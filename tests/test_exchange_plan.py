from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI
from scipy.sparse import csr_matrix

from tatva.compound import Compound, FieldSize, FieldType, field
from tatva.lifter import Lifter
from tatva.mesh import Mesh, PartitionInfo
from tatva.mpi import ExchangePlan

jax.config.update("jax_enable_x64", True)


def test_exchange_plan_layout():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Mock partition info: 2 nodes, 2 ranks.
    # Rank 0: owns node 0, ghosts node 1.
    # Rank 1: owns node 1, ghosts node 0.
    if rank == 0:
        n_owned_nodes = 1
        l2g = np.array([0, 1], dtype=np.int32)
    elif rank == 1:
        n_owned_nodes = 1
        l2g = np.array([1, 0], dtype=np.int32)
    else:
        n_owned_nodes = 0
        l2g = np.array([], dtype=np.int32)

    p_info = PartitionInfo(nodes_local_to_global=l2g, n_owned_nodes=n_owned_nodes)

    mock_mesh = Mesh(
        coords=jnp.zeros((len(l2g), 1)), elements=jnp.zeros((0, 2), dtype=int)
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(2, 2), field_type=FieldType.NODAL)  # 2 nodes, 2 DOFs per node
        s = field(shape=(1,), field_type=FieldType.SHARED)  # 1 shared scalar
        v = field(shape=(1,), field_type=FieldType.LOCAL)  # 1 local scalar

    # Rank 0 owned count: 1*2 (u) + 1 (s) + 1 (v) = 4
    # Rank 1 owned count: 1*2 (u) + 0 (s) + 1 (v) = 3
    # Total global size = 7

    lifter = Lifter(MyState.size)
    layout_reduced, lifter_aug = lifter.augment_layout(MyState.get_layout(), comm)
    plan = ExchangePlan(layout_reduced, comm)

    if rank == 0:
        assert plan.local_size == 4
        assert plan.global_size == 7
        assert plan.rstart == 0
        assert plan.rend == 4
        # Global indices expected:
        # u: [node0_d0=0, node0_d1=1, node1_d0=4, node1_d1=5]
        # s: [2]
        # v: [3]
        np.testing.assert_array_equal(plan.layout.local_to_global, [0, 1, 4, 5, 2, 3])
        np.testing.assert_array_equal(
            plan.layout.owned_mask, [True, True, False, False, True, True]
        )

    elif rank == 1:
        assert plan.local_size == 3
        assert plan.global_size == 7
        assert plan.rstart == 4
        assert plan.rend == 7
        # Global indices expected:
        # u: [node1_d0=4, node1_d1=5, node0_d0=0, node0_d1=1]
        # s: [2]
        # v: [6]
        np.testing.assert_array_equal(plan.layout.local_to_global, [4, 5, 0, 1, 2, 6])
        np.testing.assert_array_equal(
            plan.layout.owned_mask, [True, True, False, False, False, True]
        )


def test_exchange_plan_communication():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Same mock partition as above
    if rank == 0:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([0, 1], dtype=np.int32), n_owned_nodes=1
        )
    elif rank == 1:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([1, 0], dtype=np.int32), n_owned_nodes=1
        )
    else:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([], dtype=np.int32), n_owned_nodes=0
        )

    mock_mesh = Mesh(
        coords=jnp.zeros((len(p_info.nodes_local_to_global), 1)),
        elements=jnp.zeros((0, 2), dtype=int),
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(2, 1), field_type=FieldType.NODAL)

    lifter = Lifter(MyState.size)
    layout_reduced, lifter_aug = lifter.augment_layout(MyState.get_layout(), comm)
    plan = ExchangePlan(layout_reduced, comm)

    # Global: [u0=0, u1=1]
    # Rank 0: owns [0], ghosts [1]
    # Rank 1: owns [1], ghosts [0]

    scatter_fwd = plan.make_scatter_fwd_set()

    if rank == 0:
        x_owned = jnp.array([10.0])
    elif rank == 1:
        x_owned = jnp.array([20.0])

    u_local = scatter_fwd(x_owned)

    if rank == 0:
        np.testing.assert_allclose(u_local, [10.0, 20.0])
    elif rank == 1:
        np.testing.assert_allclose(u_local, [20.0, 10.0])

    # Test reverse add (assembly)
    def local_fn(u):
        return u * 2.0

    scatter_rev = plan.make_scatter_rev_add(local_fn)

    # u_local was [10, 20] on rank 0, [20, 10] on rank 1.
    # local_fn(u_local) is [20, 40] on rank 0, [40, 20] on rank 1.
    # Assembly:
    # Global 0 gets contribution from: rank 0 (idx 0) + rank 1 (idx 1) = 20 + 20 = 40
    # Global 1 gets contribution from: rank 1 (idx 0) + rank 0 (idx 1) = 40 + 40 = 80

    x_owned_new = scatter_rev(u_local)

    if rank == 0:
        np.testing.assert_allclose(x_owned_new, [40.0])
    elif rank == 1:
        np.testing.assert_allclose(x_owned_new, [80.0])


def test_exchange_plan_incomplete_nodal():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # 3 nodes total: 0, 1, 2.
    # Subset: nodes [0, 2].
    # Rank 0: owns [0], ghosts [1]. Subset local: [0].
    # Rank 1: owns [1, 2]. Subset local: [2].
    if rank == 0:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([0, 1], dtype=np.int32), n_owned_nodes=1
        )
        subset = np.array([0], dtype=np.int32)
    elif rank == 1:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([1, 2], dtype=np.int32), n_owned_nodes=2
        )
        subset = np.array([2], dtype=np.int32)
    else:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([], dtype=np.int32), n_owned_nodes=0
        )
        subset = np.array([], dtype=np.int32)

    mock_mesh = Mesh(
        coords=jnp.zeros((len(p_info.nodes_local_to_global), 1)),
        elements=jnp.zeros((0, 2), dtype=int),
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(FieldSize.AUTO, 1), field_type=FieldType.NODAL)
        l = field(
            shape=(len(subset), 1),
            field_type=FieldType.NODAL,
            nodal_local_to_global=subset,
        )

    lifter = Lifter(MyState.size)
    layout_reduced, lifter_aug = lifter.augment_layout(MyState.get_layout(), comm)
    plan = ExchangePlan(layout_reduced, comm)

    # Nodal U (full):
    # Global size = 3
    # Rank 0 owns U[0] (global 0). Rank 1 owns U[1] (global 1) and U[2] (global 2).
    # Contingous block for Rank 0: [U[0]] -> indices [0]
    # Contingous block for Rank 1: [U[1], U[2]] -> indices [1, 2]

    # Nodal L (incomplete, nodes 0 and 2):
    # Global size = 2
    # Rank 0 owns L[node 0] (global 0). Rank 1 owns L[node 2] (global 2).
    # Contingous block for Rank 0: [U[0], L[0]] -> indices [0, 1]
    # Contingous block for Rank 1: [U[1], U[2], L[2]] -> indices [2, 3, 4]

    # Wait, my logic for CONTIGUOUS block:
    # Rank 0 owns 2 DOFs (U0, L0). rstart=0, rend=2.
    # Rank 1 owns 3 DOFs (U1, U2, L2). rstart=2, rend=5.

    # Global indices (Algebraic, solver-aligned):
    # Field U: [node 0=0, node 1=2, node 2=3]
    # Field L: [node 0=1, node 2=4]

    if rank == 0:
        assert plan.local_size == 2
        assert plan.global_size == 5
        # u local: [node 0 (owned), node 1 (ghost)]. Global IDs: [0, 2]
        # l local: [node 0 (owned)]. Global IDs: [1]
        np.testing.assert_array_equal(plan.layout.local_to_global, [0, 2, 1])
        np.testing.assert_array_equal(plan.layout.owned_mask, [True, False, True])
    elif rank == 1:
        assert plan.local_size == 3
        # u local: [node 1 (owned), node 2 (owned)]. Global IDs: [2, 3]
        # l local: [node 2 (owned)]. Global IDs: [4]
        np.testing.assert_array_equal(plan.layout.local_to_global, [2, 3, 4])
        np.testing.assert_array_equal(plan.layout.owned_mask, [True, True, True])


def test_exchange_plan_hessian():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Simple 2-rank setup, 1 node each.
    if rank == 0:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([0, 1], dtype=np.int32), n_owned_nodes=1
        )
    elif rank == 1:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([1, 0], dtype=np.int32), n_owned_nodes=1
        )
    else:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([], dtype=np.int32), n_owned_nodes=0
        )

    mock_mesh = Mesh(
        coords=jnp.zeros((len(p_info.nodes_local_to_global), 1)),
        elements=jnp.zeros((0, 2), dtype=int),
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(2, 1), field_type=FieldType.NODAL)

    from tatva.sparse import ColoredMatrix

    # Each rank has its local ColoredMatrix.
    # Local mesh: 2 nodes, 1 element connecting them.
    # Local CSR: [[1, 2], [3, 4]] -> nonzeros (0,0), (0,1), (1,0), (1,1)
    # Note: local indices are [0, 1].
    # Local indptr: [0, 2, 4], indices: [0, 1, 0, 1]
    indptr = np.array([0, 2, 4], dtype=np.int32)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)

    # Colored matrix from this CSR
    cm = ColoredMatrix.from_csr(
        csr_matrix(([1.0, 1.0, 1.0, 1.0], indices, indptr), shape=(2, 2))
    )

    # Data values: rank 0 uses 10s, rank 1 uses 100s
    data = jnp.array([1.0, 2.0, 3.0, 4.0]) * (10.0 if rank == 0 else 100.0)
    cm = replace(cm, data=data)

    lifter = Lifter(MyState.size)
    layout_reduced, lifter_aug = lifter.augment_layout(MyState.get_layout(), comm)
    plan = ExchangePlan(layout_reduced, comm, local_colored_matrix=cm)

    # Expected Global layout:
    # Rank 0 owns global node 0 (global index 0).
    # Rank 1 owns global node 1 (global index 1).
    # Global CSR should be (2x2): [[G00, G01], [G10, G11]]

    # Routing:
    # Rank 0 local (0,0) -> global (0,0) -> owned by rank 0.
    # Rank 0 local (0,1) -> global (0,1) -> owned by rank 0.
    # Rank 0 local (1,0) -> global (1,0) -> owned by rank 1.
    # Rank 0 local (1,1) -> global (1,1) -> owned by rank 1.

    # Rank 1 local (0,0) -> global (1,1) -> owned by rank 1.
    # Rank 1 local (0,1) -> global (1,0) -> owned by rank 1.
    # Rank 1 local (1,0) -> global (0,1) -> owned by rank 0.
    # Rank 1 local (1,1) -> global (0,0) -> owned by rank 0.

    # Global Values (Summed):
    # G00 = Rank0(0,0) + Rank1(1,1) = 10 + 400 = 410
    # G01 = Rank0(0,1) + Rank1(1,0) = 20 + 300 = 320
    # G10 = Rank0(1,0) + Rank1(0,1) = 30 + 200 = 230
    # G11 = Rank0(1,1) + Rank1(0,0) = 40 + 100 = 140

    def local_fn(u):
        return cm

    hessian_fn = plan.make_scatter_rev_add(local_fn, is_hessian=True)

    # Dummy input
    u_dummy = jnp.zeros(2)
    hess_assembled = hessian_fn(u_dummy)

    assert isinstance(hess_assembled, ColoredMatrix)

    # Verify owned CSR content
    # Rank 0 owns global rows 0.
    # global_rows [0] has cols [0, 1]. indptr=[0, 2], indices=[0, 1]
    if rank == 0:
        assert plan.owned_nnz == 2
        np.testing.assert_array_equal(plan.owned_csr[0], [0, 2])
        np.testing.assert_array_equal(plan.owned_csr[1], [0, 1])
        np.testing.assert_allclose(hess_assembled.data, [410.0, 320.0])

    elif rank == 1:
        # Rank 1 owns global row 1.
        # global_rows [1] has cols [0, 1]. indptr=[0, 2], indices=[0, 1]
        assert plan.owned_nnz == 2
        np.testing.assert_array_equal(plan.owned_csr[0], [0, 2])
        np.testing.assert_array_equal(plan.owned_csr[1], [0, 1])
        np.testing.assert_allclose(hess_assembled.data, [230.0, 140.0])


def test_exchange_plan_with_constraints():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # 2 nodes, 2 ranks.
    if rank == 0:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([0, 1], dtype=np.int32), n_owned_nodes=1
        )
    elif rank == 1:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([1, 0], dtype=np.int32), n_owned_nodes=1
        )
    else:
        p_info = PartitionInfo(
            nodes_local_to_global=np.array([], dtype=np.int32), n_owned_nodes=0
        )

    mock_mesh = Mesh(
        coords=jnp.zeros((len(p_info.nodes_local_to_global), 1)),
        elements=jnp.zeros((0, 2), dtype=int),
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(2, 1), field_type=FieldType.NODAL)

    # Constrain node 0.
    from tatva.lifter import Fixed

    # Find local index of global node 0
    node0_local = np.where(p_info.nodes_local_to_global == 0)[0]
    fixed_dofs = node0_local.astype(np.int32)
    lifter = Lifter(MyState.size, Fixed(fixed_dofs, 0.0))

    layout_reduced, lifter_aug = lifter.augment_layout(MyState.get_layout(), comm)
    plan = ExchangePlan(layout_reduced, comm)

    # Global node 0 is fixed. Global node 1 is free.
    # Total global size = 1 (only node 1).
    # Rank 0: owns node 0 (fixed), ghosts node 1 (free).
    # Rank 1: owns node 1 (free), ghosts node 0 (fixed).

    assert plan.global_size == 1

    if rank == 0:
        # node 0 fixed (not in reduced), node 1 free ghost -> 0 (owned by rank 1)
        np.testing.assert_array_equal(plan.layout.local_to_global, [0])
        assert plan.local_size == 0
        assert plan.rstart == 0
        assert plan.rend == 0
    elif rank == 1:
        # node 1 free owned -> 0, node 0 fixed (not in reduced)
        np.testing.assert_array_equal(plan.layout.local_to_global, [0])
        assert plan.local_size == 1
        assert plan.rstart == 0
        assert plan.rend == 1


if __name__ == "__main__":
    test_exchange_plan_layout()
    test_exchange_plan_communication()
    test_exchange_plan_incomplete_nodal()
    test_exchange_plan_hessian()
    test_exchange_plan_with_constraints()
