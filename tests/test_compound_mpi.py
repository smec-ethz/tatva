import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva.compound import Compound, FieldSize, field, stack_fields
from tatva.compound.field_types import Nodal
from tatva.mesh import Mesh, PartitionInfo

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py required")


def test_global_indices():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    if rank == 0:
        l2g = np.array([0, 1, 2], dtype=np.int32)
        n_owned = 2
        local_l_nodes = jnp.array([1])  # global 1 is local 1
    elif rank == 1:
        l2g = np.array([2, 3, 1], dtype=np.int32)
        n_owned = 2
        local_l_nodes = jnp.array([2, 1])  # global 1 is local 2, global 3 is local 1
    else:
        l2g = np.array([], dtype=np.int32)
        n_owned = 0
        local_l_nodes = jnp.array([], dtype=int)

    p_info = PartitionInfo(nodes_local_to_global=l2g, n_owned_nodes=n_owned)
    mock_mesh = Mesh(
        coords=jnp.zeros((len(l2g), 1)), elements=jnp.zeros((0, 2), dtype=int)
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(FieldSize.AUTO, 2))
        l = field(
            shape=(FieldSize.AUTO, 1),
            field_type=Nodal(node_ids=local_l_nodes),
        )

    # Full field u (global shape 4x2)
    np.testing.assert_array_equal(MyState._g.u[0], [0, 1])
    np.testing.assert_array_equal(MyState._g.u[3, 1], [7])

    # Test class-level access
    np.testing.assert_array_equal(MyState._g.u[0], [0, 1])

    # Incomplete field l (global shape 2x1, subset [1, 3])
    # l starts after u (8 DOFs)
    np.testing.assert_array_equal(MyState._g.l[1], [8])
    np.testing.assert_array_equal(MyState._g.l[3], [9])

    with pytest.raises(IndexError):
        MyState._g.l[0]


def test_stacked_global_indices():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Simple 1D mesh: 3 nodes total. Rank 0 owns nodes 0, 1. Rank 1 owns node 2.
    if rank == 0:
        l2g = np.array([0, 1, 2], dtype=np.int32)
        n_owned = 2
    elif rank == 1:
        l2g = np.array([2, 0, 1], dtype=np.int32)  # node 2 is owned, 0 and 1 are ghosts
        n_owned = 1
    else:
        l2g = np.array([], dtype=np.int32)
        n_owned = 0

    p_info = PartitionInfo(nodes_local_to_global=l2g, n_owned_nodes=n_owned)
    mock_mesh = Mesh(
        coords=jnp.zeros((len(l2g), 1)), elements=jnp.zeros((0, 2), dtype=int)
    )

    @stack_fields("u", "v")
    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(FieldSize.AUTO, 1), field_type=Nodal())
        v = field(shape=(FieldSize.AUTO, 2), field_type=Nodal())

    # Total global nodes = 3.
    # Field u has 1 DOF per node. Field v has 2 DOFs per node.
    # Stacked shape (3, 3). Total global DOFs = 9.

    # Global layout for stack:
    # Node 0: DOFs [0, 1, 2]  (u: 0, v: 1, 2)
    # Node 1: DOFs [3, 4, 5]  (u: 3, v: 4, 5)
    # Node 2: DOFs [6, 7, 8]  (u: 6, v: 7, 8)

    # Check global indices for u
    # u[node_id, 0]
    indices_u = MyState._g.u[:]
    expected_u = np.array([0, 3, 6])
    np.testing.assert_array_equal(indices_u, expected_u)

    # Check global indices for v
    # v[node_id, dof_idx]
    indices_v = MyState._g.v[:]
    expected_v = np.array([1, 2, 4, 5, 7, 8])
    np.testing.assert_array_equal(indices_v, expected_v)

    # Check specific indexing
    assert MyState._g.u[1, 0] == 3
    assert MyState._g.v[2, 1] == 8

    # Test GlobalDataView (gathering)
    # Create an instance and fill it with some data
    state = MyState()
    # Fill u with [10, 20, 30] globally
    # rank 0 owns nodes 0, 1. rank 1 owns node 2.
    if rank == 0:
        state = state.at("u").set(state.u.at[0, 0].set(10.0).at[1, 0].set(20.0))
    elif rank == 1:
        state = state.at("u").set(
            state.u.at[0, 0].set(30.0)
        )  # node 2 is local index 0 on rank 1

    # Fill v with [[1, 2], [3, 4], [5, 6]] globally
    if rank == 0:
        state = state.at("v").set(
            state.v.at[0, :]
            .set(jnp.array([1.0, 2.0]))
            .at[1, :]
            .set(jnp.array([3.0, 4.0]))
        )
    elif rank == 1:
        state = state.at("v").set(state.v.at[0, :].set(jnp.array([5.0, 6.0])))

    # Gather via _g
    g_u = state._g.u
    g_v = state._g.v

    if rank == 0:  # allreduce will make it same on all ranks
        np.testing.assert_allclose(g_u, [[10.0], [20.0], [30.0]])
        np.testing.assert_allclose(g_v, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
