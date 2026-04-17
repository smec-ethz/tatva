"""
Test that neighbor-exchange parallel FEM assembly (grad, hessian) matches serial.

Run with:
    uv run pytest tests/test_neighbor_exchange.py          # single rank
    mpirun -n 2 uv run pytest tests/test_neighbor_exchange.py
    mpirun -n 4 uv run pytest tests/test_neighbor_exchange.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_autovmap import autovmap
from scipy.sparse import csr_matrix

jax.config.update("jax_enable_x64", True)

try:
    import mpi4jax  # noqa: F401
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva import Mesh, Operator, element, sparse
from tatva.lifter import Fixed, Lifter
from tatva.mpi import NeighborExchangePlan, PartitionInfo

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py and mpi4jax required")

N_DOFS_PER_NODE = 2


@autovmap(grad_u=2)
def _strain_energy_density(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    mu, lmbda = 1.0, 0.3
    sigma = 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)
    return 0.5 * jnp.einsum("ij,ij->", sigma, eps)


def _partition_elements(mesh, rank, size):
    """Partition mesh elements by index chunks; node ownership = minimum rank touching it."""
    n_elem = np.asarray(mesh.elements).shape[0]
    chunk = (n_elem + size - 1) // size
    elem_partition = np.empty(n_elem, dtype=np.int32)
    for r in range(size):
        elem_partition[r * chunk : min((r + 1) * chunk, n_elem)] = r

    local_elements_global = np.asarray(mesh.elements)[elem_partition == rank]
    active_nodes = np.unique(local_elements_global.ravel())

    node_g2l = np.full(len(mesh.coords), -1, dtype=np.int64)
    node_g2l[active_nodes] = np.arange(len(active_nodes), dtype=np.int64)

    local_mesh = Mesh(
        coords=jnp.array(np.asarray(mesh.coords)[active_nodes]),
        elements=jnp.array(node_g2l[local_elements_global]),
    )

    node_owner = np.full(len(mesh.coords), size, dtype=np.int32)
    for col in range(np.asarray(mesh.elements).shape[1]):
        np.minimum.at(node_owner, np.asarray(mesh.elements)[:, col], elem_partition)

    partition_info = PartitionInfo(
        active_nodes=active_nodes,
        owned_nodes_mask=node_owner[active_nodes] == rank,
        nodes_global_to_local=node_g2l,
    )

    return local_mesh, partition_info


def _build_serial_reference(raw_mesh):
    """Serial grad and hessian functions over the full free DOF space."""
    n_dofs = raw_mesh.coords.shape[0] * N_DOFS_PER_NODE
    full_op = Operator(raw_mesh, element.Tri3())

    coords_np = np.asarray(raw_mesh.coords)
    fixed_nodes = jnp.array(np.where(coords_np[:, 1] == coords_np[:, 1].min())[0])
    fixed_dofs = jnp.concatenate(
        [fixed_nodes * N_DOFS_PER_NODE, fixed_nodes * N_DOFS_PER_NODE + 1]
    )
    lifter_global = Lifter(n_dofs, Fixed(fixed_dofs, 0.0))

    def energy_free(u_free):
        u = lifter_global.lift_from_zeros(u_free).reshape(-1, N_DOFS_PER_NODE)
        return full_op.integrate(_strain_energy_density(full_op.grad(u)))

    free_sparsity = sparse.reduce_sparsity_pattern(
        sparse.create_sparsity_pattern(raw_mesh, N_DOFS_PER_NODE),
        lifter_global.free_dofs,
    )
    colored_global = sparse.ColoredMatrix.from_csr(free_sparsity)

    grad_ref_fn = jax.jit(jax.grad(energy_free))
    hess_ref_fn = jax.jit(sparse.jacfwd(jax.grad(energy_free), colored_global))

    return lifter_global, grad_ref_fn, hess_ref_fn


def _build_parallel_problem(raw_mesh, rank, size, comm):
    """Build the parallel neighbor-exchange problem."""
    local_mesh, partition_info = _partition_elements(raw_mesh, rank, size)
    n_active = local_mesh.coords.shape[0] * N_DOFS_PER_NODE
    local_op = Operator(local_mesh, element.Tri3())

    coords_np = np.asarray(local_mesh.coords)
    y_min = np.asarray(raw_mesh.coords)[:, 1].min()
    fixed_nodes = jnp.array(np.where(coords_np[:, 1] == y_min)[0])
    fixed_dofs = jnp.concatenate(
        [fixed_nodes * N_DOFS_PER_NODE, fixed_nodes * N_DOFS_PER_NODE + 1]
    )
    lifter = Lifter(n_active, Fixed(fixed_dofs, 0.0))

    local_free_sparsity = sparse.reduce_sparsity_pattern(
        sparse.create_sparsity_pattern(local_mesh, N_DOFS_PER_NODE),
        lifter.free_dofs,
    )
    local_colored = sparse.ColoredMatrix.from_csr(local_free_sparsity)

    def local_energy_free(u_free_active):
        u = lifter.lift_from_zeros(u_free_active).reshape(-1, N_DOFS_PER_NODE)
        return local_op.integrate(_strain_energy_density(local_op.grad(u)))

    local_grad = jax.grad(local_energy_free)
    local_hessian = jax.jit(sparse.jacfwd(local_grad, local_colored))

    global_sparsity = sparse.create_sparsity_pattern(raw_mesh, N_DOFS_PER_NODE)
    nbr_plan = NeighborExchangePlan(
        global_sparsity, partition_info, N_DOFS_PER_NODE, local_colored, lifter, comm
    )

    scatter_fwd = nbr_plan.make_scatter_fwd_set(n_free_local=lifter.size_reduced)
    grad_fn = nbr_plan.make_scatter_rev_add(local_fn=local_grad)
    hessian_fn = nbr_plan.make_scatter_rev_add(local_fn=local_hessian, is_hessian=True)

    return nbr_plan, scatter_fwd, grad_fn, hessian_fn


def test_neighbor_exchange_grad():
    """Gradient assembled via neighbor exchange matches serial reference."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    raw_mesh = Mesh.unit_square(10, 10)
    lifter_global, grad_ref_fn, _, = _build_serial_reference(raw_mesh)
    n_free_global = lifter_global.size_reduced

    nbr_plan, scatter_fwd, grad_fn, _ = _build_parallel_problem(raw_mesh, rank, size, comm)

    u_free_global = jnp.arange(n_free_global, dtype=jnp.float64) * 1e-4

    u_owned = u_free_global[nbr_plan.rstart : nbr_plan.rend]
    u_active_free = scatter_fwd(u_owned)
    grad_owned = grad_fn(u_active_free)

    all_grad = comm.gather(np.asarray(grad_owned), root=0)
    all_ranges = comm.gather((nbr_plan.rstart, nbr_plan.rend), root=0)

    if rank == 0:
        grad_ref = np.asarray(grad_ref_fn(u_free_global))
        grad_assembled = np.empty(n_free_global)
        for g, (rs, re) in zip(all_grad, all_ranges):
            grad_assembled[rs:re] = g
        np.testing.assert_allclose(
            grad_assembled,
            grad_ref,
            atol=1e-12,
            rtol=1e-12,
            err_msg="Gradient mismatch: neighbor exchange vs serial",
        )


def test_neighbor_exchange_hessian():
    """Hessian assembled via neighbor exchange matches serial reference."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    raw_mesh = Mesh.unit_square(10, 10)
    lifter_global, _, hess_ref_fn = _build_serial_reference(raw_mesh)
    n_free_global = lifter_global.size_reduced

    nbr_plan, scatter_fwd, _, hessian_fn = _build_parallel_problem(raw_mesh, rank, size, comm)

    u_free_global = jnp.zeros(n_free_global, dtype=jnp.float64)

    u_owned = u_free_global[nbr_plan.rstart : nbr_plan.rend]
    u_active_free = scatter_fwd(u_owned)
    hess_owned = hessian_fn(u_active_free)

    owned_ptr, owned_indices = nbr_plan.owned_csr
    all_data = comm.gather(np.asarray(hess_owned.data), root=0)
    all_ptrs = comm.gather(np.asarray(owned_ptr), root=0)
    all_cols = comm.gather(np.asarray(owned_indices), root=0)
    all_ranges = comm.gather((nbr_plan.rstart, nbr_plan.rend), root=0)

    if rank == 0:
        rows = np.concatenate(
            [
                np.repeat(np.arange(rs, re), np.diff(ptr))
                for ptr, (rs, re) in zip(all_ptrs, all_ranges)
            ]
        )
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_data)
        K_assembled = csr_matrix(
            (vals, (rows, cols)), shape=(n_free_global, n_free_global)
        ).toarray()

        hess_ref = np.asarray(hess_ref_fn(u_free_global).to_dense())
        np.testing.assert_allclose(
            K_assembled,
            hess_ref,
            atol=1e-12,
            rtol=1e-12,
            err_msg="Hessian mismatch: neighbor exchange vs serial",
        )


if __name__ == "__main__":
    test_neighbor_exchange_grad()
    test_neighbor_exchange_hessian()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("All checks passed.")
