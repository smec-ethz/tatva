"""
Test that allreduce parallel FEM assembly (grad, hessian) matches serial.

Run with:
    uv run pytest tests/test_allreduce.py          # single rank
    mpirun -n 2 uv run pytest tests/test_allreduce.py
    mpirun -n 4 uv run pytest tests/test_allreduce.py
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
from tatva.mpi import AllreducePlan

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py and mpi4jax required")

N_DOFS_PER_NODE = 2


@autovmap(grad_u=2)
def _strain_energy_density(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    mu, lmbda = 1.0, 0.3
    sigma = 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)
    return 0.5 * jnp.einsum("ij,ij->", sigma, eps)


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
    """Build the allreduce parallel problem."""
    n_elem = np.asarray(raw_mesh.elements).shape[0]
    chunk = (n_elem + size - 1) // size
    e_start = rank * chunk
    e_end = min((rank + 1) * chunk, n_elem)

    local_mesh = Mesh(
        coords=raw_mesh.coords,
        elements=jnp.array(np.asarray(raw_mesh.elements)[e_start:e_end]),
    )
    local_op = Operator(local_mesh, element.Tri3())

    n_dofs = raw_mesh.coords.shape[0] * N_DOFS_PER_NODE
    coords_np = np.asarray(raw_mesh.coords)
    fixed_nodes = jnp.array(np.where(coords_np[:, 1] == coords_np[:, 1].min())[0])
    fixed_dofs = jnp.concatenate(
        [fixed_nodes * N_DOFS_PER_NODE, fixed_nodes * N_DOFS_PER_NODE + 1]
    )
    lifter = Lifter(n_dofs, Fixed(fixed_dofs, 0.0))

    free_sparsity = sparse.reduce_sparsity_pattern(
        sparse.create_sparsity_pattern(raw_mesh, N_DOFS_PER_NODE),
        lifter.free_dofs,
    )
    global_colored = sparse.ColoredMatrix.from_csr(free_sparsity)

    def local_energy_free(u_free):
        u = lifter.lift_from_zeros(u_free).reshape(-1, N_DOFS_PER_NODE)
        return local_op.integrate(_strain_energy_density(local_op.grad(u)))

    local_grad = jax.grad(local_energy_free)
    local_hessian = jax.jit(sparse.jacfwd(local_grad, global_colored))

    plan = AllreducePlan(global_colored, lifter, comm)

    allgather = plan.make_allgather()
    grad_fn = plan.make_allreduce(local_fn=local_grad)
    hessian_fn = plan.make_allreduce(local_fn=local_hessian, is_hessian=True)

    return plan, allgather, grad_fn, hessian_fn, lifter


def test_allreduce_grad():
    """Gradient assembled via allreduce matches serial reference."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    raw_mesh = Mesh.unit_square(10, 10)
    lifter_global, grad_ref_fn, _ = _build_serial_reference(raw_mesh)
    n_free_global = lifter_global.size_reduced

    plan, allgather, grad_fn, _, _ = _build_parallel_problem(raw_mesh, rank, size, comm)

    u_free_global = jnp.arange(n_free_global, dtype=jnp.float64) * 1e-4

    u_owned = u_free_global[plan.rstart : plan.rend]
    u_full = allgather(u_owned)
    grad_owned = grad_fn(u_full)

    all_grad = comm.gather(np.asarray(grad_owned), root=0)
    all_ranges = comm.gather((plan.rstart, plan.rend), root=0)

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
            err_msg="Gradient mismatch: allreduce vs serial",
        )


def test_allreduce_hessian():
    """Hessian assembled via allreduce matches serial reference."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    raw_mesh = Mesh.unit_square(10, 10)
    lifter_global, _, hess_ref_fn = _build_serial_reference(raw_mesh)
    n_free_global = lifter_global.size_reduced

    plan, allgather, _, hessian_fn, _ = _build_parallel_problem(raw_mesh, rank, size, comm)

    u_free_global = jnp.zeros(n_free_global, dtype=jnp.float64)

    u_owned = u_free_global[plan.rstart : plan.rend]
    u_full = allgather(u_owned)
    hess_owned = hessian_fn(u_full)

    owned_ptr, owned_indices = plan.owned_csr
    all_data = comm.gather(np.asarray(hess_owned.data), root=0)
    all_ptrs = comm.gather(np.asarray(owned_ptr), root=0)
    all_cols = comm.gather(np.asarray(owned_indices), root=0)
    all_ranges = comm.gather((plan.rstart, plan.rend), root=0)

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
            err_msg="Hessian mismatch: allreduce vs serial",
        )


if __name__ == "__main__":
    test_allreduce_grad()
    test_allreduce_hessian()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("All checks passed.")
