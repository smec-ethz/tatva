"""
Test that MPI element-distributed parallelism (grad, HVP, Hessian) matches serial.

Run with a single rank (serial correctness check):
    uv run pytest tests/test_mpi_equivalence.py

Run with multiple MPI ranks:
    mpirun -n 2 uv run pytest tests/test_mpi_equivalence.py
    mpirun -n 4 uv run pytest tests/test_mpi_equivalence.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_autovmap import autovmap

jax.config.update("jax_enable_x64", True)

try:
    import mpi4jax  # noqa: F401
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva import Mesh, Operator, element, sparse
from tatva.utils import mpi_reduce

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py and mpi4jax required")


@autovmap(grad_u=2)
def _compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


@autovmap(grad_u=2)
def _strain_energy_density(grad_u):
    eps = _compute_strain(grad_u)
    mu, lmbda = 1.0, 0.3
    sigma = 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)
    return 0.5 * jnp.einsum("ij,ij->", sigma, eps)


def _split_elements(elements, rank, size):
    n = elements.shape[0]
    chunk = (n + size - 1) // size
    start = rank * chunk
    end = min(start + chunk, n)
    return elements[start:end]


def test_mpi_equivalence():
    """
    Verify that MPI element-distributed grad, HVP, and Hessian match the
    serial reference on all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    raw_mesh = Mesh.unit_square(10, 10)
    n_dofs = raw_mesh.coords.shape[0] * 2
    u = jnp.arange(n_dofs, dtype=jnp.float64) * 0.01
    v = jnp.ones(n_dofs, dtype=jnp.float64)

    # Serial reference (full mesh, no MPI)
    full_op = Operator(raw_mesh, element.Tri3())

    def energy_fn_full(u_flat):
        u = u_flat.reshape(-1, 2)
        return full_op.integrate(_strain_energy_density(full_op.grad(u)))

    grad_ref = jax.jit(jax.grad(energy_fn_full))(u)
    hvp_ref = jax.jit(lambda u, v: jax.jvp(jax.grad(energy_fn_full), (u,), (v,))[1])(
        u, v
    )

    sparsity = sparse.create_sparsity_pattern(raw_mesh, n_dofs_per_node=2)
    colored_matrix = sparse.ColoredMatrix.from_csr(sparsity)
    hess_ref = sparse.jacfwd(
        jax.grad(energy_fn_full), colored_matrix, color_batch_size=5
    )(u).to_dense()

    # MPI element-distributed (local sub-mesh per rank)
    local_elements = _split_elements(raw_mesh.elements, rank, size)
    local_mesh = Mesh(coords=raw_mesh.coords, elements=local_elements)
    local_op = Operator(local_mesh, element.Tri3())

    def energy_fn_local(u_flat):
        u = u_flat.reshape(-1, 2)
        return local_op.integrate(_strain_energy_density(local_op.grad(u)))

    grad_fn = jax.grad(energy_fn_local)

    def hvp_fn(u, v):
        return jax.jvp(grad_fn, (u,), (v,))[1]

    grad_dist = mpi_reduce(grad_fn, comm, jit=True)(u)
    hvp_dist = mpi_reduce(hvp_fn, comm, jit=True)(u, v)
    hess_dist = mpi_reduce(
        sparse.jacfwd(jax.grad(energy_fn_local), colored_matrix, color_batch_size=5),
        comm,
        jit=True,
    )(u).to_dense()

    err = f"rank {rank}/{size}"
    np.testing.assert_allclose(
        grad_dist,
        grad_ref,
        atol=1e-12,
        rtol=1e-12,
        err_msg=f"Gradient mismatch ({err})",
    )
    np.testing.assert_allclose(
        hvp_dist, hvp_ref, atol=1e-12, rtol=1e-12, err_msg=f"HVP mismatch ({err})"
    )
    np.testing.assert_allclose(
        hess_dist, hess_ref, atol=1e-12, rtol=1e-12, err_msg=f"Hessian mismatch ({err})"
    )


if __name__ == "__main__":
    test_mpi_equivalence()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("All checks passed.")
