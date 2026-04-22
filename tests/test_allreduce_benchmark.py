"""Benchmark allreduce parallel FEM assembly (grad, hessian).

Measures steady-state assembly time per rank, reporting the max across ranks
(the straggler) which is the wall-clock bottleneck.

Run with:
    TATVA_RUN_BENCHMARKS=1 mpirun -n 4 uv run pytest tests/test_allreduce_benchmark.py -s
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_autovmap import autovmap

jax.config.update("jax_enable_x64", True)

try:
    import mpi4jax  # noqa: F401
    from mpi4py import MPI

    from tatva.mpi import AllreducePlan

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva import Mesh, Operator, element, sparse
from tatva.lifter import Fixed, Lifter

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py and mpi4jax required")

N_DOFS_PER_NODE = 2


@autovmap(grad_u=2)
def _strain_energy_density(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    mu, lmbda = 1.0, 0.3
    sigma = 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)
    return 0.5 * jnp.einsum("ij,ij->", sigma, eps)


def _build_problem(nx, ny, rank, size, comm):
    raw_mesh = Mesh.unit_square(nx, ny)

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

    n_free_global = plan.global_size
    n_dofs_global = raw_mesh.coords.shape[0] * N_DOFS_PER_NODE
    n_elements = raw_mesh.elements.shape[0]

    return (
        plan,
        allgather,
        grad_fn,
        hessian_fn,
        n_free_global,
        n_dofs_global,
        n_elements,
    )


@pytest.mark.skipif(
    os.environ.get("TATVA_RUN_BENCHMARKS") != "1",
    reason="Set TATVA_RUN_BENCHMARKS=1 to run performance benchmarks.",
)
@pytest.mark.parametrize(
    "nx, ny", [(20, 20), (40, 40), (80, 80), (160, 160), (320, 320)]
)
def test_allreduce_benchmark(nx, ny):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    (
        plan,
        allgather,
        grad_fn,
        hessian_fn,
        n_free_global,
        n_dofs_global,
        n_elements,
    ) = _build_problem(nx, ny, rank, size, comm)

    u_free_global = jnp.zeros(n_free_global, dtype=jnp.float64)
    u_owned = u_free_global[plan.rstart : plan.rend]
    u_full = allgather(u_owned)

    # Warmup
    g = grad_fn(u_full)
    jax.block_until_ready(g)
    K = hessian_fn(u_full)
    jax.block_until_ready(K)

    n_loops = 5

    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(n_loops):
        g = grad_fn(u_full)
        jax.block_until_ready(g)
    t_grad = (time.perf_counter() - t0) / n_loops

    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(n_loops):
        K = hessian_fn(u_full)
        jax.block_until_ready(K)
    t_hess = (time.perf_counter() - t0) / n_loops

    t_grad_max = comm.reduce(t_grad, op=MPI.MAX, root=0)
    t_hess_max = comm.reduce(t_hess, op=MPI.MAX, root=0)

    if rank == 0:
        print(
            f"\nAllreduce benchmark | P={size} | mesh={nx}×{ny} "
            f"({n_elements} elems) | n_free={n_free_global} DOFs\n"
            f"  grad  (max across ranks): {t_grad_max:.4f}s\n"
            f"  hess  (max across ranks): {t_hess_max:.4f}s"
        )
