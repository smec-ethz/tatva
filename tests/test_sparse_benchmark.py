"""Benchmark the performance of Sparse differentiation for a simple 2D linear elasticity
problem. The benchmark compares the performance of sparse differentiation with and without
auxiliary arguments.

All timings should be approximately the similar, demonstrating that the presence of
auxiliary arguments does not significantly affect performance.


Run this benchmark with `TATVA_RUN_BENCHMARKS=1 pytest tests/test_sparse_benchmark.py -s`
to see the timings.
"""

import os
import time
from typing import Callable, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
import scipy.sparse as sp
from jax import Array
from jax_autovmap import autovmap
from tatva import Mesh, Operator, element, sparse
from tatva_coloring import distance2_color_and_seeds


class Material(NamedTuple):
    mu: float
    lmbda: float

    @classmethod
    def from_youngs_poisson_2d(
        cls, E: float, nu: float, plane_stress: bool = False
    ) -> "Material":
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * nu * mu / (1 - nu)
        else:
            lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
        return cls(mu=mu, lmbda=lmbda)


def _build_problem(
    nx: int = 100, ny: int = 100
) -> tuple[
    Operator,
    Callable[[Array], Array],
    Callable[[Array, Array], Array],
    sp.csr_matrix,
    Array,
]:
    mesh = Mesh.unit_square(nx, ny)

    op = Operator(mesh, element.Tri3())
    mat = Material.from_youngs_poisson_2d(E=1.0, nu=0.3)

    @autovmap(grad_u=2)
    def compute_strain(grad_u):
        return 0.5 * (grad_u + grad_u.T)

    @autovmap(eps=2, mu=0, lmbda=0)
    def compute_stress(eps, mu, lmbda):
        return 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)

    @autovmap(grad_u=2, mu=0, lmbda=0)
    def strain_energy_density(grad_u, mu, lmbda):
        eps = compute_strain(grad_u)
        sigma = compute_stress(eps, mu, lmbda)
        return 0.5 * jnp.einsum("ij,ij->", sigma, eps)

    @jax.jit
    def total_energy(u_flat: Array) -> Array:
        u = u_flat.reshape(-1, 2)
        u_grad = op.grad(u)
        e_density = strain_energy_density(u_grad, mat.mu, mat.lmbda)
        return op.integrate(e_density)

    @jax.jit
    def total_energy_with_args(u_flat: Array, damage: Array) -> Array:
        return total_energy(u_flat) * jnp.mean(damage)

    sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=2)
    sparsity_pattern_csr = sp.csr_matrix(
        (
            sparsity_pattern.data,
            (sparsity_pattern.indices[:, 0], sparsity_pattern.indices[:, 1]),
        )
    )

    colors = distance2_color_and_seeds(
        row_ptr=sparsity_pattern_csr.indptr,
        col_idx=sparsity_pattern_csr.indices,
        n_dofs=mesh.coords.shape[0] * 2,
    )[0]

    return op, total_energy, total_energy_with_args, sparsity_pattern_csr, colors


@pytest.mark.skipif(
    os.environ.get("TATVA_RUN_BENCHMARKS") != "1",
    reason="Set TATVA_RUN_BENCHMARKS=1 to run performance benchmarks.",
)
@pytest.mark.parametrize(
    "nx, ny", [(100, 100), (200, 200)]
)  # Test with different mesh sizes
def test_sparse_benchmark(nx, ny):
    op, total_energy, total_energy_with_args, sparsity_pattern_csr, colors = (
        _build_problem(nx=nx, ny=ny)
    )

    hessian_sparse_with_args = sparse.jacfwd_with_batch(
        gradient=jax.jacrev(total_energy_with_args, argnums=0),
        row_ptr=jnp.array(sparsity_pattern_csr.indptr),
        col_indices=jnp.array(sparsity_pattern_csr.indices),
        colors=jnp.array(colors),
        color_batch_size=10,  # Batch size for evaluating the element routine
        has_aux_args=True,  # Whether the gradient function has auxiliary arguments (x, y, damage, etc.)
    )

    hessian_sparse = sparse.jacfwd_with_batch(
        gradient=jax.jacrev(total_energy, argnums=0),
        row_ptr=jnp.array(sparsity_pattern_csr.indptr),
        col_indices=jnp.array(sparsity_pattern_csr.indices),
        colors=jnp.array(colors),
        color_batch_size=10,  # Batch size for evaluating the element routine
        has_aux_args=False,  # Whether the gradient function has auxiliary arguments (x, y, damage, etc.)
    )

    u_flat = jnp.zeros(op.mesh.coords.shape[0] * 2)  # Initial guess for displacements
    damage = jnp.ones(op.mesh.coords.shape[0])  # Dummy damage variable for testing)

    print(
        f"\nBenchmarking sparse differentiation with and without auxiliary arguments, n_dofs={op.mesh.coords.shape[0] * 2}:\n"
    )

    # Benchmark without auxiliary arguments
    K = hessian_sparse(u_flat)
    jax.block_until_ready(K)

    start = time.perf_counter()
    n_loops = 5
    for i in range(n_loops):
        K = hessian_sparse(u_flat)
        jax.block_until_ready(K)
    end_time = time.perf_counter()
    mean_time_without_args = (end_time - start) / n_loops
    print(
        f"{'Without Args:':>15} Time for {n_loops} loops: {mean_time_without_args:.4f} seconds"
    )

    # Benchmark with auxiliary arguments
    K = hessian_sparse_with_args(u_flat, damage)
    jax.block_until_ready(K)

    start = time.perf_counter()
    for i in range(n_loops):
        K = hessian_sparse_with_args(u_flat, damage)
        jax.block_until_ready(K)
    end_time = time.perf_counter()
    mean_time_with_args = (end_time - start) / n_loops
    print(
        f"{'With Args:':>15} Time for {n_loops} loops: {mean_time_with_args:.4f} seconds"
    )
