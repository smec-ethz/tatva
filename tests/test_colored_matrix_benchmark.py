"""
Benchmark: ColoredMatrix setup, compile, and steady-state run times.

Measures three costs separately:

  setup_time   — from_csr() + jacfwd() preprocessing
                 (jnp.asarray conversions, jnp.repeat to build row indices,
                  fancy indexing to build col_colors)
  compile_time — first call: JIT tracing + XLA compilation
  run_time     — subsequent calls: steady-state (cache hits only)

Run this benchmark with `TATVA_RUN_BENCHMARKS=1 pytest tests/test_colored_matrix_benchmark.py -s`
to see the timings.
"""

import os
import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax import Array
from jax_autovmap import autovmap
from tatva import Mesh, Operator, element
from tatva.sparse import ColoredMatrix, create_sparsity_pattern, jacfwd

jax.config.update("jax_enable_x64", True)


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
    n_dofs = mesh.coords.shape[0] * 2

    op = Operator(mesh, element.Tri3())

    mu, lmbda = 1.0, 1.0

    @autovmap(grad_u=2)
    def strain_energy_density(grad_u):
        eps = 0.5 * (grad_u + grad_u.T)
        return 0.5 * (
            2 * mu * jnp.einsum("ij,ij->", eps, eps) + lmbda * jnp.trace(eps) ** 2
        )

    def energy_fn(u_flat):
        u = u_flat.reshape(-1, 2)
        return op.integrate(strain_energy_density(op.grad(u)))

    grad_fn = jax.grad(energy_fn)

    sparsity = create_sparsity_pattern(mesh, 2)

    return op, energy_fn, grad_fn, sparsity, n_dofs


@pytest.mark.skipif(
    os.environ.get("TATVA_RUN_BENCHMARKS") != "1",
    reason="Set TATVA_RUN_BENCHMARKS=1 to run performance benchmarks.",
)
@pytest.mark.parametrize(
    "nx, ny", [(100, 100), (200, 200), (280, 280)]
)  # Test with different mesh sizes
def test_colored_matrix_benchmark(nx, ny):

    op, energy_fn, grad_fn, sparsity, n_dofs = _build_problem(nx, ny)

    u = jnp.zeros(n_dofs)

    t0 = time.perf_counter()
    cm = ColoredMatrix.from_csr(sparsity)
    n_colors = int(cm.colors.max()) + 1
    hess_fn = jacfwd(grad_fn, cm, color_batch_size=n_colors)
    hess_fn = jax.jit(hess_fn)
    setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    hess_fn(u).data.block_until_ready()
    compile_time = time.perf_counter() - t0

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        hess_fn(u).data.block_until_ready()
        times.append(time.perf_counter() - t0)
    run_time = float(np.median(times))

    print(
        f"mesh={nx}x{ny} | n_dofs={n_dofs} | nnz={sparsity.nnz} | n_colors={n_colors}"
    )
    print(f"  setup:   {setup * 1000:8.1f} ms")
    print(f"  compile: {compile_time * 1000:8.1f} ms")
    print(f"  run:     {run_time * 1000:8.1f} ms")
