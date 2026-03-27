"""Benchmark the performance of the Lifter class for a simple 2D linear elasticity
problem. The benchmark compares the performance of different ways of using the Lifter to
compute the total energy and its gradient.

All timings should be approximately the same, demonstrating that the Lifter does not
introduce significant overhead compared to manually assembling the full solution vector.
The "change_value_time" measures the time taken if we change the value of the "top"
constraint.

Good practices for using the Lifter are:
    - Pass the Lifter as a static argument to the JIT-compiled function, or don't pass it
      at all. Pass the concrete values of the constraints as dynamic arguments instead,
      and use the Lifter's `at` method to update the constraints inside the function.
    - Pass the Lifter as a dynamic argument. Then call the function with an updated Lifter
      instance with the new constraint values using the `at` method.

To avoid:
    - Passing the Lifter as a static argument, but calling the function with an updated
      Lifter instance with the new constraint values using the `at` method. This will
      cause recompilation every time the constraint values change, which is slower.

Run this benchmark with `TATVA_RUN_BENCHMARKS=1 pytest tests/test_lifter_benchmark.py -s`
to see the timings.
"""

import os
import time
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.typing import ArrayLike
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element
from tatva.compound import Compound, field
from tatva.lifter import Fixed, Lifter, RuntimeValue

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
    nx: int = 20, ny: int = 20
) -> tuple[Operator, Callable[[Array], Array]]:
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

    return op, total_energy


def _timed_first_and_mean(
    fn: Callable,
    fn_energy: Callable,
    u_free: Array,
    arg: Lifter | ArrayLike,
    *args,
    warmup_runs: int = 2,
    measure_runs: int = 5,
    change_value_runs: int = 5,
) -> tuple[float, float, float | None, float]:
    t0 = time.perf_counter()
    out = fn(u_free, arg, *args)
    jax.block_until_ready(out)
    first = time.perf_counter() - t0

    for _ in range(warmup_runs):
        out = fn(u_free, arg, *args)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(measure_runs):
        out = fn(u_free, arg, *args)
        jax.block_until_ready(out)
    mean = (time.perf_counter() - t0) / measure_runs

    if isinstance(arg, Lifter) and not args:
        t0 = time.perf_counter()
        for i in range(change_value_runs):
            # Change the value of the "top" constraint to force recompilation
            disp_top = 0.5 + 0.1 * i
            lifter_updated = arg.at["top"].set(disp_top)
            out = fn(u_free, lifter_updated)
            jax.block_until_ready(out)
        change_value_time = (time.perf_counter() - t0) / change_value_runs
    elif args:
        t0 = time.perf_counter()
        for i in range(change_value_runs):
            # Change the value of the "top" constraint to force recompilation
            disp_top = 0.5 + 0.1 * i
            out = fn(u_free, arg, disp_top)
            jax.block_until_ready(out)
        change_value_time = (time.perf_counter() - t0) / change_value_runs
    else:
        t0 = time.perf_counter()
        for i in range(change_value_runs):
            # Change the value of the "top" constraint to force recompilation
            disp_top = 0.5 + 0.1 * i
            out = fn(u_free, disp_top)
            jax.block_until_ready(out)
        change_value_time = (time.perf_counter() - t0) / change_value_runs

    disp_top = 1.0
    if isinstance(arg, Lifter) and not args:
        out = fn_energy(u_free, arg.at["top"].set(disp_top), *args)
    elif args:
        out = fn_energy(u_free, arg, disp_top)
    else:
        out = fn_energy(u_free, disp_top, *args)
    jax.block_until_ready(out)
    e = out

    return first, mean, change_value_time, e


@pytest.mark.skipif(
    os.environ.get("TATVA_RUN_BENCHMARKS") != "1",
    reason="Set TATVA_RUN_BENCHMARKS=1 to run performance benchmarks.",
)
def test_lifter_total_energy_benchmark():
    op, energy_fn = _build_problem(100, 100)
    mesh = op.mesh

    class Solution(Compound):
        u = field(mesh.coords.shape)

    bottom = jnp.where(mesh.coords[:, 1] == 0)[0]
    top = jnp.where(mesh.coords[:, 1] == 1)[0]
    dofs_bottom = Solution.u[bottom]
    dofs_top = Solution.u[top, 1]

    lifter = Lifter(
        Solution.size,
        Fixed(dofs_bottom, RuntimeValue("bottom", 0.0)),
        Fixed(dofs_top, RuntimeValue("top", 0.5)),
    )

    @partial(jax.jit, static_argnames=("lifter",))
    def energy_free_static(u_free: Array, lifter: Lifter) -> Array:
        u_full = lifter.lift_from_zeros(u_free)
        return energy_fn(u_full)

    @partial(jax.jit, static_argnames=("lifter",))
    def energy_free_static_disp_top(
        u_free: Array, lifter: Lifter, disp_top: Array
    ) -> Array:
        u_full = lifter.at["top"].set(disp_top).lift_from_zeros(u_free)
        return energy_fn(u_full)

    @jax.jit
    def energy_free_dynamic(u_free: Array, lifter: Lifter) -> Array:
        u_full = lifter.lift_from_zeros(u_free)
        return energy_fn(u_full)

    @jax.jit
    def energy_free_disp_top(u_free: Array, disp_top: Array) -> Array:
        u_full = lifter.at["top"].set(disp_top).lift_from_zeros(u_free)
        return energy_fn(u_full)

    @jax.jit
    def energy_free_manual(u_free: Array, disp_top: Array) -> Array:
        u_full = jnp.zeros(lifter.size, dtype=jnp.float64)
        u_full = u_full.at[lifter.free_dofs].set(u_free)
        u_full = u_full.at[dofs_top].set(disp_top)
        return energy_fn(u_full)

    timings: dict[str, tuple[float, float, float | None, float]] = {}

    print(f"\nBenchmarking lifter with different energy functions [n={op.mesh.coords.shape[0]*2}]:\n")

    # first 2, passing lifter as second arg
    for energy, name in zip(
        (energy_free_static, energy_free_static_disp_top), ("static", "static_disp_top")
    ):
        # res = jax.jacrev(energy)
        # jac = jax.jit(jax.jacfwd(res), static_argnames=("lifter",))

        args = (0.5,) if "disp_top" in name else ()

        timings[name] = _timed_first_and_mean(
            energy, energy, jnp.zeros(lifter.size_reduced), lifter, *args
        )
        print(
            f"{name:>15}: first={timings[name][0]:.6f}s, mean={timings[name][1]:.6f}s, "
            f"change_value_time={timings[name][2]:.6f}s, "
            f"energy={timings[name][3]:.6e}"
        )

    for energy, name in zip((energy_free_dynamic,), ("dynamic",)):
        # res = jax.jacrev(energy)
        # jac = jax.jit(jax.jacfwd(res))

        timings[name] = _timed_first_and_mean(
            energy, energy, jnp.zeros(lifter.size_reduced), lifter
        )
        print(
            f"{name:>15}: first={timings[name][0]:.6f}s, mean={timings[name][1]:.6f}s, "
            f"change_value_time={timings[name][2]:.6f}s, "
            f"energy={timings[name][3]:.6e}"
        )

    for energy, name in zip(
        (energy_free_disp_top, energy_free_manual),
        ("disp_top", "manual"),
    ):
        # res = jax.jacrev(energy)
        # jac = jax.jit(jax.jacfwd(res))

        timings[name] = _timed_first_and_mean(
            energy, energy, jnp.zeros(lifter.size_reduced), 0.5
        )
        print(
            f"{name:>15}: first={timings[name][0]:.6f}s, mean={timings[name][1]:.6f}s, "
            f"change_value_time={timings[name][2]:.6f}s, "
            f"energy={timings[name][3]:.6e}"
        )
