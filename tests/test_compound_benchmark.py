"""Benchmark compound indexing and update paths.

Run this benchmark with `TATVA_RUN_BENCHMARKS=1 pytest tests/test_compound_benchmark.py -s`
to see the timings.
"""

import os
import time

import jax
import jax.numpy as jnp
import pytest

from tatva.compound import Compound, field, stack_fields

jax.config.update("jax_enable_x64", True)


def _time_mean(fn, *args, warmup_runs: int = 2, measure_runs: int = 20):
    for _ in range(warmup_runs):
        out = fn(*args)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(measure_runs):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / measure_runs


def _make_plain_state(n_nodes: int) -> type[Compound]:
    class PlainState(Compound):
        u = field((n_nodes, 2))
        v = field((n_nodes,))

    return PlainState


def _make_stacked_state(n_nodes: int) -> type[Compound]:
    @stack_fields("u", "v", axis=-1)
    class StackedState(Compound):
        u = field((n_nodes, 2))
        v = field((n_nodes,))

    return StackedState


@pytest.mark.skipif(
    os.environ.get("TATVA_RUN_BENCHMARKS") != "1",
    reason="Set TATVA_RUN_BENCHMARKS=1 to run performance benchmarks.",
)
@pytest.mark.parametrize("n_nodes", [1_000, 10_000, 100_000])
def test_compound_index_and_set_benchmark(n_nodes: int):
    """Test that compound field indexing and setting fields doesn't add significant
    overhead compared to manual indexing and setting."""

    PlainState = _make_plain_state(n_nodes)
    StackedState = _make_stacked_state(n_nodes)

    plain = PlainState(jnp.zeros(PlainState.size))
    stacked = StackedState(jnp.zeros(StackedState.size))

    node_ids = jnp.arange(0, n_nodes, max(1, n_nodes // 10), dtype=int)
    u_value = jnp.arange(n_nodes * 2, dtype=jnp.float64).reshape(n_nodes, 2)
    v_value = jnp.arange(n_nodes, dtype=jnp.float64)

    def plain_indices():
        return PlainState.u[node_ids]  # pyright: ignore[reportAttributeAccessIssue]

    def stacked_indices():
        return StackedState.u[node_ids]  # pyright: ignore[reportAttributeAccessIssue]

    def plain_set():
        return plain.at("u").set(u_value).at("v").set(v_value).arr

    def stacked_set():
        return stacked.at("u").set(u_value).at("v").set(v_value).arr

    def manual_set_u():
        return (
            stacked.arr.reshape(-1, 3)
            .at[:, :2]
            .set(u_value)
            .at[:, 2]
            .set(v_value)
            .reshape(-1)
        )

    timings = {
        "plain_indices": _time_mean(plain_indices),
        "stacked_indices": _time_mean(stacked_indices),
        "plain_set": _time_mean(plain_set),
        "plain_set_jit": _time_mean(jax.jit(plain_set)),
        "stacked_set": _time_mean(stacked_set),
        "stacked_set_jit": _time_mean(jax.jit(stacked_set)),
        "manual_set": _time_mean(manual_set_u),
        "manual_set_jit": _time_mean(jax.jit(manual_set_u)),
    }

    print(f"\nCompound benchmark, n_nodes={n_nodes}:")
    for name, mean_time in timings.items():
        print(f"{name:>16}: {mean_time:.6e}s")

    assert plain_indices().shape[0] == node_ids.shape[0] * 2
    assert stacked_indices().shape[0] == node_ids.shape[0] * 2
    assert plain_set().shape == (PlainState.size,)
    assert stacked_set().shape == (StackedState.size,)
