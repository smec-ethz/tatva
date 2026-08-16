import jax
import jax.numpy as jnp
import numpy as np

from tatva.tracer.api import CapturedJaxpr, trace
from tatva.tracer.local.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local.plan import build_local_plan
from tatva.tracer.lowering.executor import build_local_executable
from tatva.tracer.lowering.partition import partition_contributions


def _bind_reverse(energy, reverse):
    def fn(values):
        return energy(values, reverse=reverse)

    return fn


def _lower_part(fn, u, *, n_parts=1, part=0):
    traced = trace(CapturedJaxpr.from_fn(fn, u))
    partition = partition_contributions(traced.contributions, n_parts=n_parts)
    owned = partition.for_part(part)
    seeds = tuple(
        DemandSeed(
            value=traced.contributions.root(item.root_id).value,
            demand=item.demand,
        )
        for item in owned
    )
    demand = backpropagate_demand(traced.resolved, seeds)
    plan = build_local_plan(traced.resolved, demand)
    executable = build_local_executable(
        plan,
        contributions=traced.contributions,
        owned=owned,
    )
    return executable(*executable.pack_global_inputs(u))


def test_scan_lowering_matches_forward_and_reverse_scans():
    u = jnp.arange(1.0, 5.0)

    def energy(u, *, reverse=False):
        def body(carry, x):
            next_carry = carry + x
            return next_carry, next_carry**2

        _, ys = jax.lax.scan(body, jnp.array(0.0), u, reverse=reverse)
        return jnp.sum(ys)

    for reverse in (False, True):
        fn = _bind_reverse(energy, reverse)
        np.testing.assert_allclose(_lower_part(fn, u), fn(u))


def test_scan_lowering_handles_partitioned_heterogeneous_iterations():
    u = jnp.arange(1.0, 5.0)

    def energy(values, *, reverse=False):
        def body(carry, x):
            next_carry = carry + x
            return next_carry, next_carry**2

        _, ys = jax.lax.scan(body, jnp.array(0.0), values, reverse=reverse)
        return jnp.sum(ys)

    for reverse in (False, True):
        fn = _bind_reverse(energy, reverse)
        parts = tuple(_lower_part(fn, u, n_parts=2, part=part) for part in range(2))
        np.testing.assert_allclose(sum(parts), fn(u))
