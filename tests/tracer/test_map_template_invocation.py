from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core
from jax.extend.core import Var

# The repository targets the newer JAX scan ABI. The execution environment
# used for this regression suite may expose an older ABI without ft_in.
if not hasattr(lax, "stack_p"):
    lax.stack_p = core.Primitive("stack_compat")

from tatva.tracer.core.nested import MapInvocation
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import build_rank_local_plan
from tatva.tracer.lowering.executor import _execute_frame, _frame_outputs
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ValueRef


class _LegacyFtIn:
    def __init__(self, *, num_consts: int, num_carry: int, num_xs: int):
        self._groups = (
            tuple(range(num_consts)),
            tuple(range(num_carry)),
            tuple(range(num_xs)),
        )

    def unpack(self):
        return self._groups


def _ensure_scan_ft_in(closed, *, num_consts=0, num_carry=0, num_xs=1):
    for eqn in closed.jaxpr.eqns:
        if eqn.primitive.name == "scan" and "ft_in" not in eqn.params:
            eqn.params["ft_in"] = _LegacyFtIn(
                num_consts=num_consts,
                num_carry=num_carry,
                num_xs=num_xs,
            )


def _localized_map(function, x, demand):
    closed = jax.make_jaxpr(function)(x)
    _ensure_scan_ft_in(closed)
    plan = analyze(closed.jaxpr)
    resolver, frame = ConcreteResolver.root(closed, (x,), plan)
    output = plan.jaxpr.outvars[0]
    assert isinstance(output, Var)
    trace = backpropagate_plan_demand(
        plan,
        frame,
        resolver,
        [DemandSeed(ValueRef((), output), demand)],
    )
    local = build_rank_local_plan(plan, frame, resolver, trace)
    nested = next(eqn.nested for eqn in local.eqns if eqn.nested is not None)
    assert nested is not None
    assert isinstance(nested.invocation, MapInvocation)
    return local, nested.invocation


def _run(plan, x):
    env = _execute_frame(plan, (x,))
    return _frame_outputs(plan, env)[0]


def test_sparse_local_map_uses_one_template_body():
    def function(x):
        return lax.map(lambda value: value * value + 1.0, x)

    x = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    selected = np.asarray([1, 3, 6], dtype=np.int64)
    demand = TensorDemand.axis_selection((8, 2), 0, selected)
    assert demand is not None

    local, invocation = _localized_map(function, x, demand)

    assert tuple(invocation.indices) == (1, 3, 6)
    assert not invocation.indices.is_all
    assert len(invocation.children()) == 1

    calls = 0

    def visit(child):
        nonlocal calls
        calls += 1
        return child.payload

    projected = invocation.map_children(visit)
    assert calls == 1
    assert projected.body is invocation.body

    np.testing.assert_allclose(
        np.asarray(_run(local, x[selected])),
        np.asarray(function(x)[selected]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_full_local_map_selection_is_compact():
    def function(x):
        return lax.map(lambda value: 3.0 * value - 2.0, x)

    x = jnp.arange(24, dtype=jnp.float32).reshape(12, 2)
    demand = TensorDemand.full((12, 2))
    assert demand is not None

    local, invocation = _localized_map(function, x, demand)

    assert invocation.indices.is_all
    assert invocation.indices.selected is None
    assert invocation.indices.count == 12
    assert len(invocation.children()) == 1

    np.testing.assert_allclose(
        np.asarray(_run(local, x)),
        np.asarray(function(x)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_stateful_scan_keeps_repeated_invocation():
    from tatva.tracer.core.nested import RepeatedInvocation, ScanSpec

    def function(x):
        def step(carry, value):
            next_carry = carry + value
            return next_carry, carry * value

        _, values = lax.scan(
            step,
            jnp.asarray(1.0, dtype=x.dtype),
            x,
        )
        return values

    x = jnp.arange(6, dtype=jnp.float32)
    closed = jax.make_jaxpr(function)(x)
    _ensure_scan_ft_in(closed, num_carry=1)
    plan = analyze(closed.jaxpr)
    assert plan.eqns[0].nested is not None
    assert isinstance(plan.eqns[0].nested.spec, ScanSpec)

    resolver, frame = ConcreteResolver.root(closed, (x,), plan)
    demand = TensorDemand.full((6,))
    assert demand is not None
    output = plan.jaxpr.outvars[0]
    assert isinstance(output, Var)
    trace = backpropagate_plan_demand(
        plan,
        frame,
        resolver,
        [DemandSeed(ValueRef((), output), demand)],
    )
    local = build_rank_local_plan(plan, frame, resolver, trace)
    nested = next(eqn.nested for eqn in local.eqns if eqn.nested is not None)
    assert nested is not None
    assert isinstance(nested.invocation, RepeatedInvocation)
    assert len(nested.invocation.children()) == 6

    np.testing.assert_allclose(
        np.asarray(_run(local, x)),
        np.asarray(function(x)),
        rtol=1e-6,
        atol=1e-6,
    )
