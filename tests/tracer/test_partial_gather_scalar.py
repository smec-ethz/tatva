from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core
from jax.extend.core import Var

if not hasattr(lax, "stack_p"):
    lax.stack_p = core.Primitive("stack_compat")

from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import build_rank_local_plan, pending_routes
from tatva.tracer.lowering.executor import _execute_frame, _frame_outputs
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ValueRef


def test_scalar_dynamic_gather_is_compactly_conservative():
    def function(x):
        index = jnp.asarray(jnp.sum(x) > 0, dtype=jnp.int32)
        indices = jnp.reshape(index, (1,))
        dnums = lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        )
        return lax.gather(x, indices, dimension_numbers=dnums, slice_sizes=(1,))

    x = jnp.arange(10, dtype=jnp.float32) + 1
    closed = jax.make_jaxpr(function)(x)
    plan = analyze(closed.jaxpr)
    resolver, frame = ConcreteResolver.root(closed, (x,), plan)
    demand = TensorDemand.full(())
    assert demand is not None
    output = plan.jaxpr.outvars[0]
    assert isinstance(output, Var)
    trace = backpropagate_plan_demand(
        plan,
        frame,
        resolver,
        [DemandSeed(ValueRef((), output), demand)],
    )
    input_demand = trace.input_demands[0]
    assert input_demand is not None
    assert input_demand.is_full
    assert input_demand.size == x.size

    local = build_rank_local_plan(plan, frame, resolver, trace)
    assert pending_routes(local) == ()
    result = _frame_outputs(local, _execute_frame(local, (x,)))[0]
    np.testing.assert_allclose(np.asarray(result), np.asarray(function(x)))
