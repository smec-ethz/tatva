import jax.numpy as jnp
from jax.extend.core import Var

from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ValueRef


def test_targeted_gather_index_demand_reaches_root_input():
    def fn(u, values, connectivity):
        return jnp.sum(values[connectivity]) + 0.0 * jnp.sum(u)

    u = jnp.zeros(1)
    values = jnp.arange(8.0)
    connectivity = jnp.array([[1, 3], [4, 7]], dtype=jnp.int32)
    captured = CapturedJaxpr.from_fn(fn, u, values, connectivity)
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )

    gather_plan = next(
        item for item in plan.eqns if item.eqn.primitive.name == "gather"
    )
    gather = gather_plan.eqn
    assert isinstance(gather.invars[1], Var)
    index_demand = TensorDemand.full(_shape_of(gather.invars[1]))
    assert index_demand is not None

    demand = backpropagate_plan_demand(
        plan,
        frame,
        resolver,
        (
            DemandSeed(
                value=ValueRef(path=(), var=gather.invars[1]),
                demand=index_demand,
            ),
        ),
    )

    assert demand.input_demands[0] is None
    assert demand.input_demands[1] is None
    assert demand.input_demands[2] is not None
    assert demand.input_demands[2].is_full
