import jax.numpy as jnp
import numpy as np
from jax.extend.core import Var

from tatva.tracer.analysis import analyze
from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.contributions import ValueRef
from tatva.tracer.demand import TensorDemand
from tatva.tracer.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.materialize import materialize_plan


def test_targeted_gather_index_demand_reaches_root_input():
    def fn(u, values, connectivity):
        return jnp.sum(values[connectivity]) + 0.0 * jnp.sum(u)

    u = jnp.zeros(1)
    values = jnp.arange(8.0)
    connectivity = jnp.array([[1, 3], [4, 7]], dtype=jnp.int32)
    captured = CapturedJaxpr.from_fn(fn, u, values, connectivity)
    analysis = analyze(captured.jaxpr)
    instance = materialize_plan(captured.closed_jaxpr, captured.flat_args, analysis)

    resolved = next(
        item for item in instance.eqns if item.plan.eqn.primitive.name == "gather"
    )
    gather = resolved.plan.eqn
    assert isinstance(gather.invars[1], Var)
    assert resolved.route is not None
    assert resolved.route.index_rows is not None

    index_rows = np.unique(resolved.route.index_rows.ravel())
    index_demand = TensorDemand.from_rows_hull(
        tuple(gather.invars[1].aval.shape), index_rows
    )
    assert index_demand is not None

    demand = backpropagate_demand(
        instance,
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
    np.testing.assert_array_equal(
        demand.input_demands[2].rows(), np.arange(connectivity.size)
    )
