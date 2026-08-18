import jax
import jax.numpy as jnp
from jax._src.lax import lax

from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import DemandContext, RegionalConcreteContext
from tatva.tracer.local.demand import TensorDemand


def _empty_eqn(shape=(4,)):
    closed = jax.make_jaxpr(lambda: jnp.empty(shape, dtype=jnp.float32))()
    return next(eqn for eqn in closed.jaxpr.eqns if eqn.primitive is lax.empty_p)


def test_empty_is_registered_as_zero_input_source():
    eqn = _empty_eqn()
    semantics = SEMANTICS.get_ordinary(lax.empty_p)
    demand = TensorDemand.axis_selection((4,), axis=0, indices=[1, 3])
    assert demand is not None

    assert semantics.demand(DemandContext(eqn, (demand,), None)) == ()
    assert semantics.tagged_demand is not None
    assert semantics.lowering is not None


def test_empty_cannot_supply_concrete_routing_data():
    eqn = _empty_eqn()
    semantics = SEMANTICS.get_ordinary(lax.empty_p)
    demand = TensorDemand.full((4,))

    decision = semantics.regional_concrete(
        RegionalConcreteContext(eqn=eqn, output_index=0, demand=demand)
    )

    assert "uninitialized" in decision.reason
