import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Literal

from tatva.tracer.api import CapturedJaxpr, trace
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import DemandContext
from tatva.tracer.local.demand import TensorDemand


def _sort_eqn(function, *args):
    closed = jax.make_jaxpr(function)(*args)
    return next(eqn for eqn in closed.jaxpr.eqns if eqn.primitive is lax.sort_p)


def test_sort_is_registered_as_an_ordinary_operation():
    assert SEMANTICS.get_ordinary(lax.sort_p).lowering is not None


def test_sort_demand_requires_full_copies_of_every_operand():
    keys = jnp.array([3.0, 1.0, 2.0, 0.0])
    values = jnp.array([30.0, 10.0, 20.0, 0.0])
    eqn = _sort_eqn(
        lambda key, value: lax.sort((key, value), dimension=0, num_keys=1),
        keys,
        values,
    )
    requested = TensorDemand.axis_selection((4,), axis=0, indices=[1])
    assert requested is not None

    demands = SEMANTICS.get_ordinary(lax.sort_p).demand(
        DemandContext(eqn=eqn, output_demands=(requested, None), route=None)
    )

    assert len(demands) == 2
    assert all(demand is not None and demand.is_full for demand in demands)
    assert all(not isinstance(atom, Literal) for atom in eqn.invars)


def test_partitioned_sort_keeps_global_inputs_and_local_sorted_rows():
    def objective(keys, values):
        _, sorted_values = lax.sort((keys, values), dimension=0, num_keys=1)
        return jnp.sum(sorted_values * jnp.array([1.0, 3.0, 5.0, 7.0]))

    keys = jnp.array([4.0, 1.0, 3.0, 2.0])
    values = jnp.array([40.0, 10.0, 30.0, 20.0])
    distributed = trace(CapturedJaxpr.from_fn(objective, keys, values)).partition(
        n_parts=2
    )

    local_values = []
    local_grads = []
    for rank in range(2):
        local = distributed.for_rank(rank)
        local_args, local_kwargs = local.localize_inputs(keys, values)
        # Sort's demand rule makes both operands available in their global form.
        assert tuple(local_args[0].shape) == tuple(keys.shape)
        assert tuple(local_args[1].shape) == tuple(values.shape)

        local_function = local.local_function()
        local_values.append(local_function(*local_args, **local_kwargs))
        local_grads.append(jax.grad(local_function, argnums=(0, 1))(*local_args))
        assert "sort[" in str(jax.make_jaxpr(local_function)(*local_args))

    np.testing.assert_allclose(sum(local_values), objective(keys, values))
    global_grads = jax.grad(objective, argnums=(0, 1))(keys, values)
    for operand in range(2):
        np.testing.assert_allclose(
            sum(grad[operand] for grad in local_grads), global_grads[operand]
        )


def test_multi_operand_sort_preserves_cosorted_pairing_after_localization():
    def objective(keys, values):
        sorted_keys, sorted_values = lax.sort((keys, values), dimension=0, num_keys=1)
        return jnp.sum(sorted_keys * sorted_values)

    keys = jnp.array([3.0, 1.0, 4.0, 2.0])
    # This ordering is intentionally different from the key order: sorting
    # operands independently would produce 300 rather than the paired 200.
    values = jnp.array([20.0, 40.0, 10.0, 30.0])
    distributed = trace(CapturedJaxpr.from_fn(objective, keys, values)).partition(
        n_parts=2
    )

    local_values = []
    for rank in range(2):
        local = distributed.for_rank(rank)
        local_args, local_kwargs = local.localize_inputs(keys, values)
        local_values.append(local.local_function()(*local_args, **local_kwargs))

    np.testing.assert_allclose(sum(local_values), objective(keys, values))
