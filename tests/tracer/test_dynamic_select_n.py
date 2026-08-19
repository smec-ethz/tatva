import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tatva.tracer import analyze


def _energy(u):
    selector = (u > 0).astype(jnp.int32)
    return jnp.sum(lax.select_n(selector, jnp.full_like(u, 2.0), u * u))


def test_dof_dependent_select_n_plans_and_lowers_locally():
    """A runtime selector must not be requested from the planning DOF frame."""
    u = jnp.asarray([-2.0, 3.0, -4.0, 5.0])
    traced = analyze(_energy, u)
    distributed = traced.distribute(parts=2)

    values = []
    gradients = []
    for rank in range(2):
        local = distributed.rank(rank)
        inputs = local.localize(u)
        compiled = local.compile()
        values.append(compiled(*inputs.args, **inputs.kwargs))
        gradients.append(jax.grad(compiled)(*inputs.args, **inputs.kwargs))

    np.testing.assert_allclose(sum(values), _energy(u))
    expected_grad = jax.grad(_energy)(u)
    for rank, gradient in enumerate(gradients):
        local = distributed.rank(rank)
        np.testing.assert_allclose(
            gradient,
            expected_grad[jnp.asarray(local.dofs.storage.global_dofs)],
        )
