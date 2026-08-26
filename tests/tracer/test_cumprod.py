import jax
import jax.numpy as jnp
import numpy as np

from tatva.tracer.api import analyze


def _element_cumprod_energy(u):
    elements = jnp.reshape(u, (6, 4))

    def kernel(values):
        products = jnp.cumprod(values)
        return jnp.sum((products + 0.25) ** 2)

    return jnp.sum(jax.lax.map(kernel, elements))


def test_invocation_local_opaque_cumprod_preserves_value_and_gradient():
    u = jnp.linspace(0.5, 1.5, 24, dtype=jnp.float32)

    distributed = analyze(_element_cumprod_energy, u).distribute(parts=3)

    value = 0.0
    gradient = np.zeros(u.shape, dtype=np.float32)
    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        args, kwargs = local.inputs(u)
        value += float(local(*args, **kwargs))
        local_gradient = np.asarray(jax.grad(local)(*args, **kwargs))
        gradient[local.dofs.storage.global_dofs] += local_gradient

    np.testing.assert_allclose(value, _element_cumprod_energy(u), rtol=1e-5)
    np.testing.assert_allclose(
        gradient,
        jax.grad(_element_cumprod_energy)(u),
        rtol=1e-5,
        atol=1e-6,
    )
