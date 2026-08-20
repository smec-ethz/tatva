from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

_JAX_VERSION = tuple(int(part) for part in jax.__version__.split(".")[:2])
pytestmark = pytest.mark.skipif(
    _JAX_VERSION < (0, 11),
    reason="these regressions target the JAX 0.11 custom_jvp staging ABI",
)


def test_custom_jvp_keeps_primal_and_jvp_captures_separate() -> None:
    from tatva.tracer import analyze

    primal_scale = jnp.asarray(3.0)
    jvp_scale = jnp.asarray(5.0)

    @jax.custom_jvp
    def custom_scale(x):
        return primal_scale * x

    @custom_scale.defjvp
    def custom_scale_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return primal_scale * x, jvp_scale * x_dot

    def objective(u):
        y = custom_scale(u)
        return jnp.sum(y * y)

    u = jnp.arange(6.0, dtype=jnp.float32)
    local = analyze(objective, u).distribute(parts=1).rank(0)
    args = local.localize(u)
    executable = local.compile()

    assert jnp.allclose(executable(*args.args, **args.kwargs), objective(u))
    assert jnp.allclose(
        jax.grad(executable)(*args.args),
        2 * primal_scale * jvp_scale * u,
    )


def test_custom_vjp_is_explicitly_unsupported() -> None:
    from tatva.tracer import analyze
    from tatva.tracer.support import SupportPreflightError

    @jax.custom_vjp
    def square(x):
        return x * x

    def square_fwd(x):
        return x * x, x

    def square_bwd(residual, cotangent):
        return (2 * residual * cotangent,)

    square.defvjp(square_fwd, square_bwd)

    def objective(u):
        return jnp.sum(square(u))

    u = jnp.arange(4.0, dtype=jnp.float32)
    with pytest.raises(SupportPreflightError, match="custom_vjp"):
        analyze(objective, u)
