import jax
import jax.numpy as jnp
import numpy as np

from tatva.tracer.api import analyze


def _bind_reverse(energy, reverse):
    def fn(values):
        return energy(values, reverse=reverse)

    return fn


def _lower_part(fn, u, *, n_parts=1, part=0):
    traced = analyze(fn, u)
    local = traced.distribute(parts=n_parts).rank(part)
    inputs = local.localize(u)
    return local.compile()(*inputs.args, **inputs.kwargs)


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
