import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.extend.core import primitives as jax_primitives

from tatva.tracer.analysis import analyze
from tatva.tracer.nested import CallKind, CallSpec, MapSpec, ScanSpec
from tatva.tracer.registry import SEMANTICS
from tatva.tracer.semantics import (
    CallAnalysisSemantics,
    NestedOperationSemantics,
    ScanAnalysisSemantics,
)


def test_nested_primitive_cannot_be_requested_as_ordinary():
    with pytest.raises(TypeError, match="nested operation"):
        SEMANTICS.get_ordinary(jax_primitives.scan_p)


def test_nested_primitives_are_registered():
    jit = SEMANTICS.get(jax_primitives.jit_p)
    remat = SEMANTICS.get(jax_primitives.remat_p)
    scan = SEMANTICS.get(jax_primitives.scan_p)

    assert isinstance(jit, NestedOperationSemantics)
    assert isinstance(jit.analysis, CallAnalysisSemantics)

    assert isinstance(remat, NestedOperationSemantics)
    assert isinstance(remat.analysis, CallAnalysisSemantics)

    assert isinstance(scan, NestedOperationSemantics)
    assert isinstance(scan.analysis, ScanAnalysisSemantics)


def _nested_plans(fn, *args):
    closed = jax.make_jaxpr(fn)(*args)
    plan = analyze(closed.jaxpr)

    return tuple(eqn.nested for eqn in plan.eqns if eqn.nested is not None)


def test_jit_is_analyzed_as_call():
    @jax.jit
    def inner(x):
        return x * x

    (nested,) = _nested_plans(lambda x: inner(x), jnp.ones(4))

    assert isinstance(nested.spec, CallSpec)
    assert nested.spec.call_kind is CallKind.JIT


def test_remat_is_analyzed_as_call():
    inner = jax.checkpoint(lambda x: x * x)

    (nested,) = _nested_plans(
        lambda x: inner(x),
        jnp.ones(4),
    )

    assert isinstance(nested.spec, CallSpec)
    assert nested.spec.call_kind is CallKind.REMAT


def test_stateful_scan_is_scan_spec():
    def fn(xs):
        def body(carry, x):
            carry = carry + x
            return carry, carry * carry

        _, ys = lax.scan(body, 0.0, xs)
        return ys

    (nested,) = _nested_plans(fn, jnp.ones(4))

    assert isinstance(nested.spec, ScanSpec)
    assert nested.spec.num_carry > 0


def test_lax_map_normalizes_to_map_spec():
    def fn(xs):
        return lax.map(lambda x: x * x, xs)

    (nested,) = _nested_plans(fn, jnp.ones(4))

    assert isinstance(nested.spec, MapSpec)
    assert nested.spec.length == 4
