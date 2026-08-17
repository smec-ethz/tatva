import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.extend.core import primitives as jax_primitives

from tatva.tracer.core.nested import CallKind, CallSpec, CondSpec, MapSpec, ScanSpec
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import (
    CallAnalysisSemantics,
    CondAnalysisSemantics,
    NestedOperationSemantics,
    ScanAnalysisSemantics,
)
from tatva.tracer.program.analysis import analyze


def test_nested_primitive_cannot_be_requested_as_ordinary():
    with pytest.raises(TypeError, match="NestedOperationSemantics"):
        SEMANTICS.get_ordinary(jax_primitives.scan_p)


def test_nested_primitives_are_registered():
    jit = SEMANTICS.get(jax_primitives.jit_p)
    remat = SEMANTICS.get(jax_primitives.remat_p)
    scan = SEMANTICS.get(jax_primitives.scan_p)
    cond = SEMANTICS.get(lax.cond_p)

    assert isinstance(jit, NestedOperationSemantics)
    assert isinstance(jit.analysis, CallAnalysisSemantics)

    assert isinstance(remat, NestedOperationSemantics)
    assert isinstance(remat.analysis, CallAnalysisSemantics)

    assert isinstance(scan, NestedOperationSemantics)
    assert isinstance(scan.analysis, ScanAnalysisSemantics)

    assert isinstance(cond, NestedOperationSemantics)
    assert isinstance(cond.analysis, CondAnalysisSemantics)


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


def test_cond_is_analyzed_as_cond_spec():
    def fn(pred, x):
        return lax.cond(pred, lambda v: v * 2.0, lambda v: v + 3.0, x)

    (nested,) = _nested_plans(fn, True, jnp.ones(4))

    assert isinstance(nested.spec, CondSpec)
    assert nested.spec.num_branches == 2
    assert 0 in nested.concrete_inputs  # predicate required concretely


def test_switch_is_analyzed_as_cond_spec():
    def fn(idx, x):
        return lax.switch(
            idx,
            [
                lambda v: v * 1.0,
                lambda v: v * 2.0,
                lambda v: v * 3.0,
            ],
            x,
        )

    (nested,) = _nested_plans(fn, 1, jnp.ones(4))

    assert isinstance(nested.spec, CondSpec)
    assert nested.spec.num_branches == 3
    assert 0 in nested.concrete_inputs
