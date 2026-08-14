import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Primitive
from jax.extend.core import primitives as jax_primitives

from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.support import (
    registration_issues,
)


def _identity_primitive(name: str) -> Primitive:
    primitive = Primitive(name)

    primitive.def_impl(lambda x: x)
    primitive.def_abstract_eval(lambda x: x)

    return primitive


def test_registration_preflight_collects_all_missing_primitives():
    foo_p = _identity_primitive("test_unsupported_foo")
    bar_p = _identity_primitive("test_unsupported_bar")

    def fn(x):
        x = foo_p.bind(x)
        return bar_p.bind(x)

    closed = jax.make_jaxpr(fn)(jnp.ones(4))

    issues = registration_issues(closed.jaxpr)

    assert {issue.primitive for issue in issues} == {
        "test_unsupported_foo",
        "test_unsupported_bar",
    }


def test_registration_preflight_descends_into_nested_jaxprs():
    unsupported_p = _identity_primitive("test_nested_unsupported")

    @jax.jit
    def inner(x):
        return unsupported_p.bind(x)

    closed = jax.make_jaxpr(lambda x: inner(x))(jnp.ones(4))

    issues = registration_issues(closed.jaxpr)

    assert len(issues) == 1
    assert issues[0].primitive == "test_nested_unsupported"


def test_describe_gather():
    description = SEMANTICS.describe(lax.gather_p)

    assert "routing: supported" in description
    assert "route localization: supported" in description
    assert "lowering: specialized" in description


def test_describe_add():
    description = SEMANTICS.describe(lax.add_p)

    assert "routing: none" in description
    assert "route localization: n/a" in description


def test_describe_scan():
    description = SEMANTICS.describe(jax_primitives.scan_p)

    assert "nested" in description
    assert "ScanAnalysisSemantics" in description
