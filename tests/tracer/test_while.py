from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lax.control_flow.loops import while_p
from jax.extend.core import Primitive

from tatva.tracer.api import DistributionTarget, distribute
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import (
    DerivativeRule,
    SparsityOnlySemantics,
    no_hessian,
)
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.derivatives import tangent_pattern, trace_form_derivatives
from tatva.tracer.program.forms import FormSpec
from tatva.tracer.rules.opaque import opaque_dependencies, prepare_dependency_union
from tatva.tracer.support import (
    SupportCapability,
    SupportPreflightError,
    registration_issues,
)


def _loop(x):
    return jax.lax.while_loop(
        lambda state: state[0] < 1,
        lambda state: (state[0] + 1, state[1] ** 2),
        (0, x),
    )[1]


def test_while_unions_every_input_dependency_into_every_output_entry():
    x = jnp.arange(4, dtype=jnp.float32) + 1
    objective = lambda values: jnp.sum(_loop(values))
    closed = jax.make_jaxpr(objective)(x)
    plan = analyze(closed.jaxpr, derivative_only=True)
    resolver, frame = ConcreteResolver.root(closed, (x,), plan)
    with pytest.warns(UserWarning, match="Limited while sparsity support"):
        trace = trace_form_derivatives(
            plan, frame, resolver, FormSpec.energy(input_index=0)
        )
    eqn = next(e for e in closed.jaxpr.eqns if e.primitive is while_p)
    deps = trace.root.dependencies[eqn.outvars[1]]
    np.testing.assert_array_equal(deps.csr.toarray(), np.ones((4, 4)))
    # This is the legacy limitation, not complete nonlinear-loop Hessian support.
    np.testing.assert_array_equal(trace.tangent.toarray(), np.zeros((4, 4)))


def test_downstream_nonlinearity_uses_while_dependency_union():
    with pytest.warns(UserWarning, match="second-order couplings are not recorded"):
        pattern = tangent_pattern(lambda x: jnp.sum(_loop(x) ** 2), (jnp.ones(4),), {})
    np.testing.assert_array_equal(pattern.toarray(), np.ones((4, 4)))


@pytest.mark.parametrize("nested", [False, True])
def test_distribute_explicitly_rejects_limited_while_support(nested):
    loop = jax.jit(_loop) if nested else _loop
    objective = lambda x: jnp.sum(loop(x))
    with pytest.raises(
        SupportPreflightError, match=r"\[execution\].*while.*distribute.*unsupported"
    ):
        distribute(objective, DistributionTarget.Rank(n_parts=1, rank=0), jnp.ones(4))


def test_internal_executable_analysis_also_rejects_while():
    closed = jax.make_jaxpr(_loop)(jnp.ones(4))
    with pytest.raises(
        NotImplementedError, match="executable decomposition and distribute"
    ):
        analyze(closed.jaxpr)
    issues = registration_issues(closed.jaxpr)
    assert len(issues) == 1
    assert issues[0].capability is SupportCapability.EXECUTION
    assert registration_issues(closed.jaxpr, derivative_only=True) == ()


def test_limited_while_does_not_inspect_or_execute_body():
    unsupported = Primitive("test_while_opaque_body")
    unsupported.def_abstract_eval(lambda aval: aval)

    def fail_if_executed(value):
        raise AssertionError("sparsity tracing must not execute the while body")

    unsupported.def_impl(fail_if_executed)

    def objective(values):
        result = jax.lax.while_loop(
            lambda x: False, lambda x: unsupported.bind(x), values
        )
        return jnp.sum(result**2)

    with pytest.warns(UserWarning, match="without inspecting the condition or body"):
        pattern = tangent_pattern(objective, (jnp.ones(3),), {})
    np.testing.assert_array_equal(pattern.toarray(), np.ones((3, 3)))


def test_while_capability_description_reports_limitations():
    description = SEMANTICS.describe(while_p)
    assert "sparsity-only" in description
    assert "prepare_while" in description
    assert "execution: unsupported" in description


def test_phase_support_is_determined_by_semantics_type(monkeypatch):
    primitive = Primitive("test_arbitrary_sparsity_only")
    primitive.def_impl(lambda x: x)
    primitive.def_abstract_eval(lambda aval: aval)
    rule = SparsityOnlySemantics(
        DerivativeRule(prepare_dependency_union, opaque_dependencies, no_hessian),
    )
    monkeypatch.setitem(SEMANTICS._rules, primitive, rule)

    def objective(x):
        return jnp.sum(primitive.bind(x) ** 2)

    pattern = tangent_pattern(objective, (jnp.ones(3),), {})
    np.testing.assert_array_equal(pattern.toarray(), np.ones((3, 3)))
    with pytest.raises(SupportPreflightError, match="sparsity-only semantics"):
        distribute(objective, DistributionTarget.Rank(n_parts=1, rank=0), jnp.ones(3))


def test_sparsity_only_while_can_supply_constant_concrete_routing():
    def objective(values):
        _, indices = jax.lax.while_loop(
            lambda state: state[0] < 1,
            lambda state: (state[0] + 1, state[1] + 1),
            (0, jnp.arange(2, dtype=jnp.int32)),
        )
        return jnp.sum(values[indices] ** 2)

    with pytest.warns(UserWarning, match="Limited while sparsity support"):
        pattern = tangent_pattern(objective, (jnp.ones(4),), {})
    np.testing.assert_array_equal(pattern.toarray(), np.diag([0, 1, 1, 0]))
