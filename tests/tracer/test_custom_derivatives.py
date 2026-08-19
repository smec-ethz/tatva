from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.tracer import FunctionalAnalysis, analyze
from tatva.tracer.core.nested import CallKind, CallSpec
from tatva.tracer.diagnostics import (
    contribution_blocks,
    global_derivatives,
    incidence,
)
from tatva.tracer.local.plan import LocalEqnPlan, LocalJaxprPlan
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
#
# Both custom derivative rules deliberately use y although the primal function
# ignores y.  This is the important regression case:
#
#   primal:     f(x, y) = x**2
#   derivative: df = 7 dx + y dy
#
# If custom_jvp/custom_vjp is incorrectly treated as a transparent ordinary
# call during incidence/liveness, the y = u[2:] slice is dropped.
#
# If lowering executes the localized primal call_jaxpr instead of preserving
# the custom derivative primitive, grad(sum(f(...))) becomes [2*x, 0] instead
# of [7, y].
# -----------------------------------------------------------------------------


@jax.custom_jvp
def custom_jvp_value(x: jax.Array, y: jax.Array) -> jax.Array:
    del y
    return x * x


@custom_jvp_value.defjvp
def custom_jvp_value_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents

    primal = x * x
    tangent = 7.0 * x_dot + y * y_dot
    return primal, tangent


@jax.custom_vjp
def custom_vjp_value(x: jax.Array, y: jax.Array) -> jax.Array:
    del y
    return x * x


def custom_vjp_value_fwd(x: jax.Array, y: jax.Array):
    # y is a derivative-only dependency carried as a residual.
    return x * x, y


def custom_vjp_value_bwd(y: jax.Array, cotangent: jax.Array):
    return 7.0 * cotangent, y * cotangent


custom_vjp_value.defvjp(custom_vjp_value_fwd, custom_vjp_value_bwd)


def energy_custom_jvp(u: jax.Array) -> jax.Array:
    x = u[:2]
    y = u[2:]
    return jnp.sum(custom_jvp_value(x, y))


def energy_custom_vjp(u: jax.Array) -> jax.Array:
    x = u[:2]
    y = u[2:]
    return jnp.sum(custom_vjp_value(x, y))


CASES = (
    pytest.param(energy_custom_jvp, CallKind.CUSTOM_JVP, id="custom_jvp"),
    pytest.param(energy_custom_vjp, CallKind.CUSTOM_VJP, id="custom_vjp"),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _example_u() -> jax.Array:
    return jnp.asarray([2.0, 3.0, 5.0, 11.0], dtype=jnp.float32)


def _expected_gradient(u: jax.Array) -> jax.Array:
    # The custom derivative is:
    #
    #   d sum(f) / d x = 7
    #   d sum(f) / d y = y
    #
    return jnp.asarray([7.0, 7.0, u[2], u[3]], dtype=u.dtype)


def _walk_eqn_plans(plan: JaxprPlan) -> Iterator[EqnPlan]:
    for eqn_plan in plan.eqns:
        yield eqn_plan
        if eqn_plan.nested is not None:
            for branch in eqn_plan.nested.branches:
                yield from _walk_eqn_plans(branch)


def _custom_structural_eqns(
    traced: FunctionalAnalysis,
    kind: CallKind,
) -> list[EqnPlan]:
    result: list[EqnPlan] = []

    for eqn_plan in _walk_eqn_plans(traced._plan):
        nested = eqn_plan.nested
        if nested is None:
            continue
        if isinstance(nested.spec, CallSpec) and nested.spec.call_kind is kind:
            result.append(eqn_plan)

    return result


def _walk_local_eqns(plan: LocalJaxprPlan) -> Iterator[LocalEqnPlan]:
    for eqn_plan in plan.eqns:
        yield eqn_plan

        nested = eqn_plan.nested
        if nested is None:
            continue

        for child in nested.invocation.children():
            yield from _walk_local_eqns(child.payload)


def _custom_local_eqns(
    plan: LocalJaxprPlan,
    kind: CallKind,
) -> list[LocalEqnPlan]:
    result: list[LocalEqnPlan] = []

    for eqn_plan in _walk_local_eqns(plan):
        nested = eqn_plan.nested
        if nested is None:
            continue
        if isinstance(nested.spec, CallSpec) and nested.spec.call_kind is kind:
            result.append(eqn_plan)

    return result


# -----------------------------------------------------------------------------
# Structural analysis
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_custom_derivative_is_analyzed_as_distinct_call_kind(energy, kind):
    u = _example_u()
    traced = analyze(energy, u)

    custom_eqns = _custom_structural_eqns(traced, kind)

    assert len(custom_eqns) == 1
    eqn_plan = custom_eqns[0]
    assert eqn_plan.nested is not None
    assert isinstance(eqn_plan.nested.spec, CallSpec)
    assert eqn_plan.nested.spec.call_kind is kind

    # The primal call_jaxpr must still be available structurally.
    assert eqn_plan.nested.body.jaxpr.outvars


# -----------------------------------------------------------------------------
# Contribution-block -> DOF incidence
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_custom_derivative_incidence_keeps_derivative_only_dofs(energy, kind):
    del kind

    u = _example_u()
    traced = analyze(energy, u)
    blocks = contribution_blocks(traced, blocks_per_root=2)
    block_incidence = incidence(traced, blocks)

    # The contribution domain has two entries. Splitting each root into two
    # diagnostic blocks therefore produces two contribution blocks.
    assert block_incidence.n_blocks == 2
    assert block_incidence.n_dofs == 4

    # Conservative first implementation: once a custom derivative boundary is
    # active, every invocation-local operand is kept FULL for that block.
    # Therefore each contribution block sees both x=u[:2] and y=u[2:].
    expected = np.arange(4, dtype=np.int64)

    for block_id in range(block_incidence.n_blocks):
        np.testing.assert_array_equal(
            block_incidence.dofs_for_block(block_id),
            expected,
        )


# -----------------------------------------------------------------------------
# Rank-local liveness/local-plan invariant
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_custom_derivative_local_plan_keeps_all_operands_full(energy, kind):
    u = _example_u()
    traced = analyze(energy, u)
    local = traced.distribute(parts=1).rank(0)

    custom_eqns = _custom_local_eqns(local._plan, kind)
    assert len(custom_eqns) == 1

    eqn_plan = custom_eqns[0]
    assert eqn_plan.nested is not None
    assert isinstance(eqn_plan.nested.spec, CallSpec)

    input_indices = eqn_plan.nested.spec.resolved_input_indices(
        len(eqn_plan.input_layouts)
    )

    # x and y must both survive, even though y is unused by the primal body.
    assert len(input_indices) == 2

    for input_index in input_indices:
        layout = eqn_plan.input_layouts[input_index]
        assert layout is not None, f"custom operand {input_index} was dropped"
        assert layout.is_full, (
            f"custom operand {input_index} is not invocation-local FULL: "
            f"global={layout.global_shape}, local={layout.local_shape}"
        )


# -----------------------------------------------------------------------------
# Lowering: primal semantics
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_compiled_custom_derivative_preserves_primal_value(energy, kind):
    del kind

    u = _example_u()
    traced = analyze(energy, u)
    local = traced.distribute(parts=1).rank(0)
    compiled = local.compile()

    expected = energy(u)
    actual = compiled(u)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# Lowering: custom AD semantics
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_compiled_custom_derivative_preserves_custom_gradient(energy, kind):
    del kind

    u = _example_u()

    # Sanity-check the custom derivative rule itself before involving Tatva.
    np.testing.assert_allclose(
        np.asarray(jax.grad(energy)(u)),
        np.asarray(_expected_gradient(u)),
        rtol=1e-6,
        atol=1e-6,
    )

    traced = analyze(energy, u)
    local = traced.distribute(parts=1).rank(0)
    compiled = local.compile()

    actual = jax.grad(compiled)(u)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(_expected_gradient(u)),
        rtol=1e-6,
        atol=1e-6,
    )

    # This explicitly catches the most likely lowering bug: if the compiler
    # executes the localized primal call_jaxpr directly, autodiff sees x**2
    # and produces [2*x, 0] instead of the registered custom rule [7, y].
    ordinary_primal_gradient = jnp.asarray(
        [2.0 * u[0], 2.0 * u[1], 0.0, 0.0], dtype=u.dtype
    )
    assert not np.allclose(
        np.asarray(actual),
        np.asarray(ordinary_primal_gradient),
    )


# -----------------------------------------------------------------------------
# Diagnostic global structural derivative analysis
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("energy, kind", CASES)
def test_global_structural_derivative_trace_is_conservative_at_custom_boundary(
    energy,
    kind,
):
    del kind

    u = _example_u()
    traced = analyze(energy, u)

    hessian = global_derivatives(traced).hessian

    assert hessian.shape == (4, 4)

    # The temporary implementation treats the custom derivative boundary as
    # opaque nonlinear over all of its input DOF dependencies.  It must not
    # report the sparsity of the primal call_jaxpr (which ignores y).
    np.testing.assert_array_equal(
        hessian.toarray(),
        np.ones((4, 4), dtype=bool),
    )
