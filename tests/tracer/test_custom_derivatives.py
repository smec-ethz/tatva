from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.tracer import FunctionalAnalysis, analyze
from tatva.tracer.core.nested import CustomJvpSpec
from tatva.tracer.diagnostics import (
    contribution_blocks,
    incidence,
)
from tatva.tracer.local.plan import LocalEqnPlan, LocalJaxprPlan
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.support import SupportPreflightError

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
#
# The custom JVP rule deliberately uses y although the primal function ignores
# y. This is the important regression case:
#
#   primal:     f(x, y) = x**2
#   derivative: df = 7 dx + y dy
#
# If custom_jvp is incorrectly treated as a transparent ordinary call during
# incidence/liveness, the y = u[2:] slice is dropped.
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


@jax.custom_jvp
def transposed_custom_jvp(x: jax.Array) -> jax.Array:
    return jnp.swapaxes(x, -1, -2)


@transposed_custom_jvp.defjvp
def transposed_custom_jvp_rule(primals, tangents):
    return (
        jnp.swapaxes(primals[0], -1, -2),
        jnp.swapaxes(tangents[0], -1, -2),
    )


def energy_custom_jvp(u: jax.Array) -> jax.Array:
    x = u[:2]
    y = u[2:]
    return jnp.sum(custom_jvp_value(x, y))


def energy_transposed_custom_jvp(u: jax.Array) -> jax.Array:
    return jnp.sum(transposed_custom_jvp(jnp.reshape(u, (200, 1, 2, 2))))


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


def _custom_jvp_structural_eqns(traced: FunctionalAnalysis) -> list[EqnPlan]:
    result: list[EqnPlan] = []

    for eqn_plan in _walk_eqn_plans(traced._plan):
        nested = eqn_plan.nested
        if nested is None:
            continue
        if isinstance(nested.spec, CustomJvpSpec):
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


def _custom_jvp_local_eqns(plan: LocalJaxprPlan) -> list[LocalEqnPlan]:
    result: list[LocalEqnPlan] = []

    for eqn_plan in _walk_local_eqns(plan):
        nested = eqn_plan.nested
        if nested is None:
            continue
        if isinstance(nested.spec, CustomJvpSpec):
            result.append(eqn_plan)

    return result


# -----------------------------------------------------------------------------
# Structural analysis
# -----------------------------------------------------------------------------


def test_custom_jvp_is_analyzed_with_primal_and_jvp_callbacks():
    u = _example_u()
    traced = analyze(energy_custom_jvp, u)

    custom_eqns = _custom_jvp_structural_eqns(traced)

    assert len(custom_eqns) == 1
    eqn_plan = custom_eqns[0]
    assert eqn_plan.nested is not None
    assert isinstance(eqn_plan.nested.spec, CustomJvpSpec)
    assert len(eqn_plan.nested.branches) == 2

    primal, jvp = eqn_plan.nested.branches
    assert primal.jaxpr.outvars
    assert jvp.jaxpr.outvars


def test_custom_vjp_is_rejected_as_unsupported():
    @jax.custom_vjp
    def value(x):
        return x * x

    def value_fwd(x):
        return x * x, x

    def value_bwd(residual, cotangent):
        return (2 * residual * cotangent,)

    value.defvjp(value_fwd, value_bwd)

    with pytest.raises(SupportPreflightError, match="custom_vjp"):
        analyze(lambda u: jnp.sum(value(u)), _example_u())


# -----------------------------------------------------------------------------
# Contribution-block -> DOF incidence
# -----------------------------------------------------------------------------


def test_custom_jvp_incidence_keeps_derivative_only_dofs():
    u = _example_u()
    traced = analyze(energy_custom_jvp, u)
    blocks = contribution_blocks(traced, blocks_per_root=2)
    block_incidence = incidence(traced, blocks)

    # The contribution domain has two entries. Splitting each root into two
    # diagnostic blocks therefore produces two contribution blocks.
    assert block_incidence.n_blocks == 2
    assert block_incidence.n_dofs == 4

    # The JVP callback couples each x entry to the corresponding y entry.
    for block_id in range(block_incidence.n_blocks):
        expected = np.asarray([block_id, block_id + 2], dtype=np.int64)
        np.testing.assert_array_equal(
            block_incidence.dofs_for_block(block_id),
            expected,
        )


# -----------------------------------------------------------------------------
# Rank-local liveness/local-plan invariant
# -----------------------------------------------------------------------------


def test_custom_jvp_local_plan_keeps_derivative_only_operand():
    u = _example_u()
    traced = analyze(energy_custom_jvp, u)
    local = traced.distribute(parts=1).rank(0)

    custom_eqns = _custom_jvp_local_eqns(local._plan)
    assert len(custom_eqns) == 1

    eqn_plan = custom_eqns[0]
    assert eqn_plan.nested is not None
    input_indices = tuple(range(len(eqn_plan.input_layouts)))

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


def test_compiled_custom_jvp_preserves_primal_value():
    u = _example_u()
    traced = analyze(energy_custom_jvp, u)
    local = traced.distribute(parts=1).rank(0)
    compiled = local.compile()

    expected = energy_custom_jvp(u)
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


def test_compiled_custom_jvp_preserves_custom_gradient():
    u = _example_u()

    # Sanity-check the custom derivative rule itself before involving Tatva.
    np.testing.assert_allclose(
        np.asarray(jax.grad(energy_custom_jvp)(u)),
        np.asarray(_expected_gradient(u)),
        rtol=1e-6,
        atol=1e-6,
    )

    traced = analyze(energy_custom_jvp, u)
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


def test_partitioned_custom_jvp_preserves_first_order_ad():
    u = _example_u()
    distributed = analyze(energy_custom_jvp, u).distribute(parts=2)
    values = []
    gradient_sum = np.zeros_like(np.asarray(u))

    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        inputs = local.localize(u)
        compiled = local.compile()
        values.append(compiled(*inputs.args, **inputs.kwargs))
        gradient = np.asarray(jax.grad(compiled)(*inputs.args, **inputs.kwargs))
        gradient_sum[local.dofs.storage.global_dofs] += gradient

    np.testing.assert_allclose(sum(values), energy_custom_jvp(u), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        gradient_sum, _expected_gradient(u), rtol=1e-6, atol=1e-6
    )


def test_custom_jvp_program_is_localized_before_transpose_lowering():
    u = jnp.arange(800.0, dtype=jnp.float32)
    distributed = analyze(energy_transposed_custom_jvp, u).distribute(parts=5)
    value = 0.0
    gradient = np.zeros_like(np.asarray(u))

    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        eqn = _custom_jvp_local_eqns(local._plan)[0]
        assert eqn.input_layouts[0] is not None
        assert eqn.output_layouts[0] is not None
        assert eqn.input_layouts[0].local_shape == (40, 1, 2, 2)
        assert eqn.output_layouts[0].local_shape == (40, 1, 2, 2)
        assert eqn.nested is not None
        for child_eqn in eqn.nested.invocation.jvp.eqns:  # ty: ignore[unresolved-attribute]
            for layout in child_eqn.output_layouts:
                if layout is not None and layout.global_shape[:1] == (200,):
                    assert layout.local_shape[0] == 40

        inputs = local.localize(u)
        compiled = local.compile()
        value += float(compiled(*inputs.args, **inputs.kwargs))
        local_gradient = np.asarray(jax.grad(compiled)(*inputs.args, **inputs.kwargs))
        gradient[local.dofs.storage.global_dofs] += local_gradient

    np.testing.assert_allclose(value, energy_transposed_custom_jvp(u))
    np.testing.assert_allclose(gradient, jax.grad(energy_transposed_custom_jvp)(u))


def test_localized_custom_jvp_supports_first_order_transforms_and_jit():
    u = _example_u()
    tangent = jnp.ones_like(u)
    compiled = analyze(energy_custom_jvp, u).distribute(parts=1).rank(0).compile()
    expected = jax.jvp(energy_custom_jvp, (u,), (tangent,))
    actual = jax.jit(lambda x, xd: jax.jvp(compiled, (x,), (xd,)))(u, tangent)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])

    _, pullback = jax.vjp(compiled, u)
    cotangent = jnp.ones((), dtype=u.dtype)
    np.testing.assert_allclose(pullback(cotangent)[0], _expected_gradient(u))
    np.testing.assert_allclose(jax.jit(jax.grad(compiled))(u), _expected_gradient(u))


@pytest.mark.parametrize(
    "transform",
    [
        lambda fn: jax.jacobian(jax.grad(fn)),
        jax.hessian,
    ],
)
def test_localized_custom_jvp_rejects_higher_order_ad(transform):
    u = _example_u()
    compiled = analyze(energy_custom_jvp, u).distribute(parts=1).rank(0).compile()
    with pytest.raises(NotImplementedError, match="higher-order AD"):
        transform(compiled)(u)


def test_symbolic_zero_custom_jvp_is_rejected_contextually():
    @jax.custom_jvp
    def value(x):
        return x * x

    def value_jvp(primals, tangents):
        return primals[0] ** 2, 2 * primals[0] * tangents[0]

    value.defjvp(value_jvp, symbolic_zeros=True)
    with pytest.raises(NotImplementedError, match="symbolic_zeros=False"):
        analyze(lambda u: jnp.sum(value(u)), jnp.ones(4))


def test_localized_custom_jvp_supports_dead_multiple_output():
    @jax.custom_jvp
    def values(x):
        return x * x, x + 1

    @values.defjvp
    def values_jvp(primals, tangents):
        x, x_dot = primals[0], tangents[0]
        return (x * x, x + 1), (7 * x_dot, 3 * x_dot)

    def energy(u):
        first, _unused = values(u)
        return jnp.sum(first)

    u = jnp.arange(4.0)
    distributed = analyze(energy, u).distribute(parts=2)
    gradient = np.zeros_like(np.asarray(u))
    for rank in range(2):
        local = distributed.rank(rank)
        inputs = local.localize(u)
        local_gradient = np.asarray(jax.grad(local.compile())(*inputs.args))
        gradient[local.dofs.storage.global_dofs] += local_gradient
    np.testing.assert_allclose(gradient, jax.grad(energy)(u))
