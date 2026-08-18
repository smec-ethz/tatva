import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import tatva.tracer.api as tracer_api
from tatva.tracer.capture import make_captured_jaxpr
from tatva.tracer.core.nested import NestedKind
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import (
    detect_contributions,
    detect_materialized_contributions,
)
from tatva.tracer.program.incidence import generate_contribution_blocks
from tatva.tracer.program.materialize import materialize_plan


def _detect_both(fn, *args):
    captured = make_captured_jaxpr(fn, *args)
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )
    structural = detect_contributions(plan, frame, resolver)
    materialized = detect_materialized_contributions(
        materialize_plan(captured.closed_jaxpr, captured.flat_args, plan)
    )
    assert structural == materialized
    return structural, resolver


def test_structural_detector_matches_scalar_coefficients_and_cancellation():
    def objective(dofs):
        terms = dofs**2
        reduced = jnp.sum(terms)
        total = 5.0 * reduced / 2.0
        return total + 3.0 * reduced - 3.0 * reduced

    traced, _resolver = _detect_both(
        objective,
        jnp.arange(7.0),
    )

    assert len(traced.roots) == 1
    assert traced.roots[0].coefficient == 2.5
    assert traced.roots[0].domain.shape == (7,)
    assert traced.roots[0].domain.partition_axes == (0,)


def test_structural_detector_resolves_a_lazy_computed_scalar():
    def objective(dofs, scale):
        converted = scale.astype(dofs.dtype)
        return converted * jnp.sum(dofs**2)

    captured = make_captured_jaxpr(
        objective,
        jnp.arange(7.0),
        jnp.asarray(2, dtype=jnp.int32),
    )
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )
    traced = detect_contributions(plan, frame, resolver)

    assert traced.roots[0].coefficient == 2.0
    assert resolver.stats.evaluated_eqns == 1


def test_structural_detector_matches_transparent_remat_path():
    @jax.checkpoint
    def reduced(dofs):
        return jnp.sum(dofs**2)

    traced, resolver = _detect_both(
        lambda dofs: 4.0 * reduced(dofs),
        jnp.arange(6.0),
    )

    assert len(traced.roots) == 1
    assert traced.roots[0].coefficient == 4.0
    assert len(traced.roots[0].value.path) == 1
    assert traced.roots[0].value.path[0].kind is NestedKind.CALL
    assert resolver.stats.frames_created == resolver.stats.frames_released + 1
    assert resolver.stats.peak_live_frames == 2


@pytest.mark.parametrize("selector", [False, True])
def test_structural_detector_matches_only_selected_conditional_branch(selector):
    def objective(dofs, choose_scaled):
        return lax.cond(
            choose_scaled,
            lambda values: 2.0 * jnp.sum(values**2),
            lambda values: jnp.sum(values**2),
            dofs,
        )

    traced, resolver = _detect_both(
        objective,
        jnp.arange(5.0),
        jnp.asarray(selector),
    )

    assert len(traced.roots) == 1
    assert traced.roots[0].coefficient == (2.0 if selector else 1)
    assert len(traced.roots[0].value.path) == 1
    assert traced.roots[0].value.path[0].kind is NestedKind.COND
    assert resolver.stats.frames_created == 2
    assert resolver.stats.frames_released == 1


def test_opaque_map_detection_does_not_expand_iterations():
    @jax.checkpoint
    def mapped_total(dofs):
        def body(_, value):
            return None, value**2

        _, terms = lax.scan(body, None, dofs)
        return jnp.sum(terms)

    def objective(dofs):
        return 3.0 * mapped_total(dofs)

    legacy_equivalent, _ = _detect_both(objective, jnp.arange(4.0))
    assert legacy_equivalent.roots[0].domain.shape == (4,)

    captured = make_captured_jaxpr(objective, jnp.arange(4096.0))
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )
    traced = detect_contributions(plan, frame, resolver)

    assert len(traced.roots) == 1
    assert len(traced.roots[0].value.path) == 1
    assert traced.roots[0].value.path[0].kind is NestedKind.CALL
    assert traced.roots[0].domain.shape == (4096,)
    assert resolver.stats.frames_created == 2
    assert resolver.stats.frames_released == 1
    assert resolver.stats.map_iterations == 0
    assert resolver.stats.scan_iterations == 0


def test_trace_and_incidence_do_not_materialize_the_global_plan(monkeypatch):
    def objective(dofs, indices):
        return jnp.sum(dofs[indices] ** 2)

    def unexpected_materialization(*_args, **_kwargs):
        raise AssertionError("global plan was materialized")

    monkeypatch.setattr(tracer_api, "materialize_plan", unexpected_materialization)
    traced = tracer_api.trace_fn(
        objective,
        jnp.arange(8.0),
        jnp.array([1, 3, 5], dtype=jnp.int32),
    )
    blocks = generate_contribution_blocks(
        traced.contributions,
        blocks_per_root=3,
    )
    incidence = traced.incidence(blocks)

    assert incidence.n_blocks == 3
    np.testing.assert_array_equal(incidence.block_dof_counts, [1, 1, 1])


def test_resolved_tree_is_materialized_once_on_first_access(monkeypatch):
    calls = 0
    original = tracer_api.materialize_plan

    def counted_materialization(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tracer_api, "materialize_plan", counted_materialization)
    traced = tracer_api.trace_fn(lambda dofs: jnp.sum(dofs**2), jnp.arange(4.0))

    assert calls == 0
    first = traced.resolved
    assert traced.resolved is first
    assert calls == 1
