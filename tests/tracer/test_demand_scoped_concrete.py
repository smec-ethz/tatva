import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import tatva.tracer.program.concrete_resolver as concrete_resolver_module
from tatva.tracer.capture import make_captured_jaxpr
from tatva.tracer.core.concrete import ConcreteRegion
from tatva.tracer.core.route_fragments import RouteRequest
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.program.analysis import EqnPlan, analyze
from tatva.tracer.program.concrete_resolver import (
    ConcreteEvaluationError,
    ConcreteFallback,
    ConcreteResolver,
)


def _setup(fn, *args, fallback=ConcreteFallback.GLOBAL):
    captured = make_captured_jaxpr(fn, *args)
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
        fallback=fallback,
    )
    return plan, resolver, frame


def _eqn(plan, primitive_name: str) -> EqnPlan:
    return next(
        eqn_plan
        for eqn_plan in plan.eqns
        if eqn_plan.eqn.primitive.name == primitive_name
    )


def test_value_returns_a_coordinate_aware_region_for_root_inputs():
    def objective(dofs, indices):
        return jnp.sum(dofs[indices])

    indices = jnp.arange(10_000, dtype=jnp.int32)
    plan, resolver, frame = _setup(objective, jnp.arange(10_000.0), indices)
    demand = TensorDemand.axis_selection((10_000,), 0, [7, 101, 9001])
    assert demand is not None

    region = resolver.value(frame, plan.jaxpr.invars[1], demand)

    assert isinstance(region, ConcreteRegion)
    assert region.global_shape == (10_000,)
    assert region.values.shape == (3,)
    np.testing.assert_array_equal(region.values, [7, 101, 9001])
    np.testing.assert_array_equal(region.read_rows([9001, 7]), [9001, 7])
    subset = TensorDemand.axis_selection((10_000,), 0, [101])
    assert subset is not None
    projected = resolver.value(frame, plan.jaxpr.invars[1], subset)
    assert isinstance(projected, ConcreteRegion)
    np.testing.assert_array_equal(projected.values, [101])
    assert resolver.stats.regional_cache_hits == 1


def test_large_computed_connectivity_is_evaluated_only_for_requested_rows():
    def objective(dofs, connectivity):
        shifted = (2 * connectivity + 1).astype(jnp.int32)
        return jnp.sum(dofs[shifted])

    connectivity = jnp.arange(40_000, dtype=jnp.int32).reshape(10_000, 4)
    plan, resolver, frame = _setup(
        objective,
        jnp.arange(80_002.0),
        connectivity,
    )
    gather = _eqn(plan, "gather")
    requested = np.arange(123 * 4, 125 * 4, dtype=np.int64)

    fragment = resolver.route_fragment(frame, gather, RouteRequest(requested))

    assert fragment is not None
    np.testing.assert_array_equal(
        fragment.source_rows,
        2 * np.arange(123 * 4, 125 * 4, dtype=np.int64) + 1,
    )
    assert resolver.stats.evaluated_eqns == 0
    assert resolver.stats.full_escalations == 0
    assert resolver.stats.regional_evaluated_eqns > 0
    assert resolver.stats.regional_entries < 100
    assert resolver.escalations == []


def test_reduction_can_remain_regional_while_covering_its_reduced_axis():
    def objective(dofs, table):
        indices = jnp.sum(table, axis=1).astype(jnp.int32)
        return jnp.sum(dofs[indices])

    table = jnp.arange(24_000, dtype=jnp.int32).reshape(6_000, 4) % 5
    plan, resolver, frame = _setup(objective, jnp.arange(32.0), table)
    gather = _eqn(plan, "gather")
    requested = np.array([10, 100, 1000], dtype=np.int64)

    fragment = resolver.route_fragment(frame, gather, RouteRequest(requested))

    assert fragment is not None
    expected = np.asarray(table)[requested].sum(axis=1)
    np.testing.assert_array_equal(fragment.source_rows, expected)
    assert resolver.stats.evaluated_eqns == 0
    assert resolver.stats.full_escalations == 0
    assert resolver.stats.regional_entries < 100


def test_sort_promotes_only_its_concrete_dependency_and_reports_it():
    def objective(dofs, metric):
        return jnp.sum(dofs[jnp.argsort(metric)])

    metric = jnp.arange(4096.0)[::-1]
    plan, resolver, frame = _setup(objective, jnp.arange(4096.0), metric)
    gather = _eqn(plan, "gather")

    fragment = resolver.route_fragment(
        frame, gather, RouteRequest(np.array([100, 101], dtype=np.int64))
    )

    assert fragment is not None
    np.testing.assert_array_equal(fragment.source_rows, [3995, 3994])
    assert resolver.stats.full_escalations == 1
    escalation = resolver.escalations[0]
    assert escalation.primitive == "sort"
    assert escalation.promoted_shape == (4096,)
    assert escalation.promoted_entries == 4096
    assert not escalation.requested.is_full


def test_batched_sort_is_concrete_regionally_for_complete_axis_slices():
    def objective(dofs, metric):
        return jnp.sum(dofs[jnp.argsort(metric, axis=1)])

    n_batch = 100
    width = 4
    metric = jnp.arange(n_batch * width, dtype=jnp.float32).reshape(n_batch, width)[
        :, ::-1
    ]
    plan, resolver, frame = _setup(objective, jnp.arange(width, dtype=float), metric)
    gather = _eqn(plan, "gather")
    requested = np.arange(37 * width, 38 * width, dtype=np.int64)

    fragment = resolver.route_fragment(frame, gather, RouteRequest(requested))

    assert fragment is not None
    np.testing.assert_array_equal(fragment.source_rows, [3, 2, 1, 0])
    assert resolver.stats.full_escalations == 0
    assert resolver.stats.regional_evaluated_eqns > 0
    assert resolver.stats.regional_entries < n_batch * width


def test_strict_fallback_rejects_a_global_concrete_operation():
    def objective(dofs, metric):
        return jnp.sum(dofs[jnp.argsort(metric)])

    plan, resolver, frame = _setup(
        objective,
        jnp.arange(32.0),
        jnp.arange(32.0)[::-1],
        fallback=ConcreteFallback.ERROR,
    )
    gather = _eqn(plan, "gather")

    with pytest.raises(ConcreteEvaluationError, match="sort.*FULL"):
        resolver.route_fragment(
            frame, gather, RouteRequest(np.array([3], dtype=np.int64))
        )


def test_arbitrary_scatter_reads_only_the_requested_index_region():
    def objective(dofs, indices):
        scattered = jnp.zeros_like(dofs).at[indices].add(dofs[: indices.size])
        return jnp.sum(scattered)

    plan, resolver, frame = _setup(
        objective,
        jnp.arange(16.0),
        jnp.array([1, 4, 9, 12], dtype=jnp.int32),
    )
    scatter = _eqn(plan, "scatter-add")

    fragment = resolver.route_fragment(
        frame, scatter, RouteRequest(np.array([4], dtype=np.int64))
    )

    assert fragment is not None
    assert resolver.stats.full_escalations == 0
    assert resolver.escalations == []


def test_full_fallback_inside_map_is_scoped_to_one_iteration():
    def objective(dofs, metrics):
        def element(metric):
            return jnp.sum(dofs[jnp.argsort(metric)])

        return jnp.sum(lax.map(element, metrics))

    metrics = jnp.arange(64.0).reshape(8, 8)[:, ::-1]
    plan, resolver, frame = _setup(objective, jnp.arange(8.0), metrics)
    map_eqn = next(eqn for eqn in plan.eqns if eqn.nested is not None)
    child = resolver.map_frame(frame, map_eqn, 3)
    try:
        gather = _eqn(child.plan, "gather")
        fragment = resolver.route_fragment(
            child, gather, RouteRequest(np.array([0], dtype=np.int64))
        )
    finally:
        resolver.release(child)

    assert fragment is not None
    assert resolver.escalations[0].promoted_shape == (8,)
    assert resolver.escalations[0].path
    assert resolver.escalations[0].path[0].iteration == 3


def test_full_fallback_inside_map_does_not_materialize_computed_parent_tensor(
    monkeypatch,
):
    def objective(dofs, raw_metrics):
        metrics = 2.0 * raw_metrics + 1.0

        def element(metric):
            return jnp.sum(dofs[jnp.argsort(metric)])

        return jnp.sum(lax.map(element, metrics))

    metrics = jnp.arange(4096.0).reshape(512, 8)[:, ::-1]
    fully_evaluated_shapes = []
    original_evaluate = concrete_resolver_module.evaluate_concrete_eqn

    def record_full_evaluation(eqn, inputs):
        fully_evaluated_shapes.extend(_shape_of(outvar) for outvar in eqn.outvars)
        return original_evaluate(eqn, inputs)

    monkeypatch.setattr(
        concrete_resolver_module,
        "evaluate_concrete_eqn",
        record_full_evaluation,
    )
    plan, resolver, frame = _setup(objective, jnp.arange(8.0), metrics)
    map_eqn = next(eqn for eqn in plan.eqns if eqn.nested is not None)
    child = resolver.map_frame(frame, map_eqn, 173)
    try:
        gather = _eqn(child.plan, "gather")
        fragment = resolver.route_fragment(
            child, gather, RouteRequest(np.array([0], dtype=np.int64))
        )
    finally:
        resolver.release(child)

    assert fragment is not None
    # Full evaluation stays inside the 8-entry child invocation. The parent
    # multiply/add chain is evaluated regionally for one map slice.
    assert (512, 8) not in fully_evaluated_shapes
    assert (8,) in fully_evaluated_shapes
    assert resolver.stats.regional_evaluated_eqns >= 2
    assert resolver.stats.regional_entries < 100
    assert resolver.stats.full_escalations == 1
    escalation = resolver.escalations[0]
    assert escalation.promoted_shape == (8,)
    assert escalation.path[0].iteration == 173


def test_full_fallback_inside_scan_does_not_materialize_computed_parent_tensor(
    monkeypatch,
):
    def objective(dofs, raw_metrics):
        metrics = 2.0 * raw_metrics + 1.0

        def body(carry, metric):
            term = jnp.sum(dofs[jnp.argsort(metric)])
            return carry + term, term

        _, terms = lax.scan(body, 0.0, metrics)
        return jnp.sum(terms)

    metrics = jnp.arange(4096.0).reshape(512, 8)[:, ::-1]
    fully_evaluated_shapes = []
    original_evaluate = concrete_resolver_module.evaluate_concrete_eqn

    def record_full_evaluation(eqn, inputs):
        fully_evaluated_shapes.extend(_shape_of(outvar) for outvar in eqn.outvars)
        return original_evaluate(eqn, inputs)

    monkeypatch.setattr(
        concrete_resolver_module,
        "evaluate_concrete_eqn",
        record_full_evaluation,
    )
    plan, resolver, frame = _setup(objective, jnp.arange(8.0), metrics)
    scan_eqn = next(eqn for eqn in plan.eqns if eqn.nested is not None)
    child = resolver.scan_frame(frame, scan_eqn, 173, {})
    try:
        gather = _eqn(child.plan, "gather")
        fragment = resolver.route_fragment(
            child, gather, RouteRequest(np.array([0], dtype=np.int64))
        )
    finally:
        resolver.release(child)

    assert fragment is not None
    assert (512, 8) not in fully_evaluated_shapes
    assert (8,) in fully_evaluated_shapes
    assert resolver.stats.regional_entries < 100
    assert resolver.stats.full_escalations == 1
    escalation = resolver.escalations[0]
    assert escalation.promoted_shape == (8,)
    assert escalation.path[0].iteration == 173
