import jax
import numpy as np
from jax import lax

from tatva.tracer.core import routes
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import (
    DynamicSliceRouteFragment,
    DynamicUpdateSliceRouteFragment,
    GatherRouteFragment,
    RouteRequest,
    ScatterRouteFragment,
    SelectNRouteFragment,
    resolve_dynamic_slice_route_fragment,
    resolve_dynamic_update_slice_route_fragment,
    resolve_gather_route_fragment,
    resolve_scatter_route_fragment,
    resolve_select_n_route_fragment,
)
from tatva.tracer.core.routes import (
    resolve_dynamic_slice_route,
    resolve_dynamic_update_slice_route,
    resolve_gather_route,
    resolve_scatter_route,
    resolve_select_n_route,
)
from tatva.tracer.program.concrete_resolver import ConcreteEnv


def _gather_equation(n_indices: int):
    dnums = lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )

    def gather(operand, indices):
        return lax.gather(
            operand,
            indices,
            dimension_numbers=dnums,
            slice_sizes=(1, 2),
        )

    indices = (np.arange(n_indices, dtype=np.int32) % 100).reshape(-1, 1)
    closed = jax.make_jaxpr(gather)(
        jax.ShapeDtypeStruct((100, 2), np.float32),
        indices,
    )
    assert len(closed.jaxpr.eqns) == 1
    return closed.jaxpr.eqns[0], indices


def test_gather_route_fragment_matches_requested_full_route_rows():
    eqn, indices = _gather_equation(20)
    concrete = {eqn.invars[1]: indices}
    requested = np.array([17, 0, 17, 31, 8], dtype=np.int64)

    full = resolve_gather_route(eqn, concrete)
    fragment = resolve_gather_route_fragment(
        eqn,
        concrete,
        RouteRequest(requested),
    )

    assert full is not None
    assert fragment is not None
    np.testing.assert_array_equal(fragment.output_rows, requested)
    np.testing.assert_array_equal(fragment.source_rows, full.source_rows[requested])
    assert fragment.index_rows is not None
    assert full.index_rows is not None
    np.testing.assert_array_equal(fragment.index_rows, full.index_rows[requested])


def test_registered_gather_fragment_scales_with_requested_rows(monkeypatch):
    n_indices = 10_000
    eqn, indices = _gather_equation(n_indices)
    concrete = {eqn.invars[1]: indices}
    requested = np.array([0, n_indices, 2 * n_indices - 1], dtype=np.int64)

    original_arange = routes.np.arange

    def reject_full_output_arange(*args, **kwargs):
        if args and args[0] == 2 * n_indices:
            raise AssertionError("fragment allocated rows for the complete output")
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(routes.np, "arange", reject_full_output_arange)

    semantics = SEMANTICS.get_ordinary(eqn.primitive)
    fragment = semantics.route_fragment(
        eqn,
        concrete,
        RouteRequest(requested),
    )

    assert isinstance(fragment, GatherRouteFragment)
    assert fragment.output_rows.size == requested.size
    assert fragment.source_rows.size == requested.size
    assert fragment.index_rows is not None
    assert fragment.index_rows.shape == (requested.size, 1)


def test_gather_route_fragment_rejects_rows_outside_output():
    eqn, indices = _gather_equation(4)
    concrete = {eqn.invars[1]: indices}

    with np.testing.assert_raises_regex(ValueError, "outside the output shape"):
        resolve_gather_route_fragment(
            eqn,
            concrete,
            RouteRequest(np.array([8], dtype=np.int64)),
        )


def test_scatter_route_fragment_matches_intersecting_full_route_relations():
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def scatter(operand, indices, updates):
        return lax.scatter_add(operand, indices, updates, dnums)

    indices = np.array([[1], [3], [3], [8], [6]], dtype=np.int32)
    closed = jax.make_jaxpr(scatter)(
        jax.ShapeDtypeStruct((8,), np.float32),
        indices,
        jax.ShapeDtypeStruct((5,), np.float32),
    )
    eqn = closed.jaxpr.eqns[0]
    concrete: ConcreteEnv = {eqn.invars[1]: indices}  # ty: ignore[invalid-assignment]
    requested = np.array([3, 7], dtype=np.int64)

    full = resolve_scatter_route(eqn, concrete)
    fragment = resolve_scatter_route_fragment(eqn, concrete, RouteRequest(requested))

    assert full is not None
    assert isinstance(fragment, ScatterRouteFragment)
    expected_updates = np.flatnonzero(np.isin(full.target_rows, requested))
    np.testing.assert_array_equal(fragment.output_rows, requested)
    np.testing.assert_array_equal(fragment.update_rows, expected_updates)
    np.testing.assert_array_equal(
        fragment.target_rows, full.target_rows[expected_updates]
    )


def test_select_n_route_fragment_matches_requested_full_route_rows():
    def select(selector, first, second):
        return lax.select_n(selector, first, second)

    selector = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    closed = jax.make_jaxpr(select)(
        selector,
        jax.ShapeDtypeStruct((6,), np.float32),
        jax.ShapeDtypeStruct((6,), np.float32),
    )
    eqn = closed.jaxpr.eqns[0]
    concrete: ConcreteEnv = {eqn.invars[0]: selector}  # ty: ignore[invalid-assignment]
    requested = np.array([5, 1, 3], dtype=np.int64)

    full = resolve_select_n_route(eqn, concrete)
    fragment = resolve_select_n_route_fragment(eqn, concrete, RouteRequest(requested))

    assert full is not None
    assert isinstance(fragment, SelectNRouteFragment)
    np.testing.assert_array_equal(fragment.output_rows, requested)
    np.testing.assert_array_equal(fragment.case_indices, full.case_indices[requested])


def test_dynamic_slice_route_fragment_matches_requested_full_route_rows():
    closed = jax.make_jaxpr(
        lambda operand, start: lax.dynamic_slice(operand, (start,), (4,))
    )(
        jax.ShapeDtypeStruct((10,), np.float32),
        np.int32(3),
    )
    eqn = next(
        eqn for eqn in closed.jaxpr.eqns if eqn.primitive.name == "dynamic_slice"
    )
    concrete: ConcreteEnv = {eqn.invars[1]: np.int32(3)}  # ty: ignore[invalid-assignment]
    requested = np.array([3, 0], dtype=np.int64)

    full = resolve_dynamic_slice_route(eqn, concrete)
    fragment = resolve_dynamic_slice_route_fragment(
        eqn, concrete, RouteRequest(requested)
    )

    assert full is not None
    assert isinstance(fragment, DynamicSliceRouteFragment)
    np.testing.assert_array_equal(fragment.output_rows, requested)
    np.testing.assert_array_equal(fragment.source_rows, full.source_rows[requested])


def test_dynamic_update_fragment_matches_intersecting_full_route_relations():
    closed = jax.make_jaxpr(
        lambda operand, update, start: lax.dynamic_update_slice(
            operand, update, (start,)
        )
    )(
        jax.ShapeDtypeStruct((10,), np.float32),
        jax.ShapeDtypeStruct((4,), np.float32),
        np.int32(3),
    )
    eqn = next(
        eqn for eqn in closed.jaxpr.eqns if eqn.primitive.name == "dynamic_update_slice"
    )
    concrete: ConcreteEnv = {eqn.invars[2]: np.int32(3)}  # ty: ignore[invalid-assignment]
    requested = np.array([1, 3, 5, 9], dtype=np.int64)

    full = resolve_dynamic_update_slice_route(eqn, concrete)
    fragment = resolve_dynamic_update_slice_route_fragment(
        eqn, concrete, RouteRequest(requested)
    )

    assert full is not None
    assert isinstance(fragment, DynamicUpdateSliceRouteFragment)
    expected_updates = np.flatnonzero(np.isin(full.target_rows, requested))
    np.testing.assert_array_equal(fragment.output_rows, requested)
    np.testing.assert_array_equal(fragment.update_rows, expected_updates)
    np.testing.assert_array_equal(
        fragment.target_rows, full.target_rows[expected_updates]
    )
