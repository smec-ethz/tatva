from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.extend.core import Var

from tatva.tracer.capture import make_captured_jaxpr
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteRequest
from tatva.tracer.program.analysis import EqnPlan, analyze
from tatva.tracer.program.concrete_resolver import (
    ConcreteResolver,
    DynamicRoutingError,
)
from tatva.tracer.program.materialize import materialize_plan


def _setup(fn, *args):
    captured = make_captured_jaxpr(fn, *args)
    plan = analyze(captured.jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )
    return captured, plan, resolver, frame


def _eqn(plan, primitive_name: str) -> EqnPlan:
    return next(
        eqn_plan
        for eqn_plan in plan.eqns
        if eqn_plan.eqn.primitive.name == primitive_name
    )


def _assert_fragment_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual is not None and expected is not None
    for field in fields(actual):
        lhs = getattr(actual, field.name)
        rhs = getattr(expected, field.name)
        if isinstance(lhs, np.ndarray):
            np.testing.assert_array_equal(lhs, rhs)
        else:
            assert lhs == rhs


def test_recursive_value_matches_legacy_materialization_and_is_memoized():
    def objective(dofs, raw_indices):
        indices = (2 * raw_indices + 1).astype(jnp.int32)
        return jnp.sum(dofs[indices])

    args = (
        jnp.arange(12.0),
        jnp.array([0, 2, 3], dtype=jnp.int32),
    )
    captured, plan, resolver, frame = _setup(objective, *args)
    gather = _eqn(plan, "gather")
    index_atom = gather.eqn.invars[1]
    assert isinstance(index_atom, Var)

    value = resolver.value(frame, index_atom)
    evaluated = resolver.stats.evaluated_eqns
    legacy = materialize_plan(captured.closed_jaxpr, captured.flat_args, plan)

    np.testing.assert_array_equal(value, legacy.concrete[index_atom])
    assert evaluated > 0
    assert resolver.value(frame, index_atom) is value
    assert resolver.stats.evaluated_eqns == evaluated
    assert resolver.stats.cache_hits > 0


def _gather_program(dofs, indices):
    return jnp.sum(dofs[(indices * 2 + 1).astype(jnp.int32)])


def _scatter_program(dofs, indices):
    routed = jnp.zeros_like(dofs).at[indices + 1].add(dofs[: indices.size])
    return jnp.sum(routed)


def _select_program(dofs, selector):
    return jnp.sum(lax.select_n(selector, dofs, -dofs))


def _dynamic_slice_program(dofs, start):
    return jnp.sum(lax.dynamic_slice(dofs, (start + 1,), (3,)))


def _dynamic_update_program(dofs, start):
    update = jnp.asarray([8.0, 9.0], dtype=dofs.dtype)
    updated = lax.dynamic_update_slice(dofs, update, (start + 1,))
    return jnp.sum(updated)


@pytest.mark.parametrize(
    ("fn", "args", "primitive_name", "requested"),
    [
        (
            _gather_program,
            (jnp.arange(12.0), jnp.array([0, 2, 3], dtype=jnp.int32)),
            "gather",
            [0, 2],
        ),
        (
            _scatter_program,
            (jnp.arange(12.0), jnp.array([0, 2, 3], dtype=jnp.int32)),
            "scatter-add",
            [1, 3],
        ),
        (
            _select_program,
            (
                jnp.arange(6.0),
                jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32),
            ),
            "select_n",
            [0, 3],
        ),
        (
            _dynamic_slice_program,
            (jnp.arange(10.0), jnp.int32(2)),
            "dynamic_slice",
            [0, 2],
        ),
        (
            _dynamic_update_program,
            (jnp.arange(10.0), jnp.int32(2)),
            "dynamic_update_slice",
            [1, 3, 8],
        ),
    ],
    ids=("gather", "scatter", "select", "dynamic-slice", "dynamic-update"),
)
def test_route_fragments_resolve_without_a_jaxpr_instance(
    fn,
    args,
    primitive_name,
    requested,
):
    captured, plan, resolver, frame = _setup(fn, *args)
    eqn_plan = _eqn(plan, primitive_name)
    request = RouteRequest(np.asarray(requested, dtype=np.int64))

    # This is the new Phase-4 path: no JaxprInstance is involved.
    actual = resolver.route_fragment(frame, eqn_plan, request)

    # The legacy materialized environment remains the correctness oracle.
    legacy = materialize_plan(captured.closed_jaxpr, captured.flat_args, plan)
    semantics = SEMANTICS.get_ordinary(eqn_plan.eqn.primitive)
    expected = semantics.route_fragment(eqn_plan.eqn, legacy.concrete, request)
    _assert_fragment_equal(actual, expected)


def test_transitive_dof_dependent_routing_is_rejected():
    def objective(dofs):
        index = jnp.asarray(dofs[0], dtype=jnp.int32).reshape(1)
        return jnp.sum(dofs[index])

    captured, plan, resolver, frame = _setup(objective, jnp.arange(8.0))
    gather = _eqn(plan, "gather")

    with pytest.raises(DynamicRoutingError, match="DOF input"):
        resolver.value(frame, plan.jaxpr.invars[0])
    with pytest.raises(DynamicRoutingError, match="DOF input"):
        resolver.route_fragment(
            frame,
            gather,
            RouteRequest(np.array([0], dtype=np.int64)),
        )
    with pytest.raises(DynamicRoutingError, match="input 0"):
        materialize_plan(captured.closed_jaxpr, captured.flat_args, plan)


def test_fallback_multi_output_equation_is_evaluated_once():
    def objective(dofs, matrix):
        lu, pivots, permutation = lax.linalg.lu(matrix)
        return jnp.sum(dofs) + jnp.sum(lu) + jnp.sum(pivots) + jnp.sum(permutation)

    matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    _captured, plan, resolver, frame = _setup(
        objective,
        jnp.arange(3.0),
        matrix,
    )
    lu = _eqn(plan, "lu")

    first = resolver.value(frame, lu.eqn.outvars[0])
    evaluated = resolver.stats.evaluated_eqns
    second = resolver.value(frame, lu.eqn.outvars[1])

    assert first.shape == (2, 2)
    assert second.shape == (2,)
    assert resolver.stats.evaluated_eqns == evaluated == 1
    expected = lax.linalg.lu(matrix)
    np.testing.assert_allclose(first, expected[0])
    np.testing.assert_array_equal(second, expected[1])


def test_closed_constant_and_literal_inputs_are_resolved():
    offset = jnp.array([1, 3], dtype=jnp.int32)

    def objective(dofs, base):
        return jnp.sum(dofs[(base + offset + 1).astype(jnp.int32)])

    captured, plan, resolver, frame = _setup(
        objective,
        jnp.arange(10.0),
        jnp.array([0, 1], dtype=jnp.int32),
    )
    assert captured.jaxpr.constvars
    gather = _eqn(plan, "gather")

    np.testing.assert_array_equal(
        resolver.value(frame, gather.eqn.invars[1]),
        np.array([[2], [5]], dtype=np.int32),
    )


def test_nested_concrete_producer_is_resolved_lazily():
    @jax.jit
    def shift(indices):
        return indices + 1

    def objective(dofs, indices):
        return jnp.sum(dofs[shift(indices)])

    _captured, plan, resolver, frame = _setup(
        objective,
        jnp.arange(8.0),
        jnp.array([0, 2], dtype=jnp.int32),
    )
    gather = _eqn(plan, "gather")

    fragment = resolver.route_fragment(
        frame,
        gather,
        RouteRequest(np.array([0], dtype=np.int64)),
    )
    assert fragment is not None
    np.testing.assert_array_equal(fragment.source_rows, [1])
    assert resolver.stats.frames_created == 2
    assert resolver.stats.frames_released == 1


def test_frame_and_equation_ownership_are_validated():
    captured, _plan, resolver, frame = _setup(
        _gather_program,
        jnp.arange(8.0),
        jnp.array([0, 2], dtype=jnp.int32),
    )
    other_plan = analyze(captured.jaxpr)
    foreign_eqn = _eqn(other_plan, "gather")

    with pytest.raises(ValueError, match="does not belong"):
        resolver.route_fragment(
            frame,
            foreign_eqn,
            RouteRequest(np.array([0], dtype=np.int64)),
        )
