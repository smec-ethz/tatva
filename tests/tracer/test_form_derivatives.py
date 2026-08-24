from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core

if not hasattr(lax, "stack_p"):
    lax.stack_p = core.Primitive("stack_compat")

from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.derivatives import trace_form_derivatives
from tatva.tracer.program.forms import (
    CoordinateBlock,
    CoordinateRole,
    FormSpec,
    ValueSource,
)


def _trace_form(fn, *args, form):
    closed = jax.make_jaxpr(fn)(*args)
    plan = analyze(closed.jaxpr)
    resolver, frame = ConcreteResolver.root(closed, args, plan)
    return trace_form_derivatives(plan, frame, resolver, form)


def _binary(pattern):
    result = pattern.astype(bool).astype(np.int8).tocsr()
    result.eliminate_zeros()
    return result


def test_energy_and_virtual_work_use_same_tangent_abstraction():
    n = 5
    u = jnp.arange(n, dtype=jnp.float32) + 1
    v = jnp.zeros_like(u)

    def energy(x):
        return 0.5 * jnp.sum(x * x)

    def weak(x, test):
        return jnp.sum(test * x)

    energy_trace = _trace_form(energy, u, form=FormSpec.energy(input_index=0, name="u"))
    weak_trace = _trace_form(
        weak,
        u,
        v,
        form=FormSpec(
            (
                CoordinateBlock("u", 0, CoordinateRole.COLUMN, ValueSource.EXTERNAL),
                CoordinateBlock("v", 1, CoordinateRole.ROW, ValueSource.ZERO),
            )
        ),
    )

    expected = np.eye(n, dtype=np.int8)
    np.testing.assert_array_equal(energy_trace.tangent.toarray(), expected)
    np.testing.assert_array_equal(energy_trace.hessian.toarray(), expected)
    np.testing.assert_array_equal(weak_trace.tangent.toarray(), expected)

    try:
        _ = weak_trace.hessian
    except AttributeError:
        pass
    else:
        raise AssertionError("weak form must not expose a Hessian alias")


def test_mixed_form_extracts_row_by_column_block_tangent():
    n = 4
    values = tuple(jnp.arange(n, dtype=jnp.float32) + shift for shift in range(4))

    def mixed(u, p, v, q):
        return jnp.sum(v * (u + p)) + jnp.sum(q * (u * p))

    trace = _trace_form(
        mixed,
        *values,
        form=FormSpec(
            (
                CoordinateBlock("u", 0, CoordinateRole.COLUMN),
                CoordinateBlock("p", 1, CoordinateRole.COLUMN),
                CoordinateBlock("v", 2, CoordinateRole.ROW, ValueSource.ZERO),
                CoordinateBlock("q", 3, CoordinateRole.ROW, ValueSource.ZERO),
            )
        ),
    )

    tangent = trace.tangent.toarray().astype(bool)
    assert tangent.shape == (2 * n, 2 * n)
    expected_block = np.eye(n, dtype=bool)
    np.testing.assert_array_equal(tangent[:n, :n], expected_block)  # v-u
    np.testing.assert_array_equal(tangent[:n, n:], expected_block)  # v-p
    np.testing.assert_array_equal(tangent[n:, :n], expected_block)  # q-u
    np.testing.assert_array_equal(tangent[n:, n:], expected_block)  # q-p


def test_custom_jvp_uses_tangent_program_for_second_order_support():
    @jax.custom_jvp
    def custom_square(x):
        return 0.5 * x * x

    @custom_square.defjvp
    def custom_square_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        primal = 0.5 * x * x
        tangent = x * x_dot
        return primal, tangent

    n = 6
    u = jnp.arange(n, dtype=jnp.float32) + 1

    def energy(x):
        return jnp.sum(custom_square(x))

    trace = _trace_form(energy, u, form=FormSpec.energy(input_index=0))

    np.testing.assert_array_equal(
        trace.hessian.toarray().astype(bool),
        np.eye(n, dtype=bool),
    )


def test_custom_jvp_override_can_remove_primal_hessian_support():
    @jax.custom_jvp
    def overridden(x):
        return x * x

    @overridden.defjvp
    def overridden_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return x * x, 3.0 * x_dot

    n = 5
    u = jnp.arange(n, dtype=jnp.float32) + 1

    def energy(x):
        return jnp.sum(overridden(x))

    trace = _trace_form(energy, u, form=FormSpec.energy(input_index=0))

    assert trace.hessian.nnz == 0


def test_distributed_weak_form_reuses_local_form_derivative_pipeline():
    from tatva.tracer import analyze_form

    n_elements = 6
    width = 2
    u = jnp.arange(n_elements * width, dtype=jnp.float32) + 1
    v = jnp.zeros_like(u)

    def weak(x, test):
        return jnp.sum(x.reshape(n_elements, width) * test.reshape(n_elements, width))

    form = FormSpec(
        (
            CoordinateBlock("u", 0, CoordinateRole.COLUMN),
            CoordinateBlock("v", 1, CoordinateRole.ROW, ValueSource.ZERO),
        )
    )
    distribution = analyze_form(form, weak, u, v).distribute(
        parts=2,
        blocks_per_part=2,
    )

    for rank in range(distribution.parts):
        derivative = distribution.rank(rank).derivatives()
        assert derivative.row_block_names == ("v",)
        assert derivative.column_block_names == ("u",)
        np.testing.assert_array_equal(
            derivative.tangent.toarray().astype(bool),
            np.eye(derivative.tangent.shape[0], dtype=bool),
        )
        np.testing.assert_array_equal(
            derivative.block_global_ids["v"],
            derivative.block_global_ids["u"],
        )


def test_packed_mixed_coordinate_selections_share_one_input_without_overlap():
    n = 3
    trial = jnp.arange(2 * n, dtype=jnp.float32) + 1
    test = jnp.zeros(2 * n, dtype=jnp.float32)

    def packed(x, y):
        u, p = x[:n], x[n:]
        v, q = y[:n], y[n:]
        return jnp.sum(v * (u + p)) + jnp.sum(q * (u * p))

    form = FormSpec(
        (
            CoordinateBlock("u", 0, CoordinateRole.COLUMN, selection=slice(0, n)),
            CoordinateBlock("p", 0, CoordinateRole.COLUMN, selection=slice(n, 2 * n)),
            CoordinateBlock("v", 1, CoordinateRole.ROW, ValueSource.ZERO, slice(0, n)),
            CoordinateBlock(
                "q", 1, CoordinateRole.ROW, ValueSource.ZERO, slice(n, 2 * n)
            ),
        )
    )
    trace = _trace_form(packed, trial, test, form=form)
    tangent = trace.tangent.toarray().astype(bool)
    expected = np.eye(n, dtype=bool)
    np.testing.assert_array_equal(tangent[:n, :n], expected)
    np.testing.assert_array_equal(tangent[:n, n:], expected)
    np.testing.assert_array_equal(tangent[n:, :n], expected)
    np.testing.assert_array_equal(tangent[n:, n:], expected)
