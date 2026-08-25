from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core

if not hasattr(lax, "stack_p"):
    lax.stack_p = core.Primitive("stack_compat")

from tatva.tracer import analyze as analyze_functional
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.derivatives import trace_form_derivatives
from tatva.tracer.program.forms import (
    CoordinateBlock,
    CoordinateRole,
    FormSpec,
    State,
    Test,
    Trial,
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


def test_scatter_mul_records_only_target_update_interactions():
    x = jnp.arange(6, dtype=jnp.float32) + 1
    targets = jnp.array([2, 0, 1])

    def energy(values):
        base = values[:3]
        updates = values[3:]
        return jnp.sum(base.at[targets].multiply(updates))

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((6, 6), dtype=bool)
    for update_row, target_row in enumerate(np.asarray(targets), start=3):
        expected[target_row, update_row] = True
        expected[update_row, target_row] = True
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_reduce_prod_records_distinct_pairs_within_each_reduction_bucket():
    x = jnp.arange(6, dtype=jnp.float32) + 1

    def energy(values):
        return jnp.sum(jnp.prod(values.reshape(2, 3), axis=1))

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((6, 6), dtype=bool)
    expected[:3, :3] = ~np.eye(3, dtype=bool)
    expected[3:, 3:] = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_cumprod_records_distinct_pairs_within_each_axis_fiber():
    x = jnp.arange(6, dtype=jnp.float32) + 1

    def energy(values):
        return jnp.sum(jnp.cumprod(values.reshape(2, 3), axis=1))

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((6, 6), dtype=bool)
    expected[:3, :3] = ~np.eye(3, dtype=bool)
    expected[3:, 3:] = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_reverse_cumprod_records_distinct_pairs_within_each_axis_fiber():
    x = jnp.arange(6, dtype=jnp.float32) + 1

    def energy(values):
        return jnp.sum(lax.cumprod(values.reshape(2, 3), axis=1, reverse=True))

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((6, 6), dtype=bool)
    expected[:3, :3] = ~np.eye(3, dtype=bool)
    expected[3:, 3:] = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_triangular_solve_hessian_is_batch_local_and_linear_in_rhs():
    x = jnp.arange(12, dtype=jnp.float32) + 1

    def energy(values):
        a = values[:8].reshape(2, 2, 2)
        b = values[8:].reshape(2, 2, 1)
        return jnp.sum(lax.linalg.triangular_solve(a, b, left_side=True))

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((12, 12), dtype=bool)
    for batch in range(2):
        a_rows = np.arange(batch * 4, batch * 4 + 4)
        b_rows = np.arange(8 + batch * 2, 8 + batch * 2 + 2)
        expected[np.ix_(a_rows, a_rows)] = True
        expected[np.ix_(a_rows, b_rows)] = True
        expected[np.ix_(b_rows, a_rows)] = True
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_lu_hessian_is_batch_local():
    x = jnp.arange(8, dtype=jnp.float32) + 1

    def energy(values):
        lu, _, _ = lax.linalg.lu(values.reshape(2, 2, 2))
        return jnp.sum(lu)

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((8, 8), dtype=bool)
    expected[:4, :4] = True
    expected[4:, 4:] = True
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_sort_hessian_is_conservative_within_each_slice_only():
    x = jnp.arange(12, dtype=jnp.float32) + 1

    def energy(values):
        keys = values[:6].reshape(2, 3)
        payload = values[6:].reshape(2, 3)
        _, sorted_payload = lax.sort((keys, payload), dimension=1, num_keys=1)
        return jnp.sum(sorted_payload)

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))

    expected = np.zeros((12, 12), dtype=bool)
    for rows in (
        np.asarray([0, 1, 2, 6, 7, 8]),
        np.asarray([3, 4, 5, 9, 10, 11]),
    ):
        expected[np.ix_(rows, rows)] = True
    np.testing.assert_array_equal(trace.hessian.toarray().astype(bool), expected)


def test_batched_inverse_hessian_does_not_couple_matrix_batches():
    n_batch = 3
    matrix_size = 2
    x = jnp.arange(n_batch * matrix_size**2, dtype=jnp.float32) / 100

    def energy(values):
        matrices = values.reshape(n_batch, matrix_size, matrix_size)
        matrices = matrices + 3.0 * jnp.eye(matrix_size)[None, :, :]
        return jnp.sum(jnp.linalg.inv(matrices) * matrices)

    trace = _trace_form(energy, x, form=FormSpec.energy(input_index=0))
    pattern = trace.hessian.toarray().astype(bool)

    expected = np.zeros_like(pattern)
    batch_size = matrix_size**2
    for batch in range(n_batch):
        rows = np.arange(batch * batch_size, (batch + 1) * batch_size)
        expected[np.ix_(rows, rows)] = True
    np.testing.assert_array_equal(pattern, expected)


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


def test_annotated_trial_and_test_inference():
    from tatva.tracer.capture import CallABI
    from tatva.tracer.program.forms import infer_form_spec

    def weak(u: Trial[jax.Array], v: Test[jax.Array], p: float) -> jax.Array:
        return jnp.sum(u * v * p)

    abi, _ = CallABI.from_call(weak, (jnp.zeros(4), jnp.zeros(4), 1.0), {})
    form = infer_form_spec(weak, abi)
    assert form is not None
    assert len(form.coordinates) == 2
    assert form.coordinates[0].name == "u"
    assert form.coordinates[0].role == CoordinateRole.COLUMN
    assert form.coordinates[0].value_source == ValueSource.EXTERNAL
    assert form.coordinates[1].name == "v"
    assert form.coordinates[1].role == CoordinateRole.ROW
    assert form.coordinates[1].value_source == ValueSource.ZERO


def test_annotated_analyze_end_to_end():
    n_elements = 6
    width = 2
    u = jnp.arange(n_elements * width, dtype=jnp.float32) + 1
    v = jnp.zeros_like(u)

    def weak(x: Trial, test: Test) -> jax.Array:
        return jnp.sum(x.reshape(n_elements, width) * test.reshape(n_elements, width))

    distribution = analyze_functional(weak, u, v).distribute(
        parts=2,
        blocks_per_part=2,
    )

    for rank in range(distribution.parts):
        derivative = distribution.rank(rank).derivatives()
        assert derivative.row_block_names == ("test",)
        assert derivative.column_block_names == ("x",)
        np.testing.assert_array_equal(
            derivative.tangent.toarray().astype(bool),
            np.eye(derivative.tangent.shape[0], dtype=bool),
        )


def test_annotated_test_may_precede_storage_backed_trial_input():
    n_elements = 8
    width = 2
    trial = jnp.arange(n_elements * width, dtype=jnp.float32) + 1
    test = jnp.arange(n_elements * width, dtype=jnp.float32) + 2

    def weak(v: Test, u: Trial) -> jax.Array:
        return jnp.sum(v.reshape(n_elements, width) * u.reshape(n_elements, width))

    distribution = analyze_functional(weak, test, trial).distribute(
        parts=2,
        blocks_per_part=2,
    )

    local_values = []
    for rank in range(distribution.parts):
        local = distribution.rank(rank)
        assert local.dof_input_index == 1
        localized = local.localize(test, trial)
        assert localized.args[0].size < test.size
        assert localized.args[1].size < trial.size
        local_values.append(local.compile()(*localized.args, **localized.kwargs))
        derivative = local.derivatives()
        assert derivative.row_block_names == ("v",)
        assert derivative.column_block_names == ("u",)

    np.testing.assert_allclose(sum(local_values), weak(test, trial))


def test_annotated_state_may_follow_a_noncoordinate_input():
    n_elements = 8
    width = 2
    scale = jnp.asarray(1.5, dtype=jnp.float32)
    state = jnp.arange(n_elements * width, dtype=jnp.float32) + 1

    def energy(coefficient, values: State) -> jax.Array:
        local = values.reshape(n_elements, width)
        return coefficient * jnp.sum(local * local)

    distribution = analyze_functional(energy, scale, state).distribute(
        parts=2,
        blocks_per_part=2,
    )

    local_values = []
    for rank in range(distribution.parts):
        local = distribution.rank(rank)
        assert local.dof_input_index == 1
        localized = local.localize(scale, state)
        assert localized.args[0] is None
        assert localized.args[1].size < state.size
        local_values.append(local.compile()(*localized.args, **localized.kwargs))
        assert local.derivatives().hessian.shape[0] == localized.args[1].size

    np.testing.assert_allclose(sum(local_values), energy(scale, state))
