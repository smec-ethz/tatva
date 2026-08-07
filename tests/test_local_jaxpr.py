import jax
import jax.numpy as jnp
import numpy as np
from jax.core import eval_jaxpr
from jax import lax

from tatva.compound import Compound, field
from tatva.lifter import Fixed, Lifter, Periodic
from tatva.mesh import Mesh
from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ArrayRows,
    ContributionDemand,
    LocalJaxprPlanner,
    RangeRows,
    materialize_local_jaxpr,
    pack_runtime_inputs,
)
from tatva.sparse.tracer.registry import TR
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


def _local_result(fn, value, rows, *, root_index=-1):
    closed = jax.make_jaxpr(fn)(value)
    analysis = _JaxprAnalyzer(closed).analyze()
    state = TraceState(analysis.n_dofs, analysis.active_ids, analysis.sub_info)
    state.attach_concrete_values(closed, [np.asarray(value)])
    state.seed_input_dependencies(closed)
    state.run_bound_eqns(analysis.bound_eqns, CouplingAccumulator(analysis.n_dofs))
    root = closed.jaxpr.outvars[root_index]
    plan = LocalJaxprPlanner(TR).plan_jaxpr(
        closed.jaxpr,
        state,
        {root: ContributionDemand(ArrayRows(np.asarray(rows, dtype=np.int64)))},
        [root],
    )
    program = materialize_local_jaxpr(plan, closed.consts)
    (result,) = eval_jaxpr(
        program.jaxpr, program.consts, *pack_runtime_inputs(program, [value])
    )
    return np.asarray(result), plan, program


def test_rowsets_normalize_and_localize_without_dense_full_storage():
    assert isinstance(ContributionDemand(np.array([1, 3])).rows, ArrayRows)
    np.testing.assert_array_equal(ArrayRows(np.array([2, 5])).localize([5, 2]), [1, 0])
    np.testing.assert_array_equal(RangeRows(4, 7).localize([4, 6]), [0, 2])
    np.testing.assert_array_equal(AllRows(10).localize([0, 9]), [0, 9])


def test_full_nonscalar_local_storage_is_flat():
    result, plan, program = _local_result(
        lambda x: x.reshape(2, 3), jnp.arange(6.0), np.arange(6)
    )
    np.testing.assert_allclose(result, np.arange(6.0))
    assert plan.layouts[plan.requested_outputs[0]].local_aval.shape == (6,)
    assert program.input_specs[0].layout.local_aval.shape == (6,)


def test_local_jaxpr_evaluates_requested_intermediate_rows():
    def fn(x):
        squared = x * x
        return squared, jnp.sum(squared)

    result, plan, program = _local_result(fn, jnp.arange(6.0), [1, 4], root_index=0)
    np.testing.assert_allclose(result, [1.0, 16.0])
    assert program.output_specs[0].original_var is plan.requested_outputs[0]
    assert program.input_specs[0].layout.local_size == 2


def test_local_jaxpr_routes_transpose_slice_and_scalar_broadcast():
    result, _, _ = _local_result(
        lambda x: jnp.transpose(x.reshape(2, 3))[:, 1:],
        jnp.arange(6.0),
        [0, 2],
    )
    np.testing.assert_allclose(result, [3.0, 5.0])

    result, _, _ = _local_result(
        lambda x: jnp.broadcast_to(x[0], (2, 3)),
        jnp.arange(5.0),
        [0, 2, 5],
    )
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    # This requires a non-contiguous compact selection, exercising generated
    # gather-index constvars in LocalJaxprBuilder.
    result, _, program = _local_result(lambda x: x[::-1], jnp.arange(6.0), [0, 2])
    np.testing.assert_allclose(result, [5.0, 3.0])
    assert len(program.consts) == 1


def test_local_jaxpr_lifter_literal():
    lifter = Lifter.make(
        100,
    )

    def fn(x):
        x = lifter.lift_from_zeros(x)
        return x[4:] * x[:-4]

    result, _, _ = _local_result(fn, jnp.ones(lifter.size_reduced), [4, 5])
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_local_jaxpr_lifter_periodic_fail():
    n = 4
    mesh = Mesh.unit_square(n, n)

    class Solution(Compound, mesh=mesh):
        u = field((-1, 2))

    bottom = np.where(np.isclose(mesh.coords[:, 1], 0))[0]
    top = np.where(np.isclose(mesh.coords[:, 1], 1))[0]
    right = np.where(mesh.coords[:, 0] == 1)[0]
    left = np.where(mesh.coords[:, 0] == 0)[0]
    corner_0 = np.where((mesh.coords[:, 0] == 0) & (mesh.coords[:, 1] == 0))[0]

    lifter = Lifter.make(
        # lifter = Lifter(
        mesh.coords.shape[0] * 2,
        Fixed(Solution.u[corner_0]),
        # Periodic(Solution.u[right, :], Solution.u[left, :]),
        Periodic(Solution.u[top, :], Solution.u[bottom, :]),
    )

    def fn(x):
        x = lifter.lift_from_zeros(x)
        return x[4:] * x[:-4]

    # fmt: off
    rows = np.array([ 2,  6,  9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34, 37,
        38, 41, 42, 45])
    # fmt: on

    x = jnp.ones(lifter.size_reduced)
    result, _, _ = _local_result(fn, x, rows)
    np.testing.assert_allclose(result, np.asarray(fn(x)).reshape(-1)[rows])


def test_local_jaxpr_specializes_gather_routes():
    result, _, program = _local_result(
        lambda x: x[jnp.array([4, 1, 3])], jnp.arange(7.0), [0, 2]
    )
    np.testing.assert_allclose(result, [4.0, 3.0])
    # The original global index array is no longer an input or constvar.
    assert len(program.input_specs) == 1


def test_local_jaxpr_emits_compact_iota_and_elementwise_predicates():
    result, _, program = _local_result(
        lambda x: x + jnp.arange(x.size), jnp.ones(8), [2, 6]
    )
    np.testing.assert_allclose(result, [3.0, 7.0])
    assert len(program.consts) == 1  # specialized compact iota values

    result, _, _ = _local_result(lambda x: x > 3, jnp.arange(8), [2, 6])
    np.testing.assert_array_equal(result, [False, True])


def test_local_jaxpr_reduces_each_selected_output_row():
    result, _, _ = _local_result(
        lambda x: jnp.sum(x.reshape(3, 4), axis=1), jnp.arange(12.0), [0, 2]
    )
    np.testing.assert_allclose(result, [6.0, 38.0])

    result, _, _ = _local_result(
        lambda x: jnp.sum(x.reshape(3, 4), axis=0), jnp.arange(12.0), [1, 3]
    )
    np.testing.assert_allclose(result, [15.0, 21.0])


def test_local_jaxpr_emits_selected_dot_general_rows():
    result, _, _ = _local_result(
        lambda x: x.reshape(2, 3) @ jnp.array([2.0, 3.0, 5.0]),
        jnp.arange(6.0),
        [1],
    )
    np.testing.assert_allclose(result, [43.0])


def test_local_jaxpr_specializes_dynamic_slice_starts():
    result, _, program = _local_result(
        lambda x: lax.dynamic_slice(x, (x[0].astype(jnp.int32),), (3,)),
        jnp.array([2.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        [0, 2],
    )
    np.testing.assert_allclose(result, [20.0, 40.0])
    # The dynamic start value is specialized and is not a local input.
    assert program.input_specs[0].layout.rows.to_array().tolist() == [2, 4]


def test_local_jaxpr_rewrites_nested_jit():
    @jax.jit
    def inner(x):
        return jnp.sin(x * 2)

    result, _, _ = _local_result(lambda x: inner(x), jnp.arange(6.0), [1, 4])
    np.testing.assert_allclose(result, np.sin(np.array([2.0, 8.0])))
