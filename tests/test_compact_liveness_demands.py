import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.handlers import _inverse_elementwise_demand
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ArrayRows,
    AxisProduct,
    ContributionRows,
    Full,
    Points,
    RangeRows,
    TensorDemand,
    merge_contribution_rows,
    merge_demands,
    plan_local_jaxpr,
    reshape_subset,
    union_tensor_subsets,
)
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


def _state_for(fn, value):
    closed = jax.make_jaxpr(fn)(value)
    plan = _JaxprAnalyzer(closed).analyze()
    state = TraceState(plan.n_dofs, plan.active_ids, plan.sub_info)
    state.attach_concrete_values(closed, [np.asarray(value)])
    state.seed_input_dependencies(closed)
    state.run_bound_eqns(plan.bound_eqns, CouplingAccumulator(plan.n_dofs))
    return plan, state


def test_compact_demand_merges_do_not_materialize_full_rows():
    all_rows = ContributionRows(AllRows(1_000_000))
    assert merge_contribution_rows(all_rows, ContributionRows(RangeRows(10, 20))) is all_rows
    assert merge_contribution_rows(
        ContributionRows(RangeRows(2, 5)), ContributionRows(RangeRows(5, 9))
    ) == ContributionRows(RangeRows(2, 9))
    assert isinstance(
        merge_contribution_rows(
            ContributionRows(RangeRows(2, 5)), ContributionRows(ArrayRows(np.array([9])))
        ),
        ContributionRows,
    )


def test_local_planner_rejects_flat_contribution_rows():
    closed = jax.make_jaxpr(lambda x: x)(jnp.ones(2))
    _, state = _state_for(lambda x: x, jnp.ones(2))
    root = closed.jaxpr.outvars[0]
    with pytest.raises(TypeError, match="TensorDemand"):
        plan_local_jaxpr(
            closed.jaxpr,
            state,
            {root: ContributionRows(AllRows(2))},
            [root],
        )


def test_slice_and_full_reduction_preserve_compact_demands():
    plan, state = _state_for(lambda x: x[10:90], jnp.ones(100))
    eqn, handler, *_ = next(bound for bound in plan.bound_eqns if bound[0].primitive.name == "slice")
    demand = handler.plan_backward(
        eqn, state, (TensorDemand(Full((80,))),)
    ).in_demands[0]
    assert isinstance(demand.subset, AxisProduct)
    np.testing.assert_array_equal(demand.rows.to_array(), np.arange(10, 90))

    plan, state = _state_for(lambda x: jnp.sum(x.reshape(10, 10), axis=1), jnp.ones(100))
    eqn, handler, *_ = next(
        bound for bound in plan.bound_eqns if bound[0].primitive.name == "reduce_sum"
    )
    demand = handler.plan_backward(
        eqn, state, (TensorDemand(Full((10,))),)
    ).in_demands[0]
    assert demand == TensorDemand(Full((10, 10)))


def test_elementwise_broadcast_projects_axis_products_without_row_inference():
    output_demand = TensorDemand(
        AxisProduct(
            (4, 3, 2),
            (ArrayRows(np.array([1, 3])), AllRows(3), ArrayRows(np.array([0]))),
        )
    )

    same_shape = _inverse_elementwise_demand(output_demand, (4, 3, 2), (4, 3, 2))
    assert same_shape is output_demand

    broadcast = _inverse_elementwise_demand(output_demand, (4, 1, 2), (4, 3, 2))
    assert isinstance(broadcast.subset, AxisProduct)
    assert broadcast.subset.shape == (4, 1, 2)
    for actual, expected in zip(
        broadcast.subset.axes,
        (ArrayRows(np.array([1, 3])), AllRows(1), ArrayRows(np.array([0]))),
    ):
        np.testing.assert_array_equal(actual.to_array(), expected.to_array())
    assert _inverse_elementwise_demand(output_demand, (), (4, 3, 2)) == TensorDemand(
        Full(())
    )

    irregular = TensorDemand(Points((4, 3, 2), ArrayRows(np.array([6, 19]))))
    routed = _inverse_elementwise_demand(irregular, (4, 1, 2), (4, 3, 2))
    assert isinstance(routed.subset, Points)
    np.testing.assert_array_equal(routed.rows.to_array(), np.array([2, 7]))


def test_reshape_subset_preserves_only_exact_axis_products():
    assert isinstance(reshape_subset(Full((2, 3)), (2, 3), (3, 2)), Full)

    compatible = reshape_subset(
        AxisProduct((2, 3), (ArrayRows(np.array([0])), ArrayRows(np.array([0, 1])))),
        (2, 3),
        (3, 2),
    )
    assert isinstance(compatible, AxisProduct)
    np.testing.assert_array_equal(compatible.to_rows().to_array(), np.array([0, 1]))

    irregular = reshape_subset(
        AxisProduct((2, 3), (ArrayRows(np.array([1])), AllRows(3))),
        (2, 3),
        (3, 2),
    )
    assert isinstance(irregular, Points)
    np.testing.assert_array_equal(irregular.to_rows().to_array(), np.array([3, 4, 5]))


def test_tensor_subset_union_preserves_only_native_structure():
    shape = (10, 2, 2)
    blocks = AxisProduct(
        shape, (ArrayRows(np.array([1, 4])), AllRows(2), AllRows(2))
    )
    assert blocks.local_shape == (2, 2, 2)

    merged = union_tensor_subsets(
        AxisProduct(shape, (ArrayRows(np.array([1])), AllRows(2), AllRows(2))),
        AxisProduct(shape, (ArrayRows(np.array([4])), AllRows(2), AllRows(2))),
    )
    assert isinstance(merged, AxisProduct)
    np.testing.assert_array_equal(merged.to_rows().to_array(), blocks.to_rows().to_array())

    merged_demand = merge_demands(
        TensorDemand(
            AxisProduct(shape, (ArrayRows(np.array([1])), AllRows(2), AllRows(2)))
        ),
        TensorDemand(
            AxisProduct(shape, (ArrayRows(np.array([4])), AllRows(2), AllRows(2)))
        ),
    )
    assert isinstance(merged_demand.subset, AxisProduct)

    irregular = union_tensor_subsets(
        blocks,
        Points(shape, ArrayRows(np.array([0]))),
    )
    assert isinstance(irregular, Points)
    np.testing.assert_array_equal(
        irregular.to_rows().to_array(),
        np.array([0, 4, 5, 6, 7, 16, 17, 18, 19]),
    )
