import jax
import jax.numpy as jnp
import numpy as np

from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ArrayRows,
    AxisProduct,
    ContributionDemand,
    Full,
    Points,
    RangeRows,
    TensorSubset,
    merge_demands,
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
    all_rows = ContributionDemand(AllRows(1_000_000))
    assert merge_demands(all_rows, ContributionDemand(RangeRows(10, 20))) is all_rows
    assert merge_demands(
        ContributionDemand(RangeRows(2, 5)), ContributionDemand(RangeRows(5, 9))
    ) == ContributionDemand(RangeRows(2, 9))
    assert isinstance(
        merge_demands(ContributionDemand(RangeRows(2, 5)), ContributionDemand(np.array([9]))),
        ContributionDemand,
    )


def test_slice_and_full_reduction_preserve_compact_demands():
    plan, state = _state_for(lambda x: x[10:90], jnp.ones(100))
    eqn, handler, *_ = next(bound for bound in plan.bound_eqns if bound[0].primitive.name == "slice")
    demand = handler.propagate_liveness_demand(
        eqn, state, [ContributionDemand(AllRows(80))]
    )[0]
    assert isinstance(demand.subset, AxisProduct)
    np.testing.assert_array_equal(demand.rows.to_array(), np.arange(10, 90))

    plan, state = _state_for(lambda x: jnp.sum(x.reshape(10, 10), axis=1), jnp.ones(100))
    eqn, handler, *_ = next(
        bound for bound in plan.bound_eqns if bound[0].primitive.name == "reduce_sum"
    )
    demand = handler.propagate_liveness_demand(
        eqn, state, [ContributionDemand(Full((10,)))]
    )[0]
    assert demand == ContributionDemand(Full((10, 10)))


def test_tensor_subset_inference_and_union_are_exact():
    shape = (10, 2, 2)
    blocks = TensorSubset.infer_from_rows(
        shape, ArrayRows(np.array([4, 5, 6, 7, 16, 17, 18, 19]))
    )
    assert isinstance(blocks, AxisProduct)
    assert blocks.local_shape == (2, 2, 2)

    product = TensorSubset.infer_from_rows(
        (4, 5), ArrayRows(np.array([1, 3, 11, 13]))
    )
    assert isinstance(product, AxisProduct)
    assert product.local_shape == (2, 2)

    merged = union_tensor_subsets(
        AxisProduct(shape, (ArrayRows(np.array([1])), AllRows(2), AllRows(2))),
        AxisProduct(shape, (ArrayRows(np.array([4])), AllRows(2), AllRows(2))),
    )
    assert isinstance(merged, AxisProduct)
    np.testing.assert_array_equal(merged.to_rows().to_array(), blocks.to_rows().to_array())

    merged_demand = merge_demands(
        ContributionDemand(
            AxisProduct(shape, (ArrayRows(np.array([1])), AllRows(2), AllRows(2)))
        ),
        ContributionDemand(
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

    assert isinstance(TensorSubset.infer_from_rows(shape, AllRows(40)), Full)
