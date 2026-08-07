import jax
import jax.numpy as jnp
import numpy as np

from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ContributionDemand,
    RangeRows,
    merge_demands,
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
    assert demand == ContributionDemand(RangeRows(10, 90))

    plan, state = _state_for(lambda x: jnp.sum(x.reshape(10, 10), axis=1), jnp.ones(100))
    eqn, handler, *_ = next(
        bound for bound in plan.bound_eqns if bound[0].primitive.name == "reduce_sum"
    )
    demand = handler.propagate_liveness_demand(
        eqn, state, [ContributionDemand(AllRows(10))]
    )[0]
    assert demand == ContributionDemand(AllRows(100))
