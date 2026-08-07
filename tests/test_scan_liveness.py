import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.partitioning import ContributionDemand
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


def test_scan_liveness_routes_selected_body_rows_to_parent_xs():
    """A scan output demand should retain only its iteration-local xs entries."""

    def fn(u):
        xs = u.reshape(1, 5000, 3, 2)
        _, ys = lax.scan(lambda _, x: ((), x[:, 0, 0:1]), (), xs)
        return ys

    values = jnp.ones(30_000)
    closed = jax.make_jaxpr(fn)(values)
    plan = _JaxprAnalyzer(closed).analyze()
    state = TraceState(plan.n_dofs, plan.active_ids, plan.sub_info)
    state.attach_concrete_values(closed, [np.asarray(values)])
    state.seed_input_dependencies(closed)
    state.run_bound_eqns(plan.bound_eqns, CouplingAccumulator(plan.n_dofs))

    scan_eqn, scan_handler, *_ = next(
        bound for bound in plan.bound_eqns if bound[0].primitive.name == "scan"
    )
    demand = ContributionDemand(np.array([0, 17, 4999], dtype=np.int64))
    in_demands = scan_handler.propagate_liveness_demand(
        scan_eqn, state, [demand]
    )

    np.testing.assert_array_equal(in_demands[0].rows, [0, 102, 29994])
    assert in_demands[0].rows.size < 30_000
