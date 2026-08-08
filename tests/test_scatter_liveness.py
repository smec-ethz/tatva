import jax
import jax.numpy as jnp
import numpy as np

from tatva.sparse.tracer.base import _JaxprAnalyzer
from tatva.sparse.tracer.partitioning import ArrayRows, Points, TensorDemand
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


def _scatter_demand(fn, *args, rows, concrete=True):
    closed = jax.make_jaxpr(fn)(*args)
    plan = _JaxprAnalyzer(closed).analyze()
    state = TraceState(plan.n_dofs, plan.active_ids, plan.sub_info)
    state.attach_concrete_values(closed, [np.asarray(arg) for arg in args])
    state.seed_input_dependencies(closed)
    state.run_bound_eqns(plan.bound_eqns, CouplingAccumulator(plan.n_dofs))
    eqn, handler, *_ = next(
        bound for bound in plan.bound_eqns if "scatter" in bound[0].primitive.name
    )
    if not concrete:
        state.val_of.pop(id(eqn.invars[1]), None)
    return handler.plan_backward(
        eqn,
        state,
        (
            TensorDemand(
                Points(
                    tuple(eqn.outvars[0].aval.shape),
                    ArrayRows(np.asarray(rows, dtype=np.int64)),
                )
            ),
        ),
    ).in_demands


def test_scatter_add_routes_only_updates_and_indices_for_live_outputs():
    def fn(x, indices, updates):
        return x.at[indices].add(updates)

    demands = _scatter_demand(
        fn, jnp.zeros(10), jnp.array([1, 3, 7]), jnp.ones(3), rows=[1, 7]
    )
    np.testing.assert_array_equal(demands[0].rows, [1, 7])
    np.testing.assert_array_equal(demands[1].rows, [0, 2])
    np.testing.assert_array_equal(demands[2].rows, [0, 2])


def test_scatter_repeated_indices_retain_every_contributing_update():
    def fn(x, indices, updates):
        return x.at[indices].add(updates)

    demands = _scatter_demand(
        fn, jnp.zeros(8), jnp.array([1, 1, 6]), jnp.ones(3), rows=[1]
    )
    np.testing.assert_array_equal(demands[1].rows, [0, 1])
    np.testing.assert_array_equal(demands[2].rows, [0, 1])


def test_scatter_routes_multidimensional_points_and_windows():
    def point(x, indices, updates):
        return x.reshape(4, 5).at[indices[:, 0], indices[:, 1]].add(updates)

    point_demands = _scatter_demand(
        point,
        jnp.zeros(20),
        jnp.array([[1, 2], [3, 4]]),
        jnp.ones(2),
        rows=[7],
    )
    np.testing.assert_array_equal(point_demands[1].rows, [0, 1])
    np.testing.assert_array_equal(point_demands[2].rows, [0])

    def window(x, indices, updates):
        return x.reshape(5, 4).at[indices, :].add(updates)

    window_demands = _scatter_demand(
        window,
        jnp.zeros(20),
        jnp.array([1, 3]),
        jnp.ones((2, 4)),
        rows=[5, 7],
    )
    np.testing.assert_array_equal(window_demands[1].rows, [0])
    np.testing.assert_array_equal(window_demands[2].rows, [1, 3])


def test_scatter_partial_large_demand_is_not_all_indices():
    n_indices = 5202

    def fn(x, indices, updates):
        return x.at[indices].add(updates)

    demands = _scatter_demand(
        fn,
        jnp.zeros(2 * n_indices),
        jnp.arange(n_indices),
        jnp.ones(n_indices),
        rows=np.arange(2646),
    )
    assert demands[1].rows.size == 2646
    assert demands[2].rows.size == 2646


def test_scatter_without_concrete_indices_falls_back_to_all_routes():
    def fn(x, indices, updates):
        return x.at[indices].add(updates)

    demands = _scatter_demand(
        fn,
        jnp.zeros(10),
        jnp.array([1, 3, 7]),
        jnp.ones(3),
        rows=[1],
        concrete=False,
    )
    np.testing.assert_array_equal(demands[1].rows, [0, 1, 2])
    np.testing.assert_array_equal(demands[2].rows, [0, 1, 2])
