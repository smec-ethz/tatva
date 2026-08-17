import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tatva.tracer.api import trace
from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.core.nested import LinearSolveSpec
from tatva.tracer.local.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local.plan import build_local_plan
from tatva.tracer.lowering.executor import build_local_executable
from tatva.tracer.lowering.partition import partition_contributions


def _find_linear_solve(plan):
    for eqn in plan.eqns:
        if eqn.eqn.primitive is lax.linear_solve_p:
            return eqn
        if eqn.nested:
            for child in eqn.nested.branches:
                found = _find_linear_solve(child)
                if found:
                    return found
    return None


def _local_executable(fn, dofs):
    traced = trace(CapturedJaxpr.from_fn(fn, dofs))
    owned = partition_contributions(traced.contributions, n_parts=1).for_part(0)
    demand = backpropagate_demand(
        traced.resolved,
        tuple(
            DemandSeed(traced.contributions.root(item.root_id).value, item.demand)
            for item in owned
        ),
    )
    plan = build_local_plan(traced.resolved, demand)
    executable = build_local_executable(
        plan, contributions=traced.contributions, owned=owned
    )
    return lambda x: executable(*executable.pack_global_inputs(x))


def _partitioned_local_executables(fn, dofs, n_parts):
    traced = trace(CapturedJaxpr.from_fn(fn, dofs))
    partition = partition_contributions(traced.contributions, n_parts=n_parts)
    result = []
    for part in range(n_parts):
        owned = partition.for_part(part)
        demand = backpropagate_demand(
            traced.resolved,
            tuple(
                DemandSeed(traced.contributions.root(item.root_id).value, item.demand)
                for item in owned
            ),
        )
        executable = build_local_executable(
            build_local_plan(traced.resolved, demand),
            contributions=traced.contributions,
            owned=owned,
        )
        result.append(
            lambda x, executable=executable: executable(
                *executable.pack_global_inputs(x)
            )
        )
    return tuple(result)


def test_linear_solve_is_three_callback_nested_operation_and_lowers_implicitly():
    fn = lambda u: jnp.sum(u + jnp.linalg.inv(2.0 * jnp.eye(2))[0, 0])
    u = jnp.ones(1)
    captured = CapturedJaxpr.from_fn(fn, u)
    traced = trace(captured)

    nested = _find_linear_solve(traced.analysis).nested
    assert isinstance(nested.spec, LinearSolveSpec)
    assert [callback.name for callback in nested.spec.callbacks()] == [
        "matvec",
        "solve",
        "transpose_solve",
    ]
    assert len(nested.branches) == 3

    local_fn = _local_executable(fn, u)

    assert jnp.allclose(local_fn(u), fn(u))
    assert jnp.allclose(jax.grad(local_fn)(u), jax.grad(fn)(u))
    assert "custom_linear_solve" in str(jax.make_jaxpr(local_fn)(u))


def test_local_custom_linear_solve_preserves_implicit_ad_not_solve_body_ad():
    def implicit_scalar_solve(a, b):
        def matvec(x):
            return a * x

        def solve(_matvec, rhs):
            # Correct primal, deliberately wrong if differentiated directly.
            return lax.stop_gradient(rhs / a)

        def transpose_solve(_vecmat, rhs):
            return lax.stop_gradient(rhs / a)

        return lax.custom_linear_solve(
            matvec, b, solve=solve, transpose_solve=transpose_solve
        )

    a = jnp.asarray(2.0)
    b = jnp.asarray(3.0)
    global_da, global_db = jax.grad(implicit_scalar_solve, argnums=(0, 1))(a, b)
    np.testing.assert_allclose(global_da, -b / a**2)
    np.testing.assert_allclose(global_db, 1 / a)

    # Tatva traces a rank-1 DOF input; expose the same scalar ABI for the
    # global/local AD comparisons below.
    dof_fn = lambda values: jnp.sum(
        jnp.reshape(implicit_scalar_solve(values[0], values[1]), (1,))
    )
    local_dof_fn = _local_executable(dof_fn, jnp.asarray([a, b]))
    local_implicit_scalar_solve = lambda aa, bb: local_dof_fn(jnp.asarray([aa, bb]))

    local_da, local_db = jax.grad(local_implicit_scalar_solve, argnums=(0, 1))(a, b)
    np.testing.assert_allclose(local_da, global_da)
    np.testing.assert_allclose(local_db, global_db)

    direction = (global_da, global_db)
    global_primal, global_tangent = jax.jvp(implicit_scalar_solve, (a, b), direction)
    local_primal, local_tangent = jax.jvp(
        local_implicit_scalar_solve, (a, b), direction
    )
    np.testing.assert_allclose(local_primal, global_primal)
    np.testing.assert_allclose(local_tangent, global_tangent)

    global_d2 = jax.grad(jax.grad(implicit_scalar_solve, argnums=0), argnums=0)(a, b)
    local_d2 = jax.grad(jax.grad(local_implicit_scalar_solve, argnums=0), argnums=0)(
        a, b
    )
    np.testing.assert_allclose(global_d2, 2 * b / a**3)
    np.testing.assert_allclose(local_d2, global_d2)


def test_partitioned_custom_linear_solve_matches_global_value_and_derivatives():
    def implicit_scalar_solve(a, b):
        def matvec(x):
            return a * x

        def solve(_matvec, rhs):
            return lax.stop_gradient(rhs / a)

        def transpose_solve(_vecmat, rhs):
            return lax.stop_gradient(rhs / a)

        return lax.custom_linear_solve(
            matvec, b, solve=solve, transpose_solve=transpose_solve
        )

    def fn(values):
        solutions = jax.vmap(implicit_scalar_solve)(values[:2], values[2:])
        return jnp.sum(solutions)

    values = jnp.asarray([2.0, 4.0, 3.0, 5.0])
    local_parts = _partitioned_local_executables(fn, values, n_parts=2)
    local_fn = lambda x: sum(part(x) for part in local_parts)

    np.testing.assert_allclose(local_fn(values), fn(values))
    np.testing.assert_allclose(jax.grad(local_fn)(values), jax.grad(fn)(values))
    np.testing.assert_allclose(
        jax.jvp(local_fn, (values,), (jnp.ones_like(values),)),
        jax.jvp(fn, (values,), (jnp.ones_like(values),)),
    )
