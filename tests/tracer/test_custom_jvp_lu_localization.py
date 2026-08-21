from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core

# Compatibility for the execution environment used here. The target JAX 0.11
# exposes lax.stack_p; older JAX versions do not.
if not hasattr(lax, "stack_p"):
    lax.stack_p = core.Primitive("stack_compat")

from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.localize import LocalDynamicGatherRoute
from tatva.tracer.local.plan import build_rank_local_plan, pending_routes
from tatva.tracer.lowering.executor import _execute_frame, _frame_outputs
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ValueRef
from tatva.tracer.support import require_registered_operations


def _custom_lu_function():
    @jax.custom_jvp
    def function(x):
        return jnp.sum(x, axis=(-1, -2))

    @function.defjvp
    def function_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        _, _, permutation = lax.linalg.lu(x)
        n_batch, n_cell, _, _ = x.shape
        batch = jnp.arange(n_batch, dtype=jnp.int32)[:, None, None]
        cell = jnp.arange(n_cell, dtype=jnp.int32)[None, :, None]
        pivoted = x[batch, cell, permutation, :]
        primal = jnp.sum(x, axis=(-1, -2))
        tangent = jnp.sum(pivoted * x_dot, axis=(-1, -2))
        return primal, tangent

    return function


def _localized(function, x, selected):
    closed = jax.make_jaxpr(function)(x)
    plan = analyze(closed.jaxpr)
    resolver, frame = ConcreteResolver.root(closed, (x,), plan)
    output_shape = tuple(closed.jaxpr.outvars[0].aval.shape)
    demand = TensorDemand.axis_selection(output_shape, 0, selected)
    assert demand is not None
    trace = backpropagate_plan_demand(
        plan,
        frame,
        resolver,
        [DemandSeed(ValueRef((), plan.jaxpr.outvars[0]), demand)],
    )
    return build_rank_local_plan(plan, frame, resolver, trace), trace


def test_custom_jvp_lu_pivot_gather_stays_in_selected_batch_domain():
    function = _custom_lu_function()
    n_batch = 8
    selected = np.asarray([1, 3, 6], dtype=np.int64)
    x = (
        jnp.arange(n_batch * 4 * 4, dtype=jnp.float32).reshape(n_batch, 1, 4, 4)
        + 2.0 * jnp.eye(4, dtype=jnp.float32)[None, None, :, :]
    )

    local, trace = _localized(function, x, selected)
    outer = trace.input_demands[0]
    assert outer is not None
    assert outer.size == selected.size * 4 * 4
    assert local.input_layouts[0] is not None
    assert local.input_layouts[0].local_shape == (selected.size, 1, 4, 4)
    assert pending_routes(local) == ()

    jvp_children = []
    for eqn in local.eqns:
        if eqn.nested is not None:
            jvp_children.extend(
                child.payload
                for child in eqn.nested.invocation.children()
                if child.logical_index == 1
            )
    assert len(jvp_children) == 1
    jvp_plan = jvp_children[0]

    lu = next(eqn for eqn in jvp_plan.eqns if eqn.primitive_name == "lu")
    assert lu.input_layouts[0] is not None
    assert lu.input_layouts[0].local_shape == (selected.size, 1, 4, 4)
    assert lu.output_layouts[2] is not None
    assert lu.output_layouts[2].local_shape == (selected.size, 1, 4)

    pivot_gather = next(
        eqn
        for eqn in jvp_plan.eqns
        if eqn.primitive_name == "gather"
        and eqn.route is not None
        and isinstance(eqn.route.local, LocalDynamicGatherRoute)
    )
    assert pivot_gather.input_layouts[0] is not None
    assert pivot_gather.input_layouts[0].local_shape == (selected.size, 1, 4, 4)
    assert pivot_gather.input_layouts[1] is not None
    assert pivot_gather.input_layouts[1].local_shape == (selected.size, 1, 4, 3)
    assert tuple(rebase.component for rebase in pivot_gather.route.local.rebases) == (0,)

    x_local = x[selected]

    def run_local(value):
        env = _execute_frame(local, (value,))
        return _frame_outputs(local, env)[0]

    np.testing.assert_allclose(
        np.asarray(run_local(x_local)),
        np.asarray(function(x)[selected]),
        rtol=1e-6,
        atol=1e-6,
    )

    local_grad = jax.grad(lambda value: jnp.sum(run_local(value)))(x_local)
    reference_grad = jax.grad(lambda value: jnp.sum(function(value)[selected]))(x)[selected]
    np.testing.assert_allclose(
        np.asarray(local_grad),
        np.asarray(reference_grad),
        rtol=1e-6,
        atol=1e-6,
    )

    local_jaxpr = jax.make_jaxpr(
        lambda flat: jnp.sum(run_local(flat.reshape(x_local.shape)))
    )(x_local.ravel())
    require_registered_operations(local_jaxpr.jaxpr)


def test_public_functional_custom_jvp_lu_distributes_without_global_lu():
    from tatva.tracer.api import analyze as analyze_functional

    local_energy = _custom_lu_function()
    n_elements = 12
    matrix_size = 4

    def energy(dofs):
        matrices = dofs.reshape(n_elements, 1, matrix_size, matrix_size)
        matrices = matrices + 2.0 * jnp.eye(matrix_size, dtype=dofs.dtype)[
            None, None, :, :
        ]
        return jnp.sum(local_energy(matrices))

    dofs = jnp.arange(n_elements * matrix_size * matrix_size, dtype=jnp.float32) / 100
    analysis = analyze_functional(energy, dofs)
    distribution = analysis.distribute(parts=3, blocks_per_part=2)

    local_values = []
    assembled_gradient = np.zeros(dofs.size, dtype=np.float32)

    for rank in range(distribution.parts):
        local = distribution.rank(rank)
        localized = local.localize(dofs)
        fn = local.compile()

        # The LU callback must execute on a strict subset of the global domain.
        assert local.dofs.compute_rows.size < dofs.size

        local_values.append(fn(*localized.args, **localized.kwargs))
        local_gradient = np.asarray(
            jax.grad(fn)(*localized.args, **localized.kwargs)
        )
        assembled_gradient[local.dofs.storage.global_dofs] += local_gradient

    np.testing.assert_allclose(
        np.asarray(sum(local_values)),
        np.asarray(energy(dofs)),
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        assembled_gradient,
        np.asarray(jax.grad(energy)(dofs)),
        rtol=1e-6,
        atol=1e-6,
    )
