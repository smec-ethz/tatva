from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.hijax import HiPrim
from jax._src.lax.control_flow import solves
from jax.extend.core import Primitive

from tatva.tracer.api import analyze
from tatva.tracer.core.nested import CustomJvpSpec
from tatva.tracer.program.analysis import analyze as analyze_jaxpr
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.custom_root import extract_custom_root_parameters
from tatva.tracer.program.derivatives import trace_form_derivatives
from tatva.tracer.program.forms import FormSpec
from tatva.tracer.support import SupportPreflightError, registration_issues

pytestmark = pytest.mark.skipif(
    not hasattr(solves, "CustomRoot"),
    reason="requires the Hijax CustomRoot ABI",
)


def _root(p, *, mode="direct"):
    def tangent(g, rhs):
        if mode == "direct":
            return rhs / g(jnp.ones_like(rhs))
        jacobian = jax.jacfwd(g) if mode == "jacfwd" else jax.jacrev(g)
        return jnp.linalg.solve(jacobian(jnp.zeros_like(rhs)), rhs)

    return jax.lax.custom_root(
        lambda x: x * x - p,
        jnp.ones_like(p),
        lambda f, guess: jax.lax.stop_gradient(jnp.sqrt(p)),
        tangent,
    )


def _distributed_function(fn, args, parts):
    distributed = analyze(fn, *args).distribute(parts=parts)
    ranks = tuple(distributed.rank(rank) for rank in range(parts))

    def execute(*values):
        outputs = []
        for rank in ranks:
            local_args, kwargs = rank.inputs(*values)
            outputs.append(rank(*local_args, **kwargs))
        return sum(outputs)

    return execute


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("mode", ["direct", "jacfwd", "jacrev"])
def test_custom_root_preserves_implicit_ad_and_jit(parts, mode):
    p = jnp.arange(4, dtype=jnp.float32) + 4

    def objective(values):
        return jnp.sum(_root(values, mode=mode))

    local = _distributed_function(objective, (p,), parts)
    tangent = jnp.arange(4, dtype=p.dtype) + 1
    np.testing.assert_allclose(jax.jit(local)(p), objective(p), rtol=1e-5)
    for actual, expected in zip(
        jax.jvp(local, (p,), (tangent,)),
        jax.jvp(objective, (p,), (tangent,)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(jax.grad(local)(p), jax.grad(objective)(p), rtol=1e-5)
    np.testing.assert_allclose(
        jax.hessian(local)(p), jax.hessian(objective)(p), rtol=1e-5
    )


def test_custom_root_implicit_hessian_support_ignores_stop_gradient_solver():
    p = jnp.arange(4, dtype=jnp.float32) + 4
    objective = lambda values: jnp.sum(_root(values))
    closed = jax.make_jaxpr(objective)(p)
    plan = analyze_jaxpr(closed.jaxpr)
    resolver, frame = ConcreteResolver.root(closed, (p,), plan)
    trace = trace_form_derivatives(
        plan, frame, resolver, FormSpec.energy(input_index=0)
    )
    np.testing.assert_array_equal(
        trace.hessian.toarray().astype(bool), np.eye(4, dtype=bool)
    )


def test_custom_root_keeps_derivative_only_captures_and_ignores_guess_derivatives():
    values = jnp.tile(jnp.arange(4, dtype=jnp.float32) + 4, 2)

    def objective(u):
        p, guess = u[:4], u[4:]
        # At this input guess is the exact root, but the solve program contains
        # no reference to p. Only implicit differentiation sees that capture.
        root = jax.lax.custom_root(
            lambda x: x - p,
            guess,
            lambda f, x: x,
            lambda g, rhs: rhs,
        )
        return jnp.sum(root)

    local = _distributed_function(objective, (values,), 2)
    np.testing.assert_allclose(local(values), objective(values))
    np.testing.assert_array_equal(jax.grad(local)(values), jax.grad(objective)(values))
    np.testing.assert_array_equal(jax.grad(local)(values)[:4], np.ones(4))
    np.testing.assert_array_equal(jax.grad(local)(values)[4:], np.zeros(4))


def test_custom_root_pytree_and_aux_have_zero_auxiliary_tangents():
    p = jnp.arange(4, dtype=jnp.float32) + 4

    def objective(values):
        root, aux = jax.lax.custom_root(
            lambda x: {"x": x["x"] ** 2 - values},
            {"x": jnp.ones_like(values)},
            lambda f, guess: (
                {"x": jax.lax.stop_gradient(jnp.sqrt(values))},
                {"diagnostic": values * 3, "count": jnp.array(2)},
            ),
            lambda g, rhs: {"x": rhs["x"] / g({"x": jnp.ones_like(values)})["x"]},
            has_aux=True,
        )
        return jnp.sum((root["x"] + aux["diagnostic"]) * aux["count"])

    local = _distributed_function(objective, (p,), 1)
    np.testing.assert_allclose(local(p), objective(p))
    np.testing.assert_allclose(jax.grad(local)(p), jax.grad(objective)(p), rtol=1e-5)
    np.testing.assert_allclose(
        jax.hessian(local)(p), jax.hessian(objective)(p), rtol=1e-5
    )


def test_custom_root_extraction_has_finite_primal_and_jvp_programs():
    p = jnp.ones(3) * 4
    closed = jax.make_jaxpr(_root)(p)
    eqn = next(e for e in closed.jaxpr.eqns if e.primitive.name == "call_hi_primitive")
    extracted = extract_custom_root_parameters(eqn)
    assert any(b.primal_output_index is not None for b in extracted.bindings)
    assert extract_custom_root_parameters(eqn) is extracted
    assert registration_issues(closed.jaxpr) == ()
    plan = analyze_jaxpr(closed.jaxpr)
    assert isinstance(plan.eqns[-1].nested.spec, CustomJvpSpec)


def test_custom_root_preflight_checks_derivative_callback():
    unsupported = Primitive("test_unsupported_root_tangent")
    unsupported.def_impl(lambda x: x)
    unsupported.def_abstract_eval(lambda x: x)

    def root(p):
        return jax.lax.custom_root(
            lambda x: x - p,
            jnp.ones_like(p),
            lambda f, guess: p,
            lambda g, rhs: unsupported.bind(rhs),
        )

    issues = registration_issues(jax.make_jaxpr(root)(jnp.ones(3)).jaxpr)
    assert len(issues) == 1
    assert issues[0].primitive == unsupported.name
    assert " / " in issues[0].location


def test_custom_root_preflight_checks_primal_callback():
    unsupported = Primitive("test_unsupported_root_solve")
    unsupported.def_impl(lambda x: x)
    unsupported.def_abstract_eval(lambda x: x)

    def root(p):
        return jax.lax.custom_root(
            lambda x: x - p,
            jnp.ones_like(p),
            lambda f, guess: unsupported.bind(p),
            lambda g, rhs: rhs,
        )

    issues = registration_issues(jax.make_jaxpr(root)(jnp.ones(3)).jaxpr)
    assert len(issues) == 1
    assert issues[0].primitive == unsupported.name


def test_custom_root_vmap_and_jit():
    p = jnp.arange(6, dtype=jnp.float32) + 4

    def objective(values):
        return jnp.sum(jax.vmap(jax.vmap(_root))(values.reshape(2, 3)))

    # Registration and structural analysis also descend through an enclosing
    # jit; executable contributions are rooted outside that call boundary.
    closed = jax.make_jaxpr(jax.jit(objective))(p)
    assert registration_issues(closed.jaxpr) == ()
    analyze_jaxpr(closed.jaxpr)
    local = _distributed_function(objective, (p,), 2)
    np.testing.assert_allclose(jax.jit(local)(p), objective(p), rtol=1e-5)
    np.testing.assert_allclose(jax.grad(local)(p), jax.grad(objective)(p), rtol=1e-5)
    np.testing.assert_allclose(
        jax.hessian(local)(p), jax.hessian(objective)(p), rtol=1e-5
    )


def test_custom_root_integer_captures_are_not_tangent_bindings():
    p = jnp.arange(4, dtype=jnp.float32) + 4
    scale = jnp.asarray(2, dtype=jnp.int32)

    def objective(values, coefficient):
        root = jax.lax.custom_root(
            lambda x: x * coefficient - values,
            jnp.ones_like(values),
            lambda f, guess: jax.lax.stop_gradient(values / coefficient),
            lambda g, rhs: rhs / g(jnp.ones_like(rhs)),
        )
        return jnp.sum(root)

    local = _distributed_function(objective, (p, scale), 1)
    np.testing.assert_allclose(local(p, scale), objective(p, scale))
    np.testing.assert_allclose(jax.grad(local)(p, scale), jax.grad(objective)(p, scale))
    np.testing.assert_allclose(
        jax.hessian(local)(p, scale), jax.hessian(objective)(p, scale)
    )


def test_effectful_custom_root_is_rejected_contextually():
    def objective(p):
        def solve(f, guess):
            jax.debug.print("root parameter: {}", p)
            return p

        return jnp.sum(
            jax.lax.custom_root(
                lambda x: x - p,
                jnp.ones_like(p),
                solve,
                lambda g, rhs: rhs,
            )
        )

    with pytest.raises(SupportPreflightError, match="effectful CustomRoot"):
        analyze(objective, jnp.ones(3))


def test_unknown_high_primitive_reports_wrapped_type():
    class Unknown(HiPrim):
        def __init__(self, aval):
            self.in_avals = (aval,)
            self.out_aval = aval
            self.params = {}
            super().__init__()

        def expand(self, x):
            return x

    def objective(x):
        return jnp.sum(Unknown(jax.typeof(x))(x))

    with pytest.raises(
        SupportPreflightError, match="Unknown.*no supported operation semantics"
    ):
        analyze(objective, jnp.ones(3))
