"""Adapt supported Hijax operations to Tatva's nested callback ABI."""

from __future__ import annotations

from functools import lru_cache

import jax
from jax._src import ad_util
from jax._src.hijax import VmapOf
from jax._src.lax.control_flow import solves
from jax.extend.core import JaxprEqn

from tatva.tracer.core.nested import CallKind, CustomJvpBinding
from tatva.tracer.core.semantics import CallAnalysisSemantics, CallTarget
from tatva.tracer.program.custom_jvp import CustomJvpParameters


class _IdentityKey:
    """Cache immutable JAX operations without hashing their pytree parameters."""

    def __init__(self, prim):
        self.prim = prim

    def __hash__(self):
        return id(self.prim)

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.prim is other.prim


def high_primitive_kind(eqn: JaxprEqn) -> str:
    prim = eqn.params.get("_prim")
    root_type = getattr(solves, "CustomRoot", None)
    map_type = getattr(solves, "_RootLinearMap", None)
    base = _unmapped_primitive(prim)
    if root_type is not None and isinstance(base, root_type):
        return "root"
    if map_type is not None and isinstance(base, map_type):
        return "linear_map"
    cls = type(prim)
    raise NotImplementedError(
        f"call_hi_primitive wrapped type {cls.__module__}.{cls.__qualname__} "
        "has no supported operation semantics"
    )


def _unmapped_primitive(prim):
    while isinstance(prim, VmapOf):
        prim = prim.prim
    return prim


def _abstract_args(avals):
    return tuple(
        jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=a.weak_type) for a in avals
    )


def _jvp_at_solution(prim, primals, tangents, solution):
    if isinstance(prim, VmapOf):
        # A vmap needs actual array leaves. Restore float0 leaves to symbolic
        # zeros in the underlying root's JVP after entering the mapped scope.
        tangents = jax.tree.map(
            ad_util.instantiate,
            tangents,
            is_leaf=lambda value: isinstance(value, ad_util.Zero),
        )
        return jax.vmap(
            lambda ps, ts, sol: _jvp_at_solution(prim.prim, ps, ts, sol),
            in_axes=(prim.in_dims, prim.in_dims, prim.out_dim),
            out_axes=(prim.out_dim, prim.out_dim),
            **prim._vmap_params,
        )(primals, tangents, solution)

    class RootAtSolution(type(prim)):
        def __call__(self, *unused):
            return self.solution

    proxy = object.__new__(RootAtSolution)
    proxy.__dict__.update(prim.__dict__)
    proxy.solution = solution
    tangents = prim.in_tree.unflatten(
        [
            ad_util.Zero(aval.to_tangent_aval())
            if aval.to_tangent_aval().dtype == jax.dtypes.float0
            else value
            for aval, value in zip(
                prim.in_avals_flat,
                prim.in_tree.flatten_up_to(tangents),
                strict=True,
            )
        ]
    )
    return proxy.jvp(primals, tangents)


@lru_cache(maxsize=128)
def _extract_root(key: _IdentityKey) -> CustomJvpParameters:
    prim = key.prim
    if prim.effects:
        raise NotImplementedError("effectful CustomRoot callbacks are unsupported")
    if len(prim.in_avals) != 4:
        raise NotImplementedError("CustomRoot requires four operand groups")
    args = _abstract_args(prim.in_avals_flat)
    out_args = _abstract_args(prim.out_avals_flat)
    tangent_avals = tuple(a.to_tangent_aval() for a in prim.in_avals_flat)
    # float0 tangent operands have a different shape/dtype ABI from primals.
    # They are symbolic zeros rather than explicit custom-JVP inputs.
    differentiable = tuple(a.dtype != jax.dtypes.float0 for a in tangent_avals)
    tangent_indices = tuple(i for i, live in enumerate(differentiable) if live)
    tangent_args = _abstract_args(tuple(tangent_avals[i] for i in tangent_indices))

    def primal(*flat):
        return tuple(
            prim.out_tree.flatten_up_to(prim.expand(*prim.in_tree.unflatten(flat)))
        )

    # CustomRoot.lin calls self(...) to get the root. Supplying it as an
    # explicit callback input avoids recursively staging that same operation,
    # while retaining JAX's authoritative lin/linearized implementation.
    def jvp(*flat):
        n = len(args)
        m = len(tangent_indices)
        primals = prim.in_tree.unflatten(flat[:n])
        tangents_flat = [ad_util.Zero(a) for a in tangent_avals]
        for i, value in zip(tangent_indices, flat[n : n + m], strict=True):
            tangents_flat[i] = value
        tangents = prim.in_tree.unflatten(tangents_flat)
        values, dots = _jvp_at_solution(
            prim,
            primals,
            tangents,
            prim.out_tree.unflatten(flat[n + m :]),
        )
        dots_flat = prim.out_tree.flatten_up_to(dots)
        # The output-zero ABI is fixed below: only inexact root outputs have
        # nonzero tangents; aux outputs are zero by CustomRoot's definition.
        return tuple(prim.out_tree.flatten_up_to(values)) + tuple(
            ad_util.instantiate(value)
            for value, zero in zip(dots_flat, output_zeros, strict=True)
            if not zero
        )

    root_avals = (
        prim.out_aval[0] if _unmapped_primitive(prim).has_aux else prim.out_aval
    )
    root_count = len(jax.tree.leaves(root_avals))
    output_zeros = tuple(
        i >= root_count or a.to_tangent_aval().dtype == jax.dtypes.float0
        for i, a in enumerate(prim.out_avals_flat)
    )
    primal_closed = jax.make_jaxpr(primal)(*args)
    jvp_closed = jax.make_jaxpr(jvp)(*args, *tangent_args, *out_args)
    bindings = (
        tuple(CustomJvpBinding(i) for i in range(len(args)))
        + tuple(CustomJvpBinding(i, tangent=True) for i in tangent_indices)
        + tuple(CustomJvpBinding(primal_output_index=i) for i in range(len(out_args)))
    )
    if primal_closed.effects or jvp_closed.effects:
        raise NotImplementedError("effectful CustomRoot callbacks are unsupported")
    if len(primal_closed.jaxpr.outvars) != len(out_args):
        raise NotImplementedError("CustomRoot primal output ABI mismatch")
    if len(jvp_closed.jaxpr.invars) != len(bindings):
        raise NotImplementedError("CustomRoot JVP input ABI mismatch")
    if len(jvp_closed.jaxpr.outvars) != len(out_args) + sum(
        not z for z in output_zeros
    ):
        raise NotImplementedError("CustomRoot JVP output ABI mismatch")
    return CustomJvpParameters(
        primal_closed.jaxpr,
        jvp_closed.jaxpr,
        tuple(jvp_closed.consts),
        0,
        output_zeros,
        tuple(primal_closed.consts),
        bindings,
    )


def extract_custom_root_parameters(eqn: JaxprEqn) -> CustomJvpParameters:
    if high_primitive_kind(eqn) != "root":
        raise TypeError("expected a CustomRoot equation")
    prim = eqn.params["_prim"]
    if len(prim.in_avals_flat) != len(eqn.invars) or len(prim.out_avals_flat) != len(
        eqn.outvars
    ):
        raise NotImplementedError("CustomRoot outer operand ABI mismatch")
    return _extract_root(_IdentityKey(prim))


@lru_cache(maxsize=128)
def _expand_linear_map(key: _IdentityKey):
    prim = key.prim
    if prim.effects:
        raise NotImplementedError("effectful root linear maps are unsupported")

    def expand(*flat):
        args = prim.in_tree.unflatten(flat)
        return tuple(prim.out_tree.flatten_up_to(prim.expand(*args)))

    return jax.make_jaxpr(expand)(*_abstract_args(prim.in_avals_flat))


def root_linear_map_target(eqn: JaxprEqn) -> CallTarget:
    if high_primitive_kind(eqn) != "linear_map":
        raise TypeError("expected a root linear map equation")
    return CallTarget(_expand_linear_map(_IdentityKey(eqn.params["_prim"])))


ROOT_LINEAR_MAP_ANALYSIS = CallAnalysisSemantics(
    call_kind=CallKind.JIT,
    target=root_linear_map_target,
)
