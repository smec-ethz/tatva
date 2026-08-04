# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from jax.core import eval_jaxpr
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Primitive, Var

if TYPE_CHECKING:
    from tatva.sparse.tracer.state import SparseDepSet

# ---------------------------------------------------------------------------
# primitive classification
# ---------------------------------------------------------------------------

_NONLINEAR_UNARY = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "log2",
        "sqrt",
        "rsqrt",
        "cbrt",
        "tanh",
        "sinh",
        "cosh",
        "atanh",
        "asinh",
        "acosh",
        "erf",
        "erfc",
        "erfinv",
        "lgamma",
        "digamma",
        "logistic",
    }
)

_NONLINEAR_BINARY = frozenset(
    {
        "mul",
        "div",
        "rem",
        "pow",
        "atan2",
        "igamma",
        "igammac",
        "nextafter",
        "complex",
    }
)

# Dense nonlinear linear-algebra primitives: the leaf ops that back jnp.linalg.inv/solve/
# det(large)/cholesky/eig(h). Each is a dense nonlinear map of its matrix argument — every
# output entry depends on all input entries and those input DOFs mutually couple. Treated as
# one dense black box (see Handlers.dense_linalg); classified nonlinear so its inputs are
# traced and its curvature is recorded rather than silently dropped by the generic fallback.
_DENSE_LINALG = frozenset(
    {
        "lu",
        "custom_linear_solve",
        "triangular_solve",
        "lu_solve",
        "cholesky",
        "eig",
        "eigh",
    }
)

# FFI targets whose vmapped (batched) call is elementwise along the leading (vmap) axis,
# i.e. output[i] depends only on input[i] -- e.g. a per-quad-point external constitutive
# solver vmapped over quad points. Declaring a target here lets the tracer recover the
# sparse (block-diagonal) coupling from the single batched ``ffi_call`` instead of the
# conservative dense couple-all. See ``register_elementwise_ffi``.
_ELEMENTWISE_FFI_TARGETS: set[str] = set()


def register_elementwise_ffi(*target_names: str) -> None:
    """Declare one or more ``jax.ffi`` target names as elementwise along the vmap axis.

    A vmapped ``ffi_call`` is a *single* batched custom-call over the whole leading axis,
    so by default the tracer must treat it as a dense opaque block (it cannot tell the
    batch axis is independent). Registering the target tells the tracer that the call is a
    per-element map (output[i] depends only on input[i]) -- as it is for a per-quad-point
    external solver vmapped over quad points -- so it records conservative coupling *per
    slice* (block-diagonal across the leading axis), recovering the sparse pattern.

    Register by the FFI target name (the string passed to ``jax.ffi.ffi_call``) -- anyone
    building or using a ``jax.ffi`` solver knows this name. Then a per-quad-point external
    solver vmapped over quad points, ``jax.vmap(solver)(strain_field)``, traces sparse.

    Only register a target if this independence genuinely holds; otherwise real
    cross-element couplings would be silently missed.
    """
    _ELEMENTWISE_FFI_TARGETS.update(target_names)


def _get_shape(var: Var | Literal) -> tuple[int, ...]:
    """Helper to safely retrieve shape from a JAX Var/Literal abstract value (satisfying
    static type checkers)."""
    return getattr(var.aval, "shape", ())


def _subjaxpr_and_consts(eqn) -> tuple[Jaxpr, Sequence]:
    """Normalize the ``jaxpr`` param of a single-subgraph higher-order primitive.

    ``pjit``/``jit``/``scan``/``map`` store a ``ClosedJaxpr`` (``.jaxpr`` + ``.consts``),
    whereas ``remat2`` (``jax.checkpoint``) stores a bare ``Jaxpr`` with no consts.
    Both expose the same 1:1 invar/outvar correspondence with the parent equation, so
    this returns ``(jaxpr, consts)`` for either form.
    """
    sub = eqn.params["jaxpr"]
    if hasattr(sub, "jaxpr"):  # ClosedJaxpr
        return sub.jaxpr, sub.consts
    return sub, ()


def _unwrap_jit(fn):
    """Recursively unwrap `@jax.jit` / `@pjit` decorators only.

    `jax.grad`, `jax.vmap`, and `functools.wraps`-based wrappers also set `__wrapped__`,
    so blindly removing `__wrapped__` would strip semantic transforms (e.g., turning
    `jax.grad(E)` back into `E`). We only unwrap `PjitFunction`-class wrappers, which
    @jax.jit` produces — this preserves any `grad`/`vmap` layers underneath.
    """
    while type(fn).__name__ in ("PjitFunction", "JitWrapped") and hasattr(
        fn, "__wrapped__"
    ):
        fn = fn.__wrapped__
    return fn


def _broadcast_single_row(row: sps.csr_matrix, N: int) -> sps.csr_matrix:
    """Replicate a single-row CSR matrix N times, ~250x faster than sps.vstack([row]*N).

    Builds the result directly via a uniform indptr (each row has the same nnz),
    bypassing the O(N) Python overhead of sps.vstack.
    """
    if N <= 0:
        return sps.csr_matrix((0, row.shape[1]), dtype=row.dtype)
    m = row.nnz
    if m == 0:
        return sps.csr_matrix((N, row.shape[1]), dtype=row.dtype)
    indptr = np.arange(N + 1, dtype=row.indptr.dtype) * m
    indices = np.tile(row.indices, N)
    data = np.tile(row.data, N)
    return sps.csr_matrix((data, indices, indptr), shape=(N, row.shape[1]))


def _reduce_union_over_axes(
    dep: sps.csr_matrix, shape: tuple[int, ...], keep_axes: list[int]
) -> sps.csr_matrix:
    """OR-reduce a dep-array over all axes NOT in ``keep_axes``.

    ``dep`` is a (prod(shape), n_dofs) CSR whose rows are the row-major flattening of a
    tensor of logical ``shape``. Returns a (prod(shape[keep_axes]), n_dofs) CSR whose rows
    are the row-major flattening over ``keep_axes`` (in the given order), where each kept
    row is the union of the dependency sets of every original element that maps to it.

    This is the exact support map for a reduction (a summed-out axis makes the result
    depend on the union along it), so it only ever preserves or tightens support -- never
    drops a real dependency.
    """
    n_rows = dep.shape[0]
    if not keep_axes:
        # Reducing over everything -> single-row total union.
        return sps.csr_matrix(dep.sum(axis=0).astype(bool)) if n_rows else dep
    orig = np.arange(n_rows)
    multi = np.unravel_index(orig, shape) if shape else (np.zeros(n_rows, int),)
    keep_dims = tuple(shape[a] for a in keep_axes)
    key = np.ravel_multi_index(tuple(multi[a] for a in keep_axes), keep_dims)
    n_keys = int(np.prod(keep_dims))
    # Boolean aggregation matrix (n_keys x n_rows); (agg @ dep) unions the rows per key.
    agg = sps.csr_matrix(
        (np.ones(n_rows, dtype=np.int8), (key, orig)), shape=(n_keys, n_rows)
    )
    return (agg @ dep.astype(np.int8)).astype(bool).tocsr()


def _dot_general_out_dep(
    lhs: "SparseDepSet",
    rhs: "SparseDepSet",
    lhs_c: Sequence[int],
    rhs_c: Sequence[int],
    lhs_b: Sequence[int],
    rhs_b: Sequence[int],
    oshp: tuple[int, ...],
    n_dofs: int,
) -> "SparseDepSet":
    """Structure-preserving support propagation for a ``dot_general``.

    Output element ``out[b, fl, fr]`` (batch ``b``, lhs-free ``fl``, rhs-free ``fr``)
    depends on exactly ``(union_c lhs[b, fl, c]) | (union_c rhs[b, c, fr])``. We reduce
    each operand over its contracting axes, then broadcast the two reduced dep-tensors to
    the output layout ``[batch..., lhs_free..., rhs_free...]`` (JAX's ``dot_general`` output
    order) and union them. This keeps per-row (per-element) support instead of collapsing
    to the whole-array ``total_union`` -- the difference between a locally-supported field
    and one that appears to depend on every DOF.
    """
    from tatva.sparse.tracer.state import SparseDepSet

    La, Lb = lhs.shape, rhs.shape
    lhs_b, rhs_b = list(lhs_b), list(rhs_b)
    lhs_c, rhs_c = list(lhs_c), list(rhs_c)
    lhs_free = [a for a in range(len(La)) if a not in lhs_b and a not in lhs_c]
    rhs_free = [a for a in range(len(Lb)) if a not in rhs_b and a not in rhs_c]

    # Reduce out contracting axes, keeping (batch, free) in output order.
    lhs_red = _reduce_union_over_axes(lhs.dep, La, lhs_b + lhs_free)
    rhs_red = _reduce_union_over_axes(rhs.dep, Lb, rhs_b + rhs_free)

    batch_sizes = tuple(La[a] for a in lhs_b)
    lhs_free_sizes = tuple(La[a] for a in lhs_free)
    rhs_free_sizes = tuple(Lb[a] for a in rhs_free)
    out_dims = batch_sizes + lhs_free_sizes + rhs_free_sizes

    n_out = int(np.prod(oshp))
    if int(np.prod(out_dims)) != n_out:
        # Shape bookkeeping disagrees with the reported output shape; fall back to the
        # conservative whole-array union rather than risk a mismatch.
        combined = sps.csr_matrix((lhs.dep + rhs.dep).astype(bool))
        return SparseDepSet(
            _broadcast_single_row(
                sps.csr_matrix(combined.sum(axis=0).astype(bool)), n_out
            ),
            oshp,
        )

    nb, nlf = len(lhs_b), len(lhs_free)
    omulti = np.unravel_index(np.arange(n_out), out_dims) if out_dims else ()
    batch_m = omulti[:nb]
    lf_m = omulti[nb : nb + nlf]
    rf_m = omulti[nb + nlf :]

    lhs_keys = (
        np.ravel_multi_index(batch_m + lf_m, batch_sizes + lhs_free_sizes)
        if (batch_sizes + lhs_free_sizes)
        else np.zeros(n_out, int)
    )
    rhs_keys = (
        np.ravel_multi_index(batch_m + rf_m, batch_sizes + rhs_free_sizes)
        if (batch_sizes + rhs_free_sizes)
        else np.zeros(n_out, int)
    )
    out_dep = (
        (lhs_red[lhs_keys].astype(np.int8) + rhs_red[rhs_keys].astype(np.int8))
        .astype(bool)
        .tocsr()
    )
    return SparseDepSet(out_dep, oshp)


def _prim_introduces_nonlinearity(eqn: JaxprEqn, invar_active: list[bool]) -> bool:
    """Whether this primitive makes its output a *non-affine* function of ``u``.

    Mirrors exactly the sites where the tracer records second-order couplings, so a
    variable is flagged nonlinear iff a coupling-recording primitive touched it. Every
    other (structural / additive) primitive is affine-preserving and returns ``False`` --
    nonlinearity then only reaches the output via a nonlinear *input* (handled by the
    caller). ``invar_active[i]`` is whether input ``i`` depends on ``u``.
    """
    p = eqn.primitive.name
    if p in _NONLINEAR_UNARY:
        return bool(invar_active) and invar_active[0]
    if p == "integer_pow":
        y = eqn.params.get("y", 0)
        return (y >= 2 or y <= -1) and bool(invar_active) and invar_active[0]
    if p in ("mul", "scatter-mul"):
        # bilinear: nonlinear only if *both* factors depend on u (scaling by a
        # constant stays affine)
        return sum(invar_active) >= 2
    if p == "div":
        # x / c is affine; nonlinear only when the divisor depends on u
        return len(invar_active) > 1 and invar_active[1]
    if p in _NONLINEAR_BINARY:  # pow, rem, atan2, igamma, igammac, nextafter, complex
        return any(invar_active)
    if p == "dot_general":
        return sum(invar_active) >= 2
    if p in (
        "custom_vjp_call",
        "custom_jvp_call",
        "pure_callback",
        "io_callback",
        "ffi_call",
    ):
        return any(invar_active)
    if p in _DENSE_LINALG:
        return any(invar_active)
    return False


# ---------------------------------------------------------------------------
# concrete value propagation (needed to route gather indices exactly)
# ---------------------------------------------------------------------------


def _try_concrete(primitive: Primitive, in_vals, par: dict):
    """Evaluate primitive on numpy values; return None if any input is unknown."""
    if any(v is None for v in in_vals):
        return None
    p = primitive.name
    try:
        v = [np.asarray(x) for x in in_vals]
        if p == "add":
            return np.add(v[0], v[1])
        if p == "sub":
            return np.subtract(v[0], v[1])
        if p == "mul":
            return np.multiply(v[0], v[1])
        if p == "div":
            if np.issubdtype(v[0].dtype, np.integer) and np.issubdtype(
                v[1].dtype, np.integer
            ):
                return np.floor_divide(v[0], v[1])
            return np.true_divide(v[0], v[1])
        if p == "neg":
            return np.negative(v[0])
        if p == "abs":
            return np.abs(v[0])
        if p in ("lt", "lt_to"):
            return np.less(v[0], v[1])
        if p in ("le", "le_to"):
            return np.less_equal(v[0], v[1])
        if p in ("gt", "gt_to"):
            return np.greater(v[0], v[1])
        if p in ("ge", "ge_to"):
            return np.greater_equal(v[0], v[1])
        if p in ("eq", "eq_to"):
            return np.equal(v[0], v[1])
        if p in ("ne", "ne_to"):
            return np.not_equal(v[0], v[1])
        if p == "min":
            return np.minimum(v[0], v[1])
        if p == "max":
            return np.maximum(v[0], v[1])
        if p == "floor":
            return np.floor(v[0])
        if p == "ceil":
            return np.ceil(v[0])
        if p == "round":
            return np.round(v[0])
        if p == "integer_pow":
            return v[0] ** par["y"]
        if p == "convert_element_type":
            return v[0].astype(par["new_dtype"])
        if p == "reshape":
            return v[0].reshape(par["new_sizes"])
        if p == "transpose":
            return np.transpose(v[0], par["permutation"])
        if p == "squeeze":
            return np.squeeze(v[0], axis=tuple(par["dimensions"]))
        if p == "slice":
            ss, ls = par["start_indices"], par["limit_indices"]
            st = par["strides"] or [1] * len(ss)
            return v[0][tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))]
        if p == "concatenate":
            return np.concatenate(v, axis=par["dimension"])
        if p == "iota":
            dim = par["dimension"]
            shp = par["shape"]
            newshp = [1] * len(shp)
            newshp[dim] = shp[dim]
            return np.broadcast_to(np.arange(shp[dim]).reshape(newshp), shp).copy()
        if p == "broadcast_in_dim":
            shape = par["shape"]
            bdims = par["broadcast_dimensions"]
            x = v[0]
            newshp = [1] * len(shape)
            for i, b in enumerate(bdims):
                newshp[b] = x.shape[i] if x.ndim > 0 else 1
            return np.broadcast_to(x.reshape(newshp), shape).copy()
        if p == "select_n":
            cond, cases = v[0], v[1:]
            if len(cases) == 2:
                return np.where(cond.astype(bool), cases[1], cases[0])
            result = cases[0].copy()
            for i, case in enumerate(cases[1:], 1):
                result = np.where(cond == i, case, result)
            return result
        if p in ("pjit", "jit", "remat2"):
            sub = par["jaxpr"]
            jaxpr_body = sub.jaxpr if hasattr(sub, "jaxpr") else sub
            consts = sub.consts if hasattr(sub, "consts") else ()
            v_casted = []
            for x, invar in zip(v, jaxpr_body.invars):
                target_dtype = getattr(invar.aval, "dtype", None)
                if target_dtype is not None and x.dtype != target_dtype:
                    x = x.astype(target_dtype)
                v_casted.append(x)
            res = eval_jaxpr(jaxpr_body, consts, *v_casted)
            return np.asarray(res[0])

        # For other primitives, use jax itself to evaluate the primitive on concrete numpy
        # values. This is a fallback for primitives that don't have a specific
        # implementation above.
        res = np.asarray(primitive.bind(*[jnp.asarray(x) for x in v], **par))
        warnings.warn(f"Concrete evaluation through bind needed for {p}")
        return res
    except Exception:
        pass
    return None
