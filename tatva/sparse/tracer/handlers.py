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

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax.core
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.special as sp
from jax.extend.core import JaxprEqn, Primitive, Var
from numpy.typing import NDArray

from tatva.sparse.tracer.common import (
    _ELEMENTWISE_FFI_TARGETS,
    _broadcast_single_row,
    _get_shape,
    _reduce_union_over_axes,
    _subjaxpr_and_consts,
)
from tatva.sparse.tracer.registry import TR
from tatva.sparse.tracer.state import (
    CouplingAccumulator,
    DistributionState,
    SparseDepSet,
    SubEqnInfo,
    TraceState,
)


def _eval_reshape(x, params):
    return x.reshape(params["new_sizes"])


def _eval_squeeze(x, params):
    return np.squeeze(x, axis=tuple(params["dimensions"]))


def _eval_convert_dtype(x, params):
    return x.astype(params["new_dtype"])


def _eval_div(a, b, **_params):
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.integer) and np.issubdtype(
        b_arr.dtype, np.integer
    ):
        return np.floor_divide(a_arr, b_arr)
    return np.true_divide(a_arr, b_arr)


def _dot_general_out_dep(
    lhs: SparseDepSet,
    rhs: SparseDepSet,
    lhs_c: Sequence[int],
    rhs_c: Sequence[int],
    lhs_b: Sequence[int],
    rhs_b: Sequence[int],
    oshp: tuple[int, ...],
    n_dofs: int,
) -> SparseDepSet:
    """Structure-preserving support propagation for a ``dot_general``.

    Output element ``out[b, fl, fr]`` (batch ``b``, lhs-free ``fl``, rhs-free ``fr``)
    depends on exactly ``(union_c lhs[b, fl, c]) | (union_c rhs[b, c, fr])``. We reduce
    each operand over its contracting axes, then broadcast the two reduced dep-tensors to
    the output layout ``[batch..., lhs_free..., rhs_free...]`` (JAX's ``dot_general`` output
    order) and union them. This keeps per-row (per-element) support instead of collapsing
    to the whole-array ``total_union`` -- the difference between a locally-supported field
    and one that appears to depend on every DOF.
    """
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


# =============================================================================
# Base Interface
# =============================================================================


@dataclass
class ContributionDemand:
    rows: NDArray[np.int64]


@dataclass
class ContributionRoot:
    var: Any
    rows: NDArray[np.int64]


@dataclass
class ContributionPropagation:
    in_demands: list[ContributionDemand | None]
    roots: list[ContributionRoot]
    valid: bool = True


def _demand(rows: NDArray[np.integer]) -> ContributionDemand | None:
    """Build a canonical demand, dropping rows with no source element."""
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[rows >= 0]
    return ContributionDemand(np.unique(rows)) if rows.size else None


def _invalid_contribution(eqn: JaxprEqn) -> ContributionPropagation:
    """Conservatively stop decomposition at the demanded output."""
    return ContributionPropagation([None] * len(eqn.invars), [], valid=False)


def _inverse_elementwise_rows(
    rows: NDArray[np.integer], in_shape: tuple, out_shape: tuple
) -> NDArray[np.int64]:
    """Map broadcasted output flat rows to the corresponding input flat rows."""
    if not in_shape:
        return np.zeros(len(rows), dtype=np.int64)
    out_coords = np.unravel_index(rows, out_shape)
    lead = len(out_shape) - len(in_shape)
    in_coords = []
    for axis, size in enumerate(in_shape):
        coord = out_coords[lead + axis]
        in_coords.append(np.zeros_like(coord) if size == 1 else coord)
    return np.ravel_multi_index(tuple(in_coords), in_shape).astype(np.int64)


def _inverse_broadcast_rows(
    rows: NDArray[np.integer],
    in_shape: tuple,
    out_shape: tuple,
    dimensions: Sequence[int],
) -> NDArray[np.int64]:
    if not in_shape:
        return np.zeros(len(rows), dtype=np.int64)
    out_coords = np.unravel_index(rows, out_shape)
    in_coords = [
        np.zeros_like(out_coords[axis]) if size == 1 else out_coords[axis]
        for size, axis in zip(in_shape, dimensions)
    ]
    return np.ravel_multi_index(tuple(in_coords), in_shape).astype(np.int64)


class PrimitiveHandler(ABC):
    """Abstract base class for all JAX primitive tracer handlers."""

    @abstractmethod
    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        """Forward dependency set propagation & coupling accumulation."""
        raise NotImplementedError

    def propagate_contribution_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> ContributionPropagation:
        """Stop safely when this primitive has no additive inverse rule.

        ``find_contribution_roots`` turns a non-valid result into roots at the
        demanded output entries.  Keeping the fallback here deliberately small makes
        unsupported primitives conservative instead of silently losing a read.
        """
        return _invalid_contribution(eqn)

    def safe_eval_concrete(
        self,
        primitive: Primitive,
        in_vals: list[NDArray | None],
        params: dict[str, Any],
    ) -> NDArray | None:
        """Evaluate concrete numpy values for this primitive. If the primitive is not
        implemented, fall back to jax itself."""
        if any(v is None for v in in_vals):
            return None
        try:
            res = self.eval_concrete(in_vals, params)
            if res is not None:
                return res

            # For other primitives, use jax itself to evaluate the primitive on concrete numpy
            # values. This is a fallback for primitives that don't have a specific
            # implementation above.
            v = [np.asarray(x) for x in in_vals]
            res = np.asarray(primitive.bind(*[jnp.asarray(x) for x in v], **params))
            warnings.warn(
                f"Concrete evaluation through bind needed for {primitive.name}"
            )
            return res
        except (TypeError, ValueError, KeyError, AttributeError):
            warnings.warn(f"Concrete evaluation failed for {primitive.name}")

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        """Evaluate concrete numpy values for routing indices (returns None if unknown)."""
        return None

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        """Determine if this primitive introduces non-affine behavior on active inputs."""
        return False

    def propagate_tags(self, eqn: JaxprEqn, invar_tags: list[int]) -> int:
        """Forward tag propagation (default: bitwise OR of all input tags)."""
        mask = 0
        for t in invar_tags:
            mask |= t
        return mask

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        """Return invar indices that are used as index arrays (e.g. gather/scatter index)."""
        return []

    def discover_distribution(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
    ) -> None:
        """Pass 1: Discover interactions, assign canonical ownership, and record required ghost DOFs."""

    def remap_local(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
        rank: int,
        var_map: dict[int, Any],
    ) -> JaxprEqn:
        """Pass 2: Remap primitive parameters (index arrays, shapes) and build localized JaxprEqn for rank."""
        invars_local = [var_map.get(id(v), v) for v in eqn.invars]

        # Shape propagation: if any invar has a modified local shape, propagate to outvars
        ref_shape = None
        for v in invars_local:
            sh = getattr(v.aval, "shape", None)
            if sh is not None and len(sh) > 0:
                ref_shape = sh
                break

        if ref_shape is not None:
            for v_out in eqn.outvars:
                if id(v_out) not in var_map:
                    out_shape = getattr(v_out.aval, "shape", ())
                    if out_shape and out_shape[0] != ref_shape[0]:
                        new_shape = (ref_shape[0],) + out_shape[1:]
                        var_map[id(v_out)] = Var(
                            jax.core.ShapedArray(new_shape, v_out.aval.dtype)
                        )

        outvars_local = [var_map.get(id(v), v) for v in eqn.outvars]
        return JaxprEqn(
            invars_local,
            outvars_local,
            eqn.primitive,
            eqn.params.copy(),
            eqn.effects,
            eqn.source_info,
            getattr(eqn, "ctx", None),
        )


# =============================================================================
# 1. NoOp & Zero Dependency Handlers
# =============================================================================


@TR.register(
    "debug_print",
    "debug_callback",
)
class NoOpHandler(PrimitiveHandler):
    """Handler for effect-only primitives (debug_print, debug_callback)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        return


@TR.register(
    ("lt", np.less),
    ("lt_to", np.less),
    ("le", np.less_equal),
    ("le_to", np.less_equal),
    ("gt", np.greater),
    ("ge", np.greater_equal),
    ("eq", np.equal),
    ("ne", np.not_equal),
    ("and", np.logical_and),
    ("or", np.logical_or),
    ("not", np.logical_not),
    ("xor", np.logical_xor),
    ("shift_left", np.left_shift),
    ("shift_right_arithmetic", np.right_shift),
    ("is_finite", np.isfinite),
    ("is_nan", np.isnan),
    ("argmax", lambda x, **p: np.argmax(x, axis=p.get("axes"))),
    ("argmin", lambda x, **p: np.argmin(x, axis=p.get("axes"))),
    ("floor", np.floor),
    ("ceil", np.ceil),
    ("round", np.round),
    ("sign", np.sign),
    "iota",
    "zeros",
    "ones",
    "full",
)
class ZeroDependencyHandler(PrimitiveHandler):
    """Handler for primitives with zero input dependency (iota, zeros, ones, full)."""

    def __init__(self, eval_fn: Callable[..., NDArray] | None = None):
        self.eval_fn = eval_fn

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        for ov in eqn.outvars:
            oshp = _get_shape(ov)
            state.set(ov, SparseDepSet.empty(oshp, state.n_dofs))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if any(v is None for v in in_vals):
            return None
        if self.eval_fn is not None:
            try:
                return np.asarray(self.eval_fn(*in_vals))
            except (TypeError, ValueError, KeyError, AttributeError):
                try:
                    return np.asarray(self.eval_fn(*in_vals, **params))
                except (TypeError, ValueError, KeyError, AttributeError):
                    return None

        # iota fallback
        dim = params.get("dimension")
        shp = params.get("shape")
        if dim is not None and shp is not None:
            newshp = [1] * len(shp)
            newshp[dim] = shp[dim]
            return np.broadcast_to(np.arange(shp[dim]).reshape(newshp), shp).copy()
        return None


# =============================================================================
# 2. Elementwise Unary & Binary Handlers
# =============================================================================


@TR.register(
    ("neg", False),
    ("abs", False),
    ("copy", False),
    ("stop_gradient", False),
    ("device_put", False),
    ("conj", False),
    ("real", False),
    ("imag", False),
    ("reshape", False, _eval_reshape),
    ("squeeze", False, _eval_squeeze),
    ("convert_element_type", False, _eval_convert_dtype),
    ("sin", True, np.sin),
    ("cos", True, np.cos),
    ("tan", True, np.tan),
    ("asin", True, np.arcsin),
    ("acos", True, np.arccos),
    ("atan", True, np.arctan),
    ("exp", True, np.exp),
    ("exp2", True, np.exp2),
    ("expm1", True, np.expm1),
    ("log", True, np.log),
    ("log1p", True, np.log1p),
    ("log2", True, np.log2),
    ("sqrt", True, np.sqrt),
    ("rsqrt", True, lambda x: 1.0 / np.sqrt(x)),
    ("cbrt", True, np.cbrt),
    ("tanh", True, np.tanh),
    ("sinh", True, np.sinh),
    ("cosh", True, np.cosh),
    ("atanh", True, np.arctanh),
    ("asinh", True, np.arcsinh),
    ("acosh", True, np.arccosh),
    ("erf", True, sp.erf),
    ("erfc", True, sp.erfc),
    ("erfinv", True, sp.erfinv),
    ("lgamma", True, sp.gammaln),
    ("digamma", True, sp.psi),
    ("logistic", True, sp.expit),
)
class ElementwiseUnary(PrimitiveHandler):
    """Handler for elementwise unary operations (1 input -> 1 output).

    Covers both:
    - linear/affine unary ops (is_nonlinear=False, e.g. neg, abs, reshape, transpose)
    - nonlinear unary ops (is_nonlinear=True, e.g. sin, cos, exp, sqrt)
    """

    def __init__(
        self, is_nonlinear: bool, eval_fn: Callable[..., NDArray] | None = None
    ):
        self.is_nonlinear = is_nonlinear
        self.eval_fn = eval_fn

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if in_d.shape != oshp and int(np.prod(in_d.shape)) == int(np.prod(oshp)):
            dep_out = in_d.reshape(*oshp)
        else:
            dep_out = in_d

        state.set(eqn.outvars[0], dep_out.copy())

        if self.is_nonlinear:
            dep_out.record_couplings(acc, trial_test_split)

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        transparent = {
            "neg",
            "copy",
            "stop_gradient",
            "device_put",
            "conj",
            "real",
            "imag",
            "reshape",
            "squeeze",
            "convert_element_type",
        }
        if eqn.primitive.name not in transparent:
            return _invalid_contribution(eqn)
        return ContributionPropagation([_demand(demand.rows)], [])

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return self.is_nonlinear and bool(invar_active) and invar_active[0]

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        if self.eval_fn is not None:
            try:
                return self.eval_fn(in_vals[0], params)
            except TypeError:
                return self.eval_fn(in_vals[0])
        return None


@TR.register(
    ("add", False, np.add),
    ("add_any", False, np.add),
    ("sub", False, np.subtract),
    ("max", False, np.maximum),
    ("min", False, np.minimum),
    # nonlinear ops
    ("mul", True, np.multiply),
    ("div", True, _eval_div),
    ("rem", True, np.remainder),
    ("pow", True, np.power),
    ("atan2", True, np.arctan2),
)
class ElementwiseBinary(PrimitiveHandler):
    """Handler for elementwise binary operations (2 inputs -> 1 output).

    Covers both:
    - linear binary ops (is_nonlinear=False, e.g. add, sub, max, min)
    - nonlinear binary ops (is_nonlinear=True, e.g. mul, div, pow, atan2)
    """

    def __init__(
        self, is_nonlinear: bool, eval_fn: Callable[..., NDArray] | None = None
    ):
        self.is_nonlinear = is_nonlinear
        self.eval_fn = eval_fn

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        dep_out = (
            in_d[0].broadcast_to(oshp).dep + in_d[1].broadcast_to(oshp).dep
        ).tocsr()
        dep_out.data[:] = 1
        res = SparseDepSet(dep_out, oshp)
        state.set(eqn.outvars[0], res)

        if self.is_nonlinear:
            # check if one or both inputs are constant
            is_const0 = in_d[0].dep.nnz == 0
            is_const1 = in_d[1].dep.nnz == 0
            is_linear = False
            if (eqn.primitive.name == "mul" and (is_const0 or is_const1)) or (
                eqn.primitive.name == "div" and is_const1
            ):
                is_linear = True
            if not is_linear:
                res.record_couplings(acc, trial_test_split)

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None, None], [])
        lhs, rhs = eqn.invars[:2]
        out_shape = _get_shape(eqn.outvars[0])
        lhs_demand = _demand(
            _inverse_elementwise_rows(demand.rows, _get_shape(lhs), out_shape)
        )
        rhs_demand = _demand(
            _inverse_elementwise_rows(demand.rows, _get_shape(rhs), out_shape)
        )
        primitive = eqn.primitive.name
        if primitive in {"add", "add_any", "sub"}:
            return ContributionPropagation(
                [
                    None if state.is_inactive(lhs) else lhs_demand,
                    None if state.is_inactive(rhs) else rhs_demand,
                ],
                [],
            )
        if primitive == "mul":
            if state.is_inactive(lhs):
                return ContributionPropagation([None, rhs_demand], [])
            if state.is_inactive(rhs):
                return ContributionPropagation([lhs_demand, None], [])
        if primitive == "div" and state.is_inactive(rhs):
            return ContributionPropagation([lhs_demand, None], [])
        return _invalid_contribution(eqn)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        if not self.is_nonlinear:
            return False
        p = eqn.primitive.name
        if p in ("mul", "scatter-mul"):
            return sum(invar_active) >= 2
        if p == "div":
            return len(invar_active) > 1 and invar_active[1]
        return any(invar_active)

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if any(v is None for v in in_vals[:2]):
            return None
        if self.eval_fn is not None:
            try:
                return self.eval_fn(in_vals[0], in_vals[1], params)
            except TypeError:
                return self.eval_fn(in_vals[0], in_vals[1])
        return None


@TR.register(
    "integer_pow",
)
class IntegerPowHandler(PrimitiveHandler):
    """Handler for integer power operations (x ** n)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        state.set(eqn.outvars[0], in_d.copy())
        if self.introduces_nonlinearity(eqn, [in_d.dep.nnz > 0]):
            in_d.record_couplings(acc, trial_test_split)

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        exponent = eqn.params.get("y", 0)
        if exponent == 1:
            return ContributionPropagation([_demand(demand.rows)], [])
        if exponent == 0:
            return ContributionPropagation([None], [])
        return _invalid_contribution(eqn)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        y = eqn.params.get("y", 0)
        return (y >= 2 or y <= -1) and bool(invar_active) and invar_active[0]

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is not None:
            return in_vals[0] ** params["y"]
        return None


# =============================================================================
# 3. Structural Indexing Handlers (Polymorphic, No Mode Branching)
# =============================================================================


@TR.register(
    "broadcast_in_dim",
)
class BroadcastHandler(PrimitiveHandler):
    """Handler for `broadcast_in_dim` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        bdims = eqn.params.get("broadcast_dimensions")
        state.set(eqn.outvars[0], in_d.broadcast_to(oshp, broadcast_dimensions=bdims))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        x = np.asarray(in_vals[0])
        shape = params.get("shape", ())
        bdims = params.get("broadcast_dimensions", ())
        if not shape:
            return x.copy()
        newshape = [1] * len(shape)
        for i, b in enumerate(bdims):
            newshape[b] = x.shape[i] if i < len(x.shape) else 1
        return np.broadcast_to(x.reshape(newshape), shape).copy()

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        rows = _inverse_broadcast_rows(
            demand.rows,
            _get_shape(eqn.invars[0]),
            _get_shape(eqn.outvars[0]),
            eqn.params["broadcast_dimensions"],
        )
        return ContributionPropagation([_demand(rows)], [])

    def remap_local(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
        rank: int,
        var_map: dict[int, Any],
    ) -> JaxprEqn:
        invars_local = [var_map.get(id(v), v) for v in eqn.invars]
        v_in = invars_local[0]
        in_shape = getattr(v_in.aval, "shape", ())

        params = eqn.params.copy()
        old_shape = list(params.get("shape", ()))
        bdims = params.get("broadcast_dimensions", ())

        if old_shape and bdims:
            new_shape = list(old_shape)
            for i, b in enumerate(bdims):
                if i < len(in_shape):
                    new_shape[b] = in_shape[i]
            params["shape"] = tuple(new_shape)

            if eqn.outvars:
                out_var = eqn.outvars[0]
                out_aval = jax.core.ShapedArray(tuple(new_shape), out_var.aval.dtype)
                var_map[id(out_var)] = Var(out_aval)

        outvars_local = [var_map.get(id(v), v) for v in eqn.outvars]
        return JaxprEqn(
            invars_local,
            outvars_local,
            eqn.primitive,
            params,
            eqn.effects,
            eqn.source_info,
            getattr(eqn, "ctx", None),
        )


@TR.register(
    "transpose",
)
class TransposeHandler(PrimitiveHandler):
    """Handler for `transpose` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        arr_indices = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
        perm = np.transpose(arr_indices, eqn.params["permutation"]).ravel()
        state.set(eqn.outvars[0], SparseDepSet(in_d.dep[perm], oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        return np.transpose(np.asarray(in_vals[0]), params["permutation"])

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        in_shape = _get_shape(eqn.invars[0])
        out_coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
        in_coords = [None] * len(in_shape)
        for out_axis, in_axis in enumerate(eqn.params["permutation"]):
            in_coords[in_axis] = out_coords[out_axis]
        rows = np.ravel_multi_index(tuple(in_coords), in_shape)
        return ContributionPropagation([_demand(rows)], [])


@TR.register(
    "slice",
)
class SliceHandler(PrimitiveHandler):
    """Handler for static `slice` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        par = eqn.params
        ss, ls = par["start_indices"], par["limit_indices"]
        st = par["strides"] or [1] * len(ss)
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if in_d.shape != oshp:
            idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
            sl = tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))
            sub_idx = idx[sl].ravel()
            dep_out = SparseDepSet(in_d.dep[sub_idx], oshp)
        else:
            dep_out = in_d

        state.set(eqn.outvars[0], dep_out)

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        ss, ls = params["start_indices"], params["limit_indices"]
        st = params["strides"] or [1] * len(ss)
        sl = tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))
        return np.asarray(in_vals[0])[sl]

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        in_shape = _get_shape(eqn.invars[0])
        out_coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
        strides = eqn.params["strides"] or (1,) * len(in_shape)
        in_coords = tuple(
            start + coord * stride
            for start, coord, stride in zip(
                eqn.params["start_indices"], out_coords, strides
            )
        )
        return ContributionPropagation(
            [_demand(np.ravel_multi_index(in_coords, in_shape))], []
        )

    def remap_local(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
        rank: int,
        var_map: dict[int, Any],
    ) -> JaxprEqn:
        invars_local = [var_map.get(id(v), v) for v in eqn.invars]
        v_in = invars_local[0]
        in_shape = getattr(v_in.aval, "shape", ())

        params = eqn.params.copy()
        ss = list(params.get("start_indices", ()))
        ls = list(params.get("limit_indices", ()))
        st = list(params.get("strides") or ([1] * len(ss)))

        n_owned = len(dist_state.owned_dofs[rank])
        glob_in_shape = getattr(eqn.invars[0].aval, "shape", ())

        if in_shape and ss and ls:
            new_ss, new_ls = [], []
            for d in range(len(ss)):
                max_len = in_shape[d] if d < len(in_shape) else ls[d]
                g_len = glob_in_shape[d] if d < len(glob_in_shape) else ls[d]

                if d < len(in_shape) and in_shape[d] == g_len:
                    s_d = ss[d]
                    l_d = ls[d]
                else:
                    k = g_len - (ls[d] - ss[d])
                    if ss[d] == 0:
                        s_d = 0
                        l_d = max(0, min(ls[d], n_owned - k, max_len))
                    else:
                        s_d = min(ss[d], max_len)
                        l_d = min(ls[d], s_d + min(n_owned, max_len - s_d), max_len)
                new_ss.append(s_d)
                new_ls.append(l_d)

            params["start_indices"] = tuple(new_ss)
            params["limit_indices"] = tuple(new_ls)

            out_shape = tuple(
                max(0, (l - s + step - 1) // step)
                for s, l, step in zip(new_ss, new_ls, st)
            )
            if eqn.outvars:
                out_var = eqn.outvars[0]
                out_aval = jax.core.ShapedArray(out_shape, out_var.aval.dtype)
                var_map[id(out_var)] = Var(out_aval)

        outvars_local = [var_map.get(id(v), v) for v in eqn.outvars]
        return JaxprEqn(
            invars_local,
            outvars_local,
            eqn.primitive,
            params,
            eqn.effects,
            eqn.source_info,
            getattr(eqn, "ctx", None),
        )


@TR.register(
    "rev",
)
class ReverseHandler(PrimitiveHandler):
    """Handler for `rev` (reverse/flip axes) primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        dimensions = eqn.params["dimensions"]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
        sl = tuple(
            slice(None, None, -1) if i in dimensions else slice(None)
            for i in range(len(in_d.shape))
        )
        sub_idx = idx[sl].ravel()
        state.set(eqn.outvars[0], SparseDepSet(in_d.dep[sub_idx], oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        return np.flip(np.asarray(in_vals[0]), axis=params["dimensions"])

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        shape = _get_shape(eqn.invars[0])
        coords = list(np.unravel_index(demand.rows, shape))
        for axis in eqn.params["dimensions"]:
            coords[axis] = shape[axis] - 1 - coords[axis]
        return ContributionPropagation(
            [_demand(np.ravel_multi_index(tuple(coords), shape))], []
        )


@TR.register(
    "pad",
)
class PadHandler(PrimitiveHandler):
    """Handler for `pad` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        low, high, interior = zip(*eqn.params["padding_config"])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
        padded_idx = np.pad(
            idx,
            [(l, h) for l, h in zip(low, high)],
            mode="constant",
            constant_values=-1,
        )
        if any(in_stride > 0 for in_stride in interior):
            slices = [
                slice(None, None, in_stride + 1) if in_stride > 0 else slice(None)
                for in_stride in interior
            ]
            final_idx = np.full(oshp, -1, dtype=int)
            final_idx[tuple(slices)] = padded_idx
            flat_map = final_idx.ravel()
        else:
            flat_map = padded_idx.ravel()

        valid_mask = flat_map >= 0
        valid_src = flat_map[valid_mask]
        out_dep = sps.csr_matrix((int(np.prod(oshp)), state.n_dofs), dtype=bool)
        if valid_src.size > 0:
            out_dep[valid_mask] = in_d.dep[valid_src]
        state.set(eqn.outvars[0], SparseDepSet(out_dep.tocsr(), oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        pad_width = [(l, h) for l, h, _ in params["padding_config"]]
        fill_val = in_vals[1] if len(in_vals) > 1 and in_vals[1] is not None else 0
        return np.pad(
            np.asarray(in_vals[0]),
            pad_width,
            mode="constant",
            constant_values=fill_val,
        )

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None, None], [])
        in_shape = _get_shape(eqn.invars[0])
        out_shape = _get_shape(eqn.outvars[0])
        low, _high, interior = zip(*eqn.params["padding_config"])
        out_coords = np.unravel_index(demand.rows, out_shape)
        in_coords = []
        valid = np.ones(len(demand.rows), dtype=bool)
        for coord, lo, step, size in zip(out_coords, low, interior, in_shape):
            shifted = coord - lo
            stride = step + 1
            valid &= shifted >= 0
            valid &= shifted % stride == 0
            source = shifted // stride
            valid &= source < size
            in_coords.append(source)
        rows = np.full(len(demand.rows), -1, dtype=np.int64)
        if np.any(valid):
            rows[valid] = np.ravel_multi_index(
                tuple(coord[valid] for coord in in_coords), in_shape
            )
        return ContributionPropagation([_demand(rows), None], [])


@TR.register(
    "concatenate",
)
class ConcatenateHandler(PrimitiveHandler):
    """Handler for `concatenate` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        axis = eqn.params.get("dimension", 0)
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if axis == 0:
            concatenated = sps.vstack([d.dep for d in in_d], format="csr")
        else:
            indices_list = []
            for d in in_d:
                n_items = d.dep.shape[0]
                arr = np.arange(n_items).reshape(d.shape)
                indices_list.append(arr)
            perm = np.concatenate(indices_list, axis=axis).ravel()
            stacked = sps.vstack([d.dep for d in in_d], format="csr")
            concatenated = stacked[perm]
        state.set(eqn.outvars[0], SparseDepSet(concatenated, oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if any(v is None for v in in_vals):
            return None
        return np.concatenate(in_vals, axis=params.get("dimension", 0))

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None] * len(eqn.invars), [])
        shapes = [_get_shape(invar) for invar in eqn.invars]
        offset = 0
        index_parts = []
        for shape in shapes:
            size = int(np.prod(shape))
            index_parts.append(np.arange(size).reshape(shape) + offset)
            offset += size
        source_ids = np.concatenate(
            index_parts, axis=eqn.params.get("dimension", 0)
        ).ravel()
        selected = source_ids[demand.rows]
        in_demands = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            in_demands.append(
                _demand(
                    selected[(selected >= offset) & (selected < offset + size)] - offset
                )
            )
            offset += size
        return ContributionPropagation(in_demands, [])


@TR.register(
    "stack",
)
class StackHandler(PrimitiveHandler):
    """Handler for `stack` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        """``stack`` inserts a new axis and places input ``i`` at position ``i`` along it —
        purely structural, so each stacked slice keeps its own per-element support (no
        globalizing union, no couplings). Same construction as ``concatenate`` but the
        input blocks are laid out along a *new* axis rather than concatenated on an
        existing one.
        """
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        idx_arrays = []
        offset = 0
        for b in in_d:
            size = int(np.prod(b.shape))
            idx_arrays.append(np.arange(size).reshape(b.shape) + offset)
            offset += size
        stacked_dep = sps.vstack([b.dep for b in in_d], format="csr")
        stack_idx = np.stack(idx_arrays, axis=eqn.params["axis"]).ravel()
        state.set(eqn.outvars[0], SparseDepSet(stacked_dep[stack_idx], oshp))

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None] * len(eqn.invars), [])
        shape = _get_shape(eqn.invars[0])
        size = int(np.prod(shape))
        indices = [
            np.arange(size).reshape(shape) + i * size for i in range(len(eqn.invars))
        ]
        source_ids = np.stack(indices, axis=eqn.params["axis"]).ravel()
        selected = source_ids[demand.rows]
        return ContributionPropagation(
            [
                _demand(
                    selected[(selected >= i * size) & (selected < (i + 1) * size)]
                    - i * size
                )
                for i in range(len(eqn.invars))
            ],
            [],
        )


@TR.register(
    "split",
)
class SplitHandler(PrimitiveHandler):
    """Handler for `split` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        """``split`` (``jnp.split``) cuts one array into several along ``axis`` -- the exact
        inverse of ``concatenate``, and just as structural: every output element *is* one
        input element, so it inherits that element's dependency row verbatim (no union, no
        couplings).

        Handled explicitly because it is a rare multi-output structural primitive:
        ``Handlers.fallback`` writes only ``outvars[0]``, which would leave every piece but
        the first with an empty dep-set (silently dropped couplings) while collapsing the
        first to a whole-array union (a spuriously dense pattern).
        """
        d = state.get(eqn.invars[0])
        axis = eqn.params["axis"]
        # Row-major positions of the input's elements, laid out in its logical shape, so a
        # slice of this index array names exactly the dep rows that piece is built from.
        src_indices = np.arange(int(np.prod(d.shape))).reshape(d.shape)
        offset = 0
        for outvar, size in zip(eqn.outvars, eqn.params["sizes"]):
            size = int(size)
            oshp = _get_shape(outvar)
            piece = np.take(
                src_indices, np.arange(offset, offset + size), axis=axis
            ).ravel()
            state.set(outvar, SparseDepSet(d.dep[piece], oshp))
            offset += size

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        source_shape = _get_shape(eqn.invars[0])
        source_ids = np.arange(int(np.prod(source_shape))).reshape(source_shape)
        axis = eqn.params["axis"]
        roots = []
        demanded_rows = []
        offset = 0
        for outvar, size, demand in zip(eqn.outvars, eqn.params["sizes"], out_demands):
            if demand is not None:
                piece = np.take(
                    source_ids, np.arange(offset, offset + int(size)), axis=axis
                ).ravel()
                demanded_rows.append(piece[demand.rows])
            offset += int(size)
        if demanded_rows:
            return ContributionPropagation(
                [_demand(np.concatenate(demanded_rows))], roots
            )
        return ContributionPropagation([None], roots)


# =============================================================================
# 4. Dynamic Indexing Handlers
# =============================================================================


@TR.register(
    "gather",
)
class GatherHandler(PrimitiveHandler):
    """Handler for `gather` primitive."""

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        return [1] if len(eqn.invars) > 1 else []

    def discover_distribution(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
    ) -> None:
        d_src = state.get(eqn.invars[0])
        idx = state.get_val(eqn.invars[1]) if len(eqn.invars) > 1 else None
        if idx is not None and d_src.dep.nnz > 0:
            flat_idx = np.asarray(idx).ravel()
            # this will be painfully slow..
            for i in flat_idx:
                if 0 <= i < d_src.dep.shape[0]:
                    dofs = d_src.dep[i].indices
                    dist_state.add_interaction(dofs)

    def remap_local(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
        rank: int,
        var_map: dict[int, Any],
    ) -> JaxprEqn:
        invars_local = [var_map.get(id(v), v) for v in eqn.invars]
        idx_var = invars_local[1] if len(invars_local) > 1 else None
        idx_shape = getattr(idx_var.aval, "shape", ()) if idx_var else ()

        if idx_shape and idx_shape[0] == 0:
            out_var = eqn.outvars[0]
            out_aval = jax.core.ShapedArray((0,), out_var.aval.dtype)
            var_map[id(out_var)] = Var(out_aval)

        outvars_local = [var_map.get(id(v), v) for v in eqn.outvars]
        return JaxprEqn(
            invars_local,
            outvars_local,
            eqn.primitive,
            eqn.params.copy(),
            eqn.effects,
            eqn.source_info,
            getattr(eqn, "ctx", None),
        )

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        d_src = in_d[0]
        idx = state.get_val(eqn.invars[1])
        par = eqn.params
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if idx is not None and d_src.dep.shape[0] >= 1:
            try:
                dnums = par["dimension_numbers"]
                ss = par["slice_sizes"]
                collapsed = dnums.collapsed_slice_dims
                sim = dnums.start_index_map

                arr_indices = np.arange(int(np.prod(d_src.shape))).reshape(d_src.shape)

                # Branch 1: 1D gather along dimension 0
                if collapsed == (0,) and sim == (0,) and ss[0] == 1:
                    rows = np.clip(idx.ravel().astype(int), 0, d_src.shape[0] - 1)
                    slc = (rows,) + tuple(slice(None, s) for s in ss[1:])
                    flat_src_indices = arr_indices[slc].ravel()
                    res_dep = d_src.dep[flat_src_indices].tocsr()
                    res_dep.data[:] = 1
                    state.set(eqn.outvars[0], SparseDepSet(res_dep, oshp))
                    return

                # Branch 2: 2D indexing with slice along axis 1
                if (
                    collapsed == (0,)
                    and sim == (0, 1)
                    and ss[0] == 1
                    and idx.ndim == 2
                    and idx.shape[1] == 2
                ):
                    r = np.clip(idx[:, 0].astype(int), 0, d_src.shape[0] - 1)
                    c0 = np.clip(idx[:, 1].astype(int), 0, d_src.shape[1] - ss[1])

                    col_offsets = np.arange(ss[1])
                    cols = c0[:, None] + col_offsets[None, :]
                    flat_src_indices = arr_indices[r[:, None], cols].ravel()

                    res_dep = d_src.dep[flat_src_indices].tocsr()
                    res_dep.data[:] = 1
                    state.set(eqn.outvars[0], SparseDepSet(res_dep, oshp))
                    return

                # Branch 3: N-D point indexing with proper start_index_map mapping
                if (
                    set(collapsed) == set(sim)
                    and idx.ndim == 2
                    and idx.shape[1] == len(sim)
                    and all(ss[a] == 1 for a in sim)
                ):
                    coords = [None] * len(d_src.shape)
                    for j, axis in enumerate(sim):
                        coords[axis] = np.clip(
                            idx[:, j].astype(int), 0, d_src.shape[axis] - 1
                        )
                    flat_src_indices = arr_indices[tuple(coords)].ravel()
                    res_dep = d_src.dep[flat_src_indices].tocsr()
                    res_dep.data[:] = 1
                    state.set(eqn.outvars[0], SparseDepSet(res_dep, oshp))
                    return
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                pass

        total = d_src.total_union()
        stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
        state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))


@TR.register(
    "dynamic_slice",
    "dynamic_update_slice",
)
class DynamicSliceHandler(PrimitiveHandler):
    """Handler for `dynamic_slice` and `dynamic_update_slice` primitive."""

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        p = eqn.primitive.name
        if p == "dynamic_slice":
            return list(range(1, len(eqn.invars)))
        elif p == "dynamic_update_slice":
            return list(range(2, len(eqn.invars)))
        return []

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        d_operand = in_d[0]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        slice_sizes = eqn.params.get("slice_sizes", ())

        start_vals = [state.get_val(v) for v in eqn.invars[1:]]
        if all(s is not None for s in start_vals) and d_operand.dep.shape[0] > 0:
            start_vals = cast(list[NDArray], start_vals)
            try:
                arr = np.arange(int(np.prod(d_operand.shape))).reshape(d_operand.shape)
                slices = tuple(
                    slice(int(s), int(s) + sz) for s, sz in zip(start_vals, slice_sizes)
                )
                sliced_idx = arr[slices].ravel()
                res_dep = d_operand.dep[sliced_idx].tocsr()
                res_dep.data[:] = 1
                state.set(eqn.outvars[0], SparseDepSet(res_dep, oshp))
                return
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                pass

        total = d_operand.total_union()
        stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
        state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))


@TR.register(
    "scatter",
    "scatter-add",
    "scatter-sub",
    "scatter-mul",
    "scatter-min",
    "scatter-max",
)
class ScatterHandler(PrimitiveHandler):
    """Handler for `scatter` family primitives."""

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        return [1] if len(eqn.invars) > 1 else []

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return eqn.primitive.name == "scatter-mul" and sum(invar_active) >= 2

    def discover_distribution(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
    ) -> None:
        d_vals = state.get(eqn.invars[2]) if len(eqn.invars) > 2 else None
        if d_vals is not None and d_vals.dep.nnz > 0:
            for i in range(d_vals.dep.shape[0]):
                dofs = d_vals.dep[i].indices
                dist_state.add_interaction(dofs)

    def remap_local(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        dist_state: DistributionState,
        rank: int,
        var_map: dict[int, Any],
    ) -> JaxprEqn:
        invars_local = [var_map.get(id(v), v) for v in eqn.invars]
        idx_var = invars_local[1] if len(invars_local) > 1 else None
        idx_shape = getattr(idx_var.aval, "shape", ()) if idx_var else ()

        if idx_shape and idx_shape[0] == 0:
            out_var = eqn.outvars[0]
            var_map[id(out_var)] = invars_local[0]

        outvars_local = [var_map.get(id(v), v) for v in eqn.outvars]
        return JaxprEqn(
            invars_local,
            outvars_local,
            eqn.primitive,
            eqn.params.copy(),
            eqn.effects,
            eqn.source_info,
            getattr(eqn, "ctx", None),
        )

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        d_tgt = in_d[0]
        d_vals = in_d[2]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        nonlinear = eqn.primitive.name == "scatter-mul"

        u_vals = sps.csr_matrix(d_vals.dep.sum(axis=0).astype(bool))
        idx = state.get_val(eqn.invars[1])

        if idx is not None and d_tgt.dep.shape[0] > 1:
            try:
                idx_flat = idx.ravel().astype(int)
                if len(idx_flat) == d_vals.dep.shape[0]:
                    coo_vals = d_vals.dep.tocoo()
                    row_mapped = idx_flat[coo_vals.row]
                    valid = (row_mapped >= 0) & (row_mapped < d_tgt.dep.shape[0])
                    scattered_row = row_mapped[valid]
                    scattered_col = coo_vals.col[valid]
                    scattered_data = coo_vals.data[valid]

                    scattered_mat = sps.csr_matrix(
                        (scattered_data, (scattered_row, scattered_col)),
                        shape=d_tgt.dep.shape,
                    )
                    res_dep = (d_tgt.dep + scattered_mat).tocsr()
                else:
                    valid_idx = idx_flat[
                        (idx_flat >= 0) & (idx_flat < d_tgt.dep.shape[0])
                    ]
                    if len(valid_idx) > 0:
                        coo_u = u_vals.tocoo()
                        u_cols = coo_u.col
                        u_data = coo_u.data

                        scattered_row = np.repeat(valid_idx, len(u_cols))
                        scattered_col = np.tile(u_cols, len(valid_idx))
                        scattered_data = np.tile(u_data, len(valid_idx))

                        scattered_mat = sps.csr_matrix(
                            (scattered_data, (scattered_row, scattered_col)),
                            shape=d_tgt.dep.shape,
                        )
                        res_dep = (d_tgt.dep + scattered_mat).tocsr()
                    else:
                        res_dep = d_tgt.dep.copy()

                res_dep.data[:] = 1
                res = SparseDepSet(res_dep, oshp)
                state.set(eqn.outvars[0], res)
                if nonlinear:
                    res.record_couplings(acc, trial_test_split)
                return
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                pass

        result = d_tgt.dep + _broadcast_single_row(u_vals, int(np.prod(oshp)))
        res_dep = result.tocsr()
        res_dep.data[:] = 1
        res = SparseDepSet(res_dep, oshp)
        state.set(eqn.outvars[0], res)
        if nonlinear:
            res.record_couplings(acc, trial_test_split)


@TR.register(
    "select_n",
)
class SelectNHandler(PrimitiveHandler):
    """Handler for `select_n` primitive."""

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        return [0] if len(eqn.invars) > 0 else []

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        cond_val = state.get_val(eqn.invars[0])
        cases_d = in_d[1:]

        if cond_val is not None and len(cases_d) >= 2:
            try:
                cond_flat = cond_val.ravel().astype(int)
                n_out = int(np.prod(oshp))
                res_rows = []
                for idx_out in range(n_out):
                    sel = cond_flat[idx_out] if idx_out < len(cond_flat) else 0
                    sel_clamped = max(0, min(sel, len(cases_d) - 1))
                    case_dep = cases_d[sel_clamped].dep
                    r_idx = idx_out % case_dep.shape[0] if case_dep.shape[0] > 0 else 0
                    res_rows.append(case_dep[r_idx])
                res_dep = sps.vstack(res_rows, format="csr")
                state.set(eqn.outvars[0], SparseDepSet(res_dep, oshp))
                return
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                pass

        merged = sps.csr_matrix((int(np.prod(oshp)), state.n_dofs), dtype=bool)
        for c_d in cases_d:
            merged = (merged + c_d.broadcast_to(oshp).dep).tocsr()
        merged.data[:] = 1
        state.set(eqn.outvars[0], SparseDepSet(merged, oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if any(v is None for v in in_vals):
            return None
        cond, cases = cast(NDArray, in_vals[0]), cast(list[NDArray], in_vals[1:])
        if len(cases) == 2:
            return np.where(cond.astype(bool), cases[1], cases[0])
        result = cases[0].copy()
        for i, case in enumerate(cases[1:], 1):
            result = np.where(cond == i, case, result)
        return result


@TR.register(
    "dot_general",
)
class DotHandler(PrimitiveHandler):
    """Handler for bilinear contraction `dot_general` primitive."""

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return sum(invar_active) >= 2

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        dn = eqn.params.get("dimension_numbers")
        lhs_batch, rhs_batch = [], []
        lhs_contract, rhs_contract = [], []
        if dn is not None:
            if hasattr(dn, "lhs_batch_dimensions"):
                lhs_batch = dn.lhs_batch_dimensions
                rhs_batch = dn.rhs_batch_dimensions
                lhs_contract = dn.lhs_contracting_dimensions
                rhs_contract = dn.rhs_contracting_dimensions
            elif len(dn) > 1:
                lhs_contract, rhs_contract = dn[0][0], dn[0][1]
                lhs_batch, rhs_batch = dn[1][0], dn[1][1]

        is_batched = (
            bool(lhs_batch and rhs_batch)
            and len(lhs_batch) > 0
            and lhs_batch[0] == 0
            and rhs_batch[0] == 0
        )

        if is_batched and len(in_d[0].shape) > 1 and len(in_d[1].shape) > 1:
            B = in_d[0].shape[0]
            S_a = int(np.prod(in_d[0].shape[1:]))
            S_b = int(np.prod(in_d[1].shape[1:]))
            S_out = int(np.prod(oshp[1:]))

            coo_a = in_d[0].dep.tocoo()
            batch_a, dof_a = coo_a.row // S_a, coo_a.col
            A_active = sps.csr_matrix(
                (np.ones(len(batch_a), dtype=bool), (batch_a, dof_a)),
                shape=(B, state.n_dofs),
                dtype=bool,
            )

            coo_b = in_d[1].dep.tocoo()
            batch_b, dof_b = coo_b.row // S_b, coo_b.col
            B_active = sps.csr_matrix(
                (np.ones(len(batch_b), dtype=bool), (batch_b, dof_b)),
                shape=(B, state.n_dofs),
                dtype=bool,
            )

            if A_active.nnz and B_active.nnz:
                if state.is_nonlinear(eqn.invars[0]):
                    acc.record_dep(A_active, trial_test_split)
                if state.is_nonlinear(eqn.invars[1]):
                    acc.record_dep(B_active, trial_test_split)

                P_cross = (A_active.T @ B_active).tocsr()
                r_c, c_c = P_cross.nonzero()
                if trial_test_split is not None:
                    mask_c = (r_c < trial_test_split) & (c_c >= trial_test_split)
                    mask_c |= (r_c >= trial_test_split) & (c_c < trial_test_split)
                    r_c, c_c = r_c[mask_c], c_c[mask_c]
                acc.add_coords(r_c, c_c)
                acc.add_coords(c_c, r_c)

            C_active = (A_active + B_active).astype(bool).tocsr()
            repeat_indices = np.repeat(np.arange(B), S_out)
            stacked_dep = C_active[repeat_indices]
            state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))
        else:
            ua = in_d[0].total_union()
            ub = in_d[1].total_union()
            ia, ib = ua.dep.indices, ub.dep.indices

            if ia.size and ib.size:
                if state.is_nonlinear(eqn.invars[0]):
                    ua.record_couplings(acc, trial_test_split)
                if state.is_nonlinear(eqn.invars[1]):
                    ub.record_couplings(acc, trial_test_split)

                r_c = np.repeat(ia, ib.size)
                c_c = np.tile(ib, ia.size)
                if trial_test_split is not None:
                    mask = ((r_c < trial_test_split) & (c_c >= trial_test_split)) | (
                        (r_c >= trial_test_split) & (c_c < trial_test_split)
                    )
                    r_c, c_c = r_c[mask], c_c[mask]
                acc.add_coords(r_c, c_c)
                acc.add_coords(c_c, r_c)

            out_dep = _dot_general_out_dep(
                in_d[0],
                in_d[1],
                lhs_contract,
                rhs_contract,
                lhs_batch,
                rhs_batch,
                oshp,
                state.n_dofs,
            )
            state.set(eqn.outvars[0], out_dep)


@TR.register(
    "reduce_sum",
    "reduce_window_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_and",
    "reduce_or",
)
class ReductionHandler(PrimitiveHandler):
    """Handler for reduction primitives (reduce_sum, reduce_max, etc.)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        axes = eqn.params.get("axes", ())
        keep_axes = [i for i in range(len(in_d.shape)) if i not in axes]

        reduced_dep = _reduce_union_over_axes(in_d.dep, in_d.shape, keep_axes)
        state.set(eqn.outvars[0], SparseDepSet(reduced_dep, oshp))

    def propagate_contribution_demand(
        self, eqn, state, out_demands
    ) -> ContributionPropagation:
        demand = out_demands[0]
        if demand is None:
            return ContributionPropagation([None], [])
        if eqn.primitive.name != "reduce_sum":
            return _invalid_contribution(eqn)
        in_shape = _get_shape(eqn.invars[0])
        axes = set(eqn.params["axes"])
        kept_axes = [axis for axis in range(len(in_shape)) if axis not in axes]
        out_shape = _get_shape(eqn.outvars[0])
        out_coords = np.unravel_index(demand.rows, out_shape) if out_shape else ()
        grids = np.indices(in_shape, sparse=False).reshape(len(in_shape), -1)
        mask = np.zeros(grids.shape[1], dtype=bool)
        for demand_index in range(len(demand.rows)):
            matches = np.ones(grids.shape[1], dtype=bool)
            for coord_index, axis in enumerate(kept_axes):
                matches &= grids[axis] == out_coords[coord_index][demand_index]
            mask |= matches
        rows = np.flatnonzero(mask).astype(np.int64)
        return ContributionPropagation([None], [ContributionRoot(eqn.invars[0], rows)])


# =============================================================================
# 5. Opaque Black-Box Handler
# =============================================================================


@TR.register(
    "lu",
    "custom_linear_solve",
    "triangular_solve",
    "lu_solve",
    "cholesky",
    "eig",
    "eigh",
    "custom_vjp_call",
    "custom_jvp_call",
    "pure_callback",
    "io_callback",
    ("while", False),
    ("switch", False),
)
class OpaqueBlackBoxHandler(PrimitiveHandler):
    """Handler for opaque black-box primitives (callbacks, dense linalg, fallback)."""

    def __init__(self, record_couplings: bool = True):
        self.record_couplings = record_couplings

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        if not eqn.outvars:
            return

        in_d = [state.get(v) for v in eqn.invars]
        cols_active = np.zeros(state.n_dofs, dtype=bool)
        has_active = False
        for d in in_d:
            if isinstance(d, SparseDepSet) and d.dep.shape[0] > 0:
                cols_active[d.dep.indices] = True
                has_active = True

        if not has_active:
            total = SparseDepSet.empty((), state.n_dofs)
        else:
            reduced = sps.csr_matrix(cols_active.reshape(1, -1))
            total = SparseDepSet(reduced, ())
            if self.record_couplings:
                acc.record_dep(total.dep, trial_test_split)

        for ov in eqn.outvars:
            oshp = _get_shape(ov)
            stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
            state.set(ov, SparseDepSet(stacked_dep, oshp))

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return any(invar_active)


# =============================================================================
# 6. Higher-Order Sub-Jaxpr Handlers
# =============================================================================


@TR.register(
    "pjit",
    "jit",
    "remat2",
)
class SubJaxprHandler(PrimitiveHandler):
    """Handler for carried sub-jaxprs (pjit, jit, remat2)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        sub, sub_consts = _subjaxpr_and_consts(eqn)
        in_d = [state.get(v) for v in eqn.invars]

        sub_active, _sub_index_set, sub_bound_eqns = cast(
            SubEqnInfo, state.sub_info[id(eqn)]
        )
        n_dofs = state.n_dofs
        sub_state = TraceState(n_dofs, sub_active, state.sub_info, state.nonlinear_ids)

        for v, d in zip(sub.invars, in_d):
            sub_state.set(v, d)
        for pv, sv in zip(eqn.invars, sub.invars):
            if state.is_nonlinear(pv):
                sub_state.nonlinear_ids.add(id(sv))
            val = state.get_val(pv)
            if val is not None:
                sub_state.val_of[id(sv)] = val
        for v, c in zip(sub.constvars, sub_consts):
            sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
            sub_state.val_of[id(v)] = np.asarray(c)

        sub_state.run_bound_eqns(sub_bound_eqns, acc, trial_test_split)

        for pv, sv in zip(eqn.outvars, sub.outvars):
            state.set(pv, sub_state.get(sv))
            if sub_state.is_nonlinear(sv):
                state.nonlinear_ids.add(id(pv))
            val = sub_state.get_val(sv)
            if val is not None:
                state.val_of[id(pv)] = val

        in_vals = [state.get_val(v) for v in eqn.invars]
        if not any(v is None for v in in_vals):
            try:
                v_casted = []
                for x, invar in zip(in_vals, sub.invars):
                    target_dtype = getattr(invar.aval, "dtype", None)
                    x_arr = np.asarray(x)
                    if target_dtype is not None and x_arr.dtype != target_dtype:
                        x_arr = x_arr.astype(target_dtype)
                    v_casted.append(x_arr)
                res = jax.core.eval_jaxpr(sub, sub_consts, *v_casted)
                for pv, r in zip(eqn.outvars, res):
                    state.val_of[id(pv)] = np.asarray(r)
            except (TypeError, ValueError, KeyError, AttributeError):
                pass

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if any(v is None for v in in_vals):
            return None
        sub = params.get("jaxpr")
        if sub is None:
            return None
        jaxpr_body = sub.jaxpr if hasattr(sub, "jaxpr") else sub
        consts = sub.consts if hasattr(sub, "consts") else ()
        v_casted = []
        for x, invar in zip(in_vals, jaxpr_body.invars):
            target_dtype = getattr(invar.aval, "dtype", None)
            x_arr = np.asarray(x)
            if target_dtype is not None and x_arr.dtype != target_dtype:
                x_arr = x_arr.astype(target_dtype)
            v_casted.append(x_arr)
        try:
            res = jax.core.eval_jaxpr(jaxpr_body, consts, *v_casted)
            return np.asarray(res[0])
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


@TR.register(
    "cond",
)
class CondHandler(PrimitiveHandler):
    """Handler for branching sub-jaxprs (cond, switch)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        operands = eqn.invars[1:]
        in_d = [state.get(v) for v in operands]
        n_dofs = state.n_dofs
        branch_sub_list = cast(list[SubEqnInfo], state.sub_info[id(eqn)])

        out_deps: dict[int, SparseDepSet | None] = {id(ov): None for ov in eqn.outvars}
        for (sub_active, sub_index_set, sub_bound_eqns), branch in zip(
            branch_sub_list, eqn.params["branches"]
        ):
            sub, sub_consts = branch.jaxpr, branch.consts
            sub_state = TraceState(
                n_dofs, sub_active, state.sub_info, state.nonlinear_ids
            )

            for sv, d, ov in zip(sub.invars, in_d, operands):
                sub_state.set(sv, d)
                if state.is_nonlinear(ov):
                    sub_state.nonlinear_ids.add(id(sv))
                val = state.get_val(ov)
                if val is not None:
                    sub_state.val_of[id(sv)] = val
            for v, c in zip(sub.constvars, sub_consts):
                sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
                sub_state.val_of[id(v)] = np.asarray(c)

            sub_state.run_bound_eqns(sub_bound_eqns, acc, trial_test_split)

            for ov, sv in zip(eqn.outvars, sub.outvars):
                d = sub_state.get(sv)
                if sub_state.is_nonlinear(sv):
                    state.nonlinear_ids.add(id(ov))
                prev = out_deps[id(ov)]
                if prev is None:
                    out_deps[id(ov)] = d
                else:
                    merged = (prev.dep + d.dep).tocsr()
                    merged.data[:] = 1
                    out_deps[id(ov)] = SparseDepSet(merged, d.shape)

        for ov in eqn.outvars:
            d = out_deps[id(ov)]
            state.set(
                ov, d if d is not None else SparseDepSet.empty(_get_shape(ov), n_dofs)
            )


@TR.register(
    "scan",
    "map",
)
class ScanMapHandler(PrimitiveHandler):
    """Handler for looping/batched sub-jaxprs (scan, map)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        closed_sub = eqn.params["jaxpr"]
        sub, sub_consts = closed_sub.jaxpr, closed_sub.consts

        num_const = eqn.params.get("num_consts", 0)
        num_carry = eqn.params.get("num_carry", 0)
        num_xs = len(eqn.invars) - num_const - num_carry

        slice_shapes, size_slices = [], []
        for k in range(num_xs):
            x = eqn.invars[num_const + num_carry + k]
            x_dep = state.get(x)
            slice_shapes.append(x_dep.shape[1:])
            size_slices.append(int(np.prod(x_dep.shape[1:])))

        batch_size = 1
        local_size_slices = []
        for k in range(num_xs):
            shp = slice_shapes[k]
            if len(shp) > 1:
                batch_size = shp[0]
                local_size_slices.append(int(np.prod(shp[1:])))
            else:
                local_size_slices.append(int(np.prod(shp)))

        n_local_dofs = sum(local_size_slices)
        length = eqn.params.get("length", 1)

        sub_active, _sub_index_set, sub_bound_eqns = cast(
            SubEqnInfo, state.sub_info[id(eqn)]
        )
        sub_state = TraceState(
            n_local_dofs, sub_active, state.sub_info, state.nonlinear_ids
        )

        for i in range(num_const):
            sub_state.set(
                sub.invars[i],
                SparseDepSet.empty(_get_shape(sub.invars[i]), n_local_dofs),
            )
            if state.is_nonlinear(eqn.invars[i]):
                sub_state.nonlinear_ids.add(id(sub.invars[i]))
            val = state.get_val(eqn.invars[i])
            if val is not None:
                sub_state.val_of[id(sub.invars[i])] = val

        for v, c in zip(sub.constvars, sub_consts):
            sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_local_dofs))
            sub_state.val_of[id(v)] = np.asarray(c)

        for i in range(num_carry):
            sub_state.set(
                sub.invars[num_const + i],
                SparseDepSet.empty(_get_shape(sub.invars[num_const + i]), n_local_dofs),
            )
            val = state.get_val(eqn.invars[num_const + i])
            if val is not None:
                sub_state.val_of[id(sub.invars[num_const + i])] = val

        symbolic_seed = SparseDepSet.singletons(n_local_dofs)
        offset = 0
        for k in range(num_xs):
            sz, shp = local_size_slices[k], slice_shapes[k]
            local_dep = symbolic_seed.dep[offset : offset + sz]

            if batch_size > 1:
                tiled_indices = np.tile(local_dep.indices, batch_size)
                tiled_data = np.tile(local_dep.data, batch_size)
                base_step = local_dep.indptr[-1]
                base = local_dep.indptr[:-1]
                tiled_indptr = (
                    base[None, :] + (np.arange(batch_size) * base_step)[:, None]
                ).ravel()
                tiled_indptr = np.concatenate(
                    [tiled_indptr, np.array([batch_size * base_step])]
                ).astype(local_dep.indptr.dtype)
                sub_dep = sps.csr_matrix(
                    (tiled_data, tiled_indices, tiled_indptr),
                    shape=(batch_size * sz, n_local_dofs),
                )
            else:
                sub_dep = local_dep

            sub_state.set(
                sub.invars[num_const + num_carry + k], SparseDepSet(sub_dep, shp)
            )
            if state.is_nonlinear(eqn.invars[num_const + num_carry + k]):
                sub_state.nonlinear_ids.add(id(sub.invars[num_const + num_carry + k]))

            val = state.get_val(eqn.invars[num_const + num_carry + k])
            if val is not None and val.shape[0] > 0:
                sub_state.val_of[id(sub.invars[num_const + num_carry + k])] = val[0]

            offset += sz

        sub_acc = CouplingAccumulator(n_local_dofs)

        sub_state.run_bound_eqns(sub_bound_eqns, sub_acc, None)

        sub_pat = sub_acc.finalize()
        sub_r, sub_c = sub_pat.nonzero()
        if sub_r.size:
            lo, hi = np.minimum(sub_r, sub_c), np.maximum(sub_r, sub_c)
            canon = sps.csr_matrix(
                (np.ones(lo.size, dtype=bool), (lo, hi)),
                shape=(n_local_dofs, n_local_dofs),
            )
            lo_arr, hi_arr = canon.nonzero()
        else:
            lo_arr = hi_arr = np.empty(0, dtype=np.intp)

        offsets = np.concatenate(([0], np.cumsum(local_size_slices)))
        slice_cache: dict[int, tuple[sps.csr_matrix, np.ndarray]] = {}

        def _get_col(local_idx: int) -> tuple[sps.csr_matrix, np.ndarray]:
            cached = slice_cache.get(local_idx)
            if cached is not None:
                return cached
            k_idx = int(np.searchsorted(offsets, local_idx, side="right")) - 1
            idx_in = local_idx - int(offsets[k_idx])
            n_k = local_size_slices[k_idx]
            dep_x = state.get(eqn.invars[num_const + num_carry + k_idx]).dep
            col = dep_x[idx_in::n_k]
            nnz_per_row = col.indptr[1:] - col.indptr[:-1]
            cached = (col, nnz_per_row)
            slice_cache[local_idx] = cached
            return cached

        for la, lb in zip(lo_arr.tolist(), hi_arr.tolist()):
            col_a, nnz_per_row_a = _get_col(la)
            col_b, nnz_per_row_b = _get_col(lb)

            if np.all(nnz_per_row_a <= 1) and np.all(nnz_per_row_b <= 1):
                active_a, active_b = nnz_per_row_a == 1, nnz_per_row_b == 1
                active = active_a & active_b
                if np.any(active):
                    c_a = col_a.indices[col_a.indptr[:-1][active]]
                    c_b = col_b.indices[col_b.indptr[:-1][active]]
                    if trial_test_split is not None:
                        mask = (c_a < trial_test_split) & (c_b >= trial_test_split)
                        mask |= (c_a >= trial_test_split) & (c_b < trial_test_split)
                        c_a, c_b = c_a[mask], c_b[mask]
                    acc.add_coords(c_a, c_b)
                    acc.add_coords(c_b, c_a)
            else:
                couplings = (col_a.T @ col_b).tocsr()
                r, c = couplings.nonzero()
                if trial_test_split is not None:
                    mask = (r < trial_test_split) & (c >= trial_test_split)
                    mask |= (r >= trial_test_split) & (c < trial_test_split)
                    r, c = r[mask], c[mask]
                acc.add_coords(r, c)
                acc.add_coords(c, r)

        total_length = length * batch_size

        for i in range(num_carry):
            state.set(
                eqn.outvars[i],
                SparseDepSet.empty(_get_shape(eqn.outvars[i]), state.n_dofs),
            )
            c_val = sub_state.get_val(sub.outvars[i])
            if c_val is not None:
                state.val_of[id(eqn.outvars[i])] = c_val

        in_vals = [state.get_val(v) for v in eqn.invars]
        if not any(v is None for v in in_vals):
            try:
                v_casted = []
                for x, invar in zip(in_vals, sub.invars):
                    target_dtype = getattr(invar.aval, "dtype", None)
                    x_arr = np.asarray(x)
                    if target_dtype is not None and x_arr.dtype != target_dtype:
                        x_arr = x_arr.astype(target_dtype)
                    v_casted.append(x_arr)
                res = jax.core.eval_jaxpr(sub, sub_consts, *v_casted)
                for i in range(len(eqn.outvars)):
                    state.val_of[id(eqn.outvars[i])] = np.asarray(res[i])
            except (TypeError, ValueError, RuntimeError, KeyError):
                pass

        if len(eqn.outvars) > num_carry:
            dep_inputs = [
                state.get(eqn.invars[num_const + num_carry + k]).dep
                for k in range(num_xs)
            ]
            dep_all = sps.vstack(dep_inputs, format="csr")

            offsets_all = [0]
            for k in range(num_xs - 1):
                offsets_all.append(
                    offsets_all[-1] + total_length * local_size_slices[k]
                )

            indices_list = []
            for k in range(num_xs):
                sz = local_size_slices[k]
                start_indices = offsets_all[k] + np.arange(total_length) * sz
                indices_list.append(start_indices[:, None] + np.arange(sz)[None, :])
            perm_idx = np.hstack(indices_list).ravel()

            In_all = dep_all[perm_idx]

            for i in range(len(eqn.outvars) - num_carry):
                sub_y = sub.outvars[num_carry + i]
                y = eqn.outvars[num_carry + i]
                sub_y_dep = sub_state.get(sub_y)
                slice_shape = sub_y_dep.shape
                sz_local_y = (
                    int(np.prod(slice_shape[1:]))
                    if len(slice_shape) > 1
                    else int(np.prod(slice_shape))
                )

                Y_coo = sub_y_dep.dep.tocoo()
                if length > 1:
                    l_arr = np.arange(length)
                    r_block = (
                        Y_coo.row[None, :] + l_arr[:, None] * (batch_size * sz_local_y)
                    ).ravel()
                    base_col = (Y_coo.row // sz_local_y) * n_local_dofs + Y_coo.col
                    c_block = (
                        base_col[None, :] + l_arr[:, None] * (batch_size * n_local_dofs)
                    ).ravel()
                    data_block = np.tile(Y_coo.data, length)
                else:
                    r_block = Y_coo.row
                    c_block = (Y_coo.row // sz_local_y) * n_local_dofs + Y_coo.col
                    data_block = Y_coo.data

                sub_y_block = sps.csr_matrix(
                    (data_block, (r_block, c_block)),
                    shape=(total_length * sz_local_y, total_length * n_local_dofs),
                )

                expanded_dep = (sub_y_block @ In_all).tocsr()
                expanded_dep.data[:] = 1

                y_shape = (length,) + slice_shape
                state.set(y, SparseDepSet(expanded_dep, y_shape))
                if sub_state.is_nonlinear(sub_y):
                    state.nonlinear_ids.add(id(y))
                sub_y_val = sub_state.get_val(sub_y)
                if sub_y_val is not None:
                    try:
                        state.val_of[id(y)] = np.broadcast_to(sub_y_val, y_shape).copy()
                    except (TypeError, ValueError, RuntimeError, KeyError):
                        pass


@TR.register(
    "ffi_call",
)
class FFICallHandler(PrimitiveHandler):
    """Handler for `ffi_call` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        target = eqn.params.get("target_name")

        lead = None
        if target in _ELEMENTWISE_FFI_TARGETS:
            shapes = [d.shape for d in in_d] + [_get_shape(ov) for ov in eqn.outvars]
            if shapes and all(len(s) >= 1 for s in shapes):
                leads = {s[0] for s in shapes}
                if len(leads) == 1:
                    lead = leads.pop()

        if not lead:
            OpaqueBlackBoxHandler(record_couplings=True).propagate_deps(
                eqn, state, acc, trial_test_split
            )
            return

        B = lead
        slice_rows: list[sps.csr_matrix] = []
        for b in range(B):
            cols_active = np.zeros(state.n_dofs, dtype=bool)
            for d in in_d:
                nrows = d.dep.shape[0]
                if nrows == 0:
                    continue
                core = nrows // B
                blk = d.dep[b * core : (b + 1) * core]
                if blk.nnz:
                    cols_active[blk.indices] = True
            row = sps.csr_matrix(cols_active.reshape(1, -1))
            if row.nnz:
                acc.record_dep(row, trial_test_split)
            slice_rows.append(row)

        for ovar in eqn.outvars:
            oshp = _get_shape(ovar)
            core_o = int(np.prod(oshp)) // B
            blocks = [_broadcast_single_row(slice_rows[b], core_o) for b in range(B)]
            stacked = sps.vstack(blocks).tocsr()
            state.set(ovar, SparseDepSet(stacked, oshp))
