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
from collections.abc import Callable
from typing import Any, cast

import jax.core
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.special as sp
from jax.extend.core import JaxprEqn, Literal, Primitive
from numpy.typing import NDArray

from tatva.sparse.tracer.common import (
    _ELEMENTWISE_FFI_TARGETS,
    _broadcast_single_row,
    _dot_general_out_dep,
    _get_shape,
    _inverse_broadcast_rows,
    _inverse_elementwise_rows,
    _reduce_union_over_axes,
    _subjaxpr_and_consts,
    gather_routes,
    scatter_routes,
)
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ContributionDemand,
    ContributionPropagation,
    ContributionRoot,
    RangeRows,
    _demand,
    _invalid_contribution,
    demand_rows,
    merge_demands,
    propagate_demands_backward,
)
from tatva.sparse.tracer.registry import TR
from tatva.sparse.tracer.state import (
    CouplingAccumulator,
    SparseDepSet,
    SubEqnInfo,
    TraceExecution,
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


def _inverse_elementwise_demand(demand, in_shape: tuple, out_shape: tuple):
    """Reverse broadcast mapping with compact identity/scalar fast paths."""
    if in_shape == out_shape:
        return demand
    if int(np.prod(in_shape)) == 1:
        return ContributionDemand(RangeRows(0, 1))
    return _demand(_inverse_elementwise_rows(demand.rows, in_shape, out_shape))


# =============================================================================
# Base Interface
# =============================================================================


class PrimitiveHandler(ABC):
    """Abstract base class for all JAX primitive tracer handlers."""

    @abstractmethod
    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        execution: TraceExecution,
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

    def propagate_liveness_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None]:
        """Map demanded output entries to required input entries."""
        # A missing rule must never make a value appear dead.  In particular,
        # inputs with an empty DOF dependency can still be numerical operands
        # (material data, masks, and routing indices).
        if not any(d is not None for d in out_demands):
            return [None] * len(eqn.invars)
        result: list[ContributionDemand | None] = []
        for var in eqn.invars:
            if isinstance(var, Literal):
                result.append(None)
                continue
            size = int(np.prod(_get_shape(var)))
            result.append(ContributionDemand(AllRows(size)) if size else None)
        return result

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
        execution: TraceExecution,
    ) -> None:
        return


@TR.register(
    ("shift_left", np.left_shift),
    ("shift_right_arithmetic", np.right_shift),
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
        execution: TraceExecution,
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
    ("is_finite", np.isfinite),
    ("is_nan", np.isnan),
    ("argmax", lambda x, **p: np.argmax(x, axis=p.get("axes"))),
    ("argmin", lambda x, **p: np.argmin(x, axis=p.get("axes"))),
    ("floor", np.floor),
    ("ceil", np.ceil),
    ("round", np.round),
    ("sign", np.sign),
)
class ElementWiseZeroDependencyHandler(ZeroDependencyHandler):
    def propagate_liveness_demand(
        self, eqn: JaxprEqn, state: TraceState, out_demands
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)

        oshp = _get_shape(eqn.outvars[0])

        return [
            None
            if isinstance(var, Literal)
            else _inverse_elementwise_demand(demand, _get_shape(var), oshp)
            for var in eqn.invars
        ]


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
        execution: TraceExecution,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if in_d.shape != oshp and int(np.prod(in_d.shape)) == int(np.prod(oshp)):
            dep_out = in_d.reshape(*oshp)
        else:
            dep_out = in_d

        state.set(eqn.outvars[0], dep_out.copy())

        if self.is_nonlinear:
            dep_out.record_couplings(acc, execution.trial_test_split)

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

    def propagate_liveness_demand(
        self, eqn, state, out_demands
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None]
        if int(np.prod(_get_shape(eqn.invars[0]))) != int(
            np.prod(_get_shape(eqn.outvars[0]))
        ):
            return super().propagate_liveness_demand(eqn, state, out_demands)
        return [demand]

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
        execution: TraceExecution,
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
                res.record_couplings(acc, execution.trial_test_split)

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

    def propagate_liveness_demand(
        self, eqn: JaxprEqn, state: TraceState, out_demands
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)
        out_shape = _get_shape(eqn.outvars[0])
        return [
            None
            if isinstance(var, Literal)
            else _inverse_elementwise_demand(demand, _get_shape(var), out_shape)
            for var in eqn.invars
        ]

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
        execution: TraceExecution,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        state.set(eqn.outvars[0], in_d.copy())
        if self.introduces_nonlinearity(eqn, [in_d.dep.nnz > 0]):
            in_d.record_couplings(acc, execution.trial_test_split)

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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        return [out_demands[0]]

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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None]
        if _get_shape(eqn.invars[0]) == _get_shape(eqn.outvars[0]):
            return [demand]
        if int(np.prod(_get_shape(eqn.invars[0]))) == 1:
            return [ContributionDemand(RangeRows(0, 1))]
        return [
            _demand(
                _inverse_broadcast_rows(
                    demand.rows,
                    _get_shape(eqn.invars[0]),
                    _get_shape(eqn.outvars[0]),
                    eqn.params["broadcast_dimensions"],
                )
            )
        ]


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
        execution: TraceExecution,
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
        rows = np.ravel_multi_index(
            tuple(cast(NDArray, coord) for coord in in_coords), in_shape
        )
        return ContributionPropagation([_demand(rows)], [])

    def propagate_liveness_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None]
        if tuple(eqn.params["permutation"]) == tuple(
            range(len(_get_shape(eqn.invars[0])))
        ):
            return [demand]
        if demand.is_all_rows():
            return [
                ContributionDemand(AllRows(int(np.prod(_get_shape(eqn.invars[0])))))
            ]
        in_shape = _get_shape(eqn.invars[0])
        out_coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
        in_coords = [None] * len(in_shape)
        for out_axis, in_axis in enumerate(eqn.params["permutation"]):
            in_coords[in_axis] = out_coords[out_axis]
        return [
            _demand(
                np.ravel_multi_index(
                    tuple(cast(NDArray, coord) for coord in in_coords), in_shape
                )
            )
        ]


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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None]
        in_shape = _get_shape(eqn.invars[0])
        if len(in_shape) == 1 and (demand.is_all_rows() or demand.is_range_rows()):
            stride = (eqn.params["strides"] or (1,))[0]
            start = (
                eqn.params["start_indices"][0] + demand.rows.start * stride
                if demand.is_range_rows()
                else eqn.params["start_indices"][0]
            )
            stop = (
                eqn.params["start_indices"][0] + demand.rows.stop * stride
                if demand.is_range_rows()
                else eqn.params["limit_indices"][0]
            )
            if stride == 1:
                return [ContributionDemand(RangeRows(int(start), int(stop)))]
        coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
        strides = eqn.params["strides"] or (1,) * len(in_shape)
        source = tuple(
            s + c * stride
            for s, c, stride in zip(eqn.params["start_indices"], coords, strides)
        )
        return [_demand(np.ravel_multi_index(source, in_shape))]


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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand.is_all_rows():
            return [
                ContributionDemand(AllRows(int(np.prod(_get_shape(eqn.invars[0])))))
            ]
        result = self.propagate_contribution_demand(eqn, state, out_demands)
        return result.in_demands


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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        result = self.propagate_contribution_demand(eqn, state, out_demands)
        # Padding values are numerical inputs for padding-only output entries;
        # retain them conservatively whenever padding output is demanded.
        if (
            out_demands[0] is not None
            and len(eqn.invars) > 1
            and not isinstance(eqn.invars[1], Literal)
        ):
            result.in_demands[1] = _demand(
                np.arange(state.get(eqn.invars[1]).dep.shape[0])
            )
        return result.in_demands


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
        execution: TraceExecution,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        axis = eqn.params.get("dimension", 0)
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if axis == 0:
            concatenated = sps.vstack([d.dep for d in in_d], format="csr")
        else:
            indices_list = []
            # TODO: apparantly the offset is required, such that indices are correctly
            # propagated if there are multiple concatenated operands. Need to check again
            # how the concatenate primitive works exactly..
            offset = 0
            for d in in_d:
                n_items = d.dep.shape[0]
                arr = np.arange(n_items).reshape(d.shape) + offset
                indices_list.append(arr)
                offset += n_items
            perm = np.concatenate(indices_list, axis=axis).ravel()
            stacked = sps.vstack([d.dep for d in in_d], format="csr")
            concatenated = stacked[perm]
        state.set(eqn.outvars[0], SparseDepSet(concatenated, oshp))

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)
        shapes = [_get_shape(v) for v in eqn.invars]
        axis = eqn.params.get("dimension", 0)
        # Axis-zero concatenation preserves each input's contiguous C-order
        # interval.  Keep full/range demands compact instead of constructing the
        # whole source-ID routing tensor.
        # Purely performance optimization!
        if axis == 0 and (demand.is_all_rows() or demand.is_range_rows()):
            start = demand.rows.start if demand.is_range_rows() else 0
            stop = (
                demand.rows.stop
                if demand.is_range_rows()
                else int(np.prod(_get_shape(eqn.outvars[0])))
            )
            result: list[ContributionDemand | None] = []
            offset = 0
            for shape, var in zip(shapes, eqn.invars):
                size = int(np.prod(shape))
                lo, hi = max(start, offset), min(stop, offset + size)
                result.append(
                    None
                    if isinstance(var, Literal) or lo >= hi
                    else ContributionDemand(RangeRows(lo - offset, hi - offset))
                )
                offset += size
            return result
        offsets, index_parts = 0, []
        for shape in shapes:
            size = int(np.prod(shape))
            index_parts.append(np.arange(size).reshape(shape) + offsets)
            offsets += size
        selected = np.concatenate(
            index_parts, axis=eqn.params.get("dimension", 0)
        ).ravel()[demand.rows]
        result, offset = [], 0
        for shape, var in zip(shapes, eqn.invars):
            size = int(np.prod(shape))
            result.append(
                None
                if isinstance(var, Literal)
                else _demand(
                    selected[(selected >= offset) & (selected < offset + size)] - offset
                )
            )
            offset += size
        return result


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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        if out_demands[0] is not None and out_demands[0].is_all_rows():
            return [
                None
                if isinstance(var, Literal)
                else ContributionDemand(AllRows(int(np.prod(_get_shape(var)))))
                for var in eqn.invars
            ]
        return self.propagate_contribution_demand(eqn, state, out_demands).in_demands


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
        execution: TraceExecution,
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        return self.propagate_contribution_demand(eqn, state, out_demands).in_demands


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

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        execution: TraceExecution,
    ) -> None:
        src_var, indices_var = eqn.invars[:2]

        src_dep = state.get(src_var)
        indices = state.get_val(indices_var)
        output_shape = _get_shape(eqn.outvars[0])

        if indices is not None:
            routes = gather_routes(eqn, np.asarray(indices), demanded_output_rows=None)

            if routes is not None:
                source_rows, _index_rows = routes

                n_output = source_rows.size
                valid_output_rows = np.flatnonzero(source_rows >= 0).astype(np.int64)

                # Build a sparse selection matrix:
                # out_dep[i] = src_dep[source_rows[i]]
                # Invalid FILL_OR_DROP outputs receive an empty dependency row.
                selection = sps.csr_matrix(
                    (
                        np.ones(valid_output_rows.size, dtype=bool),
                        (valid_output_rows, source_rows[valid_output_rows]),
                    ),
                    shape=(n_output, src_dep.dep.shape[0]),
                    dtype=bool,
                )

                output_dep = (selection @ src_dep.dep).astype(bool).tocsr()

                state.set(eqn.outvars[0], SparseDepSet(output_dep, output_shape))
                return

        # Conservative fallback.
        total = src_dep.total_union()
        output_dep = _broadcast_single_row(total.dep, int(np.prod(output_shape)))

        state.set(eqn.outvars[0], SparseDepSet(output_dep, output_shape))

    def propagate_liveness_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None, None]
        # Use the same concrete routes understood by forward propagation.  The
        # remaining general gather forms deliberately retain the full operand.
        src, indices_var = eqn.invars[:2]
        indices = state.get_val(indices_var)

        if indices is not None:
            routes = gather_routes(eqn, np.asarray(indices), demand_rows(demand))

            if routes is not None:
                source_rows, index_rows = routes
                # _demand removes the -1 sentinel used for FILL_OR_DROP outputs
                return [
                    _demand(source_rows),
                    _demand(index_rows),
                ]

        # concrete indices unavailable or unsupported gather config
        return [
            _demand(np.arange(state.get(src).dep.shape[0], dtype=np.int64)),
            _demand(np.arange(state.get(indices_var).dep.shape[0], dtype=np.int64)),
        ]


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
        execution: TraceExecution,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        d_operand = in_d[0]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        slice_sizes = eqn.params.get("slice_sizes", ())

        # TODO: this is wrong for dynamic_update_slice, because it ignores the update
        # operand. This needs to be fixed, but for now we just handle the dynamic_slice
        # case.
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)
        if eqn.primitive.name != "dynamic_slice":
            return super().propagate_liveness_demand(eqn, state, out_demands)
        source, starts = eqn.invars[0], eqn.invars[1:]
        start_vals = [state.get_val(v) for v in starts]
        index_demands = [
            None
            if isinstance(v, Literal)
            else _demand(np.arange(state.get(v).dep.shape[0]))
            for v in starts
        ]
        if not all(v is not None for v in start_vals):
            return [_demand(np.arange(state.get(source).dep.shape[0])), *index_demands]
        try:
            coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
            sizes = _get_shape(source)
            start = [
                min(max(int(np.asarray(v)), 0), dim - extent)
                for v, dim, extent in zip(start_vals, sizes, eqn.params["slice_sizes"])
            ]
            src_coords = tuple(c + s for c, s in zip(coords, start))
            return [_demand(np.ravel_multi_index(src_coords, sizes)), *index_demands]
        except (ValueError, TypeError):
            return [_demand(np.arange(state.get(source).dep.shape[0])), *index_demands]


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

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        execution: TraceExecution,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        d_tgt = in_d[0]
        d_vals = in_d[2]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        nonlinear = eqn.primitive.name == "scatter-mul"

        u_vals = sps.csr_matrix(d_vals.dep.sum(axis=0).astype(bool))
        idx = state.get_val(eqn.invars[1])

        if idx is not None and d_tgt.dep.shape[0] > 1:
            routes = scatter_routes(eqn, idx, include_index_rows=False)
            if routes is not None:
                target_rows, _index_rows = routes
                coo_vals = d_vals.dep.tocoo()
                mapped = target_rows[coo_vals.row]
                valid = mapped >= 0
                scattered_mat = sps.csr_matrix(
                    (coo_vals.data[valid], (mapped[valid], coo_vals.col[valid])),
                    shape=d_tgt.dep.shape,
                )
                res_dep = (d_tgt.dep + scattered_mat).tocsr()
                res_dep.data[:] = 1
                res = SparseDepSet(res_dep, oshp)
                state.set(eqn.outvars[0], res)
                if nonlinear:
                    res.record_couplings(acc, execution.trial_test_split)
                return

        result = d_tgt.dep + _broadcast_single_row(u_vals, int(np.prod(oshp)))
        res_dep = result.tocsr()
        res_dep.data[:] = 1
        res = SparseDepSet(res_dep, oshp)
        state.set(eqn.outvars[0], res)
        if nonlinear:
            res.record_couplings(acc, execution.trial_test_split)

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)
        operand, indices, updates = eqn.invars[:3]
        base = None if isinstance(operand, Literal) else demand
        if isinstance(indices, Literal) or isinstance(updates, Literal):
            return [base, None, None]
        concrete_indices = state.get_val(indices)
        if concrete_indices is not None:
            # First pass only identifies update rows that land in the live
            # output set.  Index rows are needed only for that selected subset.
            routes = scatter_routes(eqn, concrete_indices, include_index_rows=False)
            if routes is not None:
                target_rows, _index_rows = routes
                update_rows = np.flatnonzero(np.isin(target_rows, demand.rows)).astype(
                    np.int64
                )
                # Reroute only selected updates so the index demand excludes
                # index vectors used solely by dead updates.
                selected_routes = scatter_routes(eqn, concrete_indices, update_rows)
                if selected_routes is not None:
                    _selected_targets, selected_index_rows = selected_routes
                    if selected_index_rows is not None:
                        return [
                            base,
                            _demand(selected_index_rows),
                            _demand(update_rows),
                        ]
        return [
            base,
            _demand(np.arange(int(np.prod(_get_shape(indices))), dtype=np.int64)),
            _demand(np.arange(int(np.prod(_get_shape(updates))), dtype=np.int64)),
        ]


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
        execution: TraceExecution,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        cond_val = state.get_val(eqn.invars[0])
        cases_d = in_d[1:]

        if cond_val is not None and len(cases_d) >= 2:
            try:
                cond_flat = cond_val.ravel().astype(int)
                n_out = int(np.prod(oshp))
                # Gather each selected case in one sparse operation.  An earlier
                # implementation sliced a CSR matrix once per output row and then
                # called ``vstack``; a gathered array with O(N) entries therefore
                # created O(N) temporary CSR matrices.
                selected = np.zeros(n_out, dtype=np.intp)
                selected[: min(n_out, cond_flat.size)] = cond_flat[:n_out]
                np.clip(selected, 0, len(cases_d) - 1, out=selected)

                blocks = []
                output_rows = []
                for case_index, case_dep_set in enumerate(cases_d):
                    rows = np.flatnonzero(selected == case_index)
                    if rows.size == 0:
                        continue
                    case_dep = case_dep_set.dep
                    if case_dep.shape[0] == 0:
                        raise ValueError("Cannot select from an empty dependency set")
                    blocks.append(case_dep[rows % case_dep.shape[0]])
                    output_rows.append(rows)

                stacked = sps.vstack(blocks, format="csr")
                # ``blocks`` are grouped by case; restore the original output order.
                restore_order = np.argsort(np.concatenate(output_rows))
                res_dep = stacked[restore_order]
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None] * len(eqn.invars)
        out_shape = _get_shape(eqn.outvars[0])
        return [
            None
            if isinstance(v, Literal)
            else _inverse_elementwise_demand(demand, _get_shape(v), out_shape)
            for v in eqn.invars
        ]


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
        execution: TraceExecution,
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
                    acc.record_dep(A_active, execution.trial_test_split)
                if state.is_nonlinear(eqn.invars[1]):
                    acc.record_dep(B_active, execution.trial_test_split)

                P_cross = (A_active.T @ B_active).tocsr()
                r_c, c_c = P_cross.nonzero()
                if execution.trial_test_split is not None:
                    mask_c = (r_c < execution.trial_test_split) & (
                        c_c >= execution.trial_test_split
                    )
                    mask_c |= (r_c >= execution.trial_test_split) & (
                        c_c < execution.trial_test_split
                    )
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
                    ua.record_couplings(acc, execution.trial_test_split)
                if state.is_nonlinear(eqn.invars[1]):
                    ub.record_couplings(acc, execution.trial_test_split)

                r_c = np.repeat(ia, ib.size)
                c_c = np.tile(ib, ia.size)
                if execution.trial_test_split is not None:
                    mask = (
                        (r_c < execution.trial_test_split)
                        & (c_c >= execution.trial_test_split)
                    ) | (
                        (r_c >= execution.trial_test_split)
                        & (c_c < execution.trial_test_split)
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        demand = out_demands[0]
        if demand is None:
            return [None, None]
        lhs, rhs = eqn.invars[:2]
        la, rb = _get_shape(lhs), _get_shape(rhs)
        try:
            dn = eqn.params["dimension_numbers"]
            if hasattr(dn, "lhs_batch_dimensions"):
                lb, rb_dims = (
                    tuple(dn.lhs_batch_dimensions),
                    tuple(dn.rhs_batch_dimensions),
                )
                lc, rc = (
                    tuple(dn.lhs_contracting_dimensions),
                    tuple(dn.rhs_contracting_dimensions),
                )
            else:
                lc, rc = tuple(dn[0][0]), tuple(dn[0][1])
                lb, rb_dims = tuple(dn[1][0]), tuple(dn[1][1])
            lf = tuple(i for i in range(len(la)) if i not in lb and i not in lc)
            rf = tuple(i for i in range(len(rb)) if i not in rb_dims and i not in rc)
            out_coords = np.unravel_index(demand.rows, _get_shape(eqn.outvars[0]))
            n_batch, n_lfree = len(lb), len(lf)
            contract_shape = tuple(la[i] for i in lc)
            contract_coords = (
                np.indices(contract_shape).reshape(len(lc), -1)
                if lc
                else np.empty((0, 1), int)
            )
            lhs_rows, rhs_rows = [], []
            for col in range(len(demand.rows)):
                lcoords = [0] * len(la)
                rcoords = [0] * len(rb)
                for j, axis in enumerate(lb):
                    lcoords[axis] = out_coords[j][col]
                for j, axis in enumerate(rb_dims):
                    rcoords[axis] = out_coords[j][col]
                for j, axis in enumerate(lf):
                    lcoords[axis] = out_coords[n_batch + j][col]
                for j, axis in enumerate(rf):
                    rcoords[axis] = out_coords[n_batch + n_lfree + j][col]
                for k in range(contract_coords.shape[1]):
                    ll, rr = list(lcoords), list(rcoords)
                    for j, axis in enumerate(lc):
                        ll[axis] = contract_coords[j, k]
                    for j, axis in enumerate(rc):
                        rr[axis] = contract_coords[j, k]
                    lhs_rows.append(np.ravel_multi_index(tuple(ll), la))
                    rhs_rows.append(np.ravel_multi_index(tuple(rr), rb))
            return [_demand(np.asarray(lhs_rows)), _demand(np.asarray(rhs_rows))]
        except (KeyError, ValueError, AttributeError, TypeError):
            return [
                _demand(np.arange(state.get(lhs).dep.shape[0])),
                _demand(np.arange(state.get(rhs).dep.shape[0])),
            ]


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
        execution: TraceExecution,
    ) -> None:
        # TODO: reduce_window_* has window geometry rather than reduction axes, so this is
        # wrong for this primitive. Never found this primitive, but must be addressed at
        # some point
        if eqn.primitive.name == "reduce_window_sum":
            warnings.warn(
                "reduce_window_sum is not fully supported in dependency propagation. Proceed with caution."
            )
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

    def propagate_liveness_demand(
        self, eqn: JaxprEqn, state: TraceState, out_demands
    ) -> list[ContributionDemand | None]:
        demand = out_demands[0]
        if demand is None:
            return [None]
        in_shape = _get_shape(eqn.invars[0])
        if demand.is_all_rows():
            return [ContributionDemand(AllRows(int(np.prod(in_shape))))]
        if "axes" not in eqn.params:
            # reduce_window_* has window geometry rather than reduction axes.
            return [ContributionDemand(AllRows(int(np.prod(in_shape))))]
        axes = set(eqn.params.get("axes", ()))
        kept = [axis for axis in range(len(in_shape)) if axis not in axes]
        out_shape = _get_shape(eqn.outvars[0])
        out_coords = np.unravel_index(demand.rows, out_shape) if out_shape else ()
        grid = np.indices(in_shape).reshape(len(in_shape), -1)
        selected = np.zeros(grid.shape[1], dtype=bool)
        for i in range(len(demand.rows)):
            match = np.ones(grid.shape[1], dtype=bool)
            for j, axis in enumerate(kept):
                # JAX reduce_* removes reduced axes; retain this fallback for
                # keepdims-compatible custom lowerings too.
                out_axis = axis if len(out_shape) == len(in_shape) else j
                match &= grid[axis] == out_coords[out_axis][i]
            selected |= match
        return [_demand(np.flatnonzero(selected))]


# =============================================================================
# 5. Opaque Black-Box Handler
# =============================================================================


@TR.register(
    "lu",
    "triangular_solve",
    "custom_linear_solve",
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
        execution: TraceExecution,
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
                acc.record_dep(total.dep, execution.trial_test_split)

        for ov in eqn.outvars:
            oshp = _get_shape(ov)
            stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
            state.set(ov, SparseDepSet(stacked_dep, oshp))

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return any(invar_active)

    def propagate_liveness_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None]:
        if not any(d is not None for d in out_demands):
            return [None] * len(eqn.invars)

        p = eqn.primitive.name
        if p == "custom_linear_solve":
            res = self._custom_linear_solve_liveness(eqn, state, out_demands)
            if res is not None:
                return res
        if p == "lu":
            res = self._lu_liveness(eqn, state, out_demands)
            if res is not None:
                return res
        if p in ("triangular_solve", "lu_solve", "cholesky", "eig", "eigh"):
            res = self._batched_linalg_liveness(eqn, state, out_demands)
            if res is not None:
                return res

        return super().propagate_liveness_demand(eqn, state, out_demands)

    @staticmethod
    def _lu_liveness(
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None] | None:
        input_var = eqn.invars[0]
        input_shape = _get_shape(input_var)

        if len(input_shape) < 2:
            return None

        batch_shape = input_shape[:-2]
        n_batches = int(np.prod(batch_shape)) if batch_shape else 1
        demanded_batches: list[np.ndarray] = []

        for output_index, (outvar, demand) in enumerate(zip(eqn.outvars, out_demands)):
            if demand is None:
                continue

            out_shape = _get_shape(outvar)
            if output_index == 0:
                block_size = int(np.prod(out_shape[-2:]))
            else:
                block_size = int(np.prod(out_shape[len(batch_shape) :]))

            demanded_batches.append(np.unique(demand_rows(demand) // block_size))

        if not demanded_batches:
            return [None]

        batch_ids = np.unique(np.concatenate(demanded_batches))

        if np.any(batch_ids >= n_batches):
            return None

        matrix_block_size = int(np.prod(input_shape[-2:]))
        input_rows = (
            batch_ids[:, None] * matrix_block_size
            + np.arange(
                matrix_block_size,
                dtype=np.int64,
            )[None, :]
        ).ravel()

        return [_demand(input_rows)]

    @staticmethod
    def _batched_linalg_liveness(
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None] | None:
        batch_ids: list[np.ndarray] = []
        batch_shape: tuple[int, ...] | None = None

        for outvar, demand in zip(eqn.outvars, out_demands):
            if demand is None:
                continue

            out_shape = _get_shape(outvar)

            if len(out_shape) < 2:
                continue

            candidate_batch_shape = out_shape[:-1]
            n_batches = int(np.prod(candidate_batch_shape))
            block_size = int(np.prod(out_shape[len(candidate_batch_shape) :]))

            if block_size == 0:
                continue

            ids = np.unique(demand_rows(demand) // block_size)

            if np.any(ids >= n_batches):
                return None

            if batch_shape is None:
                batch_shape = candidate_batch_shape
            elif batch_shape != candidate_batch_shape:
                return None

            batch_ids.append(ids)

        if batch_shape is None or not batch_ids:
            return None

        selected_batches = np.unique(np.concatenate(batch_ids))
        n_batches = int(np.prod(batch_shape))

        result: list[ContributionDemand | None] = []

        for invar in eqn.invars:
            if isinstance(invar, Literal):
                result.append(None)
                continue

            in_shape = _get_shape(invar)

            if (
                len(in_shape) >= len(batch_shape)
                and tuple(in_shape[: len(batch_shape)]) == batch_shape
            ):
                block_size = int(np.prod(in_shape[len(batch_shape) :]))

                rows = (
                    selected_batches[:, None] * block_size
                    + np.arange(block_size, dtype=np.int64)[None, :]
                ).ravel()

                result.append(_demand(rows))
            else:
                result.append(
                    _demand(
                        np.arange(
                            int(np.prod(in_shape)),
                            dtype=np.int64,
                        )
                    )
                )

        return result

    @staticmethod
    def _custom_linear_solve_liveness(
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None] | None:
        """Preserve independent leading batch systems.

        This handles layouts such as:

            matrix: (n_batch, m, m)
            rhs:    (n_batch, m)
            result: (n_batch, m)

        A demand on selected result entries retains complete matrix/RHS blocks
        only for the corresponding batch systems.
        """
        demanded_output = next(
            (
                (outvar, demand)
                for outvar, demand in zip(eqn.outvars, out_demands)
                if demand is not None
            ),
            None,
        )

        if demanded_output is None:
            return [None] * len(eqn.invars)

        outvar, demand = demanded_output
        out_shape = _get_shape(outvar)

        # A linear solve result normally has at least one batch axis and one
        # event axis. Scalar/unbatched cases use the conservative fallback.
        if len(out_shape) < 2:
            return None

        # Treat the final result axis as the solve/event dimension. All leading
        # dimensions identify independent systems.
        batch_shape = out_shape[:-1]
        event_shape = out_shape[len(batch_shape) :]

        n_batches = int(np.prod(batch_shape))
        output_event_size = int(np.prod(event_shape))

        if n_batches == 0 or output_event_size == 0:
            return [None] * len(eqn.invars)

        batch_ids = np.unique(demand_rows(demand) // output_event_size).astype(np.int64)

        if np.any(batch_ids < 0) or np.any(batch_ids >= n_batches):
            return None

        input_demands: list[ContributionDemand | None] = []

        for invar in eqn.invars:
            if isinstance(invar, Literal):
                input_demands.append(None)
                continue

            in_shape = _get_shape(invar)
            in_size = int(np.prod(in_shape))

            # Inputs beginning with the same batch shape are interpreted as
            # per-system operands. Retain complete trailing blocks only for
            # demanded systems.
            if len(in_shape) >= len(batch_shape) and tuple(
                in_shape[: len(batch_shape)]
            ) == tuple(batch_shape):
                block_size = int(np.prod(in_shape[len(batch_shape) :]))

                rows = (
                    batch_ids[:, None] * block_size
                    + np.arange(
                        block_size,
                        dtype=np.int64,
                    )[None, :]
                ).ravel()

                input_demands.append(_demand(rows))
                continue

            # Small shared coefficients or other unbatched operands are used
            # by every selected solve and must be retained completely.
            input_demands.append(_demand(np.arange(in_size, dtype=np.int64)))

        return input_demands


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
        execution: TraceExecution,
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

        sub_state.run_bound_eqns(sub_bound_eqns, acc, execution.trial_test_split)

        for pv, sv in zip(eqn.outvars, sub.outvars):
            state.set(pv, sub_state.get(sv))
            if sub_state.is_nonlinear(sv):
                state.nonlinear_ids.add(id(pv))
            val = sub_state.get_val(sv)
            if val is not None:
                state.val_of[id(pv)] = val

        # Propagating concrete values equation-by-equation is sufficient for
        # most sub-jaxprs.  Evaluate the complete sub-jaxpr only when the
        # backwards routing analysis explicitly demands an output value.
        in_vals = [state.get_val(v) for v in eqn.invars]
        if execution.needs_concrete and not any(v is None for v in in_vals):
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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        """Run the ordinary reverse rules inside a call boundary."""
        sub, sub_consts = _subjaxpr_and_consts(eqn)
        info = cast(SubEqnInfo, state.sub_info[id(eqn)])

        _active, _indices, bound = info
        sub_state = TraceState(state.n_dofs, set(), state.sub_info, state.nonlinear_ids)
        for parent, child in zip(eqn.invars, sub.invars):
            sub_state.set(child, state.get(parent))
            value = state.get_val(parent)
            if value is not None:
                sub_state.val_of[id(child)] = value
        for child, value in zip(sub.constvars, sub_consts):
            sub_state.set(child, SparseDepSet.empty(_get_shape(child), state.n_dofs))
            sub_state.val_of[id(child)] = np.asarray(value)
        demands = {id(v): d for v, d in zip(sub.outvars, out_demands) if d is not None}
        for subeqn, handler, _active, _concrete in reversed(bound):
            outgoing = [demands.pop(id(v), None) for v in subeqn.outvars]
            if not any(outgoing):
                continue
            incoming = handler.propagate_liveness_demand(subeqn, sub_state, outgoing)
            for var, required in zip(subeqn.invars, incoming):
                if required is not None:
                    demands[id(var)] = merge_demands(demands.get(id(var)), required)

        # TODO: for rewriting the jaxpr later, the nested liveness demands must be
        # stored/retained! Likely in state.sub_liveness[id(eqn)] = SubJaxprLivenessPlan(...)
        return [demands.get(id(v)) for v in sub.invars]


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
        execution: TraceExecution,
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

            sub_state.run_bound_eqns(sub_bound_eqns, acc, execution.trial_test_split)

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

    def propagate_liveness_demand(self, eqn, state, out_demands):
        if not any(out_demands):
            return [None] * len(eqn.invars)
        # Predicate is always required: both branches are retained because its
        # concrete value must not select away a potentially live branch.
        predicate = (
            None
            if isinstance(eqn.invars[0], Literal)
            else _demand(np.arange(state.get(eqn.invars[0]).dep.shape[0]))
        )
        merged: list[ContributionDemand | None] = [None] * (len(eqn.invars) - 1)
        branch_info = cast(list[SubEqnInfo], state.sub_info.get(id(eqn), []))
        for info, closed in zip(branch_info, eqn.params["branches"]):
            bound = info[2]
            sub = closed.jaxpr
            sub_state = TraceState(
                state.n_dofs, set(), state.sub_info, state.nonlinear_ids
            )
            for parent, child in zip(eqn.invars[1:], sub.invars):
                sub_state.set(child, state.get(parent))
            demands = {
                id(v): d for v, d in zip(sub.outvars, out_demands) if d is not None
            }
            for subeqn, handler, _active, _concrete in reversed(bound):
                outgoing = [demands.pop(id(v), None) for v in subeqn.outvars]
                if any(outgoing):
                    for var, required in zip(
                        subeqn.invars,
                        handler.propagate_liveness_demand(subeqn, sub_state, outgoing),
                    ):
                        if required is not None:
                            demands[id(var)] = merge_demands(
                                demands.get(id(var)), required
                            )
            for i, var in enumerate(sub.invars):
                merged[i] = merge_demands(merged[i], demands.get(id(var)))
        # Missing sub-info should still be sound.
        if not branch_info:
            merged = super().propagate_liveness_demand(eqn, state, out_demands)[1:]
        return [predicate, *merged]


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
        execution: TraceExecution,
    ) -> None:
        closed_sub = eqn.params["jaxpr"]
        sub, sub_consts = closed_sub.jaxpr, closed_sub.consts

        # A scan's body jaxpr describes one iteration, so evaluating that jaxpr
        # directly does not produce the scan outputs.  Index routing inside a
        # surrounding jit (as generated by ``jnp.searchsorted`` in
        # ``BinarySearchStrategy.lift``) needs the actual carry/output values.
        # Evaluate the higher-order primitive itself when possible and retain all
        # of its outputs for downstream gather/select handlers.
        in_vals = [state.get_val(v) for v in eqn.invars]
        if execution.needs_concrete and not any(v is None for v in in_vals):
            try:
                concrete_outs = eqn.primitive.bind(
                    *[jnp.asarray(v) for v in in_vals], **eqn.params
                )
                if not isinstance(concrete_outs, tuple):
                    concrete_outs = (concrete_outs,)
                for ovar, value in zip(eqn.outvars, concrete_outs):
                    state.val_of[id(ovar)] = np.asarray(value)
            except (TypeError, ValueError, RuntimeError, KeyError, AttributeError):
                pass

        # The binary-search implementation lowers to a scan over only static
        # index arrays.  Once its concrete outputs are available, tracing its
        # symbolic body would merely lose the per-iteration branch choices and
        # introduce spurious dependencies.  More generally, a higher-order
        # primitive with no dependent inputs has no dependent outputs.
        if all(state.is_inactive(v) for v in eqn.invars):
            for ovar in eqn.outvars:
                state.set(ovar, SparseDepSet.empty(_get_shape(ovar), state.n_dofs))
            return

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
                    if execution.trial_test_split is not None:
                        mask = (c_a < execution.trial_test_split) & (
                            c_b >= execution.trial_test_split
                        )
                        mask |= (c_a >= execution.trial_test_split) & (
                            c_b < execution.trial_test_split
                        )
                        c_a, c_b = c_a[mask], c_b[mask]
                    acc.add_coords(c_a, c_b)
                    acc.add_coords(c_b, c_a)
            else:
                couplings = (col_a.T @ col_b).tocsr()
                r, c = couplings.nonzero()
                if execution.trial_test_split is not None:
                    mask = (r < execution.trial_test_split) & (
                        c >= execution.trial_test_split
                    )
                    mask |= (r >= execution.trial_test_split) & (
                        c < execution.trial_test_split
                    )
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
                if sub_y_val is not None and id(y) not in state.val_of:
                    try:
                        state.val_of[id(y)] = np.broadcast_to(sub_y_val, y_shape).copy()
                    except (TypeError, ValueError, RuntimeError, KeyError):
                        pass

    # def propagate_liveness_demand(
    #     self, eqn: JaxprEqn, state: TraceState, out_demands
    # ) -> list[ContributionDemand | None]:
    #     if not any(out_demands):
    #         return [None] * len(eqn.invars)
    #     # Conservative by design: a live scan output may depend on every carry
    #     # state and every iteration of each xs input.  Iteration compaction needs
    #     # a fixed-point body analysis and is intentionally deferred.
    #     return [
    #         None
    #         if isinstance(v, Literal)
    #         else _demand(np.arange(state.get(v).dep.shape[0], dtype=np.int64))
    #         for v in eqn.invars
    #     ]
    def propagate_liveness_demand(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        out_demands: list[ContributionDemand | None],
    ) -> list[ContributionDemand | None]:
        if not any(d is not None for d in out_demands):
            return [None] * len(eqn.invars)

        num_consts = eqn.params.get("num_consts", 0)
        num_carry = eqn.params.get("num_carry", 0)
        # Exact iteration-local propagation currently supports only scans without
        # carries.  Carries need a reverse fixed point across iterations, so retain
        # the established conservative all-input rule for now.
        if num_carry:
            return super().propagate_liveness_demand(eqn, state, out_demands)

        try:
            length = int(eqn.params["length"])
            sub, sub_consts = _subjaxpr_and_consts(eqn)
            sub_active, _sub_index_set, sub_bound_eqns = cast(
                SubEqnInfo, state.sub_info[id(eqn)]
            )
            if len(eqn.outvars) != len(sub.outvars) or len(eqn.invars) < num_consts:
                raise ValueError("incompatible scan JAXPR arity")
            if any(needs_concrete for *_prefix, needs_concrete in sub_bound_eqns):
                routed_parents = eqn.invars[:num_consts] + eqn.invars[num_consts:]
                if any(state.get_val(parent) is None for parent in routed_parents):
                    raise ValueError("missing concrete values for scan body routing")

            # parent output -> iteration -> body output rows
            seeds_by_iteration: dict[int, dict[int, ContributionDemand]] = {}
            for parent_out, body_out, demand in zip(
                eqn.outvars, sub.outvars, out_demands
            ):
                if demand is None:
                    continue
                body_size = int(np.prod(_get_shape(body_out)))
                parent_shape = _get_shape(parent_out)
                if (
                    not parent_shape
                    or parent_shape[0] != length
                    or int(np.prod(parent_shape[1:])) != body_size
                ):
                    raise ValueError("incompatible stacked scan output shape")
                iterations, body_rows = divmod(demand_rows(demand), body_size)
                for iteration in np.unique(iterations):
                    rows = body_rows[iterations == iteration]
                    seed = _demand(rows)
                    if seed is not None:
                        bucket = seeds_by_iteration.setdefault(int(iteration), {})
                        bucket[id(body_out)] = merge_demands(
                            bucket.get(id(body_out)), seed
                        )

            in_demands: list[ContributionDemand | None] = [None] * len(eqn.invars)
            xs_start = num_consts
            if not seeds_by_iteration:
                return in_demands

            for iteration, seeds in seeds_by_iteration.items():
                sub_state = TraceState(
                    state.n_dofs, sub_active, state.sub_info, state.nonlinear_ids
                )
                # Constants are shared by every scan iteration.
                for parent, child in zip(
                    eqn.invars[:num_consts], sub.invars[:num_consts]
                ):
                    value = state.get_val(parent)
                    sub_state.set(child, state.get(parent))
                    if value is not None:
                        sub_state.val_of[id(child)] = np.asarray(value)
                for child, value in zip(sub.constvars, sub_consts):
                    sub_state.set(
                        child, SparseDepSet.empty(_get_shape(child), state.n_dofs)
                    )
                    sub_state.val_of[id(child)] = np.asarray(value)

                # xs are stacked in the parent and one slice is supplied to the body.
                for parent, child in zip(eqn.invars[xs_start:], sub.invars[xs_start:]):
                    parent_value = state.get_val(parent)
                    parent_shape, body_shape = _get_shape(parent), _get_shape(child)
                    if (
                        not parent_shape
                        or parent_shape[0] != length
                        or parent_shape[1:] != body_shape
                    ):
                        raise ValueError("incompatible scan xs slice")
                    sub_state.set(child, SparseDepSet.empty(body_shape, state.n_dofs))
                    if parent_value is not None:
                        sub_state.val_of[id(child)] = np.asarray(parent_value)[
                            iteration
                        ]

                # Populate concrete routing values produced inside the body before
                # reverse propagation (e.g. gather/scatter index intermediates).
                sub_state.run_bound_eqns(
                    sub_bound_eqns, CouplingAccumulator(state.n_dofs)
                )
                body_demands = propagate_demands_backward(
                    sub_bound_eqns, sub_state, seeds
                )

                for i, body_input in enumerate(sub.invars[:num_consts]):
                    required = body_demands.get(id(body_input))
                    if required is not None:
                        in_demands[i] = merge_demands(in_demands[i], required)
                for offset, body_input in enumerate(sub.invars[xs_start:]):
                    required = body_demands.get(id(body_input))
                    if required is None:
                        continue
                    input_index = xs_start + offset
                    body_size = int(np.prod(_get_shape(body_input)))
                    parent_rows = iteration * body_size + demand_rows(required)
                    mapped = _demand(parent_rows)
                    in_demands[input_index] = merge_demands(
                        in_demands[input_index], mapped
                    )
            return in_demands
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return super().propagate_liveness_demand(eqn, state, out_demands)


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
        execution: TraceExecution,
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
                eqn, state, acc, execution
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
                acc.record_dep(row, execution.trial_test_split)
            slice_rows.append(row)

        for ovar in eqn.outvars:
            oshp = _get_shape(ovar)
            core_o = int(np.prod(oshp)) // B
            blocks = [_broadcast_single_row(slice_rows[b], core_o) for b in range(B)]
            stacked = sps.vstack(blocks).tocsr()
            state.set(ovar, SparseDepSet(stacked, oshp))
