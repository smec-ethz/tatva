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

import numpy as np
import scipy.special as sp

from tatva.sparse.tracer.handlers import (
    BroadcastHandler,
    ConcatenateHandler,
    CondHandler,
    DotHandler,
    DynamicSliceHandler,
    ElementwiseBinary,
    ElementwiseUnary,
    FFICallHandler,
    GatherHandler,
    IntegerPowHandler,
    NoOpHandler,
    OpaqueBlackBoxHandler,
    PadHandler,
    PrimitiveHandler,
    ReductionHandler,
    ReverseHandler,
    ScanMapHandler,
    ScatterHandler,
    SelectNHandler,
    SliceHandler,
    SubJaxprHandler,
    TransposeHandler,
    ZeroDependencyHandler,
)


class TracerRegistry:
    """Registry for JAX primitive dependency propagation handlers."""

    def __init__(self):
        self._handlers: dict[str, PrimitiveHandler] = {}
        self.fallback = OpaqueBlackBoxHandler(record_couplings=False)

    def register(self, primitive_name: str, handler: PrimitiveHandler):
        """Register a handler for a specific JAX primitive."""
        self._handlers[primitive_name] = handler

    def register_many(
        self, primitive_names: tuple[str, ...], handler: PrimitiveHandler
    ):
        """Register a shared handler for multiple JAX primitives."""
        for name in primitive_names:
            self._handlers[name] = handler

    def get(
        self, primitive_name: str, default: PrimitiveHandler | None = None
    ) -> PrimitiveHandler:
        """Get the registered handler, or return the fallback."""
        if default is None:
            default = self.fallback
        handler = self._handlers.get(primitive_name)
        if handler is None:
            warnings.warn(
                f"No handler registered for primitive '{primitive_name}'. Using fallback handler."
            )
            return default
        return handler


# Global registry instance
TR = TracerRegistry()

# -----------------------------------------------------------------------------
# 1. NoOp & Zero Dependency Primitives
# -----------------------------------------------------------------------------

TR.register_many(("debug_print", "debug_callback"), NoOpHandler())
TR.register_many(("iota", "zeros", "ones", "full"), ZeroDependencyHandler())

_COMPARISONS_MAP = {
    # Relational Comparisons
    "lt": np.less,
    "lt_to": np.less,
    "le": np.less_equal,
    "le_to": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    "eq": np.equal,
    "ne": np.not_equal,
    # Logical / Bitwise
    "and": np.logical_and,
    "or": np.logical_or,
    "not": np.logical_not,
    "xor": np.logical_xor,
    "shift_left": np.left_shift,
    "shift_right_arithmetic": np.right_shift,
    # Predicates
    "is_finite": np.isfinite,
    "is_nan": np.isnan,
    # Arg Reductions & Rounding
    "argmax": lambda x, **p: np.argmax(x, axis=p.get("axes")),
    "argmin": lambda x, **p: np.argmin(x, axis=p.get("axes")),
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "sign": np.sign,
}
for prim_name, eval_fn in _COMPARISONS_MAP.items():
    TR.register(prim_name, ZeroDependencyHandler(eval_fn))  # ty: ignore[invalid-argument-type]

# -----------------------------------------------------------------------------
# 2. Linear / Affine Unary & Nonlinear Unary Primitives
# -----------------------------------------------------------------------------

PASSTHROUGH_PRIMITIVES = (
    "neg",
    "abs",
    "copy",
    "stop_gradient",
    "device_put",
    "conj",
    "real",
    "imag",
)
TR.register_many(PASSTHROUGH_PRIMITIVES, ElementwiseUnary(is_nonlinear=False))


def _eval_reshape(x, params):
    return x.reshape(params["new_sizes"])


def _eval_transpose(x, params):
    return x.transpose(params["permutation"])


def _eval_squeeze(x, params):
    return np.squeeze(x, axis=tuple(params["dimensions"]))


def _eval_convert_dtype(x, params):
    return x.astype(params["new_dtype"])


TR.register("reshape", ElementwiseUnary(False, _eval_reshape))
TR.register("transpose", TransposeHandler())
TR.register("squeeze", ElementwiseUnary(False, _eval_squeeze))
TR.register("convert_element_type", ElementwiseUnary(False, _eval_convert_dtype))

_NONLINEAR_UNARY_MAP = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "cbrt": np.cbrt,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "atanh": np.arctanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "erfinv": sp.erfinv,
    "lgamma": sp.gammaln,
    "digamma": sp.psi,
    "logistic": sp.expit,
}
for prim_name, eval_fn in _NONLINEAR_UNARY_MAP.items():
    TR.register(prim_name, ElementwiseUnary(is_nonlinear=True, eval_fn=eval_fn))

TR.register("integer_pow", IntegerPowHandler())

# -----------------------------------------------------------------------------
# 3. Linear & Nonlinear Binary Primitives
# -----------------------------------------------------------------------------

_LINEAR_BINARY_MAP = {
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "max": np.maximum,
    "min": np.minimum,
}
for prim_name, eval_fn in _LINEAR_BINARY_MAP.items():
    TR.register(prim_name, ElementwiseBinary(is_nonlinear=False, eval_fn=eval_fn))


def _eval_div(a, b, **_params):
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.integer) and np.issubdtype(
        b_arr.dtype, np.integer
    ):
        return np.floor_divide(a_arr, b_arr)
    return np.true_divide(a_arr, b_arr)


TR.register("mul", ElementwiseBinary(is_nonlinear=True, eval_fn=np.multiply))
TR.register("div", ElementwiseBinary(is_nonlinear=True, eval_fn=_eval_div))
TR.register("rem", ElementwiseBinary(is_nonlinear=True, eval_fn=np.remainder))
TR.register("pow", ElementwiseBinary(is_nonlinear=True, eval_fn=np.power))
TR.register("atan2", ElementwiseBinary(is_nonlinear=True, eval_fn=np.arctan2))

# -----------------------------------------------------------------------------
# 4. Structural & Dynamic Indexing Primitives
# -----------------------------------------------------------------------------

TR.register("broadcast_in_dim", BroadcastHandler())
TR.register("slice", SliceHandler())
TR.register("rev", ReverseHandler())
TR.register("pad", PadHandler())
TR.register("concatenate", ConcatenateHandler())

TR.register("gather", GatherHandler())
TR.register_many(
    (
        "scatter",
        "scatter-add",
        "scatter-sub",
        "scatter-mul",
        "scatter-min",
        "scatter-max",
    ),
    ScatterHandler(),
)
TR.register("select_n", SelectNHandler())
TR.register("dot_general", DotHandler())

# -----------------------------------------------------------------------------
# 5. Reductions
# -----------------------------------------------------------------------------

TR.register_many(
    ("reduce_sum", "reduce_window_sum", "reduce_max", "reduce_min", "reduce_prod"),
    ReductionHandler(),
)

# -----------------------------------------------------------------------------
# 6. Opaque Black-Box, Callbacks, & Dense Linalg
# -----------------------------------------------------------------------------

_DENSE_LINALG = (
    "lu",
    "custom_linear_solve",
    "triangular_solve",
    "lu_solve",
    "cholesky",
    "eig",
    "eigh",
)
TR.register_many(_DENSE_LINALG, OpaqueBlackBoxHandler(record_couplings=True))

_CALLBACKS = ("custom_vjp_call", "custom_jvp_call", "pure_callback", "io_callback")
TR.register_many(_CALLBACKS, OpaqueBlackBoxHandler(record_couplings=True))

TR.register("dynamic_slice", DynamicSliceHandler())
_OPAQUE_FALLBACKS = ("dynamic_update_slice", "while", "switch")
TR.register_many(_OPAQUE_FALLBACKS, OpaqueBlackBoxHandler(record_couplings=False))

# -----------------------------------------------------------------------------
# 7. Sub-Jaxpr Higher-Order Handlers
# -----------------------------------------------------------------------------

TR.register_many(("pjit", "jit", "remat2"), SubJaxprHandler())
TR.register("cond", CondHandler())
TR.register_many(("scan", "map"), ScanMapHandler())
TR.register("ffi_call", FFICallHandler())
