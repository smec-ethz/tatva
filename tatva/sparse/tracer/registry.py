from typing import Any

import numpy as np
import scipy.special as sp

from tatva.sparse.tracer.handlers_new import (
    Broadcast,
    ElementwiseUnary,
    IntegerPowHandler,
    PadHandler,
    PrimitiveHandler,
    ReverseHandler,
    SliceHandler,
)


class TracerRegistry:
    """Registry for JAX primitive dependency propagation handlers."""

    def __init__(self):
        self._handlers = {}

    def register(self, primitive_name: str, handler: PrimitiveHandler):
        """Decorator to register a handler for one or more JAX primitives."""
        self._handlers[primitive_name] = handler

    def register_class(self, *args: tuple[str, tuple[Any, ...]]):
        def decorator(cls):
            for name, cls_args in args:
                self._handlers[name] = cls(*cls_args)
            return cls

        return decorator

    def get(self, primitive_name: str, default):
        """Get the registered handler, or return the default."""
        return self._handlers.get(primitive_name, default)


# Global registry instance
TR = TracerRegistry()

# ------------------------------------
# Linear / Affine Unary Primitives
# ------------------------------------

# passthrough primitives
PASSTHROUGH_PRIMITIVES = (
    "neg",
    "abs",
    "convert_element_type",
    "copy",
    "stop_gradient",
    "device_put",
    "conj",
    "real",
    "imag",
)
for prim in PASSTHROUGH_PRIMITIVES:
    TR.register(prim, ElementwiseUnary(False))


def _eval_reshape(x, params):
    return x.reshape(params["new_sizes"])


def _eval_transpose(x, params):
    return x.transpose(params["permutation"])


def _eval_squeeze(x, params):
    return np.squeeze(x, axis=tuple(params["dimensions"]))


def _eval_convert_dtype(x, params):
    return x.astype(params["new_dtype"])


TR.register("reshape", ElementwiseUnary(False, _eval_reshape))
TR.register("transpose", ElementwiseUnary(False, _eval_transpose))
TR.register("squeeze", ElementwiseUnary(False, _eval_squeeze))
TR.register("convert_element_type", ElementwiseUnary(False, _eval_convert_dtype))

# ------------------------------------
# Nonlinear Unary Primitives
# ------------------------------------

_NONLINEAR_UNARY_MAP = {
    # Trigonometric & Inverse Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    # Exponential & Logarithmic
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    # Powers & Roots
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "cbrt": np.cbrt,
    # Hyperbolic & Inverse Hyperbolic
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "atanh": np.arctanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    # Special Functions (scipy.special)
    "erf": sp.erf,
    "erfc": sp.erfc,
    "erfinv": sp.erfinv,
    "lgamma": sp.gammaln,
    "digamma": sp.psi,
    "logistic": sp.expit,  # 1 / (1 + exp(-x))
}
for prim_name, eval_fn in _NONLINEAR_UNARY_MAP.items():
    TR.register(prim_name, ElementwiseUnary(is_nonlinear=True, eval_fn=eval_fn))


TR.register("integer_pow", IntegerPowHandler())

# ------------------------------------
# Other
# ------------------------------------
TR.register("broadcast_in_dim", Broadcast())
TR.register("slice", SliceHandler())
TR.register("rev", ReverseHandler())
TR.register("pad", PadHandler())
