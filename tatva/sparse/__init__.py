from ._extraction import (
    pattern_from_compound,
    pattern_from_mesh,
)
from .base import ColoredMatrix, jacfwd, linearized_jacfwd
from .tracer import (
    pattern_from_energy,
    pattern_from_virtual_work,
    register_elementwise_ffi,
)


def __getattr__(name):
    deprecated_fn = (
        "create_sparsity_pattern",
        "reduce_sparsity_pattern",
        "create_sparsity_pattern_KKT",
        "create_sparsity_pattern_master_slave",
        "get_bc_indices",
    )
    if name in deprecated_fn:
        raise ImportError(
            f"The function {name!r} is deprecated and does not exist anymore. "
            "Please use the new sparsity functions. "
            "Check the documentation for more information. "
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ColoredMatrix",
    "jacfwd",
    "linearized_jacfwd",
    "pattern_from_compound",
    "pattern_from_energy",
    "pattern_from_mesh",
    "pattern_from_virtual_work",
    "register_elementwise_ffi",
]
