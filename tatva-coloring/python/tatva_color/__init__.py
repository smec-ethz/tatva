"""
Python package wrapper for the `_tatva_color` native extension.

Provides lightweight typing hints and user-facing docs when importing
`tatva_color` from Python. The compiled extension lives in `_tatva_color`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    from ._tatva_color import (
        distance2_color_and_seeds as _distance2_color_and_seeds_ext,
    )
except Exception:
    _distance2_color_and_seeds_ext = None


def distance2_color_and_seeds(
    row_ptr: Sequence[int], col_idx: Sequence[int], n_dofs: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Color the distance-2 graph induced by a CSR sparse matrix.

    Args:
        row_ptr: CSR row pointer of length ``n_dofs + 1``.
        col_idx: CSR column indices.
        n_dofs: Number of rows/cols (degrees of freedom).

    Returns:
        colors: ``np.ndarray`` of shape ``(n_dofs,)`` with color ids (int32).
        seeds: ``List[np.ndarray]`` of one-hot vectors (float64), one per color.
    """
    if _distance2_color_and_seeds_ext is None:
        raise ImportError(
            "tatva_color extension is not built. Run `maturin develop` in rust/coloring`."
        )

    row_ptr_arr = np.asarray(row_ptr, dtype=np.int64)
    col_idx_arr = np.asarray(col_idx, dtype=np.int64)
    return _distance2_color_and_seeds_ext(row_ptr_arr, col_idx_arr, n_dofs)


__all__ = ["distance2_color_and_seeds"]
