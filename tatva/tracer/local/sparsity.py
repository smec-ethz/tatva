"""Rank-local matrix sparsity with explicit global coordinates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.local.inputs import LocalInputPlan
from tatva.tracer.program.derivatives import tangent_pattern
from tatva.tracer.program.forms import LocalForm


def _freeze_i64(values) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64).ravel().copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class LocalMatrixPattern:
    """A local CSR pattern embedded in global row and column coordinates."""

    pattern: sps.csr_matrix
    row_global_ids: NDArray[np.int64]
    column_global_ids: NDArray[np.int64]
    global_shape: tuple[int, int]
    row_block_names: tuple[str, ...]
    column_block_names: tuple[str, ...]

    def __post_init__(self) -> None:
        pattern = self.pattern.astype(bool).tocsr(copy=True)
        pattern.sum_duplicates()
        pattern.eliminate_zeros()
        row_ids = _freeze_i64(self.row_global_ids)
        column_ids = _freeze_i64(self.column_global_ids)
        global_shape = tuple(int(size) for size in self.global_shape)

        if pattern.shape != (row_ids.size, column_ids.size):
            raise ValueError(
                f"local pattern shape {pattern.shape} does not match coordinate "
                f"sizes {(row_ids.size, column_ids.size)}"
            )
        if len(global_shape) != 2 or any(size < 0 for size in global_shape):
            raise ValueError("global matrix shape must contain two nonnegative sizes")
        if np.any((row_ids < 0) | (row_ids >= global_shape[0])):
            raise ValueError("row global IDs are outside the global matrix shape")
        if np.any((column_ids < 0) | (column_ids >= global_shape[1])):
            raise ValueError("column global IDs are outside the global matrix shape")
        if np.unique(row_ids).size != row_ids.size:
            raise ValueError("row global IDs must be unique")
        if np.unique(column_ids).size != column_ids.size:
            raise ValueError("column global IDs must be unique")

        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "row_global_ids", row_ids)
        object.__setattr__(self, "column_global_ids", column_ids)
        object.__setattr__(self, "global_shape", global_shape)

    def global_coo(self) -> sps.coo_matrix:
        """Translate local nonzeros into the global matrix coordinate space."""
        local = self.pattern.tocoo()
        return sps.coo_matrix(
            (
                local.data.copy(),
                (
                    self.row_global_ids[local.row],
                    self.column_global_ids[local.col],
                ),
            ),
            shape=self.global_shape,
        )


def trace_local_matrix_pattern(
    function: Callable,
    inputs: LocalInputPlan,
    form: LocalForm,
) -> LocalMatrixPattern:
    example = inputs.example_call(preserve_dead=True)

    pattern = tangent_pattern(
        function,
        example.args,
        example.kwargs,
        form=form.spec,
    )

    return LocalMatrixPattern(
        pattern=pattern,
        row_global_ids=form.row_global_ids,
        column_global_ids=form.column_global_ids,
        global_shape=form.global_shape,
        row_block_names=form.row_block_names,
        column_block_names=form.column_block_names,
    )
