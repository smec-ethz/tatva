from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.core.routes import Shape


@dataclass
class DependencySet:
    csr: sps.csr_matrix
    """Boolean relation ``value scalar rows × symbolic coordinates``."""

    shape: Shape
    """Logical shape of the array this dependency set describes."""

    def __post_init__(self):
        if self.csr.shape[0] != math.prod(self.shape):
            raise ValueError(
                f"CSR matrix has {self.csr.shape[0]} rows, but shape {self.shape} "
                f"implies {math.prod(self.shape)} rows"
            )

    @classmethod
    def empty(cls, shape: Shape, n_symbols: int) -> DependencySet:
        size = int(np.prod(shape))
        dep = sps.csr_matrix((size, n_symbols), dtype=bool)
        return cls(dep, shape)

    @classmethod
    def singletons(cls, n_symbols: int) -> DependencySet:
        dep = sps.eye(n_symbols, format="csr", dtype=bool)
        return cls(dep, (n_symbols,))

    def copy(self) -> DependencySet:
        return DependencySet(self.csr.copy(), self.shape)

    def reshape(self, *ns) -> DependencySet:
        if math.prod(ns) != math.prod(self.shape):
            raise ValueError(
                f"Cannot reshape dep-set of shape {self.shape} to shape {ns}: "
                f"number of elements does not match"
            )
        return DependencySet(self.csr, ns)

    def total_union(self) -> DependencySet:
        if self.csr.shape[0] == 0:
            return DependencySet.empty((), self.csr.shape[1])
        reduced = sps.csr_matrix(self.csr.sum(axis=0).astype(bool))
        return DependencySet(reduced, ())

    def broadcast_to(
        self,
        S_out: Shape,
        broadcast_dimensions: tuple[int, ...] | Sequence[int] | None = None,
    ) -> DependencySet:
        S_in = self.shape
        if S_in == S_out:
            return self
        src_indices = np.arange(int(np.prod(S_in)))
        if broadcast_dimensions is not None:
            newshape = [1] * len(S_out)
            for i, b in enumerate(broadcast_dimensions):
                newshape[b] = S_in[i] if len(S_in) > 0 else 1
            src_indices = src_indices.reshape(newshape)
        else:
            src_indices = src_indices.reshape(S_in)
        mapped_indices = np.broadcast_to(src_indices, S_out).ravel()
        return DependencySet(self.csr[mapped_indices], S_out)


@dataclass
class InteractionGraph:
    """Accumulate structural nonlinear interactions between symbolic coordinates.

    The graph is square over the complete symbolic coordinate system.  Energy,
    weak, and mixed operators are views obtained by selecting the declared row
    and column blocks from this single graph.
    """

    n_symbols: int

    def __post_init__(self):
        if self.n_symbols < 0:
            raise ValueError("n_symbols must be nonnegative")
        self._row_chunks: list[np.ndarray] = []
        self._col_chunks: list[np.ndarray] = []

    def add_cross(self, lhs_dep: DependencySet, rhs_dep: DependencySet) -> None:
        if lhs_dep.csr.nnz == 0 or rhs_dep.csr.nnz == 0:
            return
        self._validate_width(lhs_dep)
        self._validate_width(rhs_dep)
        pattern = (lhs_dep.csr.T @ rhs_dep.csr).tocsr()
        rows, cols = pattern.nonzero()
        self._add_coords(rows, cols)
        self._add_coords(cols, rows)

    def add_self(self, dep: DependencySet) -> None:
        if dep.csr.nnz == 0:
            return
        self._validate_width(dep)
        pattern = (dep.csr.T @ dep.csr).tocsr()
        rows, cols = pattern.nonzero()
        self._add_coords(rows, cols)

    def add_paired_cross(
        self,
        lhs: DependencySet,
        lhs_rows: NDArray,
        rhs: DependencySet,
        rhs_rows: NDArray,
    ) -> None:
        lhs_rows = np.asarray(lhs_rows, dtype=np.int64).ravel()
        rhs_rows = np.asarray(rhs_rows, dtype=np.int64).ravel()
        if lhs_rows.shape != rhs_rows.shape:
            raise ValueError(
                f"lhs_rows shape {lhs_rows.shape} does not match rhs_rows shape "
                f"{rhs_rows.shape}"
            )
        if lhs_rows.size == 0:
            return
        self._validate_width(lhs)
        self._validate_width(rhs)
        lhs_selected = lhs.csr[lhs_rows]
        rhs_selected = rhs.csr[rhs_rows]
        if lhs_selected.nnz == 0 or rhs_selected.nnz == 0:
            return
        cross = (lhs_selected.T @ rhs_selected).tocsr()
        rows, cols = cross.nonzero()
        self._add_coords(rows, cols)
        self._add_coords(cols, rows)

    def add_pattern(
        self,
        pattern: sps.spmatrix,
        *,
        symmetric: bool = False,
    ) -> None:
        matrix = sps.csr_matrix(pattern)
        if matrix.shape != (self.n_symbols, self.n_symbols):
            raise ValueError(
                f"interaction pattern shape {matrix.shape} does not match "
                f"({self.n_symbols}, {self.n_symbols})"
            )
        rows, cols = matrix.nonzero()
        self._add_coords(rows, cols)
        if symmetric:
            self._add_coords(cols, rows)

    def _validate_width(self, dep: DependencySet) -> None:
        if dep.csr.shape[1] != self.n_symbols:
            raise ValueError(
                f"dependency width {dep.csr.shape[1]} does not match interaction "
                f"graph size {self.n_symbols}"
            )

    def _add_coords(self, rows: NDArray, cols: NDArray) -> None:
        if rows.size == 0:
            return
        self._row_chunks.append(np.asarray(rows, dtype=np.int64))
        self._col_chunks.append(np.asarray(cols, dtype=np.int64))

    def finalize(self) -> sps.csr_matrix:
        if not self._row_chunks:
            return sps.csr_matrix((self.n_symbols, self.n_symbols), dtype=np.int8)
        rows = np.concatenate(self._row_chunks)
        cols = np.concatenate(self._col_chunks)
        data = np.ones(rows.shape[0], dtype=np.int8)
        pattern = sps.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_symbols, self.n_symbols),
        )
        pattern.sum_duplicates()
        pattern.data[:] = 1
        return pattern


# Transitional import compatibility for primitive-rule modules.  There is one
# implementation only; old code importing HessianAccumulator gets the generic
# symbolic interaction graph.
HessianAccumulator = InteractionGraph
