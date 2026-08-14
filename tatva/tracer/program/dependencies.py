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
    """scipy CSR matrix of shape (prod(shape), n_dofs) with boolean data, where each row
    corresponds to a flattened array element and each column corresponds to a DOF. A True
    value indicates that the array element depends on the corresponding DOF."""

    shape: Shape
    """logical shape of the array this dep-set corresponds to (e.g., (3,4) for a 3×4 array);"""

    def __post_init__(self):
        if self.csr.shape[0] != math.prod(self.shape):
            raise ValueError(
                f"CSR matrix has {self.csr.shape[0]} rows, but shape {self.shape} "
                f"implies {math.prod(self.shape)} rows"
            )

    @classmethod
    def empty(cls, shape: Shape, n_dofs: int) -> DependencySet:
        """Create a zero-dependency SparseDepSet of shape (*shape, n_dofs)."""
        size = int(np.prod(shape))
        dep = sps.csr_matrix((size, n_dofs), dtype=bool)
        return cls(dep, shape)

    @classmethod
    def singletons(cls, n_dofs: int) -> DependencySet:
        """Create an identity-seeded SparseDepSet of shape (n_dofs,) where element i
        depends only on DOF i."""
        dep = sps.eye(n_dofs, format="csr", dtype=bool)
        return cls(dep, (n_dofs,))

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
        """OR all dependency sets in this array into a single 1D vector of shape ()."""
        if self.csr.shape[0] == 0:
            return DependencySet.empty((), self.csr.shape[1])
        reduced = sps.csr_matrix(self.csr.sum(axis=0).astype(bool))
        return DependencySet(reduced, ())

    def broadcast_to(
        self,
        S_out: Shape,
        broadcast_dimensions: tuple[int, ...] | Sequence[int] | None = None,
    ) -> DependencySet:
        """Broadcast this dep-array to a new logical shape S_out."""
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
class HessianAccumulator:
    """Accumulates Hessian coupling pairs as numpy-array chunks; fingerprints dep matrices
    to skip redundant recordings."""

    n_dofs: int
    trial_test_split: int | None = None

    def __post_init__(self):
        self._row_chunks: list[np.ndarray] = []
        self._col_chunks: list[np.ndarray] = []
        # we may do that if it becomes a bottleneck
        # self._seen_fingerprints: set[int] = set()

    def add_cross(self, lhs_dep: DependencySet, rhs_dep: DependencySet) -> None:
        # does that need special handling for trial/test split? Not really, because the
        # cross product is already the right block, and the self-blocks are not included
        # in the cross product. I don't know yet tbh
        if lhs_dep.csr.nnz == 0 or rhs_dep.csr.nnz == 0:
            return

        P = (lhs_dep.csr.T @ rhs_dep.csr).tocsr()
        r, c = P.nonzero()
        self._add_coords(r, c)
        self._add_coords(c, r)  # TODO: is it needed?

    def add_self(self, dep: DependencySet) -> None:
        if dep.csr.nnz == 0:
            return

        csr = dep.csr

        if self.trial_test_split is not None:
            # Only trial<->test cross couplings survive, so compute just that block
            # (trial_part.T @ test_part) instead of the full dep.T @ dep over all
            # columns followed by masking — avoids the discarded self-blocks.
            s = self.trial_test_split
            trial_part = csr[:, :s]
            test_part = csr[:, s:]
            cross = (trial_part.T @ test_part).tocsr()
            r, c = cross.nonzero()
            c = c + s
            self._add_coords(r, c)
            self._add_coords(c, r)
        else:
            P = (csr.T @ csr).tocsr()
            r, c = P.nonzero()
            self._add_coords(r, c)

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

        lhs_selected = lhs.csr[lhs_rows]
        rhs_selected = rhs.csr[rhs_rows]

        if lhs_selected.nnz == 0 or rhs_selected.nnz == 0:
            return

        cross = (lhs_selected.T @ rhs_selected).tocsr()
        rows, cols = cross.nonzero()
        self._add_coords(rows, cols)
        self._add_coords(cols, rows)

    def _add_coords(self, rows: NDArray, cols: NDArray) -> None:
        """Append a chunk of coordinate pairs."""
        if rows.size == 0:
            return
        self._row_chunks.append(np.asarray(rows))
        self._col_chunks.append(np.asarray(cols))

    def finalize(self) -> sps.csr_matrix:
        """Build the final binary CSR sparsity pattern from the accumulated chunks."""
        if not self._row_chunks:
            return sps.csr_matrix((self.n_dofs, self.n_dofs), dtype=np.int8)
        rows = np.concatenate(self._row_chunks)
        cols = np.concatenate(self._col_chunks)
        data = np.ones(rows.shape[0], dtype=np.int8)
        pat = sps.csr_matrix((data, (rows, cols)), shape=(self.n_dofs, self.n_dofs))
        pat.sum_duplicates()
        pat.data[:] = 1
        return pat
