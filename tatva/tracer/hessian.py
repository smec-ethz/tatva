from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.dependencies import DependencySet


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
        P = (lhs_dep.csr.T @ rhs_dep.csr).tocsr()
        r, c = P.nonzero()
        self._add_coords(r, c)

    def add_self(self, dep: DependencySet) -> None:
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
