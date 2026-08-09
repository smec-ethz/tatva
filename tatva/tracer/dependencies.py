from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps


@dataclass
class DependencySet:
    csr: sps.csr_matrix
    """scipy CSR matrix of shape (prod(shape), n_dofs) with boolean data, where each row
    corresponds to a flattened array element and each column corresponds to a DOF. A True
    value indicates that the array element depends on the corresponding DOF."""

    shape: tuple
    """logical shape of the array this dep-set corresponds to (e.g., (3,4) for a 3×4 array);"""

    @classmethod
    def empty(cls, shape: tuple, n_dofs: int) -> DependencySet:
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
        return DependencySet(self.csr, ns)

    def total_union(self) -> DependencySet:
        """OR all dependency sets in this array into a single 1D vector of shape ()."""
        if self.csr.shape[0] == 0:
            return DependencySet.empty((), self.csr.shape[1])
        reduced = sps.csr_matrix(self.csr.sum(axis=0).astype(bool))
        return DependencySet(reduced, ())

    def broadcast_to(
        self,
        S_out: tuple,
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
