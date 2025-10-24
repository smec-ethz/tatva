from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

import equinox
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO


class Constraint(Hashable):
    """Abstract base class for conditions applied during lifting."""

    dofs: Array
    """The constrained dofs"""

    def apply_lift(self, u_full: Array) -> Array:  # override in subclasses
        return u_full


@dataclass
class PeriodicMap(Constraint):
    dofs: Array
    master_dofs: Array

    def apply_lift(self, u_full: Array) -> Array:
        return u_full.at[self.dofs].set(u_full[self.master_dofs])


@dataclass
class DirichletBC(Constraint):
    dofs: Array
    values: Array

    def apply_lift(self, u_full: Array) -> Array:
        return u_full.at[self.dofs].set(self.values)


class Lifter(equinox.Module):
    free_dofs: Array
    constrained_dofs: Array
    size: int
    size_reduced: int
    constraints: tuple[Constraint, ...] = ()

    def __init__(
        self,
        size: int,
        dirichlet_dofs: Array,
        *additional_constraints: Constraint,
        **kwargs,
    ):
        self.size = size
        self.constraints = additional_constraints

        self._compute_sizes(dirichlet_dofs)

    def __hash__(self):
        return hash(self.constraints)

    def _compute_sizes(self, dirichlet_dofs: Array):
        all_dofs = jnp.arange(self.size)
        constrained = jnp.concatenate(
            [dirichlet_dofs] + [cond.dofs for cond in self.constraints]
        )
        constrained = jnp.unique(constrained)
        free = jnp.setdiff1d(all_dofs, constrained, assume_unique=True)

        self.free_dofs = free
        self.constrained_dofs = constrained
        self.size_reduced = free.size

    def add(self, condition: Constraint) -> Self:
        """Add a condition to the lifter."""
        return equinox.tree_at(
            lambda lf: lf.constraints, self, self.constraints + (condition,)
        )

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size."""
        assert u_reduced.shape[0] == self.size_reduced, (
            f"Reduced displacement vector has incorrect size: "
            f"expected {self.size_reduced}, got {u_reduced.shape[0]}"
        )
        u_full = u_full.at[self.free_dofs].set(u_reduced)
        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)
        return u_full

    def lift_from_null(self, u_reduced: Array) -> Array:
        """Lift reduced displacement vector to full size from zero."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Reduce full displacement vector to free dofs."""
        return u_full[self.free_dofs]

    def reduce_sparsity_pattern(self, pattern: BCOO) -> BCOO:
        """Reduce a sparse matrix pattern to free dofs.

        Args:
            pattern (BCOO): Sparse matrix pattern in BCOO format.

        Returns:
            BCOO: Reduced sparse matrix pattern in BCOO format.
        """
        # Pull to host (avoid device OOM for big masks)
        I = np.asarray(pattern.indices[:, 0])
        J = np.asarray(pattern.indices[:, 1])
        D = np.asarray(pattern.data)

        n_full = int(self.size)
        free = np.asarray(self.free_dofs, dtype=np.int64)

        # Membership mask: O(n_full) setup, O(nnz) index
        is_free = np.zeros(n_full, dtype=bool)
        is_free[free] = True
        mask = is_free[I] & is_free[J]

        I = I[mask]
        J = J[mask]
        D = D[mask]

        # Full -> reduced reindex
        index_map = -np.ones(n_full, dtype=np.int64)
        index_map[free] = np.arange(free.size, dtype=np.int64)
        I_red = index_map[I]
        J_red = index_map[J]

        # Deduplicate (sum data; for pure pattern set to 1)
        keys = I_red * free.size + J_red
        uniq, inv = np.unique(keys, return_inverse=True)
        # accumulate
        D_red = np.bincount(inv, weights=D, minlength=uniq.size)
        I_red = (uniq // free.size).astype(np.int32)
        J_red = (uniq % free.size).astype(np.int32)

        # Back to JAX
        indices_red = jnp.stack([jnp.asarray(I_red), jnp.asarray(J_red)], axis=1)
        data_red = jnp.asarray(D_red)
        shape = (free.size, free.size)

        return BCOO((data_red, indices_red), shape=shape)
