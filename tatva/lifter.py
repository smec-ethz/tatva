from typing import Callable, Concatenate, ParamSpec, Protocol

import equinox
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO

P = ParamSpec("P")


class LiftedEnergyFunction(Protocol[P]):
    @staticmethod
    def __call__(
        u_reduced: Array, u_full: Array, *args: P.args, **kwargs: P.kwargs
    ) -> Array: ...


class Lifter(equinox.Module):
    free_dofs: Array
    constrained_dofs: Array
    size: int
    ndof_reduced: int

    def __init__(
        self, size: int, free: Array | None = None, constrained: Array | None = None
    ):
        """Class to handle lifting and reducing displacement vectors.

        Args:
            size: Size of the full displacement vector.
            free: Indices of free dofs. If None, all dofs are free.
            constrained: Indices of constrained dofs. If None, inferred from free dofs.
        """
        assert free is not None or constrained is not None, (
            "Either free or constrained dofs must be provided."
        )
        if free is None:
            free = jnp.setdiff1d(jnp.arange(size), constrained)  # type: ignore
        else:
            assert jnp.all(free < size) and jnp.all(free >= 0), (
                "Free dof indices out of bounds."
            )
            assert len(jnp.unique(free)) == len(free), (
                "Free dof indices must be unique."
            )

        if constrained is None:
            constrained = jnp.setdiff1d(jnp.arange(size), free)
        else:
            assert jnp.all(constrained < size) and jnp.all(constrained >= 0), (
                "Constrained dof indices out of bounds."
            )
            assert len(jnp.unique(constrained)) == len(constrained), (
                "Constrained dof indices must be unique."
            )

        self.free_dofs = free
        self.constrained_dofs = (
            constrained
            if constrained is not None
            else jnp.setdiff1d(jnp.arange(size), free)
        )
        self.size = size
        self.ndof_reduced = self.free_dofs.size

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size."""
        return u_full.at[self.free_dofs].set(u_reduced)

    def reduce(self, u_full: Array) -> Array:
        """Reduce full displacement vector to free dofs."""
        return u_full[self.free_dofs]

    def reduce_energy_function(
        self,
        energy_fn: Callable[Concatenate[Array, P], Array],
    ) -> LiftedEnergyFunction[P]:
        """Compute energy given reduced displacement vector."""

        def new_energy_fn(
            u_reduced: Array, u_full: Array, *args: P.args, **kwargs: P.kwargs
        ) -> Array:
            u_full = self.lift(u_reduced, u_full)
            return energy_fn(u_full, *args, **kwargs)

        return new_energy_fn

    def reduce_sparsity_pattern(self, pattern: BCOO) -> BCOO:
        """Reduce a sparse matrix pattern to the free dofs.

        Args:
            pattern: Sparse matrix pattern in BCOO format.

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
