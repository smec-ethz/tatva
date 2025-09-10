from typing import Callable, Concatenate, ParamSpec, Protocol, overload

import equinox
import jax.numpy as jnp
from jax import Array

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

    def __init__(
        self, size: int, free: Array | None = None, constrained: Array | None = None
    ):
        """Class to handle lifting and reducing displacement vectors.

        Args:
            size (int): Size of the full displacement vector.
            free (Array | None): Indices of free dofs. If None, all dofs are free.
            constrained (Array | None): Indices of constrained dofs. If None, inferred from free dofs.
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

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size."""
        return u_full.at[self.free_dofs].set(u_reduced)

    def reduce(self, u_full: Array) -> Array:
        """Reduce full displacement vector to free dofs."""
        return u_full[self.free_dofs]

    def lifted_energy_function(
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
