# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Literal,
    ParamSpec,
    Self,
)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike
from numpy.typing import NDArray

from tatva.lifter.common import (
    LifterError,
    RuntimeValueMap,
    _runtime_value_map_is_equal,
)
from tatva.lifter.constraints import Constraint

if TYPE_CHECKING:
    from mpi4py import MPI

    from tatva.mpi import _LocalLayout

__all__ = ["Lifter"]

P = ParamSpec("P")


def _compute_mask(size: int, constraints: tuple[Constraint, ...]) -> NDArray[np.bool]:
    """Compute free/constrained dofs and reduced size."""
    if not constraints:
        # base case: no constraints
        return np.ones(size, dtype=bool)

    free = np.ones(size, dtype=bool)

    for cond in constraints:
        free[cond.dofs] = False

    return free


@dataclass(frozen=True)
class LiftStrategy(ABC):
    """Base class for lifter variants. This is used to define the interface for lifter
    variants, which are used to implement different lifting strategies (e.g., scatter,
    binary search)."""

    indices: Array
    """Unsigned integer array of indices that map from the reduced vector to the full
    vector. This can be either the free dofs (for scatter) or the constrained dofs (for
    binary search)."""

    size: int = field(metadata={"static": True})
    """Total number of dofs in the full vector."""

    @property
    @abstractmethod
    def free_dofs(self) -> Array:
        """Array of free dofs as integer indices (not constrained)."""

    @property
    @abstractmethod
    def constrained_dofs(self) -> Array:
        """Array of constrained dofs as integer indices (not free)."""

    @abstractmethod
    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size.

        Args:
            u_reduced: Vector on free dofs (length ``size_reduced``).
            u_full: Base full vector to modify; typically previous solution.

        Returns:
            Full vector with free dofs set to ``u_reduced`` and constraints
            applied (Dirichlet, periodic, etc.).
        """

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return u_full[self.free_dofs]


@register_dataclass
@dataclass(frozen=True)
class ScatterStrategy(LiftStrategy):
    @property
    def free_dofs(self) -> Array:
        return self.indices

    @property
    def constrained_dofs(self) -> Array:
        mask = jnp.ones(self.size, dtype=bool).at[self.indices].set(False)
        return jnp.nonzero(mask)[0].astype(jnp.uint32)

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        return u_full.at[self.indices].set(u_reduced)


@register_dataclass
@dataclass(frozen=True)
class BinarySearchStrategy(LiftStrategy):
    @property
    def constrained_dofs(self) -> Array:
        return self.indices

    @property
    def free_dofs(self) -> Array:
        mask = jnp.ones(self.size, dtype=bool).at[self.indices].set(False)
        return jnp.nonzero(mask)[0].astype(jnp.uint32)

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        grid = jnp.arange(self.size, dtype=jnp.uint32)
        # binary search: counts how many constrained DOFs are <= grid index
        shift = jnp.searchsorted(self.indices, grid, side="left")
        # check if the current index is a constrained dof
        # idx_to_check = jnp.maximum(0, shift - 1)
        # is_constrained = (shift > 0) & (self.indices[idx_to_check] == grid)
        # is_constrained = jnp.isin(grid, self.indices)
        is_constrained = self.indices[shift] == grid

        return jnp.where(is_constrained, u_full, u_reduced[grid - shift])

    def reduce(self, u_full: Array) -> Array:
        # reduces u_full by removing the constrained dofs directly
        # avoiding allocating a full free_dofs index array
        return jnp.delete(u_full, self.indices)


@register_dataclass
@dataclass(frozen=True)
class Lifter:
    size: int = field(metadata={"static": True})
    """Total number of dofs in the full vector."""

    size_reduced: int = field(metadata={"static": True})
    """Number of dofs in the reduced vector (free dofs only)."""

    constraints: tuple[Constraint, ...] = field(metadata={"static": True})
    """Tuple of constraints, which are applied in order during lifting. Constraints must
    specify which dofs they apply to, and can optionally specify runtime values that are
    provided to the lifter at runtime. These constraints are bound to the lifter instance,
    which allows them to access the lifter's runtime values when applying the lift."""

    _runtime_values: RuntimeValueMap
    """Mapping of runtime values for dynamic constraints; keys are RuntimeValue keys."""

    _nb_extra_ghost_dofs: int = field(metadata={"static": True})
    """For MPI only: extra ghost dofs that are added to the local layout to account for
    constraints like PeriodicMPI. lifter.lift(...) will not include them in the lifted
    full vector."""

    strategy: LiftStrategy

    def __post_init__(self):
        # ensure that all constraints are bound to this lifter instance
        object.__setattr__(
            self,
            "constraints",
            tuple(cond._bind(self) for cond in self.constraints),
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # check that property OR field name for "free_dofs" and "constrained_dofs" is already defined in subclass
        if not any(hasattr(cls, name) for name in ("free_dofs", "constrained_dofs")):
            raise TypeError(
                f"Subclass {cls.__name__} must define either a property or field for "
                "'free_dofs' and 'constrained_dofs'."
            )

    @classmethod
    def make(
        cls,
        size: int,
        /,
        *constraints: Constraint,
        strategy: Literal["scatter", "binary_search"] = "scatter",
    ) -> Lifter:
        """Create a lifter that maps between reduced and full vectors.

        Args:
            size: Total number of dofs in the full vector.
            *constraints: Extra constraints (e.g., periodic maps).
            type: Type of lifter to create. "scatter" uses a scatter operation to lift the
                reduced vector to the full vector. "binary_search" uses a binary search to
                find the free dofs in the full vector. The binary search variant is faster
                when the number of constrained dofs is small compared to the total number of
                dofs, and when the architecture is a GPU.

        Examples::

            lifter = Lifter(
                6,
                Fixed(jnp.array([0, 5]), 0.0),
                Periodic(dofs=jnp.array([2]), master_dofs=jnp.array([1])),
            )
            u_reduced = jnp.array([10.0, 20.0, 30.0])
            u_full = lifter.lift_from_zeros(u_reduced)
            # u_full -> [0., 10., 10., 20., 30., 0.]
            u_reduced_back = lifter.reduce(u_full)

        """
        # collect all runtime specs from the constraints and store their keys and default
        # values for easy access during lifting.
        runtime_specs = tuple(
            spec for cond in constraints for spec in cond._get_runtime_specs()
        )
        runtime_values = {spec.key: spec.default for spec in runtime_specs}

        mask_free = _compute_mask(size, constraints)
        size_reduced = int(jnp.sum(mask_free))
        if strategy == "scatter":
            free_dofs = jnp.nonzero(mask_free)[0].astype(jnp.uint32)
            space_recreator = ScatterStrategy(free_dofs, size)
        elif strategy == "binary_search":
            constrained_dofs = jnp.nonzero(~mask_free)[0].astype(jnp.uint32)
            space_recreator = BinarySearchStrategy(constrained_dofs, size)
        else:
            raise ValueError(f"Unknown lifter type: {strategy}")

        return Lifter(
            size=size,
            size_reduced=size_reduced,
            constraints=tuple(constraints),
            _runtime_values=runtime_values,
            _nb_extra_ghost_dofs=0,
            strategy=space_recreator,
        )

    @property
    def free_dofs(self) -> Array:
        """Array of free dofs as integer indices (not constrained)."""
        return self.strategy.free_dofs

    @property
    def constrained_dofs(self) -> Array:
        """Array of constrained dofs as integer indices (not free)."""
        return self.strategy.constrained_dofs

    @property
    def at(self) -> RuntimeValueIndexer:
        """Return a ValueIndexer for setting runtime values by key."""
        return RuntimeValueIndexer(self)

    def __hash__(self):
        return hash((self.size, self.constraints, self._nb_extra_ghost_dofs))

    def __eq__(self, other) -> bool:
        """Check equality based on size, constraints, and runtime values. If a lifter is a
        ``static_arg`` in a jax transformation, the runtime values must be equal for
        the lifter to be considered equal.
        """
        return (
            isinstance(other, Self.__class__)
            and self.size == other.size
            and self.constraints == other.constraints
            and _runtime_value_map_is_equal(self._runtime_values, other._runtime_values)
            and self._nb_extra_ghost_dofs == other._nb_extra_ghost_dofs
        )

    @property
    def _local_size(self) -> int:
        """The size of the local vector from the problem definition, which does not
        include any extra ghost dofs added by constraints like PeriodicMPI. This is the
        size that lifter.lift(...) will return, and the size that lifter.reduce(...)
        expects as input."""
        return self.size - self._nb_extra_ghost_dofs

    def add(self, condition: Constraint) -> Self:
        """Return a new lifter with ``condition`` appended to constraints."""
        new_constraints = self.constraints + (condition._bind(self),)
        return self._replace(constraints=new_constraints)

    def with_values(self, updates: RuntimeValueMap) -> Self:
        """Update the internal runtime values mapping with the given updates."""
        for key in updates:
            if key not in self._runtime_values:
                raise LifterError(
                    f"There is no runtime value with key={key} in the lifter's constraints"
                )
        return self._replace(_runtime_values=self._runtime_values | updates)

    def _replace(self, **updates) -> Self:
        return replace(self, **updates)

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size.

        Args:
            u_reduced: Vector on free dofs (length ``size_reduced``).
            u_full: Base full vector to modify; typically previous solution.

        Returns:
            Full vector with free dofs set to ``u_reduced`` and constraints
            applied (Dirichlet, periodic, etc.).
        """
        u_full = self.strategy.lift(u_reduced, u_full)

        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)

        return u_full[: self._local_size]  # in case constraints added extra dofs

    def lift_from_zeros(
        self,
        u_reduced: Array,
    ) -> Array:
        """Lift reduced vector to a full vector starting from zeros."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return self.strategy.reduce(u_full)

    def reduce_adjoint(self, r_full: Array) -> Array:
        """Extract the reduced dual (residual/gradient) vector from ``r_full``.

        This applies the adjoint of each constraint in reverse order to ensure that
        contributions from constrained dofs are correctly mapped back to the reduced
        space (e.g., summing residuals for periodic dofs).

        Args:
            r_full: Full dual vector (e.g., residual or gradient).

        Returns:
            Reduced dual vector on free dofs.
        """
        for condition in reversed(self.constraints):
            r_full = condition.apply_transpose(r_full)

        return self.strategy.reduce(r_full)

    def reduce_sparsity(self, sparsity: sps.csr_matrix) -> sps.csr_matrix:
        """Reduce the sparsity pattern to the free DOF layout.

        Args:
            sparsity: Full sparsity pattern in SciPy CSR format.

        Returns:
            Reduced sparsity pattern in SciPy CSR format.
        """
        free_dofs = self.strategy.free_dofs
        return sparsity[free_dofs][:, free_dofs]

    def adapt_sparsity(self, sparsity: sps.csr_matrix) -> sps.csr_matrix:
        """Augment and reduce the sparsity pattern to account for all constraints. From
        the built-in constraints, only Periodic and PeriodicMPI have non-trivial
        implementations of this method, which add entries to the sparsity pattern to
        account for the coupling between master and slave dofs.
        Returns the reduced sparsity pattern for the free DOFs.

        Args:
            sparsity: Sparsity pattern in SciPy CSR format.

        Returns:
            Augmented and reduced sparsity pattern in SciPy CSR format.
        """
        augmented = self.augment_sparsity(sparsity)
        reduced = self.reduce_sparsity(augmented)
        return reduced

    def augment_sparsity(self, sparsity: sps.csr_matrix) -> sps.csr_matrix:
        """Augment the sparsity pattern to account for all constraints.

        Args:
            sparsity: Sparsity pattern in SciPy CSR format.

        Returns:
            Augmented sparsity pattern in SciPy CSR format.
        """
        if sparsity.shape[0] < self.size:
            n_orig = sparsity.shape[0]
            extra_ptr = np.full(
                self.size - n_orig, sparsity.indptr[-1], dtype=sparsity.indptr.dtype
            )
            new_indptr = np.concatenate([sparsity.indptr, extra_ptr])
            sparsity = sps.csr_matrix(
                (sparsity.data, sparsity.indices, new_indptr),
                shape=(self.size, self.size),
            )

        for cond in self.constraints:
            sparsity = cond.augment_sparsity(sparsity)
        return sparsity

    def adapt_layout(
        self, layout: _LocalLayout, comm: MPI.Comm
    ) -> tuple[_LocalLayout, Self]:
        """MPI only. Augment the local layout to account for all constraints.

        This may add extra ghost dofs to the local array to account for periodic
        constraints, for example.

        Args:
            layout: Local layout to augment.

        Returns:
            A tuple of (augmented_reduced_layout, augmented_lifter).
        """
        from mpi4py import MPI

        from tatva.mpi import _create_dof_layout, _LocalLayout

        _dtype = np.int32

        # gather all extra ghost dofs from constraints that need to be included
        extra_ghosts = []
        for cond in self.constraints:
            if hasattr(cond, "_extra_ghost_dofs"):
                extra_ghosts.append(cond._extra_ghost_dofs)

        if extra_ghosts:
            # unique extra ghosts for this rank
            local_extra_g = np.unique(np.concatenate(extra_ghosts))

            # re-create the layout with these extra ghosts.
            # natural global indices stay within system bounds, so n_global is unchanged.
            new_natural_l2g = np.concatenate([layout.natural_l2g, local_extra_g])
            new_owned_mask = np.concatenate(
                [layout.owned_mask, np.zeros(len(local_extra_g), dtype=bool)]
            )

            full_layout = _create_dof_layout(
                new_natural_l2g, new_owned_mask, layout.n_global, comm
            )
        else:
            full_layout = layout

        # resolve lifter against the (potentially augmented) full layout.
        # This ensures constraints like PeriodicMPI have correct local indices
        # for masters.
        new_constraints = tuple(
            cond._resolve_indices(full_layout) for cond in self.constraints
        )
        lifter = self._replace(
            constraints=new_constraints,
            _nb_extra_ghost_dofs=len(local_extra_g) if extra_ghosts else 0,
        )

        # 3. Reduce the full layout to the free-DOF layout
        mask_free = lifter.strategy.free_dofs
        mask_free_owned = full_layout.owned_mask[mask_free]
        local_to_global_free = full_layout.local_to_global[mask_free]

        n_owned = int(np.sum(mask_free_owned))
        n_global = comm.allreduce(n_owned, op=MPI.SUM)

        # New contiguous global indexing for free DOFs
        n_per_rank = comm.allgather(n_owned)
        offset = np.cumsum([0] + n_per_rank[:-1])[comm.rank]

        l2g = np.full(self.size_reduced, -1, dtype=_dtype)
        l2g[mask_free_owned] = offset + np.arange(n_owned, dtype=_dtype)

        # Resolve ghosts using full_layout.l2g as the unique global identifier
        local_directory = np.full(full_layout.n_global, -1, dtype=_dtype)
        # The global ID of the owned free DOFs in the ALL-DOF layout
        local_directory[local_to_global_free[mask_free_owned]] = l2g[mask_free_owned]

        global_directory = np.empty_like(local_directory)
        comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

        l2g[~mask_free_owned] = global_directory[local_to_global_free[~mask_free_owned]]

        reduced_layout = _LocalLayout(
            local_to_global=l2g,
            offset=offset,
            n_owned=n_owned,
            n_total=self.size_reduced,
            n_global=n_global,
            owned_mask=mask_free_owned,
            natural_l2g=local_to_global_free,
        )

        return reduced_layout, lifter


class RuntimeValueSetter:
    """Set runtime values on a lifter by position. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter, key: str):
        self.lifter = lifter
        self.key = key

    def set(self, value: ArrayLike) -> Lifter:
        """Set the runtime value for this key and return a new lifter with the updated value."""
        return self.lifter.with_values({self.key: value})


class RuntimeValueIndexer:
    """Set runtime values on a lifter by key. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter):
        self.lifter = lifter

    def __getitem__(self, key) -> RuntimeValueSetter:
        return RuntimeValueSetter(self.lifter, key)

    def __call__(self, key: str) -> RuntimeValueSetter:
        return RuntimeValueSetter(self.lifter, key)
