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

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Self,
    TypeVar,
    overload,
)
from uuid import uuid4

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from tatva.lifter.common import (
    LifterError,
    RuntimeValue,
    RuntimeValueMap,
    _iter_runtime_values,
)

if TYPE_CHECKING:
    from mpi4py import MPI

    from tatva.lifter.base import Lifter
    from tatva.mpi import _LocalLayout

__all__ = ["Constraint", "Fixed", "Periodic"]

T = TypeVar("T")
T_ArrayLike = TypeVar("T_ArrayLike", bound=ArrayLike)


class Constraint:
    """Base class for conditions applied during lifting.

    Subclasses define how constrained degrees of freedom (dofs) are enforced
    when mapping a reduced vector back to the full vector.
    """

    dofs: Array
    """The dofs to constrain; every constraint must specify which dofs it applies to."""

    _runtime_specs: tuple[RuntimeValue[Any], ...]
    """Tuple of all RuntimeValue attributes in this constraint instance, collected at
    init. Used for resolving runtime values when the constraint is bound to a lifter."""

    _lifter: Lifter | None
    """Reference to the lifter this constraint is bound to; set when the constraint is
    bound to a lifter, used for resolving runtime values."""

    _constraint_id: Hashable
    """Unique id for this constraint instance that survives `._bind(lifter)`, used for
    hashing and equality; assigned at init."""

    def __init__(self, dofs: Array):
        self.dofs = dofs
        self._lifter = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs) -> None:
            # Call the original __init__ to set up the instance, then find all
            # RuntimeValue attributes and store them in _runtime_specs and
            # _runtime_values.
            orig_init(self, *args, **kwargs)
            self._runtime_specs = tuple(_iter_runtime_values(self))

            # assign a unique id for this constraint instance, used for hashing and equality
            self._constraint_id = uuid4()

        cls.__init__ = wrapped_init  # ty:ignore[invalid-assignment]

    def __eq__(self, other) -> bool:
        """Check equality based on class and dofs; runtime values are not considered."""
        return type(self) is type(other) and self._constraint_id == other._constraint_id

    def __hash__(self):
        # hashing is required for using lifters and constraints as static args in jax
        # transformations.
        # We hash based on the class and a unique id for this constraint instance, which
        # allows us to treat constraints as unique even if they are bound to a lifter and
        # have potentially different runtime values.
        return hash((type(self), self._constraint_id))

    def augment_sparsity(self, sparsity: sps.csr_matrix) -> sps.csr_matrix:
        """Augment the sparsity pattern to account for this constraint.

        Args:
            sparsity: Sparsity pattern in SciPy CSR format.

        Returns:
            Augmented sparsity pattern in SciPy CSR format.
        """
        return sparsity

    def apply_lift(self, u_full: Array) -> Array:
        """Apply the constraint to a full vector and return the modified copy."""
        return u_full

    def apply_transpose(self, r_full: Array) -> Array:
        """Apply the transpose of the constraint to a full residual vector."""
        return r_full

    def _get_runtime_specs(self) -> tuple[RuntimeValue, ...]:
        """Return a tuple of all RuntimeValue attributes in this constraint instance."""
        return self._runtime_specs

    def _resolve_indices(self, layout: _LocalLayout) -> Self:
        """Resolve local indices for this constraint against the given layout.
        Base implementation returns self. Override in subclasses that need resolution
        against a specific MPI layout."""
        return self

    def _bind(self, lifter: Lifter) -> Self:
        """Return a shallow bound copy of this constraint for the given lifter."""
        bound = self.__class__.__new__(self.__class__)
        bound.__dict__ = dict(self.__dict__)
        bound._lifter = lifter
        return bound

    @overload
    def _resolve_runtime(
        self,
        obj: RuntimeValue[T_ArrayLike],
        runtime_values: RuntimeValueMap | None = None,
    ) -> T_ArrayLike: ...

    @overload
    def _resolve_runtime(
        self, obj: T_ArrayLike, runtime_values: RuntimeValueMap | None = None
    ) -> T_ArrayLike: ...

    def _resolve_runtime(self, obj, runtime_values: RuntimeValueMap | None = None):
        """Recursively resolve any RuntimeValue attributes in obj using runtime_values."""
        if runtime_values is None:
            if self._lifter is None:
                raise LifterError("Constraint is not bound to a lifter")
            runtime_values = self._lifter._runtime_values

        if isinstance(
            obj, (Array, np.ndarray, np.bool_, np.number, bool, int, float, complex)
        ):
            return obj
        elif isinstance(obj, RuntimeValue):
            return obj.get_value(runtime_values)
        elif isinstance(obj, Mapping):
            return type(obj)(
                (k, self._resolve_runtime(v, runtime_values)) for k, v in obj.items()
            )
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return type(obj)(self._resolve_runtime(o, runtime_values) for o in obj)
        else:
            return obj


class Periodic(Constraint):
    dofs: Array
    """The dofs to constrain; these will be set equal to the corresponding ``master_dofs`` during lifting."""
    master_dofs: Array
    """The master dofs that the constrained ``dofs`` will be set equal to during lifting."""

    def __init__(self, dofs: Array, master_dofs: Array):
        super().__init__(dofs)
        self.master_dofs = master_dofs

    def augment_sparsity(self, sparsity: sps.csr_matrix) -> sps.csr_matrix:
        n_full = sparsity.shape[0]
        dofs = np.asarray(self.dofs, dtype=np.int64)
        master_dofs = np.asarray(self.master_dofs, dtype=np.int64)

        # M is the operator such that u_full = M @ u_reduced_with_slaves_zeroed
        # M_ij = 1 if i == j OR (i is slave and j is its master)
        rows = np.concatenate([np.arange(n_full), dofs])
        cols = np.concatenate([np.arange(n_full), master_dofs])
        data = np.ones(rows.shape[0], dtype=np.int8)
        M = sps.csr_matrix((data, (rows, cols)), shape=(n_full, n_full))

        # Triple product to propagate connectivity: S_aug = M.T @ S @ M
        # This makes master rows/cols inherit from slave rows/cols.
        S_aug = M.T @ (sparsity.astype(np.int8) @ M)

        # Ensure it's a binary pattern (data=1)
        S_aug.data = np.ones_like(S_aug.data, dtype=np.int8)
        return S_aug.tocsr()

    def apply_lift(self, u_full: Array) -> Array:
        """Copy values from ``master_dofs`` into the constrained ``dofs``."""
        return u_full.at[self.dofs].set(u_full[self.master_dofs])

    def apply_transpose(self, r_full: Array) -> Array:
        """Add residuals from constrained ``dofs`` to ``master_dofs``."""
        return r_full.at[self.master_dofs].add(r_full[self.dofs]).at[self.dofs].set(0.0)


def create_g2l(l2g: NDArray) -> Callable[[NDArray], NDArray]:
    # this method may not be optimal
    # it works for now
    sort_idx = np.argsort(l2g)
    sorted_l2g = l2g[sort_idx]

    def lookup(global_indices: NDArray) -> NDArray:
        pos = np.searchsorted(sorted_l2g, global_indices)
        valid = (pos < len(sorted_l2g)) & (
            sorted_l2g[np.minimum(pos, len(sorted_l2g) - 1)] == global_indices
        )
        local_indices = np.full_like(global_indices, -1)
        local_indices[valid] = sort_idx[pos[valid]]
        return local_indices

    return lookup


class PeriodicMPI(Periodic):
    """Periodicity based on nodal mapping, for use in MPI parallel computations where dofs
    on different processes may be constrained together.

    This is a global constraint that is synchronized across all processes. It requests the
    global dof indices of the slaves and masters.
    You can access these with `CompoundCls._g.field_name[global_node, slice, ...]`.
    """

    def __init__(
        self,
        dofs: Array,
        master_dofs: Array,
        layout: _LocalLayout,
        *,
        comm: MPI.Comm | None = None,
    ):
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD

        self._comm = comm

        dofs_np = np.asarray(dofs, dtype=np.int32)
        master_dofs_np = np.asarray(master_dofs, dtype=np.int32)

        # find which of 'dofs' are in our local layout
        lookup = create_g2l(layout.natural_l2g)
        local_idx = lookup(dofs_np)

        valid = local_idx >= 0

        # store natural global indices for slaves we have locally and their corresponding masters
        self._slave_natural_g = dofs_np[valid]
        self._master_natural_g = master_dofs_np[valid]

        # extra ghosts: masters of local slaves that are NOT in our current layout
        master_local = lookup(self._master_natural_g)
        self._extra_ghost_dofs = self._master_natural_g[master_local < 0]

        # initialize as Periodic with local slaves; masters will be resolved later
        super().__init__(
            jnp.asarray(local_idx[valid]),
            jnp.zeros(np.sum(valid), dtype=jnp.int32),
        )

    def _resolve_indices(self, layout: _LocalLayout) -> Self:
        lookup = create_g2l(layout.natural_l2g)
        local_slaves = lookup(self._slave_natural_g)
        local_masters = lookup(self._master_natural_g)

        if np.any(local_slaves < 0) or np.any(local_masters < 0):
            raise LifterError(
                f"PeriodicMPI rank {self._comm.Get_rank()}: Failed to resolve local "
                "indices for periodic DOFs. Ensure all required masters were added "
                "as ghosts to the layout."
            )

        # Create a new resolved instance
        resolved = self.__class__.__new__(self.__class__)
        resolved.__dict__ = dict(self.__dict__)
        resolved.dofs = jnp.asarray(local_slaves)
        resolved.master_dofs = jnp.asarray(local_masters)
        return resolved


class Fixed(Constraint):
    dofs: Array
    """The dofs to constrain; these will be set to fixed values during lifting."""
    values: ArrayLike | RuntimeValue[ArrayLike]
    """Values to set on the constrained ``dofs`` during lifting; Defaults to 0.0 if not
    provided at init or runtime."""

    def __init__(
        self,
        dofs: Array,
        values: ArrayLike | RuntimeValue[ArrayLike] | None = None,
    ):
        """Initialize a Dirichlet boundary condition.

        Args:
            dofs: The dofs to constrain; these will be set to fixed values during lifting.
            values: Fixed values to set on the constrained ``dofs`` during lifting; Can be
            an instance of ``RuntimeValue``.
        """
        self.dofs = dofs
        self.values = values if values is not None else 0.0

    def apply_lift(self, u_full: Array) -> Array:
        """Set constrained ``dofs`` to ``values``."""
        return u_full.at[self.dofs].set(self._resolve_runtime(self.values))

    def apply_transpose(self, r_full: Array) -> Array:
        """Zero out residuals on fixed ``dofs``."""
        return r_full.at[self.dofs].set(0.0)
