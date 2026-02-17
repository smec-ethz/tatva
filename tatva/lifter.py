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

import copy
from typing import (
    Any,
    Generic,
    Hashable,
    MutableMapping,
    Self,
    TypeVar,
    overload,
)

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class LifterError(ValueError):
    """Error raised when there is a problem with the lifter, e.g., missing runtime values."""


T = TypeVar("T", bound=ArrayLike)


class RuntimeValue(Generic[T]):
    """Descriptor for a constraint value that must be provided at runtime."""

    def __init__(self, key: Hashable | None = None):
        self.key = key
        self._name: str | None = None

    def __set_name__(self, owner: type[Constraint], name: str):
        self._name = name
        if self.key is None:
            # default key is (constraint class, attribute name)
            self.key = (owner, name)

    @overload
    def __get__(self, instance: None, owner: type[Constraint]) -> RuntimeValue[T]: ...
    @overload
    def __get__(self, instance: Constraint, owner: type[Constraint]) -> T: ...
    def __get__(self, instance: Constraint, owner: type[Constraint]) -> T:
        if instance is None:
            return self
        if self.key not in instance._runtime_values:
            raise LifterError(
                f"Dynamic value for {self.key} not set on instance {instance}"
            )
        return instance._runtime_values[self.key]

    def __set__(self, instance: Constraint, value: T):
        instance._runtime_values[self.key] = value


class Constraint:
    """Base class for conditions applied during lifting.

    Subclasses define how constrained degrees of freedom (dofs) are enforced
    when mapping a reduced vector back to the full vector.
    """

    _dynamic: bool = False
    """Whether this constraint is dynamic and requires runtime values during lifting."""

    dofs: Array
    """The dofs to constrain; every constraint must specify which dofs it applies to."""

    _runtime_values: MutableMapping[Hashable, Any]
    """Mapping of keys to dynamic attributes that require runtime values during lifting."""

    def __init__(self, dofs: Array):
        self.dofs = dofs
        self._runtime_values = {}

    def __hash__(self):
        # hashing is required for using lifters and constraints as static args in jax
        # transformations.
        #
        # This is the cheapest possible hash implementation, but it means that two
        # constraints with the same parameters will not be considered equal. If we want to
        # have value-based equality and hashing, we have to implement __eq__ and __hash__,
        # e.g., by using dataclasses or manually comparing the relevant attributes.
        return id(self)

    def apply_lift(self, u_full: Array) -> Array:  # override in subclasses
        """Apply the constraint to a full vector and return the modified copy."""
        return u_full

    @classmethod
    def _get_runtime_specs(cls) -> tuple[RuntimeValue, ...]:
        """Return a tuple of all RuntimeValue attributes in this constraint."""
        out: list[RuntimeValue] = []
        for c in cls.__mro__:
            out.extend(
                obj for obj in c.__dict__.values() if isinstance(obj, RuntimeValue)
            )
        return tuple(out)

    def _semi_shallow_copy(self) -> Self:
        """Return a copy of this constraint with a separate _dynamic_attrs dict."""
        new = copy.copy(self)
        new._runtime_values = dict(self._runtime_values)
        return new

    def _is_set(self, attr: RuntimeValue[Any]) -> bool:
        """Check if a RuntimeValue attribute has been set on this instance."""
        return attr.key in self._runtime_values


class PeriodicMap(Constraint):
    dofs: Array
    """The dofs to constrain; these will be set equal to the corresponding ``master_dofs`` during lifting."""
    master_dofs: Array
    """The master dofs that the constrained ``dofs`` will be set equal to during lifting."""

    def __init__(self, dofs: Array, master_dofs: Array):
        super().__init__(dofs)
        self.master_dofs = master_dofs

    def apply_lift(self, u_full: Array) -> Array:
        """Copy values from ``master_dofs`` into the constrained ``dofs``."""
        return u_full.at[self.dofs].set(u_full[self.master_dofs])


class DirichletBC(Constraint):
    dofs: Array
    """The dofs to constrain; these will be set to fixed values during lifting."""
    values: RuntimeValue[ArrayLike] = RuntimeValue()
    """Values to set on the constrained ``dofs`` during lifting; if None, these
    values must be provided at runtime via ``Lifter.lift(..., dynamic_values=...)``"""

    def __init__(
        self,
        dofs: Array,
        values: ArrayLike | None = None,
    ):
        """Initialize a Dirichlet boundary condition.

        Args:
            dofs: The dofs to constrain; these will be set to fixed values during lifting.
            values: Fixed values to set on the constrained ``dofs`` during lifting; if None, these
                values must be provided at runtime via ``Lifter.set(*values)``.
        """
        super().__init__(dofs)
        if values is not None:
            self.values = values

    def apply_lift(self, u_full: Array) -> Array:
        """Set constrained ``dofs`` to ``values``."""
        return u_full.at[self.dofs].set(self.values)


class Lifter:
    """Create a lifter that maps between reduced and full vectors.

    Args:
        size: Total number of dofs in the full vector.
        *constraints: Extra constraints (e.g., periodic maps).
        **kwargs: Ignored; kept for compatibility with equinox.Module init.

    Examples::

        lifter = Lifter(
            6,
            DirichletBC(jnp.array([0, 5]), 0.0),
            PeriodicMap(dofs=jnp.array([2]), master_dofs=jnp.array([1])),
        )
        u_reduced = jnp.array([10.0, 20.0, 30.0])
        u_full = lifter.lift_from_zeros(u_reduced)
        # u_full -> [0., 10., 10., 20., 30., 0.]
        u_reduced_back = lifter.reduce(u_full)

    """

    free_dofs: Array
    """Array of free dofs as integer indices (not constrained)."""

    constrained_dofs: Array
    """Array of constrained dofs as integer indices."""

    size: int
    """Total number of dofs in the full vector."""

    size_reduced: int
    """Number of dofs in the reduced vector (free dofs only)."""

    constraints: tuple[Constraint, ...] = ()
    """Tuple of additional constraints (e.g., periodic maps)."""

    def __init__(
        self,
        size: int,
        /,
        *constraints: Constraint,
        **kwargs,
    ):
        self.size = size
        self.constraints = constraints
        self._runtime_pairs = tuple(
            (cond, attr)
            for cond in constraints
            for attr in cond._get_runtime_specs()
            if attr.key not in cond._runtime_values  # not fixed -> dynamic at runtime
        )
        self._compute_sizes()

    def __hash__(self):
        return hash((self.size, self.constraints))

    def add(self, condition: Constraint) -> Self:
        """Return a new lifter with ``condition`` appended to constraints."""
        return self.__class__(self.size, *self.constraints, condition)

    def set(self, *values: ArrayLike) -> Self:
        """Set the dynamic values for this lifter's constraints and return a new lifter with those values.

        Args:
            *values: Values to set on the dynamic attributes of this lifter's constraints;
                the order of values must match the order of the lifter's constraints with
                dynamic attributes (i.e., the order of ``lifter._dynamic_pairs``).

        Returns:
            a new lifter with dynamic constraints set to provided values
        """
        if len(values) != len(self._runtime_pairs):
            raise LifterError(
                f"Expected {len(self._runtime_pairs)} dynamic values, but got {len(values)}"
            )

        # create a new lifter with the same constraints but separate _dynamic_attrs dicts
        new_constraints = tuple(c._semi_shallow_copy() for c in self.constraints)

        # recompute dynamic pairs for the new constraints
        new_dynamic_pairs = self._compute_runtime_pairs(new_constraints)

        # set the dynamic values on the new constraints
        for (cond, attr), val in zip(new_dynamic_pairs, values):
            attr.__set__(cond, val)

        # return new lifter with updated constraints/pairs; reuse dof arrays/sizes
        return self._replace(
            constraints=new_constraints, _dynamic_pairs=new_dynamic_pairs
        )

    def lift(
        self,
        u_reduced: Array,
        u_full: Array,
    ) -> Array:
        """Lift reduced displacement vector to full size.

        Args:
            u_reduced: Vector on free dofs (length ``size_reduced``).
            u_full: Base full vector to modify; typically previous solution.

        Returns:
            Full vector with free dofs set to ``u_reduced`` and constraints
            applied (Dirichlet, periodic, etc.).
        """
        u_full = u_full.at[self.free_dofs].set(u_reduced)

        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)

        return u_full

    def lift_from_zeros(self, u_reduced: Array) -> Array:
        """Lift reduced vector to a full vector starting from zeros."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return u_full[self.free_dofs]

    def _replace(self, **updates) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {**self.__dict__, **updates}
        return new

    def _compute_sizes(self):
        """Compute free/constrained dofs and reduced size."""
        all_dofs = jnp.arange(self.size)

        if not self.constraints:
            # base case: no constraints
            self.free_dofs = all_dofs
            self.constrained_dofs = jnp.array([], dtype=jnp.int32)
            self.size_reduced = self.size
            return

        constrained = jnp.concatenate([cond.dofs for cond in self.constraints])
        constrained = jnp.unique(constrained)
        free = jnp.setdiff1d(all_dofs, constrained, assume_unique=True)

        self.free_dofs = free
        self.constrained_dofs = constrained
        self.size_reduced = free.size

    @staticmethod
    def _compute_runtime_pairs(
        constraints: tuple[Constraint, ...],
    ) -> tuple[tuple[Constraint, RuntimeValue], ...]:
        return tuple(
            (c, a)
            for c in constraints
            for a in c._get_runtime_specs()
            if a.key not in c._runtime_values
        )
