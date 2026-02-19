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
    Any,
    Generator,
    Generic,
    Hashable,
    Self,
    Sequence,
    Set,
    TypeAlias,
    TypeVar,
)
from uuid import uuid4

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

__all__ = [
    "Lifter",
    "Constraint",
    "DirichletBC",
    "PeriodicMap",
    "RuntimeValue",
    "LifterError",
]


class LifterError(ValueError):
    """Error raised when there is a problem with the lifter, e.g., missing runtime values."""


RuntimeValueMap: TypeAlias = dict[Hashable, Any]
PyTree: TypeAlias = Any

T = TypeVar("T", bound=ArrayLike)
V = TypeVar("V")


class RuntimeValue(Generic[T]):
    """Descriptor for a constraint value that must be provided at runtime.

    Args:
        key: A hashable key to identify this runtime value; used for setting values on the
            lifter and for error messages when the value is missing at runtime.
        default: An optional default value to use. If not provided, the lifter will raise
            an error if this value is not set at runtime.
    """

    def __init__(self, key: Hashable | None = None, default: T | None = None):
        self.key = key or uuid4()
        self.default = default

    def get_value(self, runtime_values: RuntimeValueMap) -> T:
        """Get the value of this attribute from the given lifter instance."""
        if self.key not in runtime_values:
            raise LifterError(f"Runtime value for (key={self.key}) not set on lifter")
        return runtime_values[self.key]


class Constraint:
    """Base class for conditions applied during lifting.

    Subclasses define how constrained degrees of freedom (dofs) are enforced
    when mapping a reduced vector back to the full vector.
    """

    dofs: Array
    """The dofs to constrain; every constraint must specify which dofs it applies to."""

    _runtime_specs: tuple[RuntimeValue[Any], ...]
    _lifter: Lifter | None

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

        cls.__init__ = wrapped_init  # ty:ignore[invalid-assignment]

    def __hash__(self):
        # hashing is required for using lifters and constraints as static args in jax
        # transformations.
        #
        # This is the cheapest possible hash implementation, but it means that two
        # constraints with the same parameters will not be considered equal. If we want to
        # have value-based equality and hashing, we have to implement __eq__ and __hash__,
        # e.g., by using dataclasses or manually comparing the relevant attributes.
        return id(self)

    def apply_lift(self, u_full: Array) -> Array:
        """Apply the constraint to a full vector and return the modified copy."""
        return u_full

    def _get_runtime_specs(self) -> tuple[RuntimeValue, ...]:
        """Return a tuple of all RuntimeValue attributes in this constraint instance."""
        return self._runtime_specs

    def _bind(self, lifter: Lifter) -> Self:
        """Return a shallow bound copy of this constraint for the given lifter."""
        bound = self.__class__.__new__(self.__class__)
        bound.__dict__ = dict(self.__dict__)
        bound._lifter = lifter
        return bound

    def _resolve_runtime(
        self, obj: V, runtime_values: RuntimeValueMap | None = None
    ) -> V:
        """Recursively resolve any RuntimeValue attributes in obj using runtime_values."""
        if runtime_values is None:
            if self._lifter is None:
                raise LifterError("Constraint is not bound to a lifter")
            runtime_values = self._lifter._runtime_values

        if isinstance(obj, ArrayLike):
            return obj
        elif isinstance(obj, RuntimeValue):
            return obj.get_value(runtime_values)  # ty:ignore[invalid-return-type]
        elif isinstance(obj, Mapping):
            return type(obj)(
                (k, self._resolve_runtime(v, runtime_values)) for k, v in obj.items()
            )
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return type(obj)(self._resolve_runtime(o, runtime_values) for o in obj)
        else:
            return obj


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
    values: ArrayLike | RuntimeValue[ArrayLike] | None = None
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


class RuntimeValueSetter:
    """Set runtime values on a lifter by position. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter, key: Hashable):
        self.lifter = lifter
        self.key = key

    def set(self, *value: Any) -> Lifter:
        """Set the runtime value for this key and return a new lifter with the updated value."""
        if isinstance(self.key, tuple):
            assert isinstance(value, tuple) and len(value) == len(self.key), (
                "Expected a tuple of values with the same length as the key tuple"
            )
            updates = {k: v for k, v in zip(self.key, value)}
        else:
            updates = {self.key: value}

        return self.lifter._update_runtime_values(updates)


class RuntimeValueIndexer:
    """Set runtime values on a lifter by key. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter):
        self.lifter = lifter

    def __getitem__(self, key: Hashable) -> Any:
        return RuntimeValueSetter(self.lifter, key)


@register_pytree_node_class
class Lifter:
    """Create a lifter that maps between reduced and full vectors.

    Args:
        size: Total number of dofs in the full vector.
        *constraints: Extra constraints (e.g., periodic maps).

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
    """Tuple of constraints, which are applied in order during lifting. Constraints must
    specify which dofs they apply to, and can optionally specify runtime values that are
    provided to the lifter at runtime. These constraints are bound to the lifter instance,
    which allows them to access the lifter's runtime values when applying the lift."""

    _runtime_values: RuntimeValueMap
    """Mapping of runtime values for dynamic constraints; keys are RuntimeValue keys."""

    def tree_flatten(self) -> tuple[tuple[PyTree, ...], tuple[Any, ...]]:
        # We want to be able to use lifters as static args in jax transformations, but they contain
        # runtime values that are not hashable. By treating the runtime values as non-static
        # children, we can keep the lifter itself as a static arg while still allowing the
        # runtime values to be passed in and used during lifting.
        children = (self._runtime_values,)
        aux_data = (
            self.size,
            self.constraints,
            self.free_dofs,
            self.constrained_dofs,
            self.size_reduced,
            self._runtime_keys,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[PyTree, ...]
    ) -> Self:
        # We create a new instance without calling __init__, since we already have all the
        # necessary data and don't want to recompute anything.
        # This is a bit hacky, but it allows us to reconstruct the lifter with the correct
        # runtime values without having to go through the normal initialization process,
        # which would require us to recompute the sizes and runtime pairs.
        size, constraints, free_dofs, constrained_dofs, size_reduced, _runtime_keys = (
            aux_data
        )
        (_runtime_values,) = children

        lifter = cls.__new__(cls)
        lifter.__dict__ = {
            "size": size,
            "constraints": tuple(c._bind(lifter) for c in constraints),
            "free_dofs": free_dofs,
            "constrained_dofs": constrained_dofs,
            "size_reduced": size_reduced,
            "_runtime_keys": _runtime_keys,
        }
        lifter._runtime_values = _runtime_values
        return lifter

    @property
    def at(self) -> RuntimeValueIndexer:
        """Return a ValueIndexer for setting runtime values by key."""
        return RuntimeValueIndexer(self)

    def __init__(
        self,
        size: int,
        /,
        *constraints: Constraint,
    ):
        self.size = size
        self.constraints = tuple(cond._bind(self) for cond in constraints)

        # collect all runtime specs from the constraints and store their keys and default
        # values for easy access during lifting.
        runtime_specs = tuple(
            spec for cond in self.constraints for spec in cond._get_runtime_specs()
        )
        self._runtime_keys = tuple(spec.key for spec in runtime_specs)
        self._runtime_values = {
            spec.key: spec.default for spec in runtime_specs if spec.default is not None
        }

        # compute free/constrained dofs and reduced size based on the constraints; this is
        # only based on the dofs specified in the constraints, not on any runtime values,
        # so we can compute it once at init and not have to worry about it changing at
        # runtime.
        self.free_dofs, self.constrained_dofs, self.size_reduced = self._compute_sizes()

    def __hash__(self):
        return hash((self.size, self.constraints))

    def __eq__(self, other) -> bool:
        """Check equality based on size and constraints; runtime values are not considered."""
        return (
            isinstance(other, Lifter)
            and self.size == other.size
            and self.constraints == other.constraints
        )

    def add(self, condition: Constraint) -> Self:
        """Return a new lifter with ``condition`` appended to constraints."""
        return self.__class__(self.size, *self.constraints, condition)

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

    def lift_from_zeros(
        self,
        u_reduced: Array,
    ) -> Array:
        """Lift reduced vector to a full vector starting from zeros."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return u_full[self.free_dofs]

    def _replace(self, **updates) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {**self.__dict__, **updates}
        new.constraints = tuple(cond._bind(new) for cond in new.constraints)
        return new

    def _update_runtime_values(self, updates: RuntimeValueMap) -> Self:
        """Update the internal runtime values mapping with the given updates."""
        for key in updates:
            if key not in self._runtime_keys:
                raise LifterError(
                    f"There is no runtime value with key={key} in the lifter's constraints"
                )
        return self._replace(_runtime_values=self._runtime_values | updates)

    def _compute_sizes(self) -> tuple[Array, Array, int]:
        """Compute free/constrained dofs and reduced size."""
        all_dofs = jnp.arange(self.size)

        if not self.constraints:
            # base case: no constraints
            return all_dofs, jnp.array([], dtype=jnp.int32), self.size

        constrained = jnp.concatenate([cond.dofs for cond in self.constraints])
        constrained = jnp.unique(constrained)
        free = jnp.setdiff1d(all_dofs, constrained, assume_unique=True)
        return free, constrained, free.size


def _iter_runtime_values(
    obj, seen: set[int] | None = None
) -> Generator[RuntimeValue[Any], None, None]:
    if seen is None:
        seen = set()

    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    if isinstance(obj, RuntimeValue):
        yield obj
        return

    # dict / mapping
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_runtime_values(v, seen)
        return

    # list / tuple / set (but not str/bytes)
    if isinstance(obj, (Sequence, Set)) and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        for v in obj:
            yield from _iter_runtime_values(v, seen)
        return

    # normal Python objects
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            yield from _iter_runtime_values(v, seen)
        return

    # optional: __slots__
    if hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            if hasattr(obj, name):
                yield from _iter_runtime_values(getattr(obj, name), seen)
