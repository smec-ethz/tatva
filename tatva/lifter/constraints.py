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
    Hashable,
    Self,
    TypeVar,
    overload,
)
from uuid import uuid4

from jax import Array
from jax.typing import ArrayLike

from tatva.lifter.common import (
    LifterError,
    RuntimeValue,
    RuntimeValueMap,
    _iter_runtime_values,
)

if TYPE_CHECKING:
    from tatva.lifter.base import Lifter

__all__ = ["Constraint", "Periodic", "Fixed"]

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


class Periodic(Constraint):
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


class Fixed(Constraint):
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
