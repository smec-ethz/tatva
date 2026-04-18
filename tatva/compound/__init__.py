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

from math import prod
from typing import Any, Generator, Generic, Self, TypeVar

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from tatva.compound.field import Field, field, stack_fields

__all__ = ["Compound", "field", "stack_fields"]


T_Compound = TypeVar("T_Compound", bound="Compound")


class Compound:
    """A compound array/state.

    A helper class to create a compound state with multiple fields. It simplifies packing
    and unpacking into and from a flat array. Useful to manage fields while working with a
    flat array for the solver.

    Args:
        arr: The flat data array. If None, initializes to zeros.

    Examples:

    Create a compound state with fields::

        class MyState(Compound):
            u = field(shape=(N, 3))
            phi = field(shape=(N,), default_factory=lambda: jnp.ones(N))

        state = MyState()

    Use `state.arr` to access the flat array, and `state.u`, `state.phi` to access the
    individual fields.

    You can use iterator unpacking to directly unpack the fields from the state::

        u, phi = MyState(arr)

    """

    fields: tuple[tuple[str, Field], ...] = ()
    arr: Array
    size: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__()

        # Collect fields from base classes first
        all_fields: list[tuple[str, Field]] = []
        # We find the nearest Compound base to start with its fields
        for base in cls.__mro__[1:]:
            if issubclass(base, Compound) and base is not Compound:
                all_fields.extend(base.fields)
                break

        # total size so far (from inherited fields)
        size = sum(int(prod(f.shape)) for _, f in all_fields)

        # find all NEW fields in the class namespace and compute their slices
        for attr_name, attr_value in list(cls.__dict__.items()):
            if isinstance(attr_value, Field):
                n = int(prod(attr_value.shape))
                s = slice(size, size + n)

                # copy (reconstruct) the field with the correct slice and set it on the class
                f = attr_value._copy_with_slice(s)

                setattr(cls, attr_name, f)
                all_fields.append((attr_name, f))
                size += n

        cls.fields = tuple(all_fields)
        cls.size = size

        if kwargs.get("stack_fields"):
            import warnings

            from tatva.compound.field import _apply_stacked_fields

            warnings.warn(
                "Class keyword arguments 'stack_fields'/'stack_axis' are deprecated. "
                "Use the @stack_fields(...) decorator instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _apply_stacked_fields(
                cls,  # pyright: ignore[reportArgumentType]
                stack_fields=tuple(kwargs["stack_fields"]),
                stack_axis=kwargs.get("stack_axis", -1),
            )

        # register as pytree node for JAX transformations
        register_pytree_node_class(cls)

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

    def __init__(self, arr: Array | None = None, /, **kwargs) -> None:
        """Initialize the state with given keyword arguments."""
        if arr is not None:
            assert arr.size == self.size, (
                f"Array size {arr.size} does not match expected size {self.size}."
            )
            self.arr = arr
        else:
            self.arr = jnp.zeros(self.size, dtype=float)
            # initialize fields with provided arrays OR default factories if available
            for name, field_obj in self.fields:
                if name in kwargs:
                    field_arr = kwargs[name]
                    self.arr = field_obj._set_in_array(self.arr, jnp.asarray(field_arr))
                elif field_obj.default_factory is not None:
                    self.arr = field_obj._set_in_array(
                        self.arr, jnp.asarray(field_obj.default_factory())
                    )

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Generator[Array, None, None]:
        for name, _ in self.fields:
            yield getattr(self, name)

    def __repr__(self) -> str:
        # print shape of each field in the class
        field_reprs = [
            f"{name}={getattr(type(self), name).shape}" for name, _ in self.fields
        ]
        return f"{self.__class__.__name__}({', '.join(field_reprs)})"

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.arr + other.arr)

    def at(self, name: str) -> _CompoundAtHelper[Self]:
        """Helper for functional updates to fields. Usage: `new_state =
        state.at('field_name').set(new_value)`.

        Args:
            name: The name of the field to update.
        """
        field_obj = dict(self.fields).get(name)
        if field_obj is None:
            raise AttributeError(f"Unknown field name: {name}")
        return _CompoundAtHelper(self, field_obj)

    def flatten(self) -> Array:
        """Return the flat array representation of the state. Same as `state.arr`."""
        return self.arr


class _CompoundAtHelper(Generic[T_Compound]):
    def __init__(self, state: T_Compound, field_obj: Field):
        self.state = state
        self.field_obj = field_obj

    def set(self, value: Array | float) -> T_Compound:
        return self.state.__class__(
            self.field_obj._set_in_array(self.state.arr, jnp.asarray(value))
        )
