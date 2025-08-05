from __future__ import annotations

from typing import Callable, Generator, Self, overload

import jax.numpy as jnp
from jax import Array


class Compound:
    """A compound array/state.

    A helper class to create a compound state with multiple fields. It simplifies packing
    and unpacking into and from a flat array. Useful to manage fields while working with a
    flat array for the solver.

    Args:
        **kwargs (Optional): Keyword arguments to initialize the fields of the state.

    Examples:

    Create a compound state with fields::

        class MyState(Compound):
            u = field(shape=(N, 3))
            phi = field(shape=(N,), default_factory=lambda: jnp.ones(N))

        state = MyState()

    Use `state.pack()` to flatten the state into a single array, and
    `state.unpack(packed_array)` to restore the state from a packed array::

        u_flat = state.pack()
        packed_state = MyState.unpack(u_flat)

    You can use iterator unpacking to directly unpack the fields from the state::

        u, phi = MyState.unpack(u_flat)

    """

    _fields: tuple[tuple[str, field, int], ...] = ()
    _splits_flattened_array: tuple[int, ...] = ()
    _data: Array

    size: int = 0

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize the subclass and register its fields."""
        super().__init_subclass__(**kwargs)

    def __init__(self, arr: Array | None = None) -> None:
        """Initialize the state with given keyword arguments."""
        if arr is not None:
            assert arr.size == self.size, (
                f"Array size {arr.size} does not match expected size {self.size}."
            )
            self._data = arr
        else:
            self._data = jnp.zeros(self.size, dtype=float)

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Generator[Array, None, None]:
        for name, _, _ in self._fields:
            yield getattr(self, name)

    def __repr__(self) -> str:
        # print shape of each field in the class
        field_reprs = [
            f"{name}={getattr(type(self), name).shape}" for name, _, _ in self._fields
        ]
        return f"{self.__class__.__name__}({', '.join(field_reprs)})"

    def pack(self) -> Array:
        """Pack the state into a single array. Flattened and concatenated."""
        return self._data

    @classmethod
    def unpack(cls, packed: Array) -> Self:
        """Unpack the state from a single flattened packed array."""
        return cls(packed)


class field:
    """A descriptor to define fields in the State class."""

    def __init__(
        self, shape: tuple[int, ...], default_factory: Callable | None = None
    ) -> None:
        self.shape = shape
        self.default_factory = default_factory

    def __set_name__(self, owner: Compound, name: str) -> None:
        # runs at class creation time to register the field
        self.public_name = name
        self.private_name = f"_{name}"

        previous_end_idx = owner._fields[-1][2] if owner._fields else 0
        size = int(jnp.prod(jnp.array(self.shape)))

        owner._fields += ((name, self, previous_end_idx + size),)
        owner.size += size

        self.slice = slice(previous_end_idx, previous_end_idx + size)

    @overload
    def __get__(self, instance: None, owner) -> field: ...
    @overload
    def __get__(self, instance: Compound, owner) -> Array: ...
    def __get__(self, instance, owner):
        # If instance is None, we are accessing the class attribute, not an instance
        # attribute. Hence, return the descriptor itself.
        if instance is None:
            return self  # pyright: ignore[reportUnreachable]

        return instance._data[self.slice].reshape(self.shape)

    def __set__(self, instance: Compound, value: Array | float | int) -> None:
        value = jnp.asarray(value)
        instance._data = instance._data.at[self.slice].set(value)

    def __delete__(self, instance):
        raise AttributeError(
            f"Cannot delete field ... from {instance.__class__.__name__}."
        )
