from __future__ import annotations

from typing import Callable, Generator, Self

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

    _fields: tuple[tuple[str, field], ...] = ()
    _splits_flattened_array: tuple[int, ...]

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize the subclass and register its fields."""
        super().__init_subclass__(**kwargs)

        splits = ()
        offset = 0
        for _, field in cls._fields:
            size = int(jnp.prod(jnp.array(field.shape)))
            offset += size
            splits += (offset,)

        cls._splits_flattened_array = splits

    def __init__(self, **kwargs) -> None:
        """Initialize the state with given keyword arguments."""
        for name, field in self._fields:
            if name in kwargs:
                value = kwargs[name]
                if isinstance(value, Array):
                    setattr(self, name, value)
                else:
                    setattr(self, name, jnp.full(field.shape, value))
            else:
                # Use default factory if provided
                if field.default_factory is not None:
                    setattr(self, name, field.default_factory())
                else:
                    setattr(self, name, jnp.zeros(field.shape, dtype=float))

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Generator[Array, None, None]:
        for name, _ in self._fields:
            yield getattr(self, name)

    def pack(self) -> Array:
        """Pack the state into a single array. Flattened and concatenated."""
        return jnp.concatenate(
            [getattr(self, field).reshape(-1) for field, _ in self._fields]
        )

    @classmethod
    def unpack(cls, packed) -> Self:
        """Unpack the state from a single flattened packed array."""
        splits = jnp.split(packed, cls._splits_flattened_array[:-1])
        kwargs = {
            name: value.reshape(field.shape)
            for (name, field), value in zip(cls._fields, splits)
        }
        return cls(**kwargs)


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

        owner._fields += ((name, self),)

    def __get__(self, instance, owner):
        return getattr(instance, self.private_name, None)

    def __set__(self, instance, value):
        if isinstance(value, Array):
            assert value.shape == self.shape, (
                f"Value shape {value.shape} does not match field shape {self.shape}."
            )
            setattr(instance, self.private_name, value)
        else:
            setattr(
                instance, self.private_name, jnp.full(self.shape, value, dtype=float)
            )

    def __delete__(self, instance):
        raise AttributeError(
            f"Cannot delete field ... from {instance.__class__.__name__}."
        )
