from __future__ import annotations

from typing import Any, Callable, Generator, Self, overload

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


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
    arr: Array

    size: int = 0

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize the subclass and register its fields."""
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    def __init__(self, arr: Array | None = None) -> None:
        """Initialize the state with given keyword arguments."""
        if arr is not None:
            assert arr.size == self.size, (
                f"Array size {arr.size} does not match expected size {self.size}."
            )
            self.arr = arr
        else:
            self.arr = jnp.zeros(self.size, dtype=float)

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

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.arr + other.arr)

    def pack(self) -> Array:
        """Pack the state into a single array. Flattened and concatenated."""
        return self.arr

    @classmethod
    def unpack(cls, packed: Array) -> Self:
        """Unpack the state from a single flattened packed array."""
        return cls(packed)


class field:
    """A descriptor to define fields in the State class."""

    shape: tuple[int, ...]
    default_factory: Callable | None
    idx: _CompoundIdx

    def __init__(
        self, shape: tuple[int, ...], default_factory: Callable | None = None
    ) -> None:
        self.shape = shape
        self.default_factory = default_factory
        self.idx = None  # pyright: ignore

    def __set_name__(self, owner: Compound, name: str) -> None:
        # compute slice indices
        start = owner._fields[-1][2] if owner._fields else 0
        size = int(jnp.prod(jnp.asarray(self.shape)))
        end = start + size

        owner._fields += ((name, self, end),)
        owner.size += size

        self.slice = slice(start, end)
        self.idx = _CompoundIdx(start, self.shape)

    @overload
    def __get__(self, instance: None, owner) -> field: ...
    @overload
    def __get__(self, instance: Compound, owner) -> Array: ...
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # get slice
        return instance.arr[self.slice].reshape(self.shape)

    def __set__(self, instance: Compound, value: Array | float | int) -> None:
        arr = jnp.asarray(value)
        instance.arr = instance.arr.at[self.slice].set(arr)

    def __delete__(self, instance):
        raise AttributeError(
            f"Cannot delete field ... from {instance.__class__.__name__}."
        )


class _CompoundIdx:
    """A helper for flat indexing into the compound array."""

    def __init__(self, slice_start: int, shape: tuple[int, ...]) -> None:
        self.slice_start = slice_start
        self.shape = shape

    def __getitem__(self, arg) -> Array:
        # Normalize to tuple
        if not isinstance(arg, tuple):
            arg = (arg,)
        # extend with full slices
        if len(arg) < len(self.shape):
            arg = arg + (slice(None),) * (len(self.shape) - len(arg))
        # build index arrays
        idxs = []
        for dim, sub in enumerate(arg):
            if isinstance(sub, slice):
                start, stop, step = sub.indices(self.shape[dim])
                idxs.append(jnp.arange(start, stop, step))
            elif isinstance(sub, (int, jnp.integer)):
                idxs.append(jnp.array([sub]))
            else:
                idxs.append(jnp.asarray(sub))
        # meshgrid
        mesh = jnp.meshgrid(*idxs, indexing="ij")
        # flatten
        multi_idx = [m.flatten() for m in mesh]
        flat_local = jnp.ravel_multi_index(multi_idx, dims=self.shape)
        # shift and return
        return jnp.array(flat_local + self.slice_start)
