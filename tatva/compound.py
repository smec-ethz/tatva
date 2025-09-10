# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
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

from typing import Any, Callable, Generator, Self, overload

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


class field:
    """A descriptor to define fields in the State class."""

    shape: tuple[int, ...]
    default_factory: Callable | None

    def __init__(
        self, shape: tuple[int, ...], default_factory: Callable | None = None
    ) -> None:
        self.shape = shape if len(shape) > 1 else (*shape, 1)
        self.default_factory = default_factory
        self.slice: slice = None  # type: ignore

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

        mesh = jnp.meshgrid(*idxs, indexing="ij")
        multi_idx = [m.flatten() for m in mesh]
        flat_local = jnp.ravel_multi_index(multi_idx, dims=self.shape)

        return jnp.array(flat_local + self.slice.start)


class CompoundMeta(type):
    fields: tuple[tuple[str, field], ...]
    size: int

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        fields: list[tuple[str, field]] = []
        size: int = 0

        # find all fields in the namespace and compute their slices in the flat array
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, field):
                n = int(jnp.prod(jnp.asarray(attr_value.shape)))
                # get slice indices
                start = size
                end = start + n
                attr_value.slice = slice(start, end)

                fields.append((attr_name, attr_value))
                size += n

        cls = super().__new__(mcls, name, bases, namespace)
        cls.fields = tuple(fields)
        cls.size = size

        # register as pytree node for JAX transformations
        register_pytree_node_class(cls)
        return cls

    def __getitem__(cls, arg) -> Array:
        node, *dofs = arg if isinstance(arg, tuple) else (arg,)
        nodal_vals = jnp.hstack(
            [f[node].reshape(-1, *f.shape[1:]) for _, f in cls.fields],
            dtype=int,
        )
        return (nodal_vals[:, *dofs] if dofs else nodal_vals).flatten()


class Compound(metaclass=CompoundMeta):
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

    arr: Array
    size: int = 0

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

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
