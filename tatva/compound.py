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


class _field:
    """A descriptor to define fields in the State class."""

    shape: tuple[int, ...]
    default_factory: Callable | None

    def __init__(
        self,
        shape: tuple[int, ...],
        default_factory: Callable | None = None,
        slice: slice | None = None,
    ) -> None:
        self.shape = shape if len(shape) > 1 else (*shape, 1)
        self.default_factory = default_factory
        self.slice: slice = slice  # type: ignore

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


class field(_field):
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
        instance.arr = instance.arr.at[self.slice].set(arr.flatten())

    def __delete__(self, instance):
        raise AttributeError(
            f"Cannot delete field ... from {instance.__class__.__name__}."
        )


class _field_sub_of_stack(field):
    """A descriptor to define fields that are sub-fields of a stacked field in the State class."""

    def __init__(
        self,
        shape: tuple[int, ...],
        parent_field: _field,
        parent_slice: tuple[slice, ...],
    ) -> None:
        super().__init__(shape)
        self.parent_field = parent_field
        self.parent_slice = parent_slice

    @overload
    def __get__(self, instance: None, owner) -> field: ...
    @overload
    def __get__(self, instance: Compound, owner) -> Array: ...
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # get slice
        return instance.arr[self.parent_field.slice].reshape(self.parent_field.shape)[
            self.parent_slice
        ]

    def __set__(self, instance: Compound, value: Array | float | int) -> None:
        arr = jnp.asarray(value)
        instance.arr = (
            instance.arr[self.parent_field.slice]
            .reshape(self.parent_field.shape)
            .at[self.parent_slice]
            .set(arr)
        ).flatten()

    def __getitem__(self, arg) -> Array:
        # Normalize to tuple
        if not isinstance(arg, tuple):
            arg = (arg,)
        # extend with full slices
        if len(arg) < len(self.shape):
            arg = arg + (slice(None),) * (len(self.shape) - len(arg))

        # get indices in the parent field
        parent_idxs = self.parent_field.__getitem__(slice(None)).reshape(
            self.parent_field.shape
        )
        return parent_idxs[self.parent_slice].__getitem__(arg).flatten()


class _CompoundMeta(type):
    fields: tuple[tuple[str, field], ...]
    size: int

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs
    ):
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
        if kwargs.get("stack_fields") is not None:
            cls._stack_fields(**kwargs)  # type: ignore

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


class Compound(metaclass=_CompoundMeta):
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

    arr: Array
    size: int = 0

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

    @classmethod
    def _stack_fields(cls, stack_fields: tuple[str, ...], stack_axis: int = -1) -> None:
        """Reorder fields by stacking specified fields along some axis. Modifies the class
        in place!

        Args:
            stack_fields: Names of the fields to be stacked.
            stack_axis: Axis along which to stack the fields. Defaults to -1.
        """
        fields = {k: v for k, v in cls.fields}
        base_shape = fields[stack_fields[0]].shape
        stack_axis = stack_axis % len(base_shape)  # get positive axis
        base_shape = jnp.asarray(base_shape)[
            jnp.array([i for i in range(len(base_shape)) if i != stack_axis])
        ]

        stack_size = sum(
            int(jnp.prod(jnp.asarray(fields[name].shape))) // base_shape.prod()
            for name in stack_fields
        )
        stacked_shape = jnp.insert(base_shape, stack_axis, stack_size)
        size = int(jnp.prod(jnp.asarray(stacked_shape)))
        stacked_slice = slice(0, size)
        stacked_field = _field(tuple(stacked_shape), slice=stacked_slice)

        new_fields = []

        # create new fields that are sub-fields of the stacked field
        for idx, name in enumerate(stack_fields):
            new_field = _field_sub_of_stack(
                shape=fields[name].shape,
                parent_field=stacked_field,
                parent_slice=tuple(
                    slice(None)
                    if i != stack_axis
                    else slice(
                        sum(
                            int(jnp.prod(jnp.asarray(fields[n].shape)))
                            // base_shape.prod()
                            for n in stack_fields[:idx]
                        ),
                        sum(
                            int(jnp.prod(jnp.asarray(fields[n].shape)))
                            // base_shape.prod()
                            for n in stack_fields[: idx + 1]
                        ),
                    )
                    for i in range(len(stacked_shape))
                ),
            )
            # check base shape compatibility or raise error
            if not jnp.all(
                jnp.asarray(new_field.shape)[
                    jnp.array(
                        [i for i in range(len(new_field.shape)) if i != stack_axis]
                    )
                ]
                == base_shape
            ):
                raise ValueError(
                    f"Field {name} with shape {new_field.shape} is not compatible with "
                    f"base shape {base_shape} along axis {stack_axis}."
                )

            # set new field in class and add to new fields list
            setattr(cls, name, new_field)
            new_fields.append((name, new_field))

        # update other fields slices
        offset = size
        for name, field_obj in cls.fields:
            if name in stack_fields:
                continue
            n = int(jnp.prod(jnp.asarray(field_obj.shape)))
            start = offset
            end = start + n
            field_obj.slice = slice(start, end)
            offset += n
            new_fields.append((name, field_obj))

        # finally update class fields and size
        cls.fields = tuple(new_fields)

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
