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
from typing import Any, Callable, Generator, Self, TypeVar, overload

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

__all__ = ["Compound", "field", "stack_fields"]


T_Compound = TypeVar("T_Compound", bound="Compound")


class CompoundStackError(ValueError):
    pass


class _FieldBase:
    def __init__(self, shape: tuple[int, ...], _slice: slice | None = None) -> None:
        self.shape = shape
        self.slice = _slice

    def indices(self, arg: int | slice | tuple[int | slice, ...]) -> Array:
        # Normalize to tuple
        if not isinstance(arg, tuple):
            arg = (arg,)
        if len(arg) < len(self.shape):
            arg = arg + (slice(None),) * (len(self.shape) - len(arg))

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

        if self.slice is None:
            raise RuntimeError(
                "Field slice is not set. This should be set by the Compound metaclass."
            )

        return jnp.array(flat_local + self.slice.start)

    def __getitem__(self, arg) -> Array:
        return self.indices(arg)


class Field(_FieldBase):
    """A descriptor to define fields in the State class."""

    shape: tuple[int, ...]
    default_factory: Callable | None

    def __init__(
        self,
        shape: tuple[int, ...],
        default_factory: Callable | None = None,
        *,
        _slice: slice | None = None,
    ) -> None:
        super().__init__(shape, _slice)
        self.default_factory = default_factory

    def _copy_with_slice(self, _slice: slice) -> Self:
        return self.__class__(
            shape=self.shape, default_factory=self.default_factory, _slice=_slice
        )

    @overload
    def __get__(self, instance: None, owner) -> Self: ...
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


def field(shape: tuple[int, ...], default_factory: Callable | None = None) -> Field:
    """Helper function to create a FieldSpec for defining fields in the Compound class."""
    return Field(shape=shape, default_factory=default_factory)


class FieldStackedView(Field):
    """A descriptor to define fields that are sub-fields of a stacked field in the State class."""

    def __init__(
        self,
        shape: tuple[int, ...],
        default_factory: Callable | None,
        parent_field: _FieldBase,
        parent_slice: tuple[slice, ...],
    ) -> None:
        self.shape = shape
        self.default_factory = default_factory
        self.parent_field = parent_field
        self.parent_slice = parent_slice

    @overload
    def __get__(self, instance: None, owner) -> Field: ...
    @overload
    def __get__(self, instance: Compound, owner) -> Array: ...
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # get slice
        return (
            instance.arr[self.parent_field.slice]
            .reshape(self.parent_field.shape)[self.parent_slice]
            .reshape(self.shape)
        )

    def __set__(self, instance: Compound, value: Array | float | int) -> None:
        arr = jnp.asarray(value)
        # arr must be reshapable to self.shape
        # if self.shape is rank 1, then we extend it to rank 2 with a singleton dimension
        # to allow broadcasting
        shape_in_parent = self.shape if len(self.shape) > 1 else (*self.shape, 1)
        arr = jnp.reshape(arr, shape_in_parent)
        parent = (
            instance.arr[self.parent_field.slice]
            .reshape(self.parent_field.shape)
            .at[self.parent_slice]
            .set(arr)
        ).reshape(-1)
        instance.arr = instance.arr.at[self.parent_field.slice].set(parent)

    def indices(self, arg) -> Array:
        # Normalize to tuple
        if not isinstance(arg, tuple):
            arg = (arg,)
        # extend with full slices
        if len(arg) < len(self.shape):
            arg = arg + (slice(None),) * (len(self.shape) - len(arg))

        # get indices in the parent field
        parent_idxs = self.parent_field.indices(slice(None)).reshape(
            self.parent_field.shape
        )
        return parent_idxs[self.parent_slice].__getitem__(arg).flatten()

    @property
    def slice(self):
        # The slice of a FieldStackedView is not contiguous, so we cannot return a single slice.
        # Instead, we return an int array of the global indices corresponding to this
        # field, which can be used for indexing.
        return self.indices(slice(None))


class _CompoundMeta(type):
    fields: tuple[tuple[str, Field], ...]
    size: int

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs
    ):
        fields: list[tuple[str, Field]] = []
        size: int = 0

        # find all fields in the namespace and compute their slices in the flat array
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Field):
                n = int(prod(attr_value.shape))
                s = slice(size, size + n)

                # copy (reconstruct) the field with the correct slice and set it in the namespace
                f = attr_value._copy_with_slice(s)

                namespace[attr_name] = f
                fields.append((attr_name, f))
                size += n

        cls = super().__new__(mcls, name, bases, namespace)
        cls.fields = tuple(fields)
        cls.size = size

        if kwargs.get("stack_fields"):
            import warnings

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

    fields: tuple[tuple[str, Field], ...]
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
            # initialize fields with default factories if provided
            for name, field_obj in self.fields:
                if field_obj.default_factory is not None:
                    self.arr = self.arr.at[field_obj.slice].set(
                        jnp.asarray(field_obj.default_factory()).flatten()
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


def _apply_stacked_fields(
    cls: type[T_Compound], stack_fields: tuple[str, ...], stack_axis: int = -1
) -> type[T_Compound]:
    """Reorder fields by stacking specified fields along an axis. Modifies class in place.

    Args:
        cls: The Compound class to modify.
        stack_fields: The field names to stack together. Must be compatible in shape
            except along the stack_axis.
        stack_axis: The axis along which to stack the fields. Negative values are
            supported and interpreted as counting from the end of the shape. Default is -1
            (stack along the last axis).
    """
    if not stack_fields:
        raise CompoundStackError("At least one field name is required.")
    if len(set(stack_fields)) != len(stack_fields):
        raise CompoundStackError(
            "Duplicate field names are not allowed in stack_fields."
        )

    fields_map = {name: field_obj for name, field_obj in cls.fields}
    missing = [name for name in stack_fields if name not in fields_map]
    if missing:
        raise CompoundStackError(f"Unknown field names in stack_fields: {missing}")

    # the dimension in the fields to stack along must be compatible, we take the first field as reference
    # if the field is rank 1 (a scalar per node), we reshape it to rank 2 with a singleton dimension
    def _reshape_rank_1_if_needed(field: Field) -> tuple[Field, tuple[int, ...]]:
        rank = len(original_shape := field.shape)
        if rank == 0:
            raise CompoundStackError("Cannot stack scalar fields.")
        if rank == 1:
            return Field(
                shape=(*field.shape, 1),
                default_factory=field.default_factory,
                _slice=field.slice,
            ), original_shape
        return field, original_shape

    base_field, _ = _reshape_rank_1_if_needed(fields_map[stack_fields[0]])
    base_shape = base_field.shape
    rank = len(base_shape)
    if not (-rank <= stack_axis < rank):
        raise CompoundStackError(
            f"Invalid stack_axis={stack_axis} for field rank {rank}. "
            f"Expected in [{-rank}, {rank - 1}]."
        )
    axis = stack_axis % rank
    base_reduced = base_shape[:axis] + base_shape[axis + 1 :]

    # Precompute per-field layout and validate compatibility in one pass.
    # collect new fields in stack_layout as (name, shape, default_factory, start, end)
    # tuples for the stacked fields.
    stack_layout: list[tuple[str, tuple[int, ...], Callable | None, int, int]] = []
    stack_size = 0
    for name in stack_fields:
        f, original_shape = _reshape_rank_1_if_needed(fields_map[name])
        shape = f.shape
        if len(shape) != rank:
            raise CompoundStackError(
                f"Field {name} with shape {shape} is not compatible with base rank {rank}."
            )
        reduced = shape[:axis] + shape[axis + 1 :]
        if reduced != base_reduced:
            raise CompoundStackError(
                f"Field {name} with shape {shape} is not compatible with "
                f"base shape {base_shape} along axis {axis}."
            )
        extent = shape[axis]
        start = stack_size
        end = start + extent
        stack_layout.append((name, original_shape, f.default_factory, start, end))
        stack_size = end

    stacked_shape = (*base_shape[:axis], stack_size, *base_shape[axis + 1 :])
    stacked_size = int(prod(stacked_shape))
    # we put the stacked fields first in the global dof array
    stacked_field = _FieldBase(stacked_shape, _slice=slice(0, stacked_size))

    # create new FieldStackedView descriptors for the stacked fields, and prepare to set
    # them on the class after validation and layout construction is complete.
    new_fields: list[tuple[str, Field]] = []
    pending_setattrs: list[tuple[str, FieldStackedView]] = []
    for name, shape, default_factory, start, end in stack_layout:
        parent_slice = tuple(
            slice(None) if i != axis else slice(start, end) for i in range(rank)
        )
        new_field = FieldStackedView(
            shape=shape,
            default_factory=default_factory,
            parent_field=stacked_field,
            parent_slice=parent_slice,
        )
        pending_setattrs.append((name, new_field))
        new_fields.append((name, new_field))

    # now we need to reset the field descriptors for the stacked fields to point to the
    # new layout, and update the slices of all trailing fields accordingly.
    stack_name_set = set(stack_fields)
    offset = stacked_size
    trailing_updates: list[tuple[Field, slice]] = []
    trailing_fields: list[tuple[str, Field]] = []
    for name, field_obj in cls.fields:
        if name in stack_name_set:
            continue
        n = int(prod(field_obj.shape))
        new_slice = slice(offset, offset + n)
        trailing_updates.append((field_obj, new_slice))
        trailing_fields.append((name, field_obj))
        offset += n

    # Commit mutations only after all validation and layout construction passes.
    for name, new_field in pending_setattrs:
        setattr(cls, name, new_field)
    for field_obj, new_slice in trailing_updates:
        field_obj.slice = new_slice

    new_fields.extend(trailing_fields)
    cls.fields = tuple(new_fields)
    cls.size = offset

    return cls


def stack_fields(
    *fields: str, axis: int = -1
) -> Callable[[type[T_Compound]], type[T_Compound]]:
    """Class decorator to stack compatible fields into a shared contiguous layout.

    Args:
        *fields: The field names to stack together. Must be compatible in shape except
            along the stack axis.
        axis: The axis along which to stack the fields. Negative values are supported and
            interpreted as counting from the end of the shape. Default is -1.
    """

    if not fields:
        raise ValueError("At least one field name is required.")

    field_names = tuple(fields)

    def decorator(cls: type[T_Compound]) -> type[T_Compound]:
        return _apply_stacked_fields(cls, stack_fields=field_names, stack_axis=axis)

    return decorator
