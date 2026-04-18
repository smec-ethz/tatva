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
from typing import TYPE_CHECKING, Callable, Self, TypeVar, overload

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from tatva.compound import Compound

T_Compound = TypeVar("T_Compound", bound="Compound")


class CompoundStackError(ValueError):
    pass


def field(shape: tuple[int, ...], default_factory: Callable | None = None) -> Field:
    """Helper function to create a FieldSpec for defining fields in the Compound class."""
    return Field(shape=shape, default_factory=default_factory)


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


def _normalize_index(
    arg: int | slice | Array | tuple[int | slice | Array, ...],
    shape: tuple[int, ...],
) -> tuple[int | slice | Array, ...]:
    if not isinstance(arg, tuple):
        arg = (arg,)
    if len(arg) < len(shape):
        arg = arg + (slice(None),) * (len(shape) - len(arg))
    return arg


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides_rev: list[int] = []
    for extent in reversed(shape):
        strides_rev.append(stride)
        stride *= extent
    return tuple(reversed(strides_rev))


def _reshape_affine_metadata(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if shape == target_shape:
        return shape, strides

    compact_shape = tuple(extent for extent in shape if extent != 1)
    if compact_shape != target_shape:
        raise ValueError(
            f"Cannot reshape affine metadata from shape {shape} to {target_shape}."
        )

    compact_strides = tuple(
        stride for extent, stride in zip(shape, strides, strict=True) if extent != 1
    )
    return compact_shape, compact_strides


class _FieldBase:
    shape: tuple[int, ...]
    size: int
    _slice: slice | None

    def __init__(
        self,
        shape: tuple[int, ...],
        _slice: slice | None = None,
    ) -> None:
        self.shape = shape
        self.size = int(prod(shape))
        self._set_slice_and_strides(_slice)

    def _set_slice_and_strides(self, _slice: slice | None) -> None:
        self._slice = _slice
        if _slice is None:
            self._base_offset = None
            self._strides = ()
            return
        self._base_offset = _slice.start
        self._strides = _row_major_strides(self.shape)

    def _view(self, arr: Array) -> Array:
        if self._slice is None:
            raise RuntimeError(
                "Field slice is not set. This should be set during Compound initialization."
            )
        return arr[self._slice].reshape(self.shape)

    def _set_in_array(self, arr: Array, value: Array) -> Array:
        if self._slice is None:
            raise RuntimeError(
                "Field slice is not set. This should be set during Compound initialization."
            )
        return arr.at[self._slice].set(value.reshape(-1))

    def _indices_impl(self, arg: tuple[int | slice | Array, ...]) -> Array:
        if self._base_offset is None:
            raise RuntimeError(
                "Field indexing metadata is not set. This should be set during Compound initialization."
            )

        axis_indices: list[Array] = []
        for sub, extent in zip(arg, self.shape, strict=True):
            if isinstance(sub, int):
                idx = sub if sub >= 0 else sub + extent
                axis_indices.append(jnp.asarray([idx], dtype=int))
            elif isinstance(sub, slice):
                start, stop, step = sub.indices(extent)
                axis_indices.append(jnp.arange(start, stop, step, dtype=int))
            else:
                values = jnp.asarray(sub, dtype=int).reshape(-1)
                axis_indices.append(jnp.where(values < 0, values + extent, values))

        # no axes means this is a scalar field, return the base offset as the only index
        # this is only reached if the field shape is (), and arg is an empty tuple too
        if not axis_indices:
            return jnp.asarray([self._base_offset], dtype=int)

        grid_shape = tuple(len(values) for values in axis_indices)
        index = jnp.full(grid_shape, self._base_offset, dtype=int)
        for axis, (stride, values) in enumerate(
            zip(self._strides, axis_indices, strict=True)
        ):
            broadcast_shape = tuple(
                len(values) if i == axis else 1 for i in range(len(axis_indices))
            )
            index = index + jnp.reshape(
                values * stride,
                broadcast_shape,
            )
        return index.reshape(-1)

    def indices(
        self, arg: int | slice | Array | tuple[int | slice | Array, ...]
    ) -> Array:
        return self._indices_impl(_normalize_index(arg, self.shape))

    def __getitem__(self, arg) -> Array:
        return self.indices(arg)


class Field(_FieldBase):
    """A descriptor to define fields in the State class."""

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
        return self._view(instance.arr)


class FieldStackedView(Field):
    """A descriptor to define fields that are sub-fields of a stacked field in the State class."""

    _root_slice: slice
    _root_shape: tuple[int, ...]
    _view_slice: tuple[slice | int, ...]

    def __init__(
        self,
        shape: tuple[int, ...],
        default_factory: Callable | None,
        parent_field: _FieldBase,
        parent_slice: tuple[slice, ...],
    ) -> None:
        if parent_field._slice is None:
            raise RuntimeError(
                "Parent field slice is not set. This should be set during Compound initialization."
            )

        _FieldBase.__init__(self, shape=shape)
        self.default_factory = default_factory

        self._root_slice = parent_field._slice
        self._root_shape = parent_field.shape
        self._view_slice = parent_slice

        if parent_field._base_offset is None:
            raise RuntimeError("Parent field indexing metadata is not set.")
        base_offset = parent_field._base_offset
        view_shape = list(parent_field.shape)
        strides = list(parent_field._strides)

        for axis, sub in enumerate(parent_slice):
            if isinstance(sub, int):
                idx = sub % parent_field.shape[axis]  # handle negative indices
                base_offset += idx * strides[axis]
                view_shape[axis] = 1
            else:  # sub is a slice
                start, stop, step = sub.indices(parent_field.shape[axis])
                base_offset += start * strides[axis]
                strides[axis] *= step
                view_shape[axis] = len(range(start, stop, step))

        _, reshaped_strides = _reshape_affine_metadata(
            tuple(view_shape), tuple(strides), self.shape
        )
        self._base_offset = base_offset
        self._strides = reshaped_strides

        # precompute the update shape for the _set_in_array method
        self._update_shape = tuple(
            len(range(*sub.indices(parent_field.shape[axis])))
            for axis, sub in enumerate(parent_slice)
        )

    @overload
    def __get__(self, instance: None, owner) -> Field: ...
    @overload
    def __get__(self, instance: Compound, owner) -> Array: ...
    def __get__(self, instance, owner) -> Field | Array:
        if instance is None:
            return self
        return self._view(instance.arr)

    def _view(self, arr: Array) -> Array:
        if self._root_slice is None:
            raise RuntimeError(
                "Field slice is not set. This should be set during Compound initialization."
            )
        return (
            arr[self._root_slice]
            .reshape(self._root_shape)[self._view_slice]
            .reshape(self.shape)
        )

    def _set_in_array(self, arr: Array, value: Array) -> Array:
        parent = arr[self._root_slice].reshape(self._root_shape)
        parent = parent.at[self._view_slice].set(value.reshape(self._update_shape))
        return arr.at[self._root_slice].set(parent.reshape(-1))


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
                _slice=field._slice,
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
        field_obj._set_slice_and_strides(new_slice)

    new_fields.extend(trailing_fields)
    cls.fields = tuple(new_fields)
    cls.size = offset

    return cls
