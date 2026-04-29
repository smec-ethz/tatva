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

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Callable, Self, TypeVar, overload

import jax.numpy as jnp
from jax import Array

from tatva.compound.field_types import FieldSize, FieldType, _FieldType

if TYPE_CHECKING:
    from tatva.compound import Compound


T_Compound = TypeVar("T_Compound", bound="Compound")


@dataclass(frozen=True)
class _FieldSpec:
    """A specification for a field in a Compound class."""

    shape: tuple[int, ...]
    default_factory: Callable | None = None
    field_type: FieldType | _FieldType = FieldType.LOCAL


if TYPE_CHECKING:
    # We lie to the type checker. We tell it field() returns a runtime Field descriptor
    def field(
        shape: tuple[int, ...],
        default_factory: Callable | None = None,
        field_type: FieldType | _FieldType = FieldType.LOCAL,
    ) -> Field: ...

else:
    field = _FieldSpec


def _normalize_index(
    arg: int | slice | Array | tuple[int | slice | Array, ...],
    shape: tuple[int, ...],
) -> tuple[int | slice | Array, ...]:
    if not isinstance(arg, tuple):
        arg = (arg,)
    if len(arg) < len(shape):
        arg = arg + (slice(None),) * (len(shape) - len(arg))  # ty:ignore[invalid-assignment]
    return arg  # ty:ignore[invalid-return-type]


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides_rev: list[int] = []
    for extent in reversed(shape):
        strides_rev.append(stride)
        stride *= extent
    return tuple(reversed(strides_rev))


def _compute_reshaped_strides(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if shape == target_shape:
        return strides

    # Find common prefix
    prefix_len = 0
    for s1, s2 in zip(shape, target_shape):
        if s1 == s2:
            prefix_len += 1
        else:
            break

    if (prefix_len < len(shape)) and (len(shape) - prefix_len == 1):
        # We expect the rest of shape to be exactly 1 dimension (the flattened suffix)
        extent = shape[prefix_len]
        suffix = target_shape[prefix_len:]
        if extent == int(prod(suffix)):
            extent_stride = strides[prefix_len]
            suffix_strides = _row_major_strides(suffix)
            scaled_suffix_strides = tuple(s * extent_stride for s in suffix_strides)
            return strides[:prefix_len] + scaled_suffix_strides

    # Fallback for dropping 1s
    compact_shape = tuple(e for e in shape if e != 1)
    if compact_shape == target_shape:
        return tuple(
            stride for extent, stride in zip(shape, strides, strict=True) if extent != 1
        )

    raise ValueError(
        f"Cannot reshape affine metadata from shape {shape} to {target_shape}."
    )


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
    field_type: _FieldType | FieldType

    def __init__(
        self,
        shape: tuple[int, ...],
        default_factory: Callable | None = None,
        field_type: _FieldType | FieldType = FieldType.LOCAL,
        *,
        _slice: slice | None = None,
    ) -> None:
        super().__init__(shape, _slice)
        self.default_factory = default_factory
        self.field_type = field_type

    def _copy_with_slice(self, _slice: slice) -> Self:
        return self.__class__(
            shape=self.shape,
            default_factory=self.default_factory,
            field_type=self.field_type,
            _slice=_slice,
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
        field_type: _FieldType | FieldType = FieldType.LOCAL,
    ) -> None:
        if parent_field._slice is None:
            raise RuntimeError(
                "Parent field slice is not set. This should be set during Compound initialization."
            )

        _FieldBase.__init__(self, shape=shape)
        self.default_factory = default_factory
        self.field_type = field_type

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

        reshaped_strides = _compute_reshaped_strides(
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
