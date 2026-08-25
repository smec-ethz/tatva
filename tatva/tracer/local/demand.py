"""
Structured backward demand propagation.

A TensorDemand describes which entries of a global tensor are required by one
localized rank. Demands preserve tensor structure: ordinary persistent demands
are either the complete tensor or a Cartesian product of per-axis selections.

Exact primitive routes may temporarily operate on flattened scalar rows. Such
rows are converted back into the smallest structured Cartesian hull before the
demand leaves the primitive rule.

No demand is represented by ``None`` rather than by an empty TensorDemand.

Core invariant:

    exact required entries
        ⊆ TensorDemand
        ⊆ global tensor

The first inclusion may be strict because Cartesian closure can conservatively
include additional tensor entries. This controlled over-computation preserves
structured local tensor layouts and avoids arbitrary point-cloud storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.routes import Shape

type Demand = TensorDemand | None


def _shape_size(shape: Shape) -> int:
    return int(math.prod(shape))


def _validate_shape(shape: Shape) -> None:
    if any(extent < 0 for extent in shape):
        raise ValueError(f"invalid shape {shape}: all extents must be non-negative")


def _validate_axis(shape: Shape, axis: int) -> None:
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"axis {axis} is out of range for shape {shape}")


def _integer_array(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)

    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer values")

    return np.asarray(array, dtype=np.int64).ravel()


@dataclass(frozen=True, slots=True)
class _FullAxis:
    """Every index along this axis."""


@dataclass(frozen=True, slots=True)
class _RangeAxis:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"range axis start must be non-negative, got {self.start}")
        if self.stop <= self.start:
            raise ValueError(f"invalid range [{self.start}, {self.stop})")

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True, eq=False)
class _IndexAxis:
    indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).ravel()

        if indices.size == 0:
            raise ValueError("IndexAxis cannot be empty")

        # Avoid np.unique when caller already supplied canonical data.
        if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
            indices = np.unique(indices)
        else:
            # Ensure we own the array before making it read-only.
            indices = indices.copy()

        if indices[0] < 0:
            raise ValueError(f"indices must be non-negative, got {indices[0]}")

        indices.setflags(write=False)
        object.__setattr__(self, "indices", indices)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IndexAxis) and np.array_equal(
            self.indices, other.indices
        )

    __hash__ = None

    @property
    def size(self) -> int:
        return self.indices.size


type AxisSubset = _FullAxis | _RangeAxis | _IndexAxis


@dataclass(frozen=True, slots=True, eq=False)
class AxisProduct:
    shape: Shape
    axes: tuple[AxisSubset, ...]

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if len(self.shape) != len(self.axes):
            raise ValueError("one axis subset required per tensor axis")

        for axis, (extent, subset) in enumerate(zip(self.shape, self.axes)):
            if isinstance(subset, _FullAxis):
                continue
            if isinstance(subset, _RangeAxis):
                if subset.stop > extent:
                    raise ValueError(
                        f"range [{subset.start}, {subset.stop}) "
                        f"exceeds axis {axis} extent {extent}"
                    )
                if subset.start == 0 and subset.stop == extent:
                    raise ValueError(
                        "complete range must be represented by FullAxis, not RangeAxis"
                    )
                continue

            if subset.indices[-1] >= extent:
                raise ValueError(
                    f"index {subset.indices[-1]} exceeds axis {axis} extent {extent}"
                )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AxisProduct)
            and self.shape == other.shape
            and len(self.axes) == len(other.axes)
            and all(lhs == rhs for lhs, rhs in zip(self.axes, other.axes))
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TensorDemand:
    subset: AxisProduct

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if _shape_size(self.shape) == 0:
            raise ValueError("zero-sized tensors have no entries to demand; use None")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TensorDemand)
            and self.shape == other.shape
            and self.subset == other.subset
        )

    __hash__ = None

    @property
    def shape(self) -> Shape:
        return self.subset.shape

    @property
    def is_full(self) -> bool:
        return all(isinstance(axis, _FullAxis) for axis in self.subset.axes)

    @property
    def size(self) -> int:
        size = 1
        for extent, axis in zip(self.shape, self.subset.axes):
            size *= extent if isinstance(axis, _FullAxis) else axis.size
        return size

    @property
    def axes(self) -> tuple[AxisSubset, ...]:
        return self.subset.axes

    def rows(self) -> NDArray[np.int64]:
        """Exact C-order scalar rows represented by this Cartesian demand."""
        if not self.shape:
            return np.array([0], dtype=np.int64)

        if self.is_full:
            return np.arange(int(np.prod(self.shape, dtype=np.int64)), dtype=np.int64)

        # C-order flat strides.
        strides = np.empty(len(self.shape), dtype=np.int64)
        stride = 1

        for axis in range(len(self.shape) - 1, -1, -1):
            strides[axis] = stride
            stride *= self.shape[axis]

        rows = np.zeros(1, dtype=np.int64)

        for extent, subset, stride in zip(self.shape, self.axes, strides, strict=True):
            indices = axis_indices(subset, extent=extent)
            if indices.size == 1:
                # Very common for gather envelopes: batch and non-pivot
                # axes are fixed.
                rows += int(indices[0]) * stride
                continue

            rows = (rows[:, None] + indices[None, :] * stride).reshape(-1)

        return rows

    @classmethod
    def full(cls, shape: tuple[int, ...]) -> Self | None:
        _validate_shape(shape)
        if _shape_size(shape) == 0:
            return None
        return cls(AxisProduct(shape=shape, axes=tuple(_FullAxis() for _ in shape)))

    @classmethod
    def axis_selection(cls, shape: Shape, axis: int, indices: ArrayLike) -> Self | None:
        _validate_shape(shape)
        _validate_axis(shape, axis)

        selected = _axis_from_indices(shape[axis], indices)
        return cls._single_axis(shape, axis, selected)

    @classmethod
    def axis_range(cls, shape: Shape, axis: int, start: int, stop: int) -> Self | None:
        _validate_shape(shape)
        _validate_axis(shape, axis)

        selected = _axis_from_range(shape[axis], start, stop)
        return cls._single_axis(shape, axis, selected)

    @classmethod
    def from_axes(cls, shape: Shape, axes: tuple[AxisSubset, ...]) -> Self | None:
        if _shape_size(shape) == 0:
            return None

        if len(shape) != len(axes):
            raise ValueError("one axis subset required per tensor axis")

        return cls(AxisProduct(shape, axes))

    @classmethod
    def _single_axis(
        cls,
        shape: Shape,
        axis: int,
        selection: AxisSubset | None,
    ) -> Self | None:
        if _shape_size(shape) == 0 or selection is None:
            return None

        axes: list[AxisSubset] = [_FullAxis() for _ in shape]
        axes[axis] = selection
        return cls.from_axes(shape, tuple(axes))

    @classmethod
    def from_rows_hull(
        cls,
        shape: Shape,
        rows: ArrayLike,
    ) -> Self | None:
        """Construct the smallest AxisProduct containing the requested flat rows.

        `rows` are C-order flattened scalar-entry indices.

        The result is exact when the requested rows already form a Cartesian
        product. Otherwise this intentionally computes the Cartesian hull.

        Example:

            shape = (3, 4)
            rows corresponding to:
                (0, 1)
                (2, 3)

        require axis indices:

            axis 0: {0, 2}
            axis 1: {1, 3}

        so the structured hull additionally contains:

            (0, 3)
            (2, 1)
        """
        shape = tuple(int(extent) for extent in shape)
        _validate_shape(shape)
        n_entries = _shape_size(shape)
        rows = _integer_array(rows, name="rows")

        if rows.size == 0 or n_entries == 0:
            return None

        if np.any(rows < 0) or np.any(rows >= n_entries):
            raise ValueError(f"flat rows are outside tensor with shape {shape}")

        rows = np.unique(rows)

        if rows.size == n_entries:
            return cls(AxisProduct(shape, tuple(_FullAxis() for _ in shape)))

        coordinates = np.unravel_index(rows, shape)
        axes: list[AxisSubset] = []

        for extent, axis_coordinates in zip(shape, coordinates):
            selected = np.unique(axis_coordinates)
            axis_subset = _axis_from_indices(extent, selected)
            if axis_subset is None:
                raise RuntimeError("non-empty row hull produced an empty axis")
            axes.append(axis_subset)

        return cls.from_axes(shape, tuple(axes))

    def merge(
        self,
        other: Self,
    ) -> Self:
        """Smallest structured demand containing both demands."""
        if self.shape != other.shape:
            raise ValueError(
                f"cannot merge demand shape {self.shape} with {other.shape}"
            )

        if self.is_full:
            return self
        if other.is_full:
            return other

        axes = tuple(
            _merge_axis_subsets(lhs, rhs, extent=extent)
            for extent, lhs, rhs in zip(self.shape, self.subset.axes, other.subset.axes)
        )
        result = type(self).from_axes(self.shape, axes)

        # Both inputs are non-empty, therefore their union is non-empty.
        assert result is not None

        return result

    def axis_subset(
        self,
        axis: int,
    ) -> AxisSubset:
        if axis < 0 or axis >= len(self.shape):
            raise ValueError(f"axis {axis} out of range for shape {self.shape}")

        return self.subset.axes[axis]

    def selected_indices(
        self,
        axis: int,
    ) -> NDArray[np.int64]:
        subset = self.axis_subset(axis)
        return axis_indices(subset, extent=self.shape[axis])


def _merge_axis_subsets(
    lhs: AxisSubset,
    rhs: AxisSubset,
    *,
    extent: int,
) -> AxisSubset:
    if isinstance(lhs, _FullAxis):
        return lhs

    if isinstance(rhs, _FullAxis):
        return rhs

    # Range ∪ Range can often remain a Range without allocating.
    if isinstance(lhs, _RangeAxis) and isinstance(rhs, _RangeAxis):
        # Overlap or adjacency:
        #
        # [2, 5) ∪ [5, 8) -> [2, 8)
        #
        if lhs.start <= rhs.stop and rhs.start <= lhs.stop:
            start = min(lhs.start, rhs.start)
            stop = max(lhs.stop, rhs.stop)
            result = _axis_from_range(extent, start, stop)

            assert result is not None
            return result

        # Disjoint ranges need an irregular representation.
        indices = np.concatenate(
            (
                np.arange(lhs.start, lhs.stop, dtype=np.int64),
                np.arange(rhs.start, rhs.stop, dtype=np.int64),
            )
        )
        result = _axis_from_indices(extent, indices)

        assert result is not None
        return result

    # At least one irregular selection.
    lhs_indices = axis_indices(lhs, extent=extent)
    rhs_indices = axis_indices(rhs, extent=extent)

    indices = np.union1d(lhs_indices, rhs_indices)
    result = _axis_from_indices(extent, indices)

    assert result is not None
    return result


def axis_indices(axis: AxisSubset, *, extent: int) -> NDArray[np.int64]:
    if isinstance(axis, _FullAxis):
        return np.arange(extent, dtype=np.int64)

    if isinstance(axis, _RangeAxis):
        return np.arange(axis.start, axis.stop, dtype=np.int64)

    return axis.indices


def axis_contains(axis: AxisSubset, index: int, *, extent: int) -> bool:
    if index < 0 or index >= extent:
        return False
    if isinstance(axis, _FullAxis):
        return True
    if isinstance(axis, _RangeAxis):
        return axis.start <= index < axis.stop

    pos = np.searchsorted(axis.indices, index)
    return bool(pos < axis.indices.size and axis.indices[pos] == index)


def _is_contiguous_indices(indices: NDArray[np.int64]) -> bool:
    if indices.size <= 1:
        return True
    return bool(np.all(indices[1:] == indices[:-1] + 1))


def _axis_from_indices(
    extent: int,
    indices: ArrayLike,
) -> AxisSubset | None:
    """Construct the canonical axis subset for a non-empty set of indices.

    Returns None for an empty selection.
    """
    selected = _integer_array(indices, name="indices")
    if selected.size == 0:
        return None

    if selected.size > 1 and np.any(selected[1:] <= selected[:-1]):
        selected = np.unique(selected)
    else:
        selected = selected.copy()

    if selected[0] < 0 or selected[-1] >= extent:
        raise ValueError(f"indices are outside axis extent {extent}")

    if selected.size == extent:
        return _FullAxis()

    if _is_contiguous_indices(selected):
        return _RangeAxis(
            start=int(selected[0]),
            stop=int(selected[-1]) + 1,
        )

    return _IndexAxis(indices=selected)


def _axis_from_range(
    extent: int,
    start: int,
    stop: int,
) -> AxisSubset | None:
    if not 0 <= start <= stop <= extent:
        raise ValueError(f"invalid range [{start}, {stop}) for extent {extent}")

    if start == stop:
        return None

    if start == 0 and stop == extent:
        return _FullAxis()

    return _RangeAxis(start=start, stop=stop)


def merge_demands(
    lhs: Demand,
    rhs: Demand,
) -> Demand:
    """Merge two possibly absent demands."""
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs

    return lhs.merge(rhs)


def take_leading_axis_demand(
    demand: Demand,
    index: int,
) -> Demand:
    """Demand on y[index] induced by a demand on stacked y."""
    if demand is None:
        return None

    if not demand.shape:
        raise ValueError("cannot slice leading axis of scalar demand")

    extent = demand.shape[0]
    first = demand.subset.axes[0]

    if not axis_contains(first, index, extent=extent):
        return None

    return TensorDemand.from_axes(
        demand.shape[1:],
        demand.subset.axes[1:],
    )


def lift_leading_axis_demand(
    demand: Demand,
    *,
    outer_shape: Shape,
    index: int,
) -> Demand:
    """Lift a demand on x_step into the corresponding x[index] slice."""
    if demand is None:
        return None

    if not outer_shape:
        raise ValueError("mapped input requires a leading axis")

    if demand.shape != outer_shape[1:]:
        raise ValueError(
            f"step demand shape {demand.shape} "
            f"does not match mapped input shape {outer_shape}"
        )

    leading = _axis_from_range(outer_shape[0], index, index + 1)
    assert leading is not None
    inner = demand.axes

    return TensorDemand.from_axes(
        outer_shape,
        (leading,) + inner,
    )
