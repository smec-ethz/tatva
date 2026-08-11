"""
Structured local tensor layouts.

A TensorLayout freezes an already-computed TensorDemand into a concrete local
storage contract.

The layout does not widen or otherwise modify demand. It only defines:

    global tensor shape
        +
    structured selected subset
        ->
    local tensor shape

and exact mappings between global and local axis coordinates / flattened
C-order scalar rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.demand import (
    AxisProduct,
    AxisSubset,
    Shape,
    TensorDemand,
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
    axis_indices,
)


def _shape_size(
    shape: Shape,
) -> int:
    return int(prod(shape))


def _axis_local_size(
    subset: AxisSubset,
    *,
    extent: int,
) -> int:
    if isinstance(subset, _FullAxis):
        return extent
    if isinstance(subset, _RangeAxis):
        return subset.size
    if isinstance(subset, _IndexAxis):
        return subset.size

    raise TypeError(f"unsupported axis subset {type(subset)!r}")


@dataclass(frozen=True, slots=True, eq=False)
class TensorLayout:
    """Local storage layout for one global tensor.

    `subset` uses global coordinates.

    For example:

        global_shape = (2601, 2)

        subset =
            AxisProduct(
                axis 0 = IndexAxis([1, 4, 8, ...]),
                axis 1 = FullAxis(),
            )

        local_shape =
            (n_selected_nodes, 2)

    Local coordinates are dense and zero-based regardless of how sparse the
    corresponding global coordinates are.
    """

    global_shape: Shape
    subset: AxisProduct

    def __post_init__(self) -> None:
        global_shape = tuple(int(extent) for extent in self.global_shape)

        if any(extent < 0 for extent in global_shape):
            raise ValueError(f"negative tensor extent in {global_shape}")

        if self.subset.shape != global_shape:
            raise ValueError(
                f"layout shape {global_shape} does not match "
                f"subset shape {self.subset.shape}"
            )

        if _shape_size(global_shape) == 0:
            raise ValueError("TensorLayout cannot represent an empty tensor demand")

        object.__setattr__(
            self,
            "global_shape",
            global_shape,
        )

    @classmethod
    def from_demand(
        cls,
        demand: TensorDemand,
    ) -> TensorLayout:
        """Freeze a demand without changing its represented subset."""
        return cls(
            global_shape=demand.shape,
            subset=demand.subset,
        )

    @property
    def ndim(self) -> int:
        return len(self.global_shape)

    @property
    def global_size(self) -> int:
        return _shape_size(self.global_shape)

    @property
    def local_shape(self) -> Shape:
        return tuple(
            _axis_local_size(axis, extent=extent)
            for extent, axis in zip(self.global_shape, self.subset.axes)
        )

    @property
    def local_size(self) -> int:
        return _shape_size(self.local_shape)

    @property
    def is_full(self) -> bool:
        return all(isinstance(axis, _FullAxis) for axis in self.subset.axes)

    def axis_subset(
        self,
        axis: int,
    ) -> AxisSubset:
        if axis < 0:
            axis += self.ndim

        if axis < 0 or axis >= self.ndim:
            raise IndexError(f"axis {axis} out of bounds for rank-{self.ndim} tensor")

        return self.subset.axes[axis]

    def global_axis_indices(
        self,
        axis: int,
    ) -> NDArray[np.int64]:
        """Global coordinates stored along one local axis.

        Returned order is exactly local storage order.
        """
        subset = self.axis_subset(axis)
        extent = self.global_shape[axis]
        return axis_indices(subset, extent=extent)

    def global_to_local_axis_or_missing(
        self,
        axis: int,
        indices: ArrayLike,
    ) -> NDArray[np.int64]:
        """Convert global coordinates on one axis to local coordinates.

        Global coordinates not stored by this layout map to -1.

        Shape of the input array is preserved.
        """
        values = np.asarray(indices, dtype=np.int64)
        extent = self.global_shape[axis]
        subset = self.axis_subset(axis)
        result = np.full(values.shape, -1, dtype=np.int64)
        valid_global = (values >= 0) & (values < extent)

        if isinstance(subset, _FullAxis):
            result[valid_global] = values[valid_global]
            return result

        if isinstance(subset, _RangeAxis):
            inside = valid_global & (values >= subset.start) & (values < subset.stop)
            result[inside] = values[inside] - subset.start
            return result

        if isinstance(subset, _IndexAxis):
            if not np.any(valid_global):
                return result

            candidates = values[valid_global]
            positions = np.searchsorted(
                subset.indices,
                candidates,
            )
            found = positions < subset.indices.size
            safe_positions = np.minimum(
                positions,
                subset.indices.size - 1,
            )
            found &= subset.indices[safe_positions] == candidates

            target = np.flatnonzero(valid_global.ravel())
            flat_result = result.ravel()
            flat_result[target[found]] = positions[found]

            return result

        raise TypeError(f"unsupported axis subset {type(subset)!r}")

    def global_to_local_axis(
        self,
        axis: int,
        indices: ArrayLike,
    ) -> NDArray[np.int64]:
        """Strict global -> local axis mapping.

        Raises if any requested global coordinate is not stored locally.
        """
        values = np.asarray(indices, dtype=np.int64)
        result = self.global_to_local_axis_or_missing(axis, values)
        missing = result < 0

        if np.any(missing):
            missing_values = np.unique(values[missing])
            preview = missing_values[:8]

            raise ValueError(
                f"global coordinates {preview.tolist()} "
                f"on axis {axis} are not present "
                "in the local layout"
            )

        return result

    def local_to_global_axis(
        self,
        axis: int,
        indices: ArrayLike,
    ) -> NDArray[np.int64]:
        """Convert dense local axis coordinates back to global coordinates."""
        values = np.asarray(indices, dtype=np.int64)
        subset = self.axis_subset(axis)
        local_extent = self.local_shape[axis]
        valid = (values >= 0) & (values < local_extent)

        if not np.all(valid):
            bad = np.unique(values[~valid])

            raise ValueError(
                f"local coordinates {bad[:8].tolist()} "
                f"out of bounds for local axis "
                f"of extent {local_extent}"
            )

        if isinstance(subset, _FullAxis):
            return values.copy()

        if isinstance(subset, _RangeAxis):
            return (values + subset.start).astype(np.int64, copy=False)

        if isinstance(subset, _IndexAxis):
            return subset.indices[values]

        raise TypeError(f"unsupported axis subset {type(subset)!r}")

    def global_rows_to_local_rows(
        self,
        rows: ArrayLike,
        *,
        allow_missing: bool = False,
    ) -> NDArray[np.int64]:
        """Convert global C-order flattened scalar rows to local flattened rows.

        When `allow_missing=True`, rows outside the local layout map to -1.
        Existing negative route sentinels also remain -1.

        This is intended for route localization.
        """
        global_rows = np.asarray(rows, dtype=np.int64)
        original_shape = global_rows.shape
        flat_rows = global_rows.ravel()
        result = np.full(flat_rows.shape, -1, dtype=np.int64)
        valid_global_row = (flat_rows >= 0) & (flat_rows < self.global_size)

        if not allow_missing and not np.all(valid_global_row):
            bad = np.unique(flat_rows[~valid_global_row])
            raise ValueError(f"global rows {bad[:8].tolist()} are out of bounds")

        valid_positions = np.flatnonzero(valid_global_row)
        if valid_positions.size == 0:
            return result.reshape(original_shape)

        selected_rows = flat_rows[valid_positions]

        # Scalars have one flattened row: 0 -> 0.
        if self.ndim == 0:
            result[valid_positions] = 0
            return result.reshape(original_shape)

        global_coords = np.unravel_index(selected_rows, self.global_shape)
        local_coords = []
        inside = np.ones(selected_rows.shape, dtype=bool)

        for axis, coords in enumerate(global_coords):
            mapped = self.global_to_local_axis_or_missing(axis, coords)
            inside &= mapped >= 0
            local_coords.append(mapped)

        if not allow_missing and not np.all(inside):
            missing_rows = selected_rows[~inside]
            raise ValueError(
                f"global rows {missing_rows[:8].tolist()} "
                "are not stored by the local layout"
            )

        if np.any(inside):
            local_inside_coords = tuple(coords[inside] for coords in local_coords)
            local_rows = np.ravel_multi_index(local_inside_coords, self.local_shape)
            result[valid_positions[inside]] = local_rows.astype(np.int64, copy=False)

        return result.reshape(original_shape)

    def local_rows_to_global_rows(
        self,
        rows: ArrayLike,
    ) -> NDArray[np.int64]:
        """Convert local C-order flattened scalar rows to global scalar rows."""
        local_rows = np.asarray(rows, dtype=np.int64)
        original_shape = local_rows.shape
        flat_rows = local_rows.ravel()
        valid = (flat_rows >= 0) & (flat_rows < self.local_size)

        if not np.all(valid):
            bad = np.unique(flat_rows[~valid])
            raise ValueError(f"local rows {bad[:8].tolist()} are out of bounds")

        if self.ndim == 0:
            return np.zeros(original_shape, dtype=np.int64)

        local_coords = np.unravel_index(flat_rows, self.local_shape)
        global_coords = tuple(
            self.local_to_global_axis(
                axis,
                coords,
            )
            for axis, coords in enumerate(local_coords)
        )
        global_rows = np.ravel_multi_index(global_coords, self.global_shape)

        return np.asarray(global_rows, dtype=np.int64).reshape(original_shape)


def finalize_layout(
    demand: TensorDemand,
) -> TensorLayout:
    """Freeze a demand into its storage layout.

    This function intentionally performs no legalization or widening.
    """
    return TensorLayout.from_demand(demand)
