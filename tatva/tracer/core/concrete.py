"""Demand-scoped concrete values with explicit global-coordinate metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.local.demand import TensorDemand, _FullAxis, _IndexAxis, _RangeAxis
from tatva.tracer.local.layout import TensorLayout


@dataclass(frozen=True, slots=True)
class ConcreteRegion:
    """Compact values representing one structured subset of a global tensor."""

    values: NDArray[Any]
    demand: TensorDemand

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if tuple(values.shape) != self.layout.local_shape:
            raise ValueError(
                f"regional concrete shape {values.shape} does not match "
                f"demand-local shape {self.layout.local_shape}"
            )
        object.__setattr__(self, "values", values)

    @property
    def global_shape(self) -> tuple[int, ...]:
        return self.demand.shape

    @property
    def layout(self) -> TensorLayout:
        return TensorLayout.from_demand(self.demand)

    @classmethod
    def from_full(cls, value: Any, demand: TensorDemand) -> ConcreteRegion:
        """Slice a caller-owned full value before converting it to host NumPy."""
        value_shape = tuple(getattr(value, "shape", np.shape(value)))
        if value_shape != demand.shape:
            raise ValueError(
                f"concrete value shape {value_shape} does not match "
                f"demand shape {demand.shape}"
            )
        result = value
        for axis, subset in enumerate(demand.axes):
            if isinstance(subset, _FullAxis):
                continue
            if isinstance(subset, _RangeAxis):
                index = [slice(None)] * len(demand.shape)
                index[axis] = slice(subset.start, subset.stop)
                result = result[tuple(index)]
            elif isinstance(subset, _IndexAxis):
                result = result.take(subset.indices, axis=axis)
            else:
                raise TypeError(f"unsupported axis subset {type(subset)!r}")
        return cls(np.asarray(result), demand)

    def read_rows(self, global_rows: ArrayLike) -> NDArray[Any]:
        """Read global flattened rows known to lie inside this region."""
        local_rows = self.layout.global_rows_to_local_rows(global_rows)
        return self.values.ravel()[local_rows]

    def project(self, demand: TensorDemand) -> ConcreteRegion | None:
        """Project a contained demand, returning ``None`` when it is not contained."""
        if demand.shape != self.global_shape:
            return None
        rows = demand.rows()
        local_rows = self.layout.global_rows_to_local_rows(rows, allow_missing=True)
        if np.any(local_rows < 0):
            return None
        layout = TensorLayout.from_demand(demand)
        values = self.values.ravel()[local_rows].reshape(layout.local_shape)
        return ConcreteRegion(values, demand)
