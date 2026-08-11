"""
rank-owned ContributionRoot subset
    ↓
backward through producers
    ↓
which global entries of every intermediate are needed?
"""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class FullAxis:
    extent: int


@dataclass(frozen=True)
class IndexAxis:
    extent: int
    indices: tuple[int, ...]


type AxisSubset = FullAxis | IndexAxis


@dataclass(frozen=True)
class AxisProduct:
    shape: tuple[int, ...]
    axes: tuple[AxisSubset, ...]

    def __post_init__(self):
        if len(self.shape) != len(self.axes):
            raise ValueError("one axis subset required per tensor axis")


@dataclass(frozen=True)
class TensorDemand:
    shape: tuple[int, ...]
    subset: AxisProduct

    @classmethod
    def full(cls, shape: tuple[int, ...]) -> Self:
        return cls(
            shape, AxisProduct(shape, tuple(FullAxis(extent) for extent in shape))
        )

    @classmethod
    def axis_selection(
        cls, shape: tuple[int, ...], axis: int, indices: tuple[int, ...]
    ) -> Self:
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"axis {axis} is out of range for shape {shape}")

        if any(i < 0 or i >= shape[axis] for i in indices):
            raise ValueError(
                f"indices {indices} are out of range for axis {axis} with extent {shape[axis]}"
            )

        axes = tuple(
            IndexAxis(shape[i], indices) if i == axis else FullAxis(shape[i])
            for i in range(len(shape))
        )
        return cls(shape, AxisProduct(shape, axes))
