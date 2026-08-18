"""Sparse colored demand relations used by distributed partition analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.routes import Shape


@dataclass(frozen=True, slots=True, eq=False)
class TaggedDemand:
    """COO-like sparse relation from flattened tensor rows to block IDs."""

    shape: Shape
    rows: NDArray[np.int64]
    blocks: NDArray[np.int64]

    def __post_init__(self) -> None:
        shape = tuple(int(extent) for extent in self.shape)
        if any(extent < 0 for extent in shape):
            raise ValueError(f"invalid tagged demand shape {shape}")

        rows = np.asarray(self.rows, dtype=np.int64).ravel()
        blocks = np.asarray(self.blocks, dtype=np.int64).ravel()
        if rows.shape != blocks.shape:
            raise ValueError("tagged demand rows and blocks must have equal length")

        n_entries = int(math.prod(shape))
        if np.any(rows < 0) or np.any(rows >= n_entries):
            raise ValueError(f"tagged demand rows are outside tensor shape {shape}")
        if np.any(blocks < 0):
            raise ValueError("tagged demand block IDs must be non-negative")

        if rows.size:
            order = np.lexsort((blocks, rows))
            rows = rows[order]
            blocks = blocks[order]
            keep = np.ones(rows.size, dtype=bool)
            keep[1:] = (rows[1:] != rows[:-1]) | (blocks[1:] != blocks[:-1])
            rows = rows[keep]
            blocks = blocks[keep]

        rows = rows.copy()
        blocks = blocks.copy()
        rows.setflags(write=False)
        blocks.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "blocks", blocks)

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    @property
    def block_ids(self) -> NDArray[np.int64]:
        return np.unique(self.blocks)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TaggedDemand)
            and self.shape == other.shape
            and np.array_equal(self.rows, other.rows)
            and np.array_equal(self.blocks, other.blocks)
        )

    def with_shape(self, shape: Shape) -> Self:
        if int(math.prod(shape)) != int(math.prod(self.shape)):
            raise ValueError(f"cannot reshape tagged demand {self.shape} to {shape}")
        return type(self)(shape, self.rows, self.blocks)

    def mapped(
        self,
        shape: Shape,
        rows: ArrayLike,
        *,
        valid: ArrayLike | None = None,
    ) -> Self | None:
        mapped_rows = np.asarray(rows, dtype=np.int64).ravel()
        if mapped_rows.shape != self.rows.shape:
            raise ValueError("one mapped input row is required per tagged pair")
        if valid is None:
            keep = mapped_rows >= 0
        else:
            keep = np.asarray(valid, dtype=bool).ravel() & (mapped_rows >= 0)
        if not np.any(keep):
            return None
        return type(self)(shape, mapped_rows[keep], self.blocks[keep])

    def for_blocks(self, block_ids: ArrayLike) -> Self | None:
        selected = np.asarray(block_ids, dtype=np.int64).ravel()
        keep = np.isin(self.blocks, selected)
        if not np.any(keep):
            return None
        return type(self)(self.shape, self.rows[keep], self.blocks[keep])

    def take_leading_axis(self, index: int) -> Self | None:
        if not self.shape:
            raise ValueError("cannot slice the leading axis of a scalar demand")
        if index < 0 or index >= self.shape[0]:
            raise IndexError(f"leading-axis index {index} outside shape {self.shape}")
        child_shape = self.shape[1:]
        child_size = int(math.prod(child_shape))
        keep = self.rows // child_size == index
        if not np.any(keep):
            return None
        return type(self)(
            child_shape,
            self.rows[keep] % child_size,
            self.blocks[keep],
        )

    def lift_leading_axis(self, *, outer_shape: Shape, index: int) -> Self:
        if outer_shape[1:] != self.shape:
            raise ValueError(
                f"cannot lift tagged shape {self.shape} into outer shape {outer_shape}"
            )
        if index < 0 or index >= outer_shape[0]:
            raise IndexError(f"leading-axis index {index} outside shape {outer_shape}")
        child_size = int(math.prod(self.shape))
        return type(self)(
            outer_shape,
            index * child_size + self.rows,
            self.blocks,
        )

    @classmethod
    def full(cls, shape: Shape, blocks: ArrayLike) -> Self | None:
        block_ids = np.unique(np.asarray(blocks, dtype=np.int64).ravel())
        n_entries = int(math.prod(shape))
        if n_entries == 0 or block_ids.size == 0:
            return None
        return cls(
            shape,
            np.tile(np.arange(n_entries, dtype=np.int64), block_ids.size),
            np.repeat(block_ids, n_entries),
        )


type Tagged = TaggedDemand | None


def merge_tagged(lhs: Tagged, rhs: Tagged) -> Tagged:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs.shape != rhs.shape:
        raise ValueError(
            f"cannot merge tagged shape {lhs.shape} with tagged shape {rhs.shape}"
        )
    return TaggedDemand(
        lhs.shape,
        np.concatenate((lhs.rows, rhs.rows)),
        np.concatenate((lhs.blocks, rhs.blocks)),
    )


def active_blocks(demands: tuple[Tagged, ...]) -> NDArray[np.int64]:
    parts = [demand.blocks for demand in demands if demand is not None]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))
