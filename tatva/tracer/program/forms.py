"""Scalar-form coordinate metadata and symbolic derivative layouts.

The compiler treats energy, weak, and mixed forms uniformly.  A scalar form is
parameterized by coordinate blocks.  Blocks marked ROW define residual
coordinates; blocks marked COLUMN define linearization coordinates.  Energy is
the special case where the same block is both ROW and COLUMN.

This module deliberately contains no FEM-specific trial/test types.  Those are
front-end conveniences that lower to ``CoordinateBlock`` metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Jaxpr
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.dependencies import DependencySet


class CoordinateRole(Enum):
    ROW = auto()
    COLUMN = auto()
    ROW_AND_COLUMN = auto()

    @property
    def is_row(self) -> bool:
        return self in (CoordinateRole.ROW, CoordinateRole.ROW_AND_COLUMN)

    @property
    def is_column(self) -> bool:
        return self in (CoordinateRole.COLUMN, CoordinateRole.ROW_AND_COLUMN)


class ValueSource(Enum):
    """How an operator evaluation obtains values for a coordinate input.

    ``ZERO`` is a runtime policy only.  It must never make that input concrete
    during structural planning: formal row/test variables remain symbolic even
    when residual evaluation later substitutes zero for their values.
    """

    EXTERNAL = auto()
    ZERO = auto()


type CoordinateSelection = slice | tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class CoordinateBlock:
    """One independent symbolic coordinate block bound to a flat JAXPR input.

    ``selection`` addresses flattened scalar rows of the bound input.  ``None``
    means the whole input.  Blocks bound to the same input must not overlap;
    if the same physical coordinates are both rows and columns (energy), use a
    single ``ROW_AND_COLUMN`` block rather than two overlapping blocks.
    """

    name: str
    input_index: int
    role: CoordinateRole
    value_source: ValueSource = ValueSource.EXTERNAL
    selection: CoordinateSelection = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coordinate block name must be nonempty")
        if self.input_index < 0:
            raise ValueError("coordinate block input_index must be nonnegative")

    def rows(self, input_size: int) -> NDArray[np.int64]:
        if input_size < 0:
            raise ValueError("input_size must be nonnegative")

        if self.selection is None:
            rows = np.arange(input_size, dtype=np.int64)
        elif isinstance(self.selection, slice):
            rows = np.arange(input_size, dtype=np.int64)[self.selection]
        else:
            rows = np.asarray(self.selection, dtype=np.int64).ravel()

        if np.any((rows < 0) | (rows >= input_size)):
            raise ValueError(
                f"coordinate block {self.name!r} selects rows outside input "
                f"extent {input_size}"
            )
        if np.unique(rows).size != rows.size:
            raise ValueError(
                f"coordinate block {self.name!r} contains duplicate input rows"
            )
        return rows


@dataclass(frozen=True, slots=True)
class FormSpec:
    """Coordinate semantics of one scalar form."""

    coordinates: tuple[CoordinateBlock, ...]

    def __post_init__(self) -> None:
        names = tuple(block.name for block in self.coordinates)
        if len(set(names)) != len(names):
            raise ValueError("coordinate block names must be unique")
        if not any(block.role.is_row for block in self.coordinates):
            raise ValueError("form must contain at least one row coordinate block")
        if not any(block.role.is_column for block in self.coordinates):
            raise ValueError("form must contain at least one column coordinate block")

    @classmethod
    def energy(
        cls,
        *,
        input_index: int = 0,
        name: str = "u",
        selection: CoordinateSelection = None,
    ) -> FormSpec:
        return cls(
            (
                CoordinateBlock(
                    name=name,
                    input_index=input_index,
                    selection=selection,
                    role=CoordinateRole.ROW_AND_COLUMN,
                    value_source=ValueSource.EXTERNAL,
                ),
            )
        )

    @property
    def coordinate_input_indices(self) -> tuple[int, ...]:
        return tuple(sorted({block.input_index for block in self.coordinates}))


@dataclass(frozen=True, slots=True)
class SymbolicBlock:
    name: str
    offset: int
    size: int
    role: CoordinateRole
    value_source: ValueSource

    @property
    def columns(self) -> NDArray[np.int64]:
        return np.arange(self.offset, self.offset + self.size, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SymbolicLayout:
    """Concatenated symbolic coordinate layout used by derivative propagation."""

    blocks: tuple[SymbolicBlock, ...]
    size: int

    @classmethod
    def from_form(cls, form: FormSpec, jaxpr: Jaxpr) -> SymbolicLayout:
        input_sizes = tuple(int(math.prod(_shape_of(var))) for var in jaxpr.invars)
        occupied: dict[int, set[int]] = {}
        blocks: list[SymbolicBlock] = []
        offset = 0

        for spec in form.coordinates:
            if spec.input_index >= len(jaxpr.invars):
                raise ValueError(
                    f"coordinate block {spec.name!r} references input "
                    f"{spec.input_index}, but JAXPR has {len(jaxpr.invars)} inputs"
                )
            rows = spec.rows(input_sizes[spec.input_index])
            used = occupied.setdefault(spec.input_index, set())
            overlap = used.intersection(int(row) for row in rows)
            if overlap:
                sample = sorted(overlap)[:8]
                raise ValueError(
                    f"coordinate block {spec.name!r} overlaps another block on "
                    f"input {spec.input_index} at rows {sample}; use one "
                    "ROW_AND_COLUMN block when row/column coordinates are identical"
                )
            used.update(int(row) for row in rows)

            block = SymbolicBlock(
                name=spec.name,
                offset=offset,
                size=rows.size,
                role=spec.role,
                value_source=spec.value_source,
            )
            blocks.append(block)
            offset += rows.size

        return cls(tuple(blocks), offset)

    @classmethod
    def from_sizes(
        cls,
        blocks: tuple[tuple[str, int, CoordinateRole, ValueSource], ...],
    ) -> SymbolicLayout:
        symbolic: list[SymbolicBlock] = []
        offset = 0
        names: set[str] = set()
        for name, size, role, value_source in blocks:
            if not name or name in names:
                raise ValueError("symbolic block names must be nonempty and unique")
            if size < 0:
                raise ValueError("symbolic block size must be nonnegative")
            names.add(name)
            symbolic.append(SymbolicBlock(name, offset, size, role, value_source))
            offset += size
        return cls(tuple(symbolic), offset)

    def block(self, name: str) -> SymbolicBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)

    @property
    def row_columns(self) -> NDArray[np.int64]:
        parts = [block.columns for block in self.blocks if block.role.is_row]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    @property
    def column_columns(self) -> NDArray[np.int64]:
        parts = [block.columns for block in self.blocks if block.role.is_column]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    @property
    def row_block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks if block.role.is_row)

    @property
    def column_block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks if block.role.is_column)

    @property
    def has_identical_rows_and_columns(self) -> bool:
        return np.array_equal(self.row_columns, self.column_columns)

    def seed_inputs(
        self,
        form: FormSpec,
        jaxpr: Jaxpr,
    ) -> tuple[DependencySet, ...]:
        """Create identity root seeds from form input bindings."""
        matrices = [
            sps.lil_matrix(
                (int(math.prod(_shape_of(var))), self.size),
                dtype=bool,
            )
            for var in jaxpr.invars
        ]

        specs = {block.name: block for block in form.coordinates}
        for symbolic in self.blocks:
            spec = specs[symbolic.name]
            input_size = matrices[spec.input_index].shape[0]
            rows = spec.rows(input_size)
            if rows.size != symbolic.size:
                raise RuntimeError(
                    f"coordinate block {symbolic.name!r} changed size while seeding"
                )
            cols = symbolic.columns
            if rows.size:
                matrices[spec.input_index][rows, cols] = True

        return tuple(
            DependencySet(matrix.tocsr(), _shape_of(var))
            for matrix, var in zip(matrices, jaxpr.invars, strict=True)
        )

    def tangent_block(self, interactions: sps.spmatrix) -> sps.csr_matrix:
        matrix = sps.csr_matrix(interactions)
        if matrix.shape != (self.size, self.size):
            raise ValueError(
                f"interaction matrix shape {matrix.shape} does not match symbolic "
                f"layout size {self.size}"
            )
        return matrix[self.row_columns][:, self.column_columns].tocsr()
