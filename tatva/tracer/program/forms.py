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
import typing
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Any

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Jaxpr
from numpy.typing import NDArray

from tatva.tracer.capture import CallABI
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.inputs import LocalInputPlan
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
class FormCoordinate:
    role: CoordinateRole
    value_source: ValueSource = ValueSource.EXTERNAL
    name: str | None = None
    selection: CoordinateSelection = None


type Trial[T] = Annotated[
    T,
    FormCoordinate(role=CoordinateRole.COLUMN, value_source=ValueSource.EXTERNAL),
]
type Test[T] = Annotated[
    T,
    FormCoordinate(role=CoordinateRole.ROW, value_source=ValueSource.ZERO),
]
type State[T] = Annotated[
    T,
    FormCoordinate(
        role=CoordinateRole.ROW_AND_COLUMN, value_source=ValueSource.EXTERNAL
    ),
]


def extract_form_coordinates(hint: Any) -> list[FormCoordinate]:
    """Recursively extract FormCoordinate instances from Annotated or TypeAliasType hints."""
    coords: list[FormCoordinate] = []

    def _walk(node: Any) -> None:
        if node is None:
            return

        origin = typing.get_origin(node)

        # Generic alias of TypeAliasType (e.g. Trial[Array])
        if isinstance(origin, typing.TypeAliasType):
            _walk(origin.__value__)
            return

        # Bare TypeAliasType (e.g. Trial)
        if isinstance(node, typing.TypeAliasType):
            _walk(node.__value__)
            return

        # typing.Annotated
        if origin is typing.Annotated:
            args = typing.get_args(node)
            for metadata in args[1:]:
                if isinstance(metadata, FormCoordinate):
                    coords.append(metadata)
                else:
                    _walk(metadata)
            _walk(args[0])

    _walk(hint)
    return coords


def infer_form_spec(fn: Callable, call_abi: CallABI) -> FormSpec | None:
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # ruff: ignore[blind-except]
        return None

    blocks: list[CoordinateBlock] = []
    param_ranges = {name: span for name, _, span in call_abi.parameter_trees()}

    for name in call_abi.signature.parameters:
        hint = hints.get(name)
        if hint is None:
            continue

        coordinates = extract_form_coordinates(hint)
        if not coordinates:
            continue

        span = param_ranges.get(name)
        if span is None:
            continue

        for metadata in coordinates:
            for flat_idx in range(span.start, span.stop):
                block_name = metadata.name or (
                    name
                    if span.stop - span.start == 1
                    else f"{name}_{flat_idx - span.start}"
                )
                blocks.append(
                    CoordinateBlock(
                        name=block_name,
                        input_index=flat_idx,
                        role=metadata.role,
                        value_source=metadata.value_source,
                        selection=metadata.selection,
                    )
                )

    if not blocks:
        return None
    return FormSpec(tuple(blocks))


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
class LocalForm:
    spec: FormSpec
    row_global_ids: NDArray[np.int64]
    column_global_ids: NDArray[np.int64]
    global_shape: tuple[int, int]
    row_block_names: tuple[str, ...]
    column_block_names: tuple[str, ...]


def localize_form(
    form: FormSpec,
    inputs: LocalInputPlan,
) -> LocalForm:
    """Project a global form onto the rank-local input coordinate spaces."""

    local_blocks: list[CoordinateBlock] = []
    block_global_ids: dict[str, NDArray[np.int64]] = {}
    block_global_sizes: dict[str, int] = {}

    # Localize every form coordinate block.
    for block in form.coordinates:
        try:
            candidate_global_rows = inputs.global_rows(block.input_index)
            global_value = inputs.global_examples[block.input_index]
        except IndexError as exc:
            raise RuntimeError(
                f"coordinate block {block.name!r} references "
                f"input {block.input_index}, which does not exist"
            ) from exc

        if candidate_global_rows is None:
            raise RuntimeError(
                f"coordinate block {block.name!r} references "
                f"compiler-dead local input {block.input_index}"
            )

        original_input_size = int(np.prod(np.shape(global_value), dtype=np.int64))
        local_block, global_ids = _local_coordinate_block(
            block, candidate_global_rows, original_input_size
        )

        local_blocks.append(local_block)
        block_global_ids[block.name] = global_ids
        block_global_sizes[block.name] = block.rows(original_input_size).size

    local_spec = FormSpec(tuple(local_blocks))

    # Convert per-coordinate-block global IDs into the flattened
    # row/column coordinate spaces used by the matrix pattern.
    def coordinate_axis(
        *, rows: bool
    ) -> tuple[NDArray[np.int64], tuple[str, ...], int]:
        ids: list[NDArray[np.int64]] = []
        names: list[str] = []

        offset = 0

        for block in form.coordinates:
            included = block.role.is_row if rows else block.role.is_column
            if not included:
                continue

            names.append(block.name)
            ids.append(block_global_ids[block.name] + offset)
            offset += block_global_sizes[block.name]

        global_ids = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)

        return (global_ids, tuple(names), offset)

    row_ids, row_names, global_rows = coordinate_axis(rows=True)
    column_ids, column_names, global_columns = coordinate_axis(rows=False)

    return LocalForm(
        spec=local_spec,
        row_global_ids=row_ids,
        column_global_ids=column_ids,
        global_shape=(global_rows, global_columns),
        row_block_names=row_names,
        column_block_names=column_names,
    )


def _local_coordinate_block(
    block: CoordinateBlock,
    candidate_global_rows: NDArray[np.int64],
    original_input_size: int,
) -> tuple[CoordinateBlock, NDArray[np.int64]]:
    selected_global_rows = block.rows(original_input_size)
    position = {
        int(global_row): index for index, global_row in enumerate(selected_global_rows)
    }
    local_rows = np.fromiter(
        (
            local_row
            for local_row, global_row in enumerate(candidate_global_rows)
            if int(global_row) in position
        ),
        dtype=np.int64,
    )
    block_global_ids = np.fromiter(
        (position[int(candidate_global_rows[row])] for row in local_rows),
        dtype=np.int64,
    )
    return (
        CoordinateBlock(
            name=block.name,
            input_index=block.input_index,
            role=block.role,
            value_source=block.value_source,
            selection=tuple(int(row) for row in local_rows),
        ),
        block_global_ids,
    )


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
