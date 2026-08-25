"""Slice-local semantics for ``lax.sort_p``.

A sort couples every entry along its sorted axis, but independent coordinates
on the remaining axes remain independent.  The rules in this module preserve
that distinction for runtime demand, tagged contribution incidence, and
conservative derivative sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Literal

from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    RuleContext,
    TaggedDemandContext,
)
from tatva.tracer.core.tagged import Tagged, TaggedDemand, merge_tagged
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand, _FullAxis, merge_demands
from tatva.tracer.program.dependencies import DependencySet, InteractionGraph
from tatva.tracer.rules.linalg import add_batch_self


def _size(shape: Shape) -> int:
    return int(np.prod(shape, dtype=np.int64))


def _sort_shape_and_dimension(ctx) -> tuple[Shape, int]:
    if not ctx.eqn.invars:
        raise ValueError("sort expected at least one operand")

    shape = _shape_of(ctx.eqn.invars[0])
    dimension = int(ctx.eqn.params["dimension"])
    if dimension < 0:
        dimension += len(shape)
    if dimension < 0 or dimension >= len(shape):
        raise ValueError(f"sort dimension {dimension} is invalid for shape {shape}")

    atoms = (*ctx.eqn.invars, *ctx.eqn.outvars)
    if any(_shape_of(atom) != shape for atom in atoms):
        raise ValueError("sort operands and results must have identical shapes")

    return shape, dimension


def _complete_slice_demand(demand: Demand, *, dimension: int) -> Demand:
    if demand is None:
        return None
    axes = list(demand.axes)
    axes[dimension] = _FullAxis()
    return TensorDemand.from_axes(demand.shape, tuple(axes))


def sort_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    """Demand complete sort slices without widening independent axes."""
    _, dimension = _sort_shape_and_dimension(ctx)
    combined: Demand = None
    for output in ctx.output_demands:
        combined = merge_demands(
            combined,
            _complete_slice_demand(output, dimension=dimension),
        )

    return tuple(
        None if isinstance(atom, Literal) else combined for atom in ctx.eqn.invars
    )


def _complete_tagged_slices(
    demand: TaggedDemand,
    *,
    dimension: int,
) -> TaggedDemand:
    shape = demand.shape
    extent = shape[dimension]
    coordinates = np.stack(np.unravel_index(demand.rows, shape), axis=1)
    expanded = np.repeat(coordinates, extent, axis=0)
    expanded[:, dimension] = np.tile(np.arange(extent, dtype=np.int64), demand.nnz)
    rows = np.ravel_multi_index(tuple(expanded.T), shape).astype(np.int64, copy=False)
    return TaggedDemand(shape, rows, np.repeat(demand.blocks, extent))


def sort_tagged_demand(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    """Attach each block only to the complete sort slices that it reaches."""
    _, dimension = _sort_shape_and_dimension(ctx)
    combined: Tagged = None
    for output in ctx.output_demands:
        if output is not None:
            combined = merge_tagged(
                combined,
                _complete_tagged_slices(output, dimension=dimension),
            )

    return tuple(
        None if isinstance(atom, Literal) else combined for atom in ctx.eqn.invars
    )


def _slice_rows(shape: Shape, *, dimension: int) -> tuple[Shape, np.ndarray]:
    """Map every tensor scalar row to its independent sort-slice row."""
    slice_shape = shape[:dimension] + shape[dimension + 1 :]
    n_entries = _size(shape)
    if not slice_shape:
        return slice_shape, np.zeros(n_entries, dtype=np.int64)

    coordinates = np.unravel_index(np.arange(n_entries, dtype=np.int64), shape)
    slice_coordinates = coordinates[:dimension] + coordinates[dimension + 1 :]
    rows = np.ravel_multi_index(slice_coordinates, slice_shape).astype(
        np.int64, copy=False
    )
    return slice_shape, rows


@dataclass(frozen=True, slots=True)
class PreparedSort:
    slice_dependencies: DependencySet
    scalar_to_slice: np.ndarray
    output_shapes: tuple[Shape, ...]


def prepare_sort(ctx: RuleContext) -> PreparedSort:
    shape, dimension = _sort_shape_and_dimension(ctx)
    slice_shape, scalar_to_slice = _slice_rows(shape, dimension=dimension)
    n_entries = _size(shape)
    n_slices = _size(slice_shape)
    aggregate = sps.csr_matrix(
        (
            np.ones(n_entries, dtype=np.int8),
            (scalar_to_slice, np.arange(n_entries, dtype=np.int64)),
        ),
        shape=(n_slices, n_entries),
    )

    combined = sps.csr_matrix((n_slices, ctx.n_symbols), dtype=bool)
    for dependency in ctx.input_deps:
        if dependency.shape != shape:
            raise ValueError("sort dependency shape differs from operand shape")
        combined = (combined + aggregate @ dependency.csr).astype(bool).tocsr()

    return PreparedSort(
        slice_dependencies=DependencySet(combined, slice_shape),
        scalar_to_slice=scalar_to_slice,
        output_shapes=tuple(_shape_of(atom) for atom in ctx.eqn.outvars),
    )


def sort_dependencies(
    ctx: RuleContext,
    prepared: PreparedSort,
) -> tuple[DependencySet, ...]:
    return tuple(
        DependencySet(
            prepared.slice_dependencies.csr[prepared.scalar_to_slice],
            shape,
        )
        for shape in prepared.output_shapes
    )


def sort_hessian(
    ctx: RuleContext,
    prepared: PreparedSort,
    acc: InteractionGraph,
) -> None:
    add_batch_self(acc, prepared.slice_dependencies)


DERIVATIVES_SORT = DerivativeRule(
    prepare_sort,
    sort_dependencies,
    sort_hessian,
)
