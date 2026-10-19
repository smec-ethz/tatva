from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    RuleContext,
    conservative_demand,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    Demand,
    TensorDemand,
    _FullAxis,
    merge_demands,
)
from tatva.tracer.program.dependencies import DependencySet, InteractionGraph
from tatva.tracer.rules import opaque


def _size(shape: Shape) -> int:
    return int(np.prod(shape, dtype=np.int64))


def broadcast_batch_coordinates(
    source_shape: Shape,
    target_shape: Shape,
) -> NDArray[np.int64]:
    """Map each flattened target batch item to its broadcast source item."""
    if len(source_shape) > len(target_shape):
        raise ValueError("source batch rank exceeds target batch rank")

    padded = (1,) * (len(target_shape) - len(source_shape)) + source_shape
    if any(source not in (1, target) for source, target in zip(padded, target_shape)):
        raise ValueError(
            f"cannot broadcast batch shape {source_shape} to {target_shape}"
        )

    target_size = _size(target_shape)
    if target_size == 0:
        return np.empty(0, dtype=np.int64)
    if not source_shape:
        return np.zeros(target_size, dtype=np.int64)

    coords = np.unravel_index(np.arange(target_size, dtype=np.int64), target_shape)
    source_coords = tuple(
        np.zeros(target_size, dtype=np.int64) if dimension == 1 else coordinate
        for dimension, coordinate in zip(padded, coords)
    )[-len(source_shape) :]
    return np.ravel_multi_index(source_coords, source_shape).astype(np.int64)


def batch_item_unions(deps: DependencySet, *, item_rank: int = 2) -> DependencySet:
    """Union scalar dependencies within each trailing matrix-like batch item."""
    if len(deps.shape) < item_rank:
        raise ValueError(f"shape {deps.shape} has fewer than {item_rank} item axes")
    batch_shape = deps.shape[:-item_rank]
    batch_size = _size(batch_shape)
    item_size = _size(deps.shape[-item_rank:])
    if batch_size == 0:
        return DependencySet.empty(batch_shape, deps.csr.shape[1])
    rows = np.repeat(np.arange(batch_size, dtype=np.int64), item_size)
    cols = np.arange(batch_size * item_size, dtype=np.int64)
    aggregate = sps.csr_matrix(
        (np.ones(cols.size, dtype=np.int8), (rows, cols)),
        shape=(batch_size, cols.size),
    )
    return DependencySet((aggregate @ deps.csr).astype(bool).tocsr(), batch_shape)


def _broadcast_batch_dependencies(
    batch_deps: DependencySet,
    target_shape: Shape,
) -> DependencySet:
    mapping = broadcast_batch_coordinates(batch_deps.shape, target_shape)
    return DependencySet(batch_deps.csr[mapping], target_shape)


def expand_batch_dependencies(
    batch_deps: DependencySet,
    output_shape: Shape,
) -> DependencySet:
    """Expand per-batch unions over every scalar in an output batch item."""
    batch_shape = batch_deps.shape
    if output_shape[: len(batch_shape)] != batch_shape:
        raise ValueError(
            f"output shape {output_shape} does not begin with {batch_shape}"
        )
    item_size = _size(output_shape[len(batch_shape) :])
    return DependencySet(
        batch_deps.csr[np.repeat(np.arange(_size(batch_shape)), item_size)],
        output_shape,
    )


def add_batch_self(acc: InteractionGraph, batch_deps: DependencySet) -> None:
    rows = np.arange(batch_deps.csr.shape[0], dtype=np.int64)
    acc.add_paired_cross(batch_deps, rows, batch_deps, rows)


def add_batch_cross(
    acc: InteractionGraph,
    lhs: DependencySet,
    rhs: DependencySet,
) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"batch shapes do not match: {lhs.shape} != {rhs.shape}")
    rows = np.arange(lhs.csr.shape[0], dtype=np.int64)
    acc.add_paired_cross(lhs, rows, rhs, rows)


def _project_batch_demand(
    demand: Demand,
    *,
    batch_shape: tuple[int, ...],
) -> Demand:
    if demand is None:
        return None

    batch_rank = len(batch_shape)

    if batch_rank == 0:
        return TensorDemand.full(())

    if demand.shape[:batch_rank] != batch_shape:
        raise ValueError(
            f"demand shape {demand.shape} does not start with batch shape {batch_shape}"
        )

    return TensorDemand.from_axes(
        batch_shape,
        demand.axes[:batch_rank],
    )


def _project_broadcast_batch_demand(
    demand: Demand,
    *,
    source_shape: Shape,
    target_shape: Shape,
) -> Demand:
    if demand is None:
        return None
    if demand.shape[: len(target_shape)] != target_shape:
        raise ValueError(
            f"demand shape {demand.shape} does not begin with {target_shape}"
        )
    target = TensorDemand.from_axes(target_shape, demand.axes[: len(target_shape)])
    assert target is not None
    rows = target.rows()
    source_rows = broadcast_batch_coordinates(source_shape, target_shape)[rows]
    return TensorDemand.from_rows_hull(source_shape, source_rows)


def _expand_batch_demand(
    batch_demand: Demand,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> Demand:
    if batch_demand is None:
        return None

    batch_rank = len(batch_shape)

    if shape[:batch_rank] != batch_shape:
        raise ValueError(f"shape {shape} does not begin with batch shape {batch_shape}")

    return TensorDemand.from_axes(
        shape,
        batch_demand.axes + tuple(_FullAxis() for _ in range(len(shape) - batch_rank)),
    )


def triangular_solve_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    """Preserve batch locality through standard right-aligned broadcasting."""
    if len(ctx.eqn.invars) != 2 or len(ctx.output_demands) != 1:
        return conservative_demand(ctx)

    output_demand = ctx.output_demands[0]
    if output_demand is None:
        return (None, None)

    a_shape = _shape_of(ctx.eqn.invars[0])
    b_shape = _shape_of(ctx.eqn.invars[1])
    out_shape = _shape_of(ctx.eqn.outvars[0])

    if len(a_shape) < 2 or len(b_shape) < 2 or len(out_shape) < 2:
        return conservative_demand(ctx)

    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    out_batch = out_shape[:-2]

    try:
        if tuple(np.broadcast_shapes(a_batch, b_batch)) != out_batch:
            return conservative_demand(ctx)
    except ValueError:
        return conservative_demand(ctx)

    a_demand = _project_broadcast_batch_demand(
        output_demand,
        source_shape=a_batch,
        target_shape=out_batch,
    )
    b_demand = _project_broadcast_batch_demand(
        output_demand,
        source_shape=b_batch,
        target_shape=out_batch,
    )

    return (
        _expand_batch_demand(
            a_demand,
            shape=a_shape,
            batch_shape=a_batch,
        ),
        _expand_batch_demand(
            b_demand,
            shape=b_shape,
            batch_shape=b_batch,
        ),
    )


def lu_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    if len(ctx.eqn.invars) != 1:
        raise ValueError(f"lu expected one input, got {len(ctx.eqn.invars)}")

    if not any(demand is not None for demand in ctx.output_demands):
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])

    if len(input_shape) < 2:
        return conservative_demand(ctx)

    batch_shape = input_shape[:-2]
    batch_rank = len(batch_shape)
    batch_demand: Demand = None

    for output_demand in ctx.output_demands:
        if output_demand is None:
            continue

        output_shape = output_demand.shape
        if output_shape[:batch_rank] != batch_shape:
            return conservative_demand(ctx)

        if batch_rank == 0:
            projected = TensorDemand.full(())
        else:
            projected = TensorDemand.from_axes(
                batch_shape,
                output_demand.axes[:batch_rank],
            )

        batch_demand = merge_demands(batch_demand, projected)

    if batch_demand is None:
        return (None,)

    return (
        _expand_batch_demand(
            batch_demand,
            shape=input_shape,
            batch_shape=batch_shape,
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedLu:
    batch_deps: DependencySet
    output_shapes: tuple[Shape, ...]


def prepare_lu(ctx: RuleContext) -> PreparedLu | opaque.OpaqueData:
    if len(ctx.input_deps) != 1 or not ctx.eqn.outvars:
        return opaque.prepare_opaque(ctx)
    deps = ctx.input_deps[0]
    if len(deps.shape) < 2:
        return opaque.prepare_opaque(ctx)
    return PreparedLu(
        batch_deps=batch_item_unions(deps),
        output_shapes=tuple(_shape_of(output) for output in ctx.eqn.outvars),
    )


def lu_dependencies(
    ctx: RuleContext,
    prepared: PreparedLu | opaque.OpaqueData,
) -> tuple[DependencySet, ...]:
    if not isinstance(prepared, PreparedLu):
        return opaque.opaque_dependencies(ctx, prepared)
    # Only the floating LU factors are differentiable.  Pivot/permutation
    # integer outputs deliberately carry no dependency relation.
    return (
        expand_batch_dependencies(prepared.batch_deps, prepared.output_shapes[0]),
        *(
            DependencySet.empty(shape, ctx.n_dofs)
            for shape in prepared.output_shapes[1:]
        ),
    )


def lu_hessian(
    ctx: RuleContext,
    prepared: PreparedLu | opaque.OpaqueData,
    acc: InteractionGraph,
) -> None:
    if isinstance(prepared, PreparedLu):
        add_batch_self(acc, prepared.batch_deps)
    else:
        opaque.opaque_nonlinear_hessian(ctx, prepared, acc)


@dataclass(frozen=True, slots=True)
class PreparedTriangularSolve:
    a_batch: DependencySet
    b_batch: DependencySet
    output_shape: Shape


def prepare_triangular_solve(
    ctx: RuleContext,
) -> PreparedTriangularSolve | opaque.OpaqueData:
    if len(ctx.input_deps) != 2 or len(ctx.eqn.outvars) != 1:
        return opaque.prepare_opaque(ctx)
    a_deps, b_deps = ctx.input_deps
    output_shape = _shape_of(ctx.eqn.outvars[0])
    if min(len(a_deps.shape), len(b_deps.shape), len(output_shape)) < 2:
        return opaque.prepare_opaque(ctx)
    a_batch_shape = a_deps.shape[:-2]
    b_batch_shape = b_deps.shape[:-2]
    output_batch_shape = output_shape[:-2]
    try:
        if (
            tuple(np.broadcast_shapes(a_batch_shape, b_batch_shape))
            != output_batch_shape
        ):
            return opaque.prepare_opaque(ctx)
        a_batch = _broadcast_batch_dependencies(
            batch_item_unions(a_deps), output_batch_shape
        )
        b_batch = _broadcast_batch_dependencies(
            batch_item_unions(b_deps), output_batch_shape
        )
    except ValueError:
        return opaque.prepare_opaque(ctx)
    return PreparedTriangularSolve(a_batch, b_batch, output_shape)


def triangular_solve_dependencies(
    ctx: RuleContext,
    prepared: PreparedTriangularSolve | opaque.OpaqueData,
) -> tuple[DependencySet, ...]:
    if not isinstance(prepared, PreparedTriangularSolve):
        return opaque.opaque_dependencies(ctx, prepared)
    batch = DependencySet(
        (prepared.a_batch.csr + prepared.b_batch.csr).astype(bool).tocsr(),
        prepared.a_batch.shape,
    )
    return (expand_batch_dependencies(batch, prepared.output_shape),)


def triangular_solve_hessian(
    ctx: RuleContext,
    prepared: PreparedTriangularSolve | opaque.OpaqueData,
    acc: InteractionGraph,
) -> None:
    if not isinstance(prepared, PreparedTriangularSolve):
        opaque.opaque_nonlinear_hessian(ctx, prepared, acc)
        return
    add_batch_self(acc, prepared.a_batch)
    add_batch_cross(acc, prepared.a_batch, prepared.b_batch)


DERIVATIVES_LU = DerivativeRule(prepare_lu, lu_dependencies, lu_hessian)
DERIVATIVES_TRIANGULAR_SOLVE = DerivativeRule(
    prepare_triangular_solve,
    triangular_solve_dependencies,
    triangular_solve_hessian,
)
