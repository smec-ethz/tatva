"""Vectorized colored-demand rules for contribution incidence analysis."""

from __future__ import annotations

import math

import numpy as np
from jax.extend.core import Literal

from tatva.tracer.core.route_fragments import (
    DynamicSliceRouteFragment,
    DynamicUpdateSliceRouteFragment,
    GatherEnvelopeFragment,
    GatherRouteFragment,
    ScatterRouteFragment,
    SelectNRouteFragment,
)
from tatva.tracer.core.routes import (
    DynamicSliceRoute,
    DynamicUpdateSliceRoute,
    GatherRoute,
    ScatterRoute,
    SelectNRoute,
    Shape,
)
from tatva.tracer.core.semantics import TaggedDemandContext
from tatva.tracer.core.tagged import (
    Tagged,
    TaggedDemand,
    active_blocks,
    merge_tagged,
)
from tatva.tracer.helpers import _shape_of


def no_input(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    if ctx.eqn.invars:
        raise ValueError(f"{ctx.eqn.primitive.name} unexpectedly has inputs")
    return ()


def no_op(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    return tuple(None for _ in ctx.eqn.invars)


def _output(ctx: TaggedDemandContext) -> Tagged:
    if len(ctx.output_demands) != 1:
        raise ValueError(f"{ctx.eqn.primitive.name} expected one output")
    return ctx.output_demands[0]


def _fragment_values(
    fragment_rows: np.ndarray,
    values: np.ndarray,
    demanded_rows: np.ndarray,
) -> np.ndarray:
    """Align fragment values with tagged incidences by global output row."""
    order = np.argsort(fragment_rows, kind="stable")
    rows = fragment_rows[order]
    positions = np.searchsorted(rows, demanded_rows)
    if np.any(positions >= rows.size) or np.any(rows[positions] != demanded_rows):
        raise ValueError("route fragment does not cover every demanded output row")
    return values[order[positions]]


def _coords(rows: np.ndarray, shape: Shape) -> np.ndarray:
    if not shape:
        return np.empty((rows.size, 0), dtype=np.int64)
    return np.stack(np.unravel_index(rows, shape), axis=1).astype(np.int64, copy=False)


def _rows(coords: np.ndarray, shape: Shape) -> np.ndarray:
    if not shape:
        return np.zeros(coords.shape[0], dtype=np.int64)
    return np.ravel_multi_index(tuple(coords.T), shape).astype(np.int64, copy=False)


def _inverse_broadcast(
    demand: TaggedDemand,
    *,
    input_shape: Shape,
    output_shape: Shape,
    dimensions: tuple[int, ...] | None = None,
) -> TaggedDemand:
    if input_shape == output_shape and dimensions is None:
        return demand
    if dimensions is None:
        offset = len(output_shape) - len(input_shape)
        dimensions = tuple(range(offset, len(output_shape)))
    if len(dimensions) != len(input_shape):
        raise ValueError("broadcast dimension rank mismatch")

    output_coords = _coords(demand.rows, output_shape)
    input_coords = np.empty((demand.nnz, len(input_shape)), dtype=np.int64)
    for input_axis, output_axis in enumerate(dimensions):
        input_extent = input_shape[input_axis]
        output_extent = output_shape[output_axis]
        if input_extent == output_extent:
            input_coords[:, input_axis] = output_coords[:, output_axis]
        elif input_extent == 1:
            input_coords[:, input_axis] = 0
        else:
            raise ValueError(f"invalid broadcast from {input_shape} to {output_shape}")
    return TaggedDemand(input_shape, _rows(input_coords, input_shape), demand.blocks)


def elementwise(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    output_shape = _shape_of(ctx.eqn.outvars[0])
    return tuple(
        None
        if isinstance(atom, Literal)
        else _inverse_broadcast(
            output,
            input_shape=_shape_of(atom),
            output_shape=output_shape,
        )
        for atom in ctx.eqn.invars
    )


def reshape(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    return (
        None if output is None else output.with_shape(_shape_of(ctx.eqn.invars[0])),
    )


def transpose(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)
    permutation = tuple(int(axis) for axis in ctx.eqn.params["permutation"])
    output_coords = _coords(output.rows, output.shape)
    input_coords = np.empty_like(output_coords)
    for output_axis, input_axis in enumerate(permutation):
        input_coords[:, input_axis] = output_coords[:, output_axis]
    shape = _shape_of(ctx.eqn.invars[0])
    return (TaggedDemand(shape, _rows(input_coords, shape), output.blocks),)


def broadcast_in_dim(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)
    return (
        _inverse_broadcast(
            output,
            input_shape=_shape_of(ctx.eqn.invars[0]),
            output_shape=_shape_of(ctx.eqn.outvars[0]),
            dimensions=tuple(
                int(axis) for axis in ctx.eqn.params["broadcast_dimensions"]
            ),
        ),
    )


def slice_(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)
    shape = _shape_of(ctx.eqn.invars[0])
    starts = np.asarray(ctx.eqn.params["start_indices"], dtype=np.int64)
    strides = np.asarray(
        ctx.eqn.params.get("strides") or (1,) * len(shape), dtype=np.int64
    )
    input_coords = starts + _coords(output.rows, output.shape) * strides
    return (TaggedDemand(shape, _rows(input_coords, shape), output.blocks),)


def tile(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    from tatva.tracer.rules.structural import tile_row_map

    output = _output(ctx)
    if output is None:
        return (None,)
    shape = _shape_of(ctx.eqn.invars[0])
    source_rows = tile_row_map(ctx.eqn).source_rows[output.rows]
    return (TaggedDemand(shape, source_rows, output.blocks),)


def rev(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)
    shape = _shape_of(ctx.eqn.invars[0])
    coords = _coords(output.rows, output.shape)
    for axis in ctx.eqn.params["dimensions"]:
        coords[:, int(axis)] = shape[int(axis)] - 1 - coords[:, int(axis)]
    return (TaggedDemand(shape, _rows(coords, shape), output.blocks),)


def concatenate(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    axis = int(ctx.eqn.params["dimension"])
    coords = _coords(output.rows, output.shape)
    coordinate = coords[:, axis]
    result: list[Tagged] = []
    offset = 0
    for atom in ctx.eqn.invars:
        shape = _shape_of(atom)
        extent = shape[axis]
        keep = (coordinate >= offset) & (coordinate < offset + extent)
        if not np.any(keep):
            result.append(None)
        else:
            local = coords[keep].copy()
            local[:, axis] -= offset
            result.append(TaggedDemand(shape, _rows(local, shape), output.blocks[keep]))
        offset += extent
    return tuple(result)


def _multi_input_row_map(ctx: TaggedDemandContext, row_map) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    operand_indices = row_map.operand_indices[output.rows]
    source_rows = row_map.source_rows[output.rows]
    result: list[Tagged] = []
    for operand_index, atom in enumerate(ctx.eqn.invars):
        keep = (operand_indices == operand_index) & (source_rows >= 0)
        result.append(
            None
            if not np.any(keep)
            else TaggedDemand(_shape_of(atom), source_rows[keep], output.blocks[keep])
        )
    return tuple(result)


def stack(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    from tatva.tracer.rules.structural import _stack_row_map

    return _multi_input_row_map(ctx, _stack_row_map(ctx.eqn))


def pad(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    from tatva.tracer.rules.structural import _pad_row_map

    return _multi_input_row_map(ctx, _pad_row_map(ctx.eqn))


def split(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    from tatva.tracer.rules.structural import _split_row_map

    row_map = _split_row_map(ctx.eqn)
    parts = []
    for index, demand in enumerate(ctx.output_demands):
        if demand is None:
            continue
        parts.append(
            TaggedDemand(
                _shape_of(ctx.eqn.invars[0]),
                row_map.source_rows[index][demand.rows],
                demand.blocks,
            )
        )
    result: Tagged = None
    for part in parts:
        result = merge_tagged(result, part)
    return (result,)


def gather(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    if isinstance(ctx.route, GatherEnvelopeFragment):
        output = ctx.output_demands[0]
        if output is None:
            return tuple(None for _ in ctx.eqn.invars)

        relation = ctx.route.materialize_source_rows()
        positions = np.searchsorted(ctx.route.output_rows, output.rows)
        valid = positions < ctx.route.output_rows.size

        if not np.all(valid):
            raise ValueError("tagged gather output row not covered by envelope")

        if not np.all(ctx.route.output_rows[positions] == output.rows):
            raise ValueError("tagged gather output row not covered by envelope")

        starts = relation.indptr[positions]
        counts = relation.indptr[positions + 1] - starts
        total = int(counts.sum(dtype=np.int64))

        if total == 0:
            operand = None

        else:
            # For each tagged output entry, identify the tagged entry
            # owning each expanded candidate.
            owners = np.repeat(np.arange(output.rows.size, dtype=np.int64), counts)

            # Offset within that entry's candidate list.
            tagged_indptr = np.empty(output.rows.size + 1, dtype=np.int64)
            tagged_indptr[0] = 0
            np.cumsum(counts, out=tagged_indptr[1:])

            local_offsets = np.arange(total, dtype=np.int64) - np.repeat(
                tagged_indptr[:-1], counts
            )
            candidate_positions = starts[owners] + local_offsets
            source_rows = relation.rows[candidate_positions]
            source_blocks = np.repeat(output.blocks, counts)
            operand = TaggedDemand(
                _shape_of(ctx.eqn.invars[0]), source_rows, source_blocks
            )

        result: list[Tagged] = [None] * len(ctx.eqn.invars)
        result[0] = operand

        return tuple(result)

    if isinstance(ctx.route, GatherRouteFragment):
        source_rows = _fragment_values(
            ctx.route.output_rows, ctx.route.source_rows, output.rows
        )
    elif isinstance(ctx.route, GatherRoute):
        source_rows = ctx.route.source_rows[output.rows]
    else:
        raise TypeError("tagged gather demand requires a gather route")
    result: list[Tagged] = [None] * len(ctx.eqn.invars)
    result[0] = output.mapped(_shape_of(ctx.eqn.invars[0]), source_rows)
    return tuple(result)


def _labels_for_targets(
    output: TaggedDemand,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = targets >= 0
    update_ids = np.flatnonzero(valid)
    wanted = targets[valid]
    left = np.searchsorted(output.rows, wanted, side="left")
    right = np.searchsorted(output.rows, wanted, side="right")
    counts = right - left
    keep = counts > 0
    if not np.any(keep):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    update_ids = update_ids[keep]
    left = left[keep]
    counts = counts[keep]
    total = int(counts.sum())
    group_start = np.repeat(np.cumsum(counts) - counts, counts)
    positions = np.repeat(left, counts) + np.arange(total) - group_start
    return np.repeat(update_ids, counts), output.blocks[positions]


def scatter(
    ctx: TaggedDemandContext,
    *,
    needs_operand_at_updates: bool,
) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    if isinstance(ctx.route, ScatterRouteFragment):
        targets = ctx.route.target_rows
        relation_update_rows = ctx.route.update_rows
    elif isinstance(ctx.route, ScatterRoute):
        targets = ctx.route.target_rows
        relation_update_rows = np.arange(targets.size, dtype=np.int64)
    else:
        raise TypeError("tagged scatter demand requires a scatter route")
    valid_targets = targets[targets >= 0]
    if needs_operand_at_updates:
        operand = output
    else:
        keep = ~np.isin(output.rows, np.unique(valid_targets))
        operand = (
            None
            if not np.any(keep)
            else TaggedDemand(output.shape, output.rows[keep], output.blocks[keep])
        )
    relation_rows, update_blocks = _labels_for_targets(output, targets)
    update_rows = relation_update_rows[relation_rows]
    updates = (
        None
        if update_rows.size == 0
        else TaggedDemand(_shape_of(ctx.eqn.invars[2]), update_rows, update_blocks)
    )
    result: list[Tagged] = [None] * len(ctx.eqn.invars)
    result[0] = operand
    result[2] = updates
    return tuple(result)


def scatter_set(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    return scatter(ctx, needs_operand_at_updates=False)


def scatter_accumulate(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    return scatter(ctx, needs_operand_at_updates=True)


def select_n(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    if ctx.route is None:
        result: list[Tagged] = [None] * len(ctx.eqn.invars)
        for index, atom in enumerate(ctx.eqn.invars):
            result[index] = _inverse_broadcast(
                output, input_shape=_shape_of(atom), output_shape=output.shape
            )
        return tuple(result)
    if isinstance(ctx.route, SelectNRouteFragment):
        selected = _fragment_values(
            ctx.route.output_rows, ctx.route.case_indices, output.rows
        )
    elif isinstance(ctx.route, SelectNRoute):
        selected = ctx.route.case_indices[output.rows]
    else:
        raise TypeError("tagged select_n demand requires a select_n route")
    result: list[Tagged] = [None] * len(ctx.eqn.invars)
    for case_index, atom in enumerate(ctx.eqn.invars[1:]):
        keep = selected == case_index
        if not np.any(keep):
            continue
        subset = TaggedDemand(output.shape, output.rows[keep], output.blocks[keep])
        result[case_index + 1] = _inverse_broadcast(
            subset,
            input_shape=_shape_of(atom),
            output_shape=output.shape,
        )
    return tuple(result)


def dynamic_slice(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    if isinstance(ctx.route, DynamicSliceRouteFragment):
        source_rows = _fragment_values(
            ctx.route.output_rows, ctx.route.source_rows, output.rows
        )
    elif isinstance(ctx.route, DynamicSliceRoute):
        source_rows = ctx.route.source_rows[output.rows]
    else:
        raise TypeError("tagged dynamic_slice demand requires a dynamic-slice route")
    result: list[Tagged] = [None] * len(ctx.eqn.invars)
    result[0] = output.mapped(_shape_of(ctx.eqn.invars[0]), source_rows)
    return tuple(result)


def dynamic_update_slice(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    if isinstance(ctx.route, DynamicUpdateSliceRouteFragment):
        targets = ctx.route.target_rows
        relation_update_rows = ctx.route.update_rows
    elif isinstance(ctx.route, DynamicUpdateSliceRoute):
        targets = ctx.route.target_rows
        relation_update_rows = np.arange(targets.size, dtype=np.int64)
    else:
        raise TypeError("tagged dynamic_update_slice demand requires a dynamic route")
    keep = ~np.isin(output.rows, np.unique(targets[targets >= 0]))
    operand = (
        None
        if not np.any(keep)
        else TaggedDemand(output.shape, output.rows[keep], output.blocks[keep])
    )
    relation_rows, update_blocks = _labels_for_targets(output, targets)
    update_rows = relation_update_rows[relation_rows]
    updates = (
        None
        if update_rows.size == 0
        else TaggedDemand(_shape_of(ctx.eqn.invars[1]), update_rows, update_blocks)
    )
    result: list[Tagged] = [None] * len(ctx.eqn.invars)
    result[0] = operand
    result[1] = updates
    return tuple(result)


def reduction(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)
    input_shape = _shape_of(ctx.eqn.invars[0])
    reduced = tuple(sorted(int(axis) for axis in ctx.eqn.params["axes"]))
    kept = tuple(axis for axis in range(len(input_shape)) if axis not in reduced)
    reduced_shape = tuple(input_shape[axis] for axis in reduced)
    expansion = int(math.prod(reduced_shape))
    out_coords = _coords(output.rows, output.shape)
    base = np.zeros((output.nnz, len(input_shape)), dtype=np.int64)
    if kept:
        base[:, kept] = out_coords
    coords = np.repeat(base, expansion, axis=0)
    if reduced:
        combinations = np.stack(
            np.unravel_index(np.arange(expansion), reduced_shape), axis=1
        )
        coords[:, reduced] = np.tile(combinations, (output.nnz, 1))
    return (
        TaggedDemand(
            input_shape,
            _rows(coords, input_shape),
            np.repeat(output.blocks, expansion),
        ),
    )


def cumulative(
    ctx: TaggedDemandContext,
) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])

    if output.shape != input_shape:
        raise ValueError("cumulative tagged rule requires input/output shapes to match")

    axis = int(ctx.eqn.params["axis"])
    if axis < 0:
        axis += len(input_shape)

    if axis < 0 or axis >= len(input_shape):
        raise ValueError(f"cumulative axis {axis} outside rank {len(input_shape)}")

    reverse = bool(ctx.eqn.params.get("reverse", False))

    extent = int(input_shape[axis])
    if extent == 0 or output.nnz == 0:
        return (None,)

    # C-order stride of the cumulative axis.
    #
    # For a flattened row:
    #
    #   row = fiber_base + axis_position * axis_stride
    #
    # where fiber_base identifies all coordinates except `axis`.
    axis_stride = int(math.prod(input_shape[axis + 1 :]))
    positions = (output.rows // axis_stride) % extent
    fiber_bases = output.rows - positions * axis_stride

    # Collapse all tagged output entries belonging to the same
    # (fiber, block).
    #
    # This is the important optimization: the union of cumulative
    # prefixes/suffixes can be represented by one boundary per group.
    order = np.lexsort((output.blocks, fiber_bases))

    sorted_bases = fiber_bases[order]
    sorted_blocks = output.blocks[order]
    sorted_positions = positions[order]

    group_start_mask = np.ones(sorted_bases.size, dtype=bool)
    group_start_mask[1:] = (sorted_bases[1:] != sorted_bases[:-1]) | (
        sorted_blocks[1:] != sorted_blocks[:-1]
    )
    group_starts = np.flatnonzero(group_start_mask)
    group_bases = sorted_bases[group_starts]
    group_blocks = sorted_blocks[group_starts]

    if reverse:
        # Union of suffixes:
        #
        #   [p1, extent) U [p2, extent)
        #       = [min(p1, p2), extent)
        boundaries = np.minimum.reduceat(sorted_positions, group_starts)
        first_axis_values = boundaries
        counts = extent - boundaries

    else:
        # Union of prefixes:
        #
        #   [0, p1] U [0, p2]
        #       = [0, max(p1, p2)]
        boundaries = np.maximum.reduceat(sorted_positions, group_starts)
        first_axis_values = np.zeros_like(boundaries)
        counts = boundaries + 1

    counts = counts.astype(np.int64, copy=False)

    total = int(counts.sum(dtype=np.int64))
    if total == 0:
        return (None,)

    # CSR-like offsets for each (fiber, block) group's interval.
    indptr = np.empty(counts.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    # Expand all groups in one vectorized operation.
    owners = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    local_offsets = np.arange(total, dtype=np.int64) - np.repeat(indptr[:-1], counts)
    axis_values = first_axis_values[owners] + local_offsets
    input_rows = group_bases[owners] + axis_values * axis_stride
    input_blocks = group_blocks[owners]

    return (TaggedDemand(input_shape, input_rows, input_blocks),)


def dot_general(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None, None)
    from tatva.tracer.rules.dot import dot_general_map

    mapping = dot_general_map(ctx.eqn)
    lhs = mapping.lhs_rows[output.rows]
    rhs = mapping.rhs_rows[output.rows]
    width = lhs.shape[1]
    blocks = np.repeat(output.blocks, width)
    return (
        TaggedDemand(_shape_of(ctx.eqn.invars[0]), lhs.ravel(), blocks),
        TaggedDemand(_shape_of(ctx.eqn.invars[1]), rhs.ravel(), blocks),
    )


def sort(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    blocks = active_blocks(ctx.output_demands)
    return tuple(
        None
        if isinstance(atom, Literal)
        else TaggedDemand.full(_shape_of(atom), blocks)
        for atom in ctx.eqn.invars
    )


def _batch_projection(demand: TaggedDemand, batch_shape: Shape) -> TaggedDemand:
    local_size = int(math.prod(demand.shape[len(batch_shape) :]))
    return TaggedDemand(batch_shape, demand.rows // local_size, demand.blocks)


def _expand_batch(demand: TaggedDemand, shape: Shape) -> TaggedDemand:
    local_size = int(math.prod(shape[len(demand.shape) :]))
    offsets = np.arange(local_size, dtype=np.int64)
    rows = np.repeat(demand.rows * local_size, local_size) + np.tile(
        offsets, demand.nnz
    )
    return TaggedDemand(shape, rows, np.repeat(demand.blocks, local_size))


def _broadcast_batch_to_operand(
    output: TaggedDemand,
    *,
    source_shape: Shape,
    target_batch_shape: Shape,
) -> TaggedDemand:
    """Map output tags through right-aligned batch broadcasting."""
    from tatva.tracer.rules.linalg import broadcast_batch_coordinates

    source_batch_shape = source_shape[:-2]
    output_item_size = int(math.prod(output.shape[len(target_batch_shape) :]))
    source_item_size = int(math.prod(source_shape[len(source_batch_shape) :]))
    output_batch_rows = output.rows // output_item_size
    source_batch_rows = broadcast_batch_coordinates(
        source_batch_shape, target_batch_shape
    )[output_batch_rows]
    offsets = np.arange(source_item_size, dtype=np.int64)
    return TaggedDemand(
        source_shape,
        np.repeat(source_batch_rows * source_item_size, source_item_size)
        + np.tile(offsets, output.nnz),
        np.repeat(output.blocks, source_item_size),
    )


def triangular_solve(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    output = _output(ctx)
    if output is None:
        return (None, None)
    a_shape = _shape_of(ctx.eqn.invars[0])
    b_shape = _shape_of(ctx.eqn.invars[1])
    out_shape = _shape_of(ctx.eqn.outvars[0])
    if len(a_shape) < 2 or len(b_shape) < 2 or len(out_shape) < 2:
        blocks = output.block_ids
        return TaggedDemand.full(a_shape, blocks), TaggedDemand.full(b_shape, blocks)
    try:
        return (
            _broadcast_batch_to_operand(
                output, source_shape=a_shape, target_batch_shape=out_shape[:-2]
            ),
            _broadcast_batch_to_operand(
                output, source_shape=b_shape, target_batch_shape=out_shape[:-2]
            ),
        )
    except ValueError:
        blocks = output.block_ids
        return TaggedDemand.full(a_shape, blocks), TaggedDemand.full(b_shape, blocks)


def lu(ctx: TaggedDemandContext) -> tuple[Tagged, ...]:
    shape = _shape_of(ctx.eqn.invars[0])
    blocks = active_blocks(ctx.output_demands)
    active = [demand for demand in ctx.output_demands if demand is not None]
    if not active:
        return (None,)
    if len(shape) < 2:
        return (TaggedDemand.full(shape, blocks),)
    batch_shape = shape[:-2]
    batch: Tagged = None
    for demand in active:
        assert demand is not None
        batch = merge_tagged(batch, _batch_projection(demand, batch_shape))
    assert batch is not None
    return (_expand_batch(batch, shape),)
