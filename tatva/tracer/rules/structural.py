from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    OperationSemantics,
    no_hessian,
    no_prepare,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    AxisSubset,
    Demand,
    TensorDemand,
    _axis_from_indices,
    _axis_from_range,
    _FullAxis,
    _RangeAxis,
    axis_indices,
)
from tatva.tracer.program.dependencies import DependencySet

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext


def reshape_like_dependencies(
    ctx: RuleContext,
    prepared: None,
) -> tuple[DependencySet, ...]:
    dep = ctx.input_deps[0]
    output_shape = _shape_of(ctx.eqn.outvars[0])

    return (dep.reshape(*output_shape),)


@dataclass(frozen=True)
class UnaryRowMap:
    source_rows: NDArray
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class MultiInputRowMap:
    operand_indices: NDArray
    source_rows: NDArray
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class MultiOutputUnaryMap:
    source_rows: tuple[NDArray[np.int64], ...]
    output_shapes: tuple[tuple[int, ...], ...]


def unary_routed_dependencies(
    ctx: RuleContext,
    prepared: UnaryRowMap,
) -> tuple[DependencySet, ...]:
    if len(ctx.input_deps) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have one input and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    source = ctx.input_deps[0]

    return (
        DependencySet(source.csr[prepared.source_rows], _shape_of(ctx.eqn.outvars[0])),
    )


def multi_input_routed_dependencies(
    ctx: RuleContext,
    prepared: MultiInputRowMap,
) -> tuple[DependencySet, ...]:
    if len(ctx.input_deps) == 0:
        raise ValueError(f"{ctx.eqn.primitive.name} expected at least one input")

    if prepared.operand_indices.shape != prepared.source_rows.shape:
        raise ValueError("operand_indices and source_rows must have the same shape")

    stacked = sps.vstack([dep.csr for dep in ctx.input_deps], format="csr")
    sizes = np.asarray([dep.csr.shape[0] for dep in ctx.input_deps], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(sizes[:-1])))
    global_rows = offsets[prepared.operand_indices] + prepared.source_rows

    return (DependencySet(stacked[global_rows], prepared.output_shape),)


def multi_output_unary_routed_dependencies(
    ctx: RuleContext,
    prepared: MultiOutputUnaryMap,
) -> tuple[DependencySet, ...]:
    source = ctx.input_deps[0]

    return tuple(
        DependencySet(source.csr[rows], shape)
        for rows, shape in zip(prepared.source_rows, prepared.output_shapes)
    )


def prepare_broadcast(ctx: RuleContext) -> UnaryRowMap:
    eqn = ctx.eqn

    if len(ctx.input_deps) != 1 or len(eqn.outvars) != 1:
        raise ValueError("broadcast_in_dim expects one input and one output")

    input_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])
    broadcast_dims = tuple(eqn.params["broadcast_dimensions"])

    source_rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(
        input_shape
    )

    expanded_shape = [1] * len(output_shape)

    for input_axis, output_axis in enumerate(broadcast_dims):
        expanded_shape[output_axis] = input_shape[input_axis]

    source_rows = np.broadcast_to(
        source_rows.reshape(expanded_shape), output_shape
    ).ravel()

    return UnaryRowMap(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_transpose(ctx: RuleContext) -> UnaryRowMap:
    eqn = ctx.eqn

    input_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])

    rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(input_shape)

    source_rows = np.transpose(rows, axes=eqn.params["permutation"]).ravel()

    return UnaryRowMap(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def slice_row_map(
    eqn: JaxprEqn,
) -> UnaryRowMap:
    input_shape = _shape_of(eqn.invars[0])
    output_shape = _shape_of(eqn.outvars[0])

    starts = eqn.params["start_indices"]
    limits = eqn.params["limit_indices"]
    strides = eqn.params["strides"]

    if strides is None:
        strides = (1,) * len(starts)

    rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(input_shape)

    slices = tuple(
        slice(start, limit, stride)
        for start, limit, stride in zip(starts, limits, strides)
    )

    return UnaryRowMap(
        source_rows=rows[slices].ravel(),
        output_shape=output_shape,
    )


def prepare_slice(ctx: RuleContext) -> UnaryRowMap:
    eqn = ctx.eqn
    return slice_row_map(eqn)


def prepare_rev(ctx: RuleContext) -> UnaryRowMap:
    eqn = ctx.eqn

    input_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])

    rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(input_shape)

    source_rows = np.flip(rows, axis=tuple(eqn.params["dimensions"])).ravel()

    return UnaryRowMap(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_concatenate(ctx: RuleContext) -> MultiInputRowMap:
    eqn = ctx.eqn

    axis = int(eqn.params["dimension"])
    output_shape = _shape_of(eqn.outvars[0])

    operand_parts = []
    source_parts = []

    for operand_index, dep in enumerate(ctx.input_deps):
        shape = dep.shape
        size = int(np.prod(shape))

        operand_parts.append(np.full(shape, operand_index, dtype=np.int64))

        source_parts.append(np.arange(size, dtype=np.int64).reshape(shape))

    return MultiInputRowMap(
        operand_indices=np.concatenate(operand_parts, axis=axis).ravel(),
        source_rows=np.concatenate(source_parts, axis=axis).ravel(),
        output_shape=output_shape,
    )


def _stack_row_map(
    eqn: JaxprEqn,
) -> MultiInputRowMap:
    axis = int(eqn.params["axis"])
    source_shapes = [_shape_of(atom) for atom in eqn.invars]
    output_shape = _shape_of(eqn.outvars[0])

    operand_parts = []
    source_parts = []

    for operand_index, shape in enumerate(source_shapes):
        size = int(np.prod(shape))
        operand_parts.append(np.full(shape, operand_index, dtype=np.int64))
        source_parts.append(np.arange(size, dtype=np.int64).reshape(shape))

    return MultiInputRowMap(
        operand_indices=np.stack(operand_parts, axis=axis).ravel(),
        source_rows=np.stack(source_parts, axis=axis).ravel(),
        output_shape=output_shape,
    )


def prepare_stack(ctx: RuleContext) -> MultiInputRowMap:
    if len(ctx.input_deps) == 0:
        raise ValueError("stack expects at least one input")

    return _stack_row_map(ctx.eqn)


def demand_stack(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    return _multi_input_row_map_demand(ctx, _stack_row_map(ctx.eqn))


def _pad_row_map(
    eqn: JaxprEqn,
) -> MultiInputRowMap:
    config = eqn.params["padding_config"]
    source_shape = _shape_of(eqn.invars[0])

    output_shape = _shape_of(eqn.outvars[0])
    n_output = int(np.prod(output_shape))
    output_rows = np.arange(n_output, dtype=np.int64)
    output_coords = np.stack(
        np.unravel_index(output_rows, output_shape),
        axis=1,
    )

    source_coords = np.zeros((n_output, len(source_shape)), dtype=np.int64)
    valid_source = np.ones(n_output, dtype=bool)

    for axis, (low, _high, interior) in enumerate(config):
        stride = interior + 1
        shifted = output_coords[:, axis] - low

        valid_source &= shifted >= 0
        valid_source &= shifted % stride == 0

        source_coord = shifted // stride

        valid_source &= source_coord >= 0
        valid_source &= source_coord < source_shape[axis]

        source_coords[:, axis] = source_coord

    operand_indices = np.ones(n_output, dtype=np.int64)

    # Padding value is scalar -> source row 0.
    source_rows = np.zeros(n_output, dtype=np.int64)

    if np.any(valid_source):
        operand_indices[valid_source] = 0

        source_rows[valid_source] = np.ravel_multi_index(
            tuple(source_coords[valid_source].T), source_shape
        )

    return MultiInputRowMap(
        operand_indices=operand_indices,
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_pad(ctx: RuleContext) -> MultiInputRowMap:
    if len(ctx.input_deps) != 2:
        raise ValueError("pad expects source and padding-value inputs")

    eqn = ctx.eqn
    return _pad_row_map(eqn)


def demand_pad(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    return _multi_input_row_map_demand(ctx, _pad_row_map(ctx.eqn))


def _split_row_map(
    eqn: JaxprEqn,
) -> MultiOutputUnaryMap:
    axis = int(eqn.params["axis"])
    sizes = tuple(int(x) for x in eqn.params["sizes"])
    source_shape = _shape_of(eqn.invars[0])

    index_tensor = np.arange(int(np.prod(source_shape)), dtype=np.int64).reshape(
        source_shape
    )

    routes: list[NDArray[np.int64]] = []
    shapes: list[tuple[int, ...]] = []

    offset = 0

    for outvar, size in zip(eqn.outvars, sizes):
        rows = np.take(
            index_tensor, np.arange(offset, offset + size), axis=axis
        ).ravel()

        routes.append(rows)
        shapes.append(_shape_of(outvar))

        offset += size

    return MultiOutputUnaryMap(source_rows=tuple(routes), output_shapes=tuple(shapes))


def prepare_split(ctx: RuleContext) -> MultiOutputUnaryMap:
    if len(ctx.input_deps) != 1:
        raise ValueError("split expects one input")

    return _split_row_map(ctx.eqn)


def demand_split(ctx: DemandContext) -> tuple[Demand, ...]:
    return _multi_output_unary_row_map_demand(ctx, _split_row_map(ctx.eqn))


# --------------------------------
# Demand rules
# --------------------------------


def demand_reshape_squeeze(ctx: DemandContext) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)
    rows = output.rows()
    return (TensorDemand.from_rows_hull(_shape_of(ctx.eqn.invars[0]), rows),)


def demand_transpose(ctx: DemandContext) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    permutation = tuple(int(axis) for axis in ctx.eqn.params["permutation"])
    input_axes: list[AxisSubset | None] = [None] * len(permutation)

    for output_axis, input_axis in enumerate(permutation):
        input_axes[input_axis] = output.axes[output_axis]

    if any(axis is None for axis in input_axes):
        raise RuntimeError("invalid transpose permutation")

    return (
        TensorDemand.from_axes(
            _shape_of(ctx.eqn.invars[0]),
            tuple(axis for axis in input_axes if axis is not None),
        ),
    )


def demand_broadcast_in_dim(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])
    output_shape = _shape_of(ctx.eqn.outvars[0])
    dimensions = tuple(int(axis) for axis in ctx.eqn.params["broadcast_dimensions"])

    if len(dimensions) != len(input_shape):
        raise ValueError("broadcast_dimensions rank mismatch")

    if not input_shape:
        return (TensorDemand.full(()),)

    input_axes = []

    for input_axis, output_axis in enumerate(dimensions):
        input_extent = input_shape[input_axis]
        output_extent = output_shape[output_axis]

        if input_extent == output_extent:
            input_axes.append(output.axes[output_axis])
        elif input_extent == 1:
            input_axes.append(_FullAxis())
        else:
            raise ValueError("invalid broadcast_in_dim geometry")

    return (
        TensorDemand.from_axes(
            input_shape,
            tuple(input_axes),
        ),
    )


def demand_slice(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])
    output_shape = _shape_of(ctx.eqn.outvars[0])
    starts = tuple(int(x) for x in ctx.eqn.params["start_indices"])
    strides = ctx.eqn.params.get("strides")

    if strides is None:
        strides = (1,) * len(input_shape)
    else:
        strides = tuple(int(x) for x in strides)

    input_axes = []

    for input_extent, output_extent, output_axis, start, stride in zip(
        input_shape, output_shape, output.axes, starts, strides
    ):
        if stride == 1 and isinstance(output_axis, _FullAxis):
            selected = _axis_from_range(input_extent, start, start + output_extent)

        elif stride == 1 and isinstance(output_axis, _RangeAxis):
            selected = _axis_from_range(
                input_extent, start + output_axis.start, start + output_axis.stop
            )

        else:
            rows = axis_indices(output_axis, extent=output_extent)
            selected = _axis_from_indices(input_extent, start + rows * stride)

        if selected is None:
            return (None,)

        input_axes.append(selected)

    return (
        TensorDemand.from_axes(
            input_shape,
            tuple(input_axes),
        ),
    )


def demand_rev(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    shape = _shape_of(ctx.eqn.invars[0])
    reversed_axes = {int(axis) for axis in ctx.eqn.params["dimensions"]}
    input_axes = []

    for axis_index, (extent, selection) in enumerate(zip(shape, output.axes)):
        if axis_index not in reversed_axes:
            input_axes.append(selection)
            continue

        if isinstance(selection, _FullAxis):
            input_axes.append(_FullAxis())

        elif isinstance(selection, _RangeAxis):
            input_axes.append(
                _RangeAxis(
                    start=extent - selection.stop,
                    stop=extent - selection.start,
                )
            )

        else:
            indices = extent - 1 - selection.indices
            input_axes.append(cast(AxisSubset, _axis_from_indices(extent, indices)))

    return (
        TensorDemand.from_axes(
            shape,
            tuple(input_axes),
        ),
    )


def demand_concatenate(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    axis = int(ctx.eqn.params["dimension"])
    output_shape = _shape_of(ctx.eqn.outvars[0])
    output_axis = output.axes[axis]
    result: list[Demand] = []
    offset = 0

    for atom in ctx.eqn.invars:
        input_shape = _shape_of(atom)
        extent = input_shape[axis]

        if isinstance(output_axis, _FullAxis):
            local_axis: AxisSubset = _FullAxis()

        elif isinstance(output_axis, _RangeAxis):
            start = max(output_axis.start, offset)
            stop = min(output_axis.stop, offset + extent)

            if start >= stop:
                local_axis = None
            else:
                local_axis = _axis_from_range(extent, start - offset, stop - offset)

        else:
            indices = output_axis.indices
            mask = (indices >= offset) & (indices < offset + extent)
            local_axis = _axis_from_indices(
                extent,
                indices[mask] - offset,
            )

        if local_axis is None:
            result.append(None)
            offset += extent
            continue

        input_axes = list(output.axes)
        input_axes[axis] = local_axis

        result.append(
            TensorDemand.from_axes(
                input_shape,
                tuple(input_axes),
            )
        )
        offset += extent

    if offset != output_shape[axis]:
        raise RuntimeError("concatenate axis geometry mismatch")

    return tuple(result)


def _multi_input_row_map_demand(
    ctx: DemandContext,
    prepared: MultiInputRowMap,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    output_rows = output.rows()
    operand_indices = prepared.operand_indices[output_rows]
    source_rows = prepared.source_rows[output_rows]
    result: list[Demand] = []

    for operand_index, atom in enumerate(ctx.eqn.invars):
        mask = operand_indices == operand_index

        rows = source_rows[mask]
        rows = rows[rows >= 0]

        result.append(TensorDemand.from_rows_hull(_shape_of(atom), rows))

    return tuple(result)


def _multi_output_unary_row_map_demand(
    ctx: DemandContext,
    prepared: MultiOutputUnaryMap,
) -> tuple[Demand, ...]:
    source_parts = []

    for output_index, demand in enumerate(ctx.output_demands):
        if demand is None:
            continue

        rows = demand.rows()
        source_parts.append(prepared.source_rows[output_index][rows])

    if not source_parts:
        return (None,)

    source_rows = np.unique(np.concatenate(source_parts))
    source_rows = source_rows[source_rows >= 0]

    return (TensorDemand.from_rows_hull(_shape_of(ctx.eqn.invars[0]), source_rows),)


RESHAPE_LIKE = OperationSemantics(
    DerivativeRule(
        no_prepare,
        reshape_like_dependencies,
        no_hessian,
    ),
    demand=demand_reshape_squeeze,
)
