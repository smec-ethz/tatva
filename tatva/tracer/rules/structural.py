from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of
from tatva.tracer.semantics import DerivativeRule, PrimitiveRule, no_hessian, no_prepare
from tatva.tracer.dependencies import DependencySet

if TYPE_CHECKING:
    from tatva.tracer.semantics import RuleContext


def reshape_like_dependencies(
    ctx: RuleContext,
    prepared: None,
) -> tuple[DependencySet, ...]:
    dep = ctx.input_deps[0]
    output_shape = _shape_of(ctx.eqn.outvars[0])

    return (dep.reshape(*output_shape),)


@dataclass(frozen=True)
class UnaryRowRoute:
    source_rows: NDArray
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class MultiInputRowRoute:
    operand_indices: NDArray
    source_rows: NDArray
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class MultiOutputUnaryRoute:
    source_rows: tuple[NDArray[np.int64], ...]
    output_shapes: tuple[tuple[int, ...], ...]


def unary_routed_dependencies(
    ctx: RuleContext,
    prepared: UnaryRowRoute,
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
    prepared: MultiInputRowRoute,
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
    prepared: MultiOutputUnaryRoute,
) -> tuple[DependencySet, ...]:
    source = ctx.input_deps[0]

    return tuple(
        DependencySet(source.csr[rows], shape)
        for rows, shape in zip(prepared.source_rows, prepared.output_shapes)
    )


def prepare_broadcast(ctx: RuleContext) -> UnaryRowRoute:
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

    return UnaryRowRoute(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_transpose(ctx: RuleContext) -> UnaryRowRoute:
    eqn = ctx.eqn

    input_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])

    rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(input_shape)

    source_rows = np.transpose(rows, axes=eqn.params["permutation"]).ravel()

    return UnaryRowRoute(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_slice(ctx: RuleContext) -> UnaryRowRoute:
    eqn = ctx.eqn

    input_shape = ctx.input_deps[0].shape
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

    return UnaryRowRoute(
        source_rows=rows[slices].ravel(),
        output_shape=output_shape,
    )


def prepare_rev(ctx: RuleContext) -> UnaryRowRoute:
    eqn = ctx.eqn

    input_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])

    rows = np.arange(int(np.prod(input_shape)), dtype=np.int64).reshape(input_shape)

    source_rows = np.flip(rows, axis=tuple(eqn.params["dimensions"])).ravel()

    return UnaryRowRoute(
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_concatenate(ctx: RuleContext) -> MultiInputRowRoute:
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

    return MultiInputRowRoute(
        operand_indices=np.concatenate(operand_parts, axis=axis).ravel(),
        source_rows=np.concatenate(source_parts, axis=axis).ravel(),
        output_shape=output_shape,
    )


def prepare_stack(ctx: RuleContext) -> MultiInputRowRoute:
    eqn = ctx.eqn

    axis = int(eqn.params["axis"])
    output_shape = _shape_of(eqn.outvars[0])

    operand_parts = []
    source_parts = []

    for operand_index, dep in enumerate(ctx.input_deps):
        shape = dep.shape
        size = int(np.prod(shape))

        operand_parts.append(np.full(shape, operand_index, dtype=np.int64))

        source_parts.append(np.arange(size, dtype=np.int64).reshape(shape))

    return MultiInputRowRoute(
        operand_indices=np.stack(operand_parts, axis=axis).ravel(),
        source_rows=np.stack(source_parts, axis=axis).ravel(),
        output_shape=output_shape,
    )


def prepare_pad(ctx: RuleContext) -> MultiInputRowRoute:
    eqn = ctx.eqn

    if len(ctx.input_deps) != 2:
        raise ValueError("pad expects source and padding-value inputs")

    source_shape = ctx.input_deps[0].shape
    output_shape = _shape_of(eqn.outvars[0])

    config = eqn.params["padding_config"]

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

    return MultiInputRowRoute(
        operand_indices=operand_indices,
        source_rows=source_rows,
        output_shape=output_shape,
    )


def prepare_split(ctx: RuleContext) -> MultiOutputUnaryRoute:
    if len(ctx.input_deps) != 1:
        raise ValueError("split expects one input")

    eqn = ctx.eqn
    source = ctx.input_deps[0]

    axis = int(eqn.params["axis"])
    sizes = tuple(int(x) for x in eqn.params["sizes"])

    index_tensor = np.arange(int(np.prod(source.shape)), dtype=np.int64).reshape(
        source.shape
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

    return MultiOutputUnaryRoute(source_rows=tuple(routes), output_shapes=tuple(shapes))


RESHAPE_LIKE = PrimitiveRule(
    DerivativeRule(no_prepare, reshape_like_dependencies, no_hessian)
)
