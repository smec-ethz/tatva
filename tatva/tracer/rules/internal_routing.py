from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn, Literal

from tatva.tracer.core.semantics import InternalRoutingBatching, RoutingSemantics
from tatva.tracer.core.tagged import Tagged, TaggedDemand
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand
from tatva.tracer.program.dependencies import DependencySet, HessianAccumulator


def _batching(
    eqn: JaxprEqn,
    routing: RoutingSemantics,
) -> InternalRoutingBatching:
    internal = routing.internal

    if internal is None:
        raise NotImplementedError(
            f"{eqn.primitive.name} has no invocation-internal routing semantics"
        )

    batching = internal.batching(eqn)
    if len(batching.input_axes) != len(eqn.invars):
        raise RuntimeError(
            f"{eqn.primitive.name}: internal batching describes "
            f"{len(batching.input_axes)} inputs, expected {len(eqn.invars)}"
        )

    n_batch = len(batching.output_axes)
    for axes in batching.input_axes:
        if len(axes) != n_batch:
            raise RuntimeError(
                f"{eqn.primitive.name}: inconsistent internal batch rank"
            )

    return batching


def demand(
    eqn: JaxprEqn,
    output_demands: tuple[Demand, ...],
    routing: RoutingSemantics,
) -> tuple[Demand, ...]:
    if len(output_demands) != 1:
        raise NotImplementedError(
            "invocation-internal routing currently requires one output"
        )

    output = output_demands[0]
    if output is None:
        return tuple(None for _ in eqn.invars)

    batching = _batching(eqn, routing)
    result: list[Demand] = []

    for atom, input_axes in zip(eqn.invars, batching.input_axes, strict=True):
        if isinstance(atom, Literal):
            result.append(None)
            continue

        mappings = tuple(
            (output_axis, input_axis)
            for output_axis, input_axis in zip(
                batching.output_axes, input_axes, strict=True
            )
            if input_axis is not None
        )
        result.append(TensorDemand.from_mapped_axes(_shape_of(atom), output, mappings))

    return tuple(result)


def _expand_tagged(
    output: TaggedDemand,
    *,
    input_shape: tuple[int, ...],
    output_axes: tuple[int, ...],
    input_axes: tuple[int | None, ...],
) -> TaggedDemand | None:
    pairs = tuple(
        (output_axis, input_axis)
        for output_axis, input_axis in zip(output_axes, input_axes, strict=True)
        if input_axis is not None
    )

    if not pairs:
        return TaggedDemand.full(input_shape, output.block_ids)

    output_coords = np.stack(
        np.unravel_index(output.rows, output.shape),
        axis=1,
    )

    fixed = np.column_stack(
        (
            *(output_coords[:, output_axis] for output_axis, _ in pairs),
            output.blocks,
        )
    )

    # One expansion per unique (batch coordinate, block).
    fixed = np.unique(fixed, axis=0)

    mapped_input_axes = {input_axis for _, input_axis in pairs}
    free_axes = tuple(
        axis for axis in range(len(input_shape)) if axis not in mapped_input_axes
    )

    free_shape = tuple(input_shape[axis] for axis in free_axes)
    free_size = int(np.prod(free_shape, dtype=np.int64))

    if free_axes:
        free_coords = np.stack(
            np.unravel_index(np.arange(free_size, dtype=np.int64), free_shape),
            axis=1,
        )
    else:
        free_size = 1
        free_coords = np.empty((1, 0), dtype=np.int64)

    coords = np.zeros((fixed.shape[0] * free_size, len(input_shape)), dtype=np.int64)

    for column, (_, input_axis) in enumerate(pairs):
        coords[:, input_axis] = np.repeat(fixed[:, column], free_size)

    for column, input_axis in enumerate(free_axes):
        coords[:, input_axis] = np.tile(free_coords[:, column], fixed.shape[0])

    rows = np.ravel_multi_index(tuple(coords.T), input_shape)
    blocks = np.repeat(fixed[:, -1], free_size)

    return TaggedDemand(input_shape, rows, blocks)


def tagged_demand(
    eqn: JaxprEqn,
    output_demands: tuple[Tagged, ...],
    routing: RoutingSemantics,
) -> tuple[Tagged, ...]:
    if len(output_demands) != 1:
        raise NotImplementedError(
            "invocation-internal routing currently requires one output"
        )

    output = output_demands[0]
    if output is None:
        return tuple(None for _ in eqn.invars)

    batching = _batching(eqn, routing)
    routing_inputs = frozenset(routing.inputs(eqn))

    result: list[Tagged] = []

    for index, (atom, input_axes) in enumerate(
        zip(eqn.invars, batching.input_axes, strict=True)
    ):
        if isinstance(atom, Literal) or index in routing_inputs:
            result.append(None)
            continue

        result.append(
            _expand_tagged(
                output,
                input_shape=_shape_of(atom),
                output_axes=batching.output_axes,
                input_axes=input_axes,
            )
        )

    return tuple(result)


def _axis_coordinates(
    flat_rows: np.ndarray,
    shape: tuple[int, ...],
    axis: int,
) -> np.ndarray:
    """Coordinate on one axis for flattened C-order rows."""
    stride = int(np.prod(shape[axis + 1 :], dtype=np.int64))
    return (flat_rows // stride) % shape[axis]


def _output_batch_rows(
    shape: tuple[int, ...],
    batch_axes: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> np.ndarray:
    """Map every flattened tensor row to its logical batch row."""

    count = int(np.prod(shape, dtype=np.int64))

    if count == 0:
        return np.empty(0, dtype=np.int64)

    if not batch_axes:
        return np.zeros(count, dtype=np.int64)

    if len(batch_axes) != len(batch_shape):
        raise ValueError("batch axis/rank mismatch")

    flat = np.arange(count, dtype=np.int64)
    coordinates = tuple(_axis_coordinates(flat, shape, axis) for axis in batch_axes)

    for logical_axis, axis in enumerate(batch_axes):
        if shape[axis] != batch_shape[logical_axis]:
            raise ValueError(
                f"batch extent mismatch: tensor axis {axis} has "
                f"extent {shape[axis]}, expected "
                f"{batch_shape[logical_axis]}"
            )

    return np.asarray(
        np.ravel_multi_index(coordinates, batch_shape),
        dtype=np.int64,
    )


def _dependencies_by_batch(
    dep: DependencySet,
    input_axes: tuple[int | None, ...],
    batch_shape: tuple[int, ...],
) -> sps.csr_matrix:
    """Union payload dependencies independently for every logical batch.

    The returned CSR has shape:

        (prod(batch_shape), n_dofs)

    A None input axis means that the input is broadcast across that logical
    batch dimension.
    """

    if len(input_axes) != len(batch_shape):
        raise ValueError("input batch mapping has the wrong rank")

    n_dofs = dep.csr.shape[1]
    batch_count = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1

    if dep.csr.shape[0] == 0:
        return sps.csr_matrix((batch_count, n_dofs), dtype=bool)

    carried = tuple(
        (logical_axis, input_axis)
        for logical_axis, input_axis in enumerate(input_axes)
        if input_axis is not None
    )

    # Input carries no batch axes: union the entire input, then broadcast that
    # dependency set to every logical batch.
    if not carried:
        union = dep.total_union().csr
        rows = np.zeros(batch_count, dtype=np.int64)
        return union[rows]

    carried_shape = tuple(batch_shape[logical_axis] for logical_axis, _ in carried)

    input_count = dep.csr.shape[0]
    flat = np.arange(input_count, dtype=np.int64)

    input_coordinates: list[np.ndarray] = []

    for logical_axis, input_axis in carried:
        assert input_axis is not None

        if dep.shape[input_axis] != batch_shape[logical_axis]:
            raise ValueError(
                f"input batch extent mismatch: input axis {input_axis} "
                f"has extent {dep.shape[input_axis]}, expected "
                f"{batch_shape[logical_axis]}"
            )

        input_coordinates.append(_axis_coordinates(flat, dep.shape, input_axis))

    # For every input scalar, determine the batch group it belongs to.
    input_batch_rows = np.asarray(
        np.ravel_multi_index(tuple(input_coordinates), carried_shape),
        dtype=np.int64,
    )

    carried_count = int(np.prod(carried_shape, dtype=np.int64))

    # OR all payload rows belonging to the same carried batch coordinate.
    reducer = sps.csr_matrix(
        (
            np.ones(input_count, dtype=bool),
            (
                input_batch_rows,
                np.arange(input_count, dtype=np.int64),
            ),
        ),
        shape=(carried_count, input_count),
        dtype=bool,
    )

    reduced = (reducer @ dep.csr).astype(bool)
    reduced.eliminate_zeros()

    # Expand batch dimensions that this input does not carry.
    if len(carried) == len(batch_shape):
        return reduced

    full_batch_rows = np.arange(batch_count, dtype=np.int64)
    full_coordinates = np.unravel_index(full_batch_rows, batch_shape)

    carried_rows = np.asarray(
        np.ravel_multi_index(
            tuple(full_coordinates[logical_axis] for logical_axis, _ in carried),
            carried_shape,
        ),
        dtype=np.int64,
    )

    return reduced[carried_rows]


def dependencies(
    eqn: JaxprEqn,
    input_deps: tuple[DependencySet, ...],
    routing: RoutingSemantics,
    *,
    n_dofs: int,
    acc: HessianAccumulator,
) -> tuple[DependencySet, ...]:
    """Conservative derivative structure for invocation-internal routing.

    Dynamic routing may mix payload entries within one logical batch, but may
    not mix different logical batches.
    """

    batching = _batching(eqn, routing)
    routing_inputs = frozenset(routing.inputs(eqn))

    if not eqn.outvars:
        return ()

    # All registered internal routed primitives currently have one logical
    # output batching description. Derive its batch extents from the first
    # output.
    output_shape = _shape_of(eqn.outvars[0])
    batch_shape = tuple(output_shape[axis] for axis in batching.output_axes)
    batch_count = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1

    # One dependency row per logical batch. Each row is the conservative union
    # of all payload dependencies from every differentiable data input in that
    # same batch.
    batch_csr = sps.csr_matrix((batch_count, n_dofs), dtype=bool)

    for input_index, (dep, input_axes) in enumerate(
        zip(input_deps, batching.input_axes, strict=True)
    ):
        # Routing selectors are discrete routing state, not differentiable
        # data dependencies.
        if input_index in routing_inputs:
            continue

        if dep.csr.nnz == 0:
            continue

        input_batch_csr = _dependencies_by_batch(dep, input_axes, batch_shape)
        batch_csr = batch_csr.maximum(input_batch_csr)

    # Conservative nonlinear coupling is also batch-local. This is the
    # important replacement for acc.add_self(total_union).
    acc.add_self(DependencySet(batch_csr, batch_shape))

    outputs: list[DependencySet] = []

    for outvar in eqn.outvars:
        shape = _shape_of(outvar)

        # Every scalar payload output in a batch receives the conservative
        # union for that batch, but never dependencies from another batch.
        output_batch_rows = _output_batch_rows(shape, batching.output_axes, batch_shape)

        matrix = batch_csr[output_batch_rows]
        outputs.append(DependencySet(matrix.tocsr(), shape))

    return tuple(outputs)


def gather_batching(eqn: JaxprEqn) -> InternalRoutingBatching:
    if len(eqn.invars) != 2 or len(eqn.outvars) != 1:
        raise ValueError("gather expects two inputs and one output")

    dnums = eqn.params["dimension_numbers"]

    offset_dims = tuple(int(x) for x in dnums.offset_dims)
    operand_batch = tuple(int(x) for x in getattr(dnums, "operand_batching_dims", ()))
    indices_batch = tuple(
        int(x) for x in getattr(dnums, "start_indices_batching_dims", ())
    )

    if len(operand_batch) != len(indices_batch):
        raise ValueError("gather batching dimensions do not match")

    output_rank = len(_shape_of(eqn.outvars[0]))
    offset_set = set(offset_dims)

    # These correspond, in order, to start_indices.shape[:-1].
    output_batch_dims = tuple(
        axis for axis in range(output_rank) if axis not in offset_set
    )

    output_axes = tuple(output_batch_dims[index_axis] for index_axis in indices_batch)

    slice_sizes = tuple(int(x) for x in eqn.params["slice_sizes"])
    for operand_axis in operand_batch:
        if slice_sizes[operand_axis] != 1:
            raise NotImplementedError(
                "invocation-internal gather batching requires "
                "slice size 1 on operand batching dimensions"
            )

    return InternalRoutingBatching(
        output_axes=output_axes,
        input_axes=(
            operand_batch,
            indices_batch,
        ),
    )


def scatter_batching(eqn: JaxprEqn) -> InternalRoutingBatching:
    if len(eqn.invars) < 3 or len(eqn.outvars) != 1:
        raise ValueError("scatter expects operand, indices, updates")

    dnums = eqn.params["dimension_numbers"]

    operand_batch = tuple(int(x) for x in getattr(dnums, "operand_batching_dims", ()))
    indices_batch = tuple(
        int(x) for x in getattr(dnums, "scatter_indices_batching_dims", ())
    )

    if len(operand_batch) != len(indices_batch):
        raise ValueError("scatter batching dimensions do not match")

    updates_shape = _shape_of(eqn.invars[2])
    window_dims = {int(x) for x in dnums.update_window_dims}

    # update dimensions corresponding to scatter_indices.shape[:-1]
    batch_update_dims = tuple(
        axis for axis in range(len(updates_shape)) if axis not in window_dims
    )

    update_batch = tuple(batch_update_dims[index_axis] for index_axis in indices_batch)

    return InternalRoutingBatching(
        output_axes=operand_batch,
        input_axes=(
            operand_batch,
            indices_batch,
            update_batch,
        ),
    )


def _broadcast_axes(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[int | None, ...]:
    offset = len(output_shape) - len(input_shape)

    if offset < 0:
        raise ValueError("input rank exceeds broadcast output rank")

    result: list[int | None] = []

    for output_axis, output_extent in enumerate(output_shape):
        input_axis = output_axis - offset

        if input_axis < 0:
            result.append(None)
            continue

        input_extent = input_shape[input_axis]

        if input_extent == output_extent:
            result.append(input_axis)
        elif input_extent == 1:
            result.append(None)
        else:
            raise ValueError(f"cannot broadcast {input_shape} to {output_shape}")

    return tuple(result)


def select_n_batching(eqn: JaxprEqn) -> InternalRoutingBatching:
    output_shape = _shape_of(eqn.outvars[0])
    output_axes = tuple(range(len(output_shape)))

    return InternalRoutingBatching(
        output_axes=output_axes,
        input_axes=tuple(
            _broadcast_axes(_shape_of(atom), output_shape) for atom in eqn.invars
        ),
    )


def dynamic_slice_batching(eqn: JaxprEqn) -> InternalRoutingBatching:
    operand_shape = _shape_of(eqn.invars[0])
    output_shape = _shape_of(eqn.outvars[0])

    batch_axes = tuple(
        axis
        for axis, (operand_extent, output_extent) in enumerate(
            zip(operand_shape, output_shape, strict=True)
        )
        if operand_extent == output_extent
    )

    n_batch = len(batch_axes)

    return InternalRoutingBatching(
        output_axes=batch_axes,
        input_axes=(
            batch_axes,
            *(tuple(None for _ in range(n_batch)) for _ in eqn.invars[1:]),
        ),
    )


def dynamic_update_slice_batching(
    eqn: JaxprEqn,
) -> InternalRoutingBatching:
    operand_shape = _shape_of(eqn.invars[0])
    update_shape = _shape_of(eqn.invars[1])

    batch_axes = tuple(
        axis
        for axis, (operand_extent, update_extent) in enumerate(
            zip(operand_shape, update_shape, strict=True)
        )
        if operand_extent == update_extent
    )

    n_batch = len(batch_axes)

    return InternalRoutingBatching(
        output_axes=batch_axes,
        input_axes=(
            batch_axes,
            batch_axes,
            *(tuple(None for _ in range(n_batch)) for _ in eqn.invars[2:]),
        ),
    )
