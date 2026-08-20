from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

import numpy as np
from jax.extend.core import JaxprEqn, Var
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of

type Shape = tuple[int, ...]

# these 2 are also in program/concrete_resolver.py.... should probably be unified
type ConcreteValue = Any
type ConcreteEnv = Mapping[Var, ConcreteValue]


class _RegionalRows(Protocol):
    global_shape: tuple[int, ...]

    def read_rows(self, global_rows: NDArray[np.int64]) -> NDArray[Any]: ...


def _is_regional_rows(value: object) -> TypeGuard[_RegionalRows]:
    return hasattr(value, "read_rows") and hasattr(value, "global_shape")


@dataclass(frozen=True, slots=True)
class GatherRoute:
    """Resolved gather geometry in global scalar rows.

    `source_rows[o]` is the operand row read for output row `o`.
    `index_rows[o, c]` is the flattened gather-index row supplying index
    component `c` for output row `o`.
    """

    source_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class ScatterRoute:
    """For each flattened update row, the flattened operand row it targets.

    A value of -1 represents an out-of-bounds/dropped update.
    """

    target_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class SelectNRoute:
    # for each output row, which case operand is selected
    # 0 means eqn.invars[1], 1 means eqn.invars[2], etc.
    case_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicSliceRoute:
    source_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicUpdateSliceRoute:
    target_rows: NDArray[np.int64]


type Route = (
    GatherRoute
    | ScatterRoute
    | SelectNRoute
    | DynamicSliceRoute
    | DynamicUpdateSliceRoute
)
type RouteEnv = Mapping[JaxprEqn, Route]


def resolve_gather_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> GatherRoute | None:
    indices = concrete.get(eqn.invars[1])
    if indices is None:
        return None

    return _compute_gather_route(eqn, np.asarray(indices))


def resolve_scatter_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> ScatterRoute | None:
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None

    # operand, indices, updates = eqn.invars[:3]
    indices = concrete.get(eqn.invars[1])
    if indices is None:
        return None

    return _compute_scatter_route(eqn, np.asarray(indices))


def resolve_select_n_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> SelectNRoute | None:
    if len(eqn.invars) < 2 or not eqn.outvars:
        return None

    selector = concrete.get(eqn.invars[0])
    if selector is None:
        return None

    output_shape = _shape_of(eqn.outvars[0])

    selected = np.broadcast_to(np.asarray(selector), output_shape).astype(
        np.int64, copy=False
    )
    n_cases = len(eqn.invars) - 1

    if np.any(selected < 0) or np.any(selected >= n_cases):
        return None

    return SelectNRoute(selected.ravel())


def resolve_dynamic_slice_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> DynamicSliceRoute | None:
    if len(eqn.invars) < 2 or len(eqn.outvars) != 1:
        return None

    operand_shape = _shape_of(eqn.invars[0])
    output_shape = _shape_of(eqn.outvars[0])

    starts = []
    for var in eqn.invars[1:]:
        value = concrete.get(var)
        if value is None:
            return None
        starts.append(int(np.asarray(value)))

    # JAX dynamic_slice clips starts into the valid range.
    max_starts = tuple(dim - size for dim, size in zip(operand_shape, output_shape))

    starts = tuple(
        min(max(start, 0), max_start) for start, max_start in zip(starts, max_starts)
    )

    rows = np.arange(int(np.prod(operand_shape)), dtype=np.int64).reshape(operand_shape)

    slices = tuple(
        slice(start, start + size) for start, size in zip(starts, output_shape)
    )

    return DynamicSliceRoute(source_rows=rows[slices].ravel())


def resolve_dynamic_update_slice_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> DynamicUpdateSliceRoute | None:
    if len(eqn.invars) < 3 or len(eqn.outvars) != 1:
        return None

    operand_shape = _shape_of(eqn.invars[0])
    update_shape = _shape_of(eqn.invars[1])

    starts = []
    for var in eqn.invars[2:]:
        value = concrete.get(var)
        if value is None:
            return None
        starts.append(int(np.asarray(value)))

    max_starts = tuple(dim - size for dim, size in zip(operand_shape, update_shape))

    starts = tuple(
        min(max(start, 0), max_start) for start, max_start in zip(starts, max_starts)
    )

    update_rows = np.arange(int(np.prod(update_shape)), dtype=np.int64)

    update_coords = np.stack(np.unravel_index(update_rows, update_shape), axis=1)

    target_coords = update_coords + np.asarray(starts, dtype=np.int64)

    target_rows = np.ravel_multi_index(tuple(target_coords.T), operand_shape).astype(
        np.int64
    )

    return DynamicUpdateSliceRoute(target_rows=target_rows)


def _compute_gather_route(
    eqn: JaxprEqn,
    indices: NDArray,
) -> GatherRoute:
    output_shape = _shape_of(eqn.outvars[0])
    output_rows = np.arange(int(np.prod(output_shape, dtype=np.int64)), dtype=np.int64)
    source_rows, index_rows = _compute_gather_route_rows(eqn, indices, output_rows)
    return GatherRoute(source_rows=source_rows, index_rows=index_rows)


def _compute_gather_route_rows(
    eqn: JaxprEqn,
    indices: NDArray,
    output_rows: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:
    """Resolve gather geometry only for the supplied flattened output rows."""
    operand_shape = _shape_of(eqn.invars[0])
    output_shape = _shape_of(eqn.outvars[0])

    regional = _is_regional_rows(indices)
    indices_shape = (
        tuple(indices.global_shape) if regional else tuple(np.asarray(indices).shape)
    )
    output_rows = np.asarray(output_rows, dtype=np.int64).ravel()

    dnums = eqn.params["dimension_numbers"]
    slice_sizes = tuple(int(x) for x in eqn.params["slice_sizes"])

    offset_dims = tuple(int(x) for x in dnums.offset_dims)
    collapsed_dims = tuple(int(x) for x in dnums.collapsed_slice_dims)
    start_index_map = tuple(int(x) for x in dnums.start_index_map)

    operand_batching_dims = tuple(
        int(x) for x in getattr(dnums, "operand_batching_dims", ())
    )
    indices_batching_dims = tuple(
        int(x) for x in getattr(dnums, "start_indices_batching_dims", ())
    )

    operand_rank = len(operand_shape)
    output_rank = len(output_shape)

    if len(indices_shape) < 1:
        raise NotImplementedError("gather scalar index arrays are not supported")

    if len(slice_sizes) != operand_rank:
        raise ValueError("gather slice_sizes rank does not match operand rank")

    index_vector_size = indices_shape[-1]

    if index_vector_size != len(start_index_map):
        raise ValueError("gather index-vector size does not match start_index_map")

    if len(operand_batching_dims) != len(indices_batching_dims):
        raise ValueError("gather operand/index batching dimensions do not match")

    # Output coordinates
    n_output = output_rows.size
    output_size = int(np.prod(output_shape, dtype=np.int64))
    if np.any(output_rows < 0) or np.any(output_rows >= output_size):
        raise ValueError("gather fragment output rows are outside the output shape")

    if output_shape:
        output_coords = np.stack(
            np.unravel_index(output_rows, output_shape),
            axis=1,
        ).astype(np.int64)
    else:
        output_coords = np.empty((n_output, 0), dtype=np.int64)

    # ------------------------------------------------------------------
    # Gather batch coordinates
    #
    # Output axes not present in offset_dims correspond to
    # indices.shape[:-1].
    # ------------------------------------------------------------------
    offset_set = set(offset_dims)
    output_batch_dims = tuple(
        axis for axis in range(output_rank) if axis not in offset_set
    )
    expected_batch_rank = len(indices_shape) - 1

    if len(output_batch_dims) != expected_batch_rank:
        raise NotImplementedError("unsupported gather output/index batch geometry")

    if output_batch_dims:
        batch_coords = output_coords[:, output_batch_dims]
    else:
        batch_coords = np.empty((n_output, 0), dtype=np.int64)

    # Index vector for every output row
    if index_vector_size:
        if len(indices_shape) == 1:
            index_rows = np.broadcast_to(
                np.arange(index_vector_size, dtype=np.int64),
                (n_output, index_vector_size),
            ).copy()
        else:
            components = np.tile(np.arange(index_vector_size, dtype=np.int64), n_output)
            flattened_coords = [
                np.repeat(batch_coords[:, axis], index_vector_size)
                for axis in range(len(indices_shape) - 1)
            ]
            flattened_coords.append(components)
            index_rows = np.ravel_multi_index(
                tuple(flattened_coords), indices_shape
            ).reshape(n_output, index_vector_size)
        if regional:
            index_vectors = np.asarray(
                indices.read_rows(index_rows), dtype=np.int64
            ).reshape(n_output, index_vector_size)
        else:
            index_vectors = (
                np.asarray(indices)
                .ravel()[index_rows]
                .reshape(n_output, index_vector_size)
            )
    else:
        index_vectors = np.empty((n_output, 0), dtype=np.int64)

    if index_vector_size == 0:
        index_rows = np.empty((n_output, 0), dtype=np.int64)
    elif len(indices_shape) == 1:
        index_rows = np.broadcast_to(
            np.arange(index_vector_size, dtype=np.int64), (n_output, index_vector_size)
        ).copy()
    else:
        components = np.tile(np.arange(index_vector_size, dtype=np.int64), n_output)
        flattened_coords = [
            np.repeat(batch_coords[:, axis], index_vector_size)
            for axis in range(len(indices_shape) - 1)
        ]
        flattened_coords.append(components)
        index_rows = np.ravel_multi_index(
            tuple(flattened_coords), indices_shape
        ).reshape(n_output, index_vector_size)

    # Start coordinate in operand space
    starts = np.zeros(
        (n_output, operand_rank),
        dtype=np.int64,
    )

    for component, operand_axis in enumerate(start_index_map):
        starts[:, operand_axis] = index_vectors[:, component]

    # Explicit batching dimensions get their coordinates from the
    # corresponding index batch dimension.
    for operand_axis, index_axis in zip(
        operand_batching_dims,
        indices_batching_dims,
    ):
        starts[:, operand_axis] = batch_coords[:, index_axis]

    # Window offsets
    excluded = set(collapsed_dims) | set(operand_batching_dims)
    window_operand_dims = tuple(
        axis for axis in range(operand_rank) if axis not in excluded
    )
    if len(window_operand_dims) != len(offset_dims):
        raise NotImplementedError("unsupported gather window geometry")

    offsets = np.zeros_like(starts)

    for output_axis, operand_axis in zip(offset_dims, window_operand_dims):
        offsets[:, operand_axis] = output_coords[:, output_axis]

    # Out-of-bounds policy
    upper_starts = np.asarray(operand_shape, dtype=np.int64) - np.asarray(
        slice_sizes, dtype=np.int64
    )

    if np.any(upper_starts < 0):
        raise ValueError("gather slice is larger than operand")

    mode = eqn.params.get("mode")
    mode_name = (
        "PROMISE_IN_BOUNDS"
        if mode is None
        else getattr(mode, "name", str(mode)).rsplit(".", 1)[-1].upper()
    )

    if mode_name in {"FILL", "DROP"}:
        mode_name = "FILL_OR_DROP"

    valid = np.ones(n_output, dtype=bool)

    if mode_name in {"CLIP", "PROMISE_IN_BOUNDS"}:
        starts = np.minimum(
            np.maximum(starts, 0),
            upper_starts,
        )
    elif mode_name == "FILL_OR_DROP":
        for component, operand_axis in enumerate(start_index_map):
            values = index_vectors[:, component]
            valid &= values >= 0
            valid &= values <= upper_starts[operand_axis]
    else:
        raise NotImplementedError(f"unsupported gather mode {mode_name!r}")

    # Final source rows
    source_coords = starts + offsets
    source_rows = np.full(n_output, -1, dtype=np.int64)

    if np.any(valid):
        coords = source_coords[valid]
        operand_bounds = np.asarray(operand_shape, dtype=np.int64)
        if np.any(coords < 0) or np.any(coords >= operand_bounds):
            raise ValueError("computed gather source coordinates are outside operand")

        source_rows[valid] = np.ravel_multi_index(tuple(coords.T), operand_shape)

    return source_rows, index_rows


def _compute_scatter_route(eqn: JaxprEqn, indices: NDArray) -> ScatterRoute | None:
    # NOTE: mostly unchecked
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None

    updates_shape = tuple(_shape_of(eqn.invars[2]))
    n_updates = int(np.prod(updates_shape))
    update_rows = np.arange(n_updates, dtype=np.int64)

    target_rows = _compute_scatter_target_rows(eqn, np.asarray(indices), update_rows)
    if target_rows is None:
        return None
    return ScatterRoute(target_rows=target_rows)


def _compute_scatter_target_rows(
    eqn: JaxprEqn,
    indices: NDArray,
    update_rows: NDArray[np.int64],
) -> NDArray[np.int64] | None:
    """Resolve scatter targets only for supplied flattened update rows."""
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None

    operand_shape = tuple(_shape_of(eqn.invars[0]))
    indices_shape = tuple(_shape_of(eqn.invars[1]))
    updates_shape = tuple(_shape_of(eqn.invars[2]))
    indices = np.asarray(indices)
    update_rows = np.asarray(update_rows, dtype=np.int64).ravel()

    if indices.ndim < 1 or tuple(indices.shape) != indices_shape:
        return None

    try:
        dnums = eqn.params["dimension_numbers"]
        window_dims = tuple(dnums.update_window_dims)
        inserted_dims = tuple(dnums.inserted_window_dims)
        scatter_dims = tuple(dnums.scatter_dims_to_operand_dims)
        operand_batch_dims = tuple(dnums.operand_batching_dims)
        indices_batch_dims = tuple(dnums.scatter_indices_batching_dims)
    except (KeyError, TypeError, AttributeError):
        return None

    index_vector_size = indices_shape[-1]
    if index_vector_size != len(scatter_dims):
        return None
    if len(operand_batch_dims) != len(indices_batch_dims):
        return None

    n_total_updates = int(np.prod(updates_shape))
    if np.any(update_rows < 0) or np.any(update_rows >= n_total_updates):
        raise ValueError("scatter fragment update rows are outside the updates shape")
    n_updates = update_rows.size

    try:
        update_coords = np.stack(
            np.unravel_index(update_rows, updates_shape),
            axis=1,
        )

        # Map update dimensions that are NOT window dimensions
        # onto indices.shape[:-1].
        batch_update_dims = tuple(
            d for d in range(len(updates_shape)) if d not in window_dims
        )

        if len(batch_update_dims) != len(indices_shape) - 1:
            return None

        index_batch_coords = update_coords[:, batch_update_dims]

        if index_vector_size:
            if len(indices_shape) == 1:
                # One index vector shared by every update-window element.
                #
                # This is produced by expressions such as
                #
                #     x.at[..., i, j].set(values)
                #
                # where leading dimensions of `values` are update-window
                # dimensions, not scatter-index batch dimensions.
                index_vectors = np.broadcast_to(
                    np.asarray(indices, dtype=np.int64).reshape(1, index_vector_size),
                    (n_updates, index_vector_size),
                )
            else:
                key = tuple(
                    index_batch_coords[:, i] for i in range(index_batch_coords.shape[1])
                )

                index_vectors = np.asarray(indices[key], dtype=np.int64)
                index_vectors = index_vectors.reshape(
                    n_updates,
                    index_vector_size,
                )
        else:
            index_vectors = np.empty((n_updates, 0), dtype=np.int64)

        # Construct operand coordinate for each update scalar.
        target_coords = np.zeros(
            (n_updates, len(operand_shape)),
            dtype=np.int64,
        )

        # Explicit scatter indices.
        for component, operand_axis in enumerate(scatter_dims):
            target_coords[:, operand_axis] = index_vectors[:, component]

        # Batched dimensions.
        for operand_axis, indices_axis in zip(
            operand_batch_dims,
            indices_batch_dims,
        ):
            target_coords[:, operand_axis] = index_batch_coords[:, indices_axis]

        # Window dimensions.
        window_operand_dims = tuple(
            d
            for d in range(len(operand_shape))
            if d not in inserted_dims and d not in operand_batch_dims
        )

        if len(window_operand_dims) != len(window_dims):
            return None

        for update_axis, operand_axis in zip(
            window_dims,
            window_operand_dims,
        ):
            target_coords[:, operand_axis] += update_coords[:, update_axis]

        # Dropped / out-of-bounds updates become -1.
        valid = np.ones(n_updates, dtype=bool)

        for axis, size in enumerate(operand_shape):
            valid &= target_coords[:, axis] >= 0
            valid &= target_coords[:, axis] < size

        target_rows = np.full(n_updates, -1, dtype=np.int64)

        if np.any(valid):
            target_rows[valid] = np.ravel_multi_index(
                tuple(target_coords[valid].T),
                operand_shape,
            )

        return target_rows

    except (ValueError, IndexError, TypeError):
        # TODO: looks fishy, necessary?
        return None
