"""Demand-scoped route geometry for sparse planning traversals.

Fragments expose route geometry for explicitly requested global rows without
allocating a mapping proportional to the complete logical output. They drive
both sparse planning traversal and rank-local route localization; full routes
remain available for legacy materialization and unsupported fragment cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax.extend.core import JaxprEqn, Literal
from numpy.typing import NDArray

from tatva.tracer.core.concrete import ConcreteRegion
from tatva.tracer.core.routes import (
    ConcreteEnv,
    _compute_gather_route_rows,
    _compute_scatter_target_rows,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand, _FullAxis, _RangeAxis


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """Global flattened output rows whose route geometry is required."""

    output_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GatherRouteFragment:
    """Gather mappings aligned with ``output_rows`` rather than a full output."""

    output_rows: NDArray[np.int64]
    source_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class GatherEnvelopeFragment:
    """Demand-scoped gather relation with unresolved index components.

    ``source_demands[i]`` is a compact Cartesian envelope of every operand
    scalar that may feed ``output_rows[i]``. Components listed in
    ``dynamic_components`` stay as runtime index values; all other index
    components were resolved during planning.

    The representation is deliberately compact: a runtime lookup along a very
    large operand dimension is represented by an axis subset rather than by an
    explicit list of all possible scalar source rows.
    """

    output_rows: NDArray[np.int64]
    source_demands: tuple[TensorDemand | None, ...]
    index_rows: NDArray[np.int64]
    dynamic_components: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(self.output_rows, dtype=np.int64).ravel().copy()
        if rows.size != len(self.source_demands):
            raise ValueError("one gather source envelope is required per output row")
        if rows.size > 1 and np.any(rows[1:] <= rows[:-1]):
            raise ValueError("gather envelope output rows must be strictly increasing")
        index_rows = np.asarray(self.index_rows, dtype=np.int64).copy()
        if index_rows.ndim != 2 or index_rows.shape[0] != rows.size:
            raise ValueError("gather envelope index rows are not output-aligned")
        rows.flags.writeable = False
        index_rows.flags.writeable = False
        object.__setattr__(self, "output_rows", rows)
        object.__setattr__(self, "index_rows", index_rows)


@dataclass(frozen=True, slots=True)
class ScatterRouteFragment:
    """Scatter relations whose targets intersect the requested output rows."""

    output_rows: NDArray[np.int64]
    update_rows: NDArray[np.int64]
    target_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SelectNRouteFragment:
    """Selected case indices aligned with requested output rows."""

    output_rows: NDArray[np.int64]
    case_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicSliceRouteFragment:
    """Dynamic-slice source rows aligned with requested output rows."""

    output_rows: NDArray[np.int64]
    source_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicUpdateSliceRouteFragment:
    """Requested outputs overwritten by dynamic-update input rows."""

    output_rows: NDArray[np.int64]
    update_rows: NDArray[np.int64]
    target_rows: NDArray[np.int64]


type RouteFragment = (
    GatherRouteFragment
    | GatherEnvelopeFragment
    | ScatterRouteFragment
    | SelectNRouteFragment
    | DynamicSliceRouteFragment
    | DynamicUpdateSliceRouteFragment
)


def _output_rows(eqn: JaxprEqn, request: RouteRequest) -> NDArray[np.int64]:
    rows = np.asarray(request.output_rows, dtype=np.int64).ravel()
    output_shape = _shape_of(eqn.outvars[0])
    output_size = int(np.prod(output_shape, dtype=np.int64))
    if np.any(rows < 0) or np.any(rows >= output_size):
        raise ValueError("route fragment output rows are outside the output shape")
    return rows


def _read_concrete(concrete: ConcreteEnv, atom):
    return atom.val if isinstance(atom, Literal) else concrete.get(atom)


def _concrete_array(value):
    return value.values if isinstance(value, ConcreteRegion) else value


def _gather_index_rows(
    eqn: JaxprEqn, output_rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    output_shape = _shape_of(eqn.outvars[0])
    indices_shape = _shape_of(eqn.invars[1])
    offset_dims = {int(axis) for axis in eqn.params["dimension_numbers"].offset_dims}
    output_batch_dims = tuple(
        axis for axis in range(len(output_shape)) if axis not in offset_dims
    )
    if len(output_batch_dims) != len(indices_shape) - 1:
        raise NotImplementedError("unsupported gather output/index batch geometry")

    output_rows = np.asarray(output_rows, dtype=np.int64).ravel()
    if output_shape:
        coords = np.stack(np.unravel_index(output_rows, output_shape), axis=1)
    else:
        # Scalar gather output has exactly one flattened row and no output
        # coordinates. Its index tensor therefore has no batch coordinates.
        coords = np.empty((output_rows.size, 0), dtype=np.int64)

    batch = (
        coords[:, output_batch_dims]
        if output_batch_dims
        else np.empty((output_rows.size, 0), dtype=np.int64)
    )
    vector_size = indices_shape[-1]
    components = np.tile(np.arange(vector_size, dtype=np.int64), output_rows.size)
    flattened = [
        np.repeat(batch[:, axis], vector_size) for axis in range(len(indices_shape) - 1)
    ]
    flattened.append(components)
    return np.ravel_multi_index(tuple(flattened), indices_shape).reshape(
        output_rows.size, vector_size
    )


def gather_route_concrete_demands(
    eqn: JaxprEqn, request: RouteRequest
) -> tuple[Demand, ...]:
    rows = _output_rows(eqn, request)
    demands: list[Demand] = [None] * len(eqn.invars)
    demands[1] = TensorDemand.from_rows_hull(
        _shape_of(eqn.invars[1]), _gather_index_rows(eqn, rows).ravel()
    )
    return tuple(demands)


def _read_requested_rows(value, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    if isinstance(value, ConcreteRegion):
        return np.asarray(value.read_rows(rows), dtype=np.int64).ravel()
    return np.asarray(value).ravel()[rows].astype(np.int64, copy=False)


def _gather_output_geometry(
    eqn: JaxprEqn,
    output_rows: NDArray[np.int64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    tuple[int, ...],
    tuple[int, ...],
]:
    output_shape = _shape_of(eqn.outvars[0])
    indices_shape = _shape_of(eqn.invars[1])
    offset_dims = tuple(int(x) for x in eqn.params["dimension_numbers"].offset_dims)
    output_rows = np.asarray(output_rows, dtype=np.int64).ravel()

    if output_shape:
        output_coords = np.stack(
            np.unravel_index(output_rows, output_shape), axis=1
        ).astype(np.int64)
    else:
        output_coords = np.empty((output_rows.size, 0), dtype=np.int64)

    offset_set = set(offset_dims)
    output_batch_dims = tuple(
        axis for axis in range(len(output_shape)) if axis not in offset_set
    )
    if len(output_batch_dims) != len(indices_shape) - 1:
        raise NotImplementedError("unsupported gather output/index batch geometry")

    batch_coords = (
        output_coords[:, output_batch_dims]
        if output_batch_dims
        else np.empty((output_rows.size, 0), dtype=np.int64)
    )
    return output_coords, batch_coords, output_batch_dims, offset_dims


def _axis_interval(extent: int, start: int, stop: int):
    if not 0 <= start < stop <= extent:
        raise ValueError(
            f"invalid gather source interval [{start}, {stop}) for extent {extent}"
        )
    if start == 0 and stop == extent:
        return _FullAxis()
    return _RangeAxis(start, stop)


def resolve_partial_gather_route_fragment(ctx) -> RouteFragment | None:
    """Resolve a gather as far as planning-time data permits.

    Index-vector components are queried independently. Concrete components
    become exact structural coordinates. Components that depend on runtime
    data remain dynamic and contribute a compact conservative source envelope.

    This is the structural/payload boundary: no primitive-specific notion of
    "local" is required. If known components constrain a dynamic lookup to an
    already selected structural domain (for example batch iotas plus an LU
    pivot), the envelope remains local. If they do not (for example a truly
    dynamic lookup into a global vector), the envelope correctly widens to that
    global domain.
    """
    eqn = ctx.eqn
    output_rows = _output_rows(eqn, ctx.request)
    operand_shape = _shape_of(eqn.invars[0])
    indices_shape = _shape_of(eqn.invars[1])
    dnums = eqn.params["dimension_numbers"]
    slice_sizes = tuple(int(x) for x in eqn.params["slice_sizes"])
    collapsed_dims = tuple(int(x) for x in dnums.collapsed_slice_dims)
    start_index_map = tuple(int(x) for x in dnums.start_index_map)
    operand_batching_dims = tuple(
        int(x) for x in getattr(dnums, "operand_batching_dims", ())
    )
    indices_batching_dims = tuple(
        int(x) for x in getattr(dnums, "start_indices_batching_dims", ())
    )

    if len(indices_shape) < 1:
        return None
    if indices_shape[-1] != len(start_index_map):
        return None
    if len(operand_batching_dims) != len(indices_batching_dims):
        return None
    if len(set(start_index_map)) != len(start_index_map):
        raise NotImplementedError(
            "multiple gather index components targeting one operand axis are unsupported"
        )
    if set(start_index_map) & set(operand_batching_dims):
        raise NotImplementedError(
            "gather start-index and batching dimensions must be disjoint"
        )

    output_coords, batch_coords, _output_batch_dims, offset_dims = (
        _gather_output_geometry(eqn, output_rows)
    )
    index_rows = _gather_index_rows(eqn, output_rows)
    n_output = output_rows.size
    n_components = len(start_index_map)

    component_values: list[NDArray[np.int64] | None] = []
    dynamic_components: list[int] = []
    for component in range(n_components):
        rows = index_rows[:, component]
        demand = TensorDemand.from_rows_hull(indices_shape, rows)
        value = ctx.read_input(1, demand)
        if value is None:
            component_values.append(None)
            dynamic_components.append(component)
        else:
            component_values.append(_read_requested_rows(value, rows))

    upper_starts = np.asarray(operand_shape, dtype=np.int64) - np.asarray(
        slice_sizes, dtype=np.int64
    )
    if np.any(upper_starts < 0):
        raise ValueError("gather slice is larger than operand")

    excluded = set(collapsed_dims) | set(operand_batching_dims)
    window_operand_dims = tuple(
        axis for axis in range(len(operand_shape)) if axis not in excluded
    )
    if len(window_operand_dims) != len(offset_dims):
        raise NotImplementedError("unsupported gather window geometry")

    mode = eqn.params.get("mode")
    mode_name = (
        "PROMISE_IN_BOUNDS"
        if mode is None
        else getattr(mode, "name", str(mode)).rsplit(".", 1)[-1].upper()
    )
    if mode_name in {"FILL", "DROP"}:
        mode_name = "FILL_OR_DROP"
    if mode_name not in {"CLIP", "PROMISE_IN_BOUNDS", "FILL_OR_DROP"}:
        raise NotImplementedError(f"unsupported gather mode {mode_name!r}")

    offsets = np.zeros((n_output, len(operand_shape)), dtype=np.int64)
    for output_axis, operand_axis in zip(offset_dims, window_operand_dims):
        offsets[:, operand_axis] = output_coords[:, output_axis]

    fixed_starts = np.zeros((n_output, len(operand_shape)), dtype=np.int64)
    for component, operand_axis in enumerate(start_index_map):
        values = component_values[component]
        if values is not None:
            fixed_starts[:, operand_axis] = values
    for operand_axis, index_axis in zip(
        operand_batching_dims, indices_batching_dims, strict=True
    ):
        fixed_starts[:, operand_axis] = batch_coords[:, index_axis]

    if not dynamic_components:
        valid_mask = np.ones(n_output, dtype=bool)
        target_coords = np.empty((n_output, len(operand_shape)), dtype=np.int64)

        for axis, extent in enumerate(operand_shape):
            offset = offsets[:, axis]
            start = fixed_starts[:, axis]
            upper = upper_starts[axis]

            if mode_name in {"CLIP", "PROMISE_IN_BOUNDS"}:
                start = np.clip(start, 0, upper)
            elif axis in start_index_map:
                valid_mask &= (start >= 0) & (start <= upper)

            coord = start + offset
            valid_mask &= (coord >= 0) & (coord < extent)
            target_coords[:, axis] = coord

        source_rows = np.full(n_output, -1, dtype=np.int64)
        if np.any(valid_mask):
            if operand_shape:
                source_rows[valid_mask] = np.ravel_multi_index(
                    tuple(
                        target_coords[valid_mask, a] for a in range(len(operand_shape))
                    ),
                    operand_shape,
                )
            else:
                source_rows[valid_mask] = 0

        return GatherRouteFragment(
            output_rows=output_rows,
            source_rows=source_rows,
            index_rows=index_rows,
        )

    dynamic_axes = frozenset(start_index_map[c] for c in dynamic_components)
    source_demands: list[TensorDemand | None] = []

    for row in range(n_output):
        valid = True
        axes = []
        for axis, extent in enumerate(operand_shape):
            offset = int(offsets[row, axis])
            if axis in dynamic_axes:
                start = offset
                stop = int(upper_starts[axis]) + offset + 1
                axes.append(_axis_interval(extent, start, stop))
                continue

            start = int(fixed_starts[row, axis])
            if mode_name in {"CLIP", "PROMISE_IN_BOUNDS"}:
                start = min(max(start, 0), int(upper_starts[axis]))
            elif axis in start_index_map and (
                start < 0 or start > int(upper_starts[axis])
            ):
                valid = False
                break

            coordinate = start + offset
            if coordinate < 0 or coordinate >= extent:
                valid = False
                break
            axes.append(_axis_interval(extent, coordinate, coordinate + 1))

        source_demands.append(
            TensorDemand.from_axes(operand_shape, tuple(axes)) if valid else None
        )

    return GatherEnvelopeFragment(
        output_rows=output_rows,
        source_demands=tuple(source_demands),
        index_rows=index_rows,
        dynamic_components=tuple(dynamic_components),
    )


def _selector_rows(eqn: JaxprEqn, output_rows: NDArray[np.int64]) -> NDArray[np.int64]:
    selector_shape = _shape_of(eqn.invars[0])
    if not selector_shape:
        return np.zeros(output_rows.size, dtype=np.int64)
    output_shape = _shape_of(eqn.outvars[0])
    coords = np.stack(np.unravel_index(output_rows, output_shape), axis=1)
    coords = coords[:, -len(selector_shape) :]
    for axis, extent in enumerate(selector_shape):
        if extent == 1:
            coords[:, axis] = 0
    return np.ravel_multi_index(tuple(coords.T), selector_shape).astype(np.int64)


def select_route_concrete_demands(
    eqn: JaxprEqn, request: RouteRequest
) -> tuple[Demand, ...]:
    rows = _output_rows(eqn, request)
    demands: list[Demand] = [None] * len(eqn.invars)
    demands[0] = TensorDemand.from_rows_hull(
        _shape_of(eqn.invars[0]), _selector_rows(eqn, rows)
    )
    return tuple(demands)


def _starts(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    *,
    first_start: int,
    operand_shape: tuple[int, ...],
    region_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    values: list[int] = []
    for atom in eqn.invars[first_start:]:
        value = _read_concrete(concrete, atom)
        if value is None:
            return None
        values.append(int(np.asarray(_concrete_array(value))))
    max_starts = tuple(dim - size for dim, size in zip(operand_shape, region_shape))
    return tuple(
        min(max(value, 0), maximum) for value, maximum in zip(values, max_starts)
    )


def resolve_gather_route_fragment(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    request: RouteRequest,
) -> GatherRouteFragment | None:
    """Resolve gather geometry for only ``request.output_rows``.

    Requested row order and duplicates are preserved so callers can align
    labels or other sparse payloads directly with the returned arrays.
    """
    indices = _read_concrete(concrete, eqn.invars[1])
    if indices is None:
        return None

    output_rows = _output_rows(eqn, request)
    source_rows, index_rows = _compute_gather_route_rows(
        eqn,
        indices,
        output_rows,
    )
    return GatherRouteFragment(
        output_rows=output_rows,
        source_rows=source_rows,
        index_rows=index_rows,
    )


def resolve_scatter_route_fragment(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    request: RouteRequest,
) -> ScatterRouteFragment | None:
    """Resolve only scatter updates targeting requested output rows.

    Arbitrary scatter indices must still be inspected, but they are processed
    in bounded chunks. The persistent fragment contains only actual demanded
    update↔output incidences.
    """
    if len(eqn.invars) < 3:
        return None
    indices = _read_concrete(concrete, eqn.invars[1])
    if indices is None:
        return None
    output_rows = _output_rows(eqn, request)
    wanted = np.unique(output_rows)
    if wanted.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return ScatterRouteFragment(output_rows, empty, empty)

    n_updates = int(np.prod(_shape_of(eqn.invars[2]), dtype=np.int64))
    chunk_size = max(4096, min(n_updates, 8 * wanted.size))
    matched_updates: list[NDArray[np.int64]] = []
    matched_targets: list[NDArray[np.int64]] = []
    for start in range(0, n_updates, chunk_size):
        update_rows = np.arange(
            start, min(start + chunk_size, n_updates), dtype=np.int64
        )
        targets = _compute_scatter_target_rows(
            eqn, np.asarray(_concrete_array(indices)), update_rows
        )
        if targets is None:
            return None
        keep = np.isin(targets, wanted)
        if np.any(keep):
            matched_updates.append(update_rows[keep])
            matched_targets.append(targets[keep])

    if matched_updates:
        update_rows = np.concatenate(matched_updates)
        target_rows = np.concatenate(matched_targets)
    else:
        update_rows = np.empty(0, dtype=np.int64)
        target_rows = np.empty(0, dtype=np.int64)
    return ScatterRouteFragment(output_rows, update_rows, target_rows)


def resolve_select_n_route_fragment(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    request: RouteRequest,
) -> SelectNRouteFragment | None:
    if len(eqn.invars) < 2:
        return None
    selector = _read_concrete(concrete, eqn.invars[0])
    if selector is None:
        return None
    output_rows = _output_rows(eqn, request)
    output_shape = _shape_of(eqn.outvars[0])
    selector_shape = _shape_of(eqn.invars[0])
    if len(selector_shape) > len(output_shape):
        return None
    padded_shape = (1,) * (len(output_shape) - len(selector_shape)) + selector_shape
    if any(
        source not in (1, target) for source, target in zip(padded_shape, output_shape)
    ):
        return None
    if not selector_shape:
        selected = np.full(
            output_rows.size,
            int(np.asarray(_concrete_array(selector))),
            dtype=np.int64,
        )
    else:
        selector_rows = _selector_rows(eqn, output_rows)
        if isinstance(selector, ConcreteRegion):
            selected = np.asarray(
                selector.read_rows(selector_rows), dtype=np.int64
            ).ravel()
        else:
            selected = (
                np.asarray(selector).ravel()[selector_rows].astype(np.int64, copy=False)
            )
    n_cases = len(eqn.invars) - 1
    if np.any(selected < 0) or np.any(selected >= n_cases):
        return None
    return SelectNRouteFragment(output_rows, selected)


def resolve_dynamic_slice_route_fragment(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    request: RouteRequest,
) -> DynamicSliceRouteFragment | None:
    if len(eqn.invars) < 2 or len(eqn.outvars) != 1:
        return None
    operand_shape = _shape_of(eqn.invars[0])
    output_shape = _shape_of(eqn.outvars[0])
    starts = _starts(
        eqn,
        concrete,
        first_start=1,
        operand_shape=operand_shape,
        region_shape=output_shape,
    )
    if starts is None:
        return None
    output_rows = _output_rows(eqn, request)
    coords = np.stack(np.unravel_index(output_rows, output_shape), axis=1)
    coords += np.asarray(starts, dtype=np.int64)
    source_rows = np.ravel_multi_index(tuple(coords.T), operand_shape).astype(np.int64)
    return DynamicSliceRouteFragment(output_rows, source_rows)


def resolve_dynamic_update_slice_route_fragment(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
    request: RouteRequest,
) -> DynamicUpdateSliceRouteFragment | None:
    if len(eqn.invars) < 3 or len(eqn.outvars) != 1:
        return None
    operand_shape = _shape_of(eqn.invars[0])
    update_shape = _shape_of(eqn.invars[1])
    starts = _starts(
        eqn,
        concrete,
        first_start=2,
        operand_shape=operand_shape,
        region_shape=update_shape,
    )
    if starts is None:
        return None
    output_rows = _output_rows(eqn, request)
    coords = np.stack(np.unravel_index(output_rows, operand_shape), axis=1)
    local = coords - np.asarray(starts, dtype=np.int64)
    valid = np.all(local >= 0, axis=1) & np.all(
        local < np.asarray(update_shape, dtype=np.int64), axis=1
    )
    target_rows = output_rows[valid]
    update_rows = np.ravel_multi_index(tuple(local[valid].T), update_shape).astype(
        np.int64
    )
    return DynamicUpdateSliceRouteFragment(output_rows, update_rows, target_rows)
