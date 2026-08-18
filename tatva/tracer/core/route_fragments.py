"""Demand-scoped route geometry for sparse planning traversals.

Full routes remain the source of truth for materialization and localization.
Fragments expose the same geometry for explicitly requested global rows without
allocating a mapping proportional to the complete logical output.
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
from tatva.tracer.local.demand import Demand, TensorDemand


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
    coords = np.stack(np.unravel_index(output_rows, output_shape), axis=1)
    batch = coords[:, output_batch_dims]
    vector_size = indices_shape[-1]
    components = np.tile(np.arange(vector_size, dtype=np.int64), len(output_rows))
    flattened = [
        np.repeat(batch[:, axis], vector_size) for axis in range(len(indices_shape) - 1)
    ]
    flattened.append(components)
    return np.ravel_multi_index(tuple(flattened), indices_shape).reshape(
        len(output_rows), vector_size
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
