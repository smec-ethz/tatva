"""
Localization of globally resolved routes.

Global routes are expressed in scalar rows of the original tensors.
Localization converts them into scalar rows of rank-local TensorLayouts.

A missing valid source row is an error: backward liveness should have ensured
that every source required by a locally stored output is present locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

from tatva.tracer.core.route_fragments import (
    DynamicSliceRouteFragment,
    GatherEnvelopeFragment,
    GatherRouteFragment,
    ScatterRouteFragment,
    SelectNRouteFragment,
)
from tatva.tracer.core.routes import (
    DynamicSliceRoute,
    GatherRoute,
    ScatterRoute,
    SelectNRoute,
)
from tatva.tracer.local.demand import _FullAxis
from tatva.tracer.local.layout import TensorLayout


@dataclass(frozen=True, slots=True, eq=False)
class LocalGatherRoute:
    """
    Gather geometry expressed entirely in local scalar rows.

    `source_rows[i]` is the flattened local operand row used to produce
    flattened local output row `i`.

    -1 preserves an invalid/fill entry from the global route.
    """

    source_rows: NDArray[np.int64]
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(
            self.source_rows,
            dtype=np.int64,
        ).ravel()

        output_shape = tuple(int(extent) for extent in self.output_shape)

        if any(extent < 0 for extent in output_shape):
            raise ValueError(f"negative output extent in {output_shape}")

        expected = int(prod(output_shape))

        if rows.size != expected:
            raise ValueError(
                f"LocalGatherRoute has {rows.size} source rows "
                f"for output shape {output_shape} "
                f"with {expected} entries"
            )

        if np.any(rows < -1):
            raise ValueError("LocalGatherRoute source rows must be >= -1")

        rows = rows.copy()
        rows.flags.writeable = False

        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(self, "output_shape", output_shape)


@dataclass(frozen=True, slots=True)
class LocalGatherIndexRebase:
    component: int
    operand_axis: int
    global_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        values = np.asarray(self.global_indices, dtype=np.int64).ravel().copy()
        if values.size == 0:
            raise ValueError("gather index rebase cannot use an empty axis")
        if values.size > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError("gather rebase coordinates must be strictly increasing")
        values.flags.writeable = False
        object.__setattr__(self, "global_indices", values)


@dataclass(frozen=True, slots=True)
class LocalDynamicGatherRoute:
    """Localized gather retaining runtime-dependent payload indices."""

    output_shape: tuple[int, ...]
    slice_sizes: tuple[int, ...]
    rebases: tuple[LocalGatherIndexRebase, ...]

    @classmethod
    def from_fragment(
        cls,
        eqn,
        route: GatherEnvelopeFragment,
        *,
        operand_layout: TensorLayout,
        index_layout: TensorLayout,
        output_layout: TensorLayout,
    ) -> "LocalDynamicGatherRoute":
        global_output_rows = output_layout.local_rows_to_global_rows(
            np.arange(output_layout.local_size, dtype=np.int64)
        )
        positions = np.searchsorted(route.output_rows, global_output_rows)
        if np.any(positions >= route.output_rows.size) or np.any(
            route.output_rows[positions] != global_output_rows
        ):
            raise ValueError("gather envelope does not cover the local output")

        start_index_map = tuple(
            int(x) for x in eqn.params["dimension_numbers"].start_index_map
        )
        dynamic = frozenset(route.dynamic_components)
        rebases: list[LocalGatherIndexRebase] = []
        for component, operand_axis in enumerate(start_index_map):
            subset = operand_layout.axis_subset(operand_axis)
            if isinstance(subset, _FullAxis):
                continue
            if component in dynamic:
                raise RuntimeError(
                    "runtime gather component would index a localized structural "
                    f"axis: component {component}, operand axis {operand_axis}"
                )
            rebases.append(
                LocalGatherIndexRebase(
                    component=component,
                    operand_axis=operand_axis,
                    global_indices=operand_layout.global_axis_indices(operand_axis),
                )
            )

        slice_sizes = list(int(x) for x in eqn.params["slice_sizes"])
        for axis, (global_extent, local_extent) in enumerate(
            zip(operand_layout.global_shape, operand_layout.local_shape, strict=True)
        ):
            if global_extent == local_extent:
                continue
            if slice_sizes[axis] == global_extent:
                slice_sizes[axis] = local_extent
            elif slice_sizes[axis] != 1:
                raise NotImplementedError(
                    "localized runtime gather cannot rewrite a partial window on "
                    f"operand axis {axis}: slice size {slice_sizes[axis]}, "
                    f"global extent {global_extent}, local extent {local_extent}"
                )

        if index_layout.local_shape[-1] != index_layout.global_shape[-1]:
            raise RuntimeError("localized gather dropped index-vector components")

        return cls(
            output_shape=output_layout.local_shape,
            slice_sizes=tuple(slice_sizes),
            rebases=tuple(rebases),
        )


def localize_unary_source_rows(
    source_rows: NDArray[np.int64],
    *,
    operand_layout: TensorLayout,
    output_layout: TensorLayout,
    allow_invalid: bool,
) -> NDArray[np.int64]:
    """Restrict a global unary output->input scalar-row relation to the local
    output and rewrite its source rows into the operand's local coordinates.
    """
    global_sources = np.asarray(source_rows, dtype=np.int64).ravel()
    if global_sources.size != output_layout.global_size:
        raise ValueError(
            "source-row route/output layout mismatch: "
            f"{global_sources.size} != "
            f"{output_layout.global_size}"
        )

    local_output_rows = np.arange(output_layout.local_size, dtype=np.int64)
    global_output_rows = output_layout.local_rows_to_global_rows(local_output_rows)
    selected_sources = global_sources[global_output_rows]
    result = np.full(selected_sources.shape, -1, dtype=np.int64)

    if allow_invalid:
        valid = selected_sources >= 0
    else:
        if np.any(selected_sources < 0):
            raise ValueError("route contains invalid source rows")
        valid = np.ones(selected_sources.shape, dtype=bool)

    if np.any(valid):
        try:
            result[valid] = operand_layout.global_rows_to_local_rows(
                selected_sources[valid]
            )

        except ValueError as exc:
            missing = np.unique(selected_sources[valid])
            raise ValueError(
                "required global source row is not present "
                "in the operand layout; rows include "
                f"{missing[:8].tolist()}"
            ) from exc

    return result


def localize_gather_route(
    route: GatherRoute | GatherRouteFragment,
    *,
    operand_layout: TensorLayout,
    output_layout: TensorLayout,
) -> LocalGatherRoute:
    """Restrict a global GatherRoute to the locally stored output and rewrite
    its source rows into local operand coordinates.

    Steps:

        local output row
            -> global output row
            -> global source row
            -> local source row

    Valid global source rows must exist in `operand_layout`. If they do not,
    liveness and route localization disagree and localization fails.
    """
    if isinstance(route, GatherRouteFragment):
        global_output_rows = output_layout.local_rows_to_global_rows(
            np.arange(output_layout.local_size, dtype=np.int64)
        )
        positions = np.searchsorted(route.output_rows, global_output_rows)

        if np.any(positions >= route.output_rows.size) or np.any(
            route.output_rows[positions] != global_output_rows
        ):
            raise ValueError("gather fragment does not cover the local output")

        sources = route.source_rows[positions]
        local_rows = np.full(sources.shape, -1, dtype=np.int64)
        valid = sources >= 0
        local_rows[valid] = operand_layout.global_rows_to_local_rows(sources[valid])

    else:
        local_rows = localize_unary_source_rows(
            route.source_rows,
            operand_layout=operand_layout,
            output_layout=output_layout,
            allow_invalid=True,
        )

    return LocalGatherRoute(
        source_rows=local_rows,
        output_shape=output_layout.local_shape,
    )


@dataclass(frozen=True, slots=True, eq=False)
class LocalScatterRoute:
    """Exact local scalar relation for a scatter.

    operand_rows[k]
        local scalar row read from the operand

    operand_output_rows[k]
        local output row receiving that operand value

    update_rows[k]
        local scalar row read from updates

    target_rows[k]
        local output row targeted by that update

    Input layouts may be structured hulls larger than the subset actually
    consumed by this equation, so these arrays need not cover every local
    operand/update row.
    """

    operand_rows: NDArray[np.int64]
    operand_output_rows: NDArray[np.int64]

    update_rows: NDArray[np.int64]
    target_rows: NDArray[np.int64]

    operand_shape: tuple[int, ...] | None
    update_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        operand_rows = np.asarray(self.operand_rows, dtype=np.int64).ravel()
        operand_output_rows = np.asarray(
            self.operand_output_rows, dtype=np.int64
        ).ravel()
        update_rows = np.asarray(self.update_rows, dtype=np.int64).ravel()
        target_rows = np.asarray(self.target_rows, dtype=np.int64).ravel()

        operand_shape = (
            None
            if self.operand_shape is None
            else tuple(int(x) for x in self.operand_shape)
        )

        update_shape = (
            None
            if self.update_shape is None
            else tuple(int(x) for x in self.update_shape)
        )

        output_shape = tuple(int(x) for x in self.output_shape)

        if operand_rows.shape != operand_output_rows.shape:
            raise ValueError("operand row arrays must have equal shape")

        if update_rows.shape != target_rows.shape:
            raise ValueError("update row arrays must have equal shape")

        output_size = int(prod(output_shape))

        if operand_shape is None:
            if operand_rows.size:
                raise ValueError("dead operand cannot have local rows")
        else:
            operand_size = int(prod(operand_shape))

            if np.any((operand_rows < 0) | (operand_rows >= operand_size)):
                raise ValueError("operand rows out of bounds")

        if update_shape is None:
            if update_rows.size:
                raise ValueError("dead updates cannot have local rows")
        else:
            update_size = int(prod(update_shape))

            if np.any((update_rows < 0) | (update_rows >= update_size)):
                raise ValueError("update rows out of bounds")

        if np.any((operand_output_rows < 0) | (operand_output_rows >= output_size)):
            raise ValueError("operand output rows out of bounds")

        if np.any((target_rows < 0) | (target_rows >= output_size)):
            raise ValueError("scatter target rows out of bounds")

        arrays = (operand_rows, operand_output_rows, update_rows, target_rows)
        arrays = tuple(array.copy() for array in arrays)

        for array in arrays:
            array.flags.writeable = False

        (operand_rows, operand_output_rows, update_rows, target_rows) = arrays

        object.__setattr__(self, "operand_rows", operand_rows)
        object.__setattr__(self, "operand_output_rows", operand_output_rows)
        object.__setattr__(self, "update_rows", update_rows)
        object.__setattr__(self, "target_rows", target_rows)
        object.__setattr__(self, "operand_shape", operand_shape)
        object.__setattr__(self, "update_shape", update_shape)
        object.__setattr__(self, "output_shape", output_shape)


def localize_scatter_route(
    route: ScatterRoute | ScatterRouteFragment,
    *,
    operand_layout: TensorLayout | None,
    update_layout: TensorLayout | None,
    output_layout: TensorLayout,
) -> LocalScatterRoute:
    """Restrict a global ScatterRoute to the exact scalar relations needed by
    the local output.

    TensorLayouts are structured hulls, so operand/update layouts may contain
    extra entries that this equation does not consume. Those entries are
    intentionally omitted from the localized route.
    """
    global_targets = np.asarray(route.target_rows, dtype=np.int64).ravel()

    if np.any(global_targets < -1):
        raise ValueError("scatter route contains invalid sentinel smaller than -1")

    # Operand -> output intersection
    if operand_layout is None:
        operand_rows = np.empty(0, dtype=np.int64)
        operand_output_rows = np.empty(0, dtype=np.int64)
        operand_shape = None

    else:
        if operand_layout.global_shape != output_layout.global_shape:
            raise ValueError(
                "scatter operand/output global shape mismatch: "
                f"{operand_layout.global_shape} != "
                f"{output_layout.global_shape}"
            )

        candidate_operand_rows = np.arange(operand_layout.local_size, dtype=np.int64)
        global_operand_rows = operand_layout.local_rows_to_global_rows(
            candidate_operand_rows
        )

        # Some stored operand rows can lie outside this equation's
        # output layout because operand_layout is a structured hull.
        candidate_output_rows = output_layout.global_rows_to_local_rows(
            global_operand_rows,
            allow_missing=True,
        )

        live = candidate_output_rows >= 0
        operand_rows = candidate_operand_rows[live]
        operand_output_rows = candidate_output_rows[live]
        operand_shape = operand_layout.local_shape

    # Update -> output intersection
    if update_layout is None:
        update_rows = np.empty(0, dtype=np.int64)
        target_rows = np.empty(0, dtype=np.int64)
        update_shape = None

    else:
        if isinstance(route, ScatterRouteFragment):
            global_update_rows = np.asarray(route.update_rows, dtype=np.int64)
            candidate_update_rows = update_layout.global_rows_to_local_rows(
                global_update_rows
            )
            selected_global_targets = global_targets

        else:
            if global_targets.size != update_layout.global_size:
                raise ValueError(
                    "ScatterRoute/update global shape mismatch: "
                    f"route has {global_targets.size} update rows, "
                    f"update tensor has {update_layout.global_size}"
                )
            candidate_update_rows = np.arange(update_layout.local_size, dtype=np.int64)
            global_update_rows = update_layout.local_rows_to_global_rows(
                candidate_update_rows
            )
            selected_global_targets = global_targets[global_update_rows]

        candidate_target_rows = output_layout.global_rows_to_local_rows(
            selected_global_targets,
            allow_missing=True,
        )

        # This removes both:
        #
        #   - globally invalid targets (-1)
        #   - valid targets outside this local output
        #
        # Both can appear because update_layout is a structured hull.
        live = candidate_target_rows >= 0
        update_rows = candidate_update_rows[live]
        target_rows = candidate_target_rows[live]
        update_shape = update_layout.local_shape

    # Coverage
    covered = np.zeros(output_layout.local_size, dtype=bool)
    covered[operand_output_rows] = True
    covered[target_rows] = True

    if not np.all(covered):
        missing_local = np.flatnonzero(~covered)
        missing_global = output_layout.local_rows_to_global_rows(missing_local)

        raise ValueError(
            "localized scatter cannot produce every "
            "local output row; missing global rows "
            f"{missing_global[:8].tolist()}"
        )

    return LocalScatterRoute(
        operand_rows=operand_rows,
        operand_output_rows=operand_output_rows,
        update_rows=update_rows,
        target_rows=target_rows,
        operand_shape=operand_shape,
        update_shape=update_shape,
        output_shape=output_layout.local_shape,
    )


@dataclass(frozen=True, slots=True, eq=False)
class LocalDynamicSliceRoute:
    """Dynamic-slice geometry expressed in local scalar rows.

    source_rows[i]
        local operand scalar row producing local output scalar row i.
    """

    source_rows: NDArray[np.int64]
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(self.source_rows, dtype=np.int64).ravel()

        output_shape = tuple(int(x) for x in self.output_shape)
        expected = int(prod(output_shape))
        if rows.size != expected:
            raise ValueError(f"expected {expected} source rows, got {rows.size}")

        if np.any(rows < 0):
            raise ValueError(
                "localized dynamic_slice cannot contain invalid source rows"
            )

        rows = rows.copy()
        rows.flags.writeable = False

        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(self, "output_shape", output_shape)


def localize_dynamic_slice_route(
    route: DynamicSliceRoute | DynamicSliceRouteFragment,
    *,
    operand_layout: TensorLayout,
    output_layout: TensorLayout,
) -> LocalDynamicSliceRoute:
    if isinstance(route, DynamicSliceRouteFragment):
        global_output_rows = output_layout.local_rows_to_global_rows(
            np.arange(output_layout.local_size, dtype=np.int64)
        )
        positions = np.searchsorted(route.output_rows, global_output_rows)

        if np.any(positions >= route.output_rows.size) or np.any(
            route.output_rows[positions] != global_output_rows
        ):
            raise ValueError("dynamic-slice fragment does not cover local output")

        local_rows = operand_layout.global_rows_to_local_rows(
            route.source_rows[positions]
        )

    else:
        local_rows = localize_unary_source_rows(
            route.source_rows,
            operand_layout=operand_layout,
            output_layout=output_layout,
            allow_invalid=False,
        )

    return LocalDynamicSliceRoute(
        source_rows=local_rows,
        output_shape=output_layout.local_shape,
    )


@dataclass(frozen=True, slots=True, eq=False)
class LocalSelectCaseRoute:
    output_rows: NDArray[np.int64]
    source_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        output_rows = np.asarray(self.output_rows, dtype=np.int64).ravel()
        source_rows = np.asarray(self.source_rows, dtype=np.int64).ravel()

        if output_rows.shape != source_rows.shape:
            raise ValueError("select case output/source row shapes differ")

        output_rows = output_rows.copy()
        source_rows = source_rows.copy()

        output_rows.flags.writeable = False
        source_rows.flags.writeable = False

        object.__setattr__(self, "output_rows", output_rows)
        object.__setattr__(self, "source_rows", source_rows)


@dataclass(frozen=True, slots=True)
class LocalSelectNRoute:
    cases: tuple[LocalSelectCaseRoute, ...]
    output_shape: tuple[int, ...]


def _broadcast_output_rows_to_input_rows(
    output_rows: NDArray[np.int64],
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> NDArray[np.int64]:
    """Map global flattened output rows through ordinary right-aligned
    broadcasting to flattened input rows.
    """
    output_rows = np.asarray(output_rows, dtype=np.int64).ravel()

    if len(input_shape) > len(output_shape):
        raise ValueError(f"cannot broadcast {input_shape} -> {output_shape}")

    if not input_shape:
        return np.zeros(output_rows.shape, dtype=np.int64)

    output_coords = np.unravel_index(output_rows, output_shape)
    offset = len(output_shape) - len(input_shape)

    input_coords = []

    for input_axis, input_extent in enumerate(input_shape):
        output_axis = offset + input_axis
        output_extent = output_shape[output_axis]

        if input_extent == 1:
            input_coords.append(np.zeros(output_rows.shape, dtype=np.int64))

        elif input_extent == output_extent:
            input_coords.append(output_coords[output_axis])

        else:
            raise ValueError(f"invalid broadcast {input_shape} -> {output_shape}")

    return np.asarray(
        np.ravel_multi_index(tuple(input_coords), input_shape), dtype=np.int64
    )


def localize_select_n_route(
    route: SelectNRoute | SelectNRouteFragment,
    *,
    case_layouts: tuple[TensorLayout | None, ...],
    output_layout: TensorLayout,
) -> LocalSelectNRoute:
    local_output_rows = np.arange(output_layout.local_size, dtype=np.int64)
    global_output_rows = output_layout.local_rows_to_global_rows(local_output_rows)

    if isinstance(route, SelectNRouteFragment):
        positions = np.searchsorted(route.output_rows, global_output_rows)
        if np.any(positions >= route.output_rows.size) or np.any(
            route.output_rows[positions] != global_output_rows
        ):
            raise ValueError("select_n fragment does not cover local output")

        selected_cases = route.case_indices[positions]

    else:
        global_case_indices = np.asarray(route.case_indices, dtype=np.int64).ravel()
        if global_case_indices.size != output_layout.global_size:
            raise ValueError("SelectNRoute/output shape mismatch")

        selected_cases = global_case_indices[global_output_rows]

    n_cases = len(case_layouts)

    if np.any((selected_cases < 0) | (selected_cases >= n_cases)):
        raise ValueError("select_n route contains invalid case index")

    cases: list[LocalSelectCaseRoute] = []
    covered = np.zeros(output_layout.local_size, dtype=bool)

    for case_index, case_layout in enumerate(case_layouts):
        output_rows = np.flatnonzero(selected_cases == case_index)

        if output_rows.size == 0:
            cases.append(
                LocalSelectCaseRoute(
                    output_rows=np.empty(0, dtype=np.int64),
                    source_rows=np.empty(0, dtype=np.int64),
                )
            )
            continue

        if case_layout is None:
            raise ValueError(
                f"select_n case {case_index} is required but has no local layout"
            )

        global_rows = global_output_rows[output_rows]
        global_case_rows = _broadcast_output_rows_to_input_rows(
            global_rows,
            input_shape=case_layout.global_shape,
            output_shape=output_layout.global_shape,
        )

        try:
            source_rows = case_layout.global_rows_to_local_rows(global_case_rows)

        except ValueError as exc:
            raise ValueError(
                f"select_n case {case_index} requires "
                "values absent from its local layout"
            ) from exc

        covered[output_rows] = True
        cases.append(
            LocalSelectCaseRoute(
                output_rows=output_rows,
                source_rows=source_rows,
            )
        )

    if not np.all(covered):
        raise RuntimeError("localized select_n does not cover all local output rows")

    return LocalSelectNRoute(
        cases=tuple(cases),
        output_shape=output_layout.local_shape,
    )


type LocalRoute = (
    LocalGatherRoute
    | LocalDynamicGatherRoute
    | LocalScatterRoute
    | LocalDynamicSliceRoute
    | LocalSelectNRoute
)
