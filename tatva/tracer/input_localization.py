from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
from jax.extend.core import Var

from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.contributions import ValueRef
from tatva.tracer.demand import TensorDemand, merge_demands
from tatva.tracer.halo import HaloPlan
from tatva.tracer.helpers import _shape_of
from tatva.tracer.layout import TensorLayout
from tatva.tracer.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local_plan import LocalJaxprPlan
from tatva.tracer.lowering import extract_local_value
from tatva.tracer.model import GatherRoute


def _resolve_batched_root_index_demand(
    *,
    local_plan: LocalJaxprPlan,
    seeds: tuple[DemandSeed, ...],
) -> tuple[int, TensorDemand]:
    if not seeds:
        raise ValueError("at least one gather index seed is required")

    trace = backpropagate_demand(
        local_plan.instance,
        seeds,
    )

    roots = [
        (flat_index, demand)
        for flat_index, demand in enumerate(trace.input_demands)
        if demand is not None
    ]

    if not roots:
        raise NotImplementedError(
            "gather indices do not originate from a captured input"
        )

    if len(roots) != 1:
        indices = [flat_index for flat_index, _ in roots]
        raise NotImplementedError(
            "batched gather indices resolve to multiple captured inputs: "
            f"{indices}; tagged localization demands are required to preserve "
            "the gather-to-input association"
        )

    return roots[0]


class InputAction(Protocol):
    def apply(self, value: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Keep:
    def apply(self, value: Any) -> Any:
        return value


@dataclass(frozen=True, slots=True)
class Slice:
    layout: TensorLayout

    def apply(self, value: Any) -> Any:
        return extract_local_value(value, self.layout)


@dataclass(frozen=True, slots=True)
class DofStorageSlice:
    global_dofs: np.ndarray

    def apply(self, value: Any) -> Any:
        return jnp.asarray(value)[jnp.asarray(self.global_dofs)]


@dataclass(frozen=True, slots=True)
class SliceAndRemapIndices:
    index_layout: TensorLayout
    operand_layout: TensorLayout
    operand_axis: int

    def apply(self, value: Any) -> Any:
        local = extract_local_value(value, self.index_layout)
        mapped = self.operand_layout.global_to_local_axis(
            self.operand_axis,
            np.asarray(local),
        )
        return jnp.asarray(mapped, dtype=local.dtype)


@dataclass(frozen=True, slots=True)
class InputLocalizationPlan:
    actions: tuple[InputAction, ...]

    def apply_flat(self, flat: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(flat) != len(self.actions):
            raise ValueError(
                f"expected {len(self.actions)} input leaves, got {len(flat)}"
            )
        return tuple(
            action.apply(value)
            for action, value in zip(self.actions, flat, strict=True)
        )


@dataclass
class _IndexRequest:
    demand: TensorDemand
    operand_layout: TensorLayout
    operand_axis: int


def _merge_index_request(
    requests: dict[int, _IndexRequest],
    *,
    flat_index: int,
    demand: TensorDemand,
    operand_layout: TensorLayout,
    operand_axis: int,
) -> None:
    previous = requests.get(flat_index)
    if previous is None:
        requests[flat_index] = _IndexRequest(
            demand=demand,
            operand_layout=operand_layout,
            operand_axis=operand_axis,
        )
        return

    if previous.operand_axis != operand_axis:
        raise NotImplementedError("one index leaf addresses multiple operand axes")

    left = previous.operand_layout.global_axis_indices(operand_axis)
    right = operand_layout.global_axis_indices(operand_axis)
    if not np.array_equal(left, right):
        raise ValueError("one index leaf is used with incompatible local operand maps")

    merged = merge_demands(previous.demand, demand)
    assert merged is not None
    previous.demand = merged


def _same_operand_axis_map(
    left: _IndexRequest,
    *,
    operand_layout: TensorLayout,
    operand_axis: int,
) -> bool:
    if left.operand_axis != operand_axis:
        return False

    left_indices = left.operand_layout.global_axis_indices(operand_axis)
    right_indices = operand_layout.global_axis_indices(operand_axis)
    return np.array_equal(left_indices, right_indices)


def build_input_localization_plan(
    *,
    captured: CapturedJaxpr,
    local_plan: LocalJaxprPlan,
    halo: HaloPlan,
) -> InputLocalizationPlan:
    if local_plan.instance.plan.jaxpr is not captured.jaxpr:
        raise ValueError("input localization currently requires the root JAXPR")

    requests: dict[int, _IndexRequest] = {}
    index_seeds: list[DemandSeed] = []
    batched_request: _IndexRequest | None = None

    for eqn_plan in local_plan.eqns:
        route_plan = eqn_plan.route
        if route_plan is None or not isinstance(route_plan.global_route, GatherRoute):
            continue

        route = route_plan.global_route
        if route.index_rows is None:
            raise RuntimeError("gather route has no retained index-row provenance")

        eqn = eqn_plan.eqn
        data_var, index_var = eqn.invars[:2]
        if not isinstance(data_var, Var) or not isinstance(index_var, Var):
            continue

        operand_layout = eqn_plan.input_layouts[0]
        output_layout = eqn_plan.output_layouts[0]
        if operand_layout is None or output_layout is None:
            continue

        dnums = eqn.params["dimension_numbers"]
        addressed_axes = tuple(int(x) for x in dnums.start_index_map)
        if len(set(addressed_axes)) != 1:
            raise NotImplementedError(
                "automatic input remapping currently requires a gather whose "
                "index components address one operand axis"
            )
        operand_axis = addressed_axes[0]

        local_output_rows = np.arange(output_layout.local_size, dtype=np.int64)
        global_output_rows = output_layout.local_rows_to_global_rows(local_output_rows)
        index_rows = np.unique(route.index_rows[global_output_rows].ravel())
        index_demand = TensorDemand.from_rows_hull(
            _shape_of(index_var),
            index_rows,
        )
        if index_demand is None:
            continue

        candidate = _IndexRequest(
            demand=index_demand,
            operand_layout=operand_layout,
            operand_axis=operand_axis,
        )

        if batched_request is None:
            batched_request = candidate
        elif not _same_operand_axis_map(
            batched_request,
            operand_layout=operand_layout,
            operand_axis=operand_axis,
        ):
            raise NotImplementedError(
                "gathers in one batched localization pass use incompatible "
                "operand axis maps; tagged localization demands are required"
            )

        index_seeds.append(
            DemandSeed(
                value=ValueRef(path=(), var=index_var),
                demand=index_demand,
            )
        )

    # Resolve all locally relevant gather indices in one backward traversal.
    # These demands build input reconstruction actions only; they never enter
    # the runtime layouts or executable liveness trace.
    if index_seeds:
        assert batched_request is not None
        index_flat, root_demand = _resolve_batched_root_index_demand(
            local_plan=local_plan,
            seeds=tuple(index_seeds),
        )

        _merge_index_request(
            requests,
            flat_index=index_flat,
            demand=root_demand,
            operand_layout=batched_request.operand_layout,
            operand_axis=batched_request.operand_axis,
        )

    actions: list[InputAction] = []
    for index, layout in enumerate(local_plan.input_layouts):
        request = requests.get(index)

        if index == 0:
            actions.append(DofStorageSlice(halo.storage.global_dofs))
        elif request is not None:
            actions.append(
                SliceAndRemapIndices(
                    index_layout=TensorLayout.from_demand(request.demand),
                    operand_layout=request.operand_layout,
                    operand_axis=request.operand_axis,
                )
            )
        elif layout is not None:
            actions.append(Slice(layout))
        else:
            actions.append(Keep())

    return InputLocalizationPlan(tuple(actions))
