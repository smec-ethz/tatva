"""Proven regional concrete-evaluation plans for ordinary primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from tatva.tracer.core.semantics import (
    DemandContext,
    FullConcrete,
    RegionalConcrete,
    RegionalConcreteContext,
    RegionalConcretePlan,
)
from tatva.tracer.local.demand import Demand
from tatva.tracer.local.layout import TensorLayout


def _bind_requested(ctx: RegionalConcreteContext, inputs: tuple[Any, ...]) -> Any:
    result = ctx.eqn.primitive.bind(*inputs, **ctx.eqn.params)
    outputs = tuple(result) if ctx.eqn.primitive.multiple_results else (result,)
    return np.asarray(outputs[ctx.output_index])


def regional_bind(
    demand_rule: Callable[[DemandContext], tuple[Demand, ...]],
):
    """Use ordinary demand propagation and bind on proven-compatible compact arrays."""

    def plan(_ctx: RegionalConcreteContext) -> RegionalConcretePlan:
        return RegionalConcrete(demand_rule, _bind_requested)

    return plan


def full(reason: str):
    def plan(_ctx: RegionalConcreteContext) -> RegionalConcretePlan:
        return FullConcrete(reason)

    return plan


def _reshape_requested(ctx: RegionalConcreteContext, inputs: tuple[Any, ...]) -> Any:
    return np.reshape(
        np.asarray(inputs[0]), TensorLayout.from_demand(ctx.demand).local_shape
    )


def regional_reshape(
    demand_rule: Callable[[DemandContext], tuple[Demand, ...]],
):
    def plan(ctx: RegionalConcreteContext) -> RegionalConcretePlan:
        outputs: list[Demand] = [None] * len(ctx.eqn.outvars)
        outputs[ctx.output_index] = ctx.demand
        inputs = demand_rule(DemandContext(ctx.eqn, tuple(outputs), None))
        demanded = [demand for demand in inputs if demand is not None]
        if len(demanded) != 1 or demanded[0].size != ctx.demand.size:
            return FullConcrete(
                "regional reshape would change the compact scalar count"
            )
        return RegionalConcrete(demand_rule, _reshape_requested)

    return plan


def _broadcast_requested(ctx: RegionalConcreteContext, inputs: tuple[Any, ...]) -> Any:
    value = np.asarray(inputs[0])
    local_shape = TensorLayout.from_demand(ctx.demand).local_shape
    dimensions = tuple(int(axis) for axis in ctx.eqn.params["broadcast_dimensions"])
    expanded = [1] * len(local_shape)
    for input_axis, output_axis in enumerate(dimensions):
        expanded[output_axis] = value.shape[input_axis]
    return np.broadcast_to(value.reshape(expanded), local_shape)


def regional_broadcast(
    demand_rule: Callable[[DemandContext], tuple[Demand, ...]],
):
    def plan(_ctx: RegionalConcreteContext) -> RegionalConcretePlan:
        return RegionalConcrete(demand_rule, _broadcast_requested)

    return plan


def _projected_unary(ctx: RegionalConcreteContext, inputs: tuple[Any, ...]) -> Any:
    return np.reshape(
        np.asarray(inputs[0]), TensorLayout.from_demand(ctx.demand).local_shape
    )


def regional_projected_unary(
    demand_rule: Callable[[DemandContext], tuple[Demand, ...]],
):
    """The demanded input is already the requested output in compact order."""

    def plan(_ctx: RegionalConcreteContext) -> RegionalConcretePlan:
        return RegionalConcrete(demand_rule, _projected_unary)

    return plan


def _iota_requested(ctx: RegionalConcreteContext, _inputs: tuple[Any, ...]) -> Any:
    layout = TensorLayout.from_demand(ctx.demand)
    dimension = int(ctx.eqn.params["dimension"])
    values = layout.global_axis_indices(dimension)
    shape = [1] * len(layout.local_shape)
    shape[dimension] = layout.local_shape[dimension]
    dtype = cast(Any, ctx.eqn.outvars[ctx.output_index].aval).dtype
    return np.broadcast_to(values.reshape(shape), layout.local_shape).astype(
        dtype, copy=False
    )


def regional_iota(_ctx: RegionalConcreteContext) -> RegionalConcretePlan:
    def no_inputs(_demand_ctx: DemandContext) -> tuple[Demand, ...]:
        return ()

    return RegionalConcrete(no_inputs, _iota_requested)
