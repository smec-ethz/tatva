"""
Recursive structured backward demand propagation.

The liveness pass starts from rank-owned contribution tensors and walks
backwards through the materialized JAXPR instance tree.

Ordinary primitives delegate to their registered DemandRule. Nested calls,
maps, and scans recurse explicitly because each materialized invocation may
have different resolved structural routes.

Demands always remain structured TensorDemand objects between equations.
Exact flattened scalar rows are used only transiently inside individual
primitive demand rules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from jax.extend.core import Literal, Var
from numpy.typing import NDArray

from tatva.tracer.analysis import (
    CallPlan,
    MapPlan,
    ScanPlan,
)
from tatva.tracer.contributions import ValueRef
from tatva.tracer.demand import (
    Demand,
    TensorDemand,
    axis_indices,
    lift_leading_axis_demand,
    merge_demands,
    take_leading_axis_demand,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.materialize import (
    CallInstance,
    FrameStep,
    JaxprInstance,
    MapInstance,
    ResolvedEqn,
    ScanInstance,
)
from tatva.tracer.registry import SEMANTICS
from tatva.tracer.semantics import (
    DemandContext,
)


@dataclass(frozen=True, slots=True)
class DemandSeed:
    value: ValueRef
    demand: TensorDemand


@dataclass(frozen=True)
class CallDemandTrace:
    body: JaxprDemandTrace


@dataclass(frozen=True)
class MapIterationDemandTrace:
    index: int
    body: JaxprDemandTrace


@dataclass(frozen=True)
class MapDemandTrace:
    iterations: tuple[MapIterationDemandTrace, ...]


@dataclass(frozen=True)
class ScanIterationDemandTrace:
    index: int
    body: JaxprDemandTrace


@dataclass(frozen=True)
class ScanDemandTrace:
    iterations: tuple[ScanIterationDemandTrace, ...]


type NestedDemandTrace = CallDemandTrace | MapDemandTrace | ScanDemandTrace


@dataclass(frozen=True)
class JaxprDemandTrace:
    demands: dict[Var, TensorDemand]
    input_demands: tuple[Demand, ...]
    nested: dict[int, NestedDemandTrace]


@dataclass(frozen=True)
class DemandTrace:
    root: JaxprDemandTrace


@dataclass
class _SeedNode:
    values: dict[Var, TensorDemand] = field(default_factory=dict)
    children: dict[FrameStep, _SeedNode] = field(default_factory=dict)


def _build_seed_tree(
    seeds: Iterable[DemandSeed],
) -> _SeedNode:
    root = _SeedNode()

    for seed in seeds:
        node = root

        for step in seed.value.path:
            node = node.children.setdefault(step, _SeedNode())

        existing = node.values.get(seed.value.var)
        merged = merge_demands(existing, seed.demand)

        assert merged is not None
        node.values[seed.value.var] = merged

    return root


def _add_demand(
    demands: dict[Var, TensorDemand],
    atom,
    demand: Demand,
) -> None:
    if demand is None:
        return

    if isinstance(atom, Literal):
        return

    if not isinstance(atom, Var):
        raise TypeError(f"unsupported atom {type(atom)!r}")

    existing = demands.get(atom)
    merged = merge_demands(existing, demand)
    assert merged is not None

    demands[atom] = merged


def _demand_fraction(demand: Demand) -> float:
    if demand is None:
        return 0.0

    total = int(math.prod(demand.shape))
    if total == 0:
        return 0.0

    return demand.size / total


def _backprop_ordinary(
    resolved: ResolvedEqn,
    output_demands: tuple[Demand, ...],
) -> tuple[Demand, ...]:
    eqn = resolved.plan.eqn
    rule = SEMANTICS.get(eqn.primitive)

    result = rule.demand(
        DemandContext(
            eqn=eqn,
            output_demands=output_demands,
            route=resolved.route,
        )
    )

    max_output_fraction = max(
        (_demand_fraction(demand) for demand in output_demands), default=0.0
    )
    for input_index, demand in enumerate(result):
        if demand is None:
            continue

        fraction = _demand_fraction(demand)
        if fraction == 1.0 and max_output_fraction < 1.0:
            print(
                "DEMAND WIDENING:",
                eqn.primitive.name,
                f"input={input_index}",
                f"input_shape={demand.shape}",
                f"fraction={fraction:.3f}",
                f"max_output_fraction={max_output_fraction:.3f}",
            )

    if len(result) != len(eqn.invars):
        raise RuntimeError(
            f"demand rule for {eqn.primitive.name!r} "
            f"returned {len(result)} inputs; "
            f"expected {len(eqn.invars)}"
        )

    return result


def _backprop_call(
    resolved: ResolvedEqn,
    output_demands: tuple[Demand, ...],
    child_seed: _SeedNode | None,
) -> tuple[
    tuple[Demand, ...],
    CallDemandTrace,
]:
    nested = resolved.nested
    plan = resolved.plan.nested

    if not isinstance(nested, CallInstance):
        raise TypeError("expected CallInstance")

    if not isinstance(plan, CallPlan):
        raise TypeError("expected CallPlan")

    child = _backprop_jaxpr(
        nested.body,
        child_seed or _SeedNode(),
        output_demands=output_demands,
    )

    return (
        child.input_demands,
        CallDemandTrace(body=child),
    )


def _map_iteration(
    nested: MapInstance,
    plan: MapPlan,
    logical_index: int,
):
    if logical_index < 0 or logical_index >= plan.length:
        raise IndexError(logical_index)

    position = plan.length - 1 - logical_index if plan.reverse else logical_index
    iteration = nested.iterations[position]

    if iteration.index != logical_index:
        raise RuntimeError("MapInstance iteration ordering invariant violated")

    return iteration


def _map_demanded_indices(
    plan: MapPlan,
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
    *,
    eqn_index: int,
) -> NDArray[np.int64]:
    parts = []

    for demand in output_demands:
        if demand is None:
            continue

        first = demand.axis_subset(0)
        parts.append(axis_indices(first, extent=plan.length))

    explicit = [
        step.iteration
        for step in seed_node.children
        if (
            step.eqn_index == eqn_index
            and step.kind == "map"
            and step.iteration is not None
        )
    ]

    if explicit:
        parts.append(np.asarray(explicit, dtype=np.int64))

    if not parts:
        return np.empty(0, dtype=np.int64)

    return np.unique(np.concatenate(parts))


def _backprop_map(
    resolved: ResolvedEqn,
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> tuple[
    tuple[Demand, ...],
    MapDemandTrace,
]:
    nested = resolved.nested
    plan = resolved.plan.nested
    eqn = resolved.plan.eqn

    if not isinstance(nested, MapInstance):
        raise TypeError("expected MapInstance")

    if not isinstance(plan, MapPlan):
        raise TypeError("expected MapPlan")

    input_demands: list[Demand] = [None] * len(eqn.invars)
    traces: list[MapIterationDemandTrace] = []
    indices = _map_demanded_indices(
        plan,
        output_demands,
        seed_node,
        eqn_index=resolved.plan.index,
    )

    for logical_index in indices:
        logical_index = int(logical_index)
        iteration = _map_iteration(nested, plan, logical_index)
        child_step = FrameStep(
            eqn_index=resolved.plan.index,
            kind="map",
            iteration=logical_index,
        )

        child_seed = seed_node.children.get(child_step, _SeedNode())

        body_outputs = tuple(
            take_leading_axis_demand(demand, logical_index) for demand in output_demands
        )

        child = _backprop_jaxpr(iteration.body, child_seed, output_demands=body_outputs)

        traces.append(
            MapIterationDemandTrace(
                index=logical_index,
                body=child,
            )
        )

        for input_index, demand in enumerate(child.input_demands):
            if demand is None:
                continue

            if input_index < plan.num_consts:
                lifted = demand

            else:
                outer_shape = _shape_of(eqn.invars[input_index])
                lifted = lift_leading_axis_demand(
                    demand,
                    outer_shape=outer_shape,
                    index=logical_index,
                )

            input_demands[input_index] = merge_demands(
                input_demands[input_index], lifted
            )

    return (
        tuple(input_demands),
        MapDemandTrace(iterations=tuple(traces)),
    )


def _backprop_scan(
    resolved: ResolvedEqn,
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> tuple[
    tuple[Demand, ...],
    ScanDemandTrace,
]:
    nested = resolved.nested
    plan = resolved.plan.nested
    eqn = resolved.plan.eqn

    if not isinstance(nested, ScanInstance):
        raise TypeError("expected ScanInstance")

    if not isinstance(plan, ScanPlan):
        raise TypeError("expected ScanPlan")

    if plan.num_carry <= 0:
        raise RuntimeError("carry-free scan should be MapPlan")

    num_consts = plan.num_consts
    num_carry = plan.num_carry

    num_xs = len(eqn.invars) - num_consts - num_carry
    carry_demands = list(output_demands[:num_carry])
    y_demands = output_demands[num_carry:]
    const_demands: list[Demand] = [None] * num_consts
    xs_demands: list[Demand] = [None] * num_xs
    traces: list[ScanIterationDemandTrace] = []

    # nested.iterations is execution order.
    # Backward liveness runs opposite execution order.
    for iteration in reversed(nested.iterations):
        logical_index = iteration.index

        child_step = FrameStep(
            eqn_index=resolved.plan.index,
            kind="scan",
            iteration=logical_index,
        )
        child_seed = seed_node.children.get(child_step)
        y_step_demands = tuple(
            take_leading_axis_demand(demand, logical_index) for demand in y_demands
        )

        body_outputs = tuple(carry_demands) + y_step_demands

        if all(demand is None for demand in body_outputs) and child_seed is None:
            continue

        child = _backprop_jaxpr(
            iteration.body,
            child_seed or _SeedNode(),
            output_demands=body_outputs,
        )

        traces.append(ScanIterationDemandTrace(index=logical_index, body=child))
        inputs = child.input_demands

        # Shared constants.
        for i in range(num_consts):
            const_demands[i] = merge_demands(const_demands[i], inputs[i])

        # Carry demand propagates to the previous iteration in
        # execution order.
        carry_demands = list(inputs[num_consts : num_consts + num_carry])

        # Scanned inputs.
        x_start = num_consts + num_carry

        for x_index in range(num_xs):
            demand = inputs[x_start + x_index]

            if demand is None:
                continue

            outer_input_index = x_start + x_index
            lifted = lift_leading_axis_demand(
                demand,
                outer_shape=_shape_of(eqn.invars[outer_input_index]),
                index=logical_index,
            )
            xs_demands[x_index] = merge_demands(xs_demands[x_index], lifted)

    result = tuple(const_demands) + tuple(carry_demands) + tuple(xs_demands)

    return (
        result,
        ScanDemandTrace(
            iterations=tuple(traces),
        ),
    )


def _backprop_jaxpr(
    instance: JaxprInstance,
    seed_node: _SeedNode,
    *,
    output_demands: tuple[Demand, ...] | None = None,
) -> JaxprDemandTrace:
    jaxpr = instance.plan.jaxpr
    demands: dict[Var, TensorDemand] = {}
    nested_traces: dict[int, NestedDemandTrace] = {}

    # Explicit contribution/internal seeds in this frame.
    for var, demand in seed_node.values.items():
        _add_demand(demands, var, demand)

    # Demands arriving through a nested wrapper boundary.
    if output_demands is not None:
        if len(output_demands) != len(jaxpr.outvars):
            raise ValueError(
                f"received {len(output_demands)} output demands "
                f"for JAXPR with {len(jaxpr.outvars)} outputs"
            )

        for atom, demand in zip(jaxpr.outvars, output_demands):
            _add_demand(demands, atom, demand)

    # --------------------------------------------------------------
    # Reverse topological traversal.
    # --------------------------------------------------------------

    for resolved in reversed(instance.eqns):
        eqn = resolved.plan.eqn
        eqn_output_demands = tuple(demands.get(outvar) for outvar in eqn.outvars)
        eqn_index = resolved.plan.index

        # ----------------------------------------------------------
        # Ordinary primitive
        # ----------------------------------------------------------

        if resolved.nested is None:
            if all(demand is None for demand in eqn_output_demands):
                continue

            input_demands = _backprop_ordinary(resolved, eqn_output_demands)

        # ----------------------------------------------------------
        # call / remat
        # ----------------------------------------------------------

        elif isinstance(resolved.nested, CallInstance):
            child_step = FrameStep(
                eqn_index=eqn_index,
                kind="call",
            )
            child_seed = seed_node.children.get(child_step)

            if (
                all(demand is None for demand in eqn_output_demands)
                and child_seed is None
            ):
                continue

            (input_demands, nested_trace) = _backprop_call(
                resolved, eqn_output_demands, child_seed
            )
            nested_traces[eqn_index] = nested_trace

        # ----------------------------------------------------------
        # independent map
        # ----------------------------------------------------------

        elif isinstance(resolved.nested, MapInstance):
            has_child_seed = any(
                step.eqn_index == eqn_index and step.kind == "map"
                for step in seed_node.children
            )

            if (
                all(demand is None for demand in eqn_output_demands)
                and not has_child_seed
            ):
                continue

            (input_demands, nested_trace) = _backprop_map(
                resolved, eqn_output_demands, seed_node
            )
            nested_traces[eqn_index] = nested_trace

        # ----------------------------------------------------------
        # recurrent scan
        # ----------------------------------------------------------

        elif isinstance(resolved.nested, ScanInstance):
            has_child_seed = any(
                step.eqn_index == eqn_index and step.kind == "scan"
                for step in seed_node.children
            )

            if (
                all(demand is None for demand in eqn_output_demands)
                and not has_child_seed
            ):
                continue

            (input_demands, nested_trace) = _backprop_scan(
                resolved, eqn_output_demands, seed_node
            )
            nested_traces[eqn_index] = nested_trace

        else:
            raise TypeError(f"unsupported nested instance {type(resolved.nested)!r}")

        # ----------------------------------------------------------
        # Merge producer requirements.
        # ----------------------------------------------------------

        if len(input_demands) != len(eqn.invars):
            raise RuntimeError(
                f"{eqn.primitive.name!r} demand propagation "
                "returned the wrong number of inputs"
            )

        for atom, demand in zip(eqn.invars, input_demands):
            _add_demand(demands, atom, demand)

    frame_inputs = tuple(demands.get(var) for var in jaxpr.invars)

    return JaxprDemandTrace(
        demands=demands,
        input_demands=frame_inputs,
        nested=nested_traces,
    )


def backpropagate_demand(
    root: JaxprInstance,
    seeds: Iterable[DemandSeed],
) -> DemandTrace:
    seed_tree = _build_seed_tree(seeds)

    return DemandTrace(
        root=_backprop_jaxpr(
            root,
            seed_tree,
        )
    )
