"""Contribution-block to global-DOF incidence construction.

Contribution domains are split into deterministic blocks. ``TaggedDemand``
then propagates their sparse entry/block relations together in one reverse
traversal. The normal distributed-planning path walks ``JaxprPlan`` templates
and creates only temporary concrete frames for demanded nested iterations.
Consumers can therefore partition computation without consulting
``DerivativeTrace`` or constructing a Hessian.

The older per-block and materialized tagged traversals remain available as
correctness oracles while localization still consumes ``JaxprInstance``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Literal, Var
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.nested import (
    CallContext,
    CallInvocation,
    CallSpec,
    CondContext,
    CondInvocation,
    CondSpec,
    FrameStep,
    IndexedChild,
    LinearSolveContext,
    LinearSolveInvocation,
    LinearSolveSpec,
    MapContext,
    MapSpec,
    NestedKind,
    RepeatedInvocation,
    ScanContext,
    ScanSpec,
    TraversalOrder,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteRequest
from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import TaggedDemandContext, no_route_fragment
from tatva.tracer.core.tagged import (
    Tagged,
    TaggedDemand,
    active_blocks,
    merge_tagged,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.liveness import (
    DemandSeed,
    backpropagate_demand,
)
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver
from tatva.tracer.program.contributions import (
    ContributionBlock,
    ContributionTrace,
    ValueRef,
)
from tatva.tracer.program.materialize import JaxprInstance, ResolvedEqn


@dataclass(frozen=True, slots=True)
class TaggedDemandSeed:
    value: ValueRef
    demand: TaggedDemand


@dataclass(frozen=True)
class JaxprTaggedTrace:
    demands: dict[Var, TaggedDemand]
    input_demands: tuple[Tagged, ...]
    nested: dict[int, object]


@dataclass
class _TaggedSeedNode:
    values: dict[Var, TaggedDemand] = field(default_factory=dict)
    children: dict[FrameStep, _TaggedSeedNode] = field(default_factory=dict)

    def block_ids(self) -> NDArray[np.int64]:
        parts = [demand.block_ids for demand in self.values.values()]
        parts.extend(child.block_ids() for child in self.children.values())
        parts = [part for part in parts if part.size]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))


def _build_tagged_seed_tree(
    seeds: Iterable[TaggedDemandSeed],
) -> _TaggedSeedNode:
    root = _TaggedSeedNode()
    for seed in seeds:
        value = seed.value
        path = value.path
        var = value.var
        node = root
        for step in path:
            node = node.children.setdefault(step, _TaggedSeedNode())
        merged = merge_tagged(node.values.get(var), seed.demand)
        assert merged is not None
        node.values[var] = merged
    return root


def _add_tagged(
    demands: dict[Var, TaggedDemand],
    atom,
    demand: Tagged,
) -> None:
    if demand is None or isinstance(atom, Literal):
        return
    if not isinstance(atom, Var):
        raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")
    if demand.shape != _shape_of(atom):
        raise ValueError(
            f"tagged demand shape {demand.shape} does not match atom shape "
            f"{_shape_of(atom)}"
        )
    merged = merge_tagged(demands.get(atom), demand)
    assert merged is not None
    demands[atom] = merged


def _backprop_tagged_ordinary(
    resolved: ResolvedEqn,
    output_demands: tuple[Tagged, ...],
    concrete,
) -> tuple[Tagged, ...]:
    eqn = resolved.plan.eqn
    rule = SEMANTICS.get_ordinary(eqn.primitive)
    fragment = None
    if rule.route_fragment is not no_route_fragment:
        active_rows = np.unique(
            np.concatenate(
                [demand.rows for demand in output_demands if demand is not None]
            )
        )
        fragment = rule.route_fragment(eqn, concrete, RouteRequest(active_rows))
    result = rule.tagged_demand(
        TaggedDemandContext(
            eqn=eqn,
            output_demands=output_demands,
            route=resolved.route if fragment is None else fragment,
        )
    )
    if len(result) != len(eqn.invars):
        raise RuntimeError(
            f"tagged demand rule for {eqn.primitive.name!r} returned "
            f"{len(result)} inputs; expected {len(eqn.invars)}"
        )
    return result


def _backprop_tagged_call(
    resolved: ResolvedEqn,
    context: CallContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    child_seed: _TaggedSeedNode | None,
) -> tuple[tuple[Tagged, ...], CallInvocation[JaxprTaggedTrace]]:
    child = _backprop_tagged_jaxpr(
        context.invocation.body,
        child_seed or _TaggedSeedNode(),
        output_demands=output_demands,
    )
    outer: list[Tagged] = [None] * len(resolved.plan.eqn.invars)
    indices = context.spec.resolved_input_indices(len(outer))
    if len(indices) != len(child.input_demands):
        raise RuntimeError("call child tagged-demand/input boundary mismatch")
    for outer_index, demand in zip(indices, child.input_demands, strict=True):
        outer[outer_index] = merge_tagged(outer[outer_index], demand)
    return tuple(outer), CallInvocation(context.invocation.eqn_index, child)


def _backprop_tagged_cond(
    resolved: ResolvedEqn,
    context: CondContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    child_seed: _TaggedSeedNode | None,
) -> tuple[tuple[Tagged, ...], CondInvocation[JaxprTaggedTrace]]:
    child = _backprop_tagged_jaxpr(
        context.invocation.body,
        child_seed or _TaggedSeedNode(),
        output_demands=output_demands,
    )
    outer: list[Tagged] = [None] * len(resolved.plan.eqn.invars)
    for child_index, demand in enumerate(child.input_demands):
        outer_index = context.spec.outer_input_index(
            child_index, outer_arity=len(outer)
        )
        outer[outer_index] = merge_tagged(outer[outer_index], demand)
    return tuple(outer), CondInvocation(
        context.invocation.eqn_index,
        context.invocation.branch_index,
        child,
    )


def _tagged_map_indices(
    context: MapContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> NDArray[np.int64]:
    parts: list[NDArray[np.int64]] = []
    for demand in output_demands:
        if demand is None:
            continue
        child_size = int(math.prod(demand.shape[1:]))
        parts.append(np.unique(demand.rows // child_size))

    child_steps = {child.frame_step for child in context.invocation.children()}
    explicit = [
        step.iteration
        for step in seed_node.children
        if step in child_steps and step.iteration is not None
    ]
    if explicit:
        parts.append(np.asarray(explicit, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


def _backprop_tagged_map(
    resolved: ResolvedEqn,
    context: MapContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], RepeatedInvocation[JaxprTaggedTrace]]:
    eqn = resolved.plan.eqn
    spec = context.spec
    outer: list[Tagged] = [None] * len(eqn.invars)
    children = []

    for logical_index_ in _tagged_map_indices(context, output_demands, seed_node):
        logical_index = int(logical_index_)
        child_step = context.invocation.frame_step(logical_index)
        child_outputs = tuple(
            demand.take_leading_axis(logical_index) if demand is not None else None
            for demand in output_demands
        )
        child = _backprop_tagged_jaxpr(
            context.invocation.child_at_index(logical_index),
            seed_node.children.get(child_step, _TaggedSeedNode()),
            output_demands=child_outputs,
        )
        children.append(IndexedChild(logical_index, child))

        for input_index, demand in enumerate(child.input_demands):
            if demand is None:
                continue
            lifted = (
                demand
                if input_index < spec.num_consts
                else demand.lift_leading_axis(
                    outer_shape=_shape_of(eqn.invars[input_index]),
                    index=logical_index,
                )
            )
            outer[input_index] = merge_tagged(outer[input_index], lifted)

    return tuple(outer), context.invocation.with_children(tuple(children))


def _backprop_tagged_scan(
    resolved: ResolvedEqn,
    context: ScanContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], RepeatedInvocation[JaxprTaggedTrace]]:
    eqn = resolved.plan.eqn
    spec = context.spec
    if spec.num_carry <= 0:
        raise RuntimeError("carry-free scan should be represented as a map")

    num_xs = len(eqn.invars) - spec.num_consts - spec.num_carry
    carry = list(output_demands[: spec.num_carry])
    ys = output_demands[spec.num_carry :]
    consts: list[Tagged] = [None] * spec.num_consts
    xs: list[Tagged] = [None] * num_xs
    children = []

    for nested_child in context.invocation.children(TraversalOrder.REVERSE_EXECUTION):
        logical_index = nested_child.logical_index
        assert logical_index is not None
        child_seed = seed_node.children.get(nested_child.frame_step)
        y_step = tuple(
            demand.take_leading_axis(logical_index) if demand is not None else None
            for demand in ys
        )
        child_outputs = tuple(carry) + y_step
        if all(demand is None for demand in child_outputs) and child_seed is None:
            continue

        child = _backprop_tagged_jaxpr(
            nested_child.payload,
            child_seed or _TaggedSeedNode(),
            output_demands=child_outputs,
        )
        children.append(IndexedChild(logical_index, child))
        inputs = child.input_demands
        for index in range(spec.num_consts):
            consts[index] = merge_tagged(consts[index], inputs[index])
        carry = list(inputs[spec.num_consts : spec.num_consts + spec.num_carry])

        x_start = spec.num_consts + spec.num_carry
        for x_index in range(num_xs):
            demand = inputs[x_start + x_index]
            if demand is None:
                continue
            lifted = demand.lift_leading_axis(
                outer_shape=_shape_of(eqn.invars[x_start + x_index]),
                index=logical_index,
            )
            xs[x_index] = merge_tagged(xs[x_index], lifted)

    return (
        tuple(consts) + tuple(carry) + tuple(xs),
        context.invocation.with_children(tuple(children)),
    )


def _backprop_tagged_linear_solve(
    resolved: ResolvedEqn,
    context: LinearSolveContext[JaxprInstance],
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], LinearSolveInvocation[JaxprTaggedTrace]]:
    eqn = resolved.plan.eqn
    outer: list[Tagged] = [None] * len(eqn.invars)
    output_blocks = active_blocks(output_demands)
    block_ids = np.union1d(output_blocks, seed_node.block_ids())
    child_traces = []
    output = output_demands[0] if len(output_demands) == 1 else None
    batch_shape = (
        output.shape[:-2] if output is not None and len(output.shape) >= 3 else ()
    )
    supported = output is not None and bool(batch_shape)
    if supported:
        for callback, child in zip(
            context.spec.callbacks(), context.invocation.children(), strict=True
        ):
            runtime_index = next(
                (
                    index
                    for index, binding in enumerate(callback.inputs)
                    if binding.runtime
                ),
                None,
            )
            supported &= (
                len(child.payload.plan.jaxpr.outvars) == 1
                and _shape_of(child.payload.plan.jaxpr.outvars[0])[: len(batch_shape)]
                == batch_shape
                and runtime_index is not None
                and _shape_of(child.payload.plan.jaxpr.invars[runtime_index])
                == output.shape
            )
    if not supported:
        warnings.warn(
            "custom_linear_solve does not have a supported batched matrix "
            "layout; using conservative full callback demands",
            UserWarning,
            stacklevel=2,
        )
        batch_demand = None
    else:
        assert output is not None
        local_size = int(math.prod(output.shape[len(batch_shape) :]))
        batch_demand = TaggedDemand(
            batch_shape,
            output.rows // local_size,
            output.blocks,
        )

    def required(shape: Shape) -> Tagged:
        if batch_demand is None or shape[: len(batch_shape)] != batch_shape:
            return TaggedDemand.full(shape, block_ids)
        local_size = int(math.prod(shape[len(batch_shape) :]))
        rows = np.repeat(batch_demand.rows * local_size, local_size) + np.tile(
            np.arange(local_size, dtype=np.int64), batch_demand.nnz
        )
        return TaggedDemand(
            shape,
            rows,
            np.repeat(batch_demand.blocks, local_size),
        )

    for callback, child_node in zip(
        context.spec.callbacks(), context.invocation.children(), strict=True
    ):
        callback_outputs = tuple(
            required(_shape_of(atom)) for atom in child_node.payload.plan.jaxpr.outvars
        )

        child = _backprop_tagged_jaxpr(
            child_node.payload,
            seed_node.children.get(child_node.frame_step, _TaggedSeedNode()),
            output_demands=callback_outputs,
        )
        child_traces.append(child)

        for binding, demand in zip(callback.inputs, child.input_demands, strict=True):
            if binding.runtime:
                if callback.name == "solve" and demand is not None:
                    rhs = context.spec.rhs_indices[0]
                    outer[rhs] = merge_tagged(outer[rhs], demand)
            elif demand is not None:
                index = binding.outer_input_index
                assert index is not None
                outer[index] = merge_tagged(outer[index], demand)

        # Captures are executable closure operands and remain live even when a
        # callback contains an opaque structural rule.
        for binding in callback.inputs:
            index = binding.outer_input_index
            if index is None:
                continue
            outer[index] = merge_tagged(
                outer[index], required(_shape_of(eqn.invars[index]))
            )

    return tuple(outer), LinearSolveInvocation(
        context.invocation.eqn_index,
        *child_traces,
    )


@dataclass(frozen=True)
class _TaggedNestedHandler:
    resolved: ResolvedEqn
    output_demands: tuple[Tagged, ...]
    seed_node: _TaggedSeedNode

    def call(self, context: CallContext[JaxprInstance]):
        child_step = context.invocation.children()[0].frame_step
        return _backprop_tagged_call(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node.children.get(child_step),
        )

    def map(self, context: MapContext[JaxprInstance]):
        return _backprop_tagged_map(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node,
        )

    def scan(self, context: ScanContext[JaxprInstance]):
        return _backprop_tagged_scan(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node,
        )

    def cond(self, context: CondContext[JaxprInstance]):
        child_step = context.invocation.children()[0].frame_step
        return _backprop_tagged_cond(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node.children.get(child_step),
        )

    def linear_solve(self, context: LinearSolveContext[JaxprInstance]):
        return _backprop_tagged_linear_solve(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node,
        )


def _backprop_tagged_jaxpr(
    instance: JaxprInstance,
    seed_node: _TaggedSeedNode,
    *,
    output_demands: tuple[Tagged, ...] | None = None,
) -> JaxprTaggedTrace:
    jaxpr = instance.plan.jaxpr
    demands: dict[Var, TaggedDemand] = {}
    nested_traces: dict[int, object] = {}

    for var, demand in seed_node.values.items():
        _add_tagged(demands, var, demand)

    if output_demands is not None:
        if len(output_demands) != len(jaxpr.outvars):
            raise ValueError(
                f"received {len(output_demands)} tagged output demands for "
                f"JAXPR with {len(jaxpr.outvars)} outputs"
            )
        for atom, demand in zip(jaxpr.outvars, output_demands, strict=True):
            _add_tagged(demands, atom, demand)

    for resolved in reversed(instance.eqns):
        eqn = resolved.plan.eqn
        outputs = tuple(demands.get(outvar) for outvar in eqn.outvars)

        if resolved.nested is None:
            if all(demand is None for demand in outputs):
                continue
            inputs = _backprop_tagged_ordinary(resolved, outputs, instance.concrete)
        else:
            nested_plan = resolved.plan.nested
            if nested_plan is None:
                raise TypeError("nested invocation has no analysis plan")
            has_child_seed = any(
                child.frame_step in seed_node.children
                for child in resolved.nested.children()
            )
            if all(demand is None for demand in outputs) and not has_child_seed:
                continue
            inputs, nested_trace = dispatch_nested(
                nested_plan.spec,
                resolved.nested,
                _TaggedNestedHandler(resolved, outputs, seed_node),
            )
            nested_traces[resolved.plan.index] = nested_trace

        if len(inputs) != len(eqn.invars):
            raise RuntimeError(
                f"{eqn.primitive.name!r} tagged propagation returned the wrong "
                "number of inputs"
            )
        for atom, demand in zip(eqn.invars, inputs, strict=True):
            _add_tagged(demands, atom, demand)

    return JaxprTaggedTrace(
        demands=demands,
        input_demands=tuple(demands.get(var) for var in jaxpr.invars),
        nested=nested_traces,
    )


def backpropagate_tagged_demand(
    root: JaxprInstance,
    seeds: Iterable[TaggedDemandSeed],
) -> JaxprTaggedTrace:
    """Propagate all contribution-block labels in one reverse traversal."""
    return _backprop_tagged_jaxpr(root, _build_tagged_seed_tree(seeds))


# -----------------------------------------------------------------------------
# Plan-driven tagged traversal
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaggedTraversalSummary:
    kind: NestedKind
    visited_indices: tuple[int, ...] = ()


def _backprop_plan_ordinary(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    output_demands: tuple[Tagged, ...],
) -> tuple[Tagged, ...]:
    eqn = eqn_plan.eqn
    rule = SEMANTICS.get_ordinary(eqn.primitive)
    fragment = None
    if rule.route_fragment is not no_route_fragment:
        active_rows = np.unique(
            np.concatenate(
                [demand.rows for demand in output_demands if demand is not None]
            )
        )
        fragment = resolver.route_fragment(frame, eqn_plan, RouteRequest(active_rows))
    route = resolver.route(frame, eqn_plan) if fragment is None else fragment
    result = rule.tagged_demand(
        TaggedDemandContext(
            eqn=eqn,
            output_demands=output_demands,
            route=route,
        )
    )
    if len(result) != len(eqn.invars):
        raise RuntimeError(
            f"tagged demand rule for {eqn.primitive.name!r} returned "
            f"{len(result)} inputs; expected {len(eqn.invars)}"
        )
    return result


def _seed_child(seed_node: _TaggedSeedNode, step: FrameStep) -> _TaggedSeedNode | None:
    return seed_node.children.get(step)


def _backprop_plan_call(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: CallSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], TaggedTraversalSummary]:
    child_frame = resolver.call_frame(frame, eqn_plan)
    step = child_frame.path[-1]
    try:
        child = _backprop_tagged_plan(
            child_frame.plan,
            child_frame,
            resolver,
            _seed_child(seed_node, step) or _TaggedSeedNode(),
            output_demands=output_demands,
        )
    finally:
        resolver.release(child_frame)
    outer: list[Tagged] = [None] * len(eqn_plan.eqn.invars)
    indices = spec.resolved_input_indices(len(outer))
    if len(indices) != len(child.input_demands):
        raise RuntimeError("call child tagged-demand/input boundary mismatch")
    for outer_index, demand in zip(indices, child.input_demands, strict=True):
        outer[outer_index] = merge_tagged(outer[outer_index], demand)
    return tuple(outer), TaggedTraversalSummary(NestedKind.CALL)


def _backprop_plan_cond(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: CondSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], TaggedTraversalSummary]:
    branch_index, child_frame = resolver.cond_frame(frame, eqn_plan)
    step = child_frame.path[-1]
    try:
        child = _backprop_tagged_plan(
            child_frame.plan,
            child_frame,
            resolver,
            _seed_child(seed_node, step) or _TaggedSeedNode(),
            output_demands=output_demands,
        )
    finally:
        resolver.release(child_frame)
    outer: list[Tagged] = [None] * len(eqn_plan.eqn.invars)
    for child_index, demand in enumerate(child.input_demands):
        outer_index = spec.outer_input_index(
            child_index, outer_arity=len(eqn_plan.eqn.invars)
        )
        outer[outer_index] = merge_tagged(outer[outer_index], demand)
    return tuple(outer), TaggedTraversalSummary(NestedKind.COND, (branch_index,))


def _plan_map_indices(
    eqn_plan: EqnPlan,
    spec: MapSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> NDArray[np.int64]:
    parts: list[NDArray[np.int64]] = []
    for demand in output_demands:
        if demand is None:
            continue
        child_size = int(math.prod(demand.shape[1:]))
        parts.append(np.unique(demand.rows // child_size))
    explicit = [
        step.iteration
        for step in seed_node.children
        if step.eqn_index == eqn_plan.index
        and step.kind is NestedKind.MAP
        and step.iteration is not None
    ]
    if explicit:
        parts.append(np.asarray(explicit, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    indices = np.unique(np.concatenate(parts))
    if np.any(indices < 0) or np.any(indices >= spec.length):
        raise IndexError("map tagged seed contains an out-of-range iteration")
    return indices


def _backprop_plan_map(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: MapSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], TaggedTraversalSummary]:
    eqn = eqn_plan.eqn
    outer: list[Tagged] = [None] * len(eqn.invars)
    visited: list[int] = []
    for index_ in _plan_map_indices(eqn_plan, spec, output_demands, seed_node):
        index = int(index_)
        child_outputs = tuple(
            demand.take_leading_axis(index) if demand is not None else None
            for demand in output_demands
        )
        child_frame = resolver.map_frame(frame, eqn_plan, index)
        step = child_frame.path[-1]
        try:
            child = _backprop_tagged_plan(
                child_frame.plan,
                child_frame,
                resolver,
                _seed_child(seed_node, step) or _TaggedSeedNode(),
                output_demands=child_outputs,
            )
        finally:
            resolver.release(child_frame)
        visited.append(index)
        for input_index, demand in enumerate(child.input_demands):
            if demand is None:
                continue
            lifted = (
                demand
                if input_index < spec.num_consts
                else demand.lift_leading_axis(
                    outer_shape=_shape_of(eqn.invars[input_index]), index=index
                )
            )
            outer[input_index] = merge_tagged(outer[input_index], lifted)
    return tuple(outer), TaggedTraversalSummary(NestedKind.MAP, tuple(visited))


def _scan_active_prefix(
    eqn_plan: EqnPlan,
    spec: ScanSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[int, ...]:
    execution = spec.execution_indices()
    if not execution:
        return ()
    positions = {logical: position for position, logical in enumerate(execution)}
    active_positions: list[int] = []
    if any(demand is not None for demand in output_demands[: spec.num_carry]):
        active_positions.append(len(execution) - 1)
    for demand in output_demands[spec.num_carry :]:
        if demand is None:
            continue
        child_size = int(math.prod(demand.shape[1:]))
        for logical in np.unique(demand.rows // child_size):
            logical_index = int(logical)
            if logical_index not in positions:
                raise IndexError("scan demand contains an out-of-range iteration")
            active_positions.append(positions[logical_index])
    for step in seed_node.children:
        if (
            step.eqn_index == eqn_plan.index
            and step.kind is NestedKind.SCAN
            and step.iteration is not None
        ):
            if step.iteration not in positions:
                raise IndexError("scan tagged seed contains an out-of-range iteration")
            active_positions.append(positions[step.iteration])
    if not active_positions:
        return ()
    return execution[: max(active_positions) + 1]


def _scan_carry_snapshots(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: ScanSpec,
    execution_prefix: tuple[int, ...],
) -> dict[int, dict[int, object]]:
    nested = eqn_plan.nested
    assert nested is not None
    required_carry = {
        input_index - spec.num_consts
        for input_index in nested.body.concrete_inputs
        if spec.num_consts <= input_index < spec.num_consts + spec.num_carry
    }
    if not required_carry:
        return {}
    carry = {
        carry_index: resolver.value(
            frame,
            eqn_plan.eqn.invars[spec.num_consts + carry_index],
        )
        for carry_index in required_carry
    }
    snapshots: dict[int, dict[int, object]] = {}
    for logical_index in execution_prefix:
        snapshots[logical_index] = dict(carry)
        child_frame = resolver.scan_frame(frame, eqn_plan, logical_index, carry)
        try:
            carry = {
                carry_index: resolver.value(
                    child_frame, child_frame.plan.jaxpr.outvars[carry_index]
                )
                for carry_index in required_carry
            }
        finally:
            resolver.release(child_frame)
    return snapshots


def _backprop_plan_scan(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: ScanSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], TaggedTraversalSummary]:
    if spec.num_carry <= 0:
        raise RuntimeError("carry-free scan should be represented as a map")
    eqn = eqn_plan.eqn
    execution_prefix = _scan_active_prefix(eqn_plan, spec, output_demands, seed_node)
    if not execution_prefix:
        if spec.length == 0:
            outer = [None] * len(eqn.invars)
            for carry_index, demand in enumerate(output_demands[: spec.num_carry]):
                outer[spec.num_consts + carry_index] = demand
            return tuple(outer), TaggedTraversalSummary(NestedKind.SCAN)
        return (
            tuple(None for _ in eqn.invars),
            TaggedTraversalSummary(NestedKind.SCAN),
        )

    snapshots = _scan_carry_snapshots(eqn_plan, frame, resolver, spec, execution_prefix)
    num_xs = len(eqn.invars) - spec.num_consts - spec.num_carry
    carry = list(output_demands[: spec.num_carry])
    ys = output_demands[spec.num_carry :]
    consts: list[Tagged] = [None] * spec.num_consts
    xs: list[Tagged] = [None] * num_xs
    visited: list[int] = []

    for logical_index in reversed(execution_prefix):
        step = FrameStep(eqn_plan.index, NestedKind.SCAN, logical_index)
        child_seed = _seed_child(seed_node, step)
        y_step = tuple(
            demand.take_leading_axis(logical_index) if demand is not None else None
            for demand in ys
        )
        child_outputs = tuple(carry) + y_step
        if all(demand is None for demand in child_outputs) and child_seed is None:
            continue
        child_frame = resolver.scan_frame(
            frame, eqn_plan, logical_index, snapshots.get(logical_index, {})
        )
        try:
            child = _backprop_tagged_plan(
                child_frame.plan,
                child_frame,
                resolver,
                child_seed or _TaggedSeedNode(),
                output_demands=child_outputs,
            )
        finally:
            resolver.release(child_frame)
        visited.append(logical_index)
        inputs = child.input_demands
        for index in range(spec.num_consts):
            consts[index] = merge_tagged(consts[index], inputs[index])
        carry = list(inputs[spec.num_consts : spec.num_consts + spec.num_carry])
        x_start = spec.num_consts + spec.num_carry
        for x_index in range(num_xs):
            demand = inputs[x_start + x_index]
            if demand is None:
                continue
            lifted = demand.lift_leading_axis(
                outer_shape=_shape_of(eqn.invars[x_start + x_index]),
                index=logical_index,
            )
            xs[x_index] = merge_tagged(xs[x_index], lifted)
    return (
        tuple(consts) + tuple(carry) + tuple(xs),
        TaggedTraversalSummary(NestedKind.SCAN, tuple(visited)),
    )


def _backprop_plan_linear_solve(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: LinearSolveSpec,
    output_demands: tuple[Tagged, ...],
    seed_node: _TaggedSeedNode,
) -> tuple[tuple[Tagged, ...], TaggedTraversalSummary]:
    eqn = eqn_plan.eqn
    outer: list[Tagged] = [None] * len(eqn.invars)
    output_blocks = active_blocks(output_demands)
    block_ids = np.union1d(output_blocks, seed_node.block_ids())
    output = output_demands[0] if len(output_demands) == 1 else None
    batch_shape = (
        output.shape[:-2] if output is not None and len(output.shape) >= 3 else ()
    )
    frames = resolver.linear_solve_frames(frame, eqn_plan)
    supported = output is not None and bool(batch_shape)
    if supported:
        for callback, child_frame in zip(spec.callbacks(), frames, strict=True):
            runtime_index = next(
                (i for i, binding in enumerate(callback.inputs) if binding.runtime),
                None,
            )
            supported &= (
                len(child_frame.plan.jaxpr.outvars) == 1
                and _shape_of(child_frame.plan.jaxpr.outvars[0])[: len(batch_shape)]
                == batch_shape
                and runtime_index is not None
                and _shape_of(child_frame.plan.jaxpr.invars[runtime_index])
                == output.shape
            )
    if not supported:
        warnings.warn(
            "custom_linear_solve does not have a supported batched matrix "
            "layout; using conservative full callback demands",
            UserWarning,
            stacklevel=2,
        )
        batch_demand = None
    else:
        assert output is not None
        local_size = int(math.prod(output.shape[len(batch_shape) :]))
        batch_demand = TaggedDemand(
            batch_shape, output.rows // local_size, output.blocks
        )

    def required(shape: Shape) -> Tagged:
        if batch_demand is None or shape[: len(batch_shape)] != batch_shape:
            return TaggedDemand.full(shape, block_ids)
        local_size = int(math.prod(shape[len(batch_shape) :]))
        rows = np.repeat(batch_demand.rows * local_size, local_size) + np.tile(
            np.arange(local_size, dtype=np.int64), batch_demand.nnz
        )
        return TaggedDemand(shape, rows, np.repeat(batch_demand.blocks, local_size))

    try:
        for callback_index, (callback, child_frame) in enumerate(
            zip(spec.callbacks(), frames, strict=True)
        ):
            callback_outputs = tuple(
                required(_shape_of(atom)) for atom in child_frame.plan.jaxpr.outvars
            )
            step = FrameStep(eqn_plan.index, NestedKind.LINEAR_SOLVE, callback_index)
            child = _backprop_tagged_plan(
                child_frame.plan,
                child_frame,
                resolver,
                _seed_child(seed_node, step) or _TaggedSeedNode(),
                output_demands=callback_outputs,
            )
            for binding, demand in zip(
                callback.inputs, child.input_demands, strict=True
            ):
                if binding.runtime:
                    if callback.name == "solve" and demand is not None:
                        rhs = spec.rhs_indices[0]
                        outer[rhs] = merge_tagged(outer[rhs], demand)
                elif demand is not None:
                    index = binding.outer_input_index
                    assert index is not None
                    outer[index] = merge_tagged(outer[index], demand)
            for binding in callback.inputs:
                index = binding.outer_input_index
                if index is not None:
                    outer[index] = merge_tagged(
                        outer[index], required(_shape_of(eqn.invars[index]))
                    )
    finally:
        for child_frame in frames:
            resolver.release(child_frame)
    return tuple(outer), TaggedTraversalSummary(NestedKind.LINEAR_SOLVE)


def _backprop_tagged_plan(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seed_node: _TaggedSeedNode,
    *,
    output_demands: tuple[Tagged, ...] | None = None,
) -> JaxprTaggedTrace:
    if frame.plan is not plan:
        raise ValueError("tagged traversal plan does not match concrete frame")
    jaxpr = plan.jaxpr
    demands: dict[Var, TaggedDemand] = {}
    nested_traces: dict[int, object] = {}
    for var, demand in seed_node.values.items():
        _add_tagged(demands, var, demand)
    if output_demands is not None:
        if len(output_demands) != len(jaxpr.outvars):
            raise ValueError(
                f"received {len(output_demands)} tagged output demands for "
                f"JAXPR with {len(jaxpr.outvars)} outputs"
            )
        for atom, demand in zip(jaxpr.outvars, output_demands, strict=True):
            _add_tagged(demands, atom, demand)

    for eqn_plan in reversed(plan.eqns):
        eqn = eqn_plan.eqn
        outputs = tuple(demands.get(outvar) for outvar in eqn.outvars)
        nested = eqn_plan.nested
        if nested is None:
            if all(demand is None for demand in outputs):
                continue
            inputs = _backprop_plan_ordinary(eqn_plan, frame, resolver, outputs)
        else:
            has_child_seed = any(
                step.eqn_index == eqn_plan.index for step in seed_node.children
            )
            if all(demand is None for demand in outputs) and not has_child_seed:
                continue
            spec = nested.spec
            if isinstance(spec, CallSpec):
                inputs, summary = _backprop_plan_call(
                    eqn_plan, frame, resolver, spec, outputs, seed_node
                )
            elif isinstance(spec, CondSpec):
                inputs, summary = _backprop_plan_cond(
                    eqn_plan, frame, resolver, spec, outputs, seed_node
                )
            elif isinstance(spec, MapSpec):
                inputs, summary = _backprop_plan_map(
                    eqn_plan, frame, resolver, spec, outputs, seed_node
                )
            elif isinstance(spec, ScanSpec):
                inputs, summary = _backprop_plan_scan(
                    eqn_plan, frame, resolver, spec, outputs, seed_node
                )
            elif isinstance(spec, LinearSolveSpec):
                inputs, summary = _backprop_plan_linear_solve(
                    eqn_plan, frame, resolver, spec, outputs, seed_node
                )
            else:
                raise AssertionError(f"unsupported nested spec {spec!r}")
            nested_traces[eqn_plan.index] = summary
        if len(inputs) != len(eqn.invars):
            raise RuntimeError(
                f"{eqn.primitive.name!r} tagged propagation returned the wrong "
                "number of inputs"
            )
        for atom, demand in zip(eqn.invars, inputs, strict=True):
            _add_tagged(demands, atom, demand)
    return JaxprTaggedTrace(
        demands=demands,
        input_demands=tuple(demands.get(var) for var in jaxpr.invars),
        nested=nested_traces,
    )


def backpropagate_plan_tagged_demand(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seeds: Iterable[TaggedDemandSeed],
) -> JaxprTaggedTrace:
    """Propagate tagged demand without constructing a JaxprInstance tree."""
    return _backprop_tagged_plan(plan, frame, resolver, _build_tagged_seed_tree(seeds))


def generate_contribution_blocks(
    contributions: ContributionTrace,
    *,
    blocks_per_root: int,
) -> tuple[ContributionBlock, ...]:
    """Overdecompose contribution roots along their first partition axis."""
    if blocks_per_root is not None and blocks_per_root <= 0:
        raise ValueError("blocks_per_root must be positive")

    blocks: list[ContributionBlock] = []

    for root in contributions.roots:
        shape = root.domain.shape

        if not root.domain.partition_axes:
            demand = TensorDemand.full(shape)
            if demand is not None:
                blocks.append(
                    ContributionBlock(
                        id=len(blocks),
                        root_id=root.id,
                        demand=demand,
                    )
                )
            continue

        axis = root.domain.partition_axes[0]
        extent = shape[axis]
        count = min(blocks_per_root, extent)

        quotient, remainder = divmod(extent, count)
        start = 0

        for index in range(count):
            width = quotient + (index < remainder)
            stop = start + width
            demand = TensorDemand.axis_range(shape, axis, start, stop)
            assert demand is not None

            blocks.append(
                ContributionBlock(
                    id=len(blocks),
                    root_id=root.id,
                    demand=demand,
                )
            )
            start = stop

    return tuple(blocks)


def _canonical_bool_csr(
    value: sps.spmatrix | ArrayLike,
    *,
    shape: tuple[int, int],
) -> sps.csr_matrix:
    result = sps.csr_matrix(value, shape=shape, dtype=bool)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.nnz:
        result.data[:] = True
    return result


@dataclass(frozen=True)
class BlockDofIncidence:
    """Sparse boolean relation from dense block IDs to global DOF IDs."""

    blocks: tuple[ContributionBlock, ...]
    csr: sps.csr_matrix

    def __post_init__(self) -> None:
        expected_ids = np.arange(len(self.blocks), dtype=np.int64)
        actual_ids = np.asarray([block.id for block in self.blocks], dtype=np.int64)
        if not np.array_equal(actual_ids, expected_ids):
            raise ValueError("contribution block IDs must be dense and ordered")

        csr = _canonical_bool_csr(
            self.csr,
            shape=(len(self.blocks), int(self.csr.shape[1])),
        )
        object.__setattr__(self, "csr", csr)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_dofs(self) -> int:
        return int(self.csr.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.csr.nnz)

    @property
    def block_dof_counts(self) -> NDArray[np.int64]:
        """Number of demanded global DOFs for every contribution block."""
        return np.diff(self.csr.indptr).astype(np.int64, copy=True)

    def dofs_for_block(self, block_id: int) -> NDArray[np.int64]:
        if block_id < 0 or block_id >= self.n_blocks:
            raise IndexError(f"block {block_id} is out of range [0, {self.n_blocks})")
        start = self.csr.indptr[block_id]
        stop = self.csr.indptr[block_id + 1]
        return self.csr.indices[start:stop].copy()

    def blocks_for_dof(self, dof_id: int) -> NDArray[np.int64]:
        if dof_id < 0 or dof_id >= self.n_dofs:
            raise IndexError(f"DOF {dof_id} is out of range [0, {self.n_dofs})")
        return self.csr.getcol(dof_id).nonzero()[0].astype(np.int64, copy=False)


def reference_block_dof_incidence(
    resolved: JaxprInstance,
    contributions: ContributionTrace,
    *,
    blocks: tuple[ContributionBlock, ...],
) -> BlockDofIncidence:
    """Compute the Phase-1 reference incidence with one liveness pass per block."""
    if not resolved.plan.jaxpr.invars:
        raise ValueError("functional JAXPR has no DOF input")

    dof_shape = _shape_of(resolved.plan.jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(f"first input must be a flat DOF vector, got {dof_shape}")
    n_dofs = dof_shape[0]

    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []

    for expected_id, block in enumerate(blocks):
        if block.id != expected_id:
            raise ValueError("contribution block IDs must be dense and ordered")

        root = contributions.root(block.root_id)
        if block.demand.shape != root.domain.shape:
            raise ValueError(
                f"block {block.id} demand shape {block.demand.shape} does not "
                f"match root {root.id} shape {root.domain.shape}"
            )

        trace = backpropagate_demand(
            resolved,
            (DemandSeed(value=root.value, demand=block.demand),),
        )
        dof_demand = trace.input_demands[0]
        if dof_demand is None:
            continue

        dofs = np.unique(dof_demand.rows())
        rows.append(np.full(dofs.size, block.id, dtype=np.int64))
        cols.append(dofs)

    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    data = np.ones(row.size, dtype=bool)
    csr = sps.csr_matrix(
        (data, (row, col)),
        shape=(len(blocks), n_dofs),
        dtype=bool,
    )

    return BlockDofIncidence(blocks=blocks, csr=csr)


def tagged_block_dof_incidence(
    resolved: JaxprInstance,
    contributions: ContributionTrace,
    *,
    blocks: tuple[ContributionBlock, ...] | None = None,
    block_size: int = 1,
) -> BlockDofIncidence:
    """Compute block↔DOF incidence in one sparse tagged reverse traversal."""
    if not resolved.plan.jaxpr.invars:
        raise ValueError("functional JAXPR has no DOF input")

    dof_shape = _shape_of(resolved.plan.jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(f"first input must be a flat DOF vector, got {dof_shape}")
    n_dofs = dof_shape[0]

    if blocks is None:
        blocks = generate_contribution_blocks(
            contributions,
            blocks_per_root=10,
        )

    seed_pairs: dict[
        ValueRef,
        tuple[list[NDArray[np.int64]], list[NDArray[np.int64]]],
    ] = {}
    for expected_id, block in enumerate(blocks):
        if block.id != expected_id:
            raise ValueError("contribution block IDs must be dense and ordered")
        root = contributions.root(block.root_id)
        if block.demand.shape != root.domain.shape:
            raise ValueError(
                f"block {block.id} demand shape {block.demand.shape} does not "
                f"match root {root.id} shape {root.domain.shape}"
            )
        rows, labels = seed_pairs.setdefault(root.value, ([], []))
        block_rows = block.demand.rows()
        rows.append(block_rows)
        labels.append(np.full(block_rows.size, block.id, dtype=np.int64))

    seeds = [
        TaggedDemandSeed(
            value=value,
            demand=TaggedDemand(
                _shape_of(value.var),
                np.concatenate(rows),
                np.concatenate(labels),
            ),
        )
        for value, (rows, labels) in seed_pairs.items()
    ]

    trace = backpropagate_tagged_demand(resolved, seeds)
    dof_demand = trace.input_demands[0]
    if dof_demand is None:
        csr = sps.csr_matrix((len(blocks), n_dofs), dtype=bool)
    else:
        if dof_demand.shape != dof_shape:
            raise RuntimeError(
                f"tagged DOF demand has shape {dof_demand.shape}; expected {dof_shape}"
            )
        if np.any(dof_demand.blocks >= len(blocks)):
            raise RuntimeError("tagged propagation produced an unknown block ID")
        csr = sps.csr_matrix(
            (
                np.ones(dof_demand.nnz, dtype=bool),
                (dof_demand.blocks, dof_demand.rows),
            ),
            shape=(len(blocks), n_dofs),
            dtype=bool,
        )

    return BlockDofIncidence(blocks=blocks, csr=csr)


def plan_tagged_block_dof_incidence(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    contributions: ContributionTrace,
    *,
    blocks: tuple[ContributionBlock, ...],
) -> BlockDofIncidence:
    """Compute block↔DOF incidence directly from a static analysis plan."""
    if frame.plan is not plan:
        raise ValueError("incidence plan does not match concrete frame")
    if not plan.jaxpr.invars:
        raise ValueError("functional JAXPR has no DOF input")

    dof_shape = _shape_of(plan.jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(f"first input must be a flat DOF vector, got {dof_shape}")

    n_dofs = dof_shape[0]
    seed_pairs: dict[
        ValueRef,
        tuple[list[NDArray[np.int64]], list[NDArray[np.int64]]],
    ] = {}

    for expected_id, block in enumerate(blocks):
        if block.id != expected_id:
            raise ValueError("contribution block IDs must be dense and ordered")
        root = contributions.root(block.root_id)
        if block.demand.shape != root.domain.shape:
            raise ValueError(
                f"block {block.id} demand shape {block.demand.shape} does not "
                f"match root {root.id} shape {root.domain.shape}"
            )
        rows, labels = seed_pairs.setdefault(root.value, ([], []))
        block_rows = block.demand.rows()
        rows.append(block_rows)
        labels.append(np.full(block_rows.size, block.id, dtype=np.int64))

    seeds = [
        TaggedDemandSeed(
            value=value,
            demand=TaggedDemand(
                _shape_of(value.var),
                np.concatenate(rows),
                np.concatenate(labels),
            ),
        )
        for value, (rows, labels) in seed_pairs.items()
    ]
    trace = backpropagate_plan_tagged_demand(plan, frame, resolver, seeds)
    dof_demand = trace.input_demands[0]
    if dof_demand is None:
        csr = sps.csr_matrix((len(blocks), n_dofs), dtype=bool)
    else:
        if dof_demand.shape != dof_shape:
            raise RuntimeError(
                f"tagged DOF demand has shape {dof_demand.shape}; expected {dof_shape}"
            )
        if np.any(dof_demand.blocks >= len(blocks)):
            raise RuntimeError("tagged propagation produced an unknown block ID")
        csr = sps.csr_matrix(
            (
                np.ones(dof_demand.nnz, dtype=bool),
                (dof_demand.blocks, dof_demand.rows),
            ),
            shape=(len(blocks), n_dofs),
            dtype=bool,
        )
    return BlockDofIncidence(blocks=blocks, csr=csr)
