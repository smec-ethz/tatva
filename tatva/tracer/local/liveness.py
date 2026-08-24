"""
Recursive structured backward demand propagation.

The normal rank-local pass starts from owned contribution tensors and walks
backwards through the structural JAXPR plan using ephemeral concrete frames.
The legacy materialized-tree entry point remains for reference consumers.

Ordinary primitives delegate to their registered DemandRule. Nested calls,
maps, and scans recurse explicitly because each demanded invocation may have
different concrete-dependent structural routes.

Demands always remain structured TensorDemand objects between equations.
Exact flattened scalar rows are used only transiently inside individual
primitive demand rules.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from jax.extend.core import Literal, Var
from numpy.typing import NDArray

from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallInvocation,
    CallSpec,
    CondInvocation,
    CondSpec,
    CustomJvpInvocation,
    CustomJvpSpec,
    FrameStep,
    IndexedChild,
    LinearSolveInvocation,
    LinearSolveSpec,
    MapSpec,
    NestedKind,
    RepeatedInvocation,
    ScanSpec,
    dispatch_nested_spec,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteRequest
from tatva.tracer.core.semantics import (
    DemandContext,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    Demand,
    TensorDemand,
    _FullAxis,
    axis_indices,
    lift_leading_axis_demand,
    merge_demands,
    take_leading_axis_demand,
)
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver
from tatva.tracer.program.contributions import ValueRef


def _expand_batch_demand(
    batch_demand: TensorDemand,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> Demand:
    """Keep selected leading batches and require complete solve-local axes."""
    batch_rank = len(batch_shape)
    if shape[:batch_rank] != batch_shape:
        return None
    return TensorDemand.from_axes(
        shape,
        batch_demand.axes + tuple(_FullAxis() for _ in shape[batch_rank:]),
    )


@dataclass(frozen=True, slots=True)
class DemandSeed:
    value: ValueRef
    demand: TensorDemand


type NestedDemandTrace = AnyNestedInvocation[JaxprDemandTrace]


@dataclass(frozen=True)
class JaxprDemandTrace:
    demands: dict[Var, TensorDemand]
    eqn_input_demands: dict[int, tuple[Demand, ...]]
    input_demands: tuple[Demand, ...]
    nested: dict[int, NestedDemandTrace]


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


# -----------------------------------------------------------------------------
# Plan-driven rank-local traversal
# -----------------------------------------------------------------------------


def _backprop_plan_ordinary(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    output_demands: tuple[Demand, ...],
) -> tuple[Demand, ...]:
    eqn = eqn_plan.eqn
    semantics = SEMANTICS.get_ordinary(eqn.primitive)
    routing = semantics.routing

    if routing is None:
        result = semantics.demand(
            DemandContext(eqn=eqn, output_demands=output_demands, route=None)
        )

    else:
        demanded_rows = [
            demand.rows() for demand in output_demands if demand is not None
        ]
        request = (
            None
            if not demanded_rows
            else RouteRequest(np.unique(np.concatenate(demanded_rows)))
        )
        route = resolver.routed(frame, eqn_plan, request)
        result = semantics.demand(
            DemandContext(eqn=eqn, output_demands=output_demands, route=route)
        )

    if len(result) != len(eqn.invars):
        raise RuntimeError(
            f"demand rule for {eqn.primitive.name!r} returned {len(result)} "
            f"inputs; expected {len(eqn.invars)}"
        )
    return result


@dataclass(frozen=True)
class _DemandPlanNestedHandler:
    eqn_plan: EqnPlan
    frame: ConcreteFrame
    resolver: ConcreteResolver
    outputs: tuple[Demand, ...]
    seed_node: _SeedNode

    def _empty_outer(self) -> list[Demand]:
        return [None] * len(self.eqn_plan.eqn.invars)

    def call(self, spec: CallSpec) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        outer = self._empty_outer()

        child_frame = self.resolver.call_frame(self.frame, self.eqn_plan)
        step = child_frame.path[-1]
        try:
            child = _backprop_plan_jaxpr(
                child_frame.plan,
                child_frame,
                self.resolver,
                self.seed_node.children.get(step, _SeedNode()),
                output_demands=self.outputs,
            )
        finally:
            self.resolver.release(child_frame)

        indices = spec.resolved_input_indices(len(outer))
        for index, demand in zip(indices, child.input_demands, strict=True):
            outer[index] = merge_demands(outer[index], demand)

        return tuple(outer), CallInvocation(self.eqn_plan.index, child)

    def custom_jvp(
        self, spec: CustomJvpSpec
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        outer = self._empty_outer()
        primal_frame, jvp_frame = self.resolver.custom_jvp_frames(
            self.frame, self.eqn_plan
        )
        primal_step, jvp_step = primal_frame.path[-1], jvp_frame.path[-1]
        jvp_outputs: list[Demand] = list(self.outputs)
        for is_zero, demand in zip(spec.output_zeros, self.outputs, strict=True):
            if not is_zero:
                jvp_outputs.append(demand)
        try:
            primal = _backprop_plan_jaxpr(
                primal_frame.plan,
                primal_frame,
                self.resolver,
                self.seed_node.children.get(primal_step, _SeedNode()),
                output_demands=self.outputs,
            )
            jvp = _backprop_plan_jaxpr(
                jvp_frame.plan,
                jvp_frame,
                self.resolver,
                self.seed_node.children.get(jvp_step, _SeedNode()),
                output_demands=tuple(jvp_outputs),
            )
        finally:
            self.resolver.release(primal_frame)
            self.resolver.release(jvp_frame)

        for index, demand in enumerate(primal.input_demands):
            outer[index] = merge_demands(outer[index], demand)

        for binding, demand in zip(spec.jvp_bindings, jvp.input_demands, strict=True):
            index = binding.outer_input_index
            outer[index] = merge_demands(outer[index], demand)

        return tuple(outer), CustomJvpInvocation(self.eqn_plan.index, primal, jvp)

    def map(self, spec: MapSpec) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        outer = self._empty_outer()
        children: list[IndexedChild[JaxprDemandTrace]] = []

        for index_ in _plan_map_indices(
            self.eqn_plan, spec, self.outputs, self.seed_node
        ):
            index = int(index_)
            child_outputs = tuple(
                take_leading_axis_demand(demand, index) for demand in self.outputs
            )
            child_frame = self.resolver.map_frame(self.frame, self.eqn_plan, index)
            step = child_frame.path[-1]
            try:
                child = _backprop_plan_jaxpr(
                    child_frame.plan,
                    child_frame,
                    self.resolver,
                    self.seed_node.children.get(step, _SeedNode()),
                    output_demands=child_outputs,
                )
            finally:
                self.resolver.release(child_frame)

            children.append(IndexedChild(index, child))

            for input_index, demand in enumerate(child.input_demands):
                if demand is None:
                    continue
                lifted = (
                    demand
                    if input_index < spec.num_consts
                    else lift_leading_axis_demand(
                        demand,
                        outer_shape=_shape_of(self.eqn_plan.eqn.invars[input_index]),
                        index=index,
                    )
                )
                outer[input_index] = merge_demands(outer[input_index], lifted)

        return tuple(outer), RepeatedInvocation.from_spec(
            self.eqn_plan.index, spec, tuple(children)
        )

    def scan(self, spec: ScanSpec) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        eqn_plan = self.eqn_plan
        execution = _plan_scan_prefix(eqn_plan, spec, self.outputs, self.seed_node)
        snapshots = _plan_scan_snapshots(
            eqn_plan, self.frame, self.resolver, spec, execution
        )

        carry = list(self.outputs[: spec.num_carry])
        ys = self.outputs[spec.num_carry :]
        consts: list[Demand] = [None] * spec.num_consts
        xs: list[Demand] = [None] * (
            len(eqn_plan.eqn.invars) - spec.num_consts - spec.num_carry
        )
        children = []

        for index in reversed(execution):
            step = FrameStep(eqn_plan.index, NestedKind.SCAN, index)
            child_seed = self.seed_node.children.get(step)
            child_outputs = tuple(carry) + tuple(
                take_leading_axis_demand(demand, index) for demand in ys
            )
            if all(demand is None for demand in child_outputs) and child_seed is None:
                continue

            child_frame = self.resolver.scan_frame(
                self.frame, eqn_plan, index, snapshots.get(index, {})
            )
            try:
                child = _backprop_plan_jaxpr(
                    child_frame.plan,
                    child_frame,
                    self.resolver,
                    child_seed or _SeedNode(),
                    output_demands=child_outputs,
                )
            finally:
                self.resolver.release(child_frame)

            children.append(IndexedChild(index, child))
            for i in range(spec.num_consts):
                consts[i] = merge_demands(consts[i], child.input_demands[i])

            carry = list(
                child.input_demands[spec.num_consts : spec.num_consts + spec.num_carry]
            )
            start = spec.num_consts + spec.num_carry

            for x_index, demand in enumerate(child.input_demands[start:]):
                if demand is not None:
                    lifted = lift_leading_axis_demand(
                        demand,
                        outer_shape=_shape_of(eqn_plan.eqn.invars[start + x_index]),
                        index=index,
                    )
                    xs[x_index] = merge_demands(xs[x_index], lifted)

        outer = tuple(consts) + tuple(carry) + tuple(xs)

        return outer, RepeatedInvocation.from_spec(
            eqn_plan.index, spec, tuple(children)
        )

    def cond(self, spec: CondSpec) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        outer = self._empty_outer()
        branch_index, child_frame = self.resolver.cond_frame(self.frame, self.eqn_plan)
        step = child_frame.path[-1]
        try:
            child = _backprop_plan_jaxpr(
                child_frame.plan,
                child_frame,
                self.resolver,
                self.seed_node.children.get(step, _SeedNode()),
                output_demands=self.outputs,
            )
        finally:
            self.resolver.release(child_frame)

        for child_index, demand in enumerate(child.input_demands):
            index = spec.outer_input_index(child_index, outer_arity=len(outer))
            outer[index] = merge_demands(outer[index], demand)

        return tuple(outer), CondInvocation(self.eqn_plan.index, branch_index, child)

    def linear_solve(
        self, spec: LinearSolveSpec
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        outer = self._empty_outer()
        eqn_plan = self.eqn_plan

        frames = self.resolver.linear_solve_frames(self.frame, eqn_plan)
        traces: list[JaxprDemandTrace] = []
        output = self.outputs[0] if len(self.outputs) == 1 else None
        batch_shape = (
            output.shape[:-2] if output is not None and len(output.shape) >= 3 else ()
        )

        supported = output is not None and bool(batch_shape)
        if supported:
            for callback, child_frame in zip(spec.callbacks(), frames, strict=True):
                runtime = next(
                    (i for i, binding in enumerate(callback.inputs) if binding.runtime),
                    None,
                )
                supported &= (
                    len(child_frame.plan.jaxpr.outvars) == 1
                    and _shape_of(child_frame.plan.jaxpr.outvars[0])[: len(batch_shape)]
                    == batch_shape
                    and runtime is not None
                    and _shape_of(child_frame.plan.jaxpr.invars[runtime])
                    == output.shape
                )

        batch_demand = (
            TensorDemand.from_axes(batch_shape, output.axes[: len(batch_shape)])
            if supported and output is not None
            else None
        )

        if not supported:
            warnings.warn(
                "custom_linear_solve does not have a supported batched "
                "matrix layout; using conservative full callback demands",
                UserWarning,
                stacklevel=2,
            )

        try:
            for callback_index, (callback, child_frame) in enumerate(
                zip(spec.callbacks(), frames, strict=True)
            ):
                callback_outputs = tuple(
                    TensorDemand.full(_shape_of(atom))
                    if batch_demand is None
                    else _expand_batch_demand(
                        batch_demand,
                        shape=_shape_of(atom),
                        batch_shape=batch_shape,
                    )
                    for atom in child_frame.plan.jaxpr.outvars
                )
                step = FrameStep(
                    eqn_plan.index,
                    NestedKind.LINEAR_SOLVE,
                    callback_index,
                )
                child = _backprop_plan_jaxpr(
                    child_frame.plan,
                    child_frame,
                    self.resolver,
                    self.seed_node.children.get(step, _SeedNode()),
                    output_demands=callback_outputs,
                )
                traces.append(child)

                for binding, demand in zip(
                    callback.inputs, child.input_demands, strict=True
                ):
                    if binding.runtime:
                        if callback.name == "solve" and demand is not None:
                            rhs = spec.rhs_indices[0]
                            outer[rhs] = merge_demands(outer[rhs], demand)
                    elif demand is not None:
                        assert binding.outer_input_index is not None
                        index = binding.outer_input_index
                        outer[index] = merge_demands(outer[index], demand)

                for binding in callback.inputs:
                    if binding.outer_input_index is not None:
                        index = binding.outer_input_index
                        required = (
                            TensorDemand.full(_shape_of(eqn_plan.eqn.invars[index]))
                            if batch_demand is None
                            else _expand_batch_demand(
                                batch_demand,
                                shape=_shape_of(eqn_plan.eqn.invars[index]),
                                batch_shape=batch_shape,
                            )
                        )
                        outer[index] = merge_demands(outer[index], required)

        finally:
            for child_frame in frames:
                self.resolver.release(child_frame)

        return tuple(outer), LinearSolveInvocation(eqn_plan.index, *traces)


def _plan_map_indices(
    eqn_plan: EqnPlan,
    spec: MapSpec,
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> NDArray[np.int64]:
    parts: list[NDArray[np.int64]] = []
    for demand in output_demands:
        if demand is not None:
            parts.append(axis_indices(demand.axis_subset(0), extent=spec.length))
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
        raise IndexError("map demand contains an out-of-range iteration")
    return indices


def _plan_scan_prefix(
    eqn_plan: EqnPlan,
    spec: ScanSpec,
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> tuple[int, ...]:
    execution = spec.execution_indices()
    if not execution:
        return ()
    positions = {logical: position for position, logical in enumerate(execution)}
    active: list[int] = []
    if any(demand is not None for demand in output_demands[: spec.num_carry]):
        active.append(len(execution) - 1)
    for demand in output_demands[spec.num_carry :]:
        if demand is None:
            continue
        for logical in axis_indices(demand.axis_subset(0), extent=spec.length):
            active.append(positions[int(logical)])
    for step in seed_node.children:
        if (
            step.eqn_index == eqn_plan.index
            and step.kind is NestedKind.SCAN
            and step.iteration is not None
        ):
            if step.iteration not in positions:
                raise IndexError("scan demand contains an out-of-range iteration")
            active.append(positions[step.iteration])
    return () if not active else execution[: max(active) + 1]


def _plan_scan_snapshots(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: ScanSpec,
    execution: tuple[int, ...],
) -> dict[int, dict[int, object]]:
    nested = eqn_plan.nested
    assert nested is not None
    required = {
        index - spec.num_consts
        for index in nested.body.concrete_inputs
        if spec.num_consts <= index < spec.num_consts + spec.num_carry
    }
    if not required:
        return {}
    carry = {
        index: resolver.value(frame, eqn_plan.eqn.invars[spec.num_consts + index])
        for index in required
    }
    result: dict[int, dict[int, object]] = {}
    for logical_index in execution:
        result[logical_index] = dict(carry)
        child = resolver.scan_frame(frame, eqn_plan, logical_index, carry)
        try:
            carry = {
                index: resolver.value(child, child.plan.jaxpr.outvars[index])
                for index in required
            }
        finally:
            resolver.release(child)
    return result


def _backprop_plan_jaxpr(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seed_node: _SeedNode,
    *,
    output_demands: tuple[Demand, ...] | None = None,
) -> JaxprDemandTrace:
    if frame.plan is not plan:
        raise ValueError("demand traversal plan does not match concrete frame")

    jaxpr = plan.jaxpr
    demands: dict[Var, TensorDemand] = {}
    eqn_input_demands: dict[int, tuple[Demand, ...]] = {}
    nested_traces: dict[int, NestedDemandTrace] = {}

    for var, demand in seed_node.values.items():
        _add_demand(demands, var, demand)

    if output_demands is not None:
        if len(output_demands) != len(jaxpr.outvars):
            raise ValueError("nested output demand arity mismatch")
        for atom, demand in zip(jaxpr.outvars, output_demands, strict=True):
            _add_demand(demands, atom, demand)

    # reverse topological traversal
    for eqn_plan in reversed(plan.eqns):
        eqn = eqn_plan.eqn
        outputs = tuple(demands.get(outvar) for outvar in eqn.outvars)
        nested = eqn_plan.nested

        # ordinary primitive
        if nested is None:
            if all(demand is None for demand in outputs):
                continue

            inputs = _backprop_plan_ordinary(eqn_plan, frame, resolver, outputs)

        else:
            has_seed = any(
                step.eqn_index == eqn_plan.index for step in seed_node.children
            )
            if all(demand is None for demand in outputs) and not has_seed:
                continue

            inputs, nested_trace = dispatch_nested_spec(
                nested.spec,
                _DemandPlanNestedHandler(
                    eqn_plan=eqn_plan,
                    frame=frame,
                    resolver=resolver,
                    outputs=outputs,
                    seed_node=seed_node,
                ),
            )
            nested_traces[eqn_plan.index] = nested_trace

        eqn_input_demands[eqn_plan.index] = tuple(inputs)
        for atom, demand in zip(eqn.invars, inputs, strict=True):
            _add_demand(demands, atom, demand)

    return JaxprDemandTrace(
        demands=demands,
        eqn_input_demands=eqn_input_demands,
        input_demands=tuple(demands.get(var) for var in jaxpr.invars),
        nested=nested_traces,
    )


def backpropagate_plan_demand(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seeds: Iterable[DemandSeed],
) -> JaxprDemandTrace:
    """Propagate one rank's structured demand without a JaxprInstance tree."""
    return _backprop_plan_jaxpr(plan, frame, resolver, _build_seed_tree(seeds))
