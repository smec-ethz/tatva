"""
Construction of rank-local lowering plans.

This phase combines:

    structural program + rank-scoped concrete resolution
    + backward structured demand
    + finalized TensorLayouts
    + localized route fragments

into a hierarchical plan suitable for JAX lowering.

No numerical computation happens here and no global JAXPR is mutated.

An equation survives if either:

    - at least one of its outputs is runtime-live, or
    - it contains a live nested computation.

Inputs with layout=None are deliberately absent from the local runtime.
This is expected for planning-only inputs such as gather indices whose
resolved route has already been localized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from jax.extend.core import JaxprEqn, Literal, Var

from tatva.tracer.core.concrete import ConcreteRegion
from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallContext,
    CallInvocation,
    CondContext,
    CondInvocation,
    CustomJvpContext,
    CustomJvpInvocation,
    IndexedChild,
    LinearSolveContext,
    LinearSolveInvocation,
    MapContext,
    NestedSpec,
    RepeatedInvocation,
    ScanContext,
    ScanSpec,
    TraversalOrder,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteFragment, RouteRequest
from tatva.tracer.core.routes import Route
from tatva.tracer.core.semantics import (
    RouteLocalizationContext,
    RoutingScope,
    no_route_fragment,
)
from tatva.tracer.local.demand import Demand, TensorDemand
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.liveness import JaxprDemandTrace
from tatva.tracer.local.localize import (
    LocalRoute,
)
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Rank-local structural route decision."""

    source_kind: str
    local: LocalRoute | None

    @property
    def is_localized(self) -> bool:
        return self.local is not None


@dataclass(frozen=True)
class LocalNestedPlan:
    spec: NestedSpec
    invocation: AnyNestedInvocation[LocalJaxprPlan]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(
            child.logical_index
            for child in self.invocation.children(TraversalOrder.LOGICAL)
            if child.logical_index is not None
        )


@dataclass(frozen=True)
class LocalEqnPlan:
    index: int
    eqn: JaxprEqn
    input_layouts: tuple[TensorLayout | None, ...]
    output_layouts: tuple[TensorLayout | None, ...]
    routing_scope: RoutingScope
    route: RoutePlan | None
    nested: LocalNestedPlan | None

    @property
    def primitive_name(self) -> str:
        return self.eqn.primitive.name


@dataclass(frozen=True)
class LocalJaxprPlan:
    """Local lowering plan for one rank-required JAXPR invocation."""

    plan: JaxprPlan
    const_values: tuple[Any | None, ...]
    layouts: Mapping[Var, TensorLayout]
    const_layouts: tuple[TensorLayout | None, ...]
    input_layouts: tuple[TensorLayout | None, ...]
    output_layouts: tuple[TensorLayout | None, ...]
    eqns: tuple[LocalEqnPlan, ...]

    @property
    def global_eqn_count(self) -> int:
        return len(self.plan.eqns)

    @property
    def local_eqn_count(self) -> int:
        return len(self.eqns)


def _finalize_layouts(
    trace: JaxprDemandTrace,
) -> Mapping[Var, TensorLayout]:
    layouts = {
        var: TensorLayout.from_demand(demand) for var, demand in trace.demands.items()
    }
    return MappingProxyType(layouts)


def _atom_layout(
    atom,
    layouts: Mapping[Var, TensorLayout],
) -> TensorLayout | None:
    if isinstance(atom, Literal):
        return None

    if not isinstance(atom, Var):
        raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

    return layouts.get(atom)


def _demand_layout(demand: Demand) -> TensorLayout | None:
    if demand is None:
        return None
    return TensorLayout.from_demand(demand)


def _rank_route_plan(
    eqn_plan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    *,
    input_layouts: tuple[TensorLayout | None, ...],
    output_layouts: tuple[TensorLayout | None, ...],
) -> RoutePlan | None:
    if frame.plan.routing_scope is RoutingScope.INVOCATION_INTERNAL:
        semantics = SEMANTICS.get_ordinary(eqn_plan.eqn.primitive)
        if semantics.routing is not None and semantics.routing.internal is None:
            raise RuntimeError(
                f"{eqn_plan.eqn.primitive.name} has no internal routing semantics"
            )

        return None

    semantics = SEMANTICS.get_ordinary(eqn_plan.eqn.primitive)
    routing = semantics.routing
    if routing is None:
        return None

    live_outputs = [layout for layout in output_layouts if layout is not None]
    if not live_outputs:
        return None

    rows = np.unique(
        np.concatenate(
            [
                layout.local_rows_to_global_rows(
                    np.arange(layout.local_size, dtype=np.int64)
                )
                for layout in live_outputs
            ]
        )
    )
    fragment = (
        None
        if routing.fragment is no_route_fragment
        else resolver.route_fragment(frame, eqn_plan, RouteRequest(rows))
    )
    route: Route | RouteFragment | None = fragment

    if route is None:
        route = resolver.route(frame, eqn_plan)

    if route is None:
        return None

    localizer = semantics.localization.localize_route
    local = (
        None
        if localizer is None
        else localizer(
            RouteLocalizationContext(
                eqn=eqn_plan.eqn,
                route=route,
                input_layouts=input_layouts,
                output_layouts=output_layouts,
            )
        )
    )
    return RoutePlan(type(route).__name__, local)


def _rank_const_values(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    layouts: tuple[TensorLayout | None, ...],
) -> tuple[Any | None, ...]:
    result: list[Any | None] = []

    for var, layout in zip(plan.jaxpr.constvars, layouts, strict=True):
        if layout is None:
            result.append(None)
            continue

        demand = TensorDemand.from_axes(layout.global_shape, layout.subset.axes)
        if demand is None:
            raise RuntimeError("live constant has an empty demand")

        region = resolver.value(frame, var, demand)
        if not isinstance(region, ConcreteRegion):
            raise TypeError("regional constant lookup did not return ConcreteRegion")

        result.append(region.values)

    return tuple(result)


def _rank_scan_snapshots(
    eqn_plan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: ScanSpec,
    indices: tuple[int, ...],
) -> dict[int, dict[int, object]]:
    if not indices:
        return {}
    nested = eqn_plan.nested
    assert nested is not None
    execution = spec.execution_indices()
    positions = {index: position for position, index in enumerate(execution)}
    prefix = execution[: max(positions[index] for index in indices) + 1]
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
    snapshots: dict[int, dict[int, object]] = {}

    for logical_index in prefix:
        snapshots[logical_index] = dict(carry)
        child = resolver.scan_frame(frame, eqn_plan, logical_index, carry)

        try:
            carry = {
                index: resolver.value(child, child.plan.jaxpr.outvars[index])
                for index in required
            }
        finally:
            resolver.release(child)

    return snapshots


@dataclass(frozen=True)
class _LocalRankPlanNestedHandler:
    eqn_plan: EqnPlan
    frame: ConcreteFrame
    resolver: ConcreteResolver

    def call(self, context: CallContext[JaxprDemandTrace]) -> LocalNestedPlan:
        child_frame = self.resolver.call_frame(self.frame, self.eqn_plan)
        try:
            body = _build_rank_local_jaxpr_plan(
                child_frame.plan, child_frame, self.resolver, context.invocation.body
            )
        finally:
            self.resolver.release(child_frame)

        return LocalNestedPlan(
            context.spec,
            CallInvocation(self.eqn_plan.index, body),
        )

    def custom_jvp(
        self, context: CustomJvpContext[JaxprDemandTrace]
    ) -> LocalNestedPlan:
        primal_frame, jvp_frame = self.resolver.custom_jvp_frames(
            self.frame, self.eqn_plan
        )
        try:
            primal = _build_rank_local_jaxpr_plan(
                primal_frame.plan,
                primal_frame,
                self.resolver,
                context.invocation.primal,
            )
            jvp = _build_rank_local_jaxpr_plan(
                jvp_frame.plan,
                jvp_frame,
                self.resolver,
                context.invocation.jvp,
            )
        finally:
            self.resolver.release(primal_frame)
            self.resolver.release(jvp_frame)
        return LocalNestedPlan(
            context.spec,
            CustomJvpInvocation(self.eqn_plan.index, primal, jvp),
        )

    def cond(self, context: CondContext[JaxprDemandTrace]) -> LocalNestedPlan:
        branch_index, child_frame = self.resolver.cond_frame(self.frame, self.eqn_plan)

        try:
            if branch_index != context.invocation.branch_index:
                raise RuntimeError("conditional branch changed between planning passes")

            body = _build_rank_local_jaxpr_plan(
                child_frame.plan, child_frame, self.resolver, context.invocation.body
            )
        finally:
            self.resolver.release(child_frame)

        return LocalNestedPlan(
            context.spec,
            CondInvocation(self.eqn_plan.index, branch_index, body),
        )

    def map(self, context: MapContext[JaxprDemandTrace]) -> LocalNestedPlan:
        children: list[IndexedChild[LocalJaxprPlan]] = []

        for child_trace in context.invocation.children(TraversalOrder.LOGICAL):
            index = cast(int, child_trace.logical_index)
            child_frame = self.resolver.map_frame(self.frame, self.eqn_plan, index)

            try:
                body = _build_rank_local_jaxpr_plan(
                    child_frame.plan, child_frame, self.resolver, child_trace.payload
                )
            finally:
                self.resolver.release(child_frame)

            children.append(IndexedChild(index, body))

        return LocalNestedPlan(
            context.spec,
            RepeatedInvocation.from_spec(
                self.eqn_plan.index, context.spec, tuple(children)
            ),
        )

    def scan(self, context: ScanContext[JaxprDemandTrace]) -> LocalNestedPlan:
        traces = {
            cast(int, child.logical_index): child.payload
            for child in context.invocation.children()
        }
        indices = tuple(
            index for index in context.spec.execution_indices() if index in traces
        )
        snapshots = _rank_scan_snapshots(
            self.eqn_plan, self.frame, self.resolver, context.spec, indices
        )

        children: list[IndexedChild[LocalJaxprPlan]] = []
        for index in indices:
            child_frame = self.resolver.scan_frame(
                self.frame, self.eqn_plan, index, snapshots.get(index, {})
            )

            try:
                body = _build_rank_local_jaxpr_plan(
                    child_frame.plan, child_frame, self.resolver, traces[index]
                )
            finally:
                self.resolver.release(child_frame)

            children.append(IndexedChild(index, body))

        return LocalNestedPlan(
            context.spec,
            RepeatedInvocation.from_spec(
                self.eqn_plan.index, context.spec, tuple(children)
            ),
        )

    def linear_solve(
        self, context: LinearSolveContext[JaxprDemandTrace]
    ) -> LocalNestedPlan:
        frames = self.resolver.linear_solve_frames(self.frame, self.eqn_plan)

        try:
            children = tuple(
                _build_rank_local_jaxpr_plan(
                    child_frame.plan,
                    child_frame,
                    self.resolver,
                    child_trace.payload,
                )
                for child_frame, child_trace in zip(
                    frames, context.invocation.children(), strict=True
                )
            )
        finally:
            for child_frame in frames:
                self.resolver.release(child_frame)

        return LocalNestedPlan(
            context.spec, LinearSolveInvocation(self.eqn_plan.index, *children)
        )


def _build_rank_nested_plan(
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    trace: JaxprDemandTrace,
    outer_input_layouts: tuple[TensorLayout | None, ...],
) -> LocalNestedPlan:
    nested = eqn_plan.nested
    if nested is None:
        raise TypeError("ordinary equation has no nested local plan")

    nested_trace = trace.nested.get(eqn_plan.index)
    if nested_trace is None:
        raise RuntimeError("live nested equation has no demand trace")

    local_nested_plan = dispatch_nested(
        nested.spec,
        nested_trace,
        _LocalRankPlanNestedHandler(
            eqn_plan,
            frame,
            resolver,
        ),
    )

    return local_nested_plan


def _build_rank_local_jaxpr_plan(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    trace: JaxprDemandTrace,
) -> LocalJaxprPlan:
    if frame.plan is not plan:
        raise ValueError("localization plan does not match concrete frame")

    jaxpr = plan.jaxpr
    layouts = _finalize_layouts(trace)
    const_layouts = tuple(_atom_layout(var, layouts) for var in jaxpr.constvars)
    input_layouts = tuple(_atom_layout(var, layouts) for var in jaxpr.invars)
    output_layouts = tuple(_atom_layout(atom, layouts) for atom in jaxpr.outvars)
    local_eqns: list[LocalEqnPlan] = []

    for eqn_plan in plan.eqns:
        eqn = eqn_plan.eqn
        input_demands = trace.eqn_input_demands.get(eqn_plan.index)
        if input_demands is None:
            inputs = tuple(None for _ in eqn.invars)
        else:
            inputs = tuple(_demand_layout(demand) for demand in input_demands)

        outputs = tuple(_atom_layout(atom, layouts) for atom in eqn.outvars)
        has_nested = eqn_plan.index in trace.nested

        if not any(layout is not None for layout in outputs) and not has_nested:
            if getattr(eqn, "effects", ()):
                raise NotImplementedError(
                    f"effectful dead equation {eqn.primitive.name} is unsupported"
                )
            continue

        nested_plan = (
            _build_rank_nested_plan(eqn_plan, frame, resolver, trace, inputs)
            if has_nested
            else None
        )
        route = (
            None
            if eqn_plan.nested is not None
            else _rank_route_plan(
                eqn_plan,
                frame,
                resolver,
                input_layouts=inputs,
                output_layouts=outputs,
            )
        )
        local_eqns.append(
            LocalEqnPlan(
                eqn_plan.index,
                eqn,
                inputs,
                outputs,
                routing_scope=frame.plan.routing_scope,
                route=route,
                nested=nested_plan,
            )
        )

    result = LocalJaxprPlan(
        plan=plan,
        const_values=_rank_const_values(plan, frame, resolver, const_layouts),
        layouts=layouts,
        const_layouts=const_layouts,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        eqns=tuple(local_eqns),
    )
    _validate_local_jaxpr_plan(result)

    return result


def build_rank_local_plan(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    demand: JaxprDemandTrace,
) -> LocalJaxprPlan:
    """Build one rank's plan without constructing a global invocation tree."""
    return _build_rank_local_jaxpr_plan(plan, frame, resolver, demand)


def _validate_local_jaxpr_plan(
    plan: LocalJaxprPlan,
) -> None:
    """Every demanded runtime input of a surviving equation must be produced by
    another surviving equation or be a JAXPR input/const.

    Inputs with layout=None are deliberately ignored because specialized
    lowering may compile them away, e.g. gather connectivity indices.
    """
    jaxpr = plan.plan.jaxpr
    available: set[Var] = set(jaxpr.invars)
    available.update(jaxpr.constvars)

    for local_eqn in plan.eqns:
        eqn = local_eqn.eqn

        for atom, layout in zip(eqn.invars, local_eqn.input_layouts):
            if layout is None:
                continue

            if isinstance(atom, Literal):
                continue

            if atom not in available:
                raise RuntimeError(
                    "local DCE produced an invalid program: "
                    f"{eqn.primitive.name} requires live variable "
                    f"{atom}, but its producer was removed"
                )

        for outvar, layout in zip(eqn.outvars, local_eqn.output_layouts):
            if layout is not None and isinstance(outvar, Var):
                available.add(outvar)

    for atom, layout in zip(jaxpr.outvars, plan.output_layouts):
        if layout is None:
            continue

        if isinstance(atom, Literal):
            continue

        if atom not in available:
            raise RuntimeError(
                "local JAXPR output is live but its producer is unavailable"
            )


def pending_routes(
    plan: LocalJaxprPlan,
) -> tuple[LocalEqnPlan, ...]:
    result: list[LocalEqnPlan] = []

    def visit(
        frame: LocalJaxprPlan,
    ) -> None:
        for eqn in frame.eqns:
            if eqn.route is not None and not eqn.route.is_localized:
                result.append(eqn)
            if eqn.route is not None and not eqn.route.is_localized:
                result.append(eqn)
            nested = eqn.nested

            if nested is not None:
                for child in nested.invocation.children():
                    visit(child.payload)

    visit(plan)

    return tuple(result)


def summarize_local_plan(
    plan: LocalJaxprPlan,
) -> str:
    lines = [
        (f"equations: {plan.global_eqn_count} global -> {plan.local_eqn_count} local")
    ]

    lines.append("inputs:")

    for index, layout in enumerate(plan.input_layouts):
        if layout is None:
            lines.append(f"  {index}: DEAD")
        else:
            lines.append(f"  {index}: {layout.global_shape} -> {layout.local_shape}")

    unresolved = pending_routes(plan)

    lines.append(f"pending routes: {len(unresolved)}")

    for eqn in unresolved:
        lines.append(
            f"  eqn {eqn.index}: "
            f"{eqn.primitive_name} "
            f"({eqn.route.source_kind if eqn.route else 'no route'})"
        )

    return "\n".join(lines)
