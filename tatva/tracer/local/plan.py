"""
Construction of rank-local lowering plans.

This phase combines:

    materialized global program
    + backward structured demand
    + finalized TensorLayouts
    + localized routes

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
from typing import cast

from jax.extend.core import JaxprEqn, Literal, Var

from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallContext,
    CallInvocation,
    CallSpec,
    CondContext,
    CondInvocation,
    CondSpec,
    IndexedChild,
    MapContext,
    MapSpec,
    NestedSpec,
    RepeatedInvocation,
    ScanContext,
    ScanSpec,
    TraversalOrder,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.routes import (
    Route,
)
from tatva.tracer.core.semantics import RouteLocalizationContext
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.liveness import JaxprDemandTrace
from tatva.tracer.local.localize import (
    LocalRoute,
)
from tatva.tracer.program.materialize import JaxprInstance, ResolvedEqn


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Localization status of one globally resolved route.

    `local` is None when localization for this route type has not yet
    been implemented.
    """

    global_route: Route
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
    route: RoutePlan | None
    nested: LocalNestedPlan | None

    @property
    def primitive_name(self) -> str:
        return self.eqn.primitive.name


@dataclass(frozen=True)
class LocalJaxprPlan:
    """Local lowering plan for one materialized JAXPR invocation."""

    instance: JaxprInstance
    layouts: Mapping[Var, TensorLayout]
    const_layouts: tuple[TensorLayout | None, ...]
    input_layouts: tuple[TensorLayout | None, ...]
    output_layouts: tuple[TensorLayout | None, ...]
    eqns: tuple[LocalEqnPlan, ...]

    @property
    def global_eqn_count(self) -> int:
        return len(self.instance.eqns)

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


def _build_route_plan(
    resolved: ResolvedEqn,
    *,
    input_layouts: tuple[TensorLayout | None, ...],
    output_layouts: tuple[TensorLayout | None, ...],
) -> RoutePlan | None:
    route = resolved.route
    if route is None:
        return None

    eqn = resolved.plan.eqn
    semantics = SEMANTICS.get_ordinary(eqn.primitive)
    localizer = semantics.localization.localize_route

    local = (
        None
        if localizer is None
        else localizer(
            RouteLocalizationContext(
                eqn=eqn,
                route=route,
                input_layouts=input_layouts,
                output_layouts=output_layouts,
            )
        )
    )

    return RoutePlan(
        global_route=route,
        local=local,
    )


def _plan_call(
    instance: CallInvocation[JaxprInstance],
    trace: CallInvocation[JaxprDemandTrace],
    spec: CallSpec,
) -> LocalNestedPlan:
    return LocalNestedPlan(
        spec=spec,
        invocation=CallInvocation(
            instance.eqn_index,
            _build_local_jaxpr_plan(
                instance.body,
                trace.body,
            ),
        ),
    )


def _plan_cond(
    instance: CondInvocation[JaxprInstance],
    trace: CondInvocation[JaxprDemandTrace],
    spec: CondSpec,
) -> LocalNestedPlan:
    return LocalNestedPlan(
        spec=spec,
        invocation=CondInvocation(
            instance.eqn_index,
            instance.branch_index,
            _build_local_jaxpr_plan(
                instance.body,
                trace.body,
            ),
        ),
    )


def _plan_map(
    instance: RepeatedInvocation[JaxprInstance],
    trace: RepeatedInvocation[JaxprDemandTrace],
    spec: MapSpec,
) -> LocalNestedPlan:
    traces_by_index = {
        cast(int, child.logical_index): child.payload
        for child in trace.children(TraversalOrder.LOGICAL)
    }

    instances_by_index = {
        cast(int, child.logical_index): child.payload
        for child in instance.children(TraversalOrder.LOGICAL)
    }

    if set(traces_by_index) - set(instances_by_index):
        raise RuntimeError(
            "map demand trace contains iterations not present in materialization"
        )

    # Independent map iterations are stored in logical index order.
    # This also matches TensorLayout axis storage order, since RangeAxis
    # and IndexAxis coordinates are canonical/increasing.
    indices = sorted(traces_by_index)

    iterations = tuple(
        IndexedChild(
            index=index,
            body=_build_local_jaxpr_plan(
                instances_by_index[index],
                traces_by_index[index],
            ),
        )
        for index in indices
    )

    return LocalNestedPlan(
        spec=spec,
        invocation=instance.with_children(iterations),
    )


def _plan_scan(
    instance: RepeatedInvocation[JaxprInstance],
    trace: RepeatedInvocation[JaxprDemandTrace],
    spec: ScanSpec,
) -> LocalNestedPlan:
    traces_by_index = {
        cast(int, child.logical_index): child.payload for child in trace.children()
    }

    iterations: list[IndexedChild[LocalJaxprPlan]] = []

    for materialized in instance.children(TraversalOrder.EXECUTION):
        body_trace = traces_by_index.get(materialized.logical_index)
        if body_trace is None:
            continue

        iterations.append(
            IndexedChild(
                index=cast(int, materialized.logical_index),
                body=_build_local_jaxpr_plan(materialized.payload, body_trace),
            )
        )

    if len(iterations) != len(trace.children()):
        raise RuntimeError(
            "scan demand trace contains an iteration not present in materialization"
        )

    return LocalNestedPlan(
        spec=spec,
        invocation=instance.with_children(tuple(iterations)),
    )


@dataclass(frozen=True)
class _LocalPlanNestedHandler:
    trace: AnyNestedInvocation[JaxprDemandTrace]

    def call(self, context: CallContext[JaxprInstance]) -> LocalNestedPlan:
        return _plan_call(
            context.invocation,
            cast(CallInvocation[JaxprDemandTrace], self.trace),
            context.spec,
        )

    def map(self, context: MapContext[JaxprInstance]) -> LocalNestedPlan:
        return _plan_map(
            context.invocation,
            cast(RepeatedInvocation[JaxprDemandTrace], self.trace),
            context.spec,
        )

    def scan(self, context: ScanContext[JaxprInstance]) -> LocalNestedPlan:
        return _plan_scan(
            context.invocation,
            cast(RepeatedInvocation[JaxprDemandTrace], self.trace),
            context.spec,
        )

    def cond(self, context: CondContext[JaxprInstance]) -> LocalNestedPlan:
        return _plan_cond(
            context.invocation,
            cast(CondInvocation[JaxprDemandTrace], self.trace),
            context.spec,
        )


def _build_nested_plan(
    resolved: ResolvedEqn,
    trace: JaxprDemandTrace,
) -> LocalNestedPlan | None:
    nested_instance = resolved.nested
    if nested_instance is None:
        return None

    eqn_index = resolved.plan.index
    nested_trace = trace.nested.get(eqn_index)

    if nested_trace is None:
        raise RuntimeError(
            f"live nested equation {eqn_index} "
            f"({resolved.plan.eqn.primitive.name}) "
            "has no nested demand trace"
        )

    if nested_instance.kind is not nested_trace.kind:
        raise TypeError("nested instance/demand trace mismatch")

    analysis_plan = resolved.plan.nested
    if analysis_plan is None:
        raise TypeError("nested instance has no analysis plan")

    return dispatch_nested(
        analysis_plan.spec,
        nested_instance,
        _LocalPlanNestedHandler(nested_trace),
    )


def _build_local_jaxpr_plan(
    instance: JaxprInstance,
    trace: JaxprDemandTrace,
) -> LocalJaxprPlan:
    jaxpr = instance.plan.jaxpr
    layouts = _finalize_layouts(trace)
    const_layouts = tuple(_atom_layout(var, layouts) for var in jaxpr.constvars)
    input_layouts = tuple(_atom_layout(var, layouts) for var in jaxpr.invars)
    output_layouts = tuple(_atom_layout(atom, layouts) for atom in jaxpr.outvars)
    local_eqns: list[LocalEqnPlan] = []

    for resolved in instance.eqns:
        eqn = resolved.plan.eqn
        eqn_index = resolved.plan.index
        input_layout = tuple(_atom_layout(atom, layouts) for atom in eqn.invars)
        output_layout = tuple(_atom_layout(atom, layouts) for atom in eqn.outvars)
        has_live_output = any(layout is not None for layout in output_layout)
        has_live_nested = eqn_index in trace.nested

        # Dead-code elimination.
        if not has_live_output and not has_live_nested:
            if getattr(eqn, "effects", ()):
                raise NotImplementedError(
                    "effectful dead equations are not yet "
                    f"supported by local planning: "
                    f"{eqn.primitive.name}"
                )

            continue

        nested_plan = (
            _build_nested_plan(resolved, trace) if resolved.nested is not None else None
        )

        route_plan = _build_route_plan(
            resolved,
            input_layouts=input_layout,
            output_layouts=output_layout,
        )

        local_eqns.append(
            LocalEqnPlan(
                index=eqn_index,
                eqn=eqn,
                input_layouts=input_layout,
                output_layouts=output_layout,
                route=route_plan,
                nested=nested_plan,
            )
        )

    result = LocalJaxprPlan(
        instance=instance,
        layouts=layouts,
        const_layouts=const_layouts,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        eqns=tuple(local_eqns),
    )

    _validate_local_jaxpr_plan(result)

    return result


def _validate_local_jaxpr_plan(
    plan: LocalJaxprPlan,
) -> None:
    """Every demanded runtime input of a surviving equation must be produced by
    another surviving equation or be a JAXPR input/const.

    Inputs with layout=None are deliberately ignored because specialized
    lowering may compile them away, e.g. gather connectivity indices.
    """
    jaxpr = plan.instance.plan.jaxpr
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


def build_local_plan(
    root: JaxprInstance,
    demand: JaxprDemandTrace,
) -> LocalJaxprPlan:
    return _build_local_jaxpr_plan(root, demand)


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
            f"({type(eqn.route.global_route).__name__ if eqn.route else 'no route'})"
        )

    return "\n".join(lines)
