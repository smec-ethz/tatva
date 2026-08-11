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

from jax.extend.core import JaxprEqn, Literal, Var

from tatva.tracer.analysis import MapPlan
from tatva.tracer.layout import TensorLayout
from tatva.tracer.liveness import (
    CallDemandTrace,
    DemandTrace,
    JaxprDemandTrace,
    MapDemandTrace,
    ScanDemandTrace,
)
from tatva.tracer.localize import (
    LocalRoute,
    localize_dynamic_slice_route,
    localize_gather_route,
    localize_scatter_route,
    localize_select_n_route,
)
from tatva.tracer.materialize import (
    CallInstance,
    JaxprInstance,
    MapInstance,
    ResolvedEqn,
    ScanInstance,
)
from tatva.tracer.model import DynamicSliceRoute, ScatterRoute, SelectNRoute
from tatva.tracer.routing import (
    GatherRoute,
    Route,
)


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
class LocalCallPlan:
    body: LocalJaxprPlan


@dataclass(frozen=True)
class LocalMapIterationPlan:
    index: int
    body: LocalJaxprPlan


@dataclass(frozen=True)
class LocalMapPlan:
    num_consts: int
    iterations: tuple[LocalMapIterationPlan, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(iteration.index for iteration in self.iterations)


@dataclass(frozen=True)
class LocalScanIterationPlan:
    index: int
    body: LocalJaxprPlan


@dataclass(frozen=True)
class LocalScanPlan:
    iterations: tuple[LocalScanIterationPlan, ...]


type NestedLocalPlan = LocalCallPlan | LocalMapPlan | LocalScanPlan


@dataclass(frozen=True)
class LocalEqnPlan:
    index: int
    eqn: JaxprEqn
    input_layouts: tuple[TensorLayout | None, ...]
    output_layouts: tuple[TensorLayout | None, ...]
    route: RoutePlan | None
    nested: NestedLocalPlan | None

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


@dataclass(frozen=True)
class LocalPlan:
    root: LocalJaxprPlan


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

    # Gather
    if isinstance(route, GatherRoute):
        if not input_layouts:
            raise RuntimeError("gather has no operand")

        operand_layout = input_layouts[0]
        if operand_layout is None:
            raise RuntimeError("live gather output has no live operand layout")

        if len(output_layouts) != 1:
            raise RuntimeError("gather expected one output")

        output_layout = output_layouts[0]
        if output_layout is None:
            raise RuntimeError("attempted to localize dead gather")

        local = localize_gather_route(
            route,
            operand_layout=operand_layout,
            output_layout=output_layout,
        )

        return RoutePlan(
            global_route=route,
            local=local,
        )

    # Scatter
    if isinstance(route, ScatterRoute):
        if len(input_layouts) < 3:
            raise RuntimeError("scatter expected operand, indices and updates")

        operand_layout = input_layouts[0]

        # input 1 is the index tensor. It is deliberately allowed to be
        # dead because its information has already been compiled into
        # ScatterRoute.
        update_layout = input_layouts[2]
        if len(output_layouts) != 1:
            raise RuntimeError("scatter expected one output")

        output_layout = output_layouts[0]
        if output_layout is None:
            raise RuntimeError("attempted to localize dead scatter")

        local = localize_scatter_route(
            route,
            operand_layout=operand_layout,
            update_layout=update_layout,
            output_layout=output_layout,
        )

        return RoutePlan(
            global_route=route,
            local=local,
        )

    # Dynamic slice
    if isinstance(route, DynamicSliceRoute):
        if not input_layouts:
            raise RuntimeError("dynamic_slice has no operand")

        operand_layout = input_layouts[0]
        if operand_layout is None:
            raise RuntimeError("live dynamic_slice output has no live operand")

        if len(output_layouts) != 1:
            raise RuntimeError("dynamic_slice expected one output")

        output_layout = output_layouts[0]

        if output_layout is None:
            raise RuntimeError("attempted to localize dead dynamic_slice")

        local = localize_dynamic_slice_route(
            route,
            operand_layout=operand_layout,
            output_layout=output_layout,
        )

        return RoutePlan(
            global_route=route,
            local=local,
        )

    if isinstance(route, SelectNRoute):
        if len(output_layouts) != 1:
            raise RuntimeError("select_n expected one output")

        output_layout = output_layouts[0]
        if output_layout is None:
            raise RuntimeError("attempted to localize dead select_n")

        # input 0 is the selector. Its concrete values are already
        # compiled into SelectNRoute.case_indices.
        case_layouts = tuple(input_layouts[1:])

        local = localize_select_n_route(
            route,
            case_layouts=case_layouts,
            output_layout=output_layout,
        )

        return RoutePlan(
            global_route=route,
            local=local,
        )

    # Other route types are intentionally still pending.
    return RoutePlan(
        global_route=route,
        local=None,
    )


def _plan_call(
    instance: CallInstance,
    trace: CallDemandTrace,
) -> LocalCallPlan:
    return LocalCallPlan(
        body=_build_local_jaxpr_plan(
            instance.body,
            trace.body,
        )
    )


def _plan_map(
    instance: MapInstance,
    trace: MapDemandTrace,
    analysis_plan: MapPlan,
) -> LocalMapPlan:
    traces_by_index = {
        iteration.index: iteration.body for iteration in trace.iterations
    }

    instances_by_index = {
        iteration.index: iteration.body for iteration in instance.iterations
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
        LocalMapIterationPlan(
            index=index,
            body=_build_local_jaxpr_plan(
                instances_by_index[index],
                traces_by_index[index],
            ),
        )
        for index in indices
    )

    return LocalMapPlan(
        num_consts=analysis_plan.num_consts,
        iterations=iterations,
    )


def _plan_scan(
    instance: ScanInstance,
    trace: ScanDemandTrace,
) -> LocalScanPlan:
    traces_by_index = {
        iteration.index: iteration.body for iteration in trace.iterations
    }

    iterations: list[LocalScanIterationPlan] = []

    for materialized in instance.iterations:
        body_trace = traces_by_index.get(materialized.index)
        if body_trace is None:
            continue

        iterations.append(
            LocalScanIterationPlan(
                index=materialized.index,
                body=_build_local_jaxpr_plan(materialized.body, body_trace),
            )
        )

    if len(iterations) != len(trace.iterations):
        raise RuntimeError(
            "scan demand trace contains an iteration not present in materialization"
        )

    return LocalScanPlan(iterations=tuple(iterations))


def _build_nested_plan(
    resolved: ResolvedEqn,
    trace: JaxprDemandTrace,
) -> NestedLocalPlan | None:
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

    if isinstance(nested_instance, CallInstance):
        if not isinstance(nested_trace, CallDemandTrace):
            raise TypeError("CallInstance/demand trace mismatch")

        return _plan_call(nested_instance, nested_trace)

    if isinstance(nested_instance, MapInstance):
        if not isinstance(nested_trace, MapDemandTrace):
            raise TypeError("MapInstance/demand trace mismatch")

        analysis_nested = resolved.plan.nested
        if not isinstance(analysis_nested, MapPlan):
            raise TypeError("MapInstance/analysis plan mismatch")
        return _plan_map(nested_instance, nested_trace, analysis_nested)

    if isinstance(nested_instance, ScanInstance):
        if not isinstance(nested_trace, ScanDemandTrace):
            raise TypeError("ScanInstance/demand trace mismatch")

        return _plan_scan(nested_instance, nested_trace)

    raise TypeError(f"unsupported nested instance {type(nested_instance)!r}")


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
    demand: DemandTrace,
) -> LocalPlan:
    return LocalPlan(
        root=_build_local_jaxpr_plan(
            root,
            demand.root,
        )
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
            nested = eqn.nested

            if isinstance(nested, LocalCallPlan):
                visit(nested.body)

            elif isinstance(nested, (LocalMapPlan, LocalScanPlan)):
                for iteration in nested.iterations:
                    visit(iteration.body)

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
