import warnings
from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Self

import numpy as np
from jax import lax
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Primitive, Var

from tatva.tracer.core.concrete import ConcreteRegion
from tatva.tracer.core.nested import (
    CallSpec,
    CondSpec,
    CustomJvpSpec,
    FramePath,
    FrameStep,
    LinearSolveSpec,
    MapSpec,
    NestedKind,
    ScanSpec,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteFragment, RouteRequest
from tatva.tracer.core.routes import Route, _compute_gather_route_rows
from tatva.tracer.core.semantics import (
    DemandContext,
    FullConcrete,
    PartialRouteContext,
    RegionalConcrete,
    RegionalConcreteContext,
    RouteRequirement,
    RoutingSemantics,
    UnsupportedConcrete,
    no_route_fragment,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand, axis_indices
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan, NestedPlan
from tatva.tracer.program.map_batch import BatchedMapProgram, build_batched_map_program


class DynamicRoutingError(RuntimeError):
    """Planning-time routing depends on a value unavailable without the DOFs."""


class RouteResolutionError(RuntimeError):
    pass


class ConcreteFallback(Enum):
    GLOBAL = auto()
    ERROR = auto()


class ConcreteEvaluationError(RuntimeError):
    """Demand-scoped concrete evaluation cannot be performed safely."""


class UnsupportedConcreteEvaluation(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ConcreteEscalation:
    path: FramePath
    eqn_index: int
    primitive: str
    requested: TensorDemand
    promoted_shape: tuple[int, ...]
    promoted_entries: int
    reason: str
    source: str | None = None


type ConcreteEvalRule = Callable[[tuple[Any, ...], dict[str, Any]], tuple[Any, ...]]
type ConcreteEqnEvalRule = Callable[[JaxprEqn, tuple[Any, ...]], tuple[Any, ...]]

type ConcreteValue = Any
type ConcreteEnv = dict[Var, ConcreteValue]


def evaluate_concrete_eqn(
    eqn: JaxprEqn,
    inputs: tuple[ConcreteValue, ...],
) -> tuple[ConcreteValue, ...]:
    eqn_evaluator = _CONCRETE_EQN_EVALS.get(eqn.primitive)
    if eqn_evaluator is not None:
        outputs = eqn_evaluator(eqn, inputs)
    else:
        # these are numpy fast paths for some primitives, but not all. For the rest we fall
        # back to the primitive's bind method.
        evaluator = _CONCRETE_EVALS.get(eqn.primitive)
        if evaluator is not None:
            outputs = evaluator(inputs, eqn.params)
        else:
            warnings.warn(
                f"Warning: no concrete evaluator for {eqn.primitive.name}, falling back to bind",
                stacklevel=2,
            )
            result = eqn.primitive.bind(*inputs, **eqn.params)
            outputs = tuple(result) if eqn.primitive.multiple_results else (result,)

    if len(outputs) != len(eqn.outvars):
        raise RuntimeError(
            f"{eqn.primitive.name} produced {len(outputs)} concrete outputs "
            f"for {len(eqn.outvars)} Jaxpr outputs"
        )

    # Keep planning data host-side.
    normalized = tuple(
        np.asarray(value) if hasattr(value, "shape") else value for value in outputs
    )

    return normalized


@dataclass(frozen=True, slots=True)
class ConcreteFrame:
    plan: JaxprPlan
    path: FramePath


@dataclass(frozen=True, slots=True)
class _ParentBinding:
    frame: ConcreteFrame
    atom: Atom
    leading_index: int | None = None


@dataclass
class _FrameState:
    plan: JaxprPlan
    values: dict[Var, ConcreteValue]
    bindings: dict[Var, _ParentBinding]
    producers: dict[Var, tuple[EqnPlan, int]]
    unavailable: dict[Var, str]
    resolving: set[Var]
    regions: dict[Var, list[ConcreteRegion]]


@dataclass
class ConcreteResolverStats:
    value_requests: int = 0
    cache_hits: int = 0
    evaluated_eqns: int = 0
    frames_created: int = 0
    frames_released: int = 0
    peak_live_frames: int = 0
    map_iterations: int = 0
    map_template_frames: int = 0
    scan_iterations: int = 0
    regional_value_requests: int = 0
    regional_cache_hits: int = 0
    regional_evaluated_eqns: int = 0
    regional_entries: int = 0
    full_escalations: int = 0


class ConcreteResolver:
    def __init__(self, *, fallback: ConcreteFallback = ConcreteFallback.GLOBAL):
        self._frames: dict[FramePath, _FrameState] = {}
        self.fallback = fallback
        self.escalations: list[ConcreteEscalation] = []
        self.stats = ConcreteResolverStats()
        self._batched_maps: dict[tuple[int, tuple[int, ...]], BatchedMapProgram] = {}

    @classmethod
    def root(
        cls,
        closed_jaxpr: ClosedJaxpr,
        flat_args: tuple[Any, ...],
        plan: JaxprPlan,
        *,
        fallback: ConcreteFallback = ConcreteFallback.GLOBAL,
        unavailable_inputs: Iterable[int] = (0,),
    ) -> tuple[Self, ConcreteFrame]:
        if closed_jaxpr.jaxpr is not plan.jaxpr:
            raise ValueError("plan does not match closed_jaxpr")

        if len(flat_args) != len(plan.jaxpr.invars):
            raise ValueError("plan does not match flat_args")
        if not plan.jaxpr.invars:
            raise ValueError("cannot create a concrete resolver without JAXPR inputs")

        resolver = cls(fallback=fallback)
        frame = ConcreteFrame(plan, ())
        values: dict[Var, ConcreteValue] = {}

        # closed-over consts are always available
        for var, value in zip(plan.jaxpr.constvars, closed_jaxpr.consts, strict=True):
            values[var] = value

        unavailable_indices = tuple(sorted(int(i) for i in unavailable_inputs))
        if any(i < 0 or i >= len(plan.jaxpr.invars) for i in unavailable_indices):
            raise ValueError("unavailable root input index is out of range")

        unavailable = {
            plan.jaxpr.invars[index]: (
                "the DOF input"
                if unavailable_indices == (0,) and index == 0
                else f"root coordinate input {index}"
            )
            for index in unavailable_indices
        }

        # Non-coordinate captured arguments are legal concrete planning inputs.
        unavailable_set = set(unavailable_indices)
        for index, (var, value) in enumerate(
            zip(plan.jaxpr.invars, flat_args, strict=True)
        ):
            if index not in unavailable_set:
                values[var] = value

        producers = _build_producer_index(plan)
        resolver._frames[frame.path] = _FrameState(
            plan=plan,
            values=values,
            bindings={},
            producers=producers,
            unavailable=unavailable,
            resolving=set(),
            regions={},
        )
        resolver.stats.frames_created = 1
        resolver.stats.peak_live_frames = 1
        return resolver, frame

    def value(
        self, frame: ConcreteFrame, atom: Atom, demand: TensorDemand | None = None
    ) -> ConcreteValue | ConcreteRegion:
        if demand is None:
            return self._full_value(frame, atom)
        if demand.shape != _shape_of(atom):
            raise ValueError(
                f"concrete demand shape {demand.shape} does not match "
                f"value shape {_shape_of(atom)}"
            )
        self.stats.value_requests += 1
        self.stats.regional_value_requests += 1
        return self._regional_value(frame, atom, demand)

    def _full_value(self, frame: ConcreteFrame, atom: Atom) -> ConcreteValue:
        self.stats.value_requests += 1

        if isinstance(atom, Literal):
            return atom.val
        if not isinstance(atom, Var):
            raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

        state = self._state(frame)
        if atom in state.values:
            self.stats.cache_hits += 1
            return state.values[atom]

        if atom in state.unavailable:
            label = state.unavailable[atom]
            raise DynamicRoutingError(
                f"concrete evaluation reached {label} ({atom}), which is "
                "unavailable during planning"
            )

        binding = state.bindings.get(atom)
        if binding is not None:
            if binding.leading_index is None:
                value = self.value(binding.frame, binding.atom)
            else:
                child_demand = TensorDemand.full(_shape_of(atom))
                if child_demand is None:
                    # Empty mapped values need no parent data at all.
                    value = np.empty(
                        _shape_of(atom),
                        dtype=getattr(atom.aval, "dtype", None),
                    )
                else:
                    value = self._regional_binding(binding, child_demand).values
            state.values[atom] = value
            return value

        producer = state.producers.get(atom)
        if producer is None:
            raise DynamicRoutingError(
                f"no concrete value or producer is available for {atom} in frame "
                f"{frame.path}"
            )
        eqn_plan, output_index = producer
        if atom in state.resolving:
            raise RuntimeError(f"cycle while resolving concrete value {atom}")

        state.resolving.add(atom)
        try:
            if eqn_plan.nested is None:
                inputs = tuple(
                    self.value(frame, invar) for invar in eqn_plan.eqn.invars
                )
                outputs = evaluate_concrete_eqn(eqn_plan.eqn, inputs)
                self.stats.evaluated_eqns += 1
                for outvar, output in zip(eqn_plan.eqn.outvars, outputs, strict=True):
                    if isinstance(outvar, Var):
                        state.values[outvar] = output
            else:
                self._evaluate_nested_output(frame, eqn_plan, output_index)
        finally:
            state.resolving.remove(atom)

        if output_index >= len(eqn_plan.eqn.outvars):
            raise RuntimeError("concrete producer output index is invalid")
        if atom not in state.values:
            raise RuntimeError(
                f"concrete producer for {atom} did not materialize its requested output"
            )
        return state.values[atom]

    def _regional_value(
        self, frame: ConcreteFrame, atom: Atom, demand: TensorDemand
    ) -> ConcreteRegion:
        if isinstance(atom, Literal):
            return ConcreteRegion.from_full(np.asarray(atom.val), demand)
        if not isinstance(atom, Var):
            raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

        state = self._state(frame)
        for cached in state.regions.get(atom, ()):
            projected = cached.project(demand)
            if projected is not None:
                self.stats.regional_cache_hits += 1
                return projected

        if atom in state.values:
            region = ConcreteRegion.from_full(state.values[atom], demand)
            self._cache_region(state, atom, region)
            return region
        if atom in state.unavailable:
            label = state.unavailable[atom]
            raise DynamicRoutingError(
                f"regional concrete evaluation reached {label} ({atom}), which "
                "is unavailable during planning"
            )

        binding = state.bindings.get(atom)
        if binding is not None:
            region = self._regional_binding(binding, demand)
            self._cache_region(state, atom, region)
            return region

        producer = state.producers.get(atom)
        if producer is None:
            raise DynamicRoutingError(
                f"no concrete value or producer is available for {atom} in frame "
                f"{frame.path}"
            )
        eqn_plan, output_index = producer
        if eqn_plan.nested is None:
            region = self._regional_ordinary(frame, eqn_plan, output_index, demand)
        else:
            region = self._regional_nested(frame, eqn_plan, output_index, demand)
        self._cache_region(state, atom, region)
        return region

    def _regional_binding(
        self, binding: _ParentBinding, demand: TensorDemand
    ) -> ConcreteRegion:
        if binding.leading_index is None:
            parent = self.value(binding.frame, binding.atom, demand)
            assert isinstance(parent, ConcreteRegion)
            return ConcreteRegion(parent.values, demand)

        parent_shape = _shape_of(binding.atom)
        if parent_shape[1:] != demand.shape:
            raise ValueError("mapped concrete binding shape mismatch")
        child_rows = demand.rows()
        child_size = int(np.prod(demand.shape, dtype=np.int64))
        parent_rows = binding.leading_index * child_size + child_rows
        parent_demand = TensorDemand.from_rows_hull(parent_shape, parent_rows)
        if parent_demand is None:
            raise RuntimeError("non-empty child demand produced no parent demand")
        parent = self.value(binding.frame, binding.atom, parent_demand)
        assert isinstance(parent, ConcreteRegion)
        return ConcreteRegion(np.squeeze(parent.values, axis=0), demand)

    def _regional_ordinary(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        output_index: int,
        demand: TensorDemand,
    ) -> ConcreteRegion:
        eqn = eqn_plan.eqn
        semantics = SEMANTICS.get_ordinary(eqn.primitive)
        ctx = RegionalConcreteContext(eqn, output_index, demand)
        decision = semantics.regional_concrete(ctx)
        if isinstance(decision, FullConcrete):
            return self._escalate(
                frame, eqn_plan, output_index, demand, decision.reason
            )
        if isinstance(decision, UnsupportedConcrete):
            raise ConcreteEvaluationError(
                f"{eqn.primitive.name} cannot be evaluated concretely: "
                f"{decision.reason}"
            )
        if not isinstance(decision, RegionalConcrete):
            raise TypeError("regional concrete rule returned an invalid plan")

        output_demands = [None] * len(eqn.outvars)
        output_demands[output_index] = demand
        input_demands = decision.backpropagate(
            DemandContext(eqn, tuple(output_demands), None)
        )
        if len(input_demands) != len(eqn.invars):
            raise ConcreteEvaluationError(
                f"{eqn.primitive.name} regional demand returned the wrong arity"
            )
        inputs: list[Any] = []
        for input_atom, input_demand in zip(eqn.invars, input_demands, strict=True):
            if isinstance(input_atom, Literal):
                inputs.append(input_atom.val)
            elif input_demand is None:
                if not decision.allow_dead_inputs:
                    raise ConcreteEvaluationError(
                        f"{eqn.primitive.name} omitted demand for non-literal input"
                    )
                inputs.append(None)
            else:
                input_region = self.value(frame, input_atom, input_demand)
                assert isinstance(input_region, ConcreteRegion)
                value = input_region.values
                dtype = getattr(input_atom.aval, "dtype", None)
                if dtype is not None:
                    value = value.astype(dtype, copy=False)
                inputs.append(value)
        output = np.asarray(decision.evaluate(ctx, tuple(inputs)))
        region = ConcreteRegion(output, demand)
        self.stats.regional_evaluated_eqns += 1
        return region

    def _regional_nested(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        output_index: int,
        demand: TensorDemand,
    ) -> ConcreteRegion:
        nested = eqn_plan.nested
        assert nested is not None
        spec = nested.spec
        if isinstance(spec, CallSpec):
            child = self.call_frame(frame, eqn_plan)
            try:
                result = self.value(
                    child, child.plan.jaxpr.outvars[output_index], demand
                )
                assert isinstance(result, ConcreteRegion)
                return ConcreteRegion(result.values, demand)
            finally:
                self.release(child)
        if isinstance(spec, CustomJvpSpec):
            primal, jvp = self.custom_jvp_frames(frame, eqn_plan)
            try:
                result = self.value(
                    primal, primal.plan.jaxpr.outvars[output_index], demand
                )
                assert isinstance(result, ConcreteRegion)
                return ConcreteRegion(result.values, demand)
            finally:
                self.release(primal)
                self.release(jvp)
        if isinstance(spec, CondSpec):
            _branch, child = self.cond_frame(frame, eqn_plan)
            try:
                result = self.value(
                    child, child.plan.jaxpr.outvars[output_index], demand
                )
                assert isinstance(result, ConcreteRegion)
                return ConcreteRegion(result.values, demand)
            finally:
                self.release(child)
        if isinstance(spec, MapSpec):
            return self._regional_map(frame, eqn_plan, spec, output_index, demand)
        return self._escalate(
            frame,
            eqn_plan,
            output_index,
            demand,
            f"{type(spec).__name__} requires invocation-wide concrete evaluation",
        )

    def _regional_map(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        spec: MapSpec,
        output_index: int,
        demand: TensorDemand,
    ) -> ConcreteRegion:
        output_shape = _shape_of(eqn_plan.eqn.outvars[output_index])
        logical_indices = axis_indices(demand.axes[0], extent=output_shape[0])
        child_demand = TensorDemand.from_axes(output_shape[1:], demand.axes[1:])
        if child_demand is None:
            raise RuntimeError("mapped output demand produced no child demand")
        values = []
        for logical_index in logical_indices:
            child = self.map_frame(frame, eqn_plan, int(logical_index))
            try:
                region = self.value(
                    child,
                    child.plan.jaxpr.outvars[output_index],
                    child_demand,
                )
                assert isinstance(region, ConcreteRegion)
                values.append(region.values)
            finally:
                self.release(child)
        return ConcreteRegion(np.stack(values, axis=0), demand)

    def _escalate(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        output_index: int,
        demand: TensorDemand,
        reason: str,
    ) -> ConcreteRegion:
        eqn = eqn_plan.eqn
        shape = _shape_of(eqn.outvars[output_index])
        if demand.is_full:
            full = self._full_value(frame, eqn.outvars[output_index])
            return ConcreteRegion.from_full(full, demand)
        escalation = ConcreteEscalation(
            path=frame.path,
            eqn_index=eqn_plan.index,
            primitive=eqn.primitive.name,
            requested=demand,
            promoted_shape=shape,
            promoted_entries=int(np.prod(shape, dtype=np.int64)),
            reason=reason,
            source=str(eqn.source_info) if eqn.source_info is not None else None,
        )
        if self.fallback is ConcreteFallback.ERROR:
            raise ConcreteEvaluationError(
                f"regional concrete evaluation for {eqn.primitive.name} "
                f"requires FULL {shape}: {reason}"
            )
        self.escalations.append(escalation)
        self.stats.full_escalations += 1
        full = self._full_value(frame, eqn.outvars[output_index])
        return ConcreteRegion.from_full(full, demand)

    def _cache_region(
        self, state: _FrameState, atom: Var, region: ConcreteRegion
    ) -> None:
        state.regions.setdefault(atom, []).append(region)
        self.stats.regional_entries += region.values.size

    def _routing_semantics(
        self, frame: ConcreteFrame, eqn_plan: EqnPlan
    ) -> RoutingSemantics | None:
        semantics = SEMANTICS.get_ordinary(eqn_plan.eqn.primitive)
        return semantics.routing

    def routed(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        request: RouteRequest | None = None,
    ) -> Route | RouteFragment | None:
        routing = self._routing_semantics(frame, eqn_plan)
        if routing is None:
            return None

        if request is not None and (
            routing.fragment is not no_route_fragment
            or routing.partial_fragment is not None
        ):
            fragment = self.route_fragment(frame, eqn_plan, request)
            if fragment is not None:
                return fragment

        return self.route(frame, eqn_plan)

    def route(self, frame: ConcreteFrame, eqn_plan: EqnPlan) -> Route | None:
        """Resolve an ordinary equation's complete structural route lazily."""
        routing = self._routing_semantics(frame, eqn_plan)
        if routing is None:
            return None

        concrete = self._route_inputs(frame, eqn_plan, routing)
        route = routing.resolve(eqn_plan.eqn, concrete)

        if route is None and routing.requirement is RouteRequirement.REQUIRED:
            raise RouteResolutionError(
                f"{eqn_plan.eqn.primitive.name} requires a structural route, "
                "but its route resolver could not represent the routing geometry "
                f"at equation {eqn_plan.index}, frame={frame.path}"
            )

        return route

    def route_fragment(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        request: RouteRequest,
    ) -> RouteFragment | None:
        routing = self._routing_semantics(frame, eqn_plan)
        if routing is None:
            return None

        if routing.partial_fragment is not None:
            eqn = eqn_plan.eqn

            def read_input(input_index: int, demand: Demand):
                if input_index < 0 or input_index >= len(eqn.invars):
                    raise ValueError(
                        f"{eqn.primitive.name}: invalid partial-route input "
                        f"index {input_index}"
                    )
                atom = eqn.invars[input_index]
                if isinstance(atom, Literal):
                    return atom.val
                if not isinstance(atom, Var):
                    raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")
                try:
                    if demand is None:
                        return self._full_value(frame, atom)
                    return self.value(frame, atom, demand)
                except DynamicRoutingError:
                    return None

            fragment = routing.partial_fragment(
                PartialRouteContext(
                    eqn=eqn,
                    request=request,
                    read_input=read_input,
                )
            )
            if fragment is not None:
                return fragment

        if routing.fragment is no_route_fragment:
            return None

        concrete = self._route_inputs(
            frame,
            eqn_plan,
            routing,
            request=request,
        )

        return routing.fragment(eqn_plan.eqn, concrete, request)

    def call_frame(self, parent: ConcreteFrame, eqn_plan: EqnPlan) -> ConcreteFrame:
        nested = self._nested(eqn_plan, CallSpec)
        spec = nested.spec
        assert isinstance(spec, CallSpec)
        bindings = tuple(
            _ParentBinding(parent, eqn_plan.eqn.invars[index])
            for index in spec.resolved_input_indices(len(eqn_plan.eqn.invars))
        )
        return self._register_frame(
            nested.body,
            parent.path + (FrameStep(eqn_plan.index, NestedKind.CALL),),
            nested.consts,
            bindings,
        )

    def custom_jvp_frames(
        self, parent: ConcreteFrame, eqn_plan: EqnPlan
    ) -> tuple[ConcreteFrame, ConcreteFrame]:
        nested = self._nested(eqn_plan, CustomJvpSpec)
        spec = nested.spec
        assert isinstance(spec, CustomJvpSpec)
        primal_bindings = tuple(
            _ParentBinding(parent, atom) for atom in eqn_plan.eqn.invars
        )
        primal = self._register_frame(
            nested.branches[0],
            parent.path + (FrameStep(eqn_plan.index, NestedKind.CUSTOM_JVP, 0),),
            nested.branch_consts[0],
            primal_bindings,
        )
        jvp_bindings: list[_ParentBinding | None] = []
        unavailable: dict[int, str] = {}

        for index, binding in enumerate(spec.jvp_bindings):
            if binding.tangent:
                jvp_bindings.append(None)
                unavailable[index] = f"custom_jvp runtime tangent input {index}"
            else:
                jvp_bindings.append(
                    _ParentBinding(
                        parent, eqn_plan.eqn.invars[binding.outer_input_index]
                    )
                )

        jvp = self._register_frame(
            nested.branches[1],
            parent.path + (FrameStep(eqn_plan.index, NestedKind.CUSTOM_JVP, 1),),
            nested.branch_consts[1],
            tuple(jvp_bindings),
            unavailable=unavailable,
        )
        return primal, jvp

    def map_frame(
        self, parent: ConcreteFrame, eqn_plan: EqnPlan, logical_index: int
    ) -> ConcreteFrame:
        nested = self._nested(eqn_plan, MapSpec)
        spec = nested.spec
        assert isinstance(spec, MapSpec)
        if logical_index < 0 or logical_index >= spec.length:
            raise IndexError(f"map index {logical_index} outside [0, {spec.length})")
        bindings = tuple(
            _ParentBinding(
                parent,
                atom,
                None if index < spec.num_consts else logical_index,
            )
            for index, atom in enumerate(eqn_plan.eqn.invars)
        )
        self.stats.map_iterations += 1
        return self._register_frame(
            nested.body,
            parent.path + (FrameStep(eqn_plan.index, NestedKind.MAP, logical_index),),
            nested.consts,
            bindings,
        )

    def batched_map_frame(
        self,
        parent: ConcreteFrame,
        eqn_plan: EqnPlan,
        program: BatchedMapProgram,
        *,
        analysis: bool,
    ) -> ConcreteFrame:
        nested = self._nested(eqn_plan, MapSpec)
        spec = nested.spec
        assert isinstance(spec, MapSpec)

        if analysis:
            plan = program.analysis_plan
            closed = program.analysis_closed_jaxpr
        else:
            plan = program.execution_plan
            closed = program.execution_closed_jaxpr

        if len(plan.jaxpr.invars) != len(eqn_plan.eqn.invars):
            raise RuntimeError("batched map input ABI mismatch")

        bindings = tuple(_ParentBinding(parent, atom) for atom in eqn_plan.eqn.invars)

        return self._register_frame(
            plan,
            parent.path + (FrameStep(eqn_plan.index, NestedKind.MAP, None),),
            closed.consts,
            bindings,
        )

    def batched_map_program(
        self, eqn_plan: EqnPlan, *, expose: tuple[Var, ...] = ()
    ) -> BatchedMapProgram:
        nested = self._nested(eqn_plan, MapSpec)
        key = (id(nested.body.jaxpr), tuple(id(var) for var in expose))
        program = self._batched_maps.get(key)
        spec = nested.spec

        if not isinstance(spec, MapSpec):
            raise TypeError(
                f"batched_map_program only supports MapSpec, got {type(spec).__name__}"
            )

        if program is None:
            program = build_batched_map_program(
                nested.body,
                nested.consts,
                num_consts=spec.num_consts,
                length=spec.length,
                outer_inputs=eqn_plan.eqn.invars,
                expose=expose,
            )
            self._batched_maps[key] = program

        return program

    def cond_frame(
        self, parent: ConcreteFrame, eqn_plan: EqnPlan
    ) -> tuple[int, ConcreteFrame]:
        nested = self._nested(eqn_plan, CondSpec)
        spec = nested.spec
        assert isinstance(spec, CondSpec)
        branch_index = int(np.asarray(self.value(parent, eqn_plan.eqn.invars[0])))
        if branch_index < 0 or branch_index >= spec.num_branches:
            raise DynamicRoutingError(
                f"cond equation {eqn_plan.index} branch index {branch_index} "
                f"out of range [0, {spec.num_branches})"
            )
        bindings = tuple(
            _ParentBinding(parent, atom) for atom in eqn_plan.eqn.invars[1:]
        )
        frame = self._register_frame(
            nested.branches[branch_index],
            parent.path + (FrameStep(eqn_plan.index, NestedKind.COND, branch_index),),
            nested.branch_consts[branch_index],
            bindings,
        )
        return branch_index, frame

    def scan_frame(
        self,
        parent: ConcreteFrame,
        eqn_plan: EqnPlan,
        logical_index: int,
        carry_values: dict[int, ConcreteValue],
    ) -> ConcreteFrame:
        nested = self._nested(eqn_plan, ScanSpec)
        spec = nested.spec
        assert isinstance(spec, ScanSpec)
        if logical_index < 0 or logical_index >= spec.length:
            raise IndexError(f"scan index {logical_index} outside [0, {spec.length})")
        bindings: list[_ParentBinding | None] = []
        unavailable: dict[int, str] = {}
        for input_index, atom in enumerate(eqn_plan.eqn.invars):
            if input_index < spec.num_consts:
                bindings.append(_ParentBinding(parent, atom))
            elif input_index < spec.num_consts + spec.num_carry:
                carry_index = input_index - spec.num_consts
                bindings.append(None)
                if carry_index not in carry_values:
                    unavailable[input_index] = f"scan carry input {carry_index}"
            else:
                bindings.append(_ParentBinding(parent, atom, logical_index))
        values = {
            nested.body.jaxpr.invars[spec.num_consts + carry_index]: value
            for carry_index, value in carry_values.items()
        }
        self.stats.scan_iterations += 1
        return self._register_frame(
            nested.body,
            parent.path + (FrameStep(eqn_plan.index, NestedKind.SCAN, logical_index),),
            nested.consts,
            tuple(bindings),
            values=values,
            unavailable=unavailable,
        )

    def linear_solve_frames(
        self, parent: ConcreteFrame, eqn_plan: EqnPlan
    ) -> tuple[ConcreteFrame, ...]:
        nested = self._nested(eqn_plan, LinearSolveSpec)
        spec = nested.spec
        assert isinstance(spec, LinearSolveSpec)
        result = []
        for callback_index, (callback, body, consts) in enumerate(
            zip(
                spec.callbacks(),
                nested.branches,
                nested.branch_consts,
                strict=True,
            )
        ):
            bindings: list[_ParentBinding | None] = []
            unavailable: dict[int, str] = {}
            for input_index, callback_binding in enumerate(callback.inputs):
                outer_index = callback_binding.outer_input_index
                if outer_index is None:
                    bindings.append(None)
                    unavailable[input_index] = (
                        f"{callback.name} runtime callback input {input_index}"
                    )
                else:
                    bindings.append(
                        _ParentBinding(parent, eqn_plan.eqn.invars[outer_index])
                    )
            result.append(
                self._register_frame(
                    body,
                    parent.path
                    + (
                        FrameStep(
                            eqn_plan.index,
                            NestedKind.LINEAR_SOLVE,
                            callback_index,
                        ),
                    ),
                    consts,
                    tuple(bindings),
                    unavailable=unavailable,
                )
            )
        return tuple(result)

    def release(self, frame: ConcreteFrame) -> None:
        if not frame.path:
            raise ValueError("cannot release the root concrete frame")
        state = self._frames.get(frame.path)
        if state is None or state.plan is not frame.plan:
            raise ValueError(f"unknown concrete frame {frame.path}")
        del self._frames[frame.path]
        self.stats.frames_released += 1

    def _route_inputs(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        routing: RoutingSemantics,
        *,
        request: RouteRequest | None = None,
    ) -> ConcreteEnv:
        eqn = eqn_plan.eqn
        concrete: ConcreteEnv = {}

        indices = routing.inputs(eqn)

        demands = None
        if request is not None and routing.concrete_demands is not None:
            demands = routing.concrete_demands(eqn, request)
            if len(demands) != len(eqn.invars):
                raise ValueError(
                    f"{eqn.primitive.name}.routing.concrete_demands returned the wrong arity"
                )

        for input_index in indices:
            if input_index < 0 or input_index >= len(eqn.invars):
                raise ValueError(
                    f"{eqn.primitive.name}.routing.inputs returned invalid input index {input_index}"
                )

            atom = eqn.invars[input_index]

            try:
                if isinstance(atom, Var):
                    if request is None:
                        concrete[atom] = self._full_value(frame, atom)
                    else:
                        demand = None if demands is None else demands[input_index]

                        if demand is None:
                            demand = TensorDemand.full(_shape_of(atom))

                        concrete[atom] = self.value(frame, atom, demand)

                elif not isinstance(atom, Literal):
                    raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

            except DynamicRoutingError as exc:
                if routing.requirement is RouteRequirement.OPTIONAL:
                    continue

                raise DynamicRoutingError(
                    f"{exc}\n"
                    f"while resolving structural route for "
                    f"{eqn.primitive.name} at equation {eqn_plan.index}, frame={frame.path}"
                ) from exc

        return concrete

    def _record_route_escalation(
        self,
        frame: ConcreteFrame,
        eqn_plan: EqnPlan,
        requested: TensorDemand,
        promoted: TensorDemand,
        input_index: int,
    ) -> None:
        eqn = eqn_plan.eqn
        reason = f"route construction requires complete concrete input {input_index}"
        if self.fallback is ConcreteFallback.ERROR:
            raise ConcreteEvaluationError(
                f"regional route resolution for {eqn.primitive.name} requires "
                f"FULL input {input_index} {_shape_of(eqn.invars[input_index])}: "
                f"{reason}"
            )
        self.escalations.append(
            ConcreteEscalation(
                path=frame.path,
                eqn_index=eqn_plan.index,
                primitive=eqn.primitive.name,
                requested=requested,
                promoted_shape=promoted.shape,
                promoted_entries=promoted.size,
                reason=reason,
                source=str(eqn.source_info) if eqn.source_info is not None else None,
            )
        )
        self.stats.full_escalations += 1

    def _nested(self, eqn_plan: EqnPlan, spec_type) -> NestedPlan:
        nested = eqn_plan.nested
        if nested is None or not isinstance(nested.spec, spec_type):
            raise TypeError(
                f"equation {eqn_plan.index} is not a {spec_type.__name__} plan"
            )
        return nested

    def _register_frame(
        self,
        plan: JaxprPlan,
        path: FramePath,
        consts: Sized[object],
        bindings: tuple[_ParentBinding | None, ...],
        *,
        values: dict[Var, ConcreteValue] | None = None,
        unavailable: dict[int, str] | None = None,
    ) -> ConcreteFrame:
        if path in self._frames:
            raise ValueError(f"concrete frame {path} is already live")
        if len(consts) != len(plan.jaxpr.constvars):
            raise ValueError("nested constants do not match child JAXPR")
        if len(bindings) != len(plan.jaxpr.invars):
            raise ValueError("nested input bindings do not match child JAXPR")

        concrete = dict(values or {})
        concrete.update(zip(plan.jaxpr.constvars, consts, strict=True))
        binding_map: dict[Var, _ParentBinding] = {}
        unavailable_map: dict[Var, str] = {}
        unavailable = unavailable or {}

        for index, (var, binding) in enumerate(
            zip(plan.jaxpr.invars, bindings, strict=True)
        ):
            if var in concrete:
                continue
            if binding is not None:
                binding_map[var] = binding
            else:
                unavailable_map[var] = unavailable.get(index, f"input {index}")

        frame = ConcreteFrame(plan, path)
        self._frames[path] = _FrameState(
            plan=plan,
            values=concrete,
            bindings=binding_map,
            producers=_build_producer_index(plan),
            unavailable=unavailable_map,
            resolving=set(),
            regions={},
        )
        self.stats.frames_created += 1
        self.stats.peak_live_frames = max(
            self.stats.peak_live_frames, len(self._frames)
        )
        return frame

    def _evaluate_nested_output(
        self, parent: ConcreteFrame, eqn_plan: EqnPlan, output_index: int
    ) -> None:
        nested = eqn_plan.nested
        assert nested is not None
        spec = nested.spec
        output_var = eqn_plan.eqn.outvars[output_index]
        parent_state = self._state(parent)

        if isinstance(spec, CallSpec):
            child = self.call_frame(parent, eqn_plan)
            try:
                value = self.value(child, child.plan.jaxpr.outvars[output_index])
            finally:
                self.release(child)
        elif isinstance(spec, CustomJvpSpec):
            primal, jvp = self.custom_jvp_frames(parent, eqn_plan)
            try:
                value = self.value(primal, primal.plan.jaxpr.outvars[output_index])
            finally:
                self.release(primal)
                self.release(jvp)
        elif isinstance(spec, CondSpec):
            _branch, child = self.cond_frame(parent, eqn_plan)
            try:
                value = self.value(child, child.plan.jaxpr.outvars[output_index])
            finally:
                self.release(child)
        elif isinstance(spec, MapSpec):
            values = []
            for logical_index in range(spec.length):
                child = self.map_frame(parent, eqn_plan, logical_index)
                try:
                    values.append(
                        np.asarray(
                            self.value(child, child.plan.jaxpr.outvars[output_index])
                        )
                    )
                finally:
                    self.release(child)
            value = np.stack(values, axis=0)
        elif isinstance(spec, ScanSpec):
            value = self._evaluate_scan_output(parent, eqn_plan, spec, output_index)
        elif isinstance(spec, LinearSolveSpec):
            raise DynamicRoutingError(
                "custom_linear_solve output cannot be required concretely "
                "during planning"
            )
        else:
            raise TypeError(f"unsupported nested spec {spec!r}")

        if isinstance(output_var, Var):
            parent_state.values[output_var] = value

    def _evaluate_scan_output(
        self,
        parent: ConcreteFrame,
        eqn_plan: EqnPlan,
        spec: ScanSpec,
        output_index: int,
    ) -> ConcreteValue:
        nested = eqn_plan.nested
        assert nested is not None
        required_carry = {
            index
            for index in range(spec.num_carry)
            if index in nested.body.concrete_outputs
        }
        carry = {
            index: self.value(parent, eqn_plan.eqn.invars[spec.num_consts + index])
            for index in required_carry
        }
        y_values: dict[int, ConcreteValue] = {}
        for logical_index in spec.execution_indices():
            child = self.scan_frame(parent, eqn_plan, logical_index, carry)
            try:
                next_carry = {
                    index: self.value(child, child.plan.jaxpr.outvars[index])
                    for index in required_carry
                }
                if output_index >= spec.num_carry:
                    y_values[logical_index] = self.value(
                        child, child.plan.jaxpr.outvars[output_index]
                    )
            finally:
                self.release(child)
            carry = next_carry
        if output_index < spec.num_carry:
            if spec.length == 0:
                return self.value(
                    parent,
                    eqn_plan.eqn.invars[spec.num_consts + output_index],
                )
            return carry[output_index]
        return np.stack(
            [np.asarray(y_values[index]) for index in range(spec.length)], axis=0
        )

    def _state(self, frame: ConcreteFrame) -> _FrameState:
        state = self._frames.get(frame.path)
        if state is None:
            raise ValueError(f"unknown concrete frame {frame.path}")
        if state.plan is not frame.plan:
            raise ValueError("concrete frame path is associated with a different plan")
        return state


def _build_producer_index(plan: JaxprPlan) -> dict[Var, tuple[EqnPlan, int]]:
    producers: dict[Var, tuple[EqnPlan, int]] = {}

    for eqn_plan in plan.eqns:
        for output_index, atom in enumerate(eqn_plan.eqn.outvars):
            if not isinstance(atom, Var):
                continue
            if atom in producers:
                raise ValueError(f"duplicate producer for {atom}")
            producers[atom] = (eqn_plan, output_index)

    return producers


# --------------------------------------
# Concrete primitive evaluation rules
# --------------------------------------

_CONCRETE_EVALS: dict[Primitive, ConcreteEvalRule] = {}
_CONCRETE_EQN_EVALS: dict[Primitive, ConcreteEqnEvalRule] = {}


def _simple_np(fn):
    def evaluate(inputs, _params):
        return (fn(*inputs),)

    return evaluate


def register(
    *primitives: Primitive,
) -> Callable[[ConcreteEvalRule], ConcreteEvalRule]:
    def decorator(
        rule: ConcreteEvalRule,
    ) -> ConcreteEvalRule:
        for primitive in primitives:
            _CONCRETE_EVALS[primitive] = rule
        return rule

    return decorator


def register_eqn(
    *primitives: Primitive,
) -> Callable[[ConcreteEqnEvalRule], ConcreteEqnEvalRule]:
    def decorator(rule: ConcreteEqnEvalRule) -> ConcreteEqnEvalRule:
        for primitive in primitives:
            _CONCRETE_EQN_EVALS[primitive] = rule
        return rule

    return decorator


register(lax.neg_p)(_simple_np(np.negative))
register(lax.abs_p)(_simple_np(np.abs))
register(lax.exp_p)(_simple_np(np.exp))
register(lax.add_p)(_simple_np(np.add))
register(lax.sub_p)(_simple_np(np.subtract))
register(lax.mul_p)(_simple_np(np.multiply))
register(lax.div_p)(_simple_np(np.divide))
register(lax.lt_p)(_simple_np(np.less))
register(lax.le_p)(_simple_np(np.less_equal))
register(lax.gt_p)(_simple_np(np.greater))
register(lax.ge_p)(_simple_np(np.greater_equal))
register(lax.eq_p)(_simple_np(np.equal))


@register(lax.copy_p, lax.stop_gradient_p)
def eval_identity(inputs, _params):
    return inputs


@register(lax.reshape_p)
def eval_reshape(inputs, params):
    (x,) = inputs
    return (np.reshape(np.asarray(x), params["new_sizes"]),)


@register(lax.transpose_p)
def eval_transpose(inputs, params):
    (x,) = inputs
    return (np.transpose(np.asarray(x), params["permutation"]),)


@register(lax.broadcast_in_dim_p)
def eval_broadcast_in_dim(inputs, params):
    (x,) = inputs

    shape = tuple(params["shape"])
    dims = tuple(params["broadcast_dimensions"])
    x = np.asarray(x)
    expanded = [1] * len(shape)

    for input_axis, output_axis in enumerate(dims):
        expanded[output_axis] = x.shape[input_axis]

    return (np.broadcast_to(x.reshape(expanded), shape),)


@register(lax.convert_element_type_p)
def _eval_convert_dtype(inputs, params):
    (x,) = inputs
    return (np.asarray(x).astype(params["new_dtype"]),)


@register(lax.iota_p)
def _eval_iota(_inputs, params):
    dim = params.get("dimension")
    shp = params.get("shape")
    newshp = [1] * len(shp)
    newshp[dim] = shp[dim]
    return (np.broadcast_to(np.arange(shp[dim]).reshape(newshp), shp),)


try:
    from jax._src.lax.control_flow import platform_index_p

    @register(platform_index_p)
    def _eval_platform_index(_inputs, params):
        platforms = params.get("platforms", ("cpu",))
        import jax

        backend = jax.default_backend()
        idx = platforms.index(backend) if backend in platforms else 0
        return (np.int32(idx),)
except ImportError:
    pass


@register(lax.select_n_p)
def eval_select_n(inputs, _params):
    selector = np.asarray(inputs[0])
    cases = tuple(np.asarray(x) for x in inputs[1:])
    if not cases:
        raise ValueError("select_n requires at least one case")

    result = cases[0]
    for index, case in enumerate(cases[1:], start=1):
        result = np.where(selector == index, case, result)

    return (result,)


@register(lax.clamp_p)
def eval_clamp(inputs, _params):
    min_val, x_val, max_val = inputs
    return (np.clip(np.asarray(x_val), np.asarray(min_val), np.asarray(max_val)),)


@register(lax.stack_p)
def eval_stack(inputs, params):
    axis = params.get("axis", 0)
    return (np.stack(tuple(np.asarray(x) for x in inputs), axis=axis),)


@register(lax.squeeze_p)
def eval_squeeze(inputs, params):
    (x,) = inputs
    return (np.squeeze(np.asarray(x), axis=params.get("dimensions")),)


@register(lax.slice_p)
def eval_slice(inputs, params):
    (x,) = inputs
    start_indices = params.get("start_indices")
    limit_indices = params.get("limit_indices")
    strides = params.get("strides", None)
    if strides is None:
        strides = (1,) * len(start_indices)

    return (
        np.asarray(x)[
            tuple(
                slice(start, limit, stride)
                for start, limit, stride in zip(start_indices, limit_indices, strides)
            )
        ],
    )


@register(lax.dot_general_p)
def eval_dot_general(inputs, params):
    lhs, rhs = (np.asarray(inputs[0]), np.asarray(inputs[1]))
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params["dimension_numbers"]

    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    lhs_labels = list(range(lhs.ndim))
    next_label = lhs.ndim

    rhs_labels: list[int | None] = [None] * rhs.ndim

    # Batch dimensions share labels.
    for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch, strict=True):
        rhs_labels[rhs_axis] = lhs_labels[lhs_axis]

    # Contracting dimensions share labels.
    for lhs_axis, rhs_axis in zip(lhs_contract, rhs_contract, strict=True):
        rhs_labels[rhs_axis] = lhs_labels[lhs_axis]

    # Everything else on RHS gets a fresh label.
    for axis in range(rhs.ndim):
        if rhs_labels[axis] is None:
            rhs_labels[axis] = next_label
            next_label += 1

    rhs_labels_int = [int(x) for x in rhs_labels]

    lhs_free = [
        axis
        for axis in range(lhs.ndim)
        if axis not in lhs_batch and axis not in lhs_contract
    ]
    rhs_free = [
        axis
        for axis in range(rhs.ndim)
        if axis not in rhs_batch and axis not in rhs_contract
    ]
    out_labels = (
        [lhs_labels[a] for a in lhs_batch]
        + [lhs_labels[a] for a in lhs_free]
        + [rhs_labels_int[a] for a in rhs_free]
    )

    preferred = params.get("preferred_element_type")
    kwargs = {}
    if preferred is not None:
        kwargs["dtype"] = np.dtype(preferred)

    result = np.einsum(
        lhs, lhs_labels, rhs, rhs_labels_int, out_labels, optimize=True, **kwargs
    )

    return (result,)


@register_eqn(lax.gather_p)
def eval_gather(
    eqn: JaxprEqn,
    inputs: tuple[Any, ...],
):
    operand = np.asarray(inputs[0])
    indices = np.asarray(inputs[1])

    output_shape = _shape_of(eqn.outvars[0])
    output_size = int(np.prod(output_shape, dtype=np.int64))
    output_rows = np.arange(output_size, dtype=np.int64)
    source_rows, _ = _compute_gather_route_rows(eqn, indices, output_rows)

    valid = source_rows >= 0

    if not np.all(valid):
        fill_value = eqn.params.get("fill_value")

        if fill_value is None:
            # Initially fall back rather than subtly implementing
            # the wrong JAX fill semantics.
            raise UnsupportedConcreteEvaluation(
                "host gather evaluator does not support implicit fill value"
            )

        result = np.empty(
            output_size,
            dtype=operand.dtype,
        )

        result[valid] = operand.ravel()[source_rows[valid]]
        result[~valid] = fill_value

    else:
        result = operand.ravel()[source_rows]

    return (result.reshape(output_shape),)
