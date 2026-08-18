import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from jax import lax
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Primitive, Var

from tatva.tracer.core.nested import (
    CallSpec,
    CondSpec,
    FramePath,
    FrameStep,
    LinearSolveSpec,
    MapSpec,
    NestedKind,
    ScanSpec,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteFragment, RouteRequest
from tatva.tracer.core.routes import Route
from tatva.tracer.core.semantics import no_route_fragment
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan


class DynamicRoutingError(RuntimeError):
    """Planning-time routing depends on a value unavailable without the DOFs."""


type ConcreteEvalRule = Callable[[tuple[Any, ...], dict[str, Any]], tuple[Any, ...]]

type ConcreteValue = Any
type ConcreteEnv = dict[Var, ConcreteValue]


def evaluate_concrete_eqn(
    eqn: JaxprEqn,
    inputs: tuple[ConcreteValue, ...],
) -> tuple[ConcreteValue, ...]:
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


@dataclass
class ConcreteResolverStats:
    value_requests: int = 0
    cache_hits: int = 0
    evaluated_eqns: int = 0
    frames_created: int = 0
    frames_released: int = 0
    peak_live_frames: int = 0
    map_iterations: int = 0
    scan_iterations: int = 0


class ConcreteResolver:
    def __init__(self):
        self._frames: dict[FramePath, _FrameState] = {}
        self.stats = ConcreteResolverStats()

    @classmethod
    def root(
        cls, closed_jaxpr: ClosedJaxpr, flat_args: tuple[Any, ...], plan: JaxprPlan
    ) -> tuple[Self, ConcreteFrame]:
        if closed_jaxpr.jaxpr is not plan.jaxpr:
            raise ValueError("plan does not match closed_jaxpr")

        if len(flat_args) != len(plan.jaxpr.invars):
            raise ValueError("plan does not match flat_args")
        if not plan.jaxpr.invars:
            raise ValueError("cannot create a concrete resolver without JAXPR inputs")

        resolver = cls()
        frame = ConcreteFrame(plan, ())
        values: dict[Var, ConcreteValue] = {}

        # closed-over consts are always available
        for var, value in zip(plan.jaxpr.constvars, closed_jaxpr.consts, strict=True):
            values[var] = value

        # input zero is unavailable at planning time
        unavailable = {plan.jaxpr.invars[0]: "the DOF input"}

        # all remaining captured args are legal concrete inputs
        for var, value in zip(plan.jaxpr.invars[1:], flat_args[1:], strict=True):
            values[var] = value

        producers = _build_producer_index(plan)
        resolver._frames[frame.path] = _FrameState(
            plan=plan,
            values=values,
            bindings={},
            producers=producers,
            unavailable=unavailable,
            resolving=set(),
        )
        resolver.stats.frames_created = 1
        resolver.stats.peak_live_frames = 1
        return resolver, frame

    def value(
        self, frame: ConcreteFrame, atom: Atom, demand: TensorDemand | None = None
    ) -> ConcreteValue:
        # Phase 4 computes complete concrete values. The argument establishes
        # the API that later phases will use for sliced concrete evaluation.
        del demand
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
            value = self.value(binding.frame, binding.atom)
            if binding.leading_index is not None:
                array = np.asarray(value)
                if array.ndim == 0:
                    raise ValueError("mapped input must have a leading iteration axis")
                value = array[binding.leading_index]
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

    def route(self, frame: ConcreteFrame, eqn_plan: EqnPlan) -> Route | None:
        """Resolve an ordinary equation's complete structural route lazily."""
        semantics, concrete = self._route_inputs(frame, eqn_plan)
        return semantics.route(eqn_plan.eqn, concrete)

    def route_fragment(
        self, frame: ConcreteFrame, eqn_plan: EqnPlan, request: RouteRequest
    ) -> RouteFragment | None:
        semantics, concrete = self._route_inputs(frame, eqn_plan)
        if semantics.route_fragment is no_route_fragment:
            return None
        return semantics.route_fragment(eqn_plan.eqn, concrete, request)

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

    def _route_inputs(self, frame: ConcreteFrame, eqn_plan: EqnPlan):
        self._state(frame)
        if not any(candidate is eqn_plan for candidate in frame.plan.eqns):
            raise ValueError("equation plan does not belong to the concrete frame")
        if eqn_plan.nested is not None:
            raise TypeError("nested primitives do not have ordinary routes")
        eqn = eqn_plan.eqn
        semantics = SEMANTICS.get_ordinary(eqn.primitive)
        concrete: ConcreteEnv = {}
        for input_index in semantics.concrete_inputs(eqn):
            if input_index < 0 or input_index >= len(eqn.invars):
                raise ValueError(
                    f"{eqn.primitive.name}.concrete_inputs returned invalid "
                    f"index {input_index}"
                )
            atom = eqn.invars[input_index]
            if isinstance(atom, Var):
                concrete[atom] = self.value(frame, atom)
            elif not isinstance(atom, Literal):
                raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")
        return semantics, concrete

    def _nested(self, eqn_plan: EqnPlan, spec_type):
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
        consts: tuple[object, ...],
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
    return (x.astype(params["new_dtype"]),)


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
