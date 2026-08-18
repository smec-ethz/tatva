from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from jax import lax
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Primitive, Var

from tatva.tracer.core.nested import FramePath
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.route_fragments import RouteFragment, RouteRequest
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
        print(
            f"Warning: no concrete evaluator for {eqn.primitive.name}, falling back to bind"
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


@dataclass
class _FrameState:
    plan: JaxprPlan
    values: dict[Var, ConcreteValue]
    producers: dict[Var, tuple[EqnPlan, int]]
    unavailable: frozenset[Var]
    resolving: set[Var]


@dataclass
class ConcreteResolverStats:
    value_requests: int = 0
    cache_hits: int = 0
    evaluated_eqns: int = 0


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
        unavailable = frozenset({plan.jaxpr.invars[0]})

        # all remaining captured args are legal concrete inputs
        for var, value in zip(plan.jaxpr.invars[1:], flat_args[1:], strict=True):
            values[var] = value

        producers = _build_producer_index(plan)
        resolver._frames[frame.path] = _FrameState(
            plan=plan,
            values=values,
            producers=producers,
            unavailable=unavailable,
            resolving=set(),
        )
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
            input_index = next(
                (
                    index
                    for index, invar in enumerate(frame.plan.jaxpr.invars)
                    if invar is atom
                ),
                None,
            )
            label = "the DOF input" if input_index == 0 else f"input {input_index}"
            raise DynamicRoutingError(
                f"concrete evaluation reached {label} ({atom}), which is "
                "unavailable during planning"
            )

        producer = state.producers.get(atom)
        if producer is None:
            raise DynamicRoutingError(
                f"no concrete value or producer is available for {atom} in frame "
                f"{frame.path}"
            )
        eqn_plan, output_index = producer
        if eqn_plan.nested is not None:
            raise NotImplementedError(
                "lazy concrete evaluation through nested primitive "
                f"{eqn_plan.eqn.primitive.name!r} is deferred to Phase 5"
            )
        if atom in state.resolving:
            raise RuntimeError(f"cycle while resolving concrete value {atom}")

        state.resolving.add(atom)
        try:
            inputs = tuple(self.value(frame, invar) for invar in eqn_plan.eqn.invars)
            outputs = evaluate_concrete_eqn(eqn_plan.eqn, inputs)
            self.stats.evaluated_eqns += 1
            for outvar, output in zip(eqn_plan.eqn.outvars, outputs, strict=True):
                if isinstance(outvar, Var):
                    state.values[outvar] = output
        finally:
            state.resolving.remove(atom)

        if output_index >= len(eqn_plan.eqn.outvars):
            raise RuntimeError("concrete producer output index is invalid")
        if atom not in state.values:
            raise RuntimeError(
                f"concrete producer for {atom} did not materialize its requested output"
            )
        return state.values[atom]

    def route_fragment(
        self, frame: ConcreteFrame, eqn_plan: EqnPlan, request: RouteRequest
    ) -> RouteFragment | None:
        self._state(frame)
        if not any(candidate is eqn_plan for candidate in frame.plan.eqns):
            raise ValueError("equation plan does not belong to the concrete frame")
        if eqn_plan.nested is not None:
            raise NotImplementedError(
                "route fragments for nested primitives are deferred to Phase 5"
            )

        eqn = eqn_plan.eqn
        semantics = SEMANTICS.get_ordinary(eqn.primitive)
        if semantics.route_fragment is no_route_fragment:
            return None

        concrete: ConcreteEnv = {}
        for input_index in semantics.concrete_inputs(eqn):
            if input_index < 0 or input_index >= len(eqn.invars):
                raise ValueError(
                    f"{eqn.primitive.name}.concrete_inputs returned invalid "
                    f"index {input_index}"
                )
            atom = eqn.invars[input_index]
            if isinstance(atom, Literal):
                continue
            elif isinstance(atom, Var):
                concrete[atom] = self.value(frame, atom)
            else:
                raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

        return semantics.route_fragment(eqn, concrete, request)

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
