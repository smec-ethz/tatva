"""
Planning-time concrete evaluation and structural route materialization.

This module instantiates a static `JaxprPlan` using planning-time concrete
values. The result is a hierarchical `JaxprInstance` tree containing the
resolved structural information required by later tracing phases.

Materialization performs two closely related tasks:

1. Evaluate only those numerical values that static analysis marked as required
   for planning.
2. Resolve concrete-dependent structural routes, such as gather, scatter,
   select, and dynamic-slice routing.

The global DOF input is deliberately unavailable during this phase. Therefore
any structural route that depends transitively on the current DOF values raises
`DynamicRoutingError` rather than silently becoming value-dependent.

Nested JAXPRs are materialized recursively:

- call/remat bodies are instantiated once;
- maps and carry-free scans create independent body instances for each mapped
  leading-axis position;
- scans with carry are instantiated in execution order so concrete carry values
  can flow between iterations.

Each invocation owns its resolved routes. This is necessary for nested repeated
computations because the same lexical JAXPR equation may have different routes
at different map or scan iterations.

Cheap planner-native numerical evaluators may be registered in
`CONCRETE_EVALS`. Unsupported primitives fall back to JAX primitive execution,
and materialization statistics record native and fallback execution counts.

This module does not propagate derivative sparsity. It produces the resolved
invocation tree consumed by `derivatives.py`.

Key invariants:

- `None` means a value was not materialized because planning does not require it.
- A value explicitly required by analysis must be present; otherwise
  materialization fails.
- Routes use global tensor coordinates.
- Nested invocation identity is preserved explicitly in the instance tree.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Var

from tatva.tracer.analysis import (
    CallPlan,
    EqnPlan,
    JaxprPlan,
    MapPlan,
    ScanPlan,
)
from tatva.tracer.concrete import CONCRETE_EVALS
from tatva.tracer.registry import SEMANTICS
from tatva.tracer.routing import Route

type ConcreteValue = Any
type ConcreteEnv = dict[Var, ConcreteValue]


class DynamicRoutingError(RuntimeError):
    """Planning-time routing depends on a value unavailable without the DOFs."""


@dataclass(frozen=True)
class CallInstance:
    body: JaxprInstance


@dataclass(frozen=True)
class ScanIteration:
    # Logical index in the scanned leading axis.
    index: int

    body: JaxprInstance


@dataclass(frozen=True)
class ScanInstance:
    iterations: tuple[ScanIteration, ...]


@dataclass(frozen=True)
class MapIteration:
    index: int
    body: JaxprInstance


@dataclass(frozen=True)
class MapInstance:
    iterations: tuple[MapIteration, ...]


type NestedInstance = CallInstance | ScanInstance | MapInstance


@dataclass(frozen=True)
class ResolvedEqn:
    plan: EqnPlan
    route: Route | None
    nested: NestedInstance | None


@dataclass(frozen=True)
class JaxprInstance:
    plan: JaxprPlan

    # All concrete values available inside this particular invocation.
    concrete: ConcreteEnv

    # Materialized equations in execution order.
    eqns: tuple[ResolvedEqn, ...]

    # Concrete values of this frame's outputs.
    # None means "not materialized / unavailable", not numerical None.
    output_values: tuple[ConcreteValue | None, ...]


@dataclass(frozen=True, slots=True)
class FrameStep:
    """One step from a parent JaxprInstance into a nested invocation.

    iteration is None for call-like wrappers and is the logical iteration index for
    map/scan bodies.
    """

    eqn_index: int
    kind: FrameKind
    iteration: int | None = None


type FrameKind = typing.Literal["call", "scan", "map"]
type FramePath = tuple[FrameStep, ...]


def _read(
    env: ConcreteEnv,
    atom: Atom,
) -> ConcreteValue | None:
    if isinstance(atom, Literal):
        return atom.val
    if isinstance(atom, Var):
        return env.get(atom)
    raise TypeError(f"unsupported Jaxpr atom {type(atom)!r}")


def _write(
    env: ConcreteEnv,
    var: Atom,
    value: ConcreteValue,
) -> None:
    if isinstance(var, Var):
        env[var] = value


def _required_value(
    env: ConcreteEnv,
    atom: Atom,
    *,
    context: str,
) -> ConcreteValue:
    if isinstance(atom, Literal):
        return atom.val

    if isinstance(atom, Var) and atom in env:
        return env[atom]

    raise DynamicRoutingError(
        f"{context} requires the concrete value of {atom}, "
        "but that value is unavailable during planning"
    )


def _execute_primitive(
    eqn: JaxprEqn,
    env: ConcreteEnv,
) -> tuple[ConcreteValue, ...]:
    inputs = tuple(
        _required_value(
            env, atom, context=f"concrete evaluation of {eqn.primitive.name}"
        )
        for atom in eqn.invars
    )

    # these are numpy fast paths for some primitives, but not all. For the rest we fall
    # back to the primitive's bind method.
    evaluator = CONCRETE_EVALS.get(eqn.primitive)
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

    for outvar, value in zip(eqn.outvars, normalized):
        _write(env, outvar, value)

    return normalized


def _seed_env(
    plan: JaxprPlan,
    *,
    input_values: tuple[ConcreteValue | None, ...],
    const_values: tuple[ConcreteValue, ...],
) -> ConcreteEnv:
    jaxpr = plan.jaxpr

    if len(input_values) != len(jaxpr.invars):
        raise ValueError(
            f"Jaxpr expects {len(jaxpr.invars)} inputs, got {len(input_values)}"
        )

    if len(const_values) != len(jaxpr.constvars):
        raise ValueError(
            f"Jaxpr expects {len(jaxpr.constvars)} constants, got {len(const_values)}"
        )

    env: ConcreteEnv = {}

    for var, value in zip(jaxpr.constvars, const_values):
        env[var] = value

    for index, (var, value) in enumerate(zip(jaxpr.invars, input_values)):
        if value is not None:
            env[var] = value

        elif index in plan.concrete_inputs:
            raise DynamicRoutingError(
                f"Jaxpr input {index} ({var}) is required concretely "
                "for routing but is unavailable"
            )

    return env


def _materialize_ordinary(
    eqn_plan: EqnPlan,
    env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    rule = SEMANTICS.get(eqn.primitive)

    # First ensure all values explicitly needed for routing exist.
    for input_index in rule.concrete_inputs(eqn):
        if input_index < 0 or input_index >= len(eqn.invars):
            raise ValueError(
                f"{eqn.primitive.name}.concrete_inputs returned "
                f"invalid index {input_index}"
            )

        _required_value(
            env,
            eqn.invars[input_index],
            context=f"routing for {eqn.primitive.name}",
        )

    # Route is resolved from the current invocation's concrete env.
    route = rule.route(eqn, env)

    # Execute this equation only if something downstream requires
    # one of its outputs concretely.
    if eqn_plan.concrete_outputs:
        _execute_primitive(eqn, env)

    return ResolvedEqn(
        plan=eqn_plan,
        route=route,
        nested=None,
    )


def _materialize_call(
    eqn_plan: EqnPlan,
    nested_plan: CallPlan,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    child_inputs = tuple(_read(parent_env, atom) for atom in eqn.invars)

    child = _materialize_jaxpr(
        nested_plan.body,
        input_values=child_inputs,
        const_values=nested_plan.consts,
    )

    # If the parent frame requires any wrapper outputs concretely,
    # pull the corresponding child outputs through the call boundary.
    for output_index in eqn_plan.concrete_outputs:
        value = child.output_values[output_index]

        if value is None:
            raise DynamicRoutingError(
                f"{nested_plan.kind} output {output_index} is required "
                "concretely but the nested computation did not "
                "materialize it"
            )

        _write(
            parent_env,
            eqn.outvars[output_index],
            value,
        )

    return ResolvedEqn(
        plan=eqn_plan,
        route=None,
        nested=CallInstance(body=child),
    )


def _leading_axis_value(
    value: ConcreteValue | None,
    index: int,
) -> ConcreteValue | None:
    if value is None:
        return None

    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("mapped input must have a leading iteration axis")

    return array[index]


def _materialize_scan(
    eqn_plan: EqnPlan,
    scan_plan: ScanPlan,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    num_consts = scan_plan.num_consts
    num_carry = scan_plan.num_carry
    length = scan_plan.length

    if num_carry <= 0:
        raise RuntimeError("carry-free scan should have been lowered to MapPlan")

    # Outer scan operand ordering:
    #
    #   consts..., carry..., xs...
    #
    outer_values = tuple(_read(parent_env, atom) for atom in eqn.invars)
    const_values = outer_values[:num_consts]
    carry_values = list(outer_values[num_consts : num_consts + num_carry])
    xs_values = outer_values[num_consts + num_carry :]

    # Ensure outer operands required by static scan analysis are
    # actually available.
    for input_index in scan_plan.concrete_inputs:
        if outer_values[input_index] is None:
            raise DynamicRoutingError(
                f"scan input {input_index} is required concretely "
                "for routing inside the scan body, but is unavailable"
            )

    execution_indices = (
        range(length - 1, -1, -1) if scan_plan.reverse else range(length)
    )

    iterations: list[ScanIteration] = []

    # Store concrete y values by logical scan index.
    # ys_by_output[j][i]
    n_y_outputs = len(eqn.outvars) - num_carry

    ys_by_output: list[list[ConcreteValue | None]] = [
        [None] * length for _ in range(n_y_outputs)
    ]

    for logical_index in execution_indices:
        x_step_values = tuple(
            _leading_axis_value(value, logical_index) for value in xs_values
        )

        body_inputs = tuple(const_values) + tuple(carry_values) + x_step_values

        body = _materialize_jaxpr(
            scan_plan.body,
            input_values=body_inputs,
            const_values=scan_plan.consts,
        )

        iterations.append(ScanIteration(index=logical_index, body=body))

        body_outputs = body.output_values

        # Carry for the next execution step.
        carry_values = list(body_outputs[:num_carry])

        # Per-step scan outputs.
        y_outputs = body_outputs[num_carry:]

        for y_index, value in enumerate(y_outputs):
            if value is not None:
                ys_by_output[y_index][logical_index] = value

    # Expose required outer scan outputs to the parent frame.
    for output_index in eqn_plan.concrete_outputs:
        if output_index < num_carry:
            # Final carry.
            value = carry_values[output_index]

            if value is None:
                raise DynamicRoutingError(
                    f"scan final carry output {output_index} is "
                    "required concretely but was not materialized"
                )

        else:
            # Stacked y output.
            y_index = output_index - num_carry
            values = ys_by_output[y_index]

            if any(value is None for value in values):
                raise DynamicRoutingError(
                    f"scan output {output_index} is required concretely "
                    "but one or more iterations did not materialize it"
                )

            value = np.stack(
                [np.asarray(item) for item in values if item is not None], axis=0
            )

        _write(parent_env, eqn.outvars[output_index], value)

    return ResolvedEqn(
        plan=eqn_plan,
        route=None,
        nested=ScanInstance(iterations=tuple(iterations)),
    )


def _materialize_map(
    eqn_plan: EqnPlan,
    map_plan: MapPlan,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    values = tuple(_read(parent_env, atom) for atom in eqn.invars)
    num_consts = map_plan.num_consts
    const_values = values[:num_consts]
    mapped_values = values[num_consts:]

    for input_index in map_plan.concrete_inputs:
        if values[input_index] is None:
            raise DynamicRoutingError(
                f"map input {input_index} is required concretely "
                "for routing inside the map body, but is unavailable"
            )

    execution_indices = (
        range(map_plan.length - 1, -1, -1)
        if map_plan.reverse
        else range(map_plan.length)
    )
    iterations: list[MapIteration] = []
    outputs_by_index: list[list[ConcreteValue | None]] = [
        [None] * map_plan.length for _ in eqn.outvars
    ]

    for logical_index in execution_indices:
        step_values = tuple(
            _leading_axis_value(value, logical_index) for value in mapped_values
        )
        body_inputs = tuple(const_values) + step_values
        body = _materialize_jaxpr(
            map_plan.body, input_values=body_inputs, const_values=map_plan.consts
        )
        iterations.append(MapIteration(index=logical_index, body=body))

        for output_index, value in enumerate(body.output_values):
            if value is not None:
                outputs_by_index[output_index][logical_index] = value

    # only expose concrete outputs actually requested by parent
    for output_index in eqn_plan.concrete_outputs:
        values_for_output = outputs_by_index[output_index]
        if any(value is None for value in values_for_output):
            raise DynamicRoutingError(
                f"map output {output_index} is required concretely "
                "but one or more iterations did not materialize it"
            )
        value = np.stack(
            [np.asarray(item) for item in values_for_output if item is not None], axis=0
        )
        _write(parent_env, eqn.outvars[output_index], value)

    return ResolvedEqn(
        eqn_plan, route=None, nested=MapInstance(iterations=tuple(iterations))
    )


def _materialize_eqn(
    eqn_plan: EqnPlan,
    env: ConcreteEnv,
) -> ResolvedEqn:
    nested = eqn_plan.nested

    if isinstance(nested, CallPlan):
        return _materialize_call(eqn_plan, nested, env)

    if isinstance(nested, ScanPlan):
        return _materialize_scan(eqn_plan, nested, env)

    if isinstance(nested, MapPlan):
        return _materialize_map(eqn_plan, nested, env)

    return _materialize_ordinary(eqn_plan, env)


def _materialize_jaxpr(
    plan: JaxprPlan,
    *,
    input_values: tuple[ConcreteValue | None, ...],
    const_values: tuple[ConcreteValue, ...],
) -> JaxprInstance:
    env = _seed_env(
        plan,
        input_values=input_values,
        const_values=const_values,
    )

    resolved_eqns: list[ResolvedEqn] = []

    for eqn_plan in plan.eqns:
        resolved = _materialize_eqn(eqn_plan, env)
        resolved_eqns.append(resolved)

    output_values = tuple(
        _read(env, atom) if output_index in plan.concrete_outputs else None
        for output_index, atom in enumerate(plan.jaxpr.outvars)
    )

    return JaxprInstance(
        plan=plan,
        concrete=env,
        eqns=tuple(resolved_eqns),
        output_values=output_values,
    )


def materialize_plan(
    closed_jaxpr: ClosedJaxpr,
    flat_args: list[Any],
    plan: JaxprPlan,
) -> JaxprInstance:
    if closed_jaxpr.jaxpr is not plan.jaxpr:
        raise ValueError("analysis plan does not belong to the supplied ClosedJaxpr")

    if len(flat_args) != len(plan.jaxpr.invars):
        raise ValueError(
            f"Jaxpr expects {len(plan.jaxpr.invars)} inputs, got {len(flat_args)}"
        )

    # The DOF vector is deliberately unavailable during planning.
    #
    # Therefore any concrete routing requirement transitively depending
    # on u will fail in _seed_env().
    input_values: tuple[ConcreteValue | None, ...] = (None, *flat_args[1:])

    return _materialize_jaxpr(
        plan,
        input_values=input_values,
        const_values=tuple(closed_jaxpr.consts),
    )


def resolved_eqn(
    instance: JaxprInstance,
    eqn_index: int,
) -> ResolvedEqn:
    for resolved in instance.eqns:
        if resolved.plan.index == eqn_index:
            return resolved

    raise KeyError(f"equation index {eqn_index} is not present in this JaxprInstance")


def resolve_frame(
    root: JaxprInstance,
    path: FramePath,
) -> JaxprInstance:
    """Resolve an invocation-qualified frame path into its JaxprInstance."""
    current = root

    for step in path:
        resolved = resolved_eqn(current, step.eqn_index)
        nested = resolved.nested

        if step.kind == "call":
            if not isinstance(nested, CallInstance):
                raise ValueError(
                    f"frame step expects call at equation {step.eqn_index}"
                )

            if step.iteration is not None:
                raise ValueError("call frame step must not specify an iteration")

            current = nested.body
            continue

        if step.kind == "map":
            if not isinstance(nested, MapInstance):
                raise ValueError(f"frame step expects map at equation {step.eqn_index}")

            if step.iteration is None:
                raise ValueError("map frame step requires an iteration")

            iteration = next(
                (item for item in nested.iterations if item.index == step.iteration),
                None,
            )
            if iteration is None:
                raise KeyError(f"map has no iteration {step.iteration}")

            current = iteration.body
            continue

        if step.kind == "scan":
            if not isinstance(nested, ScanInstance):
                raise ValueError(
                    f"frame step expects scan at equation {step.eqn_index}"
                )

            if step.iteration is None:
                raise ValueError("scan frame step requires an iteration")

            iteration = next(
                (item for item in nested.iterations if item.index == step.iteration),
                None,
            )
            if iteration is None:
                raise KeyError(f"scan has no iteration {step.iteration}")

            current = iteration.body
            continue

        raise ValueError(f"unknown frame kind {step.kind!r}")

    return current
