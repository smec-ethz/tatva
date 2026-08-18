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

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Var

from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallInvocation,
    CallSpec,
    CondInvocation,
    CondSpec,
    IndexedChild,
    LinearSolveInvocation,
    LinearSolveSpec,
    MapSpec,
    RepeatedInvocation,
    ScanSpec,
    collect_logical_output,
    dispatch_nested_spec,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.routes import Route
from tatva.tracer.program.analysis import (
    EqnPlan,
    JaxprPlan,
    NestedPlan,
)
from tatva.tracer.program.concrete_resolver import (
    ConcreteEnv,
    ConcreteValue,
    DynamicRoutingError,
    evaluate_concrete_eqn,
)


@dataclass(frozen=True)
class ResolvedEqn:
    plan: EqnPlan
    route: Route | None
    nested: AnyNestedInvocation[JaxprInstance] | None


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

    normalized = evaluate_concrete_eqn(eqn, inputs)

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

    rule = SEMANTICS.get_ordinary(eqn.primitive)

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
    nested_plan: NestedPlan,
    spec: CallSpec,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    outer_inputs = tuple(_read(parent_env, atom) for atom in eqn.invars)
    child_inputs = spec.select_inputs(outer_inputs)

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
                f"{spec.call_kind.name.lower()} output {output_index} is required "
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
        nested=CallInvocation(eqn_index=eqn_plan.index, body=child),
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
    scan_plan: NestedPlan,
    spec: ScanSpec,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    num_consts = spec.num_consts
    num_carry = spec.num_carry
    length = spec.length

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

    iterations: list[IndexedChild[JaxprInstance]] = []

    for logical_index in spec.execution_indices():
        x_step_values = tuple(
            _leading_axis_value(value, logical_index) for value in xs_values
        )

        body_inputs = tuple(const_values) + tuple(carry_values) + x_step_values

        body = _materialize_jaxpr(
            scan_plan.body,
            input_values=body_inputs,
            const_values=scan_plan.consts,
        )

        iterations.append(IndexedChild(index=logical_index, body=body))

        body_outputs = body.output_values

        # Carry for the next execution step.
        carry_values = list(body_outputs[:num_carry])

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
            try:
                values = collect_logical_output(
                    ((item.index, item.body.output_values) for item in iterations),
                    output_index=output_index,
                    length=length,
                    label="scan",
                )
            except RuntimeError as exc:
                raise DynamicRoutingError(str(exc)) from exc
            value = np.stack([np.asarray(item) for item in values], axis=0)

        _write(parent_env, eqn.outvars[output_index], value)

    return ResolvedEqn(
        plan=eqn_plan,
        route=None,
        nested=RepeatedInvocation.from_spec(eqn_plan.index, spec, tuple(iterations)),
    )


def _materialize_map(
    eqn_plan: EqnPlan,
    map_plan: NestedPlan,
    spec: MapSpec,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn

    values = tuple(_read(parent_env, atom) for atom in eqn.invars)
    num_consts = spec.num_consts
    const_values = values[:num_consts]
    mapped_values = values[num_consts:]

    for input_index in map_plan.concrete_inputs:
        if values[input_index] is None:
            raise DynamicRoutingError(
                f"map input {input_index} is required concretely "
                "for routing inside the map body, but is unavailable"
            )

    iterations: list[IndexedChild[JaxprInstance]] = []

    for logical_index in spec.execution_indices():
        step_values = tuple(
            _leading_axis_value(value, logical_index) for value in mapped_values
        )
        body_inputs = tuple(const_values) + step_values
        body = _materialize_jaxpr(
            map_plan.body, input_values=body_inputs, const_values=map_plan.consts
        )
        iterations.append(IndexedChild(index=logical_index, body=body))

    # only expose concrete outputs actually requested by parent
    for output_index in eqn_plan.concrete_outputs:
        try:
            values_for_output = collect_logical_output(
                ((item.index, item.body.output_values) for item in iterations),
                output_index=output_index,
                length=spec.length,
                label="map",
            )
        except RuntimeError as exc:
            raise DynamicRoutingError(str(exc)) from exc
        value = np.stack([np.asarray(item) for item in values_for_output], axis=0)
        _write(parent_env, eqn.outvars[output_index], value)

    return ResolvedEqn(
        eqn_plan,
        route=None,
        nested=RepeatedInvocation.from_spec(eqn_plan.index, spec, tuple(iterations)),
    )


def _materialize_cond(
    eqn_plan: EqnPlan,
    nested_plan: NestedPlan,
    spec: CondSpec,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    eqn = eqn_plan.eqn
    pred_val = _read(parent_env, eqn.invars[0])
    if pred_val is None:
        raise DynamicRoutingError(
            f"cond equation {eqn_plan.index} predicate depends on dynamic "
            "DOF values that are unavailable during planning"
        )
    branch_index = int(np.asarray(pred_val))
    if branch_index < 0 or branch_index >= spec.num_branches:
        raise DynamicRoutingError(
            f"cond equation {eqn_plan.index} branch index {branch_index} "
            f"out of range [0, {spec.num_branches})"
        )

    operand_inputs = tuple(
        _read(parent_env, atom) for atom in spec.select_inputs(eqn.invars)
    )
    for child_index in nested_plan.branches[branch_index].concrete_inputs:
        if operand_inputs[child_index] is None:
            outer_index = spec.outer_input_index(
                child_index, outer_arity=len(eqn.invars)
            )
            raise DynamicRoutingError(
                f"cond branch {branch_index} input {child_index} (outer input {outer_index}) "
                "is required concretely for routing inside the branch, but is unavailable"
            )

    child = _materialize_jaxpr(
        nested_plan.branches[branch_index],
        input_values=operand_inputs,
        const_values=nested_plan.branch_consts[branch_index],
    )

    for output_index in eqn_plan.concrete_outputs:
        val = child.output_values[output_index]
        if val is None:
            raise DynamicRoutingError(
                f"cond output {output_index} is required concretely but "
                "the active branch computation did not materialize it"
            )
        _write(parent_env, eqn.outvars[output_index], val)

    return ResolvedEqn(
        plan=eqn_plan,
        route=None,
        nested=CondInvocation(
            eqn_index=eqn_plan.index,
            branch_index=branch_index,
            body=child,
        ),
    )


def _materialize_linear_solve(
    eqn_plan: EqnPlan,
    nested_plan: NestedPlan,
    spec: LinearSolveSpec,
    parent_env: ConcreteEnv,
) -> ResolvedEqn:
    """Materialize callback captures; callback runtime vectors are intentionally absent."""
    eqn = eqn_plan.eqn
    outer = tuple(_read(parent_env, atom) for atom in eqn.invars)
    children: list[JaxprInstance] = []
    for callback, body, consts in zip(
        spec.callbacks(), nested_plan.branches, nested_plan.branch_consts, strict=True
    ):
        inputs: list[ConcreteValue | None] = []
        for binding in callback.inputs:
            outer_index = binding.outer_input_index
            inputs.append(None if outer_index is None else outer[outer_index])
        children.append(
            _materialize_jaxpr(
                body, input_values=tuple(inputs), const_values=consts
            )
        )
    if eqn_plan.concrete_outputs:
        raise DynamicRoutingError(
            "custom_linear_solve output cannot be required concretely during planning"
        )
    return ResolvedEqn(
        eqn_plan, route=None, nested=LinearSolveInvocation(eqn_plan.index, *children)
    )


def _materialize_eqn(
    eqn_plan: EqnPlan,
    env: ConcreteEnv,
) -> ResolvedEqn:
    nested = eqn_plan.nested

    if nested is None:
        return _materialize_ordinary(eqn_plan, env)

    return dispatch_nested_spec(
        nested.spec, _MaterializeNestedHandler(eqn_plan, nested, env)
    )


@dataclass(frozen=True)
class _MaterializeNestedHandler:
    eqn_plan: EqnPlan
    nested_plan: NestedPlan
    env: ConcreteEnv

    def call(self, spec: CallSpec) -> ResolvedEqn:
        return _materialize_call(self.eqn_plan, self.nested_plan, spec, self.env)

    def map(self, spec: MapSpec) -> ResolvedEqn:
        return _materialize_map(self.eqn_plan, self.nested_plan, spec, self.env)

    def scan(self, spec: ScanSpec) -> ResolvedEqn:
        return _materialize_scan(self.eqn_plan, self.nested_plan, spec, self.env)

    def cond(self, spec: CondSpec) -> ResolvedEqn:
        return _materialize_cond(self.eqn_plan, self.nested_plan, spec, self.env)

    def linear_solve(self, spec: LinearSolveSpec) -> ResolvedEqn:
        return _materialize_linear_solve(
            self.eqn_plan, self.nested_plan, spec, self.env
        )


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
    flat_args: tuple[Any, ...],
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
