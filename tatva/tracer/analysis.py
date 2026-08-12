"""
Static analysis and hierarchical planning for traced JAXPR programs.

This module analyzes a JAXPR without executing numerical computations. Its main
output is a tree of `JaxprPlan` objects describing which equations are relevant,
which values must be available concretely during planning, and how nested JAXPR
primitives should be interpreted.

The analysis distinguishes ordinary primitives from higher-order/nested
constructs such as calls, maps, and scans:

- `CallSpec` represents transparent call-like wrappers such as `jit` and remat.
- `MapSpec` represents independent repeated applications of a body JAXPR,
  including carry-free scans.
- `ScanSpec` represents recurrent scans whose carry creates dependencies between
  iterations.

Concrete requirements are propagated backwards. Primitive rules can request
specific concrete inputs for route construction, while nested plans propagate
concrete requirements across JAXPR boundaries. Stateful scans additionally
solve a fixed point for concrete carry requirements.

This module does not evaluate concrete values, resolve routes, or propagate
derivative dependencies. Those phases are handled by `materialize.py` and
`derivatives.py`.

Key invariants:

- Plans are hierarchical and follow the nesting structure of the source JAXPR.
- Equation indices refer to positions in the original `jaxpr.eqns`.
- Concrete requirements are expressed at JAXPR input/output boundaries.
- Carry-free scans use a `MapSpec`; only scans with carry use a `ScanSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax.extend.core import Jaxpr, JaxprEqn, Var

from tatva.tracer.helpers import _shape_of
from tatva.tracer.nested import (
    CallKind,
    CallSpec,
    MapSpec,
    NestedJaxpr,
    NestedSpec,
    ScanSpec,
    normalize_nested_jaxpr,
)
from tatva.tracer.registry import SEMANTICS


@dataclass(frozen=True)
class NestedPlan:
    """A phase-independent nested body plus its control-flow specification."""

    spec: NestedSpec
    body: JaxprPlan
    consts: tuple[object, ...]
    concrete_inputs: frozenset[int]


@dataclass(frozen=True)
class EqnPlan:
    index: int
    eqn: JaxprEqn
    nested: NestedPlan | None
    # which outputs of this eqn must be available concretely in this jaxpr frame
    concrete_outputs: frozenset[int]


@dataclass(frozen=True)
class JaxprPlan:
    jaxpr: Jaxpr
    # output reachable eqns
    eqns: tuple[EqnPlan, ...]
    # inputs of this jaxpr which must be concretely available to materialize routing
    # inside this frame
    concrete_inputs: frozenset[int]
    # outputs that the parent requires concretely
    concrete_outputs: frozenset[int]


def backward_output_slice(
    jaxpr: Jaxpr,
) -> tuple[tuple[int, JaxprEqn], ...]:
    """Keep equations that can influence any Jaxpr output.

    Returned indices refer to the original jaxpr.eqns tuple.
    """
    required: set[Var] = {var for var in jaxpr.outvars if isinstance(var, Var)}

    kept_reversed: list[tuple[int, JaxprEqn]] = []

    for index in range(len(jaxpr.eqns) - 1, -1, -1):
        eqn = jaxpr.eqns[index]

        if not any(
            isinstance(outvar, Var) and outvar in required for outvar in eqn.outvars
        ):
            continue

        kept_reversed.append((index, eqn))

        for invar in eqn.invars:
            if isinstance(invar, Var):
                required.add(invar)

    kept_reversed.reverse()
    return tuple(kept_reversed)


def analyze(
    jaxpr: Jaxpr,
    *,
    concrete_outputs: frozenset[int] = frozenset(),
) -> JaxprPlan:
    # validate output indices before doing any work
    for index in concrete_outputs:
        if index < 0 or index >= len(jaxpr.outvars):
            raise ValueError(
                f"Jaxpr output index {index} is invalid for "
                f"{len(jaxpr.outvars)} outputs"
            )

    relevant = backward_output_slice(jaxpr)

    # Variables whose values must be known concretely in this frame.
    required: set[Var] = set()

    # Seed requirements requested by the parent.
    for output_index in concrete_outputs:
        atom = jaxpr.outvars[output_index]

        if isinstance(atom, Var):
            required.add(atom)

    nested_plans: dict[int, NestedPlan] = {}
    concrete_outputs_by_eqn: dict[int, frozenset[int]] = {}

    # Walk backwards through this frame.
    for index, eqn in reversed(relevant):
        required_outputs = frozenset(
            output_index
            for output_index, outvar in enumerate(eqn.outvars)
            if isinstance(outvar, Var) and outvar in required
        )

        if required_outputs:
            concrete_outputs_by_eqn[index] = required_outputs

        # Nested primitive.
        nested = _analyze_nested(eqn, concrete_outputs=required_outputs)

        if nested is not None:
            nested_plans[index] = nested

            # Any child input needed concretely becomes a concrete
            # requirement on the corresponding outer equation input.
            for input_index in nested.concrete_inputs:
                if input_index >= len(eqn.invars):
                    raise ValueError(
                        f"nested plan for {eqn.primitive.name} requires "
                        f"input {input_index}, but equation only has "
                        f"{len(eqn.invars)} inputs"
                    )

                atom = eqn.invars[input_index]

                if isinstance(atom, Var):
                    required.add(atom)

            continue

        # Ordinary primitive.
        rule = SEMANTICS.get(eqn.primitive)

        # A route-aware primitive can explicitly request particular
        # inputs to be concretely available.
        for input_index in rule.concrete_inputs(eqn):
            if input_index < 0 or input_index >= len(eqn.invars):
                raise ValueError(
                    f"{eqn.primitive.name}.concrete_inputs returned "
                    f"invalid input index {input_index}"
                )

            atom = eqn.invars[input_index]

            if isinstance(atom, Var):
                required.add(atom)

        # If one of this primitive's outputs itself must be concrete,
        # then evaluating the primitive requires all non-literal inputs.
        #
        # This is deliberately distinct from rule.concrete_inputs():
        #
        #   rule.concrete_inputs
        #       inputs needed to resolve this equation's routing
        #
        #   required_outputs
        #       this equation must run in the concrete subgraph because
        #       a downstream equation needs its value
        #
        if required_outputs:
            for atom in eqn.invars:
                if isinstance(atom, Var):
                    required.add(atom)

    concrete_inputs = frozenset(
        input_index
        for input_index, invar in enumerate(jaxpr.invars)
        if invar in required
    )

    eqn_plans = tuple(
        EqnPlan(
            index=index,
            eqn=eqn,
            nested=nested_plans.get(index),
            concrete_outputs=concrete_outputs_by_eqn.get(index, frozenset()),
        )
        for index, eqn in relevant
    )

    return JaxprPlan(
        jaxpr=jaxpr,
        eqns=eqn_plans,
        concrete_inputs=concrete_inputs,
        concrete_outputs=concrete_outputs,
    )


def _analyze_nested(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan | None:
    match eqn.primitive.name:
        case "jit" | "pjit":
            return _analyze_call(
                eqn, kind=CallKind.JIT, concrete_outputs=concrete_outputs
            )
        case "remat2":
            return _analyze_call(
                eqn, kind=CallKind.REMAT, concrete_outputs=concrete_outputs
            )
        case "scan":
            return _analyze_scan(eqn, concrete_outputs=concrete_outputs)
        case "map":
            return _analyze_map(eqn, concrete_outputs=concrete_outputs)
        case _:
            return None


def _analyze_call(
    eqn: JaxprEqn,
    *,
    kind: CallKind,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    nested = normalize_nested_jaxpr(eqn.params["jaxpr"])

    if len(nested.jaxpr.outvars) != len(eqn.outvars):
        raise ValueError(
            f"{eqn.primitive.name} has {len(eqn.outvars)} outer outputs "
            f"but nested Jaxpr has {len(nested.jaxpr.outvars)} outputs"
        )

    if len(nested.jaxpr.invars) != len(eqn.invars):
        raise ValueError(
            f"{eqn.primitive.name} has {len(eqn.invars)} outer inputs "
            f"but nested Jaxpr has {len(nested.jaxpr.invars)} inputs"
        )

    body = analyze(nested.jaxpr, concrete_outputs=concrete_outputs)

    return NestedPlan(
        spec=CallSpec(call_kind=kind),
        body=body,
        consts=nested.consts,
        concrete_inputs=body.concrete_inputs,
    )


def _analyze_scan(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    nested = normalize_nested_jaxpr(eqn.params["jaxpr"])

    consts_group, carry_group, xs_group = eqn.params["ft_in"].unpack()
    num_consts = len(consts_group)
    num_carry = len(carry_group)
    num_xs = len(xs_group)
    length = int(eqn.params["length"])
    reverse = bool(eqn.params["reverse"])

    if num_consts + num_carry + num_xs != len(eqn.invars):
        raise ValueError(
            "invalid scan metadata: ft_in does not partition all scan inputs"
        )

    if num_carry == 0:
        return _analyze_carry_free_scan(
            eqn, nested, concrete_outputs, num_consts, length, reverse
        )

    body = nested.jaxpr

    if len(body.invars) != len(eqn.invars):
        raise ValueError(
            f"scan has {len(eqn.invars)} outer inputs "
            f"but body has {len(body.invars)} inputs"
        )

    if len(body.outvars) != len(eqn.outvars):
        raise ValueError(
            f"scan has {len(eqn.outvars)} outer outputs "
            f"but body has {len(body.outvars)} outputs"
        )

    if num_carry > len(eqn.outvars):
        raise ValueError("invalid scan metadata: num_carry exceeds output count")

    # ------------------------------------------------------------------
    # Parent concrete-output requirements
    #
    # scan output:
    #
    #   carry_final[0:num_carry]
    #   ys[...]
    #
    # body output:
    #
    #   carry_next[0:num_carry]
    #   y_step[...]
    # ------------------------------------------------------------------

    required_carry_outputs = {
        output_index for output_index in concrete_outputs if output_index < num_carry
    }

    required_y_outputs = {
        output_index for output_index in concrete_outputs if output_index >= num_carry
    }

    # No iteration means final carry == initial carry.
    if length == 0:
        body_plan = analyze(body)

        required_outer_inputs = {
            num_consts + carry_index for carry_index in required_carry_outputs
        }

        return NestedPlan(
            spec=ScanSpec(
                num_consts=num_consts,
                num_carry=num_carry,
                length=length,
                reverse=reverse,
            ),
            body=body_plan,
            consts=nested.consts,
            concrete_inputs=frozenset(required_outer_inputs),
        )

    # ------------------------------------------------------------------
    # Carry fixed point.
    #
    # If routing inside the body requires carry[i] concretely, then the
    # previous iteration must produce carry_out[i] concretely.
    #
    # That additional output requirement can itself require more body
    # inputs, including other carry components.
    # ------------------------------------------------------------------

    required_carry = set(required_carry_outputs)

    while True:
        required_body_outputs = frozenset(required_carry | required_y_outputs)

        body_plan = analyze(
            body,
            concrete_outputs=required_body_outputs,
        )

        required_carry_inputs = {
            body_input_index - num_consts
            for body_input_index in body_plan.concrete_inputs
            if (num_consts <= body_input_index < num_consts + num_carry)
        }

        expanded = required_carry | required_carry_inputs

        if expanded == required_carry:
            break

        required_carry = expanded

    # Body inputs and outer scan inputs use the same ordering:
    #
    #   consts, carry, xs
    #
    # so the body's concrete input requirements directly tell us which
    # outer scan operands must be concrete.
    scan_concrete_inputs = body_plan.concrete_inputs

    return NestedPlan(
        spec=ScanSpec(
            num_consts=num_consts,
            num_carry=num_carry,
            length=length,
            reverse=reverse,
        ),
        body=body_plan,
        consts=nested.consts,
        concrete_inputs=scan_concrete_inputs,
    )


def _analyze_carry_free_scan(
    eqn: JaxprEqn,
    nested: NestedJaxpr,
    concrete_outputs: frozenset[int],
    num_consts: int,
    length: int,
    reverse: bool,
) -> NestedPlan:
    body = nested.jaxpr

    # With no carry:
    # outer inputs: consts..., xs...
    # body inputs:  consts..., x_step...
    # outer outputs: stacked ys...
    # body outputs:  y_step...
    if len(body.outvars) != len(eqn.outvars):
        raise ValueError(
            f"carry-free scan has {len(eqn.outvars)} outer outputs "
            f"but body has {len(body.outvars)} outputs"
        )

    body_plan = analyze(
        body,
        concrete_outputs=concrete_outputs,
    )

    # Body and outer input ordering coincide:
    # consts stay constant;
    # every input after num_consts is sliced along leading axis.
    concrete_inputs = body_plan.concrete_inputs

    return NestedPlan(
        spec=MapSpec(num_consts=num_consts, length=length, reverse=reverse),
        body=body_plan,
        consts=nested.consts,
        concrete_inputs=concrete_inputs,
    )


def _analyze_map(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    nested = normalize_nested_jaxpr(eqn.params["jaxpr"])

    # Your historical map representation apparently used the same
    # scan-style convention. Keep this isolated here because "map"
    # is not something I would bake into generic code.
    num_consts = int(eqn.params.get("num_consts", 0))

    body_plan = analyze(
        nested.jaxpr,
        concrete_outputs=concrete_outputs,
    )

    # Prefer explicit length metadata if present.
    length = eqn.params.get("length")

    if length is None:
        # Infer from first mapped outer operand.
        mapped_inputs = eqn.invars[num_consts:]
        if not mapped_inputs:
            raise ValueError("map has no mapped inputs from which to infer length")

        shape = _shape_of(mapped_inputs[0])
        if not shape:
            raise ValueError("map input must have a leading mapped axis")

        length = shape[0]

    return NestedPlan(
        spec=MapSpec(num_consts=num_consts, length=int(length), reverse=False),
        body=body_plan,
        consts=nested.consts,
        concrete_inputs=body_plan.concrete_inputs,
    )
