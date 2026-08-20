"""
Static analysis and hierarchical planning for traced JAXPR programs.

This module analyzes a JAXPR without executing numerical computations. Its main
output is a tree of `JaxprPlan` objects describing which equations are relevant,
which values must be available concretely during planning, and how nested JAXPR
primitives should be interpreted.

The analysis distinguishes ordinary primitives from higher-order/nested
constructs such as calls, maps, and scans:

- `CallSpec` represents transparent call-like wrappers such as jit and remat.
- `MapSpec` is Tatva's normalized representation for independent repeated applications. In
  JAX 0.11 this is produced from carry-free scans.
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

from tatva.tracer.core.nested import (
    CallbackBinding,
    CallSpec,
    CondSpec,
    CustomJvpBinding,
    CustomJvpSpec,
    LinearSolveCallbackSpec,
    LinearSolveSpec,
    MapSpec,
    NestedJaxpr,
    NestedSpec,
    ScanSpec,
    normalize_nested_jaxpr,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import (
    CallAnalysisSemantics,
    CondAnalysisSemantics,
    CustomJvpAnalysisSemantics,
    LinearSolveAnalysisSemantics,
    NestedAnalysisSemantics,
    NestedOperationSemantics,
    RouteRequirement,
    ScanAnalysisSemantics,
)
from tatva.tracer.program.custom_jvp import extract_custom_jvp_parameters


@dataclass(frozen=True)
class NestedPlan:
    """A phase-independent nested body plus its control-flow specification."""

    spec: NestedSpec
    branches: tuple[JaxprPlan, ...]
    branch_consts: tuple[tuple[object, ...], ...]
    concrete_inputs: frozenset[int]

    @property
    def body(self) -> JaxprPlan:
        return self.branches[0]

    @property
    def consts(self) -> tuple[object, ...]:
        return self.branch_consts[0]


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
        semantics = SEMANTICS.get(eqn.primitive)

        if isinstance(semantics, NestedOperationSemantics):
            nested = _analyze_nested(
                eqn,
                semantics=semantics.analysis,
                concrete_outputs=required_outputs,
            )
            nested_plans[index] = nested

            # any child input needed concretely becomes a concrete requirement on the
            # corresponding outer equation input
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

        routing = semantics.routing

        if routing is not None and routing.requirement is RouteRequirement.REQUIRED:
            for input_index in routing.inputs(eqn):
                if input_index < 0 or input_index >= len(eqn.invars):
                    raise ValueError(
                        f"{eqn.primitive.name}.routing.inputs returned "
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
    semantics: NestedAnalysisSemantics,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    if isinstance(semantics, CustomJvpAnalysisSemantics):
        return _analyze_custom_jvp(
            eqn,
            concrete_outputs=concrete_outputs,
        )

    if isinstance(semantics, CallAnalysisSemantics):
        return _analyze_call(
            eqn,
            semantics=semantics,
            concrete_outputs=concrete_outputs,
        )

    if isinstance(semantics, ScanAnalysisSemantics):
        return _analyze_scan(
            eqn,
            concrete_outputs=concrete_outputs,
        )

    if isinstance(semantics, CondAnalysisSemantics):
        return _analyze_cond(
            eqn,
            concrete_outputs=concrete_outputs,
        )
    if isinstance(semantics, LinearSolveAnalysisSemantics):
        return _analyze_linear_solve(
            eqn,
            concrete_outputs=concrete_outputs,
        )

    raise TypeError(f"unsupported nested analysis semantics {type(semantics).__name__}")


def _analyze_custom_jvp(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    extracted = extract_custom_jvp_parameters(eqn)
    if len(extracted.primal_jaxpr.invars) != len(eqn.invars):
        raise NotImplementedError("custom_jvp primal input ABI does not match its call")
    if len(extracted.primal_jaxpr.outvars) != len(eqn.outvars):
        raise NotImplementedError(
            "custom_jvp primal output ABI does not match its call"
        )

    primal = analyze(
        extracted.primal_jaxpr,
        concrete_outputs=concrete_outputs,
    )
    # Derivative callback values are runtime-only. Concrete output propagation
    # belongs to the primal callback, while JVP routing requirements are still
    # discovered by analyzing its complete program.
    jvp = analyze(
        extracted.jvp_jaxpr,
    )
    dynamic_arity = len(eqn.invars) - extracted.num_consts
    bindings = tuple(
        CustomJvpBinding(extracted.num_consts + index) for index in range(dynamic_arity)
    ) + tuple(
        CustomJvpBinding(extracted.num_consts + index, tangent=True)
        for index in range(dynamic_arity)
    )

    # Optional structural routing (for example a partially resolved gather) does
    # not create a concrete-input requirement. Genuine planning requirements
    # (for example a dynamic cond predicate) still cross this boundary and remain
    # errors if they depend on a runtime tangent.
    concrete: set[int] = set(primal.concrete_inputs)
    for child_index in jvp.concrete_inputs:
        binding = bindings[child_index]
        if binding.tangent:
            raise NotImplementedError(
                "custom_jvp planning may not require a runtime tangent concretely"
            )
        concrete.add(binding.outer_input_index)

    return NestedPlan(
        spec=CustomJvpSpec(
            jvp_bindings=bindings,
            output_zeros=extracted.output_zeros,
        ),
        branches=(primal, jvp),
        branch_consts=((), extracted.jvp_consts),
        concrete_inputs=frozenset(concrete),
    )


def _analyze_linear_solve(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    """Analyze all executable custom-linear-solve callbacks.

    JAX stores captured callback constants in consecutive operand blocks; each
    callback receives those captures followed by one runtime vector argument.
    """
    if bool(eqn.params.get("has_aux", False)):
        raise NotImplementedError("custom_linear_solve(has_aux=True) is not supported")
    if len(eqn.outvars) != 1 or len(eqn.invars) < 1:
        raise NotImplementedError(
            "custom_linear_solve currently requires one RHS/result"
        )
    lengths = eqn.params["const_lengths"]
    jaxprs = eqn.params["jaxprs"]
    sizes = (int(lengths.matvec), int(lengths.solve), int(lengths.transpose_solve))
    starts = (
        0,
        int(lengths.matvec) + int(lengths.vecmat),
        int(lengths.matvec) + int(lengths.vecmat) + int(lengths.solve),
    )
    rhs_start = starts[2] + int(lengths.transpose_solve)
    rhs_indices = tuple(range(rhs_start, len(eqn.invars)))
    if len(rhs_indices) != 1:
        raise NotImplementedError("custom_linear_solve currently requires one RHS")
    raw = (jaxprs.matvec, jaxprs.solve, jaxprs.transpose_solve)
    names = ("matvec", "solve", "transpose_solve")
    bodies: list[JaxprPlan] = []
    consts: list[tuple[object, ...]] = []
    callbacks: list[LinearSolveCallbackSpec] = []
    concrete: set[int] = set()
    for name, value, start, size in zip(names, raw, starts, sizes, strict=True):
        child = normalize_nested_jaxpr(value)
        bindings = tuple(CallbackBinding(i) for i in range(start, start + size)) + (
            CallbackBinding(),
        )
        if len(child.jaxpr.invars) != len(bindings):
            raise ValueError(f"custom_linear_solve {name} callback input mismatch")
        # Callback output routing is independent of solution demand, but it is
        # still traversed so captured concrete requirements are propagated.
        # The runtime vector is unavailable during planning, so solution demand
        # must not be converted into a concrete callback-output requirement.
        body = analyze(child.jaxpr)
        for i in body.concrete_inputs:
            binding = bindings[i]
            if binding.outer_input_index is not None:
                concrete.add(binding.outer_input_index)
        bodies.append(body)
        consts.append(child.consts)
        callbacks.append(LinearSolveCallbackSpec(name, bindings))
    return NestedPlan(
        spec=LinearSolveSpec(*callbacks, rhs_indices=rhs_indices, has_aux=False),
        branches=tuple(bodies),
        branch_consts=tuple(consts),
        concrete_inputs=frozenset(concrete),
    )


def _analyze_call(
    eqn: JaxprEqn,
    *,
    semantics: CallAnalysisSemantics,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    target = semantics.target(eqn)
    nested = normalize_nested_jaxpr(target.body)

    spec = CallSpec(
        call_kind=semantics.call_kind,
        input_indices=target.input_indices,
    )
    input_indices = spec.resolved_input_indices(len(eqn.invars))

    if len(nested.jaxpr.outvars) != len(eqn.outvars):
        raise ValueError(
            f"{eqn.primitive.name} has {len(eqn.outvars)} outer outputs "
            f"but nested Jaxpr has {len(nested.jaxpr.outvars)} outputs"
        )

    if len(nested.jaxpr.invars) != len(input_indices):
        raise ValueError(
            f"{eqn.primitive.name} call boundary selects {len(input_indices)} inputs "
            f"but nested Jaxpr has {len(nested.jaxpr.invars)} inputs"
        )

    body = analyze(
        nested.jaxpr,
        concrete_outputs=concrete_outputs,
    )
    concrete_inputs = frozenset(
        spec.outer_input_index(child_index, outer_arity=len(eqn.invars))
        for child_index in body.concrete_inputs
    )

    return NestedPlan(
        spec=spec,
        branches=(body,),
        branch_consts=(nested.consts,),
        concrete_inputs=concrete_inputs,
    )


def _analyze_cond(
    eqn: JaxprEqn,
    *,
    concrete_outputs: frozenset[int],
) -> NestedPlan:
    branches_raw = eqn.params.get("branches")
    if branches_raw is None:
        raise ValueError(f"{eqn.primitive.name} is missing 'branches' parameter")

    branches = tuple(normalize_nested_jaxpr(b) for b in branches_raw)
    spec = CondSpec(num_branches=len(branches))

    num_operands = len(eqn.invars) - 1
    if num_operands < 0:
        raise ValueError(f"{eqn.primitive.name} has no branch selector input")

    for idx, b in enumerate(branches):
        if len(b.jaxpr.outvars) != len(eqn.outvars):
            raise ValueError(
                f"{eqn.primitive.name} has {len(eqn.outvars)} outer outputs "
                f"but branch {idx} has {len(b.jaxpr.outvars)} outputs"
            )
        if len(b.jaxpr.invars) != num_operands:
            raise ValueError(
                f"{eqn.primitive.name} has {num_operands} operand inputs "
                f"but branch {idx} has {len(b.jaxpr.invars)} inputs"
            )

    branch_plans = tuple(
        analyze(
            b.jaxpr,
            concrete_outputs=concrete_outputs,
        )
        for b in branches
    )

    concrete_inputs = {0}
    for b_plan in branch_plans:
        for child_index in b_plan.concrete_inputs:
            concrete_inputs.add(
                spec.outer_input_index(child_index, outer_arity=len(eqn.invars))
            )

    return NestedPlan(
        spec=spec,
        branches=branch_plans,
        branch_consts=tuple(b.consts for b in branches),
        concrete_inputs=frozenset(concrete_inputs),
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
            eqn,
            nested,
            concrete_outputs,
            num_consts,
            length,
            reverse,
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
        body_plan = analyze(
            body,
        )

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
            branches=(body_plan,),
            branch_consts=(nested.consts,),
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
        branches=(body_plan,),
        branch_consts=(nested.consts,),
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
        branches=(body_plan,),
        branch_consts=(nested.consts,),
        concrete_inputs=concrete_inputs,
    )
