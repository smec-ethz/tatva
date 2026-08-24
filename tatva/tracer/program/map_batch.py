from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass

import jax
from jax import core
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, Jaxpr, Var

from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.analysis import JaxprPlan, analyze


@dataclass(frozen=True)
class BatchedMapProgram:
    analysis_closed_jaxpr: ClosedJaxpr
    analysis_plan: JaxprPlan

    execution_closed_jaxpr: ClosedJaxpr
    execution_plan: JaxprPlan

    # Number of real map outputs. Additional outputs can be exposed
    # temporarily for nested contribution seeds.
    num_outputs: int

    # original body Var -> extra batched output position
    exposed: dict[Var, int]

    def __post_init__(self):
        if self.analysis_plan.jaxpr.eqns != self.execution_plan.jaxpr.eqns:
            raise RuntimeError(
                "batched analysis/execution JAXPRs do not share equation structure"
            )


def _exposable_body_vars(jaxpr: Jaxpr) -> tuple[Var, ...]:
    seen: set[Var] = set()
    result: list[Var] = []

    def add(atom) -> None:
        if isinstance(atom, Var) and atom not in seen:
            seen.add(atom)
            result.append(atom)

    for var in jaxpr.invars:
        add(var)

    for var in jaxpr.constvars:
        add(var)

    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            add(outvar)

    return tuple(result)


def build_batched_map_program(
    body_plan: JaxprPlan,
    body_consts: tuple[object, ...],
    *,
    num_consts: int,
    length: int,
    outer_inputs: Sized[Atom],
    expose: tuple[Var, ...] = (),
) -> BatchedMapProgram:
    body = body_plan.jaxpr
    num_outputs = len(body.outvars)

    original_output_positions = {
        var: index for index, var in enumerate(body.outvars) if isinstance(var, Var)
    }

    extras = tuple(
        var
        for var in _exposable_body_vars(body)
        if var not in original_output_positions
    )

    augmented = body.replace(
        outvars=tuple(body.outvars) + extras,
    )

    exposed = dict(original_output_positions)
    for offset, var in enumerate(extras):
        exposed[var] = num_outputs + offset

    if not (0 <= num_consts <= len(body.invars)):
        raise ValueError(
            f"map num_consts={num_consts} is incompatible with "
            f"{len(body.invars)} body inputs"
        )

    if len(outer_inputs) != len(body.invars):
        raise ValueError(
            "map outer/body input arity mismatch: "
            f"{len(outer_inputs)} outer inputs vs "
            f"{len(body.invars)} body inputs"
        )

    for input_index, atom in enumerate(outer_inputs[num_consts:], start=num_consts):
        shape = _shape_of(atom)

        if not shape:
            raise ValueError(
                f"mapped map input {input_index} is scalar and has no leading map axis"
            )

        if shape[0] != length:
            raise ValueError(
                f"mapped map input {input_index} has leading "
                f"extent {shape[0]}, expected map length {length}"
            )

    def one_iteration(*args):
        return tuple(core.eval_jaxpr(augmented, body_consts, *args))

    in_axes = (None,) * num_consts + (0,) * (len(body.invars) - num_consts)
    batched = jax.vmap(one_iteration, in_axes=in_axes, out_axes=0)
    examples = tuple(
        jax.ShapeDtypeStruct(
            _shape_of(atom),
            atom.aval.dtype,
        )
        for atom in outer_inputs
    )

    analysis_closed = jax.make_jaxpr(batched)(*examples)
    analysis_jaxpr = analysis_closed.jaxpr

    execution_jaxpr = analysis_jaxpr.replace(
        outvars=analysis_jaxpr.outvars[:num_outputs]
    )
    execution_closed = ClosedJaxpr(execution_jaxpr, analysis_closed.consts)

    return BatchedMapProgram(
        analysis_closed_jaxpr=analysis_closed,
        analysis_plan=analyze(analysis_jaxpr),
        execution_closed_jaxpr=execution_closed,
        execution_plan=analyze(execution_jaxpr),
        num_outputs=num_outputs,
        exposed=exposed,
    )
