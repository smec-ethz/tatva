from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Var

from tatva.tracer.analysis import AnalysisPlan
from tatva.tracer.model import ConcreteValue

# Separate “concrete” from “route-stable”. concrete.py currently seeds all function
# inputs, including the DOF vector, so it can happily evaluate indices = f(u) and bake
# those indices into a route. Numerically concrete does not mean safe to compile
# statically. Add a tiny forward value-provenance analysis:
#
# depends_on_dofs[u] = True
# depends_on_dofs[other_inputs] = False
#
# for eqn:
#     depends_on_dofs[out] = any(depends_on_dofs[in] for in invars)
#
# Importantly, stop_gradient(u) is still value-dependent on u, even though its derivative
# dependency is zero. Reject a routing operand when depends_on_dofs[index_var] is true.


def evaluate_concrete(
    closed_jaxpr: ClosedJaxpr,
    flat_args: list[Any],
    plan: AnalysisPlan,
) -> dict[Var, ConcreteValue]:
    env: dict[Var, ConcreteValue] = {}

    _seed_inputs(env, closed_jaxpr, flat_args)

    concrete_eqns = set(plan.concrete_eqns)

    for eqn in plan.eqns:
        if eqn not in concrete_eqns:
            continue

        inputs = tuple(_read(atom, env) for atom in eqn.invars)

        # The backward concrete slice should guarantee these are available.
        if any(value is None for value in inputs):
            missing = [atom for atom, value in zip(eqn.invars, inputs) if value is None]
            raise RuntimeError(
                f"Missing concrete inputs for {eqn.primitive.name}: {missing}"
            )

        outputs = _eval_eqn(eqn, inputs)

        if len(outputs) != len(eqn.outvars):
            raise RuntimeError(
                f"{eqn.primitive.name} returned {len(outputs)} outputs, "
                f"expected {len(eqn.outvars)}"
            )

        for var, value in zip(eqn.outvars, outputs):
            if isinstance(var, Var):
                env[var] = value

    return env


def _seed_inputs(
    env: dict[Var, ConcreteValue],
    closed_jaxpr: ClosedJaxpr,
    flat_args: list[Any],
) -> None:
    for var, value in zip(closed_jaxpr.jaxpr.invars, flat_args):
        env[var] = np.asarray(value)

    for var, value in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
        env[var] = np.asarray(value)


def _read(
    atom: Atom,
    env: Mapping[Var, ConcreteValue],
) -> ConcreteValue | None:
    if isinstance(atom, Literal):
        return atom.val

    return env.get(atom)


def _eval_eqn(
    eqn: JaxprEqn,
    inputs: tuple[ConcreteValue, ...],
) -> tuple[ConcreteValue, ...]:
    result = eqn.primitive.bind(
        *(jnp.asarray(x) for x in inputs),
        **eqn.params,
    )

    if eqn.primitive.multiple_results:
        return tuple(np.asarray(x) for x in result)

    return (np.asarray(result),)
