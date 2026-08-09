from dataclasses import dataclass

from jax.extend.core import Jaxpr, JaxprEqn, Var

from tatva.tracer.rules import SEMANTICS


@dataclass(frozen=True)
class AnalysisPlan:
    eqns: tuple[JaxprEqn, ...]
    concrete_vars: frozenset[Var]
    concrete_eqns: frozenset[JaxprEqn]


def analyze(jaxpr: Jaxpr) -> AnalysisPlan:
    relevant_eqns = backward_output_slice(jaxpr)
    concrete_vars, concrete_eqns = backward_concrete_slice(relevant_eqns)

    return AnalysisPlan(relevant_eqns, concrete_vars, concrete_eqns)


def backward_output_slice(jaxpr: Jaxpr) -> tuple[JaxprEqn, ...]:
    """Keep exactly the equations that can influence a JAXPR output. Assumes ordinary SSA
    JAXPR semantics: every Var is produced once."""
    required: set[Var] = {var for var in jaxpr.outvars if isinstance(var, Var)}

    kept_reversed: list[JaxprEqn] = []

    for eqn in reversed(jaxpr.eqns):
        if not any(
            isinstance(outvar, Var) and outvar in required for outvar in eqn.outvars
        ):
            continue

        kept_reversed.append(eqn)

        for invar in eqn.invars:
            if isinstance(invar, Var):
                required.add(invar)

    kept_reversed.reverse()
    return tuple(kept_reversed)


def backward_concrete_slice(
    eqns: tuple[JaxprEqn, ...],
) -> tuple[frozenset[Var], frozenset[JaxprEqn]]:
    """Find variables whose concrete values must be available to resolve structural
    routing. A primitive seeds this requirement through PrimitiveRule.concrete_inputs. The
    requirement is then propagated backwards through the producers."""
    # firsst, seed variables directly requested by structural primitive rules
    required: set[Var] = set()

    for eqn in eqns:
        rule = SEMANTICS.get(eqn.primitive)
        if rule is None:
            raise ValueError(f"Primitive {eqn.primitive} has no registered rule.")

        for index in rule.concrete_inputs(eqn):
            atom = eqn.invars[index]
            if isinstance(atom, Var):
                required.add(atom)

    concrete_eqns: set[JaxprEqn] = set()

    # then propagate backwards through the producers
    for eqn in reversed(eqns):
        if not any(isinstance(v, Var) and v in required for v in eqn.outvars):
            continue

        concrete_eqns.add(eqn)

        for invar in eqn.invars:
            if isinstance(invar, Var):
                required.add(invar)

    return frozenset(required), frozenset(concrete_eqns)
