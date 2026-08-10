from dataclasses import dataclass

from jax.extend.core import Jaxpr, JaxprEqn, Var

from tatva.tracer.registry import SEMANTICS


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


def dof_value_dependencies(jaxpr: Jaxpr) -> dict[Var, bool]:
    """Conservative value provenance.

    True means the concrete value may change when the DOF vector changes.
    This is deliberately different from derivative dependence:
    stop_gradient(u) is still value-dependent on u.
    """
    if not jaxpr.invars:
        raise ValueError("Expected a DOF-vector input")

    depends: dict[Var, bool] = {}

    depends[jaxpr.invars[0]] = True

    for var in jaxpr.invars[1:]:
        depends[var] = False

    for var in jaxpr.constvars:
        depends[var] = False

    for eqn in jaxpr.eqns:
        output_depends = any(
            depends[invar] for invar in eqn.invars if isinstance(invar, Var)
        )

        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                depends[outvar] = output_depends

    return depends


def validate_static_concrete_inputs(
    plan: AnalysisPlan,
    value_dependencies: dict[Var, bool],
) -> None:
    for eqn in plan.eqns:
        rule = SEMANTICS.get(eqn.primitive)

        for input_index in rule.concrete_inputs(eqn):
            atom = eqn.invars[input_index]

            if isinstance(atom, Var) and value_dependencies[atom]:
                raise ValueError(
                    f"{eqn.primitive.name} requires a static routing value, "
                    f"but input {input_index} depends on the DOF vector: {atom}"
                )
