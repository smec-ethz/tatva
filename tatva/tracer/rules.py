from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.extend.core import JaxprEqn, Primitive, Var
from numpy.typing import NDArray

from tatva.tracer.contributions import ContributionDemand
from tatva.tracer.dependencies import DependencySet
from tatva.tracer.hessian import HessianAccumulator
from tatva.tracer.liveness import TensorDemand
from tatva.tracer.routing import Route

type ConcreteValue = NDArray[Any] | np.generic | int | float | bool
type ConcreteEnv = Mapping[Var, ConcreteValue]


@dataclass(frozen=True)
class ContributionResult:
    inputs: tuple[ContributionDemand | None, ...]
    # if decomposition stops at this eqn, these are the newly discovered roots
    roots: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DemandResult:
    inputs: tuple[TensorDemand | None, ...]


type DependencyRule = Callable[
    [JaxprEqn, tuple[DependencySet, ...], Route | None], tuple[DependencySet, ...]
]
type HessianRule = Callable[
    [JaxprEqn, tuple[DependencySet, ...], Route | None, HessianAccumulator], None
]
type ConcreteInputRule = Callable[[JaxprEqn], tuple[int, ...]]
type RouteRule = Callable[[JaxprEqn, ConcreteEnv], Route | None]
type ContributionRule = Callable[
    [JaxprEqn, tuple[ContributionDemand | None, ...], Route | None], ContributionResult
]
type DemandRule = Callable[
    [JaxprEqn, tuple[TensorDemand, ...], Route | None], DemandResult
]


# Default functions
def no_concrete_inputs(eqn: JaxprEqn) -> tuple[int, ...]:
    return ()


def no_route(eqn: JaxprEqn, env: ConcreteEnv) -> None:
    return None


def stop_contributions(
    eqn: JaxprEqn,
    inputs: tuple[ContributionDemand | None, ...],
    route: Route | None,
) -> ContributionResult:
    # conservative: do not attempt to push additive decomposition through unknown
    # primitives
    raise NotImplementedError(
        f"contribution rule not implemented for {eqn.primitive.name}"
    )


def conservative_demand(
    eqn: JaxprEqn,
    inputs: tuple[TensorDemand, ...],
    route: Route | None,
) -> DemandResult:
    # replace this with implementation that demands Full(shape) for every non-literal
    # input whenever any output is live
    raise NotImplementedError(f"demand rule not implemented for {eqn.primitive.name}")


@dataclass(frozen=True, slots=True)
class PrimitiveRule:
    """Global structural semantics of one JAX primitive."""

    dependencies: DependencyRule
    hessian: HessianRule

    concrete_inputs: ConcreteInputRule = no_concrete_inputs
    route: RouteRule = no_route
    contributions: ContributionRule = stop_contributions
    demand: DemandRule = conservative_demand


SEMANTICS: dict[Primitive, PrimitiveRule] = {}
