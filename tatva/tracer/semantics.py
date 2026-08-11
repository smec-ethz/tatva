from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from jax.extend.core import JaxprEqn, Literal

from tatva.tracer.demand import Demand, TensorDemand
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import ConcreteEnv, Route

if TYPE_CHECKING:
    from tatva.tracer.dependencies import DependencySet, HessianAccumulator


# --------------------------------
# default rules used many times
# --------------------------------
def no_hessian(
    ctx: RuleContext,
    prepared: object,
    acc: HessianAccumulator,
) -> None:
    """A linear primitive contributes no primitive-local second derivatives."""


def no_prepare(ctx: RuleContext) -> None:
    return None


def no_concrete_inputs(eqn: JaxprEqn) -> tuple[int, ...]:
    return ()


def no_route(eqn: JaxprEqn, env: ConcreteEnv) -> None:
    return None


def conservative_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    if not any(demand is not None for demand in ctx.output_demands):
        return tuple(None for _ in ctx.eqn.invars)

    warnings.warn(f"Conservative demand rule used for {ctx.eqn.primitive.name}. ")

    result: list[Demand] = []
    for atom in ctx.eqn.invars:
        if isinstance(atom, Literal):
            result.append(None)
        else:
            result.append(TensorDemand.full(_shape_of(atom)))

    return tuple(result)


# --------------------------------
# Rule context and protocols
# --------------------------------
@dataclass(frozen=True)
class RuleContext:
    eqn: JaxprEqn
    input_deps: tuple[DependencySet, ...]
    route: Route | None
    n_dofs: int


class PrepareRule[T](Protocol):
    def __call__(self, ctx: RuleContext) -> T: ...


class DependencyRule[T](Protocol):
    def __call__(self, ctx: RuleContext, prepared: T) -> tuple[DependencySet, ...]: ...


class HessianRule[T](Protocol):
    def __call__(
        self, ctx: RuleContext, prepared: T, acc: HessianAccumulator
    ) -> None: ...


@dataclass(frozen=True)
class DemandContext:
    eqn: JaxprEqn
    output_demands: tuple[Demand, ...]
    route: Route | None


@dataclass(frozen=True)
class DemandResult:
    inputs: tuple[TensorDemand | None, ...]


type ConcreteInputRule = Callable[[JaxprEqn], tuple[int, ...]]
type RouteRule = Callable[[JaxprEqn, ConcreteEnv], Route | None]
type DemandRule = Callable[[DemandContext], tuple[Demand, ...]]


@dataclass(frozen=True, slots=True)
class DerivativeRule[T]:
    prepare: PrepareRule[T]
    dependencies: DependencyRule[T]
    hessian: HessianRule[T]


@dataclass(frozen=True, slots=True)
class PrimitiveRule[T]:
    """Global structural semantics of one JAX primitive."""

    derivatives: DerivativeRule[T]

    concrete_inputs: ConcreteInputRule = no_concrete_inputs
    route: RouteRule = no_route

    demand: DemandRule = conservative_demand
