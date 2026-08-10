from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from jax.extend.core import JaxprEqn

from tatva.tracer.model import ConcreteEnv, Route

if TYPE_CHECKING:
    from tatva.tracer.dependencies import DependencySet, HessianAccumulator


type ContributionDemand = object  # placeholder until contribution analysis is added
type TensorDemand = object  # placeholder until liveness analysis is added


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
class ContributionResult:
    inputs: tuple[ContributionDemand | None, ...]
    # if decomposition stops at this eqn, these are the newly discovered roots
    roots: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DemandResult:
    inputs: tuple[TensorDemand | None, ...]


type ConcreteInputRule = Callable[[JaxprEqn], tuple[int, ...]]
type RouteRule = Callable[[JaxprEqn, ConcreteEnv], Route | None]
type ContributionRule = Callable[
    [JaxprEqn, tuple[ContributionDemand | None, ...], Route | None], ContributionResult
]
type DemandRule = Callable[
    [JaxprEqn, tuple[TensorDemand, ...], Route | None], DemandResult
]


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
    contributions: ContributionRule = stop_contributions
    demand: DemandRule = conservative_demand
