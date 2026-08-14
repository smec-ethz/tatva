from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

from jax.extend.core import JaxprEqn, Literal

from tatva.tracer.core.nested import CallKind
from tatva.tracer.core.routes import ConcreteEnv, Route
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand

if TYPE_CHECKING:
    from tatva.tracer.local.layout import TensorLayout
    from tatva.tracer.local.localize import LocalRoute
    from tatva.tracer.lowering.rules import LoweringRule
    from tatva.tracer.program.dependencies import DependencySet, HessianAccumulator
    from tatva.tracer.program.materialize import JaxprInstance, ResolvedEqn

type ContributionCoefficient = int | float | complex


class ContributionMode(Enum):
    SCALAR = auto()
    DOMAIN = auto()


@dataclass(frozen=True, slots=True)
class ContributionInput:
    input_index: int
    coefficient: ContributionCoefficient
    mode: ContributionMode


@dataclass(frozen=True, slots=True)
class ContributionDecision:
    inputs: tuple[ContributionInput, ...] = ()
    root: bool = False

    unsupported_reason: str | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ContributionContext:
    instance: JaxprInstance
    resolved: ResolvedEqn
    output_index: int
    coefficient: ContributionCoefficient
    mode: ContributionMode


type ContributionRule = Callable[[ContributionContext], ContributionDecision]


def contribution_barrier(ctx: ContributionContext) -> ContributionDecision:
    if ctx.mode is ContributionMode.DOMAIN:
        return ContributionDecision(root=True)

    return ContributionDecision(
        unsupported_reason=(
            "cannot decompose scalar objective through primitive "
            f"{ctx.resolved.plan.eqn.primitive.name!r}; expected an additive "
            "scalar tail ending, e.g. in reduce_sum"
        )
    )


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
class RouteLocalizationContext:
    """Inputs available when converting a global route to rank-local rows."""

    eqn: JaxprEqn
    route: Route
    input_layouts: tuple[TensorLayout | None, ...]
    output_layouts: tuple[TensorLayout | None, ...]


type RouteLocalizationRule = Callable[[RouteLocalizationContext], LocalRoute]


@dataclass(frozen=True, slots=True)
class LocalizationSemantics:
    """Primitive-local capabilities used while constructing a local plan."""

    localize_route: RouteLocalizationRule | None = None


NO_LOCALIZATION = LocalizationSemantics()


@dataclass(frozen=True, slots=True)
class DerivativeRule[T]:
    prepare: PrepareRule[T]
    dependencies: DependencyRule[T]
    hessian: HessianRule[T]


@dataclass(frozen=True, slots=True)
class OperationSemantics[T]:
    """Global structural semantics of one JAX primitive."""

    derivatives: DerivativeRule[T]

    concrete_inputs: ConcreteInputRule = no_concrete_inputs
    route: RouteRule = no_route
    demand: DemandRule = conservative_demand
    contribution: ContributionRule = contribution_barrier
    localization: LocalizationSemantics = NO_LOCALIZATION
    lowering: LoweringRule | None = None


# Nested-operation semantics
@dataclass(frozen=True, slots=True)
class CallAnalysisSemantics:
    call_kind: CallKind


@dataclass(frozen=True, slots=True)
class ScanAnalysisSemantics:
    pass


type NestedAnalysisSemantics = CallAnalysisSemantics | ScanAnalysisSemantics


@dataclass(frozen=True, slots=True)
class NestedOperationSemantics:
    """Semantics for a primitive containing a nested JAXPR."""

    analysis: NestedAnalysisSemantics


type RegisteredOperationSemantics = OperationSemantics | NestedOperationSemantics
