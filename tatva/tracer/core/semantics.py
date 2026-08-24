from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal

from tatva.tracer.core.nested import CallKind
from tatva.tracer.core.route_fragments import RouteFragment, RouteRequest
from tatva.tracer.core.routes import ConcreteEnv, Route
from tatva.tracer.core.tagged import Tagged, TaggedDemand, active_blocks
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand

if TYPE_CHECKING:
    from tatva.tracer.local.layout import TensorLayout
    from tatva.tracer.local.localize import LocalRoute
    from tatva.tracer.lowering.rules import LoweringRule
    from tatva.tracer.program.dependencies import DependencySet, InteractionGraph

type ContributionCoefficient = int | float | complex


@dataclass(frozen=True, slots=True)
class RegionalConcreteContext:
    eqn: JaxprEqn
    output_index: int
    demand: TensorDemand


type RegionalConcreteEvaluator = Callable[
    [RegionalConcreteContext, tuple[Any, ...]], Any
]


@dataclass(frozen=True, slots=True)
class RegionalConcrete:
    backpropagate: DemandRule
    evaluate: RegionalConcreteEvaluator
    allow_dead_inputs: bool = False


@dataclass(frozen=True, slots=True)
class FullConcrete:
    reason: str


@dataclass(frozen=True, slots=True)
class UnsupportedConcrete:
    reason: str


type RegionalConcretePlan = RegionalConcrete | FullConcrete | UnsupportedConcrete
type RegionalConcreteRule = Callable[[RegionalConcreteContext], RegionalConcretePlan]


def full_concrete_evaluation(ctx: RegionalConcreteContext) -> FullConcrete:
    return FullConcrete(
        f"{ctx.eqn.primitive.name} has no regional concrete evaluation rule"
    )


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
    eqn: JaxprEqn
    output_index: int
    coefficient: ContributionCoefficient
    mode: ContributionMode
    concrete_scalar: Callable[[int], ContributionCoefficient | None]


type ContributionRule = Callable[[ContributionContext], ContributionDecision]


def contribution_barrier(ctx: ContributionContext) -> ContributionDecision:
    if ctx.mode is ContributionMode.DOMAIN:
        return ContributionDecision(root=True)

    return ContributionDecision(
        unsupported_reason=(
            "cannot decompose scalar objective through primitive "
            f"{ctx.eqn.primitive.name!r}; expected an additive "
            "scalar tail ending, e.g. in reduce_sum"
        )
    )


# --------------------------------
# default rules used many times
# --------------------------------
def no_hessian(
    ctx: RuleContext,
    prepared: object,
    acc: InteractionGraph,
) -> None:
    """A linear primitive contributes no primitive-local second derivatives."""


def no_prepare(ctx: RuleContext) -> None:
    return None


def no_concrete_inputs(eqn: JaxprEqn) -> tuple[int, ...]:
    return ()


def no_route(eqn: JaxprEqn, env: ConcreteEnv) -> None:
    return None


def no_route_fragment(
    eqn: JaxprEqn,
    env: ConcreteEnv,
    request: RouteRequest,
) -> None:
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


def conservative_tagged_demand(
    ctx: TaggedDemandContext,
) -> tuple[Tagged, ...]:
    """Conservatively attach every active color to every input entry."""
    blocks = active_blocks(ctx.output_demands)
    return tuple(
        None
        if isinstance(atom, Literal)
        else TaggedDemand.full(_shape_of(atom), blocks)
        for atom in ctx.eqn.invars
    )


# --------------------------------
# Rule context and protocols
# --------------------------------
@dataclass(frozen=True)
class RuleContext:
    eqn: JaxprEqn
    input_deps: tuple[DependencySet, ...]
    route: Route | None
    n_symbols: int

    @property
    def n_dofs(self) -> int:
        """Compatibility alias for primitive rules not yet renamed."""
        return self.n_symbols


class PrepareRule[T](Protocol):
    def __call__(self, ctx: RuleContext) -> T: ...


class DependencyRule[T](Protocol):
    def __call__(self, ctx: RuleContext, prepared: T) -> tuple[DependencySet, ...]: ...


class InteractionRule[T](Protocol):
    def __call__(
        self, ctx: RuleContext, prepared: T, acc: InteractionGraph
    ) -> None: ...


@dataclass(frozen=True)
class DemandContext:
    eqn: JaxprEqn
    output_demands: tuple[Demand, ...]
    route: Route | RouteFragment | None


@dataclass(frozen=True)
class TaggedDemandContext:
    eqn: JaxprEqn
    output_demands: tuple[Tagged, ...]
    route: Route | RouteFragment | None


type ConcreteInputRule = Callable[[JaxprEqn], tuple[int, ...]]
type RouteRule = Callable[[JaxprEqn, ConcreteEnv], Route | None]
type RouteFragmentRule = Callable[
    [JaxprEqn, ConcreteEnv, RouteRequest], RouteFragment | None
]
type RouteConcreteDemandRule = Callable[[JaxprEqn, RouteRequest], tuple[Demand, ...]]


@dataclass(frozen=True, slots=True)
class PartialRouteContext:
    """Demand-scoped access used by partially resolvable route rules.

    ``read_input`` returns the requested concrete region when planning can
    prove it without evaluating unavailable runtime values, otherwise ``None``.
    """

    eqn: JaxprEqn
    request: RouteRequest
    read_input: Callable[[int, Demand], object | None]


type PartialRouteFragmentRule = Callable[[PartialRouteContext], RouteFragment | None]
type DemandRule = Callable[[DemandContext], tuple[Demand, ...]]
type TaggedDemandRule = Callable[[TaggedDemandContext], tuple[Tagged, ...]]


@dataclass(frozen=True, slots=True)
class RouteLocalizationContext:
    """Inputs available when converting a global route to rank-local rows."""

    eqn: JaxprEqn
    route: Route | RouteFragment
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
    interactions: InteractionRule[T]

    @property
    def hessian(self) -> InteractionRule[T]:
        """Compatibility alias; interactions is the canonical compiler term."""
        return self.interactions


class RouteRequirement(Enum):
    REQUIRED = auto()
    OPTIONAL = auto()


@dataclass(frozen=True, slots=True)
class RoutingSemantics:
    """Structural routing semantics for an ordinary primitive.

    inputs:
        Inputs needed to specialize the primitive into global scalar-row
        routing geometry.

    requirement:
        REQUIRED:
            failure to specialize is a planning error.

        OPTIONAL:
            specialization improves precision but is not required for
            correctness. The primitive's ordinary demand/lowering rules must
            support route=None.
    """

    inputs: ConcreteInputRule
    resolve: RouteRule
    fragment: RouteFragmentRule = no_route_fragment
    concrete_demands: RouteConcreteDemandRule | None = None
    partial_fragment: PartialRouteFragmentRule | None = None
    requirement: RouteRequirement = RouteRequirement.REQUIRED


@dataclass(frozen=True, slots=True)
class OperationSemantics[T]:
    """Global structural semantics of one JAX primitive."""

    derivatives: DerivativeRule[T]

    routing: RoutingSemantics | None = None

    demand: DemandRule = conservative_demand
    tagged_demand: TaggedDemandRule = conservative_tagged_demand
    regional_concrete: RegionalConcreteRule = full_concrete_evaluation
    contribution: ContributionRule = contribution_barrier
    localization: LocalizationSemantics = NO_LOCALIZATION
    lowering: LoweringRule | None = None


# Nested-operation semantics
@dataclass(frozen=True, slots=True)
class CallTarget:
    """Executable child of a call-like primitive.

    input_indices maps child input i to outer equation input input_indices[i]. None means
    the normal identity boundary used by jit/remat.
    """

    body: object
    input_indices: tuple[int, ...] | None = None


type CallTargetRule = Callable[[JaxprEqn], CallTarget]


def direct_call_target(eqn: JaxprEqn) -> CallTarget:
    value = eqn.params.get("jaxpr")
    if not isinstance(value, (Jaxpr, ClosedJaxpr)):
        raise TypeError(
            f"call-like primitive {eqn.primitive.name} does not contain a Jaxpr-valued 'jaxpr' parameter"
        )
    return CallTarget(body=value)


@dataclass(frozen=True, slots=True)
class CallAnalysisSemantics:
    call_kind: CallKind
    target: CallTargetRule = direct_call_target


@dataclass(frozen=True, slots=True)
class ScanAnalysisSemantics:
    pass


@dataclass(frozen=True, slots=True)
class CondAnalysisSemantics:
    pass


@dataclass(frozen=True, slots=True)
class LinearSolveAnalysisSemantics:
    pass


@dataclass(frozen=True, slots=True)
class CustomJvpAnalysisSemantics:
    pass


type NestedAnalysisSemantics = (
    CallAnalysisSemantics
    | ScanAnalysisSemantics
    | CondAnalysisSemantics
    | LinearSolveAnalysisSemantics
    | CustomJvpAnalysisSemantics
)


@dataclass(frozen=True, slots=True)
class NestedOperationSemantics:
    """Semantics for a primitive containing a nested JAXPR."""

    analysis: NestedAnalysisSemantics


type RegisteredOperationSemantics = OperationSemantics | NestedOperationSemantics
