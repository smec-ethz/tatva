from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import lax
from jax.extend.core import JaxprEqn, Primitive, Var
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of
from tatva.tracer.routing import Route
from tatva.tracer.types import (
    ContributionDemand,
    DependencySet,
    HessianAccumulator,
    TensorDemand,
)

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


def no_hessian(
    eqn: JaxprEqn,
    input_deps: tuple[DependencySet, ...],
    route: Route | None,
    acc: HessianAccumulator,
) -> None:
    """A linear primitive contributes no primitive-local second derivatives."""


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


def elementwise_binary_union_dependencies(
    eqn: JaxprEqn,
    input_deps: tuple[DependencySet, ...],
    route: Route | None,
) -> tuple[DependencySet, ...]:
    """Propagate support through a two-input elementwise linear primitive.

    Each output entry depends on the union of the corresponding (after JAX
    broadcasting) entries of both operands.  This is structural Jacobian
    support, so adding the sparse boolean matrices implements set union.
    """
    if len(input_deps) != 2 or len(eqn.outvars) != 1:
        raise ValueError(
            f"{eqn.primitive.name} must have two inputs and one output; got "
            f"{len(input_deps)} inputs and {len(eqn.outvars)} outputs"
        )

    output_shape = _shape_of(eqn.outvars[0])
    lhs, rhs = (dep.broadcast_to(output_shape) for dep in input_deps)
    output_csr = (lhs.csr + rhs.csr).astype(bool).tocsr()
    output_csr.eliminate_zeros()

    return (DependencySet(output_csr, output_shape),)


SEMANTICS[lax.add_p] = PrimitiveRule(
    dependencies=elementwise_binary_union_dependencies,
    hessian=no_hessian,
)
