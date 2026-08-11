from __future__ import annotations

from typing import TYPE_CHECKING

from tatva.tracer.demand import Demand
from tatva.tracer.dependencies import DependencySet
from tatva.tracer.helpers import _shape_of
from tatva.tracer.rules.elementwise import elementwise_demand
from tatva.tracer.semantics import (
    DemandContext,
    DerivativeRule,
    PrimitiveRule,
    no_hessian,
    no_prepare,
)

if TYPE_CHECKING:
    from tatva.tracer.semantics import RuleContext


def no_output_dependencies(
    ctx: RuleContext,
    prepared: None,
) -> tuple[DependencySet, ...]:
    return ()


def zero_output_dependencies(
    ctx: RuleContext,
    prepared: None,
) -> tuple[DependencySet, ...]:
    return tuple(DependencySet.empty(_shape_of(v), ctx.n_dofs) for v in ctx.eqn.outvars)


def no_input_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    if ctx.eqn.invars:
        raise ValueError(
            f"{ctx.eqn.primitive.name} has inputs, but no demand rule is defined."
        )
    return ()


IOTA = PrimitiveRule(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    ),
    demand=no_input_demand,
)

ZERO_DEPENDENCY = PrimitiveRule(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    ),
    demand=elementwise_demand,
)
