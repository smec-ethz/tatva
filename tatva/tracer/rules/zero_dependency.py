from __future__ import annotations

from typing import TYPE_CHECKING

from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    OperationSemantics,
    no_hessian,
    no_prepare,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand
from tatva.tracer.program.dependencies import DependencySet
from tatva.tracer.rules.elementwise import elementwise_demand

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext


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


def no_op_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    return tuple(None for _ in ctx.eqn.invars)


IOTA = OperationSemantics(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    ),
    demand=no_input_demand,
)

ZERO_DEPENDENCY = OperationSemantics(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    ),
    demand=elementwise_demand,
)

NO_OP = OperationSemantics(
    DerivativeRule(
        no_prepare,
        no_output_dependencies,
        no_hessian,
    ),
    demand=no_op_demand,
)
