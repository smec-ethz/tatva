from __future__ import annotations

from typing import TYPE_CHECKING

from tatva.tracer.helpers import _shape_of
from tatva.tracer.semantics import DerivativeRule, PrimitiveRule, no_hessian, no_prepare
from tatva.tracer.dependencies import DependencySet

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


IOTA = PrimitiveRule(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    )
)

ZERO_DEPENDENCY = PrimitiveRule(
    DerivativeRule(
        no_prepare,
        zero_output_dependencies,
        no_hessian,
    )
)
