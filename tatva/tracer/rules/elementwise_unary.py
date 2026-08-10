from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tatva.tracer.helpers import _shape_of
from tatva.tracer.semantics import DerivativeRule, PrimitiveRule, no_hessian, no_prepare
from tatva.tracer.dependencies import DependencySet, HessianAccumulator

if TYPE_CHECKING:
    from tatva.tracer.semantics import RuleContext


@dataclass(frozen=True)
class ElementwiseUnaryData:
    dep: DependencySet
    output_shape: tuple[int, ...]


def prepare_elementwise_unary(ctx: RuleContext) -> ElementwiseUnaryData:
    if len(ctx.input_deps) != 1 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have one input and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    dep = ctx.input_deps[0]
    output_shape = _shape_of(ctx.eqn.outvars[0])

    if dep.shape != output_shape:
        dep = dep.reshape(*output_shape)

    return ElementwiseUnaryData(dep, output_shape)


def unary_passthrough_dependencies(
    ctx: RuleContext,
    prepared: ElementwiseUnaryData,
) -> tuple[DependencySet, ...]:
    return (prepared.dep,)


def nonlinear_unary_hessian(
    ctx: RuleContext,
    prepared: ElementwiseUnaryData,
    acc: HessianAccumulator,
) -> None:
    acc.add_self(prepared.dep)


# -------------------
# integer_pow
# -------------------


def integer_pow_dependencies(
    ctx: RuleContext,
    prepared: None,
) -> tuple[DependencySet, ...]:
    if len(ctx.input_deps) != 1 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have one input and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    return (ctx.input_deps[0],)


def integer_pow_hessian(
    ctx: RuleContext,
    prepared: None,
    acc: HessianAccumulator,
) -> None:
    n = int(ctx.eqn.params["y"])
    if n in (0, 1):
        return

    acc.add_self(ctx.input_deps[0])


LINEAR_UNARY = PrimitiveRule(
    DerivativeRule(
        prepare_elementwise_unary,
        unary_passthrough_dependencies,
        no_hessian,
    )
)
NONLINEAR_UNARY = PrimitiveRule(
    DerivativeRule(
        prepare_elementwise_unary,
        unary_passthrough_dependencies,
        nonlinear_unary_hessian,
    )
)
INTEGER_POW = PrimitiveRule(
    DerivativeRule(
        no_prepare,
        integer_pow_dependencies,
        integer_pow_hessian,
    )
)
