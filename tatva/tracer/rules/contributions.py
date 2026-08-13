import numpy as np
from jax.extend.core import Literal, Var

from tatva.tracer.semantics import (
    ContributionCoefficient,
    ContributionContext,
    ContributionDecision,
    ContributionInput,
    ContributionMode,
    ContributionRule,
)


def _unsupported(reason: str) -> ContributionDecision:
    return ContributionDecision(unsupported_reason=reason)


def _require_arity(
    ctx: ContributionContext, expected: int
) -> ContributionDecision | None:
    eqn = ctx.resolved.plan.eqn
    actual = len(eqn.invars)
    if actual == expected:
        return None

    return ContributionDecision(
        invalid_reason=(
            f"{eqn.primitive.name} expected {expected} inputs, got {actual}"
        )
    )


def additive_binary(
    lhs_factor: ContributionCoefficient = 1,
    rhs_factor: ContributionCoefficient = 1,
) -> ContributionRule:
    def rule(ctx: ContributionContext) -> ContributionDecision:
        invalid = _require_arity(ctx, 2)
        if invalid is not None:
            return invalid

        return ContributionDecision(
            inputs=(
                ContributionInput(
                    input_index=0,
                    coefficient=ctx.coefficient * lhs_factor,
                    mode=ctx.mode,
                ),
                ContributionInput(
                    input_index=1,
                    coefficient=ctx.coefficient * rhs_factor,
                    mode=ctx.mode,
                ),
            )
        )

    return rule


def additive_unary(
    factor: ContributionCoefficient = 1,
) -> ContributionRule:
    def rule(ctx: ContributionContext) -> ContributionDecision:
        invalid = _require_arity(ctx, 1)
        if invalid is not None:
            return invalid

        return ContributionDecision(
            inputs=(
                ContributionInput(
                    input_index=0,
                    coefficient=ctx.coefficient * factor,
                    mode=ctx.mode,
                ),
            )
        )

    return rule


transparent_unary = additive_unary()
negative_unary = additive_unary(-1)

additive_add = additive_binary(1, 1)
additive_sub = additive_binary(1, -1)


def reduce_sum(
    ctx: ContributionContext,
) -> ContributionDecision:
    invalid = _require_arity(ctx, 1)
    if invalid is not None:
        return invalid

    if ctx.mode is ContributionMode.DOMAIN:
        return ContributionDecision(root=True)

    return ContributionDecision(
        inputs=(
            ContributionInput(
                input_index=0,
                coefficient=ctx.coefficient,
                mode=ContributionMode.DOMAIN,
            ),
        )
    )


def _concrete_scalar(
    ctx: ContributionContext,
    input_index: int,
) -> ContributionCoefficient | None:
    atom = ctx.resolved.plan.eqn.invars[input_index]

    if isinstance(atom, Literal):
        value = atom.val
    elif isinstance(atom, Var):
        value = ctx.instance.concrete.get(atom)
        if value is None:
            return None
    else:
        return None

    array = np.asarray(value)
    if array.shape != ():
        return None

    return array.item()


def scalar_multiply(ctx: ContributionContext) -> ContributionDecision:
    """Trace ``mul`` only while still in scalar-additive mode."""

    if ctx.mode is ContributionMode.DOMAIN:
        return ContributionDecision(root=True)

    invalid = _require_arity(ctx, 2)
    if invalid is not None:
        return invalid

    lhs_scalar = _concrete_scalar(ctx, 0)
    rhs_scalar = _concrete_scalar(ctx, 1)

    if lhs_scalar is not None and rhs_scalar is None:
        return ContributionDecision(
            inputs=(
                ContributionInput(
                    input_index=1,
                    coefficient=ctx.coefficient * lhs_scalar,
                    mode=ContributionMode.SCALAR,
                ),
            )
        )

    if rhs_scalar is not None and lhs_scalar is None:
        return ContributionDecision(
            inputs=(
                ContributionInput(
                    input_index=0,
                    coefficient=ctx.coefficient * rhs_scalar,
                    mode=ContributionMode.SCALAR,
                ),
            )
        )

    if lhs_scalar is not None and rhs_scalar is not None:
        # Purely concrete term: no partitionable contribution domain.
        return ContributionDecision()

    return _unsupported(
        "contribution scalar multiplication requires exactly one concrete operand"
    )


def scalar_divide(ctx: ContributionContext) -> ContributionDecision:
    """Trace ``div`` by a concrete, non-zero scalar denominator."""

    if ctx.mode is ContributionMode.DOMAIN:
        return ContributionDecision(root=True)

    invalid = _require_arity(ctx, 2)
    if invalid is not None:
        return invalid

    denominator = _concrete_scalar(ctx, 1)
    if denominator is None:
        return _unsupported(
            "contribution scalar division requires a concrete denominator"
        )
    if denominator == 0:
        return _unsupported(
            "contribution scalar division requires a nonzero denominator"
        )

    return ContributionDecision(
        inputs=(
            ContributionInput(
                input_index=0,
                coefficient=ctx.coefficient / denominator,
                mode=ContributionMode.SCALAR,
            ),
        )
    )
