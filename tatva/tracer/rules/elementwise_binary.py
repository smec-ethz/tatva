from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tatva.tracer.helpers import _shape_of
from tatva.tracer.semantics import DerivativeRule, PrimitiveRule, no_hessian
from tatva.tracer.dependencies import DependencySet, HessianAccumulator

if TYPE_CHECKING:
    from tatva.tracer.semantics import RuleContext


# ------------------------------
# Prepare rules
# ------------------------------


@dataclass(frozen=True)
class ElementwiseBinaryData:
    lhs: DependencySet
    rhs: DependencySet
    output_shape: tuple[int, ...]


def prepare_elementwise_binary(ctx: RuleContext) -> ElementwiseBinaryData:
    input_deps, eqn = ctx.input_deps, ctx.eqn
    if len(input_deps) != 2 or len(eqn.outvars) != 1:
        raise ValueError(
            f"{eqn.primitive.name} must have two inputs and one output; got "
            f"{len(input_deps)} inputs and {len(eqn.outvars)} outputs"
        )

    output_shape = _shape_of(eqn.outvars[0])
    lhs, rhs = (dep.broadcast_to(output_shape) for dep in input_deps)

    return ElementwiseBinaryData(lhs, rhs, output_shape)


# ------------------------------
# Dependency propagation rules
# ------------------------------


def union_dependencies(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
) -> tuple[DependencySet, ...]:
    """Propagate support through a two-input elementwise linear primitive.

    Each output entry depends on the union of the corresponding (after JAX
    broadcasting) entries of both operands.  This is structural Jacobian
    support, so adding the sparse boolean matrices implements set union.
    """
    output_csr = (prepared.lhs.csr + prepared.rhs.csr).astype(bool).tocsr()
    output_csr.eliminate_zeros()  # todo: may not be necessary (didn't have that before)

    return (DependencySet(output_csr, prepared.output_shape),)


# ------------------------------
# Hessian propagation rules
# ------------------------------


def elementwise_mul_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: HessianAccumulator,
) -> None:
    acc.add_cross(prepared.lhs, prepared.rhs)


def elementwise_div_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: HessianAccumulator,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    # d²(a / b) / da db
    acc.add_cross(lhs, rhs)

    # d²(a / b) / db²
    acc.add_self(rhs)


def elementwise_pow_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: HessianAccumulator,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    acc.add_self(lhs)
    acc.add_cross(lhs, rhs)
    acc.add_self(rhs)


def elementwise_atan2_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: HessianAccumulator,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    acc.add_self(lhs)
    acc.add_cross(lhs, rhs)
    acc.add_self(rhs)


ELEMENTWISE_BINARY_BASIC = PrimitiveRule(
    DerivativeRule(
        prepare=prepare_elementwise_binary,
        dependencies=union_dependencies,
        hessian=no_hessian,
    )
)
