from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jax.extend.core import Literal

from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    OperationSemantics,
    no_hessian,
    no_prepare,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand, _FullAxis
from tatva.tracer.program.dependencies import DependencySet, InteractionGraph
from tatva.tracer.rules import concrete, tagged

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext


# --------------------------------
# Elementwise unary rules
# --------------------------------


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
    acc: InteractionGraph,
) -> None:
    acc.add_self(prepared.dep)


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
    acc: InteractionGraph,
) -> None:
    n = int(ctx.eqn.params["y"])
    if n in (0, 1):
        return

    acc.add_self(ctx.input_deps[0])


# ------------------------------
# Elementwise binary rules
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


def elementwise_mul_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: InteractionGraph,
) -> None:
    acc.add_cross(prepared.lhs, prepared.rhs)


def elementwise_div_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: InteractionGraph,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    # d²(a / b) / da db
    acc.add_cross(lhs, rhs)

    # d²(a / b) / db²
    acc.add_self(rhs)


def elementwise_pow_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: InteractionGraph,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    acc.add_self(lhs)
    acc.add_cross(lhs, rhs)
    acc.add_self(rhs)


def elementwise_atan2_hessian(
    ctx: RuleContext,
    prepared: ElementwiseBinaryData,
    acc: InteractionGraph,
) -> None:
    lhs, rhs = prepared.lhs, prepared.rhs
    acc.add_self(lhs)
    acc.add_cross(lhs, rhs)
    acc.add_self(rhs)


# ------------------------
# Demand propagation rules
# ------------------------


def inverse_elementwise_broadcast(
    demand: Demand,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> Demand:
    if demand is None:
        return None

    if input_shape == output_shape:
        return demand

    if len(input_shape) > len(output_shape):
        raise ValueError(f"cannot inverse-broadcast {output_shape} to {input_shape}")

    if not input_shape:
        return TensorDemand.full(())

    output_axes = demand.axes
    offset = len(output_shape) - len(input_shape)
    input_axes = []

    for input_axis, input_extent in enumerate(input_shape):
        output_axis = offset + input_axis
        output_extent = output_shape[output_axis]

        if input_extent == output_extent:
            input_axes.append(output_axes[output_axis])
        elif input_extent == 1:
            # Every demanded broadcast entry comes from x[0].
            input_axes.append(_FullAxis())
        else:
            raise ValueError(f"invalid broadcast from {input_shape} to {output_shape}")

    return TensorDemand.from_axes(
        input_shape,
        tuple(input_axes),
    )


def elementwise_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    if len(ctx.output_demands) != 1:
        raise ValueError("elementwise primitive expected one output")

    output_demand = ctx.output_demands[0]
    if output_demand is None:
        return tuple(None for _ in ctx.eqn.invars)

    output_shape = _shape_of(ctx.eqn.outvars[0])
    result: list[Demand] = []

    for atom in ctx.eqn.invars:
        if isinstance(atom, Literal):
            result.append(None)
            continue

        result.append(
            inverse_elementwise_broadcast(
                output_demand,
                input_shape=_shape_of(atom),
                output_shape=output_shape,
            )
        )

    return tuple(result)


LINEAR_UNARY = OperationSemantics(
    DerivativeRule(
        prepare_elementwise_unary,
        unary_passthrough_dependencies,
        no_hessian,
    ),
    demand=elementwise_demand,
    tagged_demand=tagged.elementwise,
    regional_concrete=concrete.regional_bind(elementwise_demand),
)
NONLINEAR_UNARY = OperationSemantics(
    DerivativeRule(
        prepare_elementwise_unary,
        unary_passthrough_dependencies,
        nonlinear_unary_hessian,
    ),
    demand=elementwise_demand,
    tagged_demand=tagged.elementwise,
    regional_concrete=concrete.regional_bind(elementwise_demand),
)
INTEGER_POW = OperationSemantics(
    DerivativeRule(
        no_prepare,
        integer_pow_dependencies,
        integer_pow_hessian,
    ),
    demand=elementwise_demand,
    tagged_demand=tagged.elementwise,
    regional_concrete=concrete.regional_bind(elementwise_demand),
)
ELEMENTWISE_BINARY_BASIC = OperationSemantics(
    DerivativeRule(
        prepare=prepare_elementwise_binary,
        dependencies=union_dependencies,
        interactions=no_hessian,
    ),
    demand=elementwise_demand,
    tagged_demand=tagged.elementwise,
    regional_concrete=concrete.regional_bind(elementwise_demand),
)


def prepare_elementwise_nary(ctx: RuleContext) -> tuple[DependencySet, ...]:
    output_shape = _shape_of(ctx.eqn.outvars[0])
    return tuple(
        dep.reshape(*output_shape) if dep.shape != output_shape else dep
        for dep in ctx.input_deps
    )


def nary_union_dependencies(
    ctx: RuleContext,
    prepared: tuple[DependencySet, ...],
) -> tuple[DependencySet, ...]:
    if not prepared:
        return (DependencySet.empty(_shape_of(ctx.eqn.outvars[0]), ctx.n_dofs),)
    res = prepared[0]
    for other in prepared[1:]:
        res = res | other  # ty: ignore[unsupported-operator]
    return (res,)


ELEMENTWISE_NARY_BASIC = OperationSemantics(
    DerivativeRule(
        prepare=prepare_elementwise_nary,
        dependencies=nary_union_dependencies,
        interactions=no_hessian,
    ),
    demand=elementwise_demand,
    tagged_demand=tagged.elementwise,
)
