from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Primitive
from jax.scipy.linalg import lu_solve

from tatva.tracer.localize import (
    LocalGatherRoute,
    LocalScatterRoute,
)

if TYPE_CHECKING:
    from tatva.tracer.local_plan import LocalEqnPlan


@dataclass(frozen=True)
class LoweringContext:
    plan: LocalEqnPlan
    inputs: tuple[Any | None, ...]


type LoweringRule = Callable[
    [LoweringContext],
    tuple[Any | None, ...],
]


def _input(
    ctx: LoweringContext,
    index: int,
):
    value = ctx.inputs[index]

    if value is None:
        raise RuntimeError(f"{ctx.plan.primitive_name}: runtime input {index} is dead")

    return value


def _single_output_layout(
    ctx: LoweringContext,
):
    if len(ctx.plan.output_layouts) != 1:
        raise RuntimeError(f"{ctx.plan.primitive_name} expected one output")

    layout = ctx.plan.output_layouts[0]

    if layout is None:
        raise RuntimeError(f"{ctx.plan.primitive_name} has no live output layout")

    return layout


def lower_default(
    ctx: LoweringContext,
) -> tuple[Any | None, ...]:
    eqn = ctx.plan.eqn

    if any(value is None for value in ctx.inputs):
        dead = [i for i, value in enumerate(ctx.inputs) if value is None]

        raise RuntimeError(
            f"default lowering for "
            f"{eqn.primitive.name!r} requires "
            f"dead inputs {dead}; add a specialized lowering"
        )

    result = eqn.primitive.bind(
        *ctx.inputs,
        **eqn.params,
    )

    if eqn.primitive.multiple_results:
        return tuple(result)

    return (result,)


def lower_broadcast_in_dim(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    x = _input(ctx, 0)
    output = _single_output_layout(ctx)
    dimensions = tuple(int(x) for x in ctx.plan.eqn.params["broadcast_dimensions"])
    result = lax.broadcast_in_dim(
        x,
        output.local_shape,
        dimensions,
    )

    return (result,)


def lower_reshape(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    x = _input(ctx, 0)
    output = _single_output_layout(ctx)

    if x.size != output.local_size:
        raise RuntimeError(
            f"localized reshape changes scalar count: {x.shape} -> {output.local_shape}"
        )

    dimensions = ctx.plan.eqn.params.get("dimensions")
    result = lax.reshape(
        x,
        output.local_shape,
        dimensions=dimensions,
    )

    return (result,)


def lower_iota(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    output = _single_output_layout(ctx)

    eqn = ctx.plan.eqn
    dimension = int(eqn.params["dimension"])
    dtype = eqn.params["dtype"]

    global_indices = output.global_axis_indices(dimension)
    values = jnp.asarray(global_indices, dtype=dtype)

    shape = [1] * len(output.local_shape)
    shape[dimension] = output.local_shape[dimension]
    values = jnp.reshape(values, tuple(shape))

    return (
        jnp.broadcast_to(
            values,
            output.local_shape,
        ),
    )


def lower_concatenate(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    output = _single_output_layout(ctx)
    dimension = int(ctx.plan.eqn.params["dimension"])
    values = tuple(value for value in ctx.inputs if value is not None)

    if not values:
        raise RuntimeError("live concatenate has no live operands")

    if len(values) == 1:
        result = values[0]
    else:
        result = jnp.concatenate(values, axis=dimension)

    if tuple(result.shape) != output.local_shape:
        raise RuntimeError(
            "localized concatenate shape mismatch: "
            f"{result.shape} != "
            f"{output.local_shape}"
        )

    return (result,)


def lower_gather(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    operand = _input(ctx, 0)
    route_plan = ctx.plan.route

    if route_plan is None or not isinstance(route_plan.local, LocalGatherRoute):
        raise RuntimeError("gather requires LocalGatherRoute")

    route = route_plan.local
    flat_operand = jnp.ravel(operand)
    rows = route.source_rows
    invalid = rows < 0

    if not invalid.any():
        result = flat_operand[jnp.asarray(rows)]

    else:
        safe_rows = jnp.asarray(rows.copy())
        safe_rows = jnp.maximum(safe_rows, 0)

        result = flat_operand[safe_rows]
        fill_value = ctx.plan.eqn.params.get("fill_value")

        if fill_value is None:
            raise NotImplementedError(
                "localized gather with invalid rows "
                "and implicit fill_value is not yet supported"
            )

        result = jnp.where(
            jnp.asarray(~invalid),
            result,
            jnp.asarray(fill_value, dtype=operand.dtype),
        )

    return (jnp.reshape(result, route.output_shape),)


def lower_scatter_set(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    route_plan = ctx.plan.route
    if route_plan is None or not isinstance(route_plan.local, LocalScatterRoute):
        raise RuntimeError("scatter requires LocalScatterRoute")

    route = route_plan.local

    operand = ctx.inputs[0]
    updates = ctx.inputs[2]

    if operand is None and updates is None:
        raise RuntimeError("localized scatter has no runtime data")

    dtype = operand.dtype if operand is not None else updates.dtype  # ty: ignore[unresolved-attribute]
    output = jnp.zeros((int(prod(route.output_shape)),), dtype=dtype)

    # Preserve surviving old operand values.
    if operand is not None:
        flat_operand = jnp.ravel(operand)

        output = output.at[jnp.asarray(route.operand_output_rows)].set(
            flat_operand[jnp.asarray(route.operand_rows)]
        )

    # Apply surviving updates.
    if updates is not None:
        flat_updates = jnp.ravel(updates)

        if np.unique(route.target_rows).size != route.target_rows.size:
            raise NotImplementedError(
                "localized scatter-set with duplicate targets is not yet supported"
            )

        output = output.at[jnp.asarray(route.target_rows)].set(
            flat_updates[jnp.asarray(route.update_rows)]
        )

    return (jnp.reshape(output, route.output_shape),)


def lower_custom_linear_solve(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    eqn = ctx.plan.eqn
    lengths = eqn.params["const_lengths"]

    n_matvec = int(lengths.matvec)
    n_vecmat = int(lengths.vecmat)
    n_solve = int(lengths.solve)
    n_transpose = int(lengths.transpose_solve)

    if n_solve != 2:
        raise NotImplementedError(
            "only LU-backed custom_linear_solve with two solve constants is supported"
        )

    solve_start = n_matvec + n_vecmat
    rhs_start = solve_start + n_solve + n_transpose

    if len(ctx.inputs) - rhs_start != 1:
        raise NotImplementedError(
            "custom_linear_solve with PyTree/multiple RHS leaves is not yet supported"
        )

    lu = _input(ctx, solve_start)
    pivots = _input(ctx, solve_start + 1)
    rhs = _input(ctx, rhs_start)
    result = lu_solve((lu, pivots), rhs)

    return (result,)


LOWERINGS: dict[
    Primitive,
    LoweringRule,
] = {}


def register(
    primitive: Primitive,
    rule: LoweringRule,
) -> None:
    existing = LOWERINGS.get(primitive)

    if existing is not None and existing is not rule:
        raise ValueError(f"lowering already registered for {primitive.name!r}")

    LOWERINGS[primitive] = rule


def register_default_lowerings() -> None:
    # Lowering rules
    register(lax.broadcast_in_dim_p, lower_broadcast_in_dim)
    register(lax.reshape_p, lower_reshape)
    register(lax.iota_p, lower_iota)
    register(lax.concatenate_p, lower_concatenate)

    register(lax.gather_p, lower_gather)
    register(lax.scatter_p, lower_scatter_set)

    register(lax.linear_solve_p, lower_custom_linear_solve)


# Register default lowerings at module import time
register_default_lowerings()
