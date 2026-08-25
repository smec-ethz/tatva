from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jax import lax

from tatva.tracer.local.localize import (
    LocalDynamicGatherRoute,
    LocalDynamicSliceRoute,
    LocalGatherRoute,
    LocalScatterRoute,
    LocalSelectNRoute,
    localize_unary_source_rows,
)
from tatva.tracer.rules.structural import slice_row_map, tile_row_map

if TYPE_CHECKING:
    from tatva.tracer.local.plan import LocalEqnPlan


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


def lower_bind(
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

    try:
        result = eqn.primitive.bind(
            *ctx.inputs,
            **eqn.params,
        )
    except Exception as exc:
        input_shapes = tuple(
            None if value is None else tuple(value.shape) for value in ctx.inputs
        )
        raise RuntimeError(
            f"default local lowering failed:\n"
            f"  eqn={ctx.plan.index}\n"
            f"  primitive={eqn.primitive.name}\n"
            f"  input_shapes={input_shapes}\n"
            f"  params={eqn.params}"
        ) from exc

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


def lower_empty(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    output = _single_output_layout(ctx)
    dtype = ctx.plan.eqn.params["dtype"]
    return (jnp.empty(output.local_shape, dtype=dtype),)


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


def lower_gather(ctx: LoweringContext) -> tuple[Any, ...]:
    operand = _input(ctx, 0)
    route_plan = ctx.plan.route
    if route_plan is None:
        raise RuntimeError("gather has no localized route decision")

    if isinstance(route_plan.local, LocalDynamicGatherRoute):
        indices = _input(ctx, 1)
        route = route_plan.local
        for rebase in route.rebases:
            values = indices[..., rebase.component]
            global_indices = jnp.asarray(
                rebase.global_indices,
                dtype=values.dtype,
            )
            positions = jnp.searchsorted(global_indices, values)
            safe = jnp.minimum(positions, global_indices.size - 1)
            matched = (positions < global_indices.size) & (
                global_indices[safe] == values
            )
            # An unmatched global coordinate is outside the compact operand.
            # Keep it out of bounds rather than accidentally aliasing a local row.
            local = jnp.where(matched, positions, global_indices.size).astype(
                values.dtype
            )
            component = rebase.component
            indices = jnp.concatenate(
                (
                    indices[..., :component],
                    local[..., None],
                    indices[..., component + 1 :],
                ),
                axis=-1,
            )
        params = dict(ctx.plan.eqn.params)
        params["slice_sizes"] = route.slice_sizes
        result = ctx.plan.eqn.primitive.bind(operand, indices, **params)
        if tuple(result.shape) != route.output_shape:
            raise RuntimeError(
                "localized runtime gather produced shape "
                f"{tuple(result.shape)}, expected {route.output_shape}"
            )
        return (result,)

    if not isinstance(route_plan.local, LocalGatherRoute):
        raise RuntimeError("gather requires a localized gather route")
    route = route_plan.local
    flat_operand = jnp.ravel(operand)
    rows = route.source_rows
    invalid = rows < 0
    if not invalid.any():
        result = flat_operand[jnp.asarray(rows)]
    else:
        safe_rows = jnp.maximum(jnp.asarray(rows.copy()), 0)
        result = flat_operand[safe_rows]
        fill_value = ctx.plan.eqn.params.get("fill_value")
        if fill_value is None:
            raise NotImplementedError(
                "localized gather with invalid rows and implicit fill_value is not yet supported"
            )
        result = jnp.where(
            jnp.asarray(~invalid), result, jnp.asarray(fill_value, dtype=operand.dtype)
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

        if np.unique(route.operand_output_rows).size != route.operand_output_rows.size:
            raise RuntimeError(
                "localized scatter operand projection contains duplicate output rows"
            )
        output = output.at[jnp.asarray(route.operand_output_rows)].set(
            flat_operand[jnp.asarray(route.operand_rows)],
            unique_indices=True,
        )

    # Apply surviving updates.
    if updates is not None:
        flat_updates = jnp.ravel(updates)

        if np.unique(route.target_rows).size != route.target_rows.size:
            raise NotImplementedError(
                "localized scatter-set with duplicate targets is not yet supported"
            )

        output = output.at[jnp.asarray(route.target_rows)].set(
            flat_updates[jnp.asarray(route.update_rows)],
            unique_indices=True,
        )

    return (jnp.reshape(output, route.output_shape),)


def lower_dynamic_slice(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    operand = _input(ctx, 0)
    route_plan = ctx.plan.route

    if route_plan is None or not isinstance(route_plan.local, LocalDynamicSliceRoute):
        raise RuntimeError("dynamic_slice requires LocalDynamicSliceRoute")

    route = route_plan.local
    flat_operand = jnp.ravel(operand)
    result = flat_operand[jnp.asarray(route.source_rows)]

    return (jnp.reshape(result, route.output_shape),)


def lower_slice(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    operand = _input(ctx, 0)
    input_layout = ctx.plan.input_layouts[0]
    output_layout = ctx.plan.output_layouts[0]

    if input_layout is None:
        raise RuntimeError("live slice has no operand layout")

    if output_layout is None:
        raise RuntimeError("live slice has no output layout")

    global_map = slice_row_map(ctx.plan.eqn)
    source_rows = localize_unary_source_rows(
        global_map.source_rows,
        operand_layout=input_layout,
        output_layout=output_layout,
        allow_invalid=False,
    )

    result = jnp.ravel(operand)[jnp.asarray(source_rows)]
    return (jnp.reshape(result, output_layout.local_shape),)


def lower_tile(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    operand = _input(ctx, 0)
    input_layout = ctx.plan.input_layouts[0]
    output_layout = ctx.plan.output_layouts[0]

    if input_layout is None:
        raise RuntimeError("live tile has no operand layout")
    if output_layout is None:
        raise RuntimeError("live tile has no output layout")

    global_map = tile_row_map(ctx.plan.eqn)
    source_rows = localize_unary_source_rows(
        global_map.source_rows,
        operand_layout=input_layout,
        output_layout=output_layout,
        allow_invalid=False,
    )
    result = jnp.ravel(operand)[jnp.asarray(source_rows)]
    return (jnp.reshape(result, output_layout.local_shape),)


def lower_select_n(
    ctx: LoweringContext,
) -> tuple[Any, ...]:
    route_plan = ctx.plan.route
    if route_plan is None or not isinstance(route_plan.local, LocalSelectNRoute):
        # No concrete selector was available while planning.  All operands
        # were kept by the dynamic demand rule, so bind the rank-local JAX
        # primitive directly.  Broadcasting must be entirely local: silently
        # relying on global layouts here would produce incorrect rank values.
        output_layout = _single_output_layout(ctx)
        output_shape = output_layout.local_shape
        values = [_input(ctx, index) for index in range(len(ctx.inputs))]
        for index, value in enumerate(values):
            try:
                np.broadcast_shapes(tuple(value.shape), output_shape)
            except ValueError as exc:
                raise NotImplementedError(
                    f"select_n dynamic selector operand {index} is not locally "
                    f"broadcast-compatible with output {output_shape}; got {value.shape}"
                ) from exc
        return (lax.select_n(values[0], *values[1:]),)

    route = route_plan.local

    # Selector ctx.inputs[0] is deliberately dead.
    dtype = None

    for case_index, case_route in enumerate(route.cases):
        if case_route.output_rows.size == 0:
            continue

        value = ctx.inputs[case_index + 1]
        if value is None:
            raise RuntimeError(f"live select_n case {case_index} is unavailable")

        dtype = value.dtype
        break

    if dtype is None:
        raise RuntimeError("select_n has no live case")

    output = jnp.empty((int(prod(route.output_shape)),), dtype=dtype)

    for case_index, case_route in enumerate(route.cases):
        if case_route.output_rows.size == 0:
            continue

        value = ctx.inputs[case_index + 1]
        assert value is not None
        source = jnp.ravel(value)[jnp.asarray(case_route.source_rows)]
        rows = np.asarray(case_route.output_rows, dtype=np.int64)
        if np.unique(rows).size != rows.size:
            raise RuntimeError(
                "localized select_n route contains duplicate output rows"
            )
        output = output.at[jnp.asarray(rows)].set(
            source,
            unique_indices=True,
        )

    return (jnp.reshape(output, route.output_shape),)


def lower_sort(
    ctx: LoweringContext,
) -> tuple[Any | None, ...]:
    """Sort complete local axis slices, then project each live result."""
    operands = []
    source_layout = None
    dimension = int(ctx.plan.eqn.params["dimension"])

    for index, (value, layout) in enumerate(
        zip(ctx.inputs, ctx.plan.input_layouts, strict=True)
    ):
        if value is None:
            raise RuntimeError(f"sort: runtime input {index} is dead")

        if layout is None:
            raise RuntimeError(f"sort: runtime operand {index} has no local layout")

        if layout.local_shape[dimension] != layout.global_shape[dimension]:
            raise RuntimeError(
                f"sort: operand {index} must contain the complete sorted axis"
            )

        if source_layout is None:
            source_layout = layout
        elif source_layout.global_shape != layout.global_shape or any(
            not np.array_equal(
                source_layout.global_axis_indices(axis),
                layout.global_axis_indices(axis),
            )
            for axis in range(layout.ndim)
        ):
            raise RuntimeError("sort: all operands must use the same local layout")

        operands.append(value)

    if source_layout is None:
        raise RuntimeError("sort: no live operand layout")

    result = lax.sort(
        tuple(operands),
        dimension=dimension,
        is_stable=ctx.plan.eqn.params["is_stable"],
        num_keys=ctx.plan.eqn.params["num_keys"],
    )

    # lax.sort returns an array for its unary form and a tuple for co-sorts.
    sorted_values = result if isinstance(result, tuple) else (result,)
    outputs: list[Any | None] = []

    for value, layout in zip(sorted_values, ctx.plan.output_layouts, strict=True):
        if layout is None:
            outputs.append(None)
            continue

        target_local_rows = np.arange(layout.local_size, dtype=np.int64)
        target_global_rows = layout.local_rows_to_global_rows(target_local_rows)
        source_local_rows = source_layout.global_rows_to_local_rows(target_global_rows)
        outputs.append(
            jnp.reshape(
                jnp.ravel(value)[jnp.asarray(source_local_rows)],
                layout.local_shape,
            )
        )

    return tuple(outputs)
