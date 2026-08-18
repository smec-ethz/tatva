from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.core.route_fragments import resolve_scatter_route_fragment
from tatva.tracer.core.routes import (
    GatherRoute,
    ScatterRoute,
    Shape,
    resolve_scatter_route,
)
from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    LocalizationSemantics,
    OperationSemantics,
    RouteLocalizationContext,
    no_hessian,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand
from tatva.tracer.local.localize import (
    LocalGatherRoute,
    LocalScatterRoute,
    localize_gather_route,
    localize_scatter_route,
)
from tatva.tracer.program.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.rules import tagged

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext


def prepare_gather(ctx: RuleContext) -> GatherRoute:
    if len(ctx.input_deps) != 2 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have two inputs and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    if not isinstance(ctx.route, GatherRoute):
        raise TypeError(f"gather route was not resolved for equation {ctx.eqn}")

    return ctx.route


def gather_dependencies(
    ctx: RuleContext,
    prepared: GatherRoute,
) -> tuple[DependencySet, ...]:
    source = ctx.input_deps[0]
    output_shape = _shape_of(ctx.eqn.outvars[0])

    n_output = int(np.prod(output_shape))

    if prepared.source_rows.shape != (n_output,):
        raise ValueError(
            f"gather route has {prepared.source_rows.size} rows, expected {n_output}"
        )

    valid = prepared.source_rows >= 0

    selection = sps.csr_matrix(
        (
            np.ones(np.count_nonzero(valid), dtype=bool),
            (np.flatnonzero(valid), prepared.source_rows[valid]),
        ),
        shape=(n_output, source.csr.shape[0]),
        dtype=bool,
    )

    output_csr = (selection @ source.csr).astype(bool).tocsr()

    return (DependencySet(output_csr, output_shape),)


def gather_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    route = ctx.route

    if not isinstance(route, GatherRoute):
        raise TypeError("gather demand requires GatherRoute")

    output_rows = output.rows()
    source_rows = route.source_rows[output_rows]
    source_rows = source_rows[source_rows >= 0]

    result: list[Demand] = [None] * len(ctx.eqn.invars)
    result[0] = TensorDemand.from_rows_hull(
        _shape_of(ctx.eqn.invars[0]),
        source_rows,
    )

    # indices are compiled into the route
    result[1] = None

    return tuple(result)


@dataclass(frozen=True)
class PreparedScatter:
    base: DependencySet
    updates: DependencySet
    target_rows: NDArray[np.int64]
    output_shape: Shape


def prepare_scatter(ctx: RuleContext) -> PreparedScatter:
    if not isinstance(ctx.route, ScatterRoute):
        raise TypeError(f"scatter route was not resolved for equation {ctx.eqn}")

    base = ctx.input_deps[0]
    updates = ctx.input_deps[2]

    return PreparedScatter(
        base, updates, ctx.route.target_rows, _shape_of(ctx.eqn.outvars[0])
    )


def scatter_concrete_inputs(eqn: JaxprEqn) -> tuple[int, ...]:
    return (1,)


def scatter_accumulate_dependencies(
    ctx: RuleContext, prepared: PreparedScatter
) -> tuple[DependencySet, ...]:
    valid = prepared.target_rows >= 0

    targets = prepared.target_rows[valid]
    update_deps = prepared.updates.csr[valid]

    coo = update_deps.tocoo()

    scattered = sps.csr_matrix(
        (
            coo.data,
            (targets[coo.row], coo.col),
        ),
        shape=prepared.base.csr.shape,
        dtype=bool,
    )

    out = (prepared.base.csr + scattered).astype(bool)

    return (DependencySet(csr=out, shape=prepared.output_shape),)


def scatter_mul_hessian(
    ctx: RuleContext, prepared: PreparedScatter, acc: HessianAccumulator
) -> None:
    valid = prepared.target_rows >= 0

    targets = prepared.target_rows[valid]
    base_deps = prepared.base.csr[targets]
    update_deps = prepared.updates.csr[valid]

    acc.add_cross(base_deps, update_deps)


def _scatter_demand(
    ctx: DemandContext,
    *,
    needs_operand_at_updates: bool,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    route = ctx.route

    if not isinstance(route, ScatterRoute):
        raise TypeError("scatter demand requires ScatterRoute")

    output_shape = _shape_of(ctx.eqn.outvars[0])
    n_output = int(math.prod(output_shape))
    output_rows = output.rows()

    wanted = np.zeros(
        n_output,
        dtype=bool,
    )
    wanted[output_rows] = True

    targets = route.target_rows
    valid = (targets >= 0) & (targets < n_output)

    update_rows = np.flatnonzero(
        valid & wanted[np.clip(targets, 0, max(n_output - 1, 0))]
    )

    if needs_operand_at_updates:
        operand_rows = output_rows
    else:
        overwritten = np.unique(targets[valid])
        operand_rows = output_rows[~np.isin(output_rows, overwritten)]

    result: list[Demand] = [None] * len(ctx.eqn.invars)
    result[0] = TensorDemand.from_rows_hull(_shape_of(ctx.eqn.invars[0]), operand_rows)
    # indices are compiled into ScatterRoute
    result[1] = None
    result[2] = TensorDemand.from_rows_hull(_shape_of(ctx.eqn.invars[2]), update_rows)

    return tuple(result)


def scatter_set_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    # Output targets replace operand entries.
    return _scatter_demand(
        ctx,
        needs_operand_at_updates=False,
    )


def scatter_accumulate_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    # add/mul/min/max combine update with existing operand.
    return _scatter_demand(
        ctx,
        needs_operand_at_updates=True,
    )


def localize_gather(
    ctx: RouteLocalizationContext,
) -> LocalGatherRoute:
    route = ctx.route

    if not isinstance(route, GatherRoute):
        raise TypeError(
            f"{ctx.eqn.primitive.name} route localization requires GatherRoute"
        )

    if not ctx.input_layouts:
        raise RuntimeError("gather has no operand")

    operand_layout = ctx.input_layouts[0]

    if operand_layout is None:
        raise RuntimeError("live gather output has no live operand layout")

    if len(ctx.output_layouts) != 1:
        raise RuntimeError("gather expected one output")

    output_layout = ctx.output_layouts[0]
    if output_layout is None:
        raise RuntimeError("attempted to localize dead gather")

    return localize_gather_route(
        route,
        operand_layout=operand_layout,
        output_layout=output_layout,
    )


def localize_scatter(
    ctx: RouteLocalizationContext,
) -> LocalScatterRoute:
    route = ctx.route

    if not isinstance(route, ScatterRoute):
        raise TypeError(
            f"{ctx.eqn.primitive.name} route localization requires ScatterRoute"
        )

    if len(ctx.input_layouts) < 3:
        raise RuntimeError("scatter expected operand, indices and updates")

    if len(ctx.output_layouts) != 1:
        raise RuntimeError("scatter expected one output")

    output_layout = ctx.output_layouts[0]
    if output_layout is None:
        raise RuntimeError("attempted to localize dead scatter")

    # input 1 is the index tensor. Its values have already been
    # compiled into ScatterRoute.
    return localize_scatter_route(
        route,
        operand_layout=ctx.input_layouts[0],
        update_layout=ctx.input_layouts[2],
        output_layout=output_layout,
    )


SCATTER_LOCALIZATION = LocalizationSemantics(
    localize_route=localize_scatter,
)


SCATTER_BASIC = OperationSemantics(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        hessian=no_hessian,
    ),
    concrete_inputs=scatter_concrete_inputs,
    route=resolve_scatter_route,
    route_fragment=resolve_scatter_route_fragment,
    demand=scatter_set_demand,
    tagged_demand=tagged.scatter_set,
    localization=SCATTER_LOCALIZATION,
)

SCATTER_ACCUMULATE = OperationSemantics(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        hessian=no_hessian,
    ),
    concrete_inputs=scatter_concrete_inputs,
    route=resolve_scatter_route,
    route_fragment=resolve_scatter_route_fragment,
    demand=scatter_accumulate_demand,
    tagged_demand=tagged.scatter_accumulate,
    localization=SCATTER_LOCALIZATION,
)
SCATTER_MUL = OperationSemantics(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        hessian=scatter_mul_hessian,
    ),
    concrete_inputs=scatter_concrete_inputs,
    route=resolve_scatter_route,
    route_fragment=resolve_scatter_route_fragment,
    demand=scatter_accumulate_demand,
    tagged_demand=tagged.scatter_accumulate,
    localization=SCATTER_LOCALIZATION,
)
