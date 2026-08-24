from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.core.route_fragments import (
    GatherEnvelopeFragment,
    GatherRouteFragment,
    ScatterRouteFragment,
    resolve_scatter_route_fragment,
)
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
    RoutingSemantics,
    no_hessian,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand, merge_demands
from tatva.tracer.local.localize import (
    LocalDynamicGatherRoute,
    LocalGatherRoute,
    LocalScatterRoute,
    localize_gather_route,
    localize_scatter_route,
)
from tatva.tracer.program.dependencies import DependencySet, InteractionGraph
from tatva.tracer.rules import tagged

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext

type GatherDerivativeRoute = (
    GatherRoute | GatherRouteFragment | GatherEnvelopeFragment | None
)


def prepare_gather(ctx: RuleContext) -> GatherDerivativeRoute:
    if len(ctx.input_deps) != 2 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have two inputs and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    if ctx.route is not None and not isinstance(
        ctx.route, (GatherRoute, GatherRouteFragment, GatherEnvelopeFragment)
    ):
        raise TypeError(f"invalid gather route for equation {ctx.eqn}")

    return ctx.route


def gather_dependencies(
    ctx: RuleContext,
    prepared: GatherDerivativeRoute,
) -> tuple[DependencySet, ...]:
    source = ctx.input_deps[0]
    output_shape = _shape_of(ctx.eqn.outvars[0])
    n_output = int(np.prod(output_shape))
    n_source = source.csr.shape[0]

    if prepared is None:
        union = source.total_union().csr
        output_csr = (
            sps.csr_matrix((0, ctx.n_dofs), dtype=bool)
            if n_output == 0
            else sps.vstack([union] * n_output, format="csr")
        )
        return (DependencySet(output_csr, output_shape),)

    if isinstance(prepared, GatherRoute):
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
        return (
            DependencySet((selection @ source.csr).astype(bool).tocsr(), output_shape),
        )

    if isinstance(prepared, GatherRouteFragment):
        if prepared.output_rows.shape != prepared.source_rows.shape:
            raise ValueError(
                f"gather fragment has {prepared.output_rows.size} output rows, "
                f"{prepared.source_rows.size} source rows"
            )
        valid = (
            (prepared.source_rows >= 0)
            & (prepared.output_rows >= 0)
            & (prepared.output_rows < n_output)
        )
        selection = sps.csr_matrix(
            (
                np.ones(np.count_nonzero(valid), dtype=bool),
                (prepared.output_rows[valid], prepared.source_rows[valid]),
            ),
            shape=(n_output, n_source),
            dtype=bool,
        )
        return (
            DependencySet((selection @ source.csr).astype(bool).tocsr(), output_shape),
        )

    if isinstance(prepared, GatherEnvelopeFragment):
        if len(prepared.source_demands) != prepared.output_rows.size:
            raise ValueError("gather envelope source demands count mismatch")

        sel_rows: list[NDArray[np.int64]] = []
        sel_cols: list[NDArray[np.int64]] = []

        for out_row, source_demand in zip(
            prepared.output_rows, prepared.source_demands
        ):
            if source_demand is None or out_row < 0 or out_row >= n_output:
                continue
            src_rows = source_demand.rows()
            if src_rows.size > 0:
                sel_rows.append(np.full(src_rows.size, out_row, dtype=np.int64))
                sel_cols.append(src_rows)

        if sel_rows:
            all_rows = np.concatenate(sel_rows)
            all_cols = np.concatenate(sel_cols)
            selection = sps.csr_matrix(
                (
                    np.ones(all_rows.size, dtype=bool),
                    (all_rows, all_cols),
                ),
                shape=(n_output, n_source),
                dtype=bool,
            )
        else:
            selection = sps.csr_matrix((n_output, n_source), dtype=bool)

        return (
            DependencySet((selection @ source.csr).astype(bool).tocsr(), output_shape),
        )

    raise TypeError(f"unsupported gather route type: {type(prepared)!r}")


def _envelope_positions(
    route: GatherEnvelopeFragment, output_rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    positions = np.searchsorted(route.output_rows, output_rows)
    if np.any(positions >= route.output_rows.size) or np.any(
        route.output_rows[positions] != output_rows
    ):
        raise ValueError("gather envelope does not cover demanded rows")
    return positions


def gather_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)
    route = ctx.route
    output_rows = output.rows()
    result: list[Demand] = [None] * len(ctx.eqn.invars)

    if isinstance(route, GatherEnvelopeFragment):
        positions = _envelope_positions(route, output_rows)
        operand: Demand = None
        for position in positions:
            operand = merge_demands(operand, route.source_demands[position])
        result[0] = operand
        index_rows = route.index_rows[positions].ravel()
        result[1] = TensorDemand.from_rows_hull(
            _shape_of(ctx.eqn.invars[1]), index_rows
        )
        return tuple(result)

    if isinstance(route, GatherRouteFragment):
        positions = np.searchsorted(route.output_rows, output_rows)
        if np.any(positions >= route.output_rows.size) or np.any(
            route.output_rows[positions] != output_rows
        ):
            raise ValueError("gather fragment does not cover demanded rows")
        source_rows = route.source_rows[positions]
    elif isinstance(route, GatherRoute):
        source_rows = route.source_rows[output_rows]
    else:
        raise TypeError("gather demand requires a gather route")

    result[0] = TensorDemand.from_rows_hull(
        _shape_of(ctx.eqn.invars[0]), source_rows[source_rows >= 0]
    )
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
    ctx: RuleContext, prepared: PreparedScatter, acc: InteractionGraph
) -> None:
    valid = prepared.target_rows >= 0

    targets = prepared.target_rows[valid]
    acc.add_paired_cross(
        prepared.base,
        targets,
        prepared.updates,
        np.flatnonzero(valid),
    )


def _scatter_demand(
    ctx: DemandContext,
    *,
    needs_operand_at_updates: bool,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return tuple(None for _ in ctx.eqn.invars)

    route = ctx.route

    output_shape = _shape_of(ctx.eqn.outvars[0])
    n_output = int(math.prod(output_shape))
    output_rows = output.rows()

    wanted = np.zeros(
        n_output,
        dtype=bool,
    )
    wanted[output_rows] = True

    if isinstance(route, ScatterRouteFragment):
        targets = route.target_rows
        relation_update_rows = route.update_rows

    elif isinstance(route, ScatterRoute):
        targets = route.target_rows
        relation_update_rows = np.arange(targets.size, dtype=np.int64)

    else:
        raise TypeError("scatter demand requires a scatter route")

    valid = (targets >= 0) & (targets < n_output)

    relation_rows = np.flatnonzero(
        valid & wanted[np.clip(targets, 0, max(n_output - 1, 0))]
    )
    update_rows = relation_update_rows[relation_rows]

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
) -> LocalGatherRoute | LocalDynamicGatherRoute:
    route = ctx.route
    if not isinstance(
        route, (GatherRoute, GatherRouteFragment, GatherEnvelopeFragment)
    ):
        raise TypeError(
            f"{ctx.eqn.primitive.name} route localization requires a gather route"
        )
    if len(ctx.input_layouts) < 2:
        raise RuntimeError("gather expected operand and indices")
    operand_layout = ctx.input_layouts[0]
    if operand_layout is None:
        raise RuntimeError("live gather output has no live operand layout")
    if len(ctx.output_layouts) != 1 or ctx.output_layouts[0] is None:
        raise RuntimeError("attempted to localize dead gather")
    output_layout = ctx.output_layouts[0]
    assert output_layout is not None

    if isinstance(route, GatherEnvelopeFragment):
        index_layout = ctx.input_layouts[1]
        if index_layout is None:
            raise RuntimeError("runtime-local gather has no live index layout")
        return LocalDynamicGatherRoute.from_fragment(
            ctx.eqn,
            route,
            operand_layout=operand_layout,
            index_layout=index_layout,
            output_layout=output_layout,
        )
    return localize_gather_route(
        route, operand_layout=operand_layout, output_layout=output_layout
    )


def localize_scatter(
    ctx: RouteLocalizationContext,
) -> LocalScatterRoute:
    route = ctx.route

    if not isinstance(route, (ScatterRoute, ScatterRouteFragment)):
        raise TypeError(
            f"{ctx.eqn.primitive.name} route localization requires a scatter route"
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
        interactions=no_hessian,
    ),
    routing=RoutingSemantics(
        inputs=scatter_concrete_inputs,
        resolve=resolve_scatter_route,
        fragment=resolve_scatter_route_fragment,
    ),
    demand=scatter_set_demand,
    tagged_demand=tagged.scatter_set,
    localization=SCATTER_LOCALIZATION,
)

SCATTER_ACCUMULATE = OperationSemantics(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        interactions=no_hessian,
    ),
    routing=RoutingSemantics(
        inputs=scatter_concrete_inputs,
        resolve=resolve_scatter_route,
        fragment=resolve_scatter_route_fragment,
    ),
    demand=scatter_accumulate_demand,
    tagged_demand=tagged.scatter_accumulate,
    localization=SCATTER_LOCALIZATION,
)
SCATTER_MUL = OperationSemantics(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        interactions=scatter_mul_hessian,
    ),
    routing=RoutingSemantics(
        inputs=scatter_concrete_inputs,
        resolve=resolve_scatter_route,
        fragment=resolve_scatter_route_fragment,
    ),
    demand=scatter_accumulate_demand,
    tagged_demand=tagged.scatter_accumulate,
    localization=SCATTER_LOCALIZATION,
)
