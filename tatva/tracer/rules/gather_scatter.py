from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import GatherRoute, ScatterRoute, Shape
from tatva.tracer.semantics import DerivativeRule, PrimitiveRule, no_hessian

if TYPE_CHECKING:
    from tatva.tracer.semantics import RuleContext


def prepare_gather(ctx: RuleContext) -> GatherRoute:
    if len(ctx.input_deps) != 2 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"{ctx.eqn.primitive.name} must have two inputs and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    if not isinstance(ctx.route, GatherRoute):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"gather route was not resolved for equation {ctx.eqn}"
        )

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


@dataclass(frozen=True)
class PreparedScatter:
    base: DependencySet
    updates: DependencySet
    target_rows: NDArray[np.int64]
    output_shape: Shape


def prepare_scatter(ctx: RuleContext) -> PreparedScatter:
    if not isinstance(ctx.route, ScatterRoute):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"scatter route was not resolved for equation {ctx.eqn}"
        )

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


SCATTER_BASIC = PrimitiveRule(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        hessian=no_hessian,
    )
)
SCATTER_MUL = PrimitiveRule(
    DerivativeRule(
        prepare=prepare_scatter,
        dependencies=scatter_accumulate_dependencies,
        hessian=scatter_mul_hessian,
    )
)
