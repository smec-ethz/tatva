from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps

from tatva.tracer.dependencies import DependencySet
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import GatherRoute

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
