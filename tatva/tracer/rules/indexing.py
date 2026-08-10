import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.dependencies import DependencySet
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import DynamicSliceRoute, DynamicUpdateSliceRoute, SelectNRoute
from tatva.tracer.semantics import RuleContext


def prepare_select_n(ctx: RuleContext) -> SelectNRoute:
    if not isinstance(ctx.route, SelectNRoute):
        raise ValueError(f"select_n route was not resolved for equation {ctx.eqn}")  # ruff: ignore[type-check-without-type-error]

    return ctx.route


def select_n_dependencies(
    ctx: RuleContext,
    prepared: SelectNRoute,
) -> tuple[DependencySet, ...]:
    output_shape = _shape_of(ctx.eqn.outvars[0])
    n_output = int(np.prod(output_shape))

    cases = tuple(dep.broadcast_to(output_shape) for dep in ctx.input_deps[1:])

    selected_rows: list[NDArray[np.int64]] = []
    selected_blocks: list[sps.csr_matrix] = []

    for case_index, dep in enumerate(cases):
        rows = np.flatnonzero(prepared.case_indices == case_index).astype(np.int64)

        if rows.size == 0:
            continue

        selected_rows.append(rows)
        selected_blocks.append(dep.csr[rows])

    if not selected_blocks:
        output = sps.csr_matrix(
            (n_output, ctx.n_dofs),
            dtype=bool,
        )
    else:
        rows = np.concatenate(selected_rows)
        stacked = sps.vstack(
            selected_blocks,
            format="csr",
        )
        # `rows[i]` says which output position stacked row i belongs to.
        permutation = np.argsort(rows)
        output = stacked[permutation]

    return (DependencySet(output, output_shape),)


def prepare_dynamic_slice(ctx: RuleContext) -> DynamicSliceRoute:
    if not isinstance(ctx.route, DynamicSliceRoute):
        raise ValueError(f"dynamic_slice route was not resolved for equation {ctx.eqn}")  # ruff: ignore[type-check-without-type-error]

    return ctx.route


def dynamic_slice_dependencies(
    ctx: RuleContext,
    prepared: DynamicSliceRoute,
) -> tuple[DependencySet, ...]:
    source = ctx.input_deps[0]

    return (
        DependencySet(source.csr[prepared.source_rows], _shape_of(ctx.eqn.outvars[0])),
    )


def prepare_dynamic_update_slice(ctx: RuleContext) -> DynamicUpdateSliceRoute:
    if not isinstance(ctx.route, DynamicUpdateSliceRoute):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"dynamic_update_slice route was not resolved for equation {ctx.eqn}"
        )

    return ctx.route


def dynamic_update_slice_dependencies(
    ctx: RuleContext,
    prepared: DynamicUpdateSliceRoute,
) -> tuple[DependencySet, ...]:
    if len(ctx.input_deps) < 2:
        raise ValueError("dynamic_update_slice expects operand and update")

    operand = ctx.input_deps[0]
    update = ctx.input_deps[1]

    output_shape = _shape_of(ctx.eqn.outvars[0])

    # Start from operand dependencies.
    output = operand.csr.copy().tolil()

    # Overwrite target rows with corresponding update dependencies.
    output[prepared.target_rows] = update.csr

    return (DependencySet(output.tocsr(), output_shape),)
