from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import (
    DemandContext,
    DerivativeRule,
    OperationSemantics,
    RuleContext,
    no_hessian,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    AxisSubset,
    Demand,
    TensorDemand,
    _FullAxis,
    _RangeAxis,
)
from tatva.tracer.program.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.rules import tagged


@dataclass(frozen=True, slots=True)
class PreparedReduction:
    deps: DependencySet
    input_to_output: NDArray[np.int64]
    n_outputs: int
    output_shape: Shape


def reduction_geometry(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[NDArray[np.int64], int]:
    ndim = len(shape)

    axes = tuple(sorted(axis % ndim for axis in axes))

    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate reduction axes: {axes}")

    keep_axes = tuple(axis for axis in range(ndim) if axis not in axes)

    n_inputs = int(np.prod(shape, dtype=np.int64))

    if not keep_axes:
        return np.zeros(n_inputs, dtype=np.int64), 1

    keep_shape = tuple(shape[axis] for axis in keep_axes)
    n_outputs = int(np.prod(keep_shape, dtype=np.int64))

    if n_inputs == 0:
        return np.empty(0, dtype=np.int64), n_outputs

    input_rows = np.arange(n_inputs, dtype=np.int64)
    coords = np.unravel_index(input_rows, shape)

    input_to_output = np.ravel_multi_index(
        tuple(coords[axis] for axis in keep_axes), keep_shape
    ).astype(np.int64)

    return input_to_output, n_outputs


def prepare_reduction(ctx: RuleContext) -> PreparedReduction:
    deps = ctx.input_deps[0]
    axes = tuple(ctx.eqn.params["axes"])
    input_to_output, n_outputs = reduction_geometry(deps.shape, axes)

    return PreparedReduction(
        deps=deps,
        input_to_output=input_to_output,
        n_outputs=n_outputs,
        output_shape=_shape_of(ctx.eqn.outvars[0]),
    )


def reduction_dependencies(
    ctx: RuleContext,
    prepared: PreparedReduction,
) -> tuple[DependencySet, ...]:
    deps = prepared.deps

    n_inputs = deps.csr.shape[0]

    if n_inputs == 0:
        reduced = sps.csr_matrix((prepared.n_outputs, ctx.n_dofs), dtype=bool)
    else:
        input_rows = np.arange(n_inputs, dtype=np.int64)

        aggregation = sps.csr_matrix(
            (np.ones(n_inputs, dtype=np.int32), (prepared.input_to_output, input_rows)),
            shape=(prepared.n_outputs, n_inputs),
        )

        reduced = (aggregation @ deps.csr.astype(np.int32)).astype(bool).tocsr()

    return (DependencySet(csr=reduced, shape=prepared.output_shape),)


def zero_reduction_dependencies(
    ctx: RuleContext,
    prepared: PreparedReduction,
) -> tuple[DependencySet, ...]:
    return (DependencySet.empty(prepared.output_shape, ctx.n_dofs),)


def reduce_prod_hessian(
    ctx: RuleContext,
    prepared: PreparedReduction,
    acc: HessianAccumulator,
) -> None:
    deps = prepared.deps.csr
    groups = prepared.input_to_output

    if deps.shape[0] <= 1:
        return

    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]

    start = 0

    while start < len(order):
        group = sorted_groups[start]

        stop = start + 1
        while stop < len(order) and sorted_groups[stop] == group:
            stop += 1

        rows = order[start:stop]

        # Product introduces interactions only between DISTINCT
        # scalar operands in the same reduction bucket.
        for i_pos in range(len(rows)):
            lhs_row = rows[i_pos]
            lhs = deps[lhs_row : lhs_row + 1]

            if lhs.nnz == 0:
                continue

            for rhs_row in rows[i_pos + 1 :]:
                rhs = deps[rhs_row : rhs_row + 1]

                if rhs.nnz:
                    # TODO: this fails, because add_cross expects two depsets, not sparse matrices. Fix this.
                    acc.add_cross(lhs, rhs)

        start = stop


def reduce_sum_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])
    reduced_axes = {int(axis) for axis in ctx.eqn.params["axes"]}
    output_axes = iter(output.axes)
    input_axes: list[AxisSubset] = []

    for axis in range(len(input_shape)):
        if axis in reduced_axes:
            # Every entry of a reduced dimension contributes.
            input_axes.append(_FullAxis())
        else:
            input_axes.append(next(output_axes))

    return (
        TensorDemand.from_axes(
            input_shape,
            tuple(input_axes),
        ),
    )


def cumulative_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])
    if input_shape != output.shape:
        raise ValueError(
            f"{ctx.eqn.primitive.name}: cumulative input/output shape mismatch"
        )

    axis = int(ctx.eqn.params["axis"])
    if axis < 0:
        axis += len(input_shape)

    reverse = bool(ctx.eqn.params.get("reverse", False))
    selected = output.selected_indices(axis)

    if reverse:
        start = int(selected.min())
        stop = input_shape[axis]
    else:
        start = 0
        stop = int(selected.max()) + 1

    axes = list(output.axes)
    if start == 0 and stop == input_shape[axis]:
        axes[axis] = _FullAxis()
    else:
        axes[axis] = _RangeAxis(start, stop)

    return (TensorDemand.from_axes(input_shape, tuple(axes)),)


REDUCE_BASIC = OperationSemantics(
    DerivativeRule(
        prepare=prepare_reduction,
        dependencies=reduction_dependencies,
        hessian=no_hessian,
    ),
    demand=reduce_sum_demand,
    tagged_demand=tagged.reduction,
)
REDUCE_PROD = OperationSemantics(
    DerivativeRule(
        prepare=prepare_reduction,
        dependencies=reduction_dependencies,
        hessian=reduce_prod_hessian,
    ),
    tagged_demand=tagged.reduction,
)

ZERO_REDUCTION = OperationSemantics(
    DerivativeRule(
        prepare_reduction,
        zero_reduction_dependencies,
        no_hessian,
    ),
    demand=reduce_sum_demand,
    tagged_demand=tagged.reduction,
)
