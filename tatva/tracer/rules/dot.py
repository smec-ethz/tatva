from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import Shape
from tatva.tracer.semantics import RuleContext


@dataclass(frozen=True)
class DotGeneralMap:
    """Scalar-slot structure of dot_general.

    For each output row `o` and contraction slot `k`:

        lhs_rows[o, k]
        rhs_rows[o, k]

    are multiplied together in the contraction.
    """

    lhs_rows: NDArray[np.int64]
    rhs_rows: NDArray[np.int64]
    output_shape: Shape


def _dot_dimension_numbers(
    dimension_numbers: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Returns:
        lhs_contract, rhs_contract, lhs_batch, rhs_batch
    """

    if hasattr(dimension_numbers, "lhs_contracting_dimensions"):
        return (
            tuple(dimension_numbers.lhs_contracting_dimensions),
            tuple(dimension_numbers.rhs_contracting_dimensions),
            tuple(dimension_numbers.lhs_batch_dimensions),
            tuple(dimension_numbers.rhs_batch_dimensions),
        )

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    return (
        tuple(lhs_contract),
        tuple(rhs_contract),
        tuple(lhs_batch),
        tuple(rhs_batch),
    )


def _grouped_dependency_union(
    dep: DependencySet,
    rows: NDArray[np.int64],
) -> sps.csr_matrix:
    """
    `rows` has shape (n_groups, n_rows_per_group).

    Returns one dependency row per group, containing the union of all
    DependencySet rows referenced by that group.
    """

    if rows.ndim != 2:
        raise ValueError("grouped dependency rows must be rank 2")

    n_groups, group_size = rows.shape
    n_dofs = dep.csr.shape[1]

    if group_size == 0:
        return sps.csr_matrix((n_groups, n_dofs), dtype=bool)

    group_indices = np.repeat(np.arange(n_groups, dtype=np.int64), group_size)

    source_indices = rows.ravel()

    selection = sps.csr_matrix(
        (
            np.ones(source_indices.size, dtype=bool),
            (group_indices, source_indices),
        ),
        shape=(n_groups, dep.csr.shape[0]),
        dtype=bool,
    )

    result = (selection @ dep.csr).astype(bool).tocsr()
    result.eliminate_zeros()

    return result


def prepare_dot_general(ctx: RuleContext) -> DotGeneralMap:
    eqn = ctx.eqn

    if len(ctx.input_deps) != 2 or len(eqn.outvars) != 1:
        raise ValueError(
            f"dot_general expects two inputs and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(eqn.outvars)} outputs"
        )

    lhs_shape = tuple(ctx.input_deps[0].shape)
    rhs_shape = tuple(ctx.input_deps[1].shape)
    output_shape = _shape_of(eqn.outvars[0])

    (
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch,
    ) = _dot_dimension_numbers(eqn.params["dimension_numbers"])

    if len(lhs_contract) != len(rhs_contract):
        raise ValueError("lhs/rhs contracting dimensions do not match")

    if len(lhs_batch) != len(rhs_batch):
        raise ValueError("lhs/rhs batch dimensions do not match")

    # Validate paired contracting dimensions.
    for lhs_axis, rhs_axis in zip(lhs_contract, rhs_contract):
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis]:
            raise ValueError(
                "dot_general contracting dimensions have incompatible sizes: "
                f"lhs axis {lhs_axis} has {lhs_shape[lhs_axis]}, "
                f"rhs axis {rhs_axis} has {rhs_shape[rhs_axis]}"
            )

    # Validate paired batch dimensions.
    for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis]:
            raise ValueError(
                "dot_general batch dimensions have incompatible sizes: "
                f"lhs axis {lhs_axis} has {lhs_shape[lhs_axis]}, "
                f"rhs axis {rhs_axis} has {rhs_shape[rhs_axis]}"
            )

    lhs_excluded = set(lhs_contract) | set(lhs_batch)
    rhs_excluded = set(rhs_contract) | set(rhs_batch)

    lhs_free = tuple(axis for axis in range(len(lhs_shape)) if axis not in lhs_excluded)
    rhs_free = tuple(axis for axis in range(len(rhs_shape)) if axis not in rhs_excluded)

    batch_shape = tuple(lhs_shape[axis] for axis in lhs_batch)
    lhs_free_shape = tuple(lhs_shape[axis] for axis in lhs_free)
    rhs_free_shape = tuple(rhs_shape[axis] for axis in rhs_free)

    expected_output_shape = batch_shape + lhs_free_shape + rhs_free_shape

    if output_shape != expected_output_shape:
        raise ValueError(
            "unexpected dot_general output shape: "
            f"expected {expected_output_shape}, got {output_shape}"
        )

    contract_shape = tuple(lhs_shape[axis] for axis in lhs_contract)

    n_output = int(np.prod(output_shape, dtype=np.int64))
    n_contract = int(np.prod(contract_shape, dtype=np.int64))

    # Empty contraction dimension => every output is an empty sum.
    if n_contract == 0:
        return DotGeneralMap(
            lhs_rows=np.empty((n_output, 0), dtype=np.int64),
            rhs_rows=np.empty((n_output, 0), dtype=np.int64),
            output_shape=output_shape,
        )

    # np.ndindex(()) correctly gives one scalar contraction coordinate.
    contraction_coords = tuple(np.ndindex(contract_shape))

    lhs_rows = np.empty((n_output, n_contract), dtype=np.int64)
    rhs_rows = np.empty((n_output, n_contract), dtype=np.int64)

    n_batch = len(batch_shape)
    n_lhs_free = len(lhs_free_shape)

    for output_row, output_coord in enumerate(np.ndindex(output_shape)):
        batch_coord = output_coord[:n_batch]

        lhs_free_coord = output_coord[n_batch : n_batch + n_lhs_free]

        rhs_free_coord = output_coord[n_batch + n_lhs_free :]

        for contraction_index, contract_coord in enumerate(contraction_coords):
            lhs_coord = [0] * len(lhs_shape)
            rhs_coord = [0] * len(rhs_shape)

            # Batch coordinates.
            for coord, lhs_axis, rhs_axis in zip(batch_coord, lhs_batch, rhs_batch):
                lhs_coord[lhs_axis] = coord
                rhs_coord[rhs_axis] = coord

            # LHS free coordinates.
            for coord, axis in zip(lhs_free_coord, lhs_free):
                lhs_coord[axis] = coord

            # RHS free coordinates.
            for coord, axis in zip(rhs_free_coord, rhs_free):
                rhs_coord[axis] = coord

            # Matching contraction coordinates.
            for coord, lhs_axis, rhs_axis in zip(
                contract_coord, lhs_contract, rhs_contract
            ):
                lhs_coord[lhs_axis] = coord
                rhs_coord[rhs_axis] = coord

            lhs_rows[output_row, contraction_index] = np.ravel_multi_index(
                tuple(lhs_coord), lhs_shape
            )

            rhs_rows[output_row, contraction_index] = np.ravel_multi_index(
                tuple(rhs_coord), rhs_shape
            )

    return DotGeneralMap(
        lhs_rows=lhs_rows,
        rhs_rows=rhs_rows,
        output_shape=output_shape,
    )


def dot_general_dependencies(
    ctx: RuleContext,
    prepared: DotGeneralMap,
) -> tuple[DependencySet, ...]:
    lhs, rhs = ctx.input_deps

    lhs_output = _grouped_dependency_union(lhs, prepared.lhs_rows)
    rhs_output = _grouped_dependency_union(rhs, prepared.rhs_rows)

    output = (lhs_output + rhs_output).astype(bool).tocsr()
    output.eliminate_zeros()

    return (DependencySet(output, prepared.output_shape),)


def dot_general_hessian(
    ctx: RuleContext,
    prepared: DotGeneralMap,
    acc: HessianAccumulator,
) -> None:
    lhs, rhs = ctx.input_deps

    acc.add_paired_cross(lhs, prepared.lhs_rows.ravel(), rhs, prepared.rhs_rows.ravel())
