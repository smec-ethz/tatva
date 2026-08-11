from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.demand import Demand, TensorDemand, demand_rows
from tatva.tracer.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import Shape
from tatva.tracer.semantics import DemandContext, RuleContext


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


@cache
def _dot_general_map(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    dimension_numbers: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> DotGeneralMap:
    (lhs_contract, rhs_contract, lhs_batch, rhs_batch) = dimension_numbers

    lhs_excluded = set(lhs_contract) | set(lhs_batch)
    rhs_excluded = set(rhs_contract) | set(rhs_batch)

    lhs_free = tuple(axis for axis in range(len(lhs_shape)) if axis not in lhs_excluded)
    rhs_free = tuple(axis for axis in range(len(rhs_shape)) if axis not in rhs_excluded)

    # Validate paired dimensions.
    for la, ra in zip(lhs_batch, rhs_batch):
        if lhs_shape[la] != rhs_shape[ra]:
            raise ValueError(
                f"incompatible batch dimensions: "
                f"lhs[{la}]={lhs_shape[la]} != "
                f"rhs[{ra}]={rhs_shape[ra]}"
            )

    for la, ra in zip(lhs_contract, rhs_contract):
        if lhs_shape[la] != rhs_shape[ra]:
            raise ValueError(
                f"incompatible contracting dimensions: "
                f"lhs[{la}]={lhs_shape[la]} != "
                f"rhs[{ra}]={rhs_shape[ra]}"
            )

    batch_shape = tuple(lhs_shape[a] for a in lhs_batch)
    lhs_free_shape = tuple(lhs_shape[a] for a in lhs_free)
    rhs_free_shape = tuple(rhs_shape[a] for a in rhs_free)
    contract_shape = tuple(lhs_shape[a] for a in lhs_contract)

    expected_output_shape = batch_shape + lhs_free_shape + rhs_free_shape

    if output_shape != expected_output_shape:
        raise ValueError(
            f"unexpected dot_general output shape: "
            f"expected {expected_output_shape}, got {output_shape}"
        )

    full_shape = output_shape + contract_shape

    n_output = int(np.prod(output_shape, dtype=np.int64))
    n_contract = int(np.prod(contract_shape, dtype=np.int64))

    # --------------------------------------------------------------
    # LHS
    #
    # Reorder:
    #
    #   original
    #       ↓
    #   batch, lhs_free, contract
    #
    # Then insert singleton rhs_free axes:
    #
    #   batch, lhs_free, 1..., contract
    #
    # and broadcast across rhs_free.
    # --------------------------------------------------------------
    lhs_ids = np.arange(
        int(np.prod(lhs_shape, dtype=np.int64)), dtype=np.int64
    ).reshape(lhs_shape)

    lhs_perm = tuple(lhs_batch) + lhs_free + tuple(lhs_contract)
    lhs_ordered = np.transpose(lhs_ids, lhs_perm)
    lhs_view_shape = (
        batch_shape + lhs_free_shape + (1,) * len(rhs_free_shape) + contract_shape
    )
    lhs_rows = np.broadcast_to(
        lhs_ordered.reshape(lhs_view_shape),
        full_shape,
    ).reshape(n_output, n_contract)

    # --------------------------------------------------------------
    # RHS
    #
    # Reorder:
    #
    #   batch, rhs_free, contract
    #
    # Then insert singleton lhs_free axes.
    # --------------------------------------------------------------
    rhs_ids = np.arange(
        int(np.prod(rhs_shape, dtype=np.int64)), dtype=np.int64
    ).reshape(rhs_shape)

    rhs_perm = tuple(rhs_batch) + rhs_free + tuple(rhs_contract)
    rhs_ordered = np.transpose(rhs_ids, rhs_perm)
    rhs_view_shape = (
        batch_shape + (1,) * len(lhs_free_shape) + rhs_free_shape + contract_shape
    )
    rhs_rows = np.broadcast_to(
        rhs_ordered.reshape(rhs_view_shape),
        full_shape,
    ).reshape(n_output, n_contract)

    return DotGeneralMap(
        lhs_rows=lhs_rows,
        rhs_rows=rhs_rows,
        output_shape=output_shape,
    )


def dot_general_map(
    eqn: JaxprEqn,
) -> DotGeneralMap:
    return _dot_general_map(
        tuple(_shape_of(eqn.invars[0])),
        tuple(_shape_of(eqn.invars[1])),
        tuple(_shape_of(eqn.outvars[0])),
        _dot_dimension_numbers(eqn.params["dimension_numbers"]),
    )


def prepare_dot_general(ctx: RuleContext) -> DotGeneralMap:
    if len(ctx.input_deps) != 2 or len(ctx.eqn.outvars) != 1:
        raise ValueError(
            f"dot_general expects two inputs and one output; got "
            f"{len(ctx.input_deps)} inputs and {len(ctx.eqn.outvars)} outputs"
        )

    return dot_general_map(ctx.eqn)


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


def dot_general_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    output = ctx.output_demands[0]
    if output is None:
        return (None, None)

    prepared = dot_general_map(ctx.eqn)
    rows = demand_rows(output)
    lhs_rows = np.unique(prepared.lhs_rows[rows].ravel())
    rhs_rows = np.unique(prepared.rhs_rows[rows].ravel())

    return (
        TensorDemand.from_rows_hull(
            _shape_of(ctx.eqn.invars[0]),
            lhs_rows,
        ),
        TensorDemand.from_rows_hull(
            _shape_of(ctx.eqn.invars[1]),
            rhs_rows,
        ),
    )
