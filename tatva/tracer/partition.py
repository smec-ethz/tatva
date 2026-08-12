"""
Partition additive contribution domains between ranks.

A contribution root is partitioned along its first declared partition axis.
Ownership is always expressed as a selection of whole slices along that axis,
preserving the tensor structure discovered by contribution analysis.

Two partitioning strategies are currently supported:

1. Contiguous partitioning divides the contribution axis into balanced,
   contiguous ranges. This is the default and requires no graph partition.

2. Dependency-based partitioning consumes a global DOF-to-part map, for example
   one obtained by partitioning the Hessian graph with METIS. For each
   contribution-axis slice, the structural dependency sets of all scalar
   entries in that slice are unioned. The slice is assigned to the part owning
   the largest number of those dependent DOFs.

Repeated occurrences of the same DOF inside a contribution slice count only
once. Ties prefer the slice's default contiguous owner when possible, otherwise
the smallest part ID. Dependency-free slices also use their default contiguous
owner.

This module assigns ownership only. It does not propagate backward tensor
demand or construct local layouts. `OwnedContribution.flat_rows()` provides the
exact contribution rows that seed the subsequent demand pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import scipy.sparse as sps
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.contributions import (
    ContributionRoot,
    ContributionTrace,
    ValueRef,
)
from tatva.tracer.demand import TensorDemand
from tatva.tracer.dependencies import DependencySet
from tatva.tracer.derivatives import (
    DerivativeTrace,
    JaxprDerivativeTrace,
)
from tatva.tracer.model import Shape


class PartitionStrategy(Enum):
    CONTIGUOUS = auto()
    DEPENDENCY = auto()


@dataclass(frozen=True)
class OwnedContribution:
    """Portion of one ContributionRoot owned by one rank. When partition_axis is not Nonw,
    axis_indices identifies complete slices of global_shape along that axis. A None
    partition_axis means the entire contribution is owned by this rank."""

    root_id: int
    part: int
    demand: TensorDemand

    @property
    def global_shape(self) -> Shape:
        return self.demand.shape


@dataclass(frozen=True)
class ContributionPartition:
    n_parts: int
    strategy: PartitionStrategy
    owned: tuple[OwnedContribution, ...]

    def for_part(self, part: int) -> tuple[OwnedContribution, ...]:
        if part < 0 or part >= self.n_parts:
            raise IndexError(f"part {part} is out of range [0, {self.n_parts})")
        return tuple(item for item in self.owned if item.part == part)

    def for_root(self, root_id: int) -> tuple[OwnedContribution, ...]:
        return tuple(item for item in self.owned if item.root_id == root_id)


def _contiguous_owners(
    n_items: int,
    n_parts: int,
) -> NDArray[np.int64]:
    """Balanced contiguous assignment of [0, n_items) to parts.

    Earlier parts receive one extra item when n_items is not divisible by
    n_parts.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative")

    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    owners = np.empty(
        n_items,
        dtype=np.int64,
    )
    quotient, remainder = divmod(
        n_items,
        n_parts,
    )
    start = 0

    for part in range(n_parts):
        count = quotient + (1 if part < remainder else 0)
        stop = start + count
        owners[start:stop] = part
        start = stop

    return owners


def _owned_from_axis_owners(
    root: ContributionRoot,
    *,
    axis: int,
    owners: NDArray[np.int64],
    n_parts: int,
) -> list[OwnedContribution]:
    result: list[OwnedContribution] = []

    for part in range(n_parts):
        indices = np.flatnonzero(owners == part)
        demand = TensorDemand.axis_selection(root.domain.shape, axis, indices)
        if demand is None:
            continue

        result.append(
            OwnedContribution(
                root_id=root.id,
                part=part,
                demand=demand,
            )
        )

    return result


def _partition_root_contiguous(
    root: ContributionRoot,
    *,
    n_parts: int,
) -> list[OwnedContribution]:
    shape = root.domain.shape

    if not root.domain.partition_axes:
        # Nothing structurally partitionable.
        # Assign the complete root to rank zero.
        demand = TensorDemand.full(shape)
        if demand is None:
            return []
        return [OwnedContribution(root_id=root.id, part=0, demand=demand)]

    axis = root.domain.partition_axes[0]
    extent = shape[axis]
    owners = _contiguous_owners(extent, n_parts)

    return _owned_from_axis_owners(
        root,
        axis=axis,
        owners=owners,
        n_parts=n_parts,
    )


def _resolve_derivative_frame(
    trace: JaxprDerivativeTrace,
    value: ValueRef,
) -> JaxprDerivativeTrace:
    current = trace

    for step in value.path:
        try:
            nested = current.nested[step.eqn_index]
        except KeyError as exc:
            raise KeyError(
                f"no nested derivative trace for equation {step.eqn_index}"
            ) from exc

        if nested.invocation.kind is not step.kind:
            raise TypeError(
                f"expected {step.kind.name.lower()} derivative trace at equation "
                f"{step.eqn_index}"
            )

        try:
            current = nested.invocation.child_at(step)
        except KeyError as exc:
            if nested.template is not None:
                raise RuntimeError(
                    "cannot resolve an iteration-qualified ValueRef "
                    "through a template-optimized MapDerivativeTrace"
                ) from exc
            raise

    return current


def _dependency_of(
    derivatives: DerivativeTrace,
    value: ValueRef,
) -> DependencySet:
    frame = _resolve_derivative_frame(derivatives.root, value)

    try:
        return frame.dependencies[value.var]
    except KeyError as exc:
        raise KeyError(
            f"no derivative dependency recorded for {value.var} at path {value.path}"
        ) from exc


def _axis_slice_dependencies(
    dep: DependencySet,
    axis: int,
) -> sps.csr_matrix:
    """Return structural dependencies for whole slices along `axis`.

    Result shape: (dep.shape[axis], n_dofs)

    Row i contains every global DOF on which any scalar entry of tensor slice
    [..., i, ...] structurally depends.
    """
    shape = dep.shape
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"axis {axis} invalid for shape {shape}")

    n_entries = int(math.prod(shape))
    if dep.csr.shape[0] != n_entries:
        raise ValueError(
            f"DependencySet CSR has {dep.csr.shape[0]} rows "
            f"but shape {shape} contains {n_entries} entries"
        )

    extent = shape[axis]
    n_dofs = dep.csr.shape[1]
    if dep.csr.nnz == 0:
        return sps.csr_matrix(
            (extent, n_dofs),
            dtype=bool,
        )

    coo = dep.csr.tocoo(copy=False)

    # C-order flat row:
    #
    #     axis_coordinate
    #         = (flat_row // suffix_size) % axis_extent
    #
    suffix_size = int(math.prod(shape[axis + 1 :]))
    axis_rows = (coo.row // suffix_size) % extent

    result = sps.csr_matrix(
        (np.ones(coo.nnz, dtype=bool), (axis_rows, coo.col)),
        shape=(extent, n_dofs),
        dtype=bool,
    )
    result.sum_duplicates()
    if result.nnz:
        result.data[:] = True

    return result


def _validate_dof_partition(
    dof_to_part: ArrayLike,
    *,
    n_dofs: int,
    n_parts: int,
) -> NDArray[np.int64]:
    mapping = np.asarray(dof_to_part, dtype=np.int64).ravel()

    if mapping.shape != (n_dofs,):
        raise ValueError(f"dof_to_part has shape {mapping.shape}; expected ({n_dofs},)")

    if np.any(mapping < 0):
        raise ValueError("dof_to_part contains negative part IDs")

    if np.any(mapping >= n_parts):
        raise ValueError(f"dof_to_part contains part IDs >= n_parts={n_parts}")

    return mapping


def dependency_partition_owners(
    dep: DependencySet,
    *,
    axis: int,
    dof_to_part: ArrayLike,
    n_parts: int,
) -> NDArray[np.int64]:
    """
    Assign every slice along `axis` to one part.

    Ownership is chosen by the largest number of unique dependent DOFs owned
    by each part.

    Ties prefer balanced contiguous ownership when that part participates in
    the tie. Otherwise the smallest tied part wins.

    Dependency-free slices use contiguous ownership.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    slice_deps = _axis_slice_dependencies(dep, axis)
    extent, n_dofs = slice_deps.shape
    mapping = _validate_dof_partition(dof_to_part, n_dofs=n_dofs, n_parts=n_parts)
    default_owners = _contiguous_owners(extent, n_parts)
    owners = default_owners.copy()

    if slice_deps.nnz == 0:
        return owners

    coo = slice_deps.tocoo(copy=False)

    # Convert:
    #
    #     slice → unique dependent DOFs
    #
    # into:
    #
    #     slice → number of dependent DOFs owned by each part
    #
    dependency_parts = mapping[coo.col]
    counts = sps.csr_matrix(
        (
            np.ones(coo.nnz, dtype=np.int64),
            (coo.row, dependency_parts),
        ),
        shape=(extent, n_parts),
        dtype=np.int64,
    )
    counts.sum_duplicates()

    for slice_index in range(extent):
        start = counts.indptr[slice_index]
        stop = counts.indptr[slice_index + 1]

        if start == stop:
            # No active DOFs: retain balanced default ownership.
            continue

        parts = counts.indices[start:stop]
        values = counts.data[start:stop]
        maximum = values.max()
        tied_parts = parts[values == maximum]
        preferred = default_owners[slice_index]

        if np.any(tied_parts == preferred):
            owners[slice_index] = preferred
        else:
            owners[slice_index] = int(tied_parts.min())

    return owners


def _partition_root_by_dependency(
    root: ContributionRoot,
    *,
    derivatives: DerivativeTrace,
    dof_to_part: ArrayLike,
    n_parts: int,
) -> list[OwnedContribution]:
    shape = root.domain.shape
    dep = _dependency_of(derivatives, root.value)
    if dep.shape != shape:
        raise ValueError(
            f"contribution root {root.id} has domain shape "
            f"{shape}, but its DependencySet has shape "
            f"{dep.shape}"
        )

    # No declared structured partition axis.
    # The entire root must stay together.
    if not root.domain.partition_axes:
        mapping = _validate_dof_partition(
            dof_to_part, n_dofs=dep.csr.shape[1], n_parts=n_parts
        )
        active_dofs = np.unique(dep.csr.indices)

        if active_dofs.size == 0:
            owner = 0
        else:
            counts = np.bincount(mapping[active_dofs], minlength=n_parts)
            owner = int(np.flatnonzero(counts == counts.max())[0])

        demand = TensorDemand.full(shape)
        if demand is None:
            return []
        return [OwnedContribution(root_id=root.id, part=owner, demand=demand)]

    # NOTE: for now, we intentionally use only the first declared axis.
    axis = root.domain.partition_axes[0]
    owners = dependency_partition_owners(
        dep,
        axis=axis,
        dof_to_part=dof_to_part,
        n_parts=n_parts,
    )

    return _owned_from_axis_owners(
        root,
        axis=axis,
        owners=owners,
        n_parts=n_parts,
    )


def partition_contributions(
    contributions: ContributionTrace,
    *,
    n_parts: int,
    derivatives: DerivativeTrace | None = None,
    dof_to_part: ArrayLike | None = None,
) -> ContributionPartition:
    """Partition detected contribution roots among `n_parts`.

    With no `dof_to_part`, contribution domains are divided contiguously along
    their first declared partition axis.

    When `dof_to_part` is provided, `derivatives` must also be provided.
    Contribution slices are assigned according to their structural global-DOF
    dependencies.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    if dof_to_part is None:
        strategy = PartitionStrategy.CONTIGUOUS
        owned = [
            item
            for root in contributions.roots
            for item in _partition_root_contiguous(root, n_parts=n_parts)
        ]

    else:
        strategy = PartitionStrategy.DEPENDENCY
        if derivatives is None:
            raise ValueError("derivatives are required when dof_to_part is supplied")

        owned = [
            item
            for root in contributions.roots
            for item in _partition_root_by_dependency(
                root, derivatives=derivatives, dof_to_part=dof_to_part, n_parts=n_parts
            )
        ]

    return ContributionPartition(
        n_parts=n_parts,
        strategy=strategy,
        owned=tuple(owned),
    )
