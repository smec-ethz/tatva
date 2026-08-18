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
demand or construct local layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Var
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.routes import Shape
from tatva.tracer.local.demand import TensorDemand, merge_demands
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.program.contributions import (
    ContributionRoot,
    ContributionTrace,
    ValueRef,
)
from tatva.tracer.program.dependencies import DependencySet
from tatva.tracer.program.derivatives import (
    DerivativeTrace,
    JaxprDerivativeTrace,
)
from tatva.tracer.program.incidence import BlockDofIncidence


class PartitionStrategy(Enum):
    CONTIGUOUS = auto()
    DEPENDENCY = auto()
    INCIDENCE = auto()


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
    block_to_part: NDArray[np.int64] | None = None

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

    owners = np.empty(n_items, dtype=np.int64)
    quotient, remainder = divmod(n_items, n_parts)
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


def _validate_block_owners(
    owners: ArrayLike,
    *,
    n_blocks: int,
    n_parts: int,
) -> NDArray[np.int64]:
    result = np.asarray(owners, dtype=np.int64).ravel()
    if result.shape != (n_blocks,):
        raise ValueError(
            f"block owners have shape {result.shape}; expected ({n_blocks},)"
        )
    if np.any(result < 0) or np.any(result >= n_parts):
        raise ValueError(f"block owners must be in [0, {n_parts})")
    return result


def _owned_from_block_owners(
    incidence: BlockDofIncidence,
    owners: ArrayLike,
    *,
    n_parts: int,
) -> tuple[OwnedContribution, ...]:
    """Merge blocks with the same root/part into executable root demands."""
    mapping = _validate_block_owners(
        owners,
        n_blocks=incidence.n_blocks,
        n_parts=n_parts,
    )
    merged: dict[tuple[int, int], TensorDemand] = {}

    for block, part in zip(incidence.blocks, mapping, strict=True):
        key = (block.root_id, int(part))
        demand = merge_demands(merged.get(key), block.demand)
        assert demand is not None
        merged[key] = demand

    return tuple(
        OwnedContribution(root_id=root_id, part=part, demand=demand)
        for (root_id, part), demand in sorted(merged.items())
    )


def contiguous_block_owners(
    incidence: BlockDofIncidence,
    *,
    n_parts: int,
) -> NDArray[np.int64]:
    """Balanced contiguous block assignment, independently for each root."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    owners = np.empty(incidence.n_blocks, dtype=np.int64)
    root_ids = sorted({block.root_id for block in incidence.blocks})

    for root_id in root_ids:
        ids = np.asarray(
            [block.id for block in incidence.blocks if block.root_id == root_id],
            dtype=np.int64,
        )
        owners[ids] = _contiguous_owners(ids.size, n_parts)

    return owners


def dof_partition_block_owners(
    incidence: BlockDofIncidence,
    *,
    dof_to_part: ArrayLike,
    n_parts: int,
) -> NDArray[np.int64]:
    """Place blocks by the plurality of their DOFs' preassigned owners."""
    mapping = _validate_dof_partition(
        dof_to_part,
        n_dofs=incidence.n_dofs,
        n_parts=n_parts,
    )
    defaults = contiguous_block_owners(incidence, n_parts=n_parts)
    owners = defaults.copy()

    for block_id in range(incidence.n_blocks):
        dofs = incidence.dofs_for_block(block_id)
        if dofs.size == 0:
            continue
        counts = np.bincount(mapping[dofs], minlength=n_parts)
        tied = np.flatnonzero(counts == counts.max())
        preferred = defaults[block_id]
        owners[block_id] = preferred if preferred in tied else int(tied[0])

    return owners


def greedy_incidence_block_owners(
    incidence: BlockDofIncidence,
    *,
    n_parts: int,
) -> NDArray[np.int64]:
    """Deterministic local block placer balancing weight and new rank DOFs.

    This is an intentionally small validation partitioner, not a replacement
    for a distributed hypergraph backend.  Under a soft equal-weight cap it
    prefers the rank on which a block introduces the fewest new DOFs.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if incidence.n_blocks == 0:
        return np.empty(0, dtype=np.int64)

    weights = np.asarray(
        [float(block.weight) for block in incidence.blocks], dtype=float
    )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("contribution block weights must be finite and positive")

    target = float(weights.sum()) / n_parts
    largest = float(weights.max())
    soft_cap = max(target, largest)
    loads = np.zeros(n_parts, dtype=float)
    rank_dofs = [set() for _ in range(n_parts)]
    owners = np.empty(incidence.n_blocks, dtype=np.int64)

    # High-incidence blocks establish shared-DOF affinity first.  Dense block
    # IDs break ties and make the result reproducible.
    order = sorted(
        range(incidence.n_blocks),
        key=lambda block_id: (
            -incidence.dofs_for_block(block_id).size,
            block_id,
        ),
    )

    for block_id in order:
        weight = weights[block_id]
        dofs = incidence.dofs_for_block(block_id)
        feasible = [
            part for part in range(n_parts) if loads[part] + weight <= soft_cap + 1e-12
        ]
        candidates = feasible or list(range(n_parts))

        def score(
            part: int,
            block_dofs: NDArray[np.int64] = dofs,
        ) -> tuple[int, float, int]:
            introduced = sum(int(dof) not in rank_dofs[part] for dof in block_dofs)
            return introduced, loads[part], part

        owner = min(candidates, key=score)
        owners[block_id] = owner
        loads[owner] += weight
        rank_dofs[owner].update(int(dof) for dof in dofs)

    return owners


def partition_contribution_blocks(
    incidence: BlockDofIncidence,
    *,
    n_parts: int,
    strategy: PartitionStrategy = PartitionStrategy.INCIDENCE,
    dof_to_part: ArrayLike | None = None,
) -> tuple[ContributionPartition, NDArray[np.int64]]:
    """Partition blocks and return both merged root demands and block owners."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    if dof_to_part is not None:
        owners = dof_partition_block_owners(
            incidence,
            dof_to_part=dof_to_part,
            n_parts=n_parts,
        )
        effective_strategy = PartitionStrategy.DEPENDENCY
    elif strategy is PartitionStrategy.CONTIGUOUS:
        owners = contiguous_block_owners(incidence, n_parts=n_parts)
        effective_strategy = strategy
    elif strategy is PartitionStrategy.INCIDENCE:
        owners = greedy_incidence_block_owners(incidence, n_parts=n_parts)
        effective_strategy = strategy
    else:
        raise ValueError(f"unsupported block partition strategy {strategy!r}")

    partition = ContributionPartition(
        n_parts=n_parts,
        strategy=effective_strategy,
        owned=_owned_from_block_owners(incidence, owners, n_parts=n_parts),
        block_to_part=owners.copy(),
    )
    return partition, owners


def dof_owner_from_incidence(
    incidence: BlockDofIncidence,
    *,
    block_to_part: ArrayLike,
    n_parts: int,
) -> NDArray[np.int64]:
    """Choose a balanced owner among the ranks that compute each DOF."""
    block_owners = _validate_block_owners(
        block_to_part,
        n_blocks=incidence.n_blocks,
        n_parts=n_parts,
    )
    owner = np.zeros(incidence.n_dofs, dtype=np.int64)
    loads = np.zeros(n_parts, dtype=np.int64)
    by_dof = incidence.csr.tocsc(copy=False)

    for dof in range(incidence.n_dofs):
        start = by_dof.indptr[dof]
        stop = by_dof.indptr[dof + 1]
        blocks = by_dof.indices[start:stop]
        if blocks.size == 0:
            owner[dof] = 0
            continue
        else:
            candidates = np.unique(block_owners[blocks])
            selected = min(
                (int(part) for part in candidates), key=lambda p: (loads[p], p)
            )
        owner[dof] = selected
        loads[selected] += 1

    return owner


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


def dof_owner_from_contributions(
    *,
    owned: tuple[OwnedContribution, ...],
    roots: tuple[ContributionRoot, ...],
    dependencies: dict[Var, DependencySet],
    n_dofs: int,
    n_parts: int,
) -> NDArray[np.int64]:
    """Derive a unique DOF owner from contribution ownership.

    A DOF is assigned to the lowest-numbered partition containing an owned
    contribution that depends on that DOF.

    DOFs unused by every contribution are assigned to partition 0.
    """
    roots_by_id = {root.id: root for root in roots}
    owner = np.full(n_dofs, -1, dtype=np.int64)

    for item in sorted(owned, key=lambda x: x.part):
        root = roots_by_id[item.root_id]
        dep = dependencies[root.value.var]

        # Restrict to this partition's owned contribution entries.
        owned_layout = TensorLayout.from_demand(item.demand)

        local_rows = np.arange(owned_layout.local_size, dtype=np.int64)
        global_rows = owned_layout.local_rows_to_global_rows(local_rows)

        # Union of all DOFs influencing these contribution entries.
        dofs = np.unique(dep.csr[global_rows].indices)
        unowned = owner[dofs] < 0
        owner[dofs[unowned]] = item.part

    # DOFs that never affect the functional can be assigned arbitrarily.
    owner[owner < 0] = 0
    if np.any((owner < 0) | (owner >= n_parts)):
        raise RuntimeError("invalid derived DOF ownership")

    return owner


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
