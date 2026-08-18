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

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.routes import Shape
from tatva.tracer.local.demand import TensorDemand, merge_demands
from tatva.tracer.program.contributions import (
    ContributionRoot,
)
from tatva.tracer.program.incidence import BlockDofIncidence


class PartitionStrategy(Enum):
    CONTIGUOUS = auto()
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
) -> tuple[ContributionPartition, NDArray[np.int64]]:
    """Partition blocks and return both merged root demands and block owners."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    if strategy is PartitionStrategy.CONTIGUOUS:
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
