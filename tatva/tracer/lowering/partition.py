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

import os
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import scipy.sparse as sps
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.core.routes import Shape
from tatva.tracer.local.demand import TensorDemand, merge_demands
from tatva.tracer.local.dof_plan import validate_dof_owner
from tatva.tracer.program.contributions import (
    ContributionBlock,
    ContributionRoot,
)
from tatva.tracer.program.incidence import BlockDofIncidence


class PartitionStrategy(Enum):
    CONTIGUOUS = auto()
    INCIDENCE = auto()
    MTKAHYPAR = auto()


@dataclass(frozen=True, slots=True)
class MtKaHyParOptions:
    imbalance: float = 0.03
    seed: int = 42
    threads: int | None = None

    def __post_init__(self) -> None:
        if self.imbalance < 0.0:
            raise ValueError("Mt-KaHyPar imbalance must be nonnegative")

        if self.threads is not None and self.threads <= 0:
            raise ValueError("Mt-KaHyPar threads must be positive")


@dataclass(frozen=True, slots=True)
class DistributionAssignments:
    n_parts: int
    strategy: PartitionStrategy
    block_to_part: NDArray[np.int64]
    dof_owner: NDArray[np.int64]

    def __post_init__(self) -> None:
        block_to_part = np.asarray(self.block_to_part, dtype=np.int64)
        dof_owner = np.asarray(self.dof_owner, dtype=np.int64)

        if np.any((block_to_part < 0) | (block_to_part >= self.n_parts)):
            raise ValueError("invalid block partition assignment")
        if np.any((dof_owner < 0) | (dof_owner >= self.n_parts)):
            raise ValueError("invalid DOF owner assignment")

        block_to_part.setflags(write=False)
        dof_owner.setflags(write=False)
        object.__setattr__(self, "block_to_part", block_to_part)
        object.__setattr__(self, "dof_owner", dof_owner)


def plan_distribution_assignments(
    partition_incidence: BlockDofIncidence,
    owner_incidence: BlockDofIncidence,
    *,
    n_parts: int,
    strategy: PartitionStrategy,
    dof_owner=None,
) -> DistributionAssignments:
    if partition_incidence.blocks != owner_incidence.blocks:
        raise ValueError("partition and owner incidence must have the same blocks")

    _, block_to_part = partition_contribution_blocks(
        partition_incidence, n_parts=n_parts, strategy=strategy
    )

    if dof_owner is None:
        owners = dof_owner_from_incidence(
            owner_incidence,
            block_to_part=block_to_part,
            n_parts=n_parts,
        )
    else:
        owners = validate_dof_owner(dof_owner, n_ranks=n_parts)

    return DistributionAssignments(
        n_parts=n_parts,
        strategy=strategy,
        block_to_part=block_to_part,
        dof_owner=owners,
    )


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
    blocks: tuple[ContributionBlock, ...],
    owners: ArrayLike,
    *,
    n_parts: int,
) -> tuple[OwnedContribution, ...]:
    """Merge blocks with the same root/part into executable root demands."""
    mapping = _validate_block_owners(
        owners,
        n_blocks=len(blocks),
        n_parts=n_parts,
    )
    merged: dict[tuple[int, int], TensorDemand] = {}

    for block, part in zip(blocks, mapping, strict=True):
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
    # A dense membership mask keeps the same rank/DOF relation as a set while
    # making block scoring and updates vectorized.  The incidence is already
    # expressed in global DOF IDs, so every row can index its rank mask
    # directly.  Retain the sparse-set implementation when the dense form
    # would be unreasonably large for a high-rank, high-DOF problem.
    dense_membership_bytes = n_parts * incidence.n_dofs
    use_dense_membership = dense_membership_bytes <= 64 * 1024 * 1024
    rank_dof_mask = (
        np.zeros((n_parts, incidence.n_dofs), dtype=bool)
        if use_dense_membership
        else None
    )
    rank_dof_sets = None if use_dense_membership else [set() for _ in range(n_parts)]
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

        def score(part: int) -> tuple[int, float, int]:
            if rank_dof_mask is not None:
                introduced = int(
                    dofs.size - np.count_nonzero(rank_dof_mask[part, dofs])
                )
            else:
                assert rank_dof_sets is not None
                introduced = sum(int(dof) not in rank_dof_sets[part] for dof in dofs)
            return introduced, loads[part], part

        owner = min(candidates, key=score)
        owners[block_id] = owner
        loads[owner] += weight
        if rank_dof_mask is not None:
            rank_dof_mask[owner, dofs] = True
        else:
            assert rank_dof_sets is not None
            rank_dof_sets[owner].update(int(dof) for dof in dofs)

    return owners


def mtkahypar_block_owners(
    incidence: BlockDofIncidence,
    *,
    n_parts: int,
    options: MtKaHyParOptions | None = None,
) -> NDArray[np.int64]:
    """Partition contribution blocks using Mt-KaHyPar.

    Hypernodes are contribution blocks and hyperedges are DOFs.
    A DOF hyperedge connects all contribution blocks incident to that DOF.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    n_blocks = incidence.n_blocks

    if n_blocks == 0:
        return np.empty(0, dtype=np.int64)

    if n_parts == 1:
        return np.zeros(n_blocks, dtype=np.int64)

    if n_parts > n_blocks:
        raise ValueError(
            f"cannot partition {n_blocks} contribution blocks "
            f"into {n_parts} nonempty parts"
        )

    try:
        import mtkahypar
    except ImportError as exc:
        raise ImportError(
            "PartitionStrategy.MTKAHYPAR requires the optional 'mtkahypar' package. "
            "Install it with `pip install mtkahypar`."
        ) from exc

    options = options or MtKaHyParOptions()
    preset = mtkahypar.PresetType.DEFAULT
    objective = mtkahypar.Objective.KM1

    # incidence.csr:
    #     block × dof
    #
    # Mt-KaHyPar wants one list of hypernodes for each hyperedge, i.e. for each DOF we
    # need the incident contribution blocks.
    by_dof = incidence.csr.tocsc(copy=False)
    by_dof.sort_indices()

    indptr = by_dof.indptr
    block_indices = by_dof.indices

    # Hyperedges of size 0 or 1 cannot affect a cut/connectivity objective, so omit them
    # entirely.
    #
    # This also avoids feeding useless degenerate nets into the partitioner.
    edge_sizes = np.diff(indptr)
    useful_dofs = np.flatnonzero(edge_sizes >= 2)

    hyperedges = [
        block_indices[indptr[dof] : indptr[dof + 1]]
        .astype(np.int64, copy=False)
        .tolist()
        for dof in useful_dofs
    ]

    # If there are no shared DOFs, there is no hypergraph objective to optimize. Use the
    # deterministic cheap fallback.
    if not hyperedges:
        return contiguous_block_owners(incidence, n_parts=n_parts)

    threads = options.threads if options.threads is not None else (os.cpu_count() or 1)
    mtk = mtkahypar.initialize(threads)
    context = mtk.context_from_preset(preset)

    context.set_partitioning_parameters(n_parts, float(options.imbalance), objective)
    mtkahypar.set_seed(int(options.seed))

    # Keep library output out of tatva unless explicitly exposed as a future debug option.
    context.logging = False

    node_weights = [1] * n_blocks
    hyperedge_weights = [1] * len(hyperedges)

    hypergraph = mtk.create_hypergraph(
        context,
        n_blocks,
        len(hyperedges),
        hyperedges,
        node_weights,
        hyperedge_weights,
    )

    partitioned = hypergraph.partition(context)

    owners = np.fromiter(
        (int(partitioned.block_id(block)) for block in range(n_blocks)),
        dtype=np.int64,
        count=n_blocks,
    )
    if owners.shape != (n_blocks,):
        raise RuntimeError(
            f"Mt-KaHyPar returned partition with shape {owners.shape}; expected {(n_blocks,)}"
        )
    if np.any((owners < 0) | (owners >= n_parts)):
        raise RuntimeError("Mt-KaHyPar returned an invalid block ID")

    return owners


def partition_contribution_from_assignments(
    blocks: tuple[ContributionBlock, ...],
    assignments: DistributionAssignments,
) -> ContributionPartition:
    return ContributionPartition(
        n_parts=assignments.n_parts,
        strategy=assignments.strategy,
        owned=_owned_from_block_owners(
            blocks,
            assignments.block_to_part,
            n_parts=assignments.n_parts,
        ),
        block_to_part=assignments.block_to_part.copy(),
    )


def contribution_partition_from_owners(
    blocks: tuple[ContributionBlock, ...],
    *,
    block_to_part: ArrayLike,
    n_parts: int,
    strategy: PartitionStrategy,
) -> ContributionPartition:
    """Construct contribution ownership from an already-computed assignment."""
    owners = _validate_block_owners(
        block_to_part, n_blocks=len(blocks), n_parts=n_parts
    )

    return ContributionPartition(
        n_parts=n_parts,
        strategy=strategy,
        owned=_owned_from_block_owners(blocks, owners, n_parts=n_parts),
        block_to_part=owners.copy(),
    )


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

    elif strategy is PartitionStrategy.INCIDENCE:
        owners = greedy_incidence_block_owners(incidence, n_parts=n_parts)

    elif strategy is PartitionStrategy.MTKAHYPAR:
        owners = mtkahypar_block_owners(incidence, n_parts=n_parts)

    else:
        raise ValueError(f"unsupported block partition strategy {strategy!r}")

    partition = contribution_partition_from_owners(
        incidence.blocks, block_to_part=owners, n_parts=n_parts, strategy=strategy
    )
    return partition, owners


def dof_owner_from_incidence(
    incidence: BlockDofIncidence,
    *,
    block_to_part: ArrayLike,
    n_parts: int,
) -> NDArray[np.int64]:
    """Choose a balanced owner among ranks that compute each DOF."""
    block_owners = _validate_block_owners(
        block_to_part,
        n_blocks=incidence.n_blocks,
        n_parts=n_parts,
    )
    n_dofs = incidence.n_dofs

    # Build DOF -> part incidence directly.
    #
    # block_part[b, p] = True iff block b belongs to part p.
    block_part = sps.csr_matrix(
        (
            np.ones(incidence.n_blocks, dtype=bool),
            (np.arange(incidence.n_blocks, dtype=np.int64), block_owners),
        ),
        shape=(incidence.n_blocks, n_parts),
        dtype=bool,
    )

    # incidence.csr is block × DOF.
    # Therefore:
    #     DOF × block @ block × part
    #       -> DOF × part
    eligible = (incidence.csr.T @ block_part).astype(bool).tocsr()
    owner = np.zeros(n_dofs, dtype=np.int64)
    loads = [0] * n_parts
    indptr = eligible.indptr
    indices = eligible.indices

    for dof in range(n_dofs):
        start = indptr[dof]
        stop = indptr[dof + 1]
        count = stop - start

        if count == 1:
            selected = int(indices[start])
            owner[dof] = selected
            loads[selected] += 1
        elif count == 0:
            owner[dof] = 0
        else:
            candidates = indices[start:stop]
            min_load = loads[candidates[0]]
            selected = int(candidates[0])
            for c in candidates[1:]:
                c_idx = int(c)
                load_c = loads[c_idx]
                if load_c < min_load:
                    min_load = load_c
                    selected = c_idx
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
