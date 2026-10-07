from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.lowering.partition import DistributionAssignments, PartitionStrategy
from tatva.tracer.program.contributions import ContributionBlock
from tatva.tracer.program.incidence import BlockCoordinateIncidence


class DistributedPlanningError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BlockShard:
    global_n_blocks: int
    start: int
    stop: int
    blocks: tuple[ContributionBlock, ...]

    def __post_init__(self) -> None:
        if self.global_n_blocks < 0:
            raise ValueError("global block count must be nonnegative")

        if not (0 <= self.start <= self.stop <= self.global_n_blocks):
            raise ValueError("invalid contribution block shard")

        if len(self.blocks) != self.stop - self.start:
            raise ValueError("block shard length mismatch")

        expected = np.arange(self.start, self.stop, dtype=np.int64)
        actual = np.fromiter(
            (block.id for block in self.blocks), dtype=np.int64, count=len(self.blocks)
        )
        if not np.array_equal(actual, expected):
            raise ValueError("block shard must contain its contiguous global ID range")

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def global_ids(self) -> NDArray[np.int64]:
        return np.arange(self.start, self.stop, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class BlockCoordinateIncidenceShard:
    shard: BlockShard
    coordinate_order: tuple[str, ...]
    # Each matrix:
    #     local block rows × global coordinate columns
    by_coordinate: dict[str, sps.csr_matrix]

    def __post_init__(self) -> None:
        if tuple(self.by_coordinate) != self.coordinate_order:
            raise ValueError("coordinate incidence order mismatch")

        normalized = {}

        for name in self.coordinate_order:
            matrix = sps.csr_matrix(self.by_coordinate[name], dtype=bool)
            if matrix.shape[0] != self.shard.size:
                raise ValueError(
                    f"coordinate {name!r} has {matrix.shape[0]} local rows; "
                    f"expected {self.shard.size}"
                )

            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()

            if matrix.nnz:
                matrix.data[:] = True

            normalized[name] = matrix

        object.__setattr__(self, "by_coordinate", normalized)


def shard_coordinate_incidence(
    incidence: BlockCoordinateIncidence,
    *,
    shard: BlockShard,
) -> BlockCoordinateIncidenceShard:
    return BlockCoordinateIncidenceShard(
        shard=shard,
        coordinate_order=incidence.coordinate_order,
        by_coordinate={
            name: incidence.by_coordinate[name] for name in incidence.coordinate_order
        },
    )


def block_shard_for_rank(
    blocks: tuple[ContributionBlock, ...],
    *,
    rank: int,
    size: int,
) -> BlockShard:
    if size <= 0:
        raise ValueError("MPI size must be positive")
    if rank < 0 or rank >= size:
        raise ValueError("MPI rank outside communicator")

    n_blocks = len(blocks)
    start = (n_blocks * rank) // size
    stop = (n_blocks * (rank + 1)) // size

    return BlockShard(
        global_n_blocks=n_blocks,
        start=start,
        stop=stop,
        blocks=blocks[start:stop],
    )


def _prefix_displacements(
    counts: NDArray[np.int64],
) -> NDArray[np.int64]:
    displs = np.empty_like(counts)
    if counts.size:
        displs[0] = 0

    if counts.size > 1:
        np.cumsum(counts[:-1], out=displs[1:])

    return displs


def gather_row_sharded_bool_csr(
    local: sps.csr_matrix,
    *,
    global_n_rows: int,
    comm,
    root: int = 0,
) -> sps.csr_matrix | None:
    """Gather contiguous CSR row shards onto one MPI rank."""
    from mpi4py import MPI

    rank = comm.Get_rank()
    size = comm.Get_size()

    local = local.tocsr(copy=False)

    local.sort_indices()
    local.sum_duplicates()
    local.eliminate_zeros()

    local_row_counts = np.diff(local.indptr).astype(np.int64, copy=False)
    local_indices = np.asarray(local.indices, dtype=np.int64)

    # Contiguous block sharding means the root knows exactly how
    # many rows each compiler rank owns.
    row_counts = np.asarray(
        [
            (global_n_rows * (r + 1)) // size - (global_n_rows * r) // size
            for r in range(size)
        ],
        dtype=np.int64,
    )
    row_displs = _prefix_displacements(row_counts)

    if rank == root:
        global_row_counts = np.empty(global_n_rows, dtype=np.int64)
    else:
        global_row_counts = None

    comm.Gatherv(
        local_row_counts,
        (global_row_counts, row_counts, row_displs, MPI.INT64_T)
        if rank == root
        else None,
        root=root,
    )

    # Gather sparse column-index stream.
    local_nnz = int(local_indices.size)
    nnz_counts_list = comm.gather(local_nnz, root=root)

    if rank == root:
        nnz_counts = np.asarray(nnz_counts_list, dtype=np.int64)
        nnz_displs = _prefix_displacements(nnz_counts)
        total_nnz = int(nnz_counts.sum(dtype=np.int64))
        global_indices = np.empty(total_nnz, dtype=np.int64)
    else:
        nnz_counts = None
        nnz_displs = None
        global_indices = None

    comm.Gatherv(
        local_indices,
        (global_indices, nnz_counts, nnz_displs, MPI.INT64_T) if rank == root else None,
        root=root,
    )

    if rank != root:
        return None

    assert global_row_counts is not None
    assert global_indices is not None

    indptr = np.empty(global_n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(global_row_counts, out=indptr[1:])
    data = np.ones(global_indices.size, dtype=bool)

    result = sps.csr_matrix(
        (data, global_indices, indptr),
        shape=(global_n_rows, local.shape[1]),
        dtype=bool,
    )

    result.sort_indices()

    return result


def gather_coordinate_incidence(
    local: BlockCoordinateIncidenceShard,
    *,
    global_blocks: tuple[ContributionBlock, ...],
    comm,
    root: int = 0,
) -> BlockCoordinateIncidence | None:
    rank = comm.Get_rank()
    matrices = {}

    for name in local.coordinate_order:
        gathered = gather_row_sharded_bool_csr(
            local.by_coordinate[name],
            global_n_rows=(local.shard.global_n_blocks),
            comm=comm,
            root=root,
        )

        if rank == root:
            assert gathered is not None
            matrices[name] = gathered

    if rank != root:
        return None

    return BlockCoordinateIncidence(
        blocks=global_blocks,
        coordinate_order=(local.coordinate_order),
        by_coordinate=matrices,
    )


def broadcast_assignments(
    assignments: DistributionAssignments | None,
    *,
    n_blocks: int,
    n_dofs: int,
    n_parts: int,
    strategy: PartitionStrategy,
    comm,
    root: int = 0,
) -> DistributionAssignments:
    rank = comm.Get_rank()

    if rank == root:
        assert assignments is not None

        block_to_part = np.asarray(assignments.block_to_part, dtype=np.int64).copy()
        dof_owner = np.asarray(assignments.dof_owner, dtype=np.int64).copy()

    else:
        block_to_part = np.empty(n_blocks, dtype=np.int64)
        dof_owner = np.empty(n_dofs, dtype=np.int64)

    comm.Bcast(block_to_part, root=root)
    comm.Bcast(dof_owner, root=root)

    return DistributionAssignments(
        n_parts=n_parts,
        strategy=strategy,
        block_to_part=block_to_part,
        dof_owner=dof_owner,
    )


def collective_check(
    comm,
    error: BaseException | None,
    *,
    phase: str,
) -> None:
    import traceback

    local_message = (
        None
        if error is None
        else (
            f"{type(error).__name__}: "
            f"{error}\n"
            f"{''.join(traceback.format_exception(error))}"
        )
    )

    messages = comm.allgather(local_message)
    failures = [
        (rank, message) for rank, message in enumerate(messages) if message is not None
    ]

    if failures:
        details = "\n\n".join(f"[rank {rank}]\n{message}" for rank, message in failures)
        raise DistributedPlanningError(
            f"MPI distribution phase {phase!r} failed:\n{details}"
        )
