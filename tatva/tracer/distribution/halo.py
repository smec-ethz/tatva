"""
Owned/ghost DOF layouts and halo communication planning.

The tracer determines which global DOFs each rank needs for its localized
functional. Given a unique global DOF owner map, this module derives:

    owned DOFs
    ghost DOFs
    runtime storage ordering
    compute-layout -> storage mapping
    neighbor send/receive schedules

Communication itself is deliberately separated from tracing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.local.layout import TensorLayout


class HaloCommunicator(Protocol):
    """Minimal collective interface for rank-local halo planning.

    `mpi4py.MPI.Comm` satisfies this protocol. MPI remains an optional Tatva
    dependency because the tracer imports only this structural interface.
    """

    def Get_rank(self) -> int: ...

    def Get_size(self) -> int: ...

    def alltoall(self, sendobj: list[Any]) -> list[Any]: ...


def _freeze_i64(
    values: ArrayLike,
) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64).ravel().copy()
    result.flags.writeable = False
    return result


def _require_sorted_unique(
    values: NDArray[np.int64],
    *,
    name: str,
) -> None:
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True, slots=True, eq=False)
class DofStorageLayout:
    """Runtime vector layout for one rank.

    Local storage order is:

        [all owned DOFs | required ghost DOFs]

    Therefore owned values form a contiguous prefix, which is convenient
    for solvers and distributed vectors.
    """

    rank: int

    owned_global: NDArray[np.int64]
    ghost_global: NDArray[np.int64]

    _lookup_global: NDArray[np.int64] = field(init=False, repr=False)
    _lookup_local: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        owned = _freeze_i64(self.owned_global)
        ghost = _freeze_i64(self.ghost_global)

        _require_sorted_unique(owned, name="owned_global")
        _require_sorted_unique(ghost, name="ghost_global")

        if np.intersect1d(owned, ghost).size:
            raise ValueError("owned and ghost DOFs overlap")

        global_dofs = np.concatenate((owned, ghost))

        if global_dofs.size:
            order = np.argsort(global_dofs, kind="stable")
            lookup_global = global_dofs[order]
            lookup_local = order.astype(np.int64, copy=False)

        else:
            lookup_global = np.empty(0, dtype=np.int64)
            lookup_local = np.empty(0, dtype=np.int64)

        lookup_global = _freeze_i64(lookup_global)
        lookup_local = _freeze_i64(lookup_local)

        object.__setattr__(self, "owned_global", owned)
        object.__setattr__(self, "ghost_global", ghost)
        object.__setattr__(self, "_lookup_global", lookup_global)
        object.__setattr__(self, "_lookup_local", lookup_local)

    @property
    def n_owned(self) -> int:
        return self.owned_global.size

    @property
    def n_ghost(self) -> int:
        return self.ghost_global.size

    @property
    def local_size(self) -> int:
        return self.n_owned + self.n_ghost

    @property
    def global_dofs(
        self,
    ) -> NDArray[np.int64]:
        return np.concatenate((self.owned_global, self.ghost_global))

    @property
    def owned_local_rows(
        self,
    ) -> NDArray[np.int64]:
        return np.arange(self.n_owned, dtype=np.int64)

    @property
    def ghost_local_rows(
        self,
    ) -> NDArray[np.int64]:
        return np.arange(self.n_owned, self.local_size, dtype=np.int64)

    def global_to_local(
        self,
        global_dofs: ArrayLike,
    ) -> NDArray[np.int64]:
        values = np.asarray(global_dofs, dtype=np.int64)

        original_shape = values.shape
        flat = values.ravel()

        if flat.size == 0:
            return np.empty(original_shape, dtype=np.int64)

        positions = np.searchsorted(self._lookup_global, flat)
        valid = positions < self._lookup_global.size
        safe = np.minimum(positions, max(self._lookup_global.size - 1, 0))

        if self._lookup_global.size:
            valid &= self._lookup_global[safe] == flat

        if not np.all(valid):
            missing = np.unique(flat[~valid])
            raise ValueError(
                f"global DOFs {missing[:8].tolist()} are not stored by rank {self.rank}"
            )

        return self._lookup_local[positions].reshape(original_shape)


@dataclass(frozen=True, slots=True, eq=False)
class NeighborExchange:
    """One directed communication edge.

    `global_dofs` defines buffer ordering and is identical on sender/receiver.

    `local_rows` gives the corresponding positions in this rank's
    DofStorageLayout.
    """

    peer: int
    global_dofs: NDArray[np.int64]
    local_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        global_dofs = _freeze_i64(self.global_dofs)
        local_rows = _freeze_i64(self.local_rows)

        if global_dofs.shape != local_rows.shape:
            raise ValueError("global_dofs/local_rows size mismatch")

        object.__setattr__(self, "global_dofs", global_dofs)
        object.__setattr__(self, "local_rows", local_rows)


@dataclass(frozen=True, slots=True, eq=False)
class HaloPlan:
    rank: int

    # Exact liveness-derived DOFs consumed by the executable.
    compute_global: NDArray[np.int64]

    # Runtime state vector.
    storage: DofStorageLayout

    # executable u_local = storage_values[compute_rows]
    compute_rows: NDArray[np.int64]

    recv: tuple[NeighborExchange, ...]
    send: tuple[NeighborExchange, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute_global", _freeze_i64(self.compute_global))
        object.__setattr__(self, "compute_rows", _freeze_i64(self.compute_rows))

    @property
    def owned_global(
        self,
    ) -> NDArray[np.int64]:
        return self.storage.owned_global

    @property
    def ghost_global(
        self,
    ) -> NDArray[np.int64]:
        return self.storage.ghost_global


def _validate_owner(
    dof_owner: ArrayLike,
    *,
    n_ranks: int,
) -> NDArray[np.int64]:
    if n_ranks <= 0:
        raise ValueError("number of ranks must be positive")

    owner = np.asarray(dof_owner, dtype=np.int64).ravel()
    if np.any((owner < 0) | (owner >= n_ranks)):
        raise ValueError("dof_owner contains invalid rank indices")
    return owner


def _compute_global(
    layout: TensorLayout,
    *,
    rank: int,
    n_dofs: int,
) -> NDArray[np.int64]:
    if layout.global_shape != (n_dofs,):
        raise ValueError(
            f"rank {rank} DOF layout has global shape "
            f"{layout.global_shape}; expected {(n_dofs,)}"
        )
    return np.asarray(layout.global_axis_indices(0), dtype=np.int64)


def _build_storage(
    *,
    rank: int,
    compute_global: NDArray[np.int64],
    owner: NDArray[np.int64],
) -> DofStorageLayout:
    owned_global = np.flatnonzero(owner == rank).astype(np.int64)
    ghost_global = compute_global[owner[compute_global] != rank]
    ghost_global = np.unique(ghost_global).astype(np.int64, copy=False)
    return DofStorageLayout(
        rank=rank,
        owned_global=owned_global,
        ghost_global=ghost_global,
    )


def _requests_by_owner(
    storage: DofStorageLayout,
    owner: NDArray[np.int64],
) -> dict[int, NDArray[np.int64]]:
    result: dict[int, NDArray[np.int64]] = {}
    ghosts = storage.ghost_global
    if not ghosts.size:
        return result

    ghost_owners = owner[ghosts]
    for peer_value in np.unique(ghost_owners):
        peer = int(peer_value)
        result[peer] = np.ascontiguousarray(
            ghosts[ghost_owners == peer], dtype=np.int64
        )
    return result


def _build_rank_halo_plan(
    *,
    rank: int,
    compute_global: NDArray[np.int64],
    storage: DofStorageLayout,
    recv_global: Mapping[int, ArrayLike],
    send_global: Mapping[int, ArrayLike],
) -> HaloPlan:
    def exchanges(
        relations: Mapping[int, ArrayLike],
    ) -> tuple[NeighborExchange, ...]:
        return tuple(
            NeighborExchange(
                peer=peer,
                global_dofs=values,
                local_rows=storage.global_to_local(values),
            )
            for peer, raw_values in sorted(relations.items())
            if (values := np.asarray(raw_values, dtype=np.int64).ravel()).size
        )

    return HaloPlan(
        rank=rank,
        compute_global=compute_global,
        storage=storage,
        compute_rows=storage.global_to_local(compute_global),
        recv=exchanges(recv_global),
        send=exchanges(send_global),
    )


def build_local_halo_plan(
    compute_layout: TensorLayout,
    dof_owner: ArrayLike,
    *,
    comm: HaloCommunicator,
) -> HaloPlan:
    """Collectively build only this communicator rank's halo plan.

    Every rank supplies its own liveness-derived root-DOF layout. Receives are
    known locally from that layout and the replicated owner array. One object
    `alltoall` exchanges ghost requests with their owners, providing exactly
    the information needed to derive the send schedule.

    All ranks in `comm` must call this function in the same collective order.
    This host-side planning collective is not part of JIT execution.
    """
    rank = int(comm.Get_rank())
    n_ranks = int(comm.Get_size())
    if rank < 0 or rank >= n_ranks:
        raise ValueError(f"communicator rank {rank} is outside [0, {n_ranks})")

    owner = _validate_owner(dof_owner, n_ranks=n_ranks)
    n_dofs = owner.size
    compute_global = _compute_global(
        compute_layout,
        rank=rank,
        n_dofs=n_dofs,
    )
    storage = _build_storage(
        rank=rank,
        compute_global=compute_global,
        owner=owner,
    )
    recv_global = _requests_by_owner(storage, owner)

    requests_to_owner = [np.empty(0, dtype=np.int64) for _ in range(n_ranks)]
    for peer, values in recv_global.items():
        requests_to_owner[peer] = values

    requests_from_rank = comm.alltoall(requests_to_owner)
    if len(requests_from_rank) != n_ranks:
        raise RuntimeError(
            "communicator alltoall returned "
            f"{len(requests_from_rank)} entries; expected {n_ranks}"
        )

    send_global: dict[int, NDArray[np.int64]] = {}
    for peer, values in enumerate(requests_from_rank):
        if peer == rank:
            continue
        global_dofs = np.asarray(values, dtype=np.int64).ravel()
        if not global_dofs.size:
            continue
        if np.any((global_dofs < 0) | (global_dofs >= n_dofs)):
            raise RuntimeError(f"rank {peer} requested out-of-range global DOFs")
        if np.any(owner[global_dofs] != rank):
            raise RuntimeError(f"rank {peer} requested DOFs not owned by rank {rank}")
        send_global[peer] = global_dofs

    return _build_rank_halo_plan(
        rank=rank,
        compute_global=compute_global,
        storage=storage,
        recv_global=recv_global,
        send_global=send_global,
    )


def build_halo_plans(
    compute_layouts: Sequence[TensorLayout],
    dof_owner: ArrayLike,
) -> tuple[HaloPlan, ...]:
    """Build the complete communication plan for all ranks.

    Args:
        compute_layouts:
            Rank-specific liveness-derived layout of the root DOF vector.
        dof_owner:
            Integer array of shape `(n_dofs,)`.
            `dof_owner[i]` is the unique rank owning global DOF i.
    """
    n_ranks = len(compute_layouts)

    if n_ranks == 0:
        return ()

    owner = _validate_owner(dof_owner, n_ranks=n_ranks)
    n_dofs = owner.size

    # Exact compute requirements from liveness.
    required = [
        _compute_global(layout, rank=rank, n_dofs=n_dofs)
        for rank, layout in enumerate(compute_layouts)
    ]

    # Storage layouts.
    #
    # Each rank stores:
    #
    #     every DOF it owns
    #     +
    #     only ghosts required by its local computation
    storage = [
        _build_storage(
            rank=rank,
            compute_global=required[rank],
            owner=owner,
        )
        for rank in range(n_ranks)
    ]

    # Receive relation.
    #
    # recv[receiver][owner] = ghost DOFs requested from owner.
    recv_global = [_requests_by_owner(rank_storage, owner) for rank_storage in storage]

    # Send relation is exactly the transpose.
    send_global: list[dict[int, NDArray[np.int64]]] = [{} for _ in range(n_ranks)]

    for receiver in range(n_ranks):
        for sender, values in recv_global[receiver].items():
            send_global[sender][receiver] = values

    plans = tuple(
        _build_rank_halo_plan(
            rank=rank,
            compute_global=required[rank],
            storage=storage[rank],
            recv_global=recv_global[rank],
            send_global=send_global[rank],
        )
        for rank in range(n_ranks)
    )

    _validate_halo_plans(plans)
    return plans


def _validate_halo_plans(
    plans: tuple[HaloPlan, ...],
) -> None:
    by_rank = {plan.rank: plan for plan in plans}

    for receiver in plans:
        for recv in receiver.recv:
            sender = by_rank[recv.peer]
            matching = [send for send in sender.send if send.peer == receiver.rank]

            if len(matching) != 1:
                raise RuntimeError("halo send/recv relation is not symmetric")

            send = matching[0]

            if not np.array_equal(recv.global_dofs, send.global_dofs):
                raise RuntimeError("halo sender/receiver buffer ordering differs")


def pack_storage_from_global(
    global_values: ArrayLike,
    plan: HaloPlan,
) -> NDArray:
    """Testing/reference helper.

    Real distributed execution should normally construct the owned prefix
    directly and leave ghost entries to halo exchange.
    """
    values = np.asarray(global_values)
    if values.ndim != 1:
        raise ValueError("DOF vector must be one-dimensional")

    return np.asarray(values[plan.storage.global_dofs]).copy()


def compute_values(
    storage_values: ArrayLike,
    plan: HaloPlan,
) -> NDArray:
    """Project runtime owned+ghost storage into the exact DOF ordering expected
    by the localized executable.
    """
    values = np.asarray(storage_values)
    if values.shape != (plan.storage.local_size,):
        raise ValueError(
            f"storage shape {values.shape} does not match ({plan.storage.local_size},)"
        )

    return values[plan.compute_rows]


def scatter_compute_gradient(
    compute_gradient: ArrayLike,
    plan: HaloPlan,
) -> NDArray:
    """Embed the gradient of the local executable back into owned+ghost storage.

    Ghost contributions are later sent back to their owners via the reverse
    halo reduction.
    """
    gradient = np.asarray(compute_gradient)
    if gradient.shape != (plan.compute_rows.size,):
        raise ValueError("compute-gradient shape mismatch")

    result = np.zeros(plan.storage.local_size, dtype=gradient.dtype)
    np.add.at(result, plan.compute_rows, gradient)

    return result
