from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.local.dof_plan import (
    DofStorageLayout,
    LocalDofPlan,
    _freeze_i64,
    validate_dof_owner,
)


class HaloCommunicator(Protocol):
    def Get_rank(self) -> int: ...
    def Get_size(self) -> int: ...

    def alltoall(
        self,
        sendobj: list[Any],
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True, eq=False)
class NeighborExchange:
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


@dataclass(frozen=True, slots=True)
class MpiHaloExchangePlan:
    """Communication schedule for an explicit MPI backend."""

    dofs: LocalDofPlan

    recv: tuple[NeighborExchange, ...]
    send: tuple[NeighborExchange, ...]

    @property
    def rank(self) -> int:
        return self.dofs.rank


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


def build_mpi_halo_exchange_plan(
    dofs: LocalDofPlan,
    dof_owner: ArrayLike,
    *,
    comm: HaloCommunicator,
) -> MpiHaloExchangePlan:
    rank = int(comm.Get_rank())
    n_ranks = int(comm.Get_size())

    if rank != dofs.rank:
        raise ValueError(
            f"LocalDofPlan is for rank {dofs.rank}, communicator rank is {rank}"
        )

    owner = validate_dof_owner(dof_owner, n_ranks=n_ranks)
    if owner.size != dofs.global_size:
        raise ValueError("dof_owner size differs from LocalDofPlan.global_size")

    recv_global = _requests_by_owner(dofs.storage, owner)
    requests = [np.empty(0, dtype=np.int64) for _ in range(n_ranks)]

    for peer, values in recv_global.items():
        requests[peer] = values

    requests_from_rank = comm.alltoall(requests)
    send_global: dict[int, NDArray[np.int64]] = {}

    for peer, values in enumerate(requests_from_rank):
        if peer == rank:
            continue

        global_dofs = np.asarray(values, dtype=np.int64).ravel()
        if not global_dofs.size:
            continue

        if np.any((global_dofs < 0) | (global_dofs >= dofs.global_size)):
            raise RuntimeError(f"rank {peer} requested out-of-range global DOFs")

        if np.any(owner[global_dofs] != rank):
            raise RuntimeError(f"rank {peer} requested DOFs not owned by rank {rank}")

        send_global[peer] = global_dofs

    def exchanges(
        relations: Mapping[int, ArrayLike],
    ) -> tuple[NeighborExchange, ...]:
        return tuple(
            NeighborExchange(
                peer=peer,
                global_dofs=values,
                local_rows=(dofs.storage.global_to_local(values)),
            )
            for peer, raw_values in sorted(relations.items())
            if (values := np.asarray(raw_values, dtype=np.int64).ravel()).size
        )

    return MpiHaloExchangePlan(
        dofs=dofs,
        recv=exchanges(recv_global),
        send=exchanges(send_global),
    )
