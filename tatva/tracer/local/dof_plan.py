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

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.local.layout import TensorLayout


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


def validate_dof_owner(
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

    _global_dofs: NDArray[np.int64] = field(init=False, repr=False)
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
        global_dofs.flags.writeable = False

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
        object.__setattr__(self, "_global_dofs", global_dofs)
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
        return self._global_dofs.size

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
class LocalDofPlan:
    """Compiler-derived rank-local DOF layout.

    This object describes storage and executable indexing only.
    It contains no communication-backend state.
    """

    rank: int
    global_size: int

    # Exact DOFs consumed by the localized executable.
    compute_global: NDArray[np.int64]

    # Runtime storage ABI: [owned | ghosts].
    storage: DofStorageLayout

    # executable_z = storage_z[compute_rows]
    compute_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        compute_global = _freeze_i64(self.compute_global)
        compute_rows = _freeze_i64(self.compute_rows)

        if self.storage.rank != self.rank:
            raise ValueError("storage rank differs from LocalDofPlan rank")

        if compute_global.shape != compute_rows.shape:
            raise ValueError("compute_global/compute_rows size mismatch")

        if np.any((compute_global < 0) | (compute_global >= self.global_size)):
            raise ValueError("compute_global contains out-of-range DOFs")

        if np.any((compute_rows < 0) | (compute_rows >= self.storage.local_size)):
            raise ValueError("compute_rows contains out-of-range storage rows")

        object.__setattr__(self, "compute_global", compute_global)
        object.__setattr__(self, "compute_rows", compute_rows)

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


def build_local_dof_plan(
    compute_layout: TensorLayout,
    dof_owner: ArrayLike,
    *,
    rank: int,
    n_ranks: int,
) -> LocalDofPlan:
    owner = validate_dof_owner(dof_owner, n_ranks=n_ranks)
    n_dofs = owner.size

    if rank < 0 or rank >= n_ranks:
        raise ValueError(f"rank {rank} is outside [0, {n_ranks})")

    compute_global = _compute_global(compute_layout, rank=rank, n_dofs=n_dofs)
    storage = _build_storage(rank=rank, compute_global=compute_global, owner=owner)

    return LocalDofPlan(
        rank=rank,
        global_size=n_dofs,
        compute_global=compute_global,
        storage=storage,
        compute_rows=storage.global_to_local(compute_global),
    )


def build_local_dof_plans(
    compute_layouts: Sequence[TensorLayout],
    dof_owner: ArrayLike,
) -> tuple[LocalDofPlan, ...]:
    n_ranks = len(compute_layouts)
    if n_ranks == 0:
        return ()

    owner = validate_dof_owner(dof_owner, n_ranks=n_ranks)

    return tuple(
        build_local_dof_plan(
            layout,
            owner,
            rank=rank,
            n_ranks=n_ranks,
        )
        for rank, layout in enumerate(compute_layouts)
    )


def pack_storage_from_global(
    global_values: ArrayLike,
    plan: LocalDofPlan,
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
    plan: LocalDofPlan,
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
    plan: LocalDofPlan,
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
