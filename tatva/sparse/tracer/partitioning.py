"""Backward decomposition of scalar JAX functionals into contribution roots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from jax.extend.core import Jaxpr, JaxprEqn
from numpy.typing import NDArray

from tatva.sparse.tracer.common import _get_shape

if TYPE_CHECKING:
    from tatva.sparse.tracer.state import BoundEqn, TraceState


@dataclass
class ContributionDemand:
    rows: NDArray[np.int64]


@dataclass
class ContributionRoot:
    var: Any
    rows: NDArray[np.int64]


@dataclass
class ContributionPropagation:
    in_demands: list[ContributionDemand | None]
    roots: list[ContributionRoot]
    valid: bool = True


@dataclass(frozen=True)
class RankPartitionPlan:
    rank: int

    contribution_rows: dict[int, NDArray[np.int64]]
    """root index -> original flat rows evaluated by this rank"""

    required_dofs: NDArray[np.int64]
    """all global dofs read by this rank's contributions"""

    owned_dofs: NDArray[np.int64]
    """all globally owned dofs according to part_map"""

    required_owned_dofs: NDArray[np.int64]
    """required dofs that are locally owned"""

    ghost_dofs: NDArray[np.int64]
    """required dofs that are not locally owned"""

    local_to_global: NDArray[np.int64]
    """distributed vector layout: owned first, ghosts second"""

    global_to_local: NDArray[np.int64]
    """dense global -> local lookup; -1 means unavailable"""


@dataclass(frozen=True)
class EnergyPartitionPlan:
    part_map: NDArray[np.int64]
    roots: tuple[ContributionRoot, ...]
    ranks: tuple[RankPartitionPlan, ...]

    def ghost_dofs_by_rank(self) -> dict[int, NDArray[np.int64]]:
        return {rank_plan.rank: rank_plan.ghost_dofs for rank_plan in self.ranks}


@dataclass(frozen=True)
class RankLivenessPlan:
    rank: int

    # demanded_rows: dict[int, NDArray[np.int64]]
    # """var id -> demanded flat entries"""
    all_demands: dict[int, ContributionDemand]
    """var id -> demanded flat entries"""

    live_eqn_ids: frozenset[int]
    """eqn ids needed by this rank"""


def _demand(rows: NDArray[np.integer]) -> ContributionDemand | None:
    """Build a canonical demand, dropping rows with no source element."""
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[rows >= 0]
    if not rows.size:
        return None
    # Most liveness rules preserve an already canonical demand (or create an
    # ``arange``).  Avoid allocating a hash table/sorted copy on that hot path.
    if rows.size == 1 or np.all(rows[1:] > rows[:-1]):
        return ContributionDemand(rows)
    return ContributionDemand(np.unique(rows))


def _invalid_contribution(eqn: JaxprEqn) -> ContributionPropagation:
    """Conservatively stop decomposition at the demanded output."""
    return ContributionPropagation([None] * len(eqn.invars), [], valid=False)


def _validate_partition_map(part_map: NDArray[np.integer], n_dofs: int) -> np.ndarray:
    """Return a validated, contiguous rank map for a global DOF vector."""
    partition = np.asarray(part_map, dtype=np.int64)
    if partition.ndim != 1:
        raise ValueError("part_map must be a one-dimensional array of rank IDs.")
    if partition.size != n_dofs:
        raise ValueError(
            f"part_map has {partition.size} entries, but the functional has {n_dofs} DOFs."
        )
    if partition.size and np.any(partition < 0):
        raise ValueError("part_map rank IDs must be non-negative.")
    if partition.size:
        ranks = np.unique(partition)
        expected = np.arange(ranks[-1] + 1, dtype=np.int64)
        if not np.array_equal(ranks, expected):
            raise ValueError("part_map rank IDs must be contiguous and start at zero.")
    return partition


def _dependency_row_owners(
    dep,
    partition: NDArray[np.int64],
    *,
    static_owner: int = 0,
) -> NDArray[np.int64]:
    """Assign each dependency row to one rank.

    Nonempty rows are owned by the rank owning their minimum global DOF.
    Empty rows are independent of the root DOF vector and are assigned once
    to ``static_owner``.
    """
    owners = np.full(dep.shape[0], static_owner, dtype=np.int64)

    for row in range(dep.shape[0]):
        start = dep.indptr[row]
        end = dep.indptr[row + 1]

        if start == end:
            continue

        minimum_dof = int(np.min(dep.indices[start:end]))
        owners[row] = partition[minimum_dof]

    return owners


@overload
def merge_demands(
    left: ContributionDemand, right: ContributionDemand
) -> ContributionDemand: ...
@overload
def merge_demands(
    left: ContributionDemand | None, right: ContributionDemand
) -> ContributionDemand: ...
@overload
def merge_demands(
    left: ContributionDemand, right: ContributionDemand | None
) -> ContributionDemand: ...
def merge_demands(
    left: ContributionDemand | None, right: ContributionDemand | None
) -> ContributionDemand | None:
    """Union two flat-entry demands, retaining canonical sorted row IDs."""
    if left is None:
        return right
    if right is None:
        return left
    if left is right or np.array_equal(left.rows, right.rows):
        return left
    return ContributionDemand(np.union1d(left.rows, right.rows).astype(np.int64))


def merge_contribution_roots(
    roots: Iterable[ContributionRoot],
) -> list[ContributionRoot]:
    """Coalesce repeated roots reached through separate additive branches."""
    merged: dict[int, ContributionRoot] = {}
    for root in roots:
        previous = merged.get(id(root.var))
        if previous is None:
            merged[id(root.var)] = root
        else:
            demand = merge_demands(
                ContributionDemand(previous.rows), ContributionDemand(root.rows)
            )
            assert demand is not None
            merged[id(root.var)] = ContributionRoot(root.var, demand.rows)
    return list(merged.values())


def find_contribution_roots(
    jaxpr: Jaxpr,
    bound_eqns: list[BoundEqn],
    state: TraceState,
) -> list[ContributionRoot]:
    """Return additively separable roots for the scalar outputs of ``jaxpr``.

    Unsupported primitives deliberately become roots at their demanded outputs.  This
    fallback is conservative: later halo extraction uses the forward dependency rows of
    that output, so it cannot lose a data dependency merely because decomposition stops.
    """
    demand_of: dict[int, ContributionDemand | None] = {}
    for outvar in jaxpr.outvars:
        if state.is_scalar(outvar):
            demand_of[id(outvar)] = ContributionDemand(np.array([0], dtype=np.int64))

    if not demand_of:
        raise ValueError(
            "Contribution decomposition requires a scalar functional output."
        )

    roots: list[ContributionRoot] = []
    for eqn, handler, is_active, _needs_concrete in reversed(bound_eqns):
        if not is_active:
            continue

        out_demands = [demand_of.pop(id(outvar), None) for outvar in eqn.outvars]
        if not any(demand is not None for demand in out_demands):
            continue

        result = handler.propagate_contribution_demand(eqn, state, out_demands)
        if not result.valid:
            roots.extend(
                ContributionRoot(outvar, demand.rows)
                for outvar, demand in zip(eqn.outvars, out_demands)
                if demand is not None
            )
            continue

        roots.extend(result.roots)
        for invar, demand in zip(eqn.invars, result.in_demands):
            if demand is None or state.is_inactive(invar):
                continue
            demand_of[id(invar)] = merge_demands(demand_of.get(id(invar)), demand)

    for var in (*jaxpr.invars, *jaxpr.constvars):
        demand = demand_of.get(id(var))
        if demand is not None and not state.is_inactive(var):
            roots.append(ContributionRoot(var, demand.rows))

    return merge_contribution_roots(roots)


def build_partition_plan(
    jaxpr: Jaxpr,
    bound_eqns: list[BoundEqn],
    state: TraceState,
    part_map: NDArray[np.integer],
) -> EnergyPartitionPlan:
    """Build contribution ownership, halo, and local DOF layouts."""
    partition = _validate_partition_map(part_map, state.n_dofs)

    roots = find_contribution_roots(
        jaxpr=jaxpr,
        bound_eqns=bound_eqns,
        state=state,
    )

    n_ranks = int(partition.max()) + 1 if partition.size else 1

    # rank -> root index -> original contribution rows
    rows_by_rank: list[dict[int, NDArray[np.int64]]] = [{} for _ in range(n_ranks)]

    for root_index, root in enumerate(roots):
        root_rows = np.asarray(root.rows, dtype=np.int64)
        full_dep = state.get(root.var).dep

        if np.any(root_rows < 0) or np.any(root_rows >= full_dep.shape[0]):
            raise ValueError(
                f"Contribution root {root_index} contains out-of-range rows."
            )

        selected_dep = full_dep[root_rows].tocsr()
        row_owners = _dependency_row_owners(
            selected_dep,
            partition,
            static_owner=0,
        )

        for rank in range(n_ranks):
            rank_mask = row_owners == rank

            if np.any(rank_mask):
                rows_by_rank[rank][root_index] = root_rows[rank_mask]

    rank_plans: list[RankPartitionPlan] = []

    for rank in range(n_ranks):
        contribution_rows = rows_by_rank[rank]
        required_chunks: list[NDArray[np.int64]] = []

        for root_index, rows in contribution_rows.items():
            root = roots[root_index]
            dep = state.get(root.var).dep[rows].tocsr()

            if dep.nnz:
                required_chunks.append(np.asarray(dep.indices, dtype=np.int64))

        if required_chunks:
            required_dofs = np.unique(np.concatenate(required_chunks)).astype(np.int64)
        else:
            required_dofs = np.empty(0, dtype=np.int64)

        owned_dofs = np.flatnonzero(partition == rank).astype(np.int64)

        required_owned_dofs = required_dofs[partition[required_dofs] == rank]

        ghost_dofs = required_dofs[partition[required_dofs] != rank]

        # Keep all partition-owned DOFs in the distributed vector.
        local_to_global = np.concatenate([owned_dofs, ghost_dofs]).astype(np.int64)

        global_to_local = np.full(
            state.n_dofs,
            -1,
            dtype=np.int64,
        )
        global_to_local[local_to_global] = np.arange(
            local_to_global.size,
            dtype=np.int64,
        )

        rank_plans.append(
            RankPartitionPlan(
                rank=rank,
                contribution_rows=contribution_rows,
                required_dofs=required_dofs,
                owned_dofs=owned_dofs,
                required_owned_dofs=required_owned_dofs,
                ghost_dofs=ghost_dofs,
                local_to_global=local_to_global,
                global_to_local=global_to_local,
            )
        )

    result = EnergyPartitionPlan(
        part_map=partition,
        roots=tuple(roots),
        ranks=tuple(rank_plans),
    )

    _validate_energy_partition_plan(result)
    return result


def _validate_energy_partition_plan(
    plan: EnergyPartitionPlan,
) -> None:
    # Every contribution row must be assigned exactly once.
    for root_index, root in enumerate(plan.roots):
        assigned_chunks = [
            rank.contribution_rows[root_index]
            for rank in plan.ranks
            if root_index in rank.contribution_rows
        ]

        assigned = (
            np.concatenate(assigned_chunks)
            if assigned_chunks
            else np.empty(0, dtype=np.int64)
        )

        expected = np.asarray(root.rows, dtype=np.int64)

        if assigned.size != np.unique(assigned).size:
            raise AssertionError(
                f"Contribution root {root_index} has duplicate assignments."
            )

        if not np.array_equal(
            np.sort(assigned),
            np.sort(expected),
        ):
            raise AssertionError(
                f"Contribution root {root_index} was not partitioned completely."
            )

    for rank in plan.ranks:
        if np.intersect1d(
            rank.owned_dofs,
            rank.ghost_dofs,
        ).size:
            raise AssertionError(
                f"Rank {rank.rank} has a DOF classified as both owned and ghost."
            )

        if np.any(plan.part_map[rank.ghost_dofs] == rank.rank):
            raise AssertionError(
                f"Rank {rank.rank} contains locally owned DOFs in its ghost set."
            )

        expected_local = np.arange(
            rank.local_to_global.size,
            dtype=np.int64,
        )

        actual_local = rank.global_to_local[rank.local_to_global]

        if not np.array_equal(actual_local, expected_local):
            raise AssertionError(
                f"Rank {rank.rank} has an invalid global-to-local map."
            )


def seed_rank_demands(
    partition_plan: EnergyPartitionPlan,
    rank: int,
) -> dict[int, ContributionDemand]:
    rank_plan = partition_plan.ranks[rank]
    demand_of: dict[int, ContributionDemand] = {}

    for root_index, rows in rank_plan.contribution_rows.items():
        root = partition_plan.roots[root_index]
        existing = demand_of.get(id(root.var))
        demand_of[id(root.var)] = merge_demands(existing, ContributionDemand(rows))

    return demand_of


def _propagate_demands_backward(
    bound_eqns: list[BoundEqn],
    state: TraceState,
    seed_demands: dict[int, ContributionDemand],
    live_eqn_ids: set[int] | None = None,
) -> dict[int, ContributionDemand]:
    """Implementation shared by rank and nested-JAXPR liveness passes."""
    pending = dict(seed_demands)
    all_demands: dict[int, ContributionDemand] = dict(pending)

    def add_demand(var: Any, demand: ContributionDemand) -> None:
        var_id = id(var)
        previous_pending = pending.get(var_id)
        previous_all = all_demands.get(var_id)
        merged_pending = merge_demands(previous_pending, demand)
        pending[var_id] = merged_pending
        # While a demand is pending, both maps normally reference the same
        # object.  Reuse that one union instead of canonicalizing twice.
        all_demands[var_id] = (
            merged_pending
            if previous_all is previous_pending
            else merge_demands(previous_all, demand)
        )

    for eqn, handler, _is_active, _needs_concrete in reversed(bound_eqns):
        out_demands = [pending.pop(id(outvar), None) for outvar in eqn.outvars]

        if not any(demand is not None for demand in out_demands):
            continue

        if len(out_demands) != len(eqn.outvars):
            raise RuntimeError(
                f"{eqn.primitive.name} has an invalid output demand list."
            )
        for var, demand in zip(eqn.outvars, out_demands):
            if demand is not None:
                size = int(np.prod(_get_shape(var)))
                if np.any((demand.rows < 0) | (demand.rows >= size)):
                    raise ValueError(
                        f"{eqn.primitive.name} received out-of-range output rows."
                    )

        if live_eqn_ids is not None:
            live_eqn_ids.add(id(eqn))
        in_demands = handler.propagate_liveness_demand(eqn, state, out_demands)

        if len(in_demands) != len(eqn.invars):
            raise RuntimeError(
                f"{eqn.primitive.name} returned "
                f"{len(in_demands)} input demands for "
                f"{len(eqn.invars)} inputs."
            )

        out_count = sum(
            len(d.rows)
            for d in out_demands
            if d is not None
        )

        for i, (invar, demand) in enumerate(zip(eqn.invars, in_demands)):
            if demand is None:
                continue

            total = int(np.prod(_get_shape(invar)))
            demanded = len(demand.rows)

            if total >= 100 and demanded / total > 0.9:
                print(
                    eqn.primitive.name,
                    {
                        "index": i,
                        "shape": _get_shape(invar),
                        "demanded": demanded,
                        "total": total,
                        "fraction": demanded / total,
                        "out_count": out_count,
                    },
                )

        for invar, demand in zip(eqn.invars, in_demands):
            if demand is None:
                continue

            size = int(np.prod(_get_shape(invar)))
            if np.any((demand.rows < 0) | (demand.rows >= size)):
                raise ValueError(
                    f"{eqn.primitive.name} produced out-of-range input rows."
                )
            add_demand(invar, demand)

    return all_demands


def propagate_demands_backward(
    bound_eqns: list[BoundEqn],
    state: TraceState,
    seed_demands: dict[int, ContributionDemand],
) -> dict[int, ContributionDemand]:
    """Propagate arbitrary demanded entries backwards through a bound JAXPR."""
    return _propagate_demands_backward(bound_eqns, state, seed_demands)


def build_rank_liveness(
    bound_eqns: list[BoundEqn],
    state: TraceState,
    partition_plan: EnergyPartitionPlan,
    rank: int,
) -> RankLivenessPlan:
    live_eqn_ids: set[int] = set()
    all_demands = _propagate_demands_backward(
        bound_eqns,
        state,
        seed_rank_demands(partition_plan, rank),
        live_eqn_ids,
    )

    return RankLivenessPlan(
        rank=rank,
        all_demands=all_demands,
        live_eqn_ids=frozenset(live_eqn_ids),
    )
