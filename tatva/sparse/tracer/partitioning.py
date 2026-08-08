"""Backward decomposition of scalar JAX functionals into contribution roots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import AbstractValue, ShapedArray
from jax.extend.core import (
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
)
from numpy.typing import NDArray

from tatva.sparse.tracer.common import _get_shape
from tatva.sparse.tracer.registry import TR

if TYPE_CHECKING:
    from tatva.sparse.tracer.base import EnergyTrace
    from tatva.sparse.tracer.handlers import PrimitiveHandler
    from tatva.sparse.tracer.state import BoundEqn, TraceState


class RowSet(Protocol):
    def __len__(self) -> int: ...
    def to_array(self) -> NDArray: ...
    def is_full(self, total_size: int) -> bool: ...
    def localize(self, global_rows: NDArray) -> NDArray:
        """Map original flat rows to compact local rows."""


@dataclass(frozen=True)
class ArrayRows:
    values: NDArray[np.int64]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.int64)

        if values.ndim != 1:
            raise ValueError("Rows must be one-dimensional")
        if len(values) > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError("Rows must be sorted and unique")

        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return self.values.size

    def to_array(self) -> NDArray[np.int64]:
        return self.values

    # Compatibility with the older ndarray-backed demand API.  These do not
    # change the stored representation: callers still receive a RowSet.
    @property
    def size(self) -> int:
        return self.values.size

    def __array__(self, dtype=None, copy=None) -> NDArray[np.int64]:
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result

    def __getitem__(self, key: Any) -> Any:
        return self.values[key]

    def is_full(self, total_size: int) -> bool:
        return self.values.size == total_size and (
            total_size == 0
            or (self.values[0] == 0 and self.values[-1] == total_size - 1)
        )

    def localize(self, global_rows: NDArray[np.int64]) -> NDArray[np.int64]:
        global_rows = np.asarray(global_rows, dtype=np.int64)
        if self.values.size == 0:
            if global_rows.size:
                raise ValueError("Rows are not present in the local layout.")
            return global_rows
        local_rows = np.searchsorted(self.values, global_rows)

        valid = local_rows < self.values.size
        valid &= (
            self.values[np.minimum(local_rows, max(self.values.size - 1, 0))]
            == global_rows
        )
        if not np.all(valid):
            missing = global_rows[~valid]
            raise ValueError(
                f"Rows are not present in the local layout: {missing[:10]}"
            )

        return local_rows


@dataclass(frozen=True)
class AllRows:
    """Every flat entry of a variable, represented without an index array."""

    size: int

    def __len__(self) -> int:
        return self.size

    def __array__(self, dtype=None, copy=None) -> NDArray[np.int64]:
        result = np.arange(self.size, dtype=dtype or np.int64)
        return result.copy() if copy else result

    def to_array(self) -> NDArray[np.int64]:
        return np.arange(self.size, dtype=np.int64)

    def is_full(self, total_size: int) -> bool:
        return self.size == total_size

    def localize(self, global_rows: NDArray[np.int64]) -> NDArray[np.int64]:
        global_rows = np.asarray(global_rows, dtype=np.int64)
        if np.any(global_rows < 0) or np.any(global_rows >= self.size):
            raise ValueError("Rows are not present in the local layout.")
        return global_rows

    def __getitem__(self, key):
        return key


@dataclass(frozen=True)
class RangeRows:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError("RangeRows must be a non-negative half-open range.")

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def size(self) -> int:
        return len(self)

    def __array__(self, dtype=None, copy=None) -> NDArray[np.int64]:
        result = np.arange(self.start, self.stop, dtype=dtype or np.int64)
        return result.copy() if copy else result

    def to_array(self) -> NDArray[np.int64]:
        return np.arange(self.start, self.stop, dtype=np.int64)

    def is_full(self, total_size: int) -> bool:
        return self.start == 0 and self.stop == total_size

    def localize(self, global_rows: NDArray[np.int64]) -> NDArray[np.int64]:
        global_rows = np.asarray(global_rows, dtype=np.int64)
        if np.any(global_rows < self.start) or np.any(global_rows >= self.stop):
            raise ValueError("Rows are not present in the local layout.")
        return global_rows - self.start


# @dataclass(frozen=True)
# class RangeRows:
#     """A contiguous half-open range of flat entries without materialization."""
#
#     start: int
#     stop: int
#
#     def __post_init__(self) -> None:
#         if self.start < 0 or self.stop < self.start:
#             raise ValueError("RangeRows must be a non-negative half-open range.")
#
#     def __len__(self) -> int:
#         return self.stop - self.start
#
#     @property
#     def size(self) -> int:
#         return self.stop - self.start
#
#     def __array__(self, dtype=None) -> NDArray[np.int64]:
#         return np.arange(self.start, self.stop, dtype=dtype or np.int64)
#
#     def __getitem__(self, key):
#         return key + self.start


type RowSelection = ArrayRows | AllRows | RangeRows


@dataclass(frozen=True)
class TensorSubset:
    """Exact subset of a tensor's C-order index space.

    ``Points`` is the universal representation and deliberately keeps the existing
    compact ``RowSet`` implementation.  The other variants retain tensor
    structure, but may always degrade to ``Points`` without changing meaning.
    """

    shape: tuple[int, ...]

    @property
    def local_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def to_rows(self) -> RowSet:
        raise NotImplementedError

    @property
    def size(self) -> int:
        return len(self.to_rows())

    @classmethod
    def infer_from_rows(cls, shape: tuple[int, ...], rows: RowSet) -> TensorSubset:
        """Compatibility inference for legacy row-oriented handlers.

        New handlers must construct ``Full``, ``AxisProduct``, or ``Points``
        directly.  This is intentionally only a migration/debugging utility.
        """
        shape = tuple(shape)
        total = int(np.prod(shape, dtype=np.int64))
        if rows.is_full(total):
            return Full(shape)
        if not shape:
            return Points(shape, rows)
        materialized = rows.to_array()
        if shape and materialized.size:
            coordinates = np.unravel_index(materialized, shape)
            axes = tuple(
                ArrayRows(np.unique(coordinate).astype(np.int64))
                for coordinate in coordinates
            )
            if (
                int(np.prod([len(axis) for axis in axes], dtype=np.int64))
                == materialized.size
            ):
                candidate = AxisProduct(shape, axes)
                if np.array_equal(candidate.to_rows().to_array(), materialized):
                    return candidate
        return Points(shape, rows)


@dataclass(frozen=True)
class Full(TensorSubset):
    @property
    def local_shape(self) -> tuple[int, ...]:
        return self.shape

    def to_rows(self) -> RowSet:
        return AllRows(int(np.prod(self.shape, dtype=np.int64)))


@dataclass(frozen=True)
class Points(TensorSubset):
    """Arbitrary correlated positions, represented by canonical flat rows."""

    rows: RowSet

    @property
    def local_shape(self) -> tuple[int, ...]:
        return () if not self.shape else (len(self.rows),)

    def to_rows(self) -> RowSet:
        return self.rows


@dataclass(frozen=True)
class AxisProduct(TensorSubset):
    """Independent selections along every axis of a tensor."""

    axes: tuple[RowSet, ...]

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.shape):
            raise ValueError("AxisProduct needs one selection per tensor axis")
        for axis, extent in zip(self.axes, self.shape):
            if not axis.is_full(extent) and np.any(axis.to_array() >= extent):
                raise ValueError("Axis selection is outside tensor bounds")

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(len(axis) for axis in self.axes)

    def to_rows(self) -> RowSet:
        if not self.shape:
            return AllRows(1)
        coordinates = np.meshgrid(
            *(axis.to_array() for axis in self.axes), indexing="ij"
        )
        return ArrayRows(
            np.ravel_multi_index(
                tuple(coord.reshape(-1) for coord in coordinates), self.shape
            )
        )


def union_tensor_subsets(left: TensorSubset, right: TensorSubset) -> TensorSubset:
    """Return the strongest exact representation of an index-space union."""
    if left.shape != right.shape:
        raise ValueError("Cannot union subsets of different tensor shapes")
    if isinstance(left, Full) or isinstance(right, Full):
        return Full(left.shape)
    if isinstance(left, AxisProduct) and isinstance(right, AxisProduct):
        different = [
            i
            for i, (a, b) in enumerate(zip(left.axes, right.axes))
            if not np.array_equal(a.to_array(), b.to_array())
        ]
        if len(different) <= 1:
            axes = list(left.axes)
            if different:
                axis = different[0]
                axes[axis] = ArrayRows(
                    np.union1d(left.axes[axis].to_array(), right.axes[axis].to_array())
                )
            return AxisProduct(left.shape, tuple(axes))
    rows = ArrayRows(np.union1d(left.to_rows().to_array(), right.to_rows().to_array()))
    return TensorSubset.infer_from_rows(left.shape, rows)


@dataclass(frozen=True)
class VarLayout:
    """The finalized local representation of one original variable.

    ``original_var`` and ``subset`` are the only stored state.  All row and
    aval metadata is derived, keeping the tensor-index subset authoritative.
    """

    original_var: Var
    subset: TensorSubset

    @property
    def original_shape(self) -> tuple[int, ...]:
        return self.subset.shape

    @property
    def rows(self) -> RowSet:
        return self.subset.to_rows()

    @property
    def local_aval(self) -> AbstractValue:
        return (
            self.original_var.aval
            if not self.original_shape
            else replace_aval_shape(self.original_var.aval, self.subset.local_shape)
        )

    @property
    def original_size(self) -> int:
        return aval_size(self.original_var.aval)

    @property
    def is_full(self) -> bool:
        return isinstance(self.subset, Full)

    @property
    def local_size(self) -> int:
        return len(self.rows)

    @property
    def structured_local_shape(self) -> tuple[int, ...]:
        """The semantic compact shape represented by the finalized subset."""
        return self.subset.local_shape


@dataclass
class BackwardResult:
    in_demands: tuple[ContributionDemand | None, ...]
    aux: Any = None
    subplans: tuple[JaxprPlan, ...] = ()


@dataclass
class EqnPlan:
    handler: PrimitiveHandler
    live_output_mask: tuple[bool, ...]
    aux: Any = None
    subplans: tuple[JaxprPlan, ...] = ()


@dataclass
class JaxprPlan[**P]:
    original_jaxpr: Jaxpr
    requested_outputs: tuple[Var, ...]

    eqn_plans: tuple[EqnPlan | None, ...]
    """aligned one-to-one with the original jaxpr eqns"""
    layouts: dict[Var, VarLayout]
    """contains exactly one finalized layout per live orig var"""
    kept_constvar_indices: tuple[int, ...]
    kept_invar_indices: tuple[int, ...]
    constvar_demands: tuple[ContributionDemand | None, ...]
    invar_demands: tuple[ContributionDemand | None, ...]


def aval_size(aval: AbstractValue) -> int:
    """Return the total number of flat entries in an abstract value."""
    shape = getattr(aval, "shape", ())
    return int(np.prod(shape, dtype=np.int64))


def replace_aval_shape(
    aval: AbstractValue,
    shape: tuple[int, ...],
) -> AbstractValue:
    # Pin this helper to your supported JAX version.
    #
    # Prefer aval.update(shape=shape) if supported by the concrete aval type.
    update = getattr(aval, "update", None)
    if update is not None:
        return update(shape=shape)

    return ShapedArray(
        shape,
        aval.dtype,  # ty: ignore[unresolved-attribute]
        weak_type=getattr(aval, "weak_type", False),
    )


def finalize_layout(
    var: Var,
    demand: ContributionDemand,
) -> VarLayout:
    subset = demand.subset
    if subset is None:
        raise ValueError("Cannot finalize an unshaped legacy demand")
    if subset.shape != tuple(getattr(var.aval, "shape", ())):
        raise ValueError("Demand shape does not match its variable")
    if subset.size == 0:
        raise ValueError("A live layout cannot contain zero rows.")
    if not _demand_in_bounds(demand, aval_size(var.aval)):
        raise ValueError(
            f"Demand outside variable bounds: {var}, size={aval_size(var.aval)}"
        )
    return VarLayout(original_var=var, subset=subset)


@dataclass(frozen=True, init=False)
class ContributionDemand:
    """An exact tensor subset, plus a temporary unshaped legacy bridge."""

    subset: TensorSubset | None
    _legacy_rows: RowSet | None

    def __init__(
        self,
        subset_or_rows: TensorSubset | RowSet | NDArray,
        subset: TensorSubset | None = None,
    ):
        # The two-argument form is retained only while row-oriented handlers
        # migrate.  Once shaped, ``subset`` is the sole source of truth.
        if subset is not None:
            subset_or_rows = subset
        if isinstance(subset_or_rows, TensorSubset):
            object.__setattr__(self, "subset", subset_or_rows)
            object.__setattr__(self, "_legacy_rows", None)
        else:
            rows = subset_or_rows
            if not isinstance(rows, (ArrayRows, AllRows, RangeRows)):
                rows = ArrayRows(np.asarray(rows, dtype=np.int64))
            object.__setattr__(self, "subset", None)
            object.__setattr__(self, "_legacy_rows", rows)

    @property
    def rows(self) -> RowSet:
        if self.subset is not None:
            return self.subset.to_rows()
        assert self._legacy_rows is not None
        return self._legacy_rows

    @property
    def shape(self) -> tuple[int, ...] | None:
        return None if self.subset is None else self.subset.shape

    def __len__(self) -> int:
        return len(self.rows)

    def is_all_rows(self) -> bool:
        return isinstance(self.rows, AllRows)

    def is_range_rows(self) -> bool:
        return isinstance(self.rows, RangeRows)

def demand_rows(demand: ContributionDemand) -> NDArray[np.int64]:
    """Materialize rows only at handlers requiring arbitrary index routing."""
    rows = demand.rows
    if isinstance(rows, AllRows):
        return np.arange(rows.size, dtype=np.int64)
    if isinstance(rows, RangeRows):
        return np.arange(rows.start, rows.stop, dtype=np.int64)
    return rows.to_array()


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
class EnergyPartitionPlan[**P]:
    part_map: NDArray[np.int64]
    roots: tuple[ContributionRoot, ...]
    ranks: tuple[RankPartitionPlan, ...]
    energy_trace: EnergyTrace

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
        if rows.size > 1 and rows[-1] - rows[0] + 1 == rows.size:
            return ContributionDemand(RangeRows(int(rows[0]), int(rows[-1]) + 1))
        return ContributionDemand(ArrayRows(rows))
    return ContributionDemand(ArrayRows(np.unique(rows)))


def seed_demand(var: Var, rows: NDArray[np.integer]) -> ContributionDemand:
    """Build a shaped planner seed from externally supplied contribution rows."""
    legacy = _demand(rows)
    if legacy is None:
        raise ValueError("A planner seed cannot be empty")
    return ContributionDemand(
        TensorSubset.infer_from_rows(_get_shape(var), legacy.rows)
    )


def _demand_in_bounds(demand: ContributionDemand, size: int) -> bool:
    """Check bounds without materializing compact full/range demands."""
    if isinstance(demand.rows, AllRows):
        return demand.rows.size == size
    if isinstance(demand.rows, RangeRows):
        return demand.rows.stop <= size
    rows = demand.rows.to_array()
    return not np.any((rows < 0) | (rows >= size))


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

    # TODO: Potential performance bottleneck!
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
def merge_demands(left: ContributionDemand, right: None) -> ContributionDemand: ...
@overload
def merge_demands(
    left: ContributionDemand | None, right: ContributionDemand | None
) -> ContributionDemand | None: ...
def merge_demands(
    left: ContributionDemand | None, right: ContributionDemand | None
) -> ContributionDemand | None:
    """Union demands, preserving an exact shared tensor-index representation."""
    if left is None:
        return right
    if right is None:
        return left
    if (
        left.subset is not None
        and right.subset is not None
        and left.subset.shape == right.subset.shape
    ):
        subset = union_tensor_subsets(left.subset, right.subset)
        return ContributionDemand(subset.to_rows(), subset)
    if isinstance(left.rows, AllRows):
        return left
    if isinstance(right.rows, AllRows):
        return right
    if (
        isinstance(left.rows, RangeRows)
        and isinstance(right.rows, RangeRows)
        and right.rows.start <= left.rows.stop
        and left.rows.start <= right.rows.stop
    ):
        return ContributionDemand(
            RangeRows(
                min(left.rows.start, right.rows.start),
                max(left.rows.stop, right.rows.stop),
            )
        )
    if left is right or np.array_equal(demand_rows(left), demand_rows(right)):
        return left
    return _demand(np.union1d(demand_rows(left), demand_rows(right)).astype(np.int64))


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
                ContributionDemand(ArrayRows(previous.rows)),
                ContributionDemand(ArrayRows(root.rows)),
            )
            assert demand is not None
            merged[id(root.var)] = ContributionRoot(root.var, demand_rows(demand))
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
            demand_of[id(outvar)] = ContributionDemand(
                ArrayRows(np.array([0], dtype=np.int64))
            )

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
                ContributionRoot(outvar, demand_rows(demand))
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
            roots.append(ContributionRoot(var, demand_rows(demand)))

    return merge_contribution_roots(roots)


def build_partition_plan[**P](
    energy_trace: EnergyTrace[P],
    part_map: NDArray[np.integer],
) -> EnergyPartitionPlan[P]:
    """Build contribution ownership, halo, and local DOF layouts."""
    state = energy_trace.state
    partition = _validate_partition_map(part_map, state.n_dofs)

    roots = find_contribution_roots(
        jaxpr=energy_trace.concrete_jaxpr.jaxpr,
        bound_eqns=energy_trace.plan.bound_eqns,
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
        energy_trace=energy_trace,
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
        demand = (
            seed_demand(root.var, rows)
            if isinstance(root.var, Var)
            else ContributionDemand(ArrayRows(rows))
        )
        demand_of[id(root.var)] = merge_demands(existing, demand)

    return demand_of


def _validate_demand_for_var(var: Any, demand: ContributionDemand) -> ContributionDemand:
    """Enforce the production invariant: every demand owns its tensor shape."""
    if demand.subset is None:
        raise NotImplementedError(
            f"Unshaped liveness demand for {var}; handlers must return a TensorSubset"
        )
    if demand.subset.shape != _get_shape(var):
        raise ValueError("Liveness subset shape does not match its variable")
    return demand


def rank_seed_demands(
    partition_plan: EnergyPartitionPlan,
    rank: int,
) -> tuple[dict[Var, ContributionDemand], tuple[Var, ...]]:
    """Turn a rank's contribution assignments into planner seed variables."""
    rank_plan = partition_plan.ranks[rank]
    seeds: dict[Var, ContributionDemand] = {}
    requested: list[Var] = []
    for root_index, rows in rank_plan.contribution_rows.items():
        var = partition_plan.roots[root_index].var
        if not isinstance(var, Var):
            raise TypeError("Contribution roots for local JAXPRs must be Vars")
        demand = seed_demand(var, rows)
        seeds[var] = merge_demands(seeds.get(var), demand)
        if var not in requested:
            requested.append(var)
    return seeds, tuple(requested)


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
        demand = _validate_demand_for_var(var, demand)
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
        out_demands = [
            None
            if (demand := pending.pop(id(outvar), None)) is None
            else _validate_demand_for_var(outvar, demand)
            for outvar in eqn.outvars
        ]

        if not any(demand is not None for demand in out_demands):
            continue

        if len(out_demands) != len(eqn.outvars):
            raise RuntimeError(
                f"{eqn.primitive.name} has an invalid output demand list."
            )
        for var, demand in zip(eqn.outvars, out_demands):
            if demand is not None:
                all_demands[id(var)] = demand
                size = int(np.prod(_get_shape(var)))
                if not _demand_in_bounds(demand, size):
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

        for invar, demand in zip(eqn.invars, in_demands):
            if demand is None:
                continue

            size = int(np.prod(_get_shape(invar)))
            if not _demand_in_bounds(demand, size):
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


def rank_liveness_from_plan(rank: int, plan: JaxprPlan) -> RankLivenessPlan:
    """Project the authoritative local JAXPR plan into rank-liveness metadata."""
    return RankLivenessPlan(
        rank=rank,
        all_demands={
            id(var): ContributionDemand(layout.subset)
            for var, layout in plan.layouts.items()
        },
        live_eqn_ids=frozenset(
            id(eqn)
            for eqn, eqn_plan in zip(plan.original_jaxpr.eqns, plan.eqn_plans)
            if eqn_plan is not None
        ),
    )


def build_rank_liveness(
    jaxpr: Jaxpr,
    state: TraceState,
    partition_plan: EnergyPartitionPlan,
    rank: int,
) -> RankLivenessPlan:
    """Build rank liveness from the same finalized plan used for execution."""
    seeds, requested_outputs = rank_seed_demands(partition_plan, rank)
    plan = plan_local_jaxpr(
        jaxpr=jaxpr,
        state=state,
        seed_demands=seeds,
        requested_outputs=requested_outputs,
    )
    return rank_liveness_from_plan(rank, plan)


def plan_local_jaxpr(
    jaxpr: Jaxpr,
    state: TraceState,
    seed_demands: Mapping[Var, ContributionDemand],
    requested_outputs: Sequence[Var],
) -> JaxprPlan:
    if jaxpr.effects:
        raise NotImplementedError(
            "First implementation supports effect-free JAXPRs only"
        )

    def _finalize_inputs(
        variables: Sequence[Var],
        pending: dict[Var, ContributionDemand],
        layouts: dict[Var, VarLayout],
    ) -> tuple[ContributionDemand | None, ...]:
        demands: list[ContributionDemand | None] = []

        for var in variables:
            demand = pending.pop(var, None)
            if demand is not None:
                demand = _validate_demand_for_var(var, demand)
            demands.append(demand)

            if demand is not None:
                layouts[var] = finalize_layout(var, demand)

        return tuple(demands)

    def _merge_pending(
        pending: dict[Var, ContributionDemand],
        var: Var,
        demand: ContributionDemand,
    ) -> None:
        demand = _validate_demand_for_var(var, demand)
        old = pending.get(var)
        pending[var] = demand if old is None else merge_demands(old, demand)

    def _validate_literal_output(
        literal: Literal,
        demand: ContributionDemand,
    ) -> None:
        total = aval_size(literal.aval)
        # rows = normalize_row_set(demand.rows)
        rows = demand.rows

        if not rows.is_full(total):
            raise NotImplementedError("Partial literal JAXPR outputs are not supported")

    def _validate_literal_input(
        literal: Literal,
        demand: ContributionDemand,
    ) -> None:
        if not _demand_in_bounds(demand, aval_size(literal.aval)):
            raise ValueError(f"Demand outside literal bounds: {literal}")

    pending: dict[Var, ContributionDemand] = {}
    layouts: dict[Var, VarLayout] = {}
    eqn_plans: list[EqnPlan | None] = [None] * len(jaxpr.eqns)

    known_vars = set(jaxpr.constvars) | set(jaxpr.invars)
    known_vars.update(outvar for eqn in jaxpr.eqns for outvar in eqn.outvars)
    for var, demand in seed_demands.items():
        if var not in known_vars:
            raise ValueError(f"Seed variable is not part of this JAXPR: {var}")
        _merge_pending(pending, var, demand)

    requested_outputs = tuple(requested_outputs)
    for var in requested_outputs:
        if var not in seed_demands:
            raise ValueError(f"Requested output has no seed demand: {var}")

    for eqn_index in range(len(jaxpr.eqns) - 1, -1, -1):
        eqn = jaxpr.eqns[eqn_index]

        if eqn.effects:
            raise NotImplementedError(
                f"Effectful equation cannot currently be rewritten: "
                f"{eqn.primitive.name}"
            )

        out_demands = tuple(
            None
            if (demand := pending.pop(outvar, None)) is None
            else _validate_demand_for_var(outvar, demand)
            for outvar in eqn.outvars
        )

        if not any(demand is not None for demand in out_demands):
            continue

        # At this point every consumer of each output has been visited.
        # Therefore each non-None output demand is final.
        for outvar, demand in zip(eqn.outvars, out_demands):
            if demand is None:
                continue

            if outvar in layouts:
                raise AssertionError(f"Layout finalized twice for {outvar}")

            layouts[outvar] = finalize_layout(outvar, demand)

        handler = TR.get(eqn.primitive.name)

        result = handler.plan_backward(
            eqn=eqn,
            state=state,
            out_demands=out_demands,
        )

        if len(result.in_demands) != len(eqn.invars):
            raise ValueError(
                f"{eqn.primitive.name} returned "
                f"{len(result.in_demands)} input demands for "
                f"{len(eqn.invars)} inputs"
            )

        for invar, demand in zip(eqn.invars, result.in_demands):
            if demand is None:
                continue

            if isinstance(invar, Literal):
                # Literals remain embedded in the generated JAXPR, so
                # they do not acquire a compact VarLayout or enter the
                # pending map.  A liveness handler can still report a
                # demand for one (notably a scalar being broadcast); it
                # means the literal is numerically required, not that it
                # needs runtime storage.
                _validate_literal_input(invar, demand)
                continue

            if invar in layouts:
                raise AssertionError(
                    "Demand reached a variable whose layout was already "
                    "finalized. The traversal is not reverse-topological "
                    "or a handler propagated to the wrong variable."
                )

            _merge_pending(pending, invar, demand)

        eqn_plans[eqn_index] = EqnPlan(
            handler=handler,
            live_output_mask=tuple(demand is not None for demand in out_demands),
            aux=result.aux,
            subplans=result.subplans,
        )

    constvar_demands = _finalize_inputs(
        jaxpr.constvars,
        pending,
        layouts,
    )
    invar_demands = _finalize_inputs(
        jaxpr.invars,
        pending,
        layouts,
    )

    if pending:
        dangling = list(pending)[:10]
        raise ValueError(
            f"Demands remain for variables without producers or inputs: {dangling}"
        )

    kept_constvars = tuple(
        i for i, demand in enumerate(constvar_demands) if demand is not None
    )
    kept_invars = tuple(
        i for i, demand in enumerate(invar_demands) if demand is not None
    )
    return JaxprPlan(
        original_jaxpr=jaxpr,
        requested_outputs=requested_outputs,
        eqn_plans=tuple(eqn_plans),
        layouts=layouts,
        kept_constvar_indices=kept_constvars,
        kept_invar_indices=kept_invars,
        constvar_demands=constvar_demands,
        invar_demands=invar_demands,
    )


@dataclass
class InputSpec:
    original_index: int
    layout: VarLayout


@dataclass
class OutputSpec:
    original_var: Var
    layout: VarLayout


@dataclass
class LocalProgram[**P]:
    fn: Callable[P, tuple[Any, ...]]
    input_specs: tuple[InputSpec, ...]
    output_specs: tuple[OutputSpec, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> tuple[Any, ...]:
        return self.fn(*args, **kwargs)


class LocalJaxprInterpreter:
    """Execute a compact plan using ordinary JAX operations.

    This deliberately is an interpreter for *values*, not a JAXPR builder:
    JAX traces this callable when a local JAXPR is wanted.
    """

    def __init__(self, plan: JaxprPlan, original_consts: Sequence[Any]):
        self.plan = plan
        self.original_consts = tuple(original_consts)

    def take_rows_from_local(
        self, value: Any, layout: VarLayout, global_rows: NDArray[np.int64]
    ):
        """Compatibility bridge from a local tensor to requested global rows."""
        rows = np.asarray(global_rows, dtype=np.int64)
        if not layout.original_shape:
            return value
        # Every structured local tensor has C-order storage corresponding to
        # ``layout.rows``; flattening here is only the compatibility bridge for
        # primitives that still request arbitrary points.
        value = jnp.ravel(value)
        if np.array_equal(rows, layout.rows.to_array()):
            return value
        local_rows = layout.rows.localize(rows)
        if local_rows.size and np.array_equal(
            local_rows, np.arange(local_rows[0], local_rows[0] + local_rows.size)
        ):
            return value[local_rows[0] : local_rows[0] + local_rows.size]
        return jnp.take(value, jnp.asarray(local_rows), axis=0)

    @staticmethod
    def normalize_local_result(value: Any, layout: VarLayout) -> Any:
        """Return exactly ``layout.subset.local_shape`` or fail loudly."""
        expected = layout.subset.local_shape
        actual = tuple(getattr(value, "shape", ()))
        if actual == expected:
            return value
        if int(np.size(value)) != layout.local_size:
            raise ValueError(
                f"Local result has shape {actual}, expected {expected} for {layout.original_var}"
            )
        return jnp.reshape(value, expected)

    def run_subplan(
        self,
        subplan: JaxprPlan,
        sub_consts: Sequence[Any],
        parent_values,
        parent_layouts,
    ):
        args = []
        for child_index in subplan.kept_invar_indices:
            value = parent_values[child_index]
            layout = parent_layouts[child_index]
            child = subplan.layouts[subplan.original_jaxpr.invars[child_index]]
            if value is None or layout is None:
                raise NotImplementedError(
                    "Nested local plan needs an eliminated parent operand"
                )
            localized = self.take_rows_from_local(value, layout, child.rows.to_array())
            args.append(self.normalize_local_result(localized, child))
        return LocalJaxprInterpreter(subplan, sub_consts).make_function()(*args)

    def make_function(self) -> Callable[..., tuple[Any, ...]]:
        plan = self.plan
        compact_consts = {
            var: jnp.asarray(
                extract_global_value(self.original_consts[i], plan.layouts[var].subset)
            )
            for i, var in enumerate(plan.original_jaxpr.constvars)
            if i in plan.kept_constvar_indices
        }

        def local_function(*local_inputs) -> tuple[Any, ...]:
            if len(local_inputs) != len(plan.kept_invar_indices):
                raise TypeError(
                    f"Expected {len(plan.kept_invar_indices)} local inputs, got {len(local_inputs)}"
                )
            env = dict(compact_consts)
            env.update(
                {
                    plan.original_jaxpr.invars[i]: value
                    for i, value in zip(plan.kept_invar_indices, local_inputs)
                }
            )
            for eqn, eqn_plan in zip(plan.original_jaxpr.eqns, plan.eqn_plans):
                if eqn_plan is None:
                    continue
                values = tuple(
                    jnp.asarray(v.val) if isinstance(v, Literal) else env.get(v)
                    for v in eqn.invars
                )
                in_layouts = tuple(
                    None if isinstance(v, Literal) else plan.layouts.get(v)
                    for v in eqn.invars
                )
                out_layouts = tuple(
                    plan.layouts.get(v) if live else None
                    for v, live in zip(eqn.outvars, eqn_plan.live_output_mask)
                )
                result = eqn_plan.handler.eval_local(
                    eqn=eqn,
                    plan=eqn_plan,
                    in_values=values,
                    in_layouts=in_layouts,
                    out_layouts=out_layouts,
                    interpreter=self,
                )
                if len(result) != len(eqn.outvars):
                    raise RuntimeError(
                        f"{eqn.primitive.name} returned an invalid local output count"
                    )
                env.update(
                    {
                        var: self.normalize_local_result(value, layout)
                        for var, value, layout in zip(eqn.outvars, result, out_layouts)
                        if value is not None and layout is not None
                    }
                )
            return tuple(env[var] for var in plan.requested_outputs)

        return local_function


def make_local_interpreter(
    plan: JaxprPlan, original_consts: Sequence[Any]
) -> Callable[..., tuple[Any, ...]]:
    return LocalJaxprInterpreter(plan, original_consts).make_function()


def _build_local_program(
    plan: JaxprPlan, original_consts: Sequence[Any]
) -> LocalProgram:
    return LocalProgram(
        fn=make_local_interpreter(plan, original_consts),
        input_specs=tuple(
            InputSpec(i, plan.layouts[plan.original_jaxpr.invars[i]])
            for i in plan.kept_invar_indices
        ),
        output_specs=tuple(
            OutputSpec(v, plan.layouts[v]) for v in plan.requested_outputs
        ),
    )


def build_local_program[**P](
    partition_plan: EnergyPartitionPlan[P], rank: int
) -> LocalProgram[P]:
    """Return a compact local JAXPR for a single rank's contributions."""
    state = partition_plan.energy_trace.state
    seed_demands, requested_outputs = rank_seed_demands(partition_plan, rank)

    jaxpr_plan = plan_local_jaxpr(
        jaxpr=partition_plan.energy_trace.concrete_jaxpr.jaxpr,
        state=state,
        seed_demands=seed_demands,
        requested_outputs=requested_outputs,
    )
    return _build_local_program(
        jaxpr_plan,
        partition_plan.energy_trace.concrete_jaxpr.consts,
    )


def trace_local_program(program: LocalProgram, *example_local_inputs: Any):
    return jax.make_jaxpr(program.fn)(*example_local_inputs)


def report_localization_coverage(plan: JaxprPlan) -> list[str]:
    """Return a compact recursive summary of local execution support."""
    report: list[str] = []
    exact = {
        "gather",
        "dynamic_slice",
        "scatter",
        "scatter-add",
        "scatter-sub",
        "scatter-mul",
        "scatter-min",
        "scatter-max",
        "reduce_sum",
        "jit",
        "pjit",
        "remat2",
    }
    for eqn, eqn_plan in zip(plan.original_jaxpr.eqns, plan.eqn_plans):
        if eqn_plan is None:
            continue
        name = eqn.primitive.name
        kind = (
            "exact specialized"
            if name in exact
            else "exact shared/local-or-full fallback"
        )
        report.append(f"{name:<20} {type(eqn_plan.handler).__name__:<28} {kind}")
        for subplan in eqn_plan.subplans:
            report.extend("  " + line for line in report_localization_coverage(subplan))
    return report


def extract_global_value(global_value: Any, subset: TensorSubset) -> Any:
    """Extract an exact local tensor from a value in its original tensor shape."""
    if not subset.shape:
        return global_value
    if isinstance(subset, Full):
        return jnp.reshape(global_value, subset.shape)
    if isinstance(subset, AxisProduct):
        return jnp.reshape(global_value, subset.shape)[
            jnp.ix_(*(jnp.asarray(axis.to_array()) for axis in subset.axes))
        ]
    flat = jnp.ravel(global_value)
    return jnp.take(flat, jnp.asarray(subset.to_rows().to_array()), axis=0)


def pack_runtime_inputs(
    program: LocalProgram,
    global_inputs: Sequence[Any],
) -> tuple[Any, ...]:
    """Pack global inputs into the local JAXPR's expected layout."""
    return tuple(
        extract_global_value(global_inputs[spec.original_index], spec.layout.subset)
        for spec in program.input_specs
    )


# Compatibility name retained while callers migrate from the manual JAXPR API.
materialize_local_jaxpr = _build_local_program
