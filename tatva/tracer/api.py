from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.sparse._coloring import csr_to_adjacency
from tatva.tracer.capture import CapturedJaxpr, make_captured_jaxpr
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.dof_plan import (
    LocalDofPlan,
    build_local_dof_plan,
    build_local_dof_plans,
)
from tatva.tracer.local.inputs import (
    LocalizeOverrides,
    localize_inputs,
)
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local.plan import LocalJaxprPlan, build_local_plan
from tatva.tracer.lowering.executor import build_local_executable
from tatva.tracer.lowering.partition import (
    ContributionPartition,
    OwnedContribution,
    PartitionStrategy,
    dof_owner_from_incidence,
    partition_contribution_blocks,
)
from tatva.tracer.program.analysis import JaxprPlan, analyze
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ContributionTrace, detect_contributions
from tatva.tracer.program.derivatives import DerivativeTrace, trace_derivatives
from tatva.tracer.program.incidence import (
    BlockDofIncidence,
    plan_tagged_block_dof_incidence,
)
from tatva.tracer.program.materialize import JaxprInstance, materialize_plan
from tatva.tracer.support import require_local_routes, require_registered_operations


@dataclass(frozen=True)
class TraceResult[**P, R]:
    captured: CapturedJaxpr[P, R]
    analysis: JaxprPlan
    contributions: ContributionTrace

    @functools.cached_property
    def resolved(self) -> JaxprInstance:
        """Materialize the invocation tree only for post-partition consumers."""
        return materialize_plan(
            self.captured.closed_jaxpr,
            self.captured.flat_args,
            self.analysis,
        )

    def incidence(self, block_size: int = 1) -> BlockDofIncidence:
        """Sparse block↔DOF incidence, built lazily on first planning use."""
        resolver, frame = ConcreteResolver.root(
            self.captured.closed_jaxpr,
            self.captured.flat_args,
            self.analysis,
        )
        return plan_tagged_block_dof_incidence(
            self.analysis,
            frame,
            resolver,
            self.contributions,
            block_size=block_size,
        )

    @functools.cached_property
    def derivatives(self) -> DerivativeTrace:
        return trace_derivatives(
            self.resolved,
            n_dofs=_shape_of(self.captured.jaxpr.invars[0])[0],
        )

    @property
    def hessian(self) -> sps.csr_matrix:
        return self.derivatives.hessian

    def _metis_dof_partition(self, n_parts: int) -> NDArray[np.int64]:
        try:
            import pymetis
        except ImportError as exc:
            raise ImportError(
                "pymetis is required for graph partitioning, but it is not "
                "installed. Please install pymetis to use this feature."
            ) from exc

        sparsity = self.hessian
        adjacency = csr_to_adjacency(
            sparsity.shape[0], sparsity.indptr, sparsity.indices
        )
        _, parts = pymetis.part_graph(n_parts, adjacency=adjacency)
        return np.asarray(parts, dtype=np.int64)

    def _partition_metadata(
        self,
        *,
        n_parts: int,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous", "incidence"],
    ) -> tuple[ContributionPartition, NDArray[np.int64]]:
        if n_parts <= 0:
            raise ValueError("n_parts must be positive")

        if isinstance(partitioning, np.ndarray):
            dof_to_part: NDArray[np.int64] | None = np.asarray(
                partitioning, dtype=np.int64
            )
        elif partitioning == "contiguous":
            dof_to_part = None
        elif partitioning == "metis":
            dof_to_part = self._metis_dof_partition(n_parts)
        elif partitioning == "incidence":
            dof_to_part = None
        else:
            raise ValueError(f"unsupported partition method {partitioning!r}")

        incidence = self.incidence(block_size=10)
        blocks_per_root = {
            root.id: sum(block.root_id == root.id for block in incidence.blocks)
            for root in self.contributions.roots
        }
        needs_finer_blocks = any(
            root.domain.partition_axes
            and root.domain.shape[root.domain.partition_axes[0]] >= n_parts
            and blocks_per_root[root.id] < n_parts
            for root in self.contributions.roots
        )
        if needs_finer_blocks:
            # Preserve the configured coarse default while ensuring every rank
            # can receive work from each structurally partitionable root.
            resolver, frame = ConcreteResolver.root(
                self.captured.closed_jaxpr,
                self.captured.flat_args,
                self.analysis,
            )
            incidence = plan_tagged_block_dof_incidence(
                self.analysis,
                frame,
                resolver,
                self.contributions,
                block_size=1,
            )

        strategy = (
            PartitionStrategy.INCIDENCE
            if isinstance(partitioning, str) and partitioning == "incidence"
            else PartitionStrategy.CONTIGUOUS
        )
        contribution_partition, block_to_part = partition_contribution_blocks(
            incidence,
            n_parts=n_parts,
            strategy=strategy,
            dof_to_part=dof_to_part,
        )

        if dof_to_part is None:
            dof_to_part = dof_owner_from_incidence(
                incidence,
                block_to_part=block_to_part,
                n_parts=n_parts,
            )

        return contribution_partition, dof_to_part

    def partition(
        self,
        *,
        n_parts: int,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous", "incidence"] = "contiguous",
    ) -> DistributedFunctional[P, R]:
        contribution_partition, dof_to_part = self._partition_metadata(
            n_parts=n_parts,
            partitioning=partitioning,
        )

        local_plans = []
        for rank in range(n_parts):
            owned = contribution_partition.for_part(rank)
            seeds = tuple(
                DemandSeed(
                    value=self.contributions.root(item.root_id).value,
                    demand=item.demand,
                )
                for item in owned
            )
            demand = backpropagate_demand(self.resolved, seeds)
            local_plans.append(build_local_plan(self.resolved, demand))

        local_plans = tuple(local_plans)
        require_local_routes(local_plans)

        compute_layouts = tuple(plan.input_layouts[0] for plan in local_plans)
        if any(layout is None for layout in compute_layouts):
            raise RuntimeError("DOF input unexpectedly dead")
        compute_layouts = cast(tuple[TensorLayout, ...], compute_layouts)

        dof_plans = build_local_dof_plans(compute_layouts, dof_to_part)

        return DistributedFunctional(
            traced=self,
            partition=contribution_partition,
            local_plans=local_plans,
            dof_plans=dof_plans,
        )

    def partition_local(
        self,
        *,
        rank: int,
        n_parts: int,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous", "incidence"] = "contiguous",
    ) -> RankLocalFunctional[P, R]:
        """Build the rank-local view of a distributed functional."""
        if rank < 0 or rank >= n_parts:
            raise ValueError(f"rank {rank} is outside [0, {n_parts})")

        if isinstance(partitioning, str) and partitioning == "metis":
            # because this runs on every rank, this only works if METIS is DETERMINISTIC!
            # if turns out to be wrong, we must bcast the partitioning instead
            effective_partitioning = self._metis_dof_partition(n_parts)
        else:
            effective_partitioning = partitioning

        contribution_partition, dof_owner = self._partition_metadata(
            n_parts=n_parts,
            partitioning=effective_partitioning,
        )
        owned = contribution_partition.for_part(rank)
        seeds = tuple(
            DemandSeed(
                value=self.contributions.root(item.root_id).value,
                demand=item.demand,
            )
            for item in owned
        )
        demand = backpropagate_demand(self.resolved, seeds)
        local_plan = build_local_plan(self.resolved, demand)

        compute_layout = local_plan.input_layouts[0]
        if compute_layout is None:
            raise RuntimeError("DOF input unexpectedly dead")

        dof_plan = build_local_dof_plan(
            compute_layout,
            dof_owner,
            rank=rank,
            n_ranks=n_parts,
        )

        return RankLocalFunctional(
            traced=self,
            rank=rank,
            n_parts=n_parts,
            owned=owned,
            local_plan=local_plan,
            dof_plan=dof_plan,
        )


@dataclass(frozen=True)
class DistributedFunctional[**P, R]:
    traced: TraceResult[P, R]
    partition: ContributionPartition
    local_plans: tuple[LocalJaxprPlan, ...]
    dof_plans: tuple[LocalDofPlan, ...]

    def __post_init__(self) -> None:
        n_parts = self.partition.n_parts
        counts = (
            len(self.local_plans),
            len(self.dof_plans),
        )
        if counts != (n_parts, n_parts):
            raise ValueError(
                "distributed plan counts do not match partition count: "
                f"{counts} != {(n_parts, n_parts)}"
            )

    @property
    def n_parts(self) -> int:
        return self.partition.n_parts

    def for_rank(self, rank: int) -> RankLocalFunctional[P, R]:
        """Return the canonical single-rank view of this all-ranks result."""
        if rank < 0 or rank >= self.n_parts:
            raise ValueError(f"rank {rank} is out of bounds for {self.n_parts}")

        return RankLocalFunctional(
            traced=self.traced,
            rank=rank,
            n_parts=self.n_parts,
            owned=self.partition.for_part(rank),
            local_plan=self.local_plans[rank],
            dof_plan=self.dof_plans[rank],
        )


@dataclass(frozen=True)
class RankLocalFunctional[**P, R]:
    """Compiled partition state owned by one communicator rank."""

    traced: TraceResult[P, R]
    rank: int
    n_parts: int
    owned: tuple[OwnedContribution, ...]
    local_plan: LocalJaxprPlan
    dof_plan: LocalDofPlan

    @property
    def hessian(self) -> sps.csr_matrix:
        l2g = self.dof_plan.storage.global_dofs
        return self.traced.hessian[l2g][:, l2g]

    def local_function(self) -> typing.Callable[P, R]:
        """Build the executable for this rank's already-localized inputs."""
        executable = build_local_executable(
            self.local_plan,
            contributions=self.traced.contributions,
            owned=self.owned,
        )
        compute_rows = self.dof_plan.compute_rows
        call_abi = self.traced.captured.call_abi
        input_layouts = self.local_plan.input_layouts

        @functools.wraps(self.traced.captured.fn)
        def _local_function(*args, **kwargs):
            bound = call_abi.bind(*args, **kwargs)
            flat = call_abi.flatten_bound(bound)
            executable_inputs = []

            for index, (value, layout) in enumerate(
                zip(flat, input_layouts, strict=True)
            ):
                if layout is None:
                    continue
                if index == 0:
                    # this is the DOF input, which is sliced according to the halo plan
                    # storage_layout -> compute_layout
                    value = value[jnp.asarray(compute_rows)]
                executable_inputs.append(value)

            return executable(*executable_inputs)

        return _local_function

    def localize_inputs(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Construct the rank-local form of the functional's input arguments.

        Input leaves that are required by the local program are sliced according to the
        compiler-derived input layouts. Leaves that are not required by the local program
        are replaced by ``None``. A ``None`` value therefore means that the corresponding
        input is dead for execution on this rank; it does not necessarily mean that the
        value is semantically meaningless to the enclosing user object.

        User-defined PyTree types can reconstruct a meaningful rank-local representation
        by implementing ``__tatva_localize__``. The method is invoked bottom-up after its
        child values have been localized, allowing, for example, a mesh object to rebuild
        local connectivity from localized coordinates even when the original connectivity
        leaf is dead and has therefore become ``None``.

        Use ``localize_inputs_with_specializers`` instead to override the default
        localization behavior for specific PyTree types. The ``specializers`` mapping is
        consulted before any ``__tatva_localize__`` method is called.

        The returned positional and keyword arguments preserve the original function
        signature and PyTree structure after any semantic reconstruction has been applied.

        Args:
            *args, **kwargs: Global input arguments matching the captured functional
                signature.

        Returns:
            args, kwargs
                Rank-local positional and keyword arguments suitable for
                ``local_function()``. Dead leaves may be ``None`` unless an enclosing
                ``__tatva_localize__`` implementation reconstructs them.
        """
        return self.localize_inputs_with_specializers({}, *args, **kwargs)

    def localize_inputs_with_specializers(
        self,
        specializers: LocalizeOverrides,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        local_bound = localize_inputs(
            self.rank,
            self.traced.captured.call_abi,
            self.dof_plan,
            specializers,
            self.local_plan.input_layouts,
            args=args,
            kwargs=kwargs,
        )
        return local_bound


def trace[**P, R](captured: CapturedJaxpr[P, R]) -> TraceResult[P, R]:
    jaxpr = captured.jaxpr

    if not jaxpr.invars:
        raise ValueError("Functional JAXPR has no inputs")

    dof_shape = _shape_of(jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(
            f"First input must be a flat DOF vector, got shape {dof_shape}"
        )

    # fail once with all unsupported operations instead of discovering the first missing registration during analysis
    require_registered_operations(jaxpr)

    # 1. Static structural analysis
    analysis = analyze(jaxpr)

    # 2. Structural contribution detection with lazy concrete scalar lookup.
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        analysis,
    )
    contributions = detect_contributions(
        analysis,
        frame,
        resolver,
    )

    return TraceResult(
        captured=captured,
        analysis=analysis,
        contributions=contributions,
    )


def trace_fn[**P, R](
    fn: typing.Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> TraceResult[P, R]:
    """Trace a function to a JAXPR and analyze its derivative structure.

    Args:
        fn: A Python callable to trace.
        *args: Positional arguments to pass to `fn` for tracing.
        **kwargs: Keyword arguments to pass to `fn` for tracing.
    """
    captured = make_captured_jaxpr(fn, *args, **kwargs)
    return trace(captured)
