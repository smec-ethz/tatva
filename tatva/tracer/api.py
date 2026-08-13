from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.sparse._coloring import csr_to_adjacency
from tatva.tracer import make_captured_jaxpr
from tatva.tracer.analysis import JaxprPlan, analyze
from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.contributions import ContributionTrace, detect_contributions
from tatva.tracer.derivatives import DerivativeTrace, trace_derivatives
from tatva.tracer.halo import (
    HaloCommunicator,
    HaloPlan,
    build_halo_plans,
    build_local_halo_plan,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.input_localization import (
    InputLocalizationPlan,
    build_input_localization_plan,
)
from tatva.tracer.layout import TensorLayout
from tatva.tracer.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local_plan import LocalJaxprPlan, build_local_plan
from tatva.tracer.lowering import build_local_executable
from tatva.tracer.materialize import JaxprInstance, materialize_plan
from tatva.tracer.partition import (
    ContributionPartition,
    OwnedContribution,
    dof_owner_from_contributions,
    partition_contributions,
)


class PartitionCommunicator(HaloCommunicator, typing.Protocol):
    """Collectives used by `TraceResult.partition_local`.

    `mpi4py.MPI.Comm` implements this interface.
    """

    def bcast(self, obj: typing.Any, root: int = 0) -> typing.Any: ...


@dataclass(frozen=True)
class TraceResult[**P, R]:
    captured: CapturedJaxpr[P, R]
    analysis: JaxprPlan
    resolved: JaxprInstance
    derivatives: DerivativeTrace
    contributions: ContributionTrace

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
        partitioning: NDArray[np.int64] | typing.Literal["metis", "contiguous"],
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
        else:
            raise ValueError(f"unsupported partition method {partitioning!r}")

        contribution_partition = partition_contributions(
            self.contributions,
            n_parts=n_parts,
            derivatives=self.derivatives,
            dof_to_part=dof_to_part,
        )

        if dof_to_part is None:
            dof_to_part = dof_owner_from_contributions(
                owned=contribution_partition.owned,
                roots=self.contributions.roots,
                dependencies=self.derivatives.root.dependencies,
                n_dofs=self.hessian.shape[0],
                n_parts=n_parts,
            )

        return contribution_partition, dof_to_part

    def partition(
        self,
        *,
        n_parts: int,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous"] = "contiguous",
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

        compute_layouts = tuple(plan.input_layouts[0] for plan in local_plans)
        if any(layout is None for layout in compute_layouts):
            raise RuntimeError("DOF input unexpectedly dead")
        compute_layouts = cast(tuple[TensorLayout, ...], compute_layouts)

        halo_plans = build_halo_plans(compute_layouts, dof_to_part)

        input_plans = tuple(
            build_input_localization_plan(
                captured=self.captured,
                local_plan=local_plan,
                halo=halo_plan,
            )
            for local_plan, halo_plan in zip(local_plans, halo_plans, strict=True)
        )

        return DistributedFunctional(
            traced=self,
            partition=contribution_partition,
            dof_owner=dof_to_part,
            local_plans=tuple(local_plans),
            halo_plans=halo_plans,
            input_plans=input_plans,
        )

    def partition_local(
        self,
        *,
        comm: PartitionCommunicator,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous"] = "contiguous",
    ) -> RankLocalFunctional[P, R]:
        """Collectively compile only this communicator rank's local functional.

        Every rank must call this method with the same traced functional and
        partitioning configuration. Static analysis remains replicated, while
        demand propagation, local planning, input localization, and executable
        construction are performed only for the calling rank.

        Halo planning performs one host-side ``alltoall`` of ghost DOF request
        arrays. For METIS, rank zero computes the DOF partition and broadcasts
        it before local planning. MPI is optional; an ``mpi4py.MPI.Comm`` can be
        passed directly when Tatva's MPI extras are installed.
        """
        rank = int(comm.Get_rank())
        n_parts = int(comm.Get_size())
        if rank < 0 or rank >= n_parts:
            raise ValueError(f"communicator rank {rank} is outside [0, {n_parts})")

        effective_partitioning = partitioning
        if isinstance(partitioning, str) and partitioning == "metis":
            root_partition = self._metis_dof_partition(n_parts) if rank == 0 else None
            broadcast = comm.bcast(root_partition, root=0)
            effective_partitioning = np.asarray(broadcast, dtype=np.int64)

        contribution_partition, dof_to_part = self._partition_metadata(
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

        halo_plan = build_local_halo_plan(
            compute_layout,
            dof_to_part,
            comm=comm,
        )
        input_plan = build_input_localization_plan(
            captured=self.captured,
            local_plan=local_plan,
            halo=halo_plan,
        )

        return RankLocalFunctional(
            traced=self,
            rank=rank,
            n_parts=n_parts,
            owned=owned,
            dof_owner=dof_to_part,
            local_plan=local_plan,
            halo_plan=halo_plan,
            input_plan=input_plan,
        )


@dataclass(frozen=True)
class DistributedFunctional[**P, R]:
    traced: TraceResult[P, R]
    partition: ContributionPartition
    dof_owner: NDArray[np.int64]
    local_plans: tuple[LocalJaxprPlan, ...]
    halo_plans: tuple[HaloPlan, ...]
    input_plans: tuple[InputLocalizationPlan, ...]

    def __post_init__(self) -> None:
        n_parts = self.partition.n_parts
        counts = (
            len(self.local_plans),
            len(self.halo_plans),
            len(self.input_plans),
        )
        if counts != (n_parts, n_parts, n_parts):
            raise ValueError(
                "distributed plan counts do not match partition count: "
                f"{counts} != {(n_parts, n_parts, n_parts)}"
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
            dof_owner=self.dof_owner,
            local_plan=self.local_plans[rank],
            halo_plan=self.halo_plans[rank],
            input_plan=self.input_plans[rank],
        )

    def local_function(self, rank: int) -> typing.Callable[P, R]:
        """Compatibility shortcut for ``for_rank(rank).local_function()``."""
        return self.for_rank(rank).local_function()

    def localize_inputs(
        self, rank: int, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
        """Compatibility shortcut for ``for_rank(rank).localize_inputs()``."""
        return self.for_rank(rank).localize_inputs(*args, **kwargs)


@dataclass(frozen=True)
class RankLocalFunctional[**P, R]:
    """Compiled partition state owned by one communicator rank.

    Unlike :class:`DistributedFunctional`, this object does not retain plans
    for other ranks. Its halo send/receive schedules were negotiated
    collectively during :meth:`TraceResult.partition_local`.
    """

    traced: TraceResult[P, R]
    rank: int
    n_parts: int
    owned: tuple[OwnedContribution, ...]
    dof_owner: NDArray[np.int64]
    local_plan: LocalJaxprPlan
    halo_plan: HaloPlan
    input_plan: InputLocalizationPlan

    def local_function(self) -> typing.Callable[P, R]:
        """Build the executable for this rank's already-localized inputs."""
        executable = build_local_executable(
            self.local_plan,
            contributions=self.traced.contributions,
            owned=self.owned,
        )
        compute_rows = self.halo_plan.compute_rows
        call_abi = self.traced.captured.call_abi
        input_layouts = self.local_plan.input_layouts

        @functools.wraps(self.traced.captured.fn)
        def _local_function(*args, **kwargs):
            flat = call_abi.flatten_call(*args, **kwargs)
            executable_inputs = []

            for index, (value, layout) in enumerate(
                zip(flat, input_layouts, strict=True)
            ):
                if layout is None:
                    continue
                if index == 0:
                    value = value[jnp.asarray(compute_rows)]
                executable_inputs.append(value)

            return executable(*executable_inputs)

        return _local_function

    def localize_inputs(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
        """Localize a global call using this rank's input reconstruction plan."""
        flat = self.traced.captured.call_abi.flatten_call(*args, **kwargs)
        local_flat = self.input_plan.apply_flat(flat)
        local_bound = self.traced.captured.call_abi.unflatten(local_flat)
        return local_bound.args, dict(local_bound.kwargs)


def trace[**P, R](captured: CapturedJaxpr[P, R]) -> TraceResult[P, R]:
    jaxpr = captured.jaxpr

    if not jaxpr.invars:
        raise ValueError("Functional JAXPR has no inputs")

    dof_shape = _shape_of(jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(
            f"First input must be a flat DOF vector, got shape {dof_shape}"
        )
    n_dofs = dof_shape[0]

    # 1. Static structural analysis
    analysis = analyze(jaxpr)

    # 2. recursive concrete evaluation + route materialization
    resolved = materialize_plan(
        captured.closed_jaxpr,
        captured.flat_args,
        analysis,
    )

    # 3. recursive derivative propagation
    derivatives = trace_derivatives(
        resolved,
        n_dofs=n_dofs,
    )

    contributions = detect_contributions(
        resolved,
    )

    return TraceResult(
        captured=captured,
        analysis=analysis,
        resolved=resolved,
        derivatives=derivatives,
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
