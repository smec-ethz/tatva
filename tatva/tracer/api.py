from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.mesh import Mesh
from tatva.sparse._coloring import csr_to_adjacency
from tatva.tracer.analysis import JaxprPlan, analyze
from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.contributions import ContributionTrace, detect_contributions
from tatva.tracer.derivatives import DerivativeTrace, trace_derivatives
from tatva.tracer.halo import HaloPlan, build_halo_plans
from tatva.tracer.helpers import _shape_of
from tatva.tracer.input_localization import (
    InputLocalizationPlan,
    build_input_localization_plan,
)
from tatva.tracer.layout import TensorLayout
from tatva.tracer.liveness import DemandSeed, backpropagate_demand
from tatva.tracer.local_plan import LocalJaxprPlan, build_local_plan
from tatva.tracer.localize import LocalGatherRoute
from tatva.tracer.lowering import build_local_executable, extract_local_value
from tatva.tracer.materialize import JaxprInstance, materialize_plan
from tatva.tracer.partition import (
    ContributionPartition,
    dof_owner_from_contributions,
    partition_contributions,
)


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

    def partition(
        self,
        *,
        n_parts: int,
        partitioning: NDArray[np.int64]
        | typing.Literal["metis", "contiguous"] = "contiguous",
        jit: bool = True,
    ) -> DistributedFunctional[P, R]:
        if isinstance(partitioning, np.ndarray):
            dof_to_part = np.asarray(partitioning, dtype=np.int64)

        elif partitioning == "contiguous":
            dof_to_part = None

        elif partitioning == "metis":
            try:
                import pymetis
            except ImportError as e:
                raise ImportError(
                    "pymetis is required for graph partitioning, but it is not installed. "
                    "Please install pymetis to use this feature."
                ) from e

            sparsity = self.hessian
            # this is temporary
            adjacency = csr_to_adjacency(
                sparsity.shape[0], sparsity.indptr, sparsity.indices
            )
            _, parts = pymetis.part_graph(
                n_parts,
                adjacency=adjacency,
            )
            dof_to_part = np.asarray(parts, dtype=np.int64)

        else:
            raise ValueError(f"unsupported partition method {partitioning!r}")

        contribution_partition = partition_contributions(
            self.contributions,
            n_parts=n_parts,
            derivatives=self.derivatives,
            dof_to_part=dof_to_part,
        )

        # If partitioning was contiguous, we need to create a dof_to_part array now
        if partitioning == "contiguous":
            dof_to_part = dof_owner_from_contributions(
                owned=contribution_partition.owned,
                roots=self.contributions.roots,
                dependencies=self.derivatives.root.dependencies,
                # shouldn't access n_dofs like this, temp hack
                n_dofs=self.hessian.shape[0],
                n_parts=n_parts,
            )
        assert dof_to_part is not None

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


@dataclass(frozen=True)
class DistributedFunctional[**P, R]:
    traced: TraceResult[P, R]
    partition: ContributionPartition
    dof_owner: NDArray[np.int64]
    local_plans: tuple[LocalJaxprPlan, ...]
    halo_plans: tuple[HaloPlan, ...]
    input_plans: tuple[InputLocalizationPlan, ...]

    def local_function(self, rank: int) -> typing.Callable[P, R]:
        if rank < 0 or rank >= len(self.halo_plans):
            raise ValueError(f"rank {rank} is out of bounds for {len(self.halo_plans)}")

        plan = self.local_plans[rank]
        executable = build_local_executable(
            plan,
            contributions=self.traced.contributions,
            owned=self.partition.for_part(rank),
        )
        halo = self.halo_plans[rank]
        call_abi = self.traced.captured.call_abi

        @functools.wraps(self.traced.captured.fn)
        def _local_function(*args, **kwargs):
            flat = call_abi.flatten_call(*args, **kwargs)
            executable_inputs = []

            for i, (value, layout) in enumerate(zip(flat, plan.input_layouts)):
                if layout is None:
                    # dead input
                    continue

                if i == 0:
                    # from storage layout to compute layout
                    value = value[jnp.asarray(halo.compute_rows)]

                executable_inputs.append(value)

            return executable(*executable_inputs)

        return _local_function

    def localize_inputs(
        self, rank: int, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
        flat = self.traced.captured.call_abi.flatten_call(*args, **kwargs)
        local_flat = self.input_plans[rank].apply_flat(flat)
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
