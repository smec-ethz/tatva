from __future__ import annotations

import functools
import typing
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from numpy.typing import ArrayLike, NDArray

from tatva.tracer.capture import CapturedJaxpr, make_captured_jaxpr
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.derivatives import LocalDerivativeTrace, trace_local_derivatives
from tatva.tracer.local.dof_plan import (
    LocalDofPlan,
    build_local_dof_plan,
    validate_dof_owner,
)
from tatva.tracer.local.inputs import LocalizeOverrides, localize_inputs
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import LocalJaxprPlan, build_rank_local_plan
from tatva.tracer.lowering.executor import build_local_executable
from tatva.tracer.lowering.partition import (
    ContributionPartition,
    OwnedContribution,
    PartitionStrategy,
    dof_owner_from_incidence,
    partition_contribution_blocks,
)
from tatva.tracer.program.analysis import JaxprPlan
from tatva.tracer.program.analysis import analyze as analyze_jaxpr
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ContributionTrace, detect_contributions
from tatva.tracer.program.incidence import (
    generate_contribution_blocks,
    plan_tagged_block_dof_incidence,
)
from tatva.tracer.support import require_local_routes, require_registered_operations


@dataclass(frozen=True)
class LocalArguments:
    """Rank-local arguments ready to pass to a compiled local functional."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True, eq=False)
class FunctionalAnalysis[**P, R]:
    """Structural analysis of a captured functional.

    The compiler artifacts are intentionally private. Expensive global inspection is
    available from :mod:`tatva.tracer.diagnostics`; normal execution proceeds through
    :meth:`distribute`.
    """

    _captured: CapturedJaxpr[P, R]
    _plan: JaxprPlan
    _contributions: ContributionTrace

    def distribute(
        self,
        *,
        parts: int,
        blocks_per_part: int = 4,
        dof_owner: ArrayLike | None = None,
    ) -> DistributionPlan[P, R]:
        """Build global partition metadata without materializing rank-local plans."""
        if parts <= 0:
            raise ValueError("parts must be positive")
        if blocks_per_part <= 0:
            raise ValueError("blocks_per_part must be positive")

        blocks = generate_contribution_blocks(
            self._contributions,
            blocks_per_root=blocks_per_part * parts,
        )
        resolver, frame = ConcreteResolver.root(
            self._captured.closed_jaxpr,
            self._captured.flat_args,
            self._plan,
        )
        incidence = plan_tagged_block_dof_incidence(
            self._plan,
            frame,
            resolver,
            self._contributions,
            blocks=blocks,
        )
        contribution_partition, block_to_part = partition_contribution_blocks(
            incidence,
            n_parts=parts,
            strategy=PartitionStrategy.INCIDENCE,
        )

        if dof_owner is None:
            owner = dof_owner_from_incidence(
                incidence,
                block_to_part=block_to_part,
                n_parts=parts,
            )
        else:
            owner = validate_dof_owner(dof_owner, n_ranks=parts)

        return DistributionPlan(
            _functional=self,
            _partition=contribution_partition,
            _dof_owner=owner,
        )


@dataclass(frozen=True, eq=False)
class DistributionPlan[**P, R]:
    """Global contribution/ownership metadata with lazy rank construction."""

    _functional: FunctionalAnalysis[P, R]
    _partition: ContributionPartition
    _dof_owner: NDArray
    _rank_cache: dict[int, LocalFunctional[P, R]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def parts(self) -> int:
        return self._partition.n_parts

    def rank(self, rank: int) -> LocalFunctional[P, R]:
        """Lazily construct and cache one rank-local functional."""
        if rank < 0 or rank >= self.parts:
            raise ValueError(f"rank {rank} is out of bounds for {self.parts}")
        if rank in self._rank_cache:
            return self._rank_cache[rank]

        functional = self._functional
        owned = self._partition.for_part(rank)
        seeds = tuple(
            DemandSeed(
                value=functional._contributions.root(item.root_id).value,
                demand=item.demand,
            )
            for item in owned
        )
        resolver, frame = ConcreteResolver.root(
            functional._captured.closed_jaxpr,
            functional._captured.flat_args,
            functional._plan,
        )
        demand = backpropagate_plan_demand(
            functional._plan,
            frame,
            resolver,
            seeds,
        )
        local_plan = build_rank_local_plan(
            functional._plan,
            frame,
            resolver,
            demand,
        )
        require_local_routes((local_plan,))

        compute_layout = local_plan.input_layouts[0]
        if compute_layout is None:
            raise RuntimeError("DOF input unexpectedly dead")

        dofs = build_local_dof_plan(
            compute_layout,
            self._dof_owner,
            rank=rank,
            n_ranks=self.parts,
        )
        local = LocalFunctional(
            rank=rank,
            parts=self.parts,
            dofs=dofs,
            _captured=functional._captured,
            _contributions=functional._contributions,
            _owned=owned,
            _plan=local_plan,
        )
        self._rank_cache[rank] = local
        return local

    def all_ranks(self) -> tuple[LocalFunctional[P, R], ...]:
        """Explicitly construct the local functional for every rank."""
        return tuple(self.rank(rank) for rank in range(self.parts))


@dataclass(frozen=True, eq=False)
class LocalFunctional[**P, R]:
    """Self-contained program and storage layout for one rank."""

    rank: int
    parts: int
    dofs: LocalDofPlan
    _captured: CapturedJaxpr[P, R]
    _contributions: ContributionTrace
    _owned: tuple[OwnedContribution, ...]
    _plan: LocalJaxprPlan

    @functools.cached_property
    def _executable(self):
        return build_local_executable(
            self._plan,
            contributions=self._contributions,
            owned=self._owned,
        )

    @functools.cached_property
    def _compiled(self) -> typing.Callable[..., R]:
        executable = self._executable
        compute_rows = self.dofs.compute_rows
        call_abi = self._captured.call_abi
        input_layouts = self._plan.input_layouts

        @functools.wraps(self._captured.fn)
        def local_function(*args, **kwargs):
            bound = call_abi.bind(*args, **kwargs)
            flat = call_abi.flatten_bound(bound)
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

        return local_function

    def compile(self) -> typing.Callable[..., R]:
        """Return the cached executable over rank-local inputs."""
        return self._compiled

    @functools.cached_property
    def _derivatives(self) -> LocalDerivativeTrace:
        return trace_local_derivatives(
            self._executable,
            self.dofs,
            self._captured.flat_args,
        )

    def derivatives(self) -> LocalDerivativeTrace:
        """Analyze derivatives in storage-local DOF coordinates."""
        return self._derivatives

    def localize(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> LocalArguments:
        """Transform global arguments into this rank's input representation."""
        return self.localize_with({}, *args, **kwargs)

    def localize_with(
        self,
        specializers: LocalizeOverrides,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> LocalArguments:
        """Localize inputs with overrides for user-defined PyTree types."""
        local_args, local_kwargs = localize_inputs(
            self.rank,
            self._captured.call_abi,
            self.dofs,
            specializers,
            self._plan.input_layouts,
            args=args,
            kwargs=kwargs,
        )
        return LocalArguments(args=local_args, kwargs=local_kwargs)


def analyze_captured[**P, R](
    captured: CapturedJaxpr[P, R],
) -> FunctionalAnalysis[P, R]:
    """Structurally analyze an already captured functional."""
    jaxpr = captured.jaxpr
    if not jaxpr.invars:
        raise ValueError("Functional JAXPR has no inputs")

    dof_shape = _shape_of(jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(
            f"First input must be a flat DOF vector, got shape {dof_shape}"
        )

    require_registered_operations(jaxpr)
    plan = analyze_jaxpr(jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )
    contributions = detect_contributions(plan, frame, resolver)
    return FunctionalAnalysis(
        _captured=captured,
        _plan=plan,
        _contributions=contributions,
    )


def analyze[**P, R](
    fn: typing.Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FunctionalAnalysis[P, R]:
    """Capture and structurally analyze a functional."""
    return analyze_captured(make_captured_jaxpr(fn, *args, **kwargs))
