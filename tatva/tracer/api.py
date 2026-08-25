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
)
from tatva.tracer.local.inputs import LocalizeOverrides, localize_inputs
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import LocalJaxprPlan, build_rank_local_plan
from tatva.tracer.lowering.executor import build_local_executable
from tatva.tracer.lowering.partition import (
    ContributionPartition,
    DistributionAssignments,
    OwnedContribution,
    PartitionStrategy,
    partition_contribution_from_assignments,
    plan_distribution_assignments,
)
from tatva.tracer.program.analysis import JaxprPlan
from tatva.tracer.program.analysis import analyze as analyze_jaxpr
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver
from tatva.tracer.program.contributions import ContributionTrace, detect_contributions
from tatva.tracer.program.forms import FormSpec, infer_form_spec
from tatva.tracer.program.incidence import (
    BlockCoordinateIncidence,
    generate_contribution_blocks,
    plan_tagged_block_coordinate_incidence,
)
from tatva.tracer.program.incidence_distributed import (
    BlockCoordinateIncidenceShard,
    BlockShard,
    DistributedPlanningError,
    block_shard_for_rank,
    broadcast_assignments,
    collective_check,
    gather_coordinate_incidence,
    shard_coordinate_incidence,
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
    _form: FormSpec
    _resolver: ConcreteResolver
    _root_frame: ConcreteFrame

    def distribute(
        self,
        *,
        parts: int,
        blocks_per_part: int = 4,
        strategy: PartitionStrategy = PartitionStrategy.INCIDENCE,
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
        coordinate_incidence = plan_tagged_block_coordinate_incidence(
            self._plan,
            self._root_frame,
            self._resolver,
            self._contributions,
            blocks=blocks,
            form=self._form,
        )

        assignments = self._distribution_assignments(
            coordinate_incidence,
            parts=parts,
            strategy=strategy,
            dof_owner=dof_owner,
        )
        partition = partition_contribution_from_assignments(blocks, assignments)

        return self._build_distribution_plan(
            partition=partition,
            assignments=assignments,
        )

    def distribute_mpi(
        self,
        *,
        comm,
        blocks_per_part: int = 4,
        strategy: PartitionStrategy = PartitionStrategy.INCIDENCE,
        dof_owner: ArrayLike | None = None,
        root: int = 0,
    ) -> LocalFunctional[P, R]:
        """Collectively build and return this MPI rank's distribution plan."""
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size <= 0:
            raise ValueError("MPI communicator is empty")

        if root < 0 or root >= size:
            raise ValueError(f"root {root} is outside communicator of size {size}")

        if blocks_per_part <= 0:
            raise ValueError("blocks_per_part must be positive")

        blocks = generate_contribution_blocks(
            self._contributions, blocks_per_root=blocks_per_part * size
        )

        # Every compiler rank receives a contiguous subset of contribution
        # blocks and performs only that subset's tagged propagation.
        shard = block_shard_for_rank(blocks, rank=rank, size=size)
        local_error: BaseException | None = None

        try:
            local_incidence = self._trace_incidence_shard(shard)
        except Exception as exc:  # noqa: BLE001
            local_incidence = None
            local_error = exc

        collective_check(comm, local_error, phase="tagged incidence")

        assert local_incidence is not None

        # Reconstruct the global block × coordinate incidence only on the
        # partition root.
        global_coordinates = gather_coordinate_incidence(
            local_incidence,
            global_blocks=blocks,
            comm=comm,
            root=root,
        )

        # The root now executes exactly the same semantic steps as serial
        # distribute():
        #   combined incidence -> block partition
        #   canonical column    -> DOF ownership
        assignments: DistributionAssignments | None = None
        root_error: BaseException | None = None

        if rank == root:
            try:
                assert global_coordinates is not None

                assignments = self._distribution_assignments(
                    global_coordinates,
                    parts=size,
                    strategy=strategy,
                    dof_owner=dof_owner,
                )

            except Exception as exc:  # noqa: BLE001
                root_error = exc

        root_error_text = (
            None
            if root_error is None
            else (f"{type(root_error).__name__}: {root_error}")
        )
        root_error_text = comm.bcast(root_error_text, root=root)
        if root_error_text is not None:
            raise DistributedPlanningError(
                f"MPI distribution planning failed on rank {root}: {root_error_text}"
            )

        # Important: the coordinate width is global even though the block
        # dimension is sharded. Therefore every rank already knows the
        # canonical DOF count; no separate metadata helper is necessary.
        first_column = next(
            coordinate
            for coordinate in self._form.coordinates
            if coordinate.role.is_column
        )
        n_dofs = int(local_incidence.by_coordinate[first_column.name].shape[1])

        assignments = broadcast_assignments(
            assignments,
            n_blocks=len(blocks),
            n_dofs=n_dofs,
            n_parts=size,
            strategy=strategy,
            comm=comm,
            root=root,
        )

        partition = partition_contribution_from_assignments(blocks, assignments)
        distribution = self._build_distribution_plan(
            partition=partition, assignments=assignments
        )

        return distribution.rank(rank)

    def _distribution_assignments(
        self,
        coordinate_incidence: BlockCoordinateIncidence,
        *,
        parts: int,
        strategy: PartitionStrategy,
        dof_owner: ArrayLike | None,
    ) -> DistributionAssignments:
        partition_incidence = coordinate_incidence.combined()

        first_column = next(
            coordinate
            for coordinate in self._form.coordinates
            if coordinate.role.is_column
        )

        owner_incidence = coordinate_incidence.coordinate(first_column.name)

        return plan_distribution_assignments(
            partition_incidence,
            owner_incidence,
            n_parts=parts,
            strategy=strategy,
            dof_owner=dof_owner,
        )

    def _build_distribution_plan(
        self,
        *,
        partition: ContributionPartition,
        assignments: DistributionAssignments,
    ) -> DistributionPlan[P, R]:
        """Build replicated global metadata from completed assignments."""
        if partition.n_parts != assignments.n_parts:
            raise ValueError(
                "partition and assignments must have the same number of parts"
            )

        return DistributionPlan(
            _functional=self,
            _partition=partition,
            _dof_owner=assignments.dof_owner.copy(),
        )

    def _trace_incidence_shard(
        self, shard: BlockShard
    ) -> BlockCoordinateIncidenceShard:

        incidence = plan_tagged_block_coordinate_incidence(
            self._plan,
            self._root_frame,
            self._resolver,
            self._contributions,
            blocks=shard.blocks,
            form=self._form,
        )
        return shard_coordinate_incidence(incidence, shard=shard)


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
        demand = backpropagate_plan_demand(
            functional._plan,
            functional._root_frame,
            functional._resolver,
            seeds,
        )
        local_plan = build_rank_local_plan(
            functional._plan,
            functional._root_frame,
            functional._resolver,
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
            _form=functional._form,
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
    _form: FormSpec

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
            form=self._form,
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
    *,
    form: FormSpec | None = None,
) -> FunctionalAnalysis[P, R]:
    """Structurally analyze an already captured scalar form."""
    jaxpr = captured.jaxpr
    if not jaxpr.invars:
        raise ValueError("Functional JAXPR has no inputs")

    if form is None:
        form = infer_form_spec(captured.fn, captured.call_abi)
    if form is None:
        form = FormSpec.energy(input_index=0)

    # Distributed storage is still backed by the canonical first column input.
    # The derivative core already supports arbitrary/mixed coordinate blocks;
    # multi-column distributed storage is intentionally a separate runtime step.
    first_column = next(
        (block for block in form.coordinates if block.role.is_column),
        None,
    )
    if first_column is None:
        raise ValueError("form has no column coordinate block")
    if first_column.input_index != 0:
        raise NotImplementedError(
            "distributed form analysis currently requires its first column "
            "coordinate block on flat input 0"
        )
    dof_shape = _shape_of(jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(
            f"First column input must be a flat DOF vector, got shape {dof_shape}"
        )

    require_registered_operations(jaxpr)
    plan = analyze_jaxpr(jaxpr)
    resolver, frame = ConcreteResolver.root(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
        unavailable_inputs=form.coordinate_input_indices,
    )
    contributions = detect_contributions(plan, frame, resolver)
    return FunctionalAnalysis(
        _captured=captured,
        _plan=plan,
        _contributions=contributions,
        _form=form,
        _resolver=resolver,
        _root_frame=frame,
    )


def analyze_form[**P, R](
    form: FormSpec,
    fn: typing.Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FunctionalAnalysis[P, R]:
    """Capture and analyze a scalar form with explicit coordinate metadata."""
    return analyze_captured(
        make_captured_jaxpr(fn, *args, **kwargs),
        form=form,
    )


def analyze[**P, R](
    fn: typing.Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FunctionalAnalysis[P, R]:
    """Capture an energy functional using the generic scalar-form pipeline."""
    return analyze_captured(make_captured_jaxpr(fn, *args, **kwargs))
