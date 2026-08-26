from __future__ import annotations

import functools
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import ArrayLike, NDArray

from tatva.tracer.capture import CapturedJaxpr, make_captured_jaxpr
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.dof_plan import (
    LocalDofPlan,
    build_local_dof_plan,
)
from tatva.tracer.local.inputs import LocalInputPlan, build_local_input_plan
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import build_rank_local_plan
from tatva.tracer.local.sparsity import LocalMatrixPattern, trace_local_matrix_pattern
from tatva.tracer.lowering.executor import build_local_function
from tatva.tracer.lowering.partition import (
    ContributionPartition,
    DistributionAssignments,
    PartitionStrategy,
    partition_contribution_from_assignments,
    plan_distribution_assignments,
)
from tatva.tracer.program.analysis import JaxprPlan
from tatva.tracer.program.analysis import analyze as analyze_jaxpr
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver
from tatva.tracer.program.contributions import ContributionTrace, detect_contributions
from tatva.tracer.program.forms import (
    FormSpec,
    LocalForm,
    infer_form_spec,
    localize_form,
)
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

if typing.TYPE_CHECKING:
    from mpi4py import MPI

    Comm = MPI.Comm


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
    _dof_input_index: int
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

    def distribute_collectively(
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

        dof_input_index = functional._dof_input_index
        compute_layout = local_plan.input_layouts[dof_input_index]
        if compute_layout is None:
            raise RuntimeError("DOF input unexpectedly dead")

        dofs = build_local_dof_plan(
            compute_layout,
            self._dof_owner,
            rank=rank,
            n_ranks=self.parts,
        )

        input_plan = build_local_input_plan(
            captured=functional._captured,
            local_plan=local_plan,
            dofs=dofs,
            dof_input_index=dof_input_index,
        )
        executable = build_local_function(
            local_plan,
            contributions=functional._contributions,
            owned=owned,
            captured=functional._captured,
            inputs=input_plan,
        )
        local_form = localize_form(functional._form, input_plan)
        local = LocalFunctional(
            dofs=dofs,
            _function=executable,
            _inputs=input_plan,
            _form=local_form,
        )

        self._rank_cache[rank] = local
        return local

    def all_ranks(self) -> tuple[LocalFunctional[P, R], ...]:
        """Explicitly construct the local functional for every rank."""
        return tuple(self.rank(rank) for rank in range(self.parts))


@dataclass(frozen=True, eq=False)
class LocalFunctional[**P, R]:
    """Self-contained program and storage layout for one rank."""

    dofs: LocalDofPlan

    _function: Callable[P, R]
    _inputs: LocalInputPlan
    _form: LocalForm

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def inputs(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return the rank-local input plan for the given arguments."""
        bound = self._inputs.localize(*args, **kwargs)
        return bound.args, bound.kwargs

    def example_inputs(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return a rank-local example input for the given arguments."""
        bound = self._inputs.example_call()
        return bound.args, bound.kwargs

    @functools.cached_property
    def sparsity(self) -> LocalMatrixPattern:
        return trace_local_matrix_pattern(self._function, self._inputs, self._form)


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

    columns = tuple(block for block in form.coordinates if block.role.is_column)
    if not columns:
        raise ValueError("form has no column coordinate block")
    if len(columns) != 1:
        raise NotImplementedError(
            "distributed form analysis currently requires one full flat "
            "column coordinate input"
        )
    column = columns[0]
    if column.selection is not None:
        raise NotImplementedError(
            "distributed form analysis does not support a partial column "
            "coordinate selection"
        )
    dof_input_index = column.input_index
    if dof_input_index >= len(jaxpr.invars):
        raise ValueError(
            f"column coordinate references input {dof_input_index}, but the "
            f"captured JAXPR has {len(jaxpr.invars)} inputs"
        )
    dof_shape = _shape_of(jaxpr.invars[dof_input_index])
    if len(dof_shape) != 1:
        raise ValueError(
            f"Column input must be a flat DOF vector, got shape {dof_shape}"
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
        _dof_input_index=dof_input_index,
        _resolver=resolver,
        _root_frame=frame,
    )


def analyze_form[**P, R](
    form: FormSpec,
    fn: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FunctionalAnalysis[P, R]:
    """Capture and analyze a scalar form with explicit coordinate metadata."""
    return analyze_captured(
        make_captured_jaxpr(fn, *args, **kwargs),
        form=form,
    )


def analyze[**P, R](
    fn: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FunctionalAnalysis[P, R]:
    """Capture an energy functional using the generic scalar-form pipeline."""
    return analyze_captured(make_captured_jaxpr(fn, *args, **kwargs))


class DistributionTarget:
    @dataclass(frozen=True, eq=False)
    class Collective:
        comm: Comm
        strategy: (
            PartitionStrategy | Literal["contiguous", "incidence", "mtkahypar"]
        ) = PartitionStrategy.INCIDENCE
        blocks_per_part: int = 4

    @dataclass(frozen=True, eq=False)
    class Rank:
        n_parts: int
        rank: int
        strategy: (
            PartitionStrategy | Literal["contiguous", "incidence", "mtkahypar"]
        ) = PartitionStrategy.INCIDENCE
        blocks_per_part: int = 4

    type Target = Collective | Rank


def distribute[**P, R](
    fn: Callable[P, R],
    target: DistributionTarget.Target,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> LocalFunctional[P, R]:
    """Analyze a scalar functional and distribute it to a target rank.

    With a `DistributionTarget.Collective`, every rank in the MPI communicator
    participates in planning and receives its rank-local functional. A
    `DistributionTarget.Rank` performs planning serially and returns the local functional
    for the selected rank.

    Args:
        fn: The scalar energy functional to analyze and distribute.
        target: The collective or serial rank distribution target.
        args: Example positional arguments used to trace ``fn``.
        kwargs: Example keyword arguments used to trace ``fn``.

    Returns:
        The executable rank-local functional for the target rank.
    """
    analysis = analyze_captured(make_captured_jaxpr(fn, *args, **kwargs))

    if isinstance(target, DistributionTarget.Rank):
        return analysis.distribute(
            parts=target.n_parts,
            strategy=PartitionStrategy(target.strategy),
            blocks_per_part=target.blocks_per_part,
        ).rank(target.rank)

    elif isinstance(target, DistributionTarget.Collective):
        return analysis.distribute_collectively(
            comm=target.comm,
            strategy=PartitionStrategy(target.strategy),
            blocks_per_part=target.blocks_per_part,
        )
