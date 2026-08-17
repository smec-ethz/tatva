"""
Detection of additive contribution domains in a materialized JAXPR program.

Contribution detection identifies tensor-valued quantities whose entries can be
partitioned independently and later reduced to reproduce the global scalar
objective.

The detector walks backwards from the scalar JAXPR output in two modes:

1. Scalar-additive mode follows operations that preserve an additive scalar
   decomposition. Supported operations are add/sub/neg, shape-transparent unary
   operations, call/remat boundaries, multiplication by a concrete scalar, and
   division by a concrete nonzero scalar. A `reduce_sum` changes the walk into
   contribution-domain mode.

2. Contribution-domain mode follows operations that preserve the one-to-one
   additive structure of tensor entries. When an operation no longer preserves
   that structure, its output becomes a `ContributionRoot`.

Call-like nested JAXPRs are transparent. Other nested constructs are treated as
opaque structured boundaries: in contribution-domain mode their outputs become
contribution roots rather than descending into individual iterations.

This module detects ownership candidates only. It does not perform partitioning,
backward liveness propagation, layout construction, or localization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from jax.core import Atom
from jax.extend.core import Literal, Var

from tatva.tracer.core.nested import (
    CallContext,
    FramePath,
    MapContext,
    ScanContext,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import (
    ContributionCoefficient,
    ContributionContext,
    ContributionMode,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.materialize import (
    JaxprInstance,
    ResolvedEqn,
)


@dataclass(frozen=True)
class ValueRef:
    """Invocation-qualified reference to a JAXPR variable."""

    path: FramePath
    var: Var


@dataclass(frozen=True)
class ContributionDomain:
    """Shape and axes along which a contribution root may be partitioned."""

    shape: Shape
    partition_axes: tuple[int, ...]


@dataclass(frozen=True)
class ContributionRoot:
    """Tensor-valued additive contribution to the scalar objective.

    Root IDs are dense and correspond to their position in `ContributionTrace.roots`.
    """

    id: int
    value: ValueRef
    domain: ContributionDomain
    coefficient: ContributionCoefficient = 1


@dataclass(frozen=True)
class ContributionTrace:
    """Result of contribution detection for a scalar JAXPR output."""

    scalar_output: ValueRef | None
    roots: tuple[ContributionRoot, ...]

    def root(self, root_id: int) -> ContributionRoot:
        """Return a contribution root by its dense root ID."""
        if root_id < 0 or root_id >= len(self.roots):
            raise KeyError(f"no contribution root with id {root_id}")

        root = self.roots[root_id]
        if root.id != root_id:
            raise InvalidMaterializedJaxprError(
                f"contribution root index {root_id} has inconsistent id {root.id}"
            )
        return root


type PartitionAxesPolicy = Callable[
    [ValueRef, tuple[int, ...]],
    tuple[int, ...],
]


def first_axis_partition(
    value: ValueRef,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Partition non-scalar contribution domains along their first axis."""
    del value
    return () if not shape else (0,)


class ContributionDetectionError(RuntimeError):
    """Base error for failures during contribution detection."""


class UnsupportedContributionError(ContributionDetectionError):
    """A valid JAXPR cannot be interpreted as the supported additive form."""


class InvalidMaterializedJaxprError(ContributionDetectionError):
    """The materialized JAXPR violates an invariant required by the detector."""


@dataclass(frozen=True)
class _Seed:
    atom: Atom
    coefficient: ContributionCoefficient
    mode: ContributionMode


@dataclass(frozen=True)
class _InputRequest:
    input_index: int
    coefficient: ContributionCoefficient
    mode: ContributionMode


@dataclass(frozen=True)
class _RootCandidate:
    value: ValueRef
    domain: ContributionDomain
    coefficient: ContributionCoefficient


@dataclass
class _FrameResult:
    roots: list[_RootCandidate]
    inputs: list[_InputRequest]


# -----------------------------------------------------------------------------
# Domain construction and materialized JAXPR helpers
# -----------------------------------------------------------------------------


def _validated_partition_axes(
    value: ValueRef,
    shape: Shape,
    *,
    partition_axes: PartitionAxesPolicy,
) -> tuple[int, ...]:
    axes = tuple(int(axis) for axis in partition_axes(value, shape))

    if len(set(axes)) != len(axes):
        raise ValueError(f"duplicate partition axes {axes}")

    if any(axis < 0 or axis >= len(shape) for axis in axes):
        raise ValueError(
            f"invalid partition axes {axes} for contribution shape {shape}"
        )

    return axes


def _root_candidate(
    path: FramePath,
    var: Var,
    coefficient: ContributionCoefficient,
    *,
    partition_axes: PartitionAxesPolicy,
) -> _RootCandidate:
    value = ValueRef(path=path, var=var)
    shape = _shape_of(var)

    return _RootCandidate(
        value=value,
        domain=ContributionDomain(
            shape=shape,
            partition_axes=_validated_partition_axes(
                value,
                shape,
                partition_axes=partition_axes,
            ),
        ),
        coefficient=coefficient,
    )


def _producer_map(
    instance: JaxprInstance,
) -> dict[Var, tuple[ResolvedEqn, int]]:
    producers: dict[Var, tuple[ResolvedEqn, int]] = {}

    for resolved in instance.eqns:
        for output_index, outvar in enumerate(resolved.plan.eqn.outvars):
            if isinstance(outvar, Var):
                producers[outvar] = (resolved, output_index)

    return producers


# -----------------------------------------------------------------------------
# Nested-frame handling
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _ContributionNestedHandler:
    # Satisfies the `NestedHandler` interface for contribution detection. The handler
    # is invoked for each nested frame, and may trace through call-like frames or
    # treat other nested constructs as opaque structured boundaries.

    resolved: ResolvedEqn
    output_index: int
    path: FramePath
    seed: _Seed
    partition_axes: PartitionAxesPolicy

    def call(
        self, context: CallContext[JaxprInstance]
    ) -> tuple[list[_RootCandidate], list[_Seed]]:
        return self._trace_call(context)

    def map(
        self, context: MapContext[JaxprInstance]
    ) -> tuple[list[_RootCandidate], list[_Seed]]:
        return ([self._trace_opaque_nested()], [])

    def scan(
        self, context: ScanContext[JaxprInstance]
    ) -> tuple[list[_RootCandidate], list[_Seed]]:
        return ([self._trace_opaque_nested()], [])

    def _trace_call(
        self, context: CallContext[JaxprInstance]
    ) -> tuple[list[_RootCandidate], list[_Seed]]:
        """Trace transparently through a call/remat child frame."""
        nested = context.invocation
        eqn = self.resolved.plan.eqn
        child = nested.body
        child_jaxpr = child.plan.jaxpr

        if self.output_index >= len(child_jaxpr.outvars):
            raise InvalidMaterializedJaxprError(
                f"{eqn.primitive.name} output mapping is inconsistent"
            )

        child_path = self.path + (nested.children()[0].frame_step,)
        child_result = _walk_frame(
            child,
            child_path,
            [
                _Seed(
                    atom=child_jaxpr.outvars[self.output_index],
                    coefficient=self.seed.coefficient,
                    mode=self.seed.mode,
                )
            ],
            partition_axes=self.partition_axes,
        )

        forwarded: list[_Seed] = []
        for request in child_result.inputs:
            try:
                outer_index = context.spec.outer_input_index(
                    request.input_index, outer_arity=len(eqn.invars)
                )
            except IndexError as exc:
                raise InvalidMaterializedJaxprError(
                    f"{eqn.primitive.name} child input mapping is inconsistent"
                ) from exc

            forwarded.append(
                _Seed(
                    atom=eqn.invars[outer_index],
                    coefficient=request.coefficient,
                    mode=request.mode,
                )
            )

        return child_result.roots, forwarded

    def _trace_opaque_nested(self) -> _RootCandidate:
        """Handle any non-call nested construct as an opaque structured boundary."""
        if self.seed.mode is ContributionMode.DOMAIN:
            return _root_candidate(
                self.path,
                cast(Var, self.seed.atom),
                self.seed.coefficient,
                partition_axes=self.partition_axes,
            )

        name = self.resolved.plan.eqn.primitive.name
        raise UnsupportedContributionError(
            f"nested primitive {name!r} produces the scalar objective "
            "without an additive reduction"
        )


# -----------------------------------------------------------------------------
# Backward traversal
# -----------------------------------------------------------------------------


def _walk_frame(
    instance: JaxprInstance,
    path: FramePath,
    seeds: list[_Seed],
    *,
    partition_axes: PartitionAxesPolicy,
) -> _FrameResult:
    """Walk one materialized JAXPR frame backwards from the supplied seeds."""
    jaxpr = instance.plan.jaxpr
    producers = _producer_map(instance)
    input_indices = {var: index for index, var in enumerate(jaxpr.invars)}
    constvars = set(jaxpr.constvars)
    stack = list(seeds)

    roots: list[_RootCandidate] = []
    input_requests: list[_InputRequest] = []

    while stack:
        seed = stack.pop()
        atom = seed.atom

        if seed.coefficient == 0:
            continue
        if isinstance(atom, Literal):
            continue
        if not isinstance(atom, Var):
            raise InvalidMaterializedJaxprError(
                f"unsupported JAXPR atom {type(atom)!r}"
            )

        input_index = input_indices.get(atom)
        if input_index is not None:
            input_requests.append(
                _InputRequest(
                    input_index=input_index,
                    coefficient=seed.coefficient,
                    mode=seed.mode,
                )
            )
            continue

        if atom in constvars:
            if seed.mode is ContributionMode.DOMAIN:
                roots.append(
                    _root_candidate(
                        path,
                        atom,
                        seed.coefficient,
                        partition_axes=partition_axes,
                    )
                )
            # Scalar constants do not create partitionable contributions.
            continue

        producer = producers.get(atom)
        if producer is None:
            raise InvalidMaterializedJaxprError(f"no producer found for {atom}")

        resolved, output_index = producer

        if resolved.nested is not None:
            if resolved.plan.nested is None:
                raise InvalidMaterializedJaxprError(
                    f"{resolved.plan.eqn.primitive.name} has nested instance but no nested plan"
                )
            child_roots, forwarded = dispatch_nested(
                resolved.plan.nested.spec,
                resolved.nested,
                _ContributionNestedHandler(
                    resolved=resolved,
                    output_index=output_index,
                    path=path,
                    seed=seed,
                    partition_axes=partition_axes,
                ),
            )
            roots.extend(child_roots)
            stack.extend(forwarded)
            continue

        semantics = SEMANTICS.get_ordinary(resolved.plan.eqn.primitive)
        decision = semantics.contribution(
            ContributionContext(
                instance=instance,
                resolved=resolved,
                output_index=output_index,
                coefficient=seed.coefficient,
                mode=seed.mode,
            )
        )

        if decision.invalid_reason is not None:
            raise InvalidMaterializedJaxprError(decision.invalid_reason)

        if decision.unsupported_reason is not None:
            raise UnsupportedContributionError(decision.unsupported_reason)

        if decision.root:
            roots.append(
                _root_candidate(
                    path, atom, seed.coefficient, partition_axes=partition_axes
                )
            )

        eqn = resolved.plan.eqn
        for request in decision.inputs:
            if request.input_index < 0 or request.input_index >= len(eqn.invars):
                raise InvalidMaterializedJaxprError(
                    f"{eqn.primitive.name} contribution rule requested invalid "
                    f"input {request.input_index}"
                )

            stack.append(
                _Seed(
                    atom=eqn.invars[request.input_index],
                    coefficient=request.coefficient,
                    mode=request.mode,
                )
            )

    return _FrameResult(
        roots=roots,
        inputs=input_requests,
    )


# -----------------------------------------------------------------------------
# Result aggregation and public entry point
# -----------------------------------------------------------------------------


def _aggregate_roots(
    candidates: list[_RootCandidate],
) -> tuple[ContributionRoot, ...]:
    """Combine equal roots, sum coefficients, and remove exact cancellation."""
    coefficients: dict[
        tuple[ValueRef, ContributionDomain],
        ContributionCoefficient,
    ] = {}
    order: list[tuple[ValueRef, ContributionDomain]] = []

    for candidate in candidates:
        key = (candidate.value, candidate.domain)

        if key not in coefficients:
            coefficients[key] = 0
            order.append(key)

        coefficients[key] += candidate.coefficient

    roots: list[ContributionRoot] = []

    for value, domain in order:
        coefficient = coefficients[(value, domain)]
        if coefficient == 0:
            continue

        roots.append(
            ContributionRoot(
                id=len(roots),
                value=value,
                domain=domain,
                coefficient=coefficient,
            )
        )

    return tuple(roots)


def detect_contributions(
    root: JaxprInstance,
    *,
    partition_axes: PartitionAxesPolicy = first_axis_partition,
) -> ContributionTrace:
    """Detect tensor-valued additive contributions to one scalar JAXPR output."""
    jaxpr = root.plan.jaxpr

    if len(jaxpr.outvars) != 1:
        raise ValueError(
            "contribution detection currently expects one scalar JAXPR output, "
            f"got {len(jaxpr.outvars)}"
        )

    output = jaxpr.outvars[0]
    output_shape = _shape_of(output)
    if output_shape != ():
        raise ValueError(
            "contribution detection expects a scalar objective, "
            f"got output shape {output_shape}"
        )

    scalar_output = ValueRef(path=(), var=output) if isinstance(output, Var) else None

    result = _walk_frame(
        root,
        (),
        [
            _Seed(
                atom=output,
                coefficient=1,
                mode=ContributionMode.SCALAR,
            )
        ],
        partition_axes=partition_axes,
    )

    candidates = list(result.roots)

    # Requests escaping the root frame cannot be forwarded farther. Domain-mode
    # requests therefore make the corresponding root JAXPR input a contribution
    # root. Scalar-mode requests mean no additive reduction was found.
    for request in result.inputs:
        var = jaxpr.invars[request.input_index]

        if request.mode is ContributionMode.SCALAR:
            raise UnsupportedContributionError(
                "scalar objective reaches root input "
                f"{request.input_index} without an additive reduction"
            )

        candidates.append(
            _root_candidate(
                (),
                var,
                request.coefficient,
                partition_axes=partition_axes,
            )
        )

    return ContributionTrace(
        scalar_output=scalar_output,
        roots=_aggregate_roots(candidates),
    )
