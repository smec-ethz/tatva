"""
Detection of additive contribution domains in a structural JAXPR plan.

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

Concrete scalar operands are resolved lazily. In particular, opaque map and
scan domains do not instantiate their iteration frames during detection.

This module detects ownership candidates only. It does not perform partitioning,
backward liveness propagation, layout construction, or localization.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from jax.core import Atom
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from tatva.tracer.core.nested import (
    CallSpec,
    CondSpec,
    FramePath,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.routes import Shape
from tatva.tracer.core.semantics import (
    ContributionCoefficient,
    ContributionContext,
    ContributionMode,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.program.concrete_resolver import (
    ConcreteFrame,
    ConcreteResolver,
    DynamicRoutingError,
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
class ContributionBlock:
    id: int
    root_id: int
    demand: TensorDemand
    weight: float = 1.0


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
# Domain construction and traversal helpers
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


def _plan_producer_map(
    plan: JaxprPlan,
) -> dict[Var, tuple[EqnPlan, int]]:
    producers: dict[Var, tuple[EqnPlan, int]] = {}

    for eqn_plan in plan.eqns:
        for output_index, outvar in enumerate(eqn_plan.eqn.outvars):
            if isinstance(outvar, Var):
                producers[outvar] = (eqn_plan, output_index)

    return producers


def _as_concrete_scalar(value: object | None) -> ContributionCoefficient | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    return cast(ContributionCoefficient, array.item())


def _resolved_concrete_scalar(
    resolver: ConcreteResolver,
    frame: ConcreteFrame,
    eqn: JaxprEqn,
    input_index: int,
) -> ContributionCoefficient | None:
    if _depends_on_opaque_nested(frame.plan, eqn.invars[input_index]):
        return None
    try:
        value = resolver.value(frame, eqn.invars[input_index])
    except DynamicRoutingError:
        return None
    return _as_concrete_scalar(value)


def _depends_on_opaque_nested(plan: JaxprPlan, atom: Atom) -> bool:
    """Avoid evaluating repeated domains merely to reject them as scalars."""
    producer_maps: dict[int, dict[Var, tuple[EqnPlan, int]]] = {}
    input_maps: dict[int, dict[Var, int]] = {}
    memo: dict[tuple[int, Var], tuple[bool, frozenset[int]]] = {}
    visiting: set[tuple[int, Var]] = set()

    def visit(current: JaxprPlan, candidate: Atom) -> tuple[bool, frozenset[int]]:
        if not isinstance(candidate, Var):
            return False, frozenset()
        key = (id(current), candidate)
        cached = memo.get(key)
        if cached is not None:
            return cached
        inputs = input_maps.setdefault(
            id(current),
            {var: index for index, var in enumerate(current.jaxpr.invars)},
        )
        input_index = inputs.get(candidate)
        if input_index is not None:
            result = False, frozenset((input_index,))
            memo[key] = result
            return result
        producers = producer_maps.setdefault(id(current), _plan_producer_map(current))
        producer = producers.get(candidate)
        if producer is None:
            result = False, frozenset()
            memo[key] = result
            return result
        if key in visiting:
            return False, frozenset()
        visiting.add(key)
        eqn_plan, output_index = producer
        nested = eqn_plan.nested
        if nested is None:
            children = tuple(visit(current, value) for value in eqn_plan.eqn.invars)
            result = (
                any(opaque for opaque, _indices in children),
                frozenset().union(*(indices for _opaque, indices in children)),
            )
        elif isinstance(nested.spec, CallSpec):
            child_opaque, child_inputs = visit(
                nested.body, nested.body.jaxpr.outvars[output_index]
            )
            outer_children = tuple(
                visit(
                    current,
                    eqn_plan.eqn.invars[
                        nested.spec.outer_input_index(
                            child_index,
                            outer_arity=len(eqn_plan.eqn.invars),
                        )
                    ],
                )
                for child_index in child_inputs
            )
            result = (
                child_opaque or any(opaque for opaque, _indices in outer_children),
                frozenset().union(*(indices for _opaque, indices in outer_children)),
            )
        else:
            # Repeated operations and callbacks are contribution-domain barriers.
            # A conditional would require selecting a branch before this preflight,
            # so conservatively avoid using its output as a scalar coefficient too.
            result = True, frozenset()
        visiting.remove(key)
        memo[key] = result
        return result

    return visit(plan, atom)[0]


# -----------------------------------------------------------------------------
# Backward traversal
# -----------------------------------------------------------------------------


def _trace_plan_nested(
    eqn_plan: EqnPlan,
    output_index: int,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seed: _Seed,
    *,
    partition_axes: PartitionAxesPolicy,
) -> tuple[list[_RootCandidate], list[_Seed]]:
    nested = eqn_plan.nested
    assert nested is not None
    spec = nested.spec
    eqn = eqn_plan.eqn

    if not isinstance(spec, (CallSpec, CondSpec)):
        if seed.mode is ContributionMode.DOMAIN:
            return (
                [
                    _root_candidate(
                        frame.path,
                        cast(Var, seed.atom),
                        seed.coefficient,
                        partition_axes=partition_axes,
                    )
                ],
                [],
            )
        raise UnsupportedContributionError(
            f"nested primitive {eqn.primitive.name!r} produces the scalar "
            "objective without an additive reduction"
        )

    if isinstance(spec, CallSpec):
        child = resolver.call_frame(frame, eqn_plan)
    else:
        _branch_index, child = resolver.cond_frame(frame, eqn_plan)

    try:
        child_jaxpr = child.plan.jaxpr
        if output_index >= len(child_jaxpr.outvars):
            raise InvalidMaterializedJaxprError(
                f"{eqn.primitive.name} output mapping is inconsistent"
            )
        child_result = _walk_plan_frame(
            child.plan,
            child,
            resolver,
            [
                _Seed(
                    atom=child_jaxpr.outvars[output_index],
                    coefficient=seed.coefficient,
                    mode=seed.mode,
                )
            ],
            partition_axes=partition_axes,
        )
    finally:
        resolver.release(child)

    forwarded: list[_Seed] = []
    for request in child_result.inputs:
        try:
            outer_index = spec.outer_input_index(
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


def _walk_plan_frame(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    seeds: list[_Seed],
    *,
    partition_axes: PartitionAxesPolicy,
) -> _FrameResult:
    """Walk one structural JAXPR frame backwards from the supplied seeds."""
    if frame.plan is not plan:
        raise ValueError("contribution plan does not match concrete frame")

    jaxpr = plan.jaxpr
    producers = _plan_producer_map(plan)
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
                        frame.path,
                        atom,
                        seed.coefficient,
                        partition_axes=partition_axes,
                    )
                )
            continue

        producer = producers.get(atom)
        if producer is None:
            raise InvalidMaterializedJaxprError(f"no producer found for {atom}")
        eqn_plan, output_index = producer

        if eqn_plan.nested is not None:
            child_roots, forwarded = _trace_plan_nested(
                eqn_plan,
                output_index,
                frame,
                resolver,
                seed,
                partition_axes=partition_axes,
            )
            roots.extend(child_roots)
            stack.extend(forwarded)
            continue

        eqn = eqn_plan.eqn
        semantics = SEMANTICS.get_ordinary(eqn.primitive)

        decision = semantics.contribution(
            ContributionContext(
                eqn=eqn,
                output_index=output_index,
                coefficient=seed.coefficient,
                mode=seed.mode,
                concrete_scalar=functools.partial(
                    _resolved_concrete_scalar, resolver, frame, eqn
                ),
            )
        )
        if decision.invalid_reason is not None:
            raise InvalidMaterializedJaxprError(decision.invalid_reason)
        if decision.unsupported_reason is not None:
            raise UnsupportedContributionError(decision.unsupported_reason)
        if decision.root:
            roots.append(
                _root_candidate(
                    frame.path,
                    atom,
                    seed.coefficient,
                    partition_axes=partition_axes,
                )
            )

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

    return _FrameResult(roots=roots, inputs=input_requests)


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


def _build_trace(
    jaxpr: Jaxpr,
    result: _FrameResult,
    *,
    partition_axes: PartitionAxesPolicy,
) -> ContributionTrace:
    output = jaxpr.outvars[0]
    scalar_output = ValueRef(path=(), var=output) if isinstance(output, Var) else None
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


def _validate_scalar_output(jaxpr: Jaxpr) -> Atom:
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
    return output


def detect_contributions(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    *,
    partition_axes: PartitionAxesPolicy = first_axis_partition,
) -> ContributionTrace:
    """Detect contributions directly from a structural plan and lazy frame."""
    if frame.plan is not plan:
        raise ValueError("contribution plan does not match concrete frame")
    jaxpr = plan.jaxpr
    output = _validate_scalar_output(jaxpr)
    result = _walk_plan_frame(
        plan,
        frame,
        resolver,
        [_Seed(atom=output, coefficient=1, mode=ContributionMode.SCALAR)],
        partition_axes=partition_axes,
    )
    return _build_trace(jaxpr, result, partition_axes=partition_axes)
