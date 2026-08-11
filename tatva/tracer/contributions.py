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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from jax.core import Atom
from jax.extend.core import Literal, Var

from tatva.tracer.helpers import _shape_of
from tatva.tracer.materialize import (
    CallInstance,
    FramePath,
    FrameStep,
    JaxprInstance,
    ResolvedEqn,
)
from tatva.tracer.model import Shape


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


type ContributionCoefficient = int | float | complex


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


class _Mode(Enum):
    SCALAR = auto()
    DOMAIN = auto()


@dataclass(frozen=True)
class _Seed:
    atom: Atom
    coefficient: ContributionCoefficient
    mode: _Mode


@dataclass(frozen=True)
class _InputRequest:
    input_index: int
    coefficient: ContributionCoefficient
    mode: _Mode


@dataclass(frozen=True)
class _RootCandidate:
    value: ValueRef
    domain: ContributionDomain
    coefficient: ContributionCoefficient


@dataclass
class _FrameResult:
    roots: list[_RootCandidate]
    inputs: list[_InputRequest]


@dataclass(frozen=True)
class _TraceStep:
    """Result of tracing one primitive inside the current frame."""

    seeds: tuple[_Seed, ...] = ()
    root: _RootCandidate | None = None


_ADDITIVE_BINARY_FACTORS: dict[str, tuple[int, int]] = {
    "add": (1, 1),
    "sub": (1, -1),
}

_ADDITIVE_UNARY_FACTORS: dict[str, int] = {
    "neg": -1,
}

_TRANSPARENT_UNARY = {
    "reshape",
    "squeeze",
    "transpose",
    "rev",
    "copy",
    "stop_gradient",
    "convert_element_type",
}


# -----------------------------------------------------------------------------
# Domain construction and validation
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


def _make_domain(
    value: ValueRef,
    *,
    partition_axes: PartitionAxesPolicy,
) -> ContributionDomain:
    shape = _shape_of(value.var)
    return ContributionDomain(
        shape=shape,
        partition_axes=_validated_partition_axes(
            value,
            shape,
            partition_axes=partition_axes,
        ),
    )


def _root_candidate(
    path: FramePath,
    var: Var,
    coefficient: ContributionCoefficient,
    *,
    partition_axes: PartitionAxesPolicy,
) -> _RootCandidate:
    value = ValueRef(path=path, var=var)
    return _RootCandidate(
        value=value,
        domain=_make_domain(value, partition_axes=partition_axes),
        coefficient=coefficient,
    )


# -----------------------------------------------------------------------------
# Materialized JAXPR helpers
# -----------------------------------------------------------------------------


def _producer_map(
    instance: JaxprInstance,
) -> dict[Var, tuple[ResolvedEqn, int]]:
    producers: dict[Var, tuple[ResolvedEqn, int]] = {}

    for resolved in instance.eqns:
        for output_index, outvar in enumerate(resolved.plan.eqn.outvars):
            if isinstance(outvar, Var):
                producers[outvar] = (resolved, output_index)

    return producers


def _require_arity(
    primitive_name: str,
    inputs: Sequence[Atom],
    expected: int,
) -> None:
    actual = len(inputs)
    if actual != expected:
        raise InvalidMaterializedJaxprError(
            f"{primitive_name} expected {expected} inputs, got {actual}"
        )


def _concrete_scalar(
    instance: JaxprInstance,
    atom: Atom,
) -> ContributionCoefficient | None:
    if isinstance(atom, Literal):
        value = atom.val
    elif isinstance(atom, Var):
        value = instance.concrete.get(atom)
        if value is None:
            return None
    else:
        return None

    array = np.asarray(value)
    if array.shape != ():
        return None

    return array.item()


# -----------------------------------------------------------------------------
# Primitive semantics
# -----------------------------------------------------------------------------


def _trace_common_primitive(
    resolved: ResolvedEqn,
    coefficient: ContributionCoefficient,
    mode: _Mode,
) -> tuple[_Seed, ...] | None:
    """Trace primitives with identical semantics in scalar and domain modes."""
    eqn = resolved.plan.eqn
    name = eqn.primitive.name

    binary_factors = _ADDITIVE_BINARY_FACTORS.get(name)
    if binary_factors is not None:
        _require_arity(name, eqn.invars, 2)
        lhs_factor, rhs_factor = binary_factors
        return (
            _Seed(eqn.invars[0], coefficient * lhs_factor, mode),
            _Seed(eqn.invars[1], coefficient * rhs_factor, mode),
        )

    unary_factor = _ADDITIVE_UNARY_FACTORS.get(name)
    if unary_factor is not None:
        _require_arity(name, eqn.invars, 1)
        return (_Seed(eqn.invars[0], coefficient * unary_factor, mode),)

    if name in _TRANSPARENT_UNARY:
        _require_arity(name, eqn.invars, 1)
        return (_Seed(eqn.invars[0], coefficient, mode),)

    return None


def _trace_scalar_primitive(
    instance: JaxprInstance,
    resolved: ResolvedEqn,
    coefficient: ContributionCoefficient,
) -> _TraceStep:
    """Trace one primitive while searching for the first additive reduction."""
    common = _trace_common_primitive(resolved, coefficient, _Mode.SCALAR)
    if common is not None:
        return _TraceStep(seeds=common)

    eqn = resolved.plan.eqn
    name = eqn.primitive.name

    if name == "reduce_sum":
        _require_arity(name, eqn.invars, 1)
        return _TraceStep(
            seeds=(
                _Seed(
                    atom=eqn.invars[0],
                    coefficient=coefficient,
                    mode=_Mode.DOMAIN,
                ),
            )
        )

    if name == "mul":
        return _trace_scalar_multiply(instance, resolved, coefficient)

    if name == "div":
        return _trace_scalar_divide(instance, resolved, coefficient)

    raise UnsupportedContributionError(
        f"cannot decompose scalar objective through primitive {name!r}; "
        "expected an additive scalar tail ending in reduce_sum"
    )


def _trace_scalar_multiply(
    instance: JaxprInstance,
    resolved: ResolvedEqn,
    coefficient: ContributionCoefficient,
) -> _TraceStep:
    eqn = resolved.plan.eqn
    name = eqn.primitive.name
    _require_arity(name, eqn.invars, 2)

    lhs, rhs = eqn.invars
    lhs_scalar = _concrete_scalar(instance, lhs)
    rhs_scalar = _concrete_scalar(instance, rhs)

    if lhs_scalar is not None and rhs_scalar is None:
        return _TraceStep(
            seeds=(
                _Seed(
                    atom=rhs,
                    coefficient=coefficient * lhs_scalar,
                    mode=_Mode.SCALAR,
                ),
            )
        )

    if rhs_scalar is not None and lhs_scalar is None:
        return _TraceStep(
            seeds=(
                _Seed(
                    atom=lhs,
                    coefficient=coefficient * rhs_scalar,
                    mode=_Mode.SCALAR,
                ),
            )
        )

    if lhs_scalar is not None and rhs_scalar is not None:
        # Purely concrete term: it contributes no partitionable tensor domain.
        return _TraceStep()

    raise UnsupportedContributionError(
        "contribution scalar multiplication requires exactly one concrete operand"
    )


def _trace_scalar_divide(
    instance: JaxprInstance,
    resolved: ResolvedEqn,
    coefficient: ContributionCoefficient,
) -> _TraceStep:
    eqn = resolved.plan.eqn
    name = eqn.primitive.name
    _require_arity(name, eqn.invars, 2)

    numerator, denominator = eqn.invars
    denominator_scalar = _concrete_scalar(instance, denominator)

    if denominator_scalar is None:
        raise UnsupportedContributionError(
            "contribution scalar division requires a concrete denominator"
        )
    if denominator_scalar == 0:
        raise UnsupportedContributionError(
            "contribution scalar division requires a nonzero denominator"
        )

    return _TraceStep(
        seeds=(
            _Seed(
                atom=numerator,
                coefficient=coefficient / denominator_scalar,
                mode=_Mode.SCALAR,
            ),
        )
    )


def _trace_domain_primitive(
    path: FramePath,
    atom: Var,
    resolved: ResolvedEqn,
    coefficient: ContributionCoefficient,
    *,
    partition_axes: PartitionAxesPolicy,
) -> _TraceStep:
    """Trace one primitive while preserving entry-wise additive structure."""
    common = _trace_common_primitive(resolved, coefficient, _Mode.DOMAIN)
    if common is not None:
        return _TraceStep(seeds=common)

    # Any other primitive breaks the one-to-one contribution structure.
    return _TraceStep(
        root=_root_candidate(path, atom, coefficient, partition_axes=partition_axes)
    )


# -----------------------------------------------------------------------------
# Nested-frame handling
# -----------------------------------------------------------------------------


def _trace_call(
    resolved: ResolvedEqn,
    output_index: int,
    path: FramePath,
    coefficient: ContributionCoefficient,
    mode: _Mode,
    *,
    partition_axes: PartitionAxesPolicy,
) -> tuple[list[_RootCandidate], list[_Seed]]:
    """Trace transparently through a call/remat child frame."""
    nested = resolved.nested
    if not isinstance(nested, CallInstance):
        raise InvalidMaterializedJaxprError(
            "_trace_call received a non-call nested instance"
        )

    eqn = resolved.plan.eqn
    child = nested.body
    child_jaxpr = child.plan.jaxpr

    if output_index >= len(child_jaxpr.outvars):
        raise InvalidMaterializedJaxprError(
            f"{eqn.primitive.name} output mapping is inconsistent"
        )

    child_path = path + (FrameStep(eqn_index=resolved.plan.index, kind="call"),)
    child_result = _walk_frame(
        child,
        child_path,
        [
            _Seed(
                atom=child_jaxpr.outvars[output_index],
                coefficient=coefficient,
                mode=mode,
            )
        ],
        partition_axes=partition_axes,
    )

    forwarded: list[_Seed] = []
    for request in child_result.inputs:
        if request.input_index >= len(eqn.invars):
            raise InvalidMaterializedJaxprError(
                f"{eqn.primitive.name} child input mapping is inconsistent"
            )

        forwarded.append(
            _Seed(
                atom=eqn.invars[request.input_index],
                coefficient=request.coefficient,
                mode=request.mode,
            )
        )

    return child_result.roots, forwarded


def _trace_opaque_nested(
    resolved: ResolvedEqn,
    path: FramePath,
    atom: Var,
    coefficient: ContributionCoefficient,
    mode: _Mode,
    *,
    partition_axes: PartitionAxesPolicy,
) -> _RootCandidate:
    """Handle any non-call nested construct as an opaque structured boundary."""
    if mode is _Mode.DOMAIN:
        return _root_candidate(path, atom, coefficient, partition_axes=partition_axes)

    name = resolved.plan.eqn.primitive.name
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
            if seed.mode is _Mode.DOMAIN:
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

        if isinstance(resolved.nested, CallInstance):
            child_roots, forwarded = _trace_call(
                resolved,
                output_index,
                path,
                seed.coefficient,
                seed.mode,
                partition_axes=partition_axes,
            )
            roots.extend(child_roots)
            stack.extend(forwarded)
            continue

        if resolved.nested is not None:
            roots.append(
                _trace_opaque_nested(
                    resolved,
                    path,
                    atom,
                    seed.coefficient,
                    seed.mode,
                    partition_axes=partition_axes,
                )
            )
            continue

        if seed.mode is _Mode.SCALAR:
            step = _trace_scalar_primitive(
                instance,
                resolved,
                seed.coefficient,
            )
        else:
            step = _trace_domain_primitive(
                path,
                atom,
                resolved,
                seed.coefficient,
                partition_axes=partition_axes,
            )

        if step.root is not None:
            roots.append(step.root)
        stack.extend(step.seeds)

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
                mode=_Mode.SCALAR,
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

        if request.mode is _Mode.SCALAR:
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
