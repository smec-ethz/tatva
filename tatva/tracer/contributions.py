"""
Detection of additive contribution domains in a materialized JAXPR program.

Contribution detection identifies tensor-valued quantities whose entries can be
partitioned independently and later reduced to reproduce the global scalar
objective.

The detector walks backwards from the scalar JAXPR output in two modes:

1. Scalar-additive mode follows operations that preserve an additive scalar
   decomposition. A `reduce_sum` changes the walk into contribution-domain mode.

2. Contribution-domain mode follows only transformations that preserve the
   one-to-one additive structure of tensor entries. When an operation no longer
   preserves that structure, its output becomes a `ContributionRoot`.

Call-like nested JAXPRs are transparent. Map and scan outputs are intentionally
kept as contribution roots rather than descending into individual iterations;
this preserves their leading structured iteration axis for later partitioning.

This module detects ownership candidates only. It does not perform partitioning,
backward liveness propagation, layout construction, or localization.

Answers:
    Which tensor entries constitute additive contributions to the final scalar?

Initial detector stays deliberately narrow:
    scalar-additive mode:
        add/sub/neg
        shape-transparent ops
        call/remat
        reduce_sum → enter contribution-domain mode

    contribution-domain mode:
        add/sub/neg
        shape-transparent ops
        call/remat
        anything else → ContributionRoot
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number

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
    path: FramePath
    var: Var


@dataclass(frozen=True)
class ContributionDomain:
    shape: Shape
    partition_axes: tuple[int, ...]


type ContributionCoefficient = int | float | complex


@dataclass(frozen=True)
class ContributionRoot:
    id: int
    # Because nested JAXPRs now exist, a plain Var is not enough. Introduce a lightweight
    # invocation-qualified reference
    value: ValueRef
    domain: ContributionDomain

    # additive sign in the scalar objective
    coefficient: ContributionCoefficient = 1


@dataclass(frozen=True)
class ContributionTrace:
    scalar_output: ValueRef | None
    roots: tuple[ContributionRoot, ...]

    def root(self, root_id: int) -> ContributionRoot:
        for root in self.roots:
            if root.id == root_id:
                return root
        raise KeyError(f"no contribution root with id {root_id}")


type PartitionAxesPolicy = Callable[
    [ValueRef, tuple[int, ...]],
    tuple[int, ...],
]


def first_axis_partition(
    value: ValueRef,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    del value

    if not shape:
        return ()

    return (0,)


type _Mode = typing.Literal["scalar", "domain"]


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


class UnsupportedContributionError(RuntimeError):
    """The scalar tail could not be interpreted as an additive decomposition."""


_SCALAR_BINARY_ADD = {
    "add",
}
_SCALAR_BINARY_SUB = {
    "sub",
}
_SCALAR_UNARY_NEG = {
    "neg",
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


def _producer_map(
    instance: JaxprInstance,
) -> dict[Var, tuple[ResolvedEqn, int]]:
    result: dict[Var, tuple[ResolvedEqn, int]] = {}

    for resolved in instance.eqns:
        eqn = resolved.plan.eqn

        for output_index, outvar in enumerate(eqn.outvars):
            if isinstance(outvar, Var):
                result[outvar] = (resolved, output_index)

    return result


def _make_domain(
    value: ValueRef,
    *,
    partition_axes: PartitionAxesPolicy,
) -> ContributionDomain:
    shape = _shape_of(value.var)
    axes = tuple(int(axis) for axis in partition_axes(value, shape))

    if len(set(axes)) != len(axes):
        raise ValueError(f"duplicate partition axes {axes}")

    if any(axis < 0 or axis >= len(shape) for axis in axes):
        raise ValueError(
            f"invalid partition axes {axes} for contribution shape {shape}"
        )

    return ContributionDomain(
        shape=shape,
        partition_axes=axes,
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


def _walk_frame(
    instance: JaxprInstance,
    path: FramePath,
    seeds: list[_Seed],
    *,
    partition_axes: PartitionAxesPolicy,
) -> _FrameResult:
    jaxpr = instance.plan.jaxpr
    producers = _producer_map(instance)
    input_indices = {var: index for index, var in enumerate(jaxpr.invars)}
    constvars = set(jaxpr.constvars)
    queue = list(seeds)

    roots: list[_RootCandidate] = []
    input_requests: list[_InputRequest] = []

    while queue:
        seed = queue.pop()

        atom = seed.atom
        coefficient = seed.coefficient
        mode = seed.mode

        if coefficient == 0:
            continue

        if isinstance(atom, Literal):
            continue

        if not isinstance(atom, Var):
            raise TypeError(f"unsupported JAXPR atom {type(atom)!r}")

        # JAXPR input boundary
        input_index = input_indices.get(atom)
        if input_index is not None:
            input_requests.append(
                _InputRequest(
                    input_index=input_index,
                    coefficient=coefficient,
                    mode=mode,
                )
            )
            continue

        # Closed-over constants have no producer inside this frame.
        # In domain mode they can still represent additive values.
        if atom in constvars:
            if mode == "domain":
                roots.append(
                    _root_candidate(
                        path,
                        atom,
                        coefficient,
                        partition_axes=partition_axes,
                    )
                )
                continue

            # A scalar constant in the additive tail does not create a
            # partitionable contribution.
            continue

        producer = producers.get(atom)
        if producer is None:
            raise RuntimeError(f"no producer found for {atom}")

        resolved, output_index = producer
        eqn = resolved.plan.eqn

        # Transparent call/remat boundary
        if isinstance(resolved.nested, CallInstance):
            child = resolved.nested.body
            child_jaxpr = child.plan.jaxpr

            if output_index >= len(child_jaxpr.outvars):
                raise RuntimeError(
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
            roots.extend(child_result.roots)

            # Child JAXPR invars correspond to wrapper inputs.
            for request in child_result.inputs:
                if request.input_index >= len(eqn.invars):
                    raise RuntimeError(
                        f"{eqn.primitive.name} child input mapping is inconsistent"
                    )

                queue.append(
                    _Seed(
                        atom=eqn.invars[request.input_index],
                        coefficient=request.coefficient,
                        mode=request.mode,
                    )
                )

            continue

        # ----------------------------------------------------------
        # Other nested constructs.
        #
        # Do NOT descend into MapInstance/ScanInstance here.
        #
        # If a mapped output appears below a reduction, preserving the
        # outer tensor is exactly what we want as a contribution root.
        # ----------------------------------------------------------
        if resolved.nested is not None:
            if mode == "domain":
                roots.append(
                    _root_candidate(
                        path,
                        atom,
                        coefficient,
                        partition_axes=partition_axes,
                    )
                )
                continue

            raise UnsupportedContributionError(
                f"nested primitive {eqn.primitive.name!r} produces "
                "the scalar objective without an additive reduction"
            )

        name = eqn.primitive.name

        # add
        if name in _SCALAR_BINARY_ADD:
            if len(eqn.invars) != 2:
                raise RuntimeError(f"{name} expected two inputs")

            queue.append(_Seed(eqn.invars[0], coefficient, mode))
            queue.append(_Seed(eqn.invars[1], coefficient, mode))
            continue

        # subtract
        if name in _SCALAR_BINARY_SUB:
            if len(eqn.invars) != 2:
                raise RuntimeError(f"{name} expected two inputs")

            queue.append(_Seed(eqn.invars[0], coefficient, mode))
            queue.append(_Seed(eqn.invars[1], -coefficient, mode))
            continue

        # negation
        if name in _SCALAR_UNARY_NEG:
            if len(eqn.invars) != 1:
                raise RuntimeError(f"{name} expected one input")

            queue.append(_Seed(eqn.invars[0], -coefficient, mode))
            continue

        # Bijective shape/value transforms
        if name in _TRANSPARENT_UNARY:
            if len(eqn.invars) != 1:
                raise RuntimeError(f"{name} expected one input")

            queue.append(_Seed(eqn.invars[0], coefficient, mode))
            continue

        # First additive reduction
        if mode == "scalar" and name == "reduce_sum":
            if len(eqn.invars) != 1:
                raise RuntimeError("reduce_sum expected one input")

            queue.append(
                _Seed(atom=eqn.invars[0], coefficient=coefficient, mode="domain")
            )
            continue

        # Contribution-domain boundary
        if mode == "domain":
            roots.append(
                _root_candidate(path, atom, coefficient, partition_axes=partition_axes)
            )
            continue

        # Multiplicative scalar tail
        if mode == "scalar" and name == "mul":
            if len(eqn.invars) != 2:
                raise RuntimeError("mul expected two inputs")

            lhs, rhs = eqn.invars
            lhs_scalar = _concrete_scalar(instance, lhs)
            rhs_scalar = _concrete_scalar(instance, rhs)

            # c * expression
            if lhs_scalar is not None and rhs_scalar is None:
                queue.append(
                    _Seed(atom=rhs, coefficient=coefficient * lhs_scalar, mode=mode)
                )
                continue
            # expression * c
            if rhs_scalar is not None and lhs_scalar is None:
                queue.append(
                    _Seed(atom=lhs, coefficient=coefficient * rhs_scalar, mode=mode)
                )
                continue
            # both sides concrete: this is just a constant term
            if lhs_scalar is not None and rhs_scalar is not None:
                continue

        # Division by concrete value
        if mode == "scalar" and name == "div":
            if len(eqn.invars) != 2:
                raise RuntimeError("div expected two inputs")

            numerator, denominator = eqn.invars
            denominator_scalar = _concrete_scalar(
                instance,
                denominator,
            )

            if denominator_scalar is None:
                raise UnsupportedContributionError(
                    "contribution scalar division requires a concrete denominator"
                )
            if denominator_scalar == 0:
                raise ZeroDivisionError("contribution scalar denominator is zero")
            queue.append(
                _Seed(
                    atom=numerator,
                    coefficient=coefficient / denominator_scalar,
                    mode=mode,
                )
            )
            continue

        # Non-additive scalar tail
        raise UnsupportedContributionError(
            f"cannot decompose scalar objective through primitive "
            f"{name!r}; expected an additive scalar tail ending in "
            "reduce_sum"
        )

    return _FrameResult(
        roots=roots,
        inputs=input_requests,
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


def _aggregate_roots(
    candidates: list[_RootCandidate],
) -> tuple[ContributionRoot, ...]:
    """This matters for: E = s + s -> which should produce coefficient 2 for the single
    root s, not two separate roots."""
    coefficients: dict[
        tuple[ValueRef, ContributionDomain],
        ContributionCoefficient,
    ] = {}
    order: list[tuple[ValueRef, ContributionDomain]] = []

    for candidate in candidates:
        key = (
            candidate.value,
            candidate.domain,
        )

        if key not in coefficients:
            coefficients[key] = 0
            order.append(key)

        coefficients[key] += candidate.coefficient

    roots: list[ContributionRoot] = []

    for value, domain in order:
        coefficient = coefficients[(value, domain)]

        # Exact additive cancellation.
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
    jaxpr = root.plan.jaxpr

    if len(jaxpr.outvars) != 1:
        raise ValueError(
            f"contribution detection currently expects one scalar "
            f"JAXPR output, got {len(jaxpr.outvars)}"
        )

    output = jaxpr.outvars[0]

    if _shape_of(output) != ():
        raise ValueError(
            f"contribution detection expects a scalar objective, "
            f"got output shape {_shape_of(output)}"
        )

    scalar_output = ValueRef(path=(), var=output) if isinstance(output, Var) else None

    result = _walk_frame(
        root,
        (),
        [_Seed(atom=output, coefficient=1, mode="scalar")],
        partition_axes=partition_axes,
    )

    candidates = list(result.roots)

    # --------------------------------------------------------------
    # Requests that escape the root frame cannot be forwarded farther.
    #
    # In domain mode they simply mean the root JAXPR input itself is
    # the contribution tensor, e.g.
    #
    #     E = sum(u)
    #
    # In scalar mode we never found an additive reduction.
    # --------------------------------------------------------------

    for request in result.inputs:
        var = jaxpr.invars[request.input_index]

        if request.mode == "scalar":
            raise UnsupportedContributionError(
                f"scalar objective reaches root input "
                f"{request.input_index} without an additive reduction"
            )

        candidates.append(
            _root_candidate((), var, request.coefficient, partition_axes=partition_axes)
        )

    return ContributionTrace(
        scalar_output=scalar_output,
        roots=_aggregate_roots(candidates),
    )
