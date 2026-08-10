"""
Recursive structural Jacobian and Hessian sparsity propagation.

This module consumes a materialized `JaxprInstance` tree and propagates
structural derivative dependencies with respect to the global DOF vector.

A `DependencySet` represents structural Jacobian support: each tensor entry maps
to the set of global DOFs that may influence it. Primitive-local derivative
rules propagate these dependency sets and contribute structural second-order
interactions to a shared `HessianAccumulator`.

Ordinary primitives are handled through `SEMANTICS`. Higher-order constructs are
handled recursively rather than through ordinary primitive rules:

- calls/remat recurse into their body once;
- maps and carry-free scans represent independent body applications;
- scans with carry are traced in execution order so carry dependencies evolve
  between iterations.

Map tracing has two modes. When every materialized iteration has the same
resolved structural program, the body is traced once using symbolic local input
DOFs. Its local Jacobian and Hessian support are then lifted to each iteration
through that iteration's actual global input-dependency matrix. If routes or
nested structure differ between iterations, tracing falls back to exact
iteration-by-iteration recursion.

Stateful scans are not eligible for this map optimization because their carry
creates genuine cross-iteration dependency propagation.

The root JAXPR is seeded specially: its first input is the global flat DOF
vector and receives singleton dependencies; all other root inputs and constants
start with empty derivative dependencies. Nested JAXPR inputs instead receive
dependency sets supplied by their parent invocation.

The Hessian accumulator is shared across the full recursive trace, so
second-order interactions originating inside nested calls, maps, or scans are
recorded directly in the global sparsity pattern.

Key invariants:

- dependency coordinates always refer to global DOFs;
- nested wrappers introduce no Hessian interactions by themselves;
- independent map iterations may be template-lifted only when their resolved
  structural programs are identical;
- recurrent scan carry dependencies are propagated iteration by iteration;
- zero-length maps/scans produce empty mapped-output dependencies and no
  second-order contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.sparse as sps
from jax.core import Atom
from jax.extend.core import Jaxpr, Literal, Var

from tatva.tracer.analysis import MapPlan, ScanPlan
from tatva.tracer.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.helpers import _shape_of
from tatva.tracer.materialize import (
    CallInstance,
    JaxprInstance,
    MapInstance,
    ResolvedEqn,
    ScanInstance,
)
from tatva.tracer.registry import SEMANTICS
from tatva.tracer.semantics import RuleContext


@dataclass(frozen=True)
class CallDerivativeTrace:
    body: JaxprDerivativeTrace


@dataclass(frozen=True)
class ScanIterationDerivativeTrace:
    index: int
    body: JaxprDerivativeTrace


@dataclass(frozen=True)
class ScanDerivativeTrace:
    iterations: tuple[ScanIterationDerivativeTrace, ...]


@dataclass(frozen=True)
class MapIterationDerivativeTrace:
    index: int
    body: JaxprDerivativeTrace


@dataclass(frozen=True)
class MapDerivativeTrace:
    template: JaxprDerivativeTrace | None
    # present only for the unrolled fallback
    iterations: tuple[MapIterationDerivativeTrace, ...]


@dataclass(frozen=True)
class MapDerivativeTemplate:
    """Handle map by tracing a single iteration and broadcasting the result to all iterations."""

    input_shapes: tuple[tuple[int, ...], ...]
    output_deps: tuple[DependencySet, ...]
    hessian: sps.csr_matrix
    n_local_dofs: int
    trace: JaxprDerivativeTrace


type NestedDerivativeTrace = (
    CallDerivativeTrace | ScanDerivativeTrace | MapDerivativeTrace
)


@dataclass(frozen=True)
class JaxprDerivativeTrace:
    dependencies: dict[Var, DependencySet]
    output_deps: tuple[DependencySet, ...]
    nested: dict[int, NestedDerivativeTrace]


@dataclass(frozen=True)
class DerivativeTrace:
    root: JaxprDerivativeTrace
    hessian: sps.csr_matrix


def trace_derivatives(
    instance: JaxprInstance,
    n_dofs: int,
) -> DerivativeTrace:
    acc = HessianAccumulator(n_dofs)

    input_deps = _seed_root_input_dependencies(instance, n_dofs)

    root = _trace_jaxpr(
        instance=instance, input_deps=input_deps, acc=acc, n_dofs=n_dofs
    )

    return DerivativeTrace(root=root, hessian=acc.finalize())


def _seed_root_input_dependencies(
    instance: JaxprInstance,
    n_dofs: int,
) -> tuple[DependencySet, ...]:
    jaxpr = instance.plan.jaxpr

    if not jaxpr.invars:
        raise ValueError("Expected the first Jaxpr input to be the DOF vector")

    dof_var = jaxpr.invars[0]
    dof_shape = _shape_of(dof_var)

    if dof_shape != (n_dofs,):
        raise ValueError(f"DOF input must have shape ({n_dofs},), got {dof_shape}")

    result: list[DependencySet] = [DependencySet.singletons(n_dofs)]

    result.extend(
        DependencySet.empty(
            _shape_of(var),
            n_dofs,
        )
        for var in jaxpr.invars[1:]
    )

    return tuple(result)


def _trace_jaxpr(
    *,
    instance: JaxprInstance,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> JaxprDerivativeTrace:
    jaxpr = instance.plan.jaxpr

    if len(input_deps) != len(jaxpr.invars):
        raise ValueError(
            f"Jaxpr expects {len(jaxpr.invars)} input dependency sets, "
            f"got {len(input_deps)}"
        )

    dependencies: dict[Var, DependencySet] = {}

    # Inputs come from the parent frame.
    for var, dep in zip(jaxpr.invars, input_deps):
        dependencies[var] = dep

    # Closed-over constants never depend on the global DOFs.
    for var in jaxpr.constvars:
        dependencies[var] = DependencySet.empty(
            _shape_of(var),
            n_dofs,
        )

    nested_traces: dict[int, NestedDerivativeTrace] = {}

    def dependency_of(atom: Atom) -> DependencySet:
        if isinstance(atom, Literal):
            return DependencySet.empty(
                _shape_of(atom),
                n_dofs,
            )
        if isinstance(atom, Var):
            try:
                return dependencies[atom]
            except KeyError as exc:
                raise RuntimeError(f"missing dependency state for {atom}") from exc
        raise TypeError(f"unsupported Jaxpr atom {type(atom)!r}")

    for resolved in instance.eqns:
        eqn = resolved.plan.eqn

        input_eqn_deps = tuple(dependency_of(atom) for atom in eqn.invars)

        if resolved.nested is None:
            output_deps = _trace_ordinary_eqn(
                resolved=resolved,
                input_deps=input_eqn_deps,
                acc=acc,
                n_dofs=n_dofs,
            )

        elif isinstance(resolved.nested, CallInstance):
            output_deps, nested_trace = _trace_call(
                resolved=resolved,
                input_deps=input_eqn_deps,
                acc=acc,
                n_dofs=n_dofs,
            )
            nested_traces[resolved.plan.index] = nested_trace

        elif isinstance(resolved.nested, MapInstance):
            output_deps, nested_trace = _trace_map_unrolled(
                resolved=resolved,
                input_deps=input_eqn_deps,
                acc=acc,
                n_dofs=n_dofs,
            )
            nested_traces[resolved.plan.index] = nested_trace

        elif isinstance(resolved.nested, ScanInstance):
            output_deps, nested_trace = _trace_scan(
                resolved=resolved,
                input_deps=input_eqn_deps,
                acc=acc,
                n_dofs=n_dofs,
            )
            nested_traces[resolved.plan.index] = nested_trace

        else:
            raise TypeError(f"unsupported nested instance {type(resolved.nested)!r}")

        if len(output_deps) != len(eqn.outvars):
            raise RuntimeError(
                f"{eqn.primitive.name} returned "
                f"{len(output_deps)} dependency sets for "
                f"{len(eqn.outvars)} outputs"
            )

        for outvar, dep in zip(eqn.outvars, output_deps):
            if isinstance(outvar, Var):
                dependencies[outvar] = dep

    output_deps = tuple(dependency_of(atom) for atom in jaxpr.outvars)

    return JaxprDerivativeTrace(
        dependencies=dependencies,
        output_deps=output_deps,
        nested=nested_traces,
    )


def _trace_ordinary_eqn(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[DependencySet, ...]:
    eqn = resolved.plan.eqn
    rule = SEMANTICS.get(eqn.primitive)

    ctx = RuleContext(
        eqn=eqn,
        input_deps=input_deps,
        route=resolved.route,
        n_dofs=n_dofs,
    )

    prepared = rule.derivatives.prepare(ctx)

    # Primitive-local second-order structure.
    rule.derivatives.hessian(ctx, prepared, acc)

    # Structural Jacobian propagation.
    return rule.derivatives.dependencies(ctx, prepared)


def _trace_call(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    CallDerivativeTrace,
]:
    nested = resolved.nested

    if not isinstance(nested, CallInstance):
        raise TypeError("expected CallInstance")

    body_trace = _trace_jaxpr(
        instance=nested.body,
        input_deps=input_deps,
        acc=acc,
        n_dofs=n_dofs,
    )

    return (
        body_trace.output_deps,
        CallDerivativeTrace(
            body=body_trace,
        ),
    )


def _leading_axis_dependency(
    dep: DependencySet,
    index: int,
) -> DependencySet:
    if len(dep.shape) == 0:
        raise ValueError("scanned input dependency must have a leading scan axis")

    length = dep.shape[0]

    if index < 0 or index >= length:
        raise IndexError(f"scan index {index} outside leading extent {length}")

    step_shape = dep.shape[1:]
    step_size = int(np.prod(step_shape, dtype=np.int64))

    start = index * step_size
    stop = start + step_size

    return DependencySet(
        dep.csr[start:stop],
        step_shape,
    )


def _stack_leading_axis_dependencies(
    steps: list[DependencySet],
) -> DependencySet:
    if not steps:
        raise ValueError("cannot stack an empty list of scan dependencies")

    step_shape = steps[0].shape

    if any(dep.shape != step_shape for dep in steps):
        raise ValueError("scan output dependency shapes differ between iterations")

    return DependencySet(
        sps.vstack(
            [dep.csr for dep in steps],
            format="csr",
        ),
        (len(steps),) + step_shape,
    )


def _trace_scan(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    ScanDerivativeTrace,
]:
    nested = resolved.nested
    scan_plan = resolved.plan.nested

    if not isinstance(nested, ScanInstance):
        raise TypeError("expected ScanInstance")

    if not isinstance(scan_plan, ScanPlan):
        raise TypeError("expected ScanPlan")

    if scan_plan.num_carry < 0:
        raise RuntimeError("carry-free scan should have been lowered to MapPlan")

    num_consts = scan_plan.num_consts
    num_carry = scan_plan.num_carry
    length = scan_plan.length

    if len(input_deps) < num_consts + num_carry:
        raise ValueError(
            "scan dependency input count is incompatible with num_consts/num_carry"
        )

    const_deps = input_deps[:num_consts]
    carry_deps = list(input_deps[num_consts : num_consts + num_carry])
    xs_deps = input_deps[num_consts + num_carry :]
    n_outputs = len(resolved.plan.eqn.outvars)
    n_y_outputs = n_outputs - num_carry

    if length == 0:
        final_carry = tuple(carry_deps)
        y_outputs: list[DependencySet] = []
        eqn = resolved.plan.eqn

        for outvar in eqn.outvars[num_carry:]:
            shape = _shape_of(outvar)
            y_outputs.append(DependencySet.empty(shape, n_dofs))

        return (
            final_carry + tuple(y_outputs),
            ScanDerivativeTrace(iterations=()),
        )

    # Store y dependencies by logical scan index.
    ys_by_output: list[list[DependencySet | None]] = [
        [None] * length for _ in range(n_y_outputs)
    ]

    iteration_traces: list[ScanIterationDerivativeTrace] = []

    # materialize.py already stored iterations in actual execution order,
    # and each invocation carries its logical xs index.
    for iteration in nested.iterations:
        logical_index = iteration.index

        x_step_deps = tuple(
            _leading_axis_dependency(
                dep,
                logical_index,
            )
            for dep in xs_deps
        )

        body_input_deps = tuple(const_deps) + tuple(carry_deps) + x_step_deps

        body_trace = _trace_jaxpr(
            instance=iteration.body,
            input_deps=body_input_deps,
            acc=acc,
            n_dofs=n_dofs,
        )

        if len(body_trace.output_deps) < num_carry:
            raise RuntimeError("scan body returned fewer outputs than num_carry")

        # Carry dependencies evolve from iteration to iteration.
        carry_deps = list(body_trace.output_deps[:num_carry])
        y_step_deps = body_trace.output_deps[num_carry:]

        if len(y_step_deps) != n_y_outputs:
            raise RuntimeError(
                f"scan body returned {len(y_step_deps)} y outputs, "
                f"expected {n_y_outputs}"
            )

        for output_index, dep in enumerate(y_step_deps):
            ys_by_output[output_index][logical_index] = dep

        iteration_traces.append(
            ScanIterationDerivativeTrace(
                index=logical_index,
                body=body_trace,
            )
        )

    # Final carry outputs.
    final_carry = tuple(carry_deps)

    # Stack y outputs in logical scan-axis order.
    stacked_y: list[DependencySet] = []

    for output_index, steps in enumerate(ys_by_output):
        if any(dep is None for dep in steps):
            raise RuntimeError(
                f"scan y output {output_index} is missing "
                "dependency information for one or more iterations"
            )

        stacked_y.append(
            _stack_leading_axis_dependencies([dep for dep in steps if dep is not None])
        )

    return (
        final_carry + tuple(stacked_y),
        ScanDerivativeTrace(
            iterations=tuple(iteration_traces),
        ),
    )


def _trace_map(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    MapDerivativeTrace,
]:
    nested = resolved.nested
    map_plan = resolved.plan.nested

    if not isinstance(nested, MapInstance):
        raise TypeError("expected MapInstance")

    if not isinstance(map_plan, MapPlan):
        raise TypeError("expected MapPlan")

    if not nested.iterations:
        ouputs = tuple(
            DependencySet.empty(_shape_of(outvar), n_dofs)
            for outvar in resolved.plan.eqn.outvars
        )
        return ouputs, MapDerivativeTrace(template=None, iterations=())

    if _map_iterations_are_structurally_equal(nested):
        return _trace_map_template(
            resolved=resolved,
            input_deps=input_deps,
            acc=acc,
            n_dofs=n_dofs,
        )

    # Value-dependent routing differs between iterations:
    # use exact iteration-by-iteration tracing.
    return _trace_map_unrolled(
        resolved=resolved,
        input_deps=input_deps,
        acc=acc,
        n_dofs=n_dofs,
    )


def _trace_map_unrolled(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    MapDerivativeTrace,
]:
    nested = resolved.nested
    map_plan = resolved.plan.nested

    if not isinstance(nested, MapInstance):
        raise TypeError("expected MapInstance")

    if not isinstance(map_plan, MapPlan):
        raise TypeError("expected MapPlan")

    num_consts = map_plan.num_consts

    const_deps = input_deps[:num_consts]
    mapped_deps = input_deps[num_consts:]

    outputs_by_index: list[list[DependencySet | None]] = [
        [None] * map_plan.length for _ in resolved.plan.eqn.outvars
    ]

    iteration_traces: list[MapIterationDerivativeTrace] = []

    for iteration in nested.iterations:
        logical_index = iteration.index
        step_deps = tuple(
            _leading_axis_dependency(dep, logical_index) for dep in mapped_deps
        )
        body_inputs = tuple(const_deps) + step_deps
        body_trace = _trace_jaxpr(
            instance=iteration.body, input_deps=body_inputs, acc=acc, n_dofs=n_dofs
        )

        if len(body_trace.output_deps) != len(outputs_by_index):
            raise RuntimeError(
                f"mapped body returned "
                f"{len(body_trace.output_deps)} outputs; "
                f"expected {len(outputs_by_index)}"
            )

        for output_index, dep in enumerate(body_trace.output_deps):
            outputs_by_index[output_index][logical_index] = dep

        iteration_traces.append(
            MapIterationDerivativeTrace(index=logical_index, body=body_trace)
        )

    outputs: list[DependencySet] = []

    for output_index, steps in enumerate(outputs_by_index):
        if map_plan.length == 0:
            shape = _shape_of(resolved.plan.eqn.outvars[output_index])
            outputs.append(DependencySet.empty(shape, n_dofs))
            continue

        if any(dep is None for dep in steps):
            raise RuntimeError(
                f"mapped output {output_index} is missing dependency information"
            )

        outputs.append(
            _stack_leading_axis_dependencies([dep for dep in steps if dep is not None])
        )

    return (
        tuple(outputs),
        MapDerivativeTrace(template=None, iterations=tuple(iteration_traces)),
    )


def _trace_map_template(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    MapDerivativeTrace,
]:
    nested = resolved.nested
    map_plan = resolved.plan.nested

    assert isinstance(nested, MapInstance)
    assert isinstance(map_plan, MapPlan)

    representative = nested.iterations[0]
    template = _build_map_derivative_template(representative.body)
    num_consts = map_plan.num_consts

    const_deps = input_deps[:num_consts]
    mapped_deps = input_deps[num_consts:]

    outputs_by_index: list[list[DependencySet | None]] = [
        [None] * map_plan.length for _ in resolved.plan.eqn.outvars
    ]

    # We no longer have distinct derivative traces of every body
    # execution because there was only one symbolic trace.
    #
    # Keep the representative trace information separately if needed
    # later; for now this can be empty.
    iteration_traces: list[MapIterationDerivativeTrace] = []

    for iteration in nested.iterations:
        logical_index = iteration.index
        step_deps = tuple(
            _leading_axis_dependency(dep, logical_index) for dep in mapped_deps
        )
        body_input_deps = tuple(const_deps) + step_deps

        lifting = _input_lifting_matrix(
            body_input_deps,
            expected_shapes=template.input_shapes,
            n_local_dofs=template.n_local_dofs,
            n_dofs=n_dofs,
        )

        # Structural Jacobian propagation
        for output_index, local_dep in enumerate(template.output_deps):
            outputs_by_index[output_index][logical_index] = _lift_template_output(
                local_dep, lifting
            )

        # Structural Hessian propagation
        _lift_template_hessian(template.hessian, lifting, acc)

    outputs: list[DependencySet] = []

    for output_index, steps in enumerate(outputs_by_index):
        if any(dep is None for dep in steps):
            raise RuntimeError(
                f"map output {output_index} is missing dependency information"
            )

        outputs.append(
            _stack_leading_axis_dependencies([dep for dep in steps if dep is not None])
        )

    return (
        tuple(outputs),
        MapDerivativeTrace(
            template=template.trace,  # symbolic trace of the representative iteration
            iterations=tuple(iteration_traces),
        ),
    )


def _values_equal(lhs: object, rhs: object) -> bool:
    """Helper to compare nested structures of values, including numpy arrays and dataclasses."""
    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, np.ndarray):
        return np.array_equal(lhs, rhs)  # ty: ignore[invalid-argument-type]

    if isinstance(lhs, tuple):
        return len(lhs) == len(rhs) and all(  # ty: ignore[invalid-argument-type]
            _values_equal(a, b)
            for a, b in zip(lhs, rhs)  # ty: ignore[not-iterable, invalid-argument-type]
        )

    if isinstance(lhs, dict):
        return lhs.keys() == rhs.keys() and all(  # ty: ignore[unresolved-attribute]
            _values_equal(lhs[key], rhs[key])  # ty: ignore[invalid-argument-type, not-subscriptable]
            for key in lhs
        )

    if hasattr(lhs, "__dataclass_fields__"):
        return all(
            _values_equal(getattr(lhs, name), getattr(rhs, name))
            for name in lhs.__dataclass_fields__  # ty: ignore[not-iterable]
        )

    return lhs == rhs


def _same_structural_instance(
    lhs: JaxprInstance,
    rhs: JaxprInstance,
) -> bool:
    # They should normally share the same static plan.
    if lhs.plan is not rhs.plan:
        return False

    if len(lhs.eqns) != len(rhs.eqns):
        return False

    for lhs_eqn, rhs_eqn in zip(lhs.eqns, rhs.eqns):
        if lhs_eqn.plan.index != rhs_eqn.plan.index:
            return False

        if not _values_equal(lhs_eqn.route, rhs_eqn.route):
            return False

        lhs_nested = lhs_eqn.nested
        rhs_nested = rhs_eqn.nested

        if type(lhs_nested) is not type(rhs_nested):
            return False

        if lhs_nested is None:
            continue

        if isinstance(lhs_nested, CallInstance):
            rhs_nested = cast(CallInstance, rhs_nested)
            if not _same_structural_instance(lhs_nested.body, rhs_nested.body):
                return False

        elif isinstance(lhs_nested, (MapInstance, ScanInstance)):
            rhs_nested = cast(type(lhs_nested), rhs_nested)
            if len(lhs_nested.iterations) != len(rhs_nested.iterations):
                return False

            for a, b in zip(lhs_nested.iterations, rhs_nested.iterations):
                if a.index != b.index:
                    return False

                if not _same_structural_instance(a.body, b.body):
                    return False

        else:
            return False

    return True


def _map_iterations_are_structurally_equal(
    instance: MapInstance,
) -> bool:
    if len(instance.iterations) <= 1:
        return True

    representative = instance.iterations[0].body

    return all(
        _same_structural_instance(representative, iteration.body)
        for iteration in instance.iterations[1:]
    )


def _symbolic_input_dependencies(
    jaxpr: Jaxpr,
) -> tuple[
    tuple[DependencySet, ...],
    int,
]:
    shapes = tuple(_shape_of(var) for var in jaxpr.invars)
    sizes = tuple(int(np.prod(shape, dtype=np.int64)) for shape in shapes)
    n_local_dofs = sum(sizes)
    dependencies: list[DependencySet] = []
    offset = 0

    for shape, size in zip(shapes, sizes):
        rows = np.arange(size, dtype=np.int64)
        cols = offset + rows
        csr = sps.csr_matrix(
            (
                np.ones(size, dtype=bool),
                (rows, cols),
            ),
            shape=(size, n_local_dofs),
            dtype=bool,
        )

        dependencies.append(DependencySet(csr, shape))
        offset += size

    return tuple(dependencies), n_local_dofs


def _build_map_derivative_template(
    representative: JaxprInstance,
) -> MapDerivativeTemplate:
    jaxpr = representative.plan.jaxpr
    symbolic_inputs, n_local_dofs = _symbolic_input_dependencies(jaxpr)
    local_acc = HessianAccumulator(n_local_dofs)
    local_trace = _trace_jaxpr(
        instance=representative,
        input_deps=symbolic_inputs,
        acc=local_acc,
        n_dofs=n_local_dofs,
    )

    return MapDerivativeTemplate(
        input_shapes=tuple(dep.shape for dep in symbolic_inputs),
        output_deps=local_trace.output_deps,
        hessian=local_acc.finalize(),
        n_local_dofs=n_local_dofs,
        trace=local_trace,
    )


def _input_lifting_matrix(
    input_deps: tuple[DependencySet, ...],
    *,
    expected_shapes: tuple[tuple[int, ...], ...],
    n_local_dofs: int,
    n_dofs: int,
) -> sps.csr_matrix:
    if len(input_deps) != len(expected_shapes):
        raise ValueError("map template/body input count mismatch")

    for dep, expected in zip(input_deps, expected_shapes):
        if dep.shape != expected:
            raise ValueError(
                f"map body input shape changed: expected {expected}, got {dep.shape}"
            )

    matrix = sps.vstack([dep.csr for dep in input_deps], format="csr")

    if matrix.shape != (n_local_dofs, n_dofs):
        raise RuntimeError(
            f"invalid map lifting matrix shape "
            f"{matrix.shape}; expected "
            f"({n_local_dofs}, {n_dofs})"
        )

    return matrix


def _lift_template_output(
    template_dep: DependencySet,
    lifting: sps.csr_matrix,
) -> DependencySet:
    csr = (template_dep.csr @ lifting).astype(bool).tocsr()
    csr.eliminate_zeros()

    return DependencySet(csr, template_dep.shape)


def _lift_template_hessian(
    template_hessian: sps.csr_matrix,
    lifting: sps.csr_matrix,
    acc: HessianAccumulator,
) -> None:
    if template_hessian.nnz == 0 or lifting.nnz == 0:
        return

    global_pattern = (lifting.T @ template_hessian @ lifting).tocsr()
    if global_pattern.nnz == 0:
        return

    rows, cols = global_pattern.nonzero()

    acc._add_coords(
        rows.astype(np.int64, copy=False),
        cols.astype(np.int64, copy=False),
    )
