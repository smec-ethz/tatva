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

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from jax.core import Atom
from jax.extend.core import Jaxpr, Literal, Var

from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallContext,
    CallInvocation,
    CondContext,
    CondInvocation,
    CustomJvpContext,
    CustomJvpInvocation,
    IndexedChild,
    LinearSolveContext,
    LinearSolveInvocation,
    MapContext,
    RepeatedInvocation,
    ScanContext,
    TraversalOrder,
    collect_logical_output,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import RuleContext
from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.dependencies import DependencySet, HessianAccumulator
from tatva.tracer.program.materialize import (
    JaxprInstance,
    ResolvedEqn,
)


@dataclass(frozen=True)
class NestedDerivativeTrace:
    invocation: AnyNestedInvocation[JaxprDerivativeTrace]
    template: JaxprDerivativeTrace | None


@dataclass(frozen=True)
class MapDerivativeTemplate:
    """Handle map by tracing a single iteration and broadcasting the result to all iterations."""

    input_shapes: tuple[tuple[int, ...], ...]
    output_deps: tuple[DependencySet, ...]
    hessian: sps.csr_matrix
    n_local_dofs: int
    trace: JaxprDerivativeTrace


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

        else:
            nested_plan = resolved.plan.nested
            if nested_plan is None:
                raise TypeError("nested invocation has no analysis plan")
            output_deps, nested_trace = dispatch_nested(
                nested_plan.spec,
                resolved.nested,
                _DerivativeNestedHandler(resolved, input_eqn_deps, acc, n_dofs),
            )
            nested_traces[resolved.plan.index] = nested_trace

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


@dataclass(frozen=True)
class _DerivativeNestedHandler:
    resolved: ResolvedEqn
    input_deps: tuple[DependencySet, ...]
    acc: HessianAccumulator
    n_dofs: int

    def call(self, context: CallContext[JaxprInstance]):
        return _trace_call(
            context=context,
            input_deps=self.input_deps,
            acc=self.acc,
            n_dofs=self.n_dofs,
        )

    def custom_jvp(self, context: CustomJvpContext[JaxprInstance]):
        return _trace_custom_jvp_opaque(
            resolved=self.resolved,
            context=context,
            input_deps=self.input_deps,
            acc=self.acc,
            n_dofs=self.n_dofs,
        )

    def map(self, context: MapContext[JaxprInstance]):
        return _trace_map(
            resolved=self.resolved,
            context=context,
            input_deps=self.input_deps,
            acc=self.acc,
            n_dofs=self.n_dofs,
        )

    def scan(self, context: ScanContext[JaxprInstance]):
        return _trace_scan(
            resolved=self.resolved,
            context=context,
            input_deps=self.input_deps,
            acc=self.acc,
            n_dofs=self.n_dofs,
        )

    def cond(self, context: CondContext[JaxprInstance]):
        return _trace_cond(
            context=context,
            input_deps=self.input_deps,
            acc=self.acc,
            n_dofs=self.n_dofs,
        )

    def linear_solve(self, context: LinearSolveContext[JaxprInstance]):
        # Implicit solve AD is not represented by the primal callback bodies.
        # Keep the outer operation conservative and retain callback traces for
        # diagnostics/local planning only.
        rhs = self.input_deps[context.spec.rhs_indices[0]]
        traces = []
        for spec, child in zip(
            context.spec.callbacks(), context.invocation.children(), strict=True
        ):
            deps = tuple(
                rhs if binding.runtime else self.input_deps[binding.outer_input_index]
                for binding in spec.inputs
            )
            traces.append(
                _trace_jaxpr(
                    instance=child.payload,
                    input_deps=deps,
                    acc=self.acc,
                    n_dofs=self.n_dofs,
                )
            )
        union = (
            sps.vstack([dep.total_union().csr for dep in self.input_deps], format="csr")
            .sum(axis=0)
            .astype(bool)
        )
        result = tuple(
            DependencySet(
                sps.vstack(
                    [sps.csr_matrix(union)] * int(np.prod(_shape_of(out))), format="csr"
                ),
                _shape_of(out),
            )
            for out in self.resolved.plan.eqn.outvars
        )
        return result, NestedDerivativeTrace(
            LinearSolveInvocation(context.invocation.eqn_index, *traces), template=None
        )


def _trace_ordinary_eqn(
    *,
    resolved: ResolvedEqn,
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[DependencySet, ...]:
    eqn = resolved.plan.eqn
    rule = SEMANTICS.get_ordinary(eqn.primitive)

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


def _trace_custom_jvp_opaque(
    *,
    resolved: ResolvedEqn,
    context: CustomJvpContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
    """Trace callbacks diagnostically but keep the outer Hessian conservative."""
    from tatva.tracer.rules.opaque import DERIVATIVES_OPAQUE_NONLINEAR

    scratch = HessianAccumulator(n_dofs)
    primal_trace = _trace_jaxpr(
        instance=context.invocation.primal,
        input_deps=input_deps,
        acc=scratch,
        n_dofs=n_dofs,
    )
    jvp_deps = tuple(
        input_deps[binding.outer_input_index] for binding in context.spec.jvp_bindings
    )
    jvp_trace = _trace_jaxpr(
        instance=context.invocation.jvp,
        input_deps=jvp_deps,
        acc=scratch,
        n_dofs=n_dofs,
    )
    eqn = resolved.plan.eqn
    ctx = RuleContext(eqn=eqn, input_deps=input_deps, route=None, n_dofs=n_dofs)
    rule = DERIVATIVES_OPAQUE_NONLINEAR
    prepared = rule.prepare(ctx)
    rule.hessian(ctx, prepared, acc)
    outputs = rule.dependencies(ctx, prepared)
    nested_trace = CustomJvpInvocation(
        eqn_index=resolved.plan.index,
        primal=primal_trace,
        jvp=jvp_trace,
    )
    return outputs, nested_trace


def _trace_call(
    *,
    context: CallContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation

    child_input_deps = context.spec.select_inputs(input_deps)

    body_trace = _trace_jaxpr(
        instance=nested.body,
        input_deps=child_input_deps,
        acc=acc,
        n_dofs=n_dofs,
    )

    return (
        body_trace.output_deps,
        NestedDerivativeTrace(
            invocation=CallInvocation(nested.eqn_index, body_trace), template=None
        ),
    )


def _trace_cond(
    *,
    context: CondContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation
    child_input_deps = context.spec.select_inputs(input_deps)

    body_trace = _trace_jaxpr(
        instance=nested.body,
        input_deps=child_input_deps,
        acc=acc,
        n_dofs=n_dofs,
    )

    return (
        body_trace.output_deps,
        NestedDerivativeTrace(
            invocation=CondInvocation(
                nested.eqn_index,
                nested.branch_index,
                body_trace,
            ),
            template=None,
        ),
    )


def _trace_scan(
    *,
    resolved: ResolvedEqn,
    context: ScanContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation
    spec = context.spec

    if spec.num_carry < 0:
        raise RuntimeError("carry-free scan should have been lowered to MapPlan")

    num_consts = spec.num_consts
    num_carry = spec.num_carry
    length = spec.length

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
            NestedDerivativeTrace(
                invocation=nested.with_children(()),
                template=None,
            ),
        )

    iteration_traces: list[IndexedChild[JaxprDerivativeTrace]] = []

    # materialize.py already stored iterations in actual execution order,
    # and each invocation carries its logical xs index.
    for child in nested.children(TraversalOrder.EXECUTION):
        logical_index = child.logical_index
        assert logical_index is not None

        x_step_deps = tuple(
            _leading_axis_dependency(
                dep,
                logical_index,
            )
            for dep in xs_deps
        )

        body_input_deps = tuple(const_deps) + tuple(carry_deps) + x_step_deps

        body_trace = _trace_jaxpr(
            instance=child.payload,
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

        iteration_traces.append(
            IndexedChild(
                index=logical_index,
                body=body_trace,
            )
        )

    # Final carry outputs.
    final_carry = tuple(carry_deps)

    # Stack y outputs in logical scan-axis order.
    stacked_y: list[DependencySet] = []

    for output_index in range(n_y_outputs):
        steps = collect_logical_output(
            ((item.index, item.body.output_deps) for item in iteration_traces),
            output_index=num_carry + output_index,
            length=length,
            label="scan derivative",
        )
        stacked_y.append(_stack_leading_axis_dependencies(list(steps)))

    return (
        final_carry + tuple(stacked_y),
        NestedDerivativeTrace(
            invocation=nested.with_children(tuple(iteration_traces)),
            template=None,
        ),
    )


def _trace_map(
    *,
    resolved: ResolvedEqn,
    context: MapContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation
    if not nested.iterations:
        ouputs = tuple(
            DependencySet.empty(_shape_of(outvar), n_dofs)
            for outvar in resolved.plan.eqn.outvars
        )
        return ouputs, NestedDerivativeTrace(
            invocation=nested.with_children(()),
            template=None,
        )

    # a template cannot amortize its construction with one execution
    if len(nested.iterations) == 1:
        return _trace_map_unrolled(
            resolved=resolved,
            context=context,
            input_deps=input_deps,
            acc=acc,
            n_dofs=n_dofs,
        )

    # template tracing assigns one symbolic DOF to every scalar body input.
    # avoid it when that local symbolic problem is larger than tracing directly
    # in the global DOF space.
    # just a heuristic rule, but it may avoid a lot of unnecessary symbolic tracing.
    representative = nested.iterations[0].body
    n_local_dofs = sum(
        math.prod(_shape_of(var)) for var in representative.plan.jaxpr.invars
    )
    if n_local_dofs <= n_dofs and _map_iterations_are_structurally_equal(nested):
        return _trace_map_template(
            resolved=resolved,
            context=context,
            input_deps=input_deps,
            acc=acc,
            n_dofs=n_dofs,
        )

    # Value-dependent routing differs between iterations:
    # use exact iteration-by-iteration tracing.
    return _trace_map_unrolled(
        resolved=resolved,
        context=context,
        input_deps=input_deps,
        acc=acc,
        n_dofs=n_dofs,
    )


def _trace_map_unrolled(
    *,
    resolved: ResolvedEqn,
    context: MapContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation
    spec = context.spec

    num_consts = spec.num_consts

    const_deps = input_deps[:num_consts]
    mapped_deps = input_deps[num_consts:]

    iteration_traces: list[IndexedChild[JaxprDerivativeTrace]] = []

    for child in nested.children(TraversalOrder.EXECUTION):
        logical_index = child.logical_index
        assert logical_index is not None
        step_deps = tuple(
            _leading_axis_dependency(dep, logical_index) for dep in mapped_deps
        )
        body_inputs = tuple(const_deps) + step_deps
        body_trace = _trace_jaxpr(
            instance=child.payload, input_deps=body_inputs, acc=acc, n_dofs=n_dofs
        )

        if len(body_trace.output_deps) != len(resolved.plan.eqn.outvars):
            raise RuntimeError(
                f"mapped body returned "
                f"{len(body_trace.output_deps)} outputs; "
                f"expected {len(resolved.plan.eqn.outvars)}"
            )

        iteration_traces.append(IndexedChild(index=logical_index, body=body_trace))

    outputs: list[DependencySet] = []

    for output_index, outvar in enumerate(resolved.plan.eqn.outvars):
        if spec.length == 0:
            shape = _shape_of(outvar)
            outputs.append(DependencySet.empty(shape, n_dofs))
            continue
        steps = collect_logical_output(
            ((item.index, item.body.output_deps) for item in iteration_traces),
            output_index=output_index,
            length=spec.length,
            label="map derivative",
        )
        outputs.append(_stack_leading_axis_dependencies(list(steps)))

    return (
        tuple(outputs),
        NestedDerivativeTrace(
            invocation=nested.with_children(tuple(iteration_traces)),
            template=None,
        ),
    )


def _trace_map_template(
    *,
    resolved: ResolvedEqn,
    context: MapContext[JaxprInstance],
    input_deps: tuple[DependencySet, ...],
    acc: HessianAccumulator,
    n_dofs: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested = context.invocation
    spec = context.spec

    representative = nested.children(TraversalOrder.EXECUTION)[0]
    template = _build_map_derivative_template(representative.payload)
    num_consts = spec.num_consts

    const_deps = input_deps[:num_consts]
    mapped_deps = input_deps[num_consts:]

    outputs_by_iteration: list[tuple[int, tuple[DependencySet, ...]]] = []

    for child in nested.children(TraversalOrder.EXECUTION):
        logical_index = child.logical_index
        assert logical_index is not None
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
        lifted_outputs = tuple(
            _lift_template_output(local_dep, lifting)
            for local_dep in template.output_deps
        )
        outputs_by_iteration.append((logical_index, lifted_outputs))

        # Structural Hessian propagation
        _lift_template_hessian(template.hessian, lifting, acc)

    outputs: list[DependencySet] = []

    for output_index in range(len(resolved.plan.eqn.outvars)):
        steps = collect_logical_output(
            outputs_by_iteration,
            output_index=output_index,
            length=spec.length,
            label="map template derivative",
        )
        outputs.append(_stack_leading_axis_dependencies(list(steps)))

    return (
        tuple(outputs),
        NestedDerivativeTrace(
            invocation=nested.with_children(()),
            template=template.trace,  # symbolic trace of the representative iteration
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

        if not _same_structural_nested(lhs_nested, rhs_nested):
            return False

    return True


def _same_structural_nested(
    lhs: AnyNestedInvocation[JaxprInstance] | None,
    rhs: AnyNestedInvocation[JaxprInstance] | None,
) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if lhs.kind is not rhs.kind or lhs.eqn_index != rhs.eqn_index:
        return False
    lhs_children = lhs.children()
    rhs_children = rhs.children()
    if len(lhs_children) != len(rhs_children):
        return False
    return all(
        a.logical_index == b.logical_index
        and _same_structural_instance(a.payload, b.payload)
        for a, b in zip(lhs_children, rhs_children)
    )


def _map_iterations_are_structurally_equal(
    instance: RepeatedInvocation[JaxprInstance],
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
