"""
Recursive structural Jacobian and Hessian sparsity propagation.

This module consumes a materialized `JaxprInstance` tree and propagates
structural derivative dependencies with respect to the declared coordinate blocks.

A `DependencySet` represents structural Jacobian support: each tensor entry maps
to the set of symbolic coordinates that may influence it. Primitive-local derivative
rules propagate these dependency sets and contribute structural second-order
interactions to a shared `InteractionGraph`.

Ordinary primitives are handled through `SEMANTICS`. Higher-order constructs are
handled recursively rather than through ordinary primitive rules:

- calls/remat recurse into their body once;
- maps and carry-free scans represent independent body applications;
- scans with carry are traced in execution order so carry dependencies evolve
  between iterations.

Map tracing has two modes. When every materialized iteration has the same
resolved structural program, the body is traced once in a compact local symbolic
coordinate system. Its dependency and interaction support are then lifted to each
iteration through that iteration's actual input-dependency matrix. If routes or
nested structure differ between iterations, tracing falls back to exact
iteration-by-iteration recursion.

Stateful scans are not eligible for this map optimization because their carry
creates genuine cross-iteration dependency propagation.

Root JAXPR inputs are seeded from a `FormSpec`: every declared coordinate
block receives independent symbolic columns, while non-coordinate inputs start
with empty derivative dependencies. Nested JAXPR inputs receive dependency sets
supplied by their parent invocation.

The interaction graph is shared across the full recursive trace, so second-order
interactions originating inside nested calls, maps, scans, or custom-JVP rules
are recorded in one symbolic coordinate system. Operator sparsity is obtained by
extracting the declared row-by-column block from this graph.

Key invariants:

- dependency coordinates always refer to symbolic coordinates;
- nested wrappers introduce no second-order interactions by themselves;
- independent map iterations may be template-lifted only when their resolved
  structural programs are identical;
- recurrent scan carry dependencies are propagated iteration by iteration;
- zero-length maps/scans produce empty mapped-output dependencies and no
  second-order contributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from jax.core import Atom
from jax.extend.core import Literal, Var

from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallInvocation,
    CallSpec,
    CondInvocation,
    CondSpec,
    CustomJvpInvocation,
    CustomJvpSpec,
    IndexedChild,
    IterationSelection,
    LinearSolveInvocation,
    LinearSolveSpec,
    MapInvocation,
    MapSpec,
    RepeatedInvocation,
    ScanSpec,
    collect_logical_output,
    dispatch_nested_spec,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import RuleContext
from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.analysis import EqnPlan, JaxprPlan
from tatva.tracer.program.concrete_resolver import ConcreteFrame, ConcreteResolver
from tatva.tracer.program.dependencies import DependencySet, InteractionGraph
from tatva.tracer.program.forms import FormSpec, SymbolicLayout
from tatva.tracer.program.repeated import map_template_requires_mapped_concrete


@dataclass(frozen=True)
class NestedDerivativeTrace:
    invocation: AnyNestedInvocation[JaxprDerivativeTrace]
    template: JaxprDerivativeTrace | None


@dataclass(frozen=True)
class MapDerivativeTemplate:
    """Handle map by tracing a single iteration and broadcasting the result to all iterations."""

    input_shapes: tuple[tuple[int, ...], ...]
    output_deps: tuple[DependencySet, ...]
    interactions: sps.csr_matrix
    n_local_symbols: int
    trace: JaxprDerivativeTrace


@dataclass(frozen=True)
class JaxprDerivativeTrace:
    dependencies: dict[Var, DependencySet]
    output_deps: tuple[DependencySet, ...]
    nested: dict[int, NestedDerivativeTrace]


@dataclass(frozen=True)
class DerivativeTrace:
    root: JaxprDerivativeTrace
    symbolic_layout: SymbolicLayout
    interactions: sps.csr_matrix

    @property
    def tangent(self) -> sps.csr_matrix:
        """Row-coordinate × column-coordinate structural tangent pattern."""
        return self.symbolic_layout.tangent_block(self.interactions)

    @property
    def hessian(self) -> sps.csr_matrix:
        """Backward-compatible energy Hessian view.

        A Hessian is defined only when the exact same symbolic coordinates are
        declared as rows and columns.  Weak/mixed forms should use ``tangent``.
        """
        if not self.symbolic_layout.has_identical_rows_and_columns:
            raise AttributeError(
                "hessian is only defined when row and column coordinates coincide; "
                "use tangent for weak or mixed forms"
            )
        return self.tangent


def trace_form_derivatives(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    form: FormSpec,
) -> DerivativeTrace:
    """Trace one scalar form in a unified symbolic coordinate system."""
    layout = SymbolicLayout.from_form(form, plan.jaxpr)
    input_deps = layout.seed_inputs(form, plan.jaxpr)
    return trace_seeded_derivatives(plan, frame, resolver, layout, input_deps)


def trace_seeded_derivatives(
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    symbolic_layout: SymbolicLayout,
    input_deps: tuple[DependencySet, ...],
) -> DerivativeTrace:
    """Trace with an explicit root relation into an existing symbolic layout.

    This is used by localized execution, where executable inputs can be compact
    projections of a larger storage coordinate block.
    """
    if len(input_deps) != len(plan.jaxpr.invars):
        raise ValueError(
            f"JAXPR has {len(plan.jaxpr.invars)} inputs but received "
            f"{len(input_deps)} dependency seeds"
        )
    for dep in input_deps:
        if dep.csr.shape[1] != symbolic_layout.size:
            raise ValueError(
                "root dependency seed width does not match symbolic layout"
            )

    acc = InteractionGraph(symbolic_layout.size)
    root = _trace_jaxpr(
        plan=plan,
        frame=frame,
        resolver=resolver,
        input_deps=input_deps,
        acc=acc,
        n_symbols=symbolic_layout.size,
    )
    return DerivativeTrace(
        root=root,
        symbolic_layout=symbolic_layout,
        interactions=acc.finalize(),
    )


def trace_derivatives(
    captured: CapturedJaxpr,
    plan: JaxprPlan,
    n_dofs: int,
) -> DerivativeTrace:
    """Backward-compatible energy entry point using the generic form tracer."""
    resolver, frame = ConcreteResolver.root(captured.jaxpr, captured.flat_args, plan)
    trace = trace_form_derivatives(
        plan, frame, resolver, FormSpec.energy(input_index=0)
    )
    if trace.symbolic_layout.size != n_dofs:
        raise ValueError(
            f"energy coordinate size {trace.symbolic_layout.size} does not match "
            f"n_dofs={n_dofs}"
        )
    return trace


def _trace_jaxpr(
    *,
    plan: JaxprPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> JaxprDerivativeTrace:
    jaxpr = plan.jaxpr

    if len(input_deps) != len(jaxpr.invars):
        raise ValueError(
            f"Jaxpr expects {len(jaxpr.invars)} input dependency sets, "
            f"got {len(input_deps)}"
        )

    dependencies: dict[Var, DependencySet] = {}

    # Inputs come from the parent frame.
    for var, dep in zip(jaxpr.invars, input_deps):
        dependencies[var] = dep

    # Closed-over constants never depend on the symbolic coordinates.
    for var in jaxpr.constvars:
        dependencies[var] = DependencySet.empty(
            _shape_of(var),
            n_symbols,
        )

    nested_traces: dict[int, NestedDerivativeTrace] = {}

    def dependency_of(atom: Atom) -> DependencySet:
        if isinstance(atom, Literal):
            return DependencySet.empty(
                _shape_of(atom),
                n_symbols,
            )
        if isinstance(atom, Var):
            try:
                return dependencies[atom]
            except KeyError as exc:
                raise RuntimeError(f"missing dependency state for {atom}") from exc
        raise TypeError(f"unsupported Jaxpr atom {type(atom)!r}")

    for eqn_plan in plan.eqns:
        eqn = eqn_plan.eqn

        input_eqn_deps = tuple(dependency_of(atom) for atom in eqn.invars)

        if eqn_plan.nested is None:
            output_deps = _trace_ordinary_eqn(
                eqn_plan=eqn_plan,
                frame=frame,
                resolver=resolver,
                input_deps=input_eqn_deps,
                acc=acc,
                n_symbols=n_symbols,
            )

        else:
            nested_plan = eqn_plan.nested
            if nested_plan is None:
                raise TypeError("nested invocation has no analysis plan")
            output_deps, nested_trace = dispatch_nested_spec(
                nested_plan.spec,
                _DerivativeNestedHandler(
                    eqn_plan, frame, resolver, input_eqn_deps, acc, n_symbols
                ),
            )
            nested_traces[eqn_plan.index] = nested_trace

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
    eqn_plan: EqnPlan
    frame: ConcreteFrame
    resolver: ConcreteResolver
    input_deps: tuple[DependencySet, ...]
    acc: InteractionGraph
    n_symbols: int

    def call(
        self, spec: CallSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        return _trace_call(
            spec=spec,
            eqn_plan=self.eqn_plan,
            frame=self.frame,
            resolver=self.resolver,
            input_deps=self.input_deps,
            acc=self.acc,
            n_symbols=self.n_symbols,
        )

    def custom_jvp(
        self, spec: CustomJvpSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        return _trace_custom_jvp(
            spec=spec,
            eqn_plan=self.eqn_plan,
            frame=self.frame,
            resolver=self.resolver,
            input_deps=self.input_deps,
            acc=self.acc,
            n_symbols=self.n_symbols,
        )

    def map(
        self, spec: MapSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        return _trace_map(
            spec=spec,
            eqn_plan=self.eqn_plan,
            frame=self.frame,
            resolver=self.resolver,
            input_deps=self.input_deps,
            acc=self.acc,
            n_symbols=self.n_symbols,
        )

    def scan(
        self, spec: ScanSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        return _trace_scan(
            spec=spec,
            eqn_plan=self.eqn_plan,
            frame=self.frame,
            resolver=self.resolver,
            input_deps=self.input_deps,
            acc=self.acc,
            n_symbols=self.n_symbols,
        )

    def cond(
        self, spec: CondSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        return _trace_cond(
            spec=spec,
            eqn_plan=self.eqn_plan,
            frame=self.frame,
            resolver=self.resolver,
            input_deps=self.input_deps,
            acc=self.acc,
            n_symbols=self.n_symbols,
        )

    def linear_solve(
        self, spec: LinearSolveSpec
    ) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
        # Implicit solve AD is not represented by the primal callback bodies.
        # Keep the outer operation conservative and retain callback traces for
        # diagnostics/local planning only.
        rhs = self.input_deps[spec.rhs_indices[0]]
        callback_frames = self.resolver.linear_solve_frames(self.frame, self.eqn_plan)
        traces: list[JaxprDerivativeTrace] = []

        try:
            for callback, child_frame in zip(
                spec.callbacks(), callback_frames, strict=True
            ):
                deps = tuple(
                    rhs
                    if binding.runtime
                    else self.input_deps[binding.outer_input_index]
                    for binding in callback.inputs
                    if binding.outer_input_index is not None
                )
                traces.append(
                    _trace_jaxpr(
                        plan=child_frame.plan,
                        frame=child_frame,
                        resolver=self.resolver,
                        input_deps=deps,
                        acc=self.acc,
                        n_symbols=self.n_symbols,
                    )
                )
        finally:
            for child_frame in callback_frames:
                self.resolver.release(child_frame)

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
            for out in self.eqn_plan.eqn.outvars
        )
        return result, NestedDerivativeTrace(
            LinearSolveInvocation(self.eqn_plan.index, *traces), template=None
        )


def _trace_ordinary_eqn(
    *,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[DependencySet, ...]:
    eqn = eqn_plan.eqn
    semantics = SEMANTICS.get_ordinary(eqn.primitive)

    # output_rows = [
    #     np.arange(int(np.prod(_shape_of(outvar))), dtype=np.int64)
    #     for outvar in eqn.outvars
    # ]
    # request = None if not output_rows else RouteRequest(np.concatenate(output_rows))
    route = resolver.route(frame, eqn_plan)

    ctx = RuleContext(
        eqn=eqn,
        input_deps=input_deps,
        route=route,
        n_symbols=n_symbols,
    )

    prepared = semantics.derivatives.prepare(ctx)
    semantics.derivatives.interactions(ctx, prepared, acc)

    return semantics.derivatives.dependencies(ctx, prepared)


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


def _augment_dependency_width(
    dep: DependencySet,
    extra_columns: int,
) -> DependencySet:
    if extra_columns < 0:
        raise ValueError("extra_columns must be nonnegative")
    if extra_columns == 0:
        return dep
    zeros = sps.csr_matrix((dep.csr.shape[0], extra_columns), dtype=bool)
    return DependencySet(
        sps.hstack((dep.csr, zeros), format="csr"),
        dep.shape,
    )


def _independent_tangent_dependency(
    shape: tuple[int, ...],
    *,
    parent_symbols: int,
    tangent_symbols: int,
    tangent_offset: int,
) -> DependencySet:
    size = int(np.prod(shape, dtype=np.int64))
    rows = np.arange(size, dtype=np.int64)
    cols = parent_symbols + tangent_offset + rows
    csr = sps.csr_matrix(
        (
            np.ones(size, dtype=bool),
            (rows, cols),
        ),
        shape=(size, parent_symbols + tangent_symbols),
        dtype=bool,
    )
    return DependencySet(csr, shape)


def _project_dependency(
    dep: DependencySet,
    projection: sps.csr_matrix,
) -> DependencySet:
    csr = (dep.csr @ projection).astype(bool).tocsr()
    csr.eliminate_zeros()
    return DependencySet(csr, dep.shape)


def _project_jaxpr_derivative_trace(
    trace: JaxprDerivativeTrace,
    projection: sps.csr_matrix,
) -> JaxprDerivativeTrace:
    nested: dict[int, NestedDerivativeTrace] = {}
    for index, item in trace.nested.items():
        invocation = item.invocation.map_children(
            lambda child: _project_jaxpr_derivative_trace(
                child.payload,
                projection,
            )
        )
        nested[index] = NestedDerivativeTrace(
            invocation=invocation, template=item.template
        )

    return JaxprDerivativeTrace(
        dependencies={
            var: _project_dependency(dep, projection)
            for var, dep in trace.dependencies.items()
        },
        output_deps=tuple(
            _project_dependency(dep, projection) for dep in trace.output_deps
        ),
        nested=nested,
    )


def _trace_custom_jvp(
    *,
    spec: CustomJvpSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[tuple[DependencySet, ...], NestedDerivativeTrace]:
    """Use the staged custom-JVP program as the authoritative derivative rule.

    The enclosing symbolic coordinate system is extended temporarily with one
    independent tangent symbol per scalar tangent input.  The JVP is traced in
    ``[parent symbols | tangent symbols]``.  First-order output dependencies
    come from tangent-output -> tangent-symbol incidence.  Second-order
    interactions come from the parent x tangent cross block.  Both are lifted
    through the outer input dependency relation and projected back to the
    enclosing symbolic system.
    """
    eqn = eqn_plan.eqn

    tangent_bindings = tuple(
        binding for binding in spec.jvp_bindings if binding.tangent
    )
    tangent_sizes = tuple(
        int(np.prod(_shape_of(eqn.invars[binding.outer_input_index]), dtype=np.int64))
        for binding in tangent_bindings
    )
    n_tangent_symbols = sum(tangent_sizes)
    n_extended_symbols = n_symbols + n_tangent_symbols

    # Each independent tangent scalar represents the directional variation of
    # the corresponding outer primal scalar.  This sparse lifting relation is
    # also what lets the same custom-JVP machinery work inside energy, weak,
    # and mixed forms without knowing their row/column roles.
    if tangent_bindings:
        tangent_lift = sps.vstack(
            [input_deps[binding.outer_input_index].csr for binding in tangent_bindings],
            format="csr",
        ).astype(bool)
    else:
        tangent_lift = sps.csr_matrix((0, n_symbols), dtype=bool)

    if tangent_lift.shape != (n_tangent_symbols, n_symbols):
        raise RuntimeError(
            "custom_jvp tangent lifting shape mismatch: "
            f"{tangent_lift.shape} != {(n_tangent_symbols, n_symbols)}"
        )

    jvp_input_deps: list[DependencySet] = []
    jvp_acc = InteractionGraph(n_extended_symbols)
    primal_frame, jvp_frame = resolver.custom_jvp_frames(frame, eqn_plan)
    tangent_offset = 0

    for binding, child_var in zip(
        spec.jvp_bindings, jvp_frame.plan.jaxpr.invars, strict=True
    ):
        outer_dep = input_deps[binding.outer_input_index]
        child_shape = _shape_of(child_var)
        if child_shape != outer_dep.shape:
            raise RuntimeError(
                "custom_jvp child input shape differs from its outer binding: "
                f"{child_shape} != {outer_dep.shape}"
            )

        if binding.tangent:
            dep = _independent_tangent_dependency(
                child_shape,
                parent_symbols=n_symbols,
                tangent_symbols=n_tangent_symbols,
                tangent_offset=tangent_offset,
            )
            tangent_offset += int(np.prod(child_shape, dtype=np.int64))
        else:
            dep = _augment_dependency_width(outer_dep, n_tangent_symbols)
        jvp_input_deps.append(dep)

    if tangent_offset != n_tangent_symbols:
        raise RuntimeError("custom_jvp tangent symbol accounting mismatch")

    try:
        primal_trace = _trace_jaxpr(
            plan=primal_frame.plan,
            frame=primal_frame,
            resolver=resolver,
            input_deps=input_deps,
            acc=InteractionGraph(n_symbols),  # diagnostics only
            n_symbols=n_symbols,
        )

        jvp_extended = _trace_jaxpr(
            plan=jvp_frame.plan,
            frame=jvp_frame,
            resolver=resolver,
            input_deps=tuple(jvp_input_deps),
            acc=jvp_acc,
            n_symbols=n_extended_symbols,
        )

    finally:
        resolver.release(jvp_frame)
        resolver.release(primal_frame)

    n_outputs = len(eqn.outvars)
    tangent_cursor = n_outputs
    outputs: list[DependencySet] = []

    for outvar, is_zero in zip(eqn.outvars, spec.output_zeros, strict=True):
        out_shape = _shape_of(outvar)
        if is_zero:
            outputs.append(DependencySet.empty(out_shape, n_symbols))
            continue

        if tangent_cursor >= len(jvp_extended.output_deps):
            raise RuntimeError("custom_jvp tangent output ABI is incomplete")
        tangent_dep = jvp_extended.output_deps[tangent_cursor]
        tangent_cursor += 1

        tangent_columns = tangent_dep.csr[:, n_symbols:]
        lifted = (tangent_columns @ tangent_lift).astype(bool).tocsr()
        lifted.eliminate_zeros()
        outputs.append(DependencySet(lifted, out_shape))

    if tangent_cursor != len(jvp_extended.output_deps):
        raise RuntimeError(
            "custom_jvp tangent output ABI has unexpected trailing outputs"
        )

    # The custom derivative's second-order support is variation of the tangent
    # output with parent coordinates: parent x tangent.  Parent x parent work
    # done while recomputing primal outputs in the callback is deliberately
    # ignored because custom_jvp overrides that derivative semantics.
    jvp_interactions = jvp_acc.finalize()
    cross = jvp_interactions[:n_symbols, n_symbols:]
    outer_cross = (cross @ tangent_lift).astype(bool).tocsr()
    outer_cross.eliminate_zeros()
    if outer_cross.nnz:
        acc.add_pattern(outer_cross, symmetric=True)

    # Nested diagnostics exposed to callers must remain in the parent symbolic
    # coordinate system; the temporary tangent block is an implementation
    # detail of this handler.
    projection = sps.vstack(
        (
            sps.eye(n_symbols, format="csr", dtype=bool),
            tangent_lift,
        ),
        format="csr",
    )
    jvp_trace = _project_jaxpr_derivative_trace(jvp_extended, projection)

    return (
        tuple(outputs),
        NestedDerivativeTrace(
            invocation=CustomJvpInvocation(
                eqn_index=eqn_plan.index, primal=primal_trace, jvp=jvp_trace
            ),
            template=None,
        ),
    )


def _trace_call(
    *,
    spec: CallSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    child_frame = resolver.call_frame(frame, eqn_plan)
    child_input_deps = spec.select_inputs(input_deps)

    try:
        child_trace = _trace_jaxpr(
            plan=child_frame.plan,
            frame=child_frame,
            resolver=resolver,
            input_deps=child_input_deps,
            acc=acc,
            n_symbols=n_symbols,
        )
    finally:
        resolver.release(child_frame)

    return (
        child_trace.output_deps,
        NestedDerivativeTrace(
            invocation=CallInvocation(eqn_plan.index, child_trace), template=None
        ),
    )


def _trace_cond(
    *,
    spec: CondSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    branch_index, child_frame = resolver.cond_frame(frame, eqn_plan)
    child_input_deps = spec.select_inputs(input_deps)

    try:
        body_trace = _trace_jaxpr(
            plan=child_frame.plan,
            frame=child_frame,
            resolver=resolver,
            input_deps=child_input_deps,
            acc=acc,
            n_symbols=n_symbols,
        )
    finally:
        resolver.release(child_frame)

    return (
        body_trace.output_deps,
        NestedDerivativeTrace(
            invocation=CondInvocation(eqn_plan.index, branch_index, body_trace),
            template=None,
        ),
    )


def _trace_scan(
    *,
    spec: ScanSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    nested_plan = eqn_plan.nested
    assert nested_plan is not None

    required_concrete_carry = {
        input_index - spec.num_consts
        for input_index in nested_plan.body.concrete_inputs
        if spec.num_consts <= input_index < spec.num_consts + spec.num_carry
    }
    carry_values = {
        index: resolver.value(frame, eqn_plan.eqn.invars[spec.num_consts + index])
        for index in required_concrete_carry
    }
    const_deps = input_deps[: spec.num_consts]
    carry_deps = input_deps[spec.num_consts : spec.num_consts + spec.num_carry]
    xs_deps = input_deps[spec.num_consts + spec.num_carry :]
    iteration_traces: list[IndexedChild[JaxprDerivativeTrace]] = []

    if spec.num_carry < 0:
        raise RuntimeError("carry-free scan should have been lowered to MapPlan")

    for logical_index in spec.execution_indices():
        x_steps_deps = tuple(
            _leading_axis_dependency(dep, logical_index) for dep in xs_deps
        )
        body_input_deps = const_deps + carry_deps + x_steps_deps
        child_frame = resolver.scan_frame(frame, eqn_plan, logical_index, carry_values)

        try:
            body_trace = _trace_jaxpr(
                plan=child_frame.plan,
                frame=child_frame,
                resolver=resolver,
                input_deps=body_input_deps,
                acc=acc,
                n_symbols=n_symbols,
            )

            # advance only the concrete carry slots needed for later routing
            carry_values = {
                index: resolver.value(
                    child_frame, child_frame.plan.jaxpr.outvars[index]
                )
                for index in required_concrete_carry
            }
        finally:
            resolver.release(child_frame)

        carry_deps = body_trace.output_deps[: spec.num_carry]
        iteration_traces.append(IndexedChild(index=logical_index, body=body_trace))

    # Final carry outputs.
    final_carry = tuple(carry_deps)

    # Stack y outputs in logical scan-axis order.
    stacked_y: list[DependencySet] = []

    for output_index in range(len(eqn_plan.eqn.outvars) - spec.num_carry):
        steps = collect_logical_output(
            ((item.index, item.body.output_deps) for item in iteration_traces),
            output_index=spec.num_carry + output_index,
            length=spec.length,
            label="scan derivative",
        )
        stacked_y.append(_stack_leading_axis_dependencies(list(steps)))

    return (
        final_carry + tuple(stacked_y),
        NestedDerivativeTrace(
            invocation=RepeatedInvocation.from_spec(
                eqn_plan.index, spec, tuple(iteration_traces)
            ),
            template=None,
        ),
    )


def _trace_map(
    *,
    spec: MapSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    if map_template_requires_mapped_concrete(eqn_plan, spec):
        return _trace_map_unrolled(
            spec=spec,
            eqn_plan=eqn_plan,
            frame=frame,
            resolver=resolver,
            input_deps=input_deps,
            acc=acc,
            n_symbols=n_symbols,
        )

    template = _build_map_derivative_template(
        eqn_plan=eqn_plan,
        frame=frame,
        resolver=resolver,
        spec=spec,
    )
    if len(template.output_deps) != len(eqn_plan.eqn.outvars):
        raise RuntimeError(
            f"mapped body returned {len(template.output_deps)} outputs; "
            f"expected {len(eqn_plan.eqn.outvars)}"
        )
    steps_by_output = [[] for _ in eqn_plan.eqn.outvars]

    for logical_index in range(spec.length):
        lifting = _map_iteration_lifting(
            template,
            spec=spec,
            input_deps=input_deps,
            logical_index=logical_index,
            n_symbols=n_symbols,
        )
        for output_index, local_dep in enumerate(template.output_deps):
            lifted = (local_dep.csr @ lifting).astype(bool).tocsr()
            lifted.eliminate_zeros()
            steps_by_output[output_index].append(DependencySet(lifted, local_dep.shape))

        if template.interactions.nnz and lifting.nnz:
            interaction = (
                (lifting.T @ template.interactions @ lifting).astype(bool).tocsr()
            )
            interaction.eliminate_zeros()
            acc.add_pattern(interaction)

    outputs = tuple(
        _stack_leading_axis_dependencies(steps) for steps in steps_by_output
    )
    invocation = MapInvocation(
        eqn_index=eqn_plan.index,
        indices=IterationSelection(length=spec.length),
        body=template.trace,
    )

    return (
        outputs,
        NestedDerivativeTrace(invocation=invocation, template=None),
    )


def _trace_map_unrolled(
    *,
    spec: MapSpec,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    input_deps: tuple[DependencySet, ...],
    acc: InteractionGraph,
    n_symbols: int,
) -> tuple[
    tuple[DependencySet, ...],
    NestedDerivativeTrace,
]:
    const_deps = input_deps[: spec.num_consts]
    mapped_deps = input_deps[spec.num_consts :]
    children: list[IndexedChild[JaxprDerivativeTrace]] = []

    for logical_index in spec.execution_range():
        step_deps = tuple(
            _leading_axis_dependency(dep, logical_index) for dep in mapped_deps
        )
        child_frame = resolver.map_frame(frame, eqn_plan, logical_index)
        try:
            body_trace = _trace_jaxpr(
                plan=child_frame.plan,
                frame=child_frame,
                resolver=resolver,
                input_deps=const_deps + step_deps,
                acc=acc,
                n_symbols=n_symbols,
            )
        finally:
            resolver.release(child_frame)

        if len(body_trace.output_deps) != len(eqn_plan.eqn.outvars):
            raise RuntimeError("mapped body/output ABI mismatch")

        children.append(IndexedChild(logical_index, body_trace))

    outputs: list[DependencySet] = []

    for output_index, outvar in enumerate(eqn_plan.eqn.outvars):
        if spec.length == 0:
            shape = _shape_of(outvar)
            outputs.append(DependencySet.empty(shape, n_symbols))
            continue
        steps = collect_logical_output(
            ((item.index, item.body.output_deps) for item in children),
            output_index=output_index,
            length=spec.length,
            label="map derivative",
        )
        outputs.append(_stack_leading_axis_dependencies(list(steps)))

    return (
        tuple(outputs),
        NestedDerivativeTrace(
            invocation=RepeatedInvocation.from_spec(
                eqn_plan.index, spec, tuple(children)
            ),
            template=None,
        ),
    )


def _build_map_derivative_template(
    *,
    eqn_plan: EqnPlan,
    frame: ConcreteFrame,
    resolver: ConcreteResolver,
    spec: MapSpec,
) -> MapDerivativeTemplate:
    if spec.length <= 0:
        raise ValueError("zero-length map has no body template")

    representative = 0
    child_frame = resolver.map_frame(frame, eqn_plan, representative)

    try:
        shapes = tuple(_shape_of(var) for var in child_frame.plan.jaxpr.invars)

        sizes = tuple(int(np.prod(shape, dtype=np.int64)) for shape in shapes)

        offsets = np.cumsum((0, *sizes[:-1]))
        n_local = sum(sizes)

        local_inputs = []

        for shape, size, offset in zip(shapes, sizes, offsets, strict=True):
            rows = np.arange(size, dtype=np.int64)
            cols = offset + rows

            local_inputs.append(
                DependencySet(
                    sps.csr_matrix(
                        (np.ones(size, dtype=bool), (rows, cols)),
                        shape=(size, n_local),
                    ),
                    shape,
                )
            )

        local_acc = InteractionGraph(n_local)

        trace = _trace_jaxpr(
            plan=child_frame.plan,
            frame=child_frame,
            resolver=resolver,
            input_deps=tuple(local_inputs),
            acc=local_acc,
            n_symbols=n_local,
        )

        return MapDerivativeTemplate(
            input_shapes=shapes,
            output_deps=trace.output_deps,
            interactions=local_acc.finalize(),
            n_local_symbols=n_local,
            trace=trace,
        )

    finally:
        resolver.release(child_frame)


def _map_iteration_lifting(
    template: MapDerivativeTemplate,
    *,
    spec: MapSpec,
    input_deps: tuple[DependencySet, ...],
    logical_index: int,
    n_symbols: int,
) -> sps.csr_matrix:
    if len(input_deps) != len(template.input_shapes):
        raise RuntimeError(
            "map template/input count mismatch"
            f" {len(template.input_shapes)} != {len(input_deps)}"
        )

    rows = []

    for input_index, dep in enumerate(input_deps):
        if input_index < spec.num_consts:
            child_dep = dep
        else:
            child_dep = _leading_axis_dependency(dep, logical_index)

        if child_dep.shape != template.input_shapes[input_index]:
            raise RuntimeError("map template/input shape mismatch")

        rows.append(child_dep.csr)

    lifting = sps.vstack(rows, format="csr").astype(bool)

    expected_shape = (template.n_local_symbols, n_symbols)
    if lifting.shape != expected_shape:
        raise RuntimeError(
            f"invalid map lifting shape {lifting.shape}; expected {expected_shape}"
        )

    return lifting
