"""
Recursive structured backward demand propagation.

The liveness pass starts from rank-owned contribution tensors and walks
backwards through the materialized JAXPR instance tree.

Ordinary primitives delegate to their registered DemandRule. Nested calls,
maps, and scans recurse explicitly because each materialized invocation may
have different resolved structural routes.

Demands always remain structured TensorDemand objects between equations.
Exact flattened scalar rows are used only transiently inside individual
primitive demand rules.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from jax.extend.core import Literal, Var
from numpy.typing import NDArray

from tatva.tracer.core.nested import (
    AnyNestedInvocation,
    CallContext,
    CallInvocation,
    CondContext,
    CondInvocation,
    FrameStep,
    IndexedChild,
    LinearSolveContext,
    LinearSolveInvocation,
    MapContext,
    RepeatedInvocation,
    ScanContext,
    TraversalOrder,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import (
    DemandContext,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    Demand,
    TensorDemand,
    _FullAxis,
    axis_indices,
    lift_leading_axis_demand,
    merge_demands,
    take_leading_axis_demand,
)
from tatva.tracer.program.contributions import ValueRef
from tatva.tracer.program.materialize import (
    JaxprInstance,
    ResolvedEqn,
)


def _expand_batch_demand(
    batch_demand: TensorDemand,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> Demand:
    """Keep selected leading batches and require complete solve-local axes."""
    batch_rank = len(batch_shape)
    if shape[:batch_rank] != batch_shape:
        return None
    return TensorDemand.from_axes(
        shape,
        batch_demand.axes + tuple(_FullAxis() for _ in shape[batch_rank:]),
    )


def _linear_solve_batch_demand(
    context: LinearSolveContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
) -> tuple[tuple[int, ...], TensorDemand] | None:
    """Return the supported non-broadcasted matrix-solve batch demand.

    We deliberately support the same layout as the LU and triangular-solve
    rules: a non-empty leading batch prefix followed by two solve axes.
    """
    # The outer equation is obtained by the caller; callback validation lives
    # here because it needs the materialized callback bodies.
    if len(output_demands) != 1 or output_demands[0] is None:
        return None

    demand = output_demands[0]
    assert demand is not None
    if len(demand.shape) < 3:
        return None

    batch_shape = demand.shape[:-2]
    if not batch_shape:
        return None

    for callback, child in zip(
        context.spec.callbacks(), context.invocation.children(), strict=True
    ):
        if len(child.payload.plan.jaxpr.outvars) != 1:
            return None

        if (
            _shape_of(child.payload.plan.jaxpr.outvars[0])[: len(batch_shape)]
            != batch_shape
        ):
            return None

        runtime_index = next(
            (index for index, binding in enumerate(callback.inputs) if binding.runtime),
            None,
        )
        if runtime_index is None:
            return None

        runtime_shape = _shape_of(child.payload.plan.jaxpr.invars[runtime_index])
        if runtime_shape != demand.shape:
            return None

    return batch_shape, TensorDemand.from_axes(
        batch_shape, demand.axes[: len(batch_shape)]
    )  # ty: ignore[invalid-return-type]


@dataclass(frozen=True, slots=True)
class DemandSeed:
    value: ValueRef
    demand: TensorDemand


type NestedDemandTrace = AnyNestedInvocation[JaxprDemandTrace]


@dataclass(frozen=True)
class JaxprDemandTrace:
    demands: dict[Var, TensorDemand]
    input_demands: tuple[Demand, ...]
    nested: dict[int, NestedDemandTrace]


@dataclass
class _SeedNode:
    values: dict[Var, TensorDemand] = field(default_factory=dict)
    children: dict[FrameStep, _SeedNode] = field(default_factory=dict)


def _build_seed_tree(
    seeds: Iterable[DemandSeed],
) -> _SeedNode:
    root = _SeedNode()

    for seed in seeds:
        node = root

        for step in seed.value.path:
            node = node.children.setdefault(step, _SeedNode())

        existing = node.values.get(seed.value.var)
        merged = merge_demands(existing, seed.demand)

        assert merged is not None
        node.values[seed.value.var] = merged

    return root


def _add_demand(
    demands: dict[Var, TensorDemand],
    atom,
    demand: Demand,
) -> None:
    if demand is None:
        return

    if isinstance(atom, Literal):
        return

    if not isinstance(atom, Var):
        raise TypeError(f"unsupported atom {type(atom)!r}")

    existing = demands.get(atom)
    merged = merge_demands(existing, demand)
    assert merged is not None

    demands[atom] = merged


def _backprop_ordinary(
    resolved: ResolvedEqn,
    output_demands: tuple[Demand, ...],
) -> tuple[Demand, ...]:
    eqn = resolved.plan.eqn
    rule = SEMANTICS.get_ordinary(eqn.primitive)

    result = rule.demand(
        DemandContext(
            eqn=eqn,
            output_demands=output_demands,
            route=resolved.route,
        )
    )

    if len(result) != len(eqn.invars):
        raise RuntimeError(
            f"demand rule for {eqn.primitive.name!r} "
            f"returned {len(result)} inputs; "
            f"expected {len(eqn.invars)}"
        )

    return result


def _backprop_call(
    resolved: ResolvedEqn,
    context: CallContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
    child_seed: _SeedNode | None,
) -> tuple[
    tuple[Demand, ...],
    CallInvocation[JaxprDemandTrace],
]:
    nested = context.invocation

    child = _backprop_jaxpr(
        nested.body,
        child_seed or _SeedNode(),
        output_demands=output_demands,
    )

    outer_demands: list[Demand] = [None] * len(resolved.plan.eqn.invars)
    input_indices = context.spec.resolved_input_indices(len(outer_demands))

    if len(child.input_demands) != len(input_indices):
        raise RuntimeError(
            f"{resolved.plan.eqn.primitive.name} child demand/input boundary mismatch"
        )

    for outer_index, demand in zip(input_indices, child.input_demands, strict=True):
        outer_demands[outer_index] = merge_demands(outer_demands[outer_index], demand)

    return (
        tuple(outer_demands),
        CallInvocation(eqn_index=nested.eqn_index, body=child),
    )


def _backprop_cond(
    resolved: ResolvedEqn,
    context: CondContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
    child_seed: _SeedNode | None,
) -> tuple[
    tuple[Demand, ...],
    CondInvocation[JaxprDemandTrace],
]:
    nested = context.invocation

    child = _backprop_jaxpr(
        nested.body,
        child_seed or _SeedNode(),
        output_demands=output_demands,
    )

    outer_demands: list[Demand] = [None] * len(resolved.plan.eqn.invars)
    for child_index, demand in enumerate(child.input_demands):
        outer_index = context.spec.outer_input_index(
            child_index, outer_arity=len(outer_demands)
        )
        outer_demands[outer_index] = merge_demands(outer_demands[outer_index], demand)

    return (
        tuple(outer_demands),
        CondInvocation(
            eqn_index=nested.eqn_index,
            branch_index=nested.branch_index,
            body=child,
        ),
    )


def _map_demanded_indices(
    context: MapContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
    *,
    eqn_index: int,
) -> NDArray[np.int64]:
    spec = context.spec
    parts = []

    for demand in output_demands:
        if demand is None:
            continue

        first = demand.axis_subset(0)
        parts.append(axis_indices(first, extent=spec.length))

    child_steps = {child.frame_step for child in context.invocation.children()}
    explicit = [
        step.iteration
        for step in seed_node.children
        if step in child_steps and step.iteration is not None
    ]

    if explicit:
        parts.append(np.asarray(explicit, dtype=np.int64))

    if not parts:
        return np.empty(0, dtype=np.int64)

    return np.unique(np.concatenate(parts))


def _backprop_map(
    resolved: ResolvedEqn,
    context: MapContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> tuple[
    tuple[Demand, ...],
    RepeatedInvocation[JaxprDemandTrace],
]:
    nested = context.invocation
    eqn = resolved.plan.eqn

    spec = context.spec

    input_demands: list[Demand] = [None] * len(eqn.invars)
    traces: list[IndexedChild[JaxprDemandTrace]] = []
    indices = _map_demanded_indices(
        context,
        output_demands,
        seed_node,
        eqn_index=resolved.plan.index,
    )

    for logical_index in indices:
        logical_index = int(logical_index)
        iteration = nested.child_at_index(logical_index)
        child_step = nested.frame_step(logical_index)

        child_seed = seed_node.children.get(child_step, _SeedNode())

        body_outputs = tuple(
            take_leading_axis_demand(demand, logical_index) for demand in output_demands
        )

        child = _backprop_jaxpr(iteration, child_seed, output_demands=body_outputs)

        traces.append(
            IndexedChild(
                index=logical_index,
                body=child,
            )
        )

        for input_index, demand in enumerate(child.input_demands):
            if demand is None:
                continue

            if input_index < spec.num_consts:
                lifted = demand

            else:
                outer_shape = _shape_of(eqn.invars[input_index])
                lifted = lift_leading_axis_demand(
                    demand,
                    outer_shape=outer_shape,
                    index=logical_index,
                )

            input_demands[input_index] = merge_demands(
                input_demands[input_index], lifted
            )

    return (
        tuple(input_demands),
        nested.with_children(tuple(traces)),
    )


def _backprop_scan(
    resolved: ResolvedEqn,
    context: ScanContext[JaxprInstance],
    output_demands: tuple[Demand, ...],
    seed_node: _SeedNode,
) -> tuple[
    tuple[Demand, ...],
    RepeatedInvocation[JaxprDemandTrace],
]:
    nested = context.invocation
    eqn = resolved.plan.eqn

    spec = context.spec

    if spec.num_carry <= 0:
        raise RuntimeError("carry-free scan should be MapPlan")

    num_consts = spec.num_consts
    num_carry = spec.num_carry

    num_xs = len(eqn.invars) - num_consts - num_carry
    carry_demands = list(output_demands[:num_carry])
    y_demands = output_demands[num_carry:]
    const_demands: list[Demand] = [None] * num_consts
    xs_demands: list[Demand] = [None] * num_xs
    traces: list[IndexedChild[JaxprDemandTrace]] = []

    # nested.iterations is execution order.
    # Backward liveness runs opposite execution order.
    for nested_child in nested.children(TraversalOrder.REVERSE_EXECUTION):
        logical_index = nested_child.logical_index
        assert logical_index is not None
        child_step = nested_child.frame_step
        child_seed = seed_node.children.get(child_step)
        y_step_demands = tuple(
            take_leading_axis_demand(demand, logical_index) for demand in y_demands
        )

        body_outputs = tuple(carry_demands) + y_step_demands

        if all(demand is None for demand in body_outputs) and child_seed is None:
            continue

        child = _backprop_jaxpr(
            nested_child.payload,
            child_seed or _SeedNode(),
            output_demands=body_outputs,
        )

        traces.append(IndexedChild(index=logical_index, body=child))
        inputs = child.input_demands

        # Shared constants.
        for i in range(num_consts):
            const_demands[i] = merge_demands(const_demands[i], inputs[i])

        # Carry demand propagates to the previous iteration in
        # execution order.
        carry_demands = list(inputs[num_consts : num_consts + num_carry])

        # Scanned inputs.
        x_start = num_consts + num_carry

        for x_index in range(num_xs):
            demand = inputs[x_start + x_index]

            if demand is None:
                continue

            outer_input_index = x_start + x_index
            lifted = lift_leading_axis_demand(
                demand,
                outer_shape=_shape_of(eqn.invars[outer_input_index]),
                index=logical_index,
            )
            xs_demands[x_index] = merge_demands(xs_demands[x_index], lifted)

    result = tuple(const_demands) + tuple(carry_demands) + tuple(xs_demands)

    return (
        result,
        nested.with_children(tuple(traces)),
    )


@dataclass(frozen=True)
class _DemandNestedHandler:
    resolved: ResolvedEqn
    output_demands: tuple[Demand, ...]
    seed_node: _SeedNode

    def call(
        self, context: CallContext[JaxprInstance]
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        child_step = context.invocation.children()[0].frame_step
        return _backprop_call(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node.children.get(child_step),
        )

    def map(
        self, context: MapContext[JaxprInstance]
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        return _backprop_map(
            self.resolved, context, self.output_demands, self.seed_node
        )

    def scan(
        self, context: ScanContext[JaxprInstance]
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        return _backprop_scan(
            self.resolved, context, self.output_demands, self.seed_node
        )

    def cond(
        self, context: CondContext[JaxprInstance]
    ) -> tuple[tuple[Demand, ...], NestedDemandTrace]:
        child_step = context.invocation.children()[0].frame_step
        return _backprop_cond(
            self.resolved,
            context,
            self.output_demands,
            self.seed_node.children.get(child_step),
        )

    def linear_solve(self, context: LinearSolveContext[JaxprInstance]):
        eqn = self.resolved.plan.eqn
        outer: list[Demand] = [None] * len(eqn.invars)
        traces = []
        batch = _linear_solve_batch_demand(context, self.output_demands)
        if (
            batch is None
            or _shape_of(eqn.invars[context.spec.rhs_indices[0]])
            != self.output_demands[0].shape  # ty: ignore[unresolved-attribute]
        ):
            warnings.warn(
                "custom_linear_solve does not have a supported batched matrix "
                "layout; using conservative full callback demands",
                UserWarning,
                stacklevel=2,
            )
            batch_shape = None
            batch_demand = None
        else:
            batch_shape, batch_demand = batch

        # The solution demand enters every executable callback.  Only solve's
        # runtime RHS maps back to the outer RHS operand.
        for callback, child_node in zip(
            context.spec.callbacks(), context.invocation.children(), strict=True
        ):
            outputs = tuple(
                (
                    TensorDemand.full(_shape_of(atom))
                    if (batch_demand is None or batch_shape is None)
                    else _expand_batch_demand(
                        batch_demand,
                        shape=_shape_of(atom),
                        batch_shape=batch_shape,
                    )
                )
                for atom in child_node.payload.plan.jaxpr.outvars
            )
            child = _backprop_jaxpr(
                child_node.payload,
                self.seed_node.children.get(child_node.frame_step, _SeedNode()),
                output_demands=outputs,
            )
            traces.append(child)

            for binding, demand in zip(
                callback.inputs, child.input_demands, strict=True
            ):
                if binding.runtime:
                    if callback.name == "solve" and demand is not None:
                        outer[context.spec.rhs_indices[0]] = merge_demands(
                            outer[context.spec.rhs_indices[0]], demand
                        )
                elif demand is not None:
                    i = binding.outer_input_index
                    assert i is not None
                    outer[i] = merge_demands(outer[i], demand)

            # Captures are closure operands of an executable callback. Keep
            # them live even when a callback body's structural demand rule
            # cannot see through an opaque linear-algebra primitive.
            for binding in callback.inputs:
                if binding.outer_input_index is not None:
                    i = binding.outer_input_index
                    required = (
                        TensorDemand.full(_shape_of(eqn.invars[i]))
                        if (batch_demand is None or batch_shape is None)
                        else _expand_batch_demand(
                            batch_demand,
                            shape=_shape_of(eqn.invars[i]),
                            batch_shape=batch_shape,
                        )
                    )
                    outer[i] = merge_demands(outer[i], required)

        return tuple(outer), LinearSolveInvocation(
            context.invocation.eqn_index, *traces
        )


def _backprop_jaxpr(
    instance: JaxprInstance,
    seed_node: _SeedNode,
    *,
    output_demands: tuple[Demand, ...] | None = None,
) -> JaxprDemandTrace:
    jaxpr = instance.plan.jaxpr
    demands: dict[Var, TensorDemand] = {}
    nested_traces: dict[int, NestedDemandTrace] = {}

    # Explicit contribution/internal seeds in this frame.
    for var, demand in seed_node.values.items():
        _add_demand(demands, var, demand)

    # Demands arriving through a nested wrapper boundary.
    if output_demands is not None:
        if len(output_demands) != len(jaxpr.outvars):
            raise ValueError(
                f"received {len(output_demands)} output demands "
                f"for JAXPR with {len(jaxpr.outvars)} outputs"
            )

        for atom, demand in zip(jaxpr.outvars, output_demands):
            _add_demand(demands, atom, demand)

    # Reverse topological traversal.
    for resolved in reversed(instance.eqns):
        eqn = resolved.plan.eqn
        eqn_output_demands = tuple(demands.get(outvar) for outvar in eqn.outvars)
        eqn_index = resolved.plan.index

        # Ordinary primitive
        if resolved.nested is None:
            if all(demand is None for demand in eqn_output_demands):
                continue

            input_demands = _backprop_ordinary(resolved, eqn_output_demands)

        else:
            nested_plan = resolved.plan.nested
            if nested_plan is None:
                raise TypeError("nested invocation has no analysis plan")

            has_child_seed = any(
                child.frame_step in seed_node.children
                for child in resolved.nested.children()
            )

            if (
                all(demand is None for demand in eqn_output_demands)
                and not has_child_seed
            ):
                continue

            input_demands, nested_trace = dispatch_nested(
                nested_plan.spec,
                resolved.nested,
                _DemandNestedHandler(resolved, eqn_output_demands, seed_node),
            )
            nested_traces[eqn_index] = nested_trace

        # Merge producer requirements.
        if len(input_demands) != len(eqn.invars):
            raise RuntimeError(
                f"{eqn.primitive.name!r} demand propagation "
                "returned the wrong number of inputs"
            )

        for atom, demand in zip(eqn.invars, input_demands):
            _add_demand(demands, atom, demand)

    frame_inputs = tuple(demands.get(var) for var in jaxpr.invars)

    return JaxprDemandTrace(
        demands=demands,
        input_demands=frame_inputs,
        nested=nested_traces,
    )


def backpropagate_demand(
    root: JaxprInstance,
    seeds: Iterable[DemandSeed],
) -> JaxprDemandTrace:
    seed_tree = _build_seed_tree(seeds)

    return _backprop_jaxpr(root, seed_tree)
