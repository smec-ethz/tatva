"""
Build executable JAX functions from LocalJaxprPlan.

The equation walk happens in Python during JAX tracing. Primitive operations
emitted by that walk become the compiled JAX computation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.extend.core import Literal, Var
from numpy.typing import NDArray

from tatva.tracer.core.nested import (
    CallContext,
    CondContext,
    LinearSolveContext,
    MapContext,
    RepeatedInvocation,
    ScanContext,
    ScanSpec,
    TraversalOrder,
    dispatch_nested,
)
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.local.demand import (
    TensorDemand,
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
)
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.localize import (
    LocalDynamicSliceRoute,
    LocalGatherRoute,
    LocalScatterRoute,
    LocalSelectNRoute,
)
from tatva.tracer.local.plan import (
    LocalEqnPlan,
    LocalJaxprPlan,
    LocalNestedPlan,
)
from tatva.tracer.lowering.partition import OwnedContribution
from tatva.tracer.lowering.rules import LoweringContext, lower_bind
from tatva.tracer.program.contributions import ContributionTrace


def extract_local_value(
    value,
    layout: TensorLayout,
) -> Array:
    """Convenience helper for testing/local input construction.

    Production distributed execution can supply already-local tensors
    directly.
    """
    result = jnp.asarray(value)

    if tuple(result.shape) != layout.global_shape:
        raise ValueError(
            f"global input shape {result.shape} does not "
            f"match layout {layout.global_shape}"
        )

    for axis in range(layout.ndim):
        subset = layout.axis_subset(axis)

        if isinstance(subset, _FullAxis):
            continue

        if isinstance(subset, _RangeAxis):
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(subset.start, subset.stop)
            result = result[tuple(slices)]
            continue

        if isinstance(subset, _IndexAxis):
            result = jnp.take(result, jnp.asarray(subset.indices), axis=axis)
            continue

        raise TypeError(f"unsupported axis subset {type(subset)!r}")

    if tuple(result.shape) != layout.local_shape:
        raise RuntimeError(
            f"localized value has shape {result.shape}; expected {layout.local_shape}"
        )

    return result


def _project_local_value(
    value,
    *,
    source_layout: TensorLayout,
    target_layout: TensorLayout,
):
    """Extract `target_layout` from a tensor currently stored using
    `source_layout`.

    Both layouts describe the same global tensor. target_layout must be a
    subset of source_layout.

    This handles the important case where a frame-level layout is a
    structured hull larger than the exact subset consumed by one child.
    """
    if source_layout.global_shape != target_layout.global_shape:
        raise ValueError(
            "cannot project between different global tensors: "
            f"{source_layout.global_shape} != "
            f"{target_layout.global_shape}"
        )

    if source_layout.local_shape == target_layout.local_shape and all(
        np.array_equal(
            source_layout.global_axis_indices(axis),
            target_layout.global_axis_indices(axis),
        )
        for axis in range(source_layout.ndim)
    ):
        return value

    target_local_rows = np.arange(target_layout.local_size, dtype=np.int64)
    target_global_rows = target_layout.local_rows_to_global_rows(target_local_rows)

    try:
        source_local_rows = source_layout.global_rows_to_local_rows(target_global_rows)

    except ValueError as exc:
        raise ValueError("target layout is not contained in source layout") from exc

    result = jnp.ravel(value)[jnp.asarray(source_local_rows)]
    return jnp.reshape(result, target_layout.local_shape)


def _read_atom(
    atom,
    layout: TensorLayout | None,
    env: dict[Var, Any],
):
    if isinstance(atom, Literal):
        return atom.val

    if not isinstance(atom, Var):
        raise TypeError(f"unsupported atom {type(atom)!r}")

    if layout is None:
        return None

    try:
        return env[atom]
    except KeyError as exc:
        raise RuntimeError(f"live variable {atom} is unavailable") from exc


def _validate_outputs(
    plan: LocalEqnPlan,
    outputs: tuple[Any | None, ...],
) -> None:
    if len(outputs) != len(plan.eqn.outvars):
        raise RuntimeError(
            f"{plan.primitive_name}: lowering returned "
            f"{len(outputs)} outputs; expected "
            f"{len(plan.eqn.outvars)}"
        )

    for index, (value, layout) in enumerate(zip(outputs, plan.output_layouts)):
        if layout is None:
            continue

        if value is None:
            raise RuntimeError(f"{plan.primitive_name}: live output {index} is None")

        shape = tuple(value.shape)

        if shape != layout.local_shape:
            raise RuntimeError(
                f"eqn {plan.index} "
                f"{plan.primitive_name}: "
                f"local output {index} has shape "
                f"{shape}, expected "
                f"{layout.local_shape}"
            )


def _frame_outputs(
    plan: LocalJaxprPlan,
    env: dict[Var, Any],
) -> tuple[Any | None, ...]:
    result: list[Any | None] = []

    for atom, layout in zip(plan.plan.jaxpr.outvars, plan.output_layouts):
        if layout is None:
            result.append(None)
            continue

        if isinstance(atom, Literal):
            result.append(atom.val)
            continue

        result.append(env[atom])

    return tuple(result)


def _lower_call(
    plan: LocalEqnPlan,
    context: CallContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    body = context.invocation.body
    input_indices = context.spec.resolved_input_indices(len(inputs))

    if len(input_indices) != len(body.input_layouts):
        raise RuntimeError(
            f"{plan.primitive_name}: call boundary selects {len(input_indices)} inputs "
            f"but child plan has {len(body.input_layouts)} inputs"
        )

    child_inputs: list[Any] = []

    for child_index, outer_index in enumerate(input_indices):
        target_layout = body.input_layouts[child_index]
        if target_layout is None:
            continue

        value = inputs[outer_index]
        source_layout = plan.input_layouts[outer_index]

        if value is None or source_layout is None:
            raise RuntimeError(
                f"{plan.primitive_name}: live child input {child_index} "
                f"maps to dead outer input {outer_index}"
            )

        child_inputs.append(
            _project_local_value(
                value,
                source_layout=source_layout,
                target_layout=target_layout,
            )
        )

    env = _execute_frame(body, tuple(child_inputs))
    child_outputs = _frame_outputs(body, env)

    result: list[Any | None] = []
    for output_index, (value, source_layout, target_layout) in enumerate(
        zip(
            child_outputs,
            body.output_layouts,
            plan.output_layouts,
            strict=True,
        )
    ):
        if target_layout is None:
            result.append(None)
            continue

        if value is None or source_layout is None:
            raise RuntimeError(
                f"{plan.primitive_name}: live outer output {output_index} "
                "is unavailable from child call"
            )

        result.append(
            _project_local_value(
                value,
                source_layout=source_layout,
                target_layout=target_layout,
            )
        )

    return tuple(result)


def _lower_cond(
    plan: LocalEqnPlan,
    context: CondContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    body = context.invocation.body
    child_inputs: list[Any] = []

    for child_index in range(len(body.input_layouts)):
        target_layout = body.input_layouts[child_index]
        if target_layout is None:
            continue

        outer_index = context.spec.outer_input_index(
            child_index, outer_arity=len(inputs)
        )
        value = inputs[outer_index]
        source_layout = plan.input_layouts[outer_index]

        if value is None or source_layout is None:
            raise RuntimeError(
                f"{plan.primitive_name}: live child input {child_index} "
                f"maps to dead outer input {outer_index}"
            )

        child_inputs.append(
            _project_local_value(
                value,
                source_layout=source_layout,
                target_layout=target_layout,
            )
        )

    env = _execute_frame(body, tuple(child_inputs))
    child_outputs = _frame_outputs(body, env)

    result: list[Any | None] = []
    for output_index, (value, source_layout, target_layout) in enumerate(
        zip(
            child_outputs,
            body.output_layouts,
            plan.output_layouts,
            strict=True,
        )
    ):
        if target_layout is None:
            result.append(None)
            continue

        if value is None or source_layout is None:
            raise RuntimeError(
                f"{plan.primitive_name}: live outer output {output_index} "
                "is unavailable from child cond"
            )

        result.append(
            _project_local_value(
                value,
                source_layout=source_layout,
                target_layout=target_layout,
            )
        )

    return tuple(result)


def _lower_map(
    plan: LocalEqnPlan,
    context: MapContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    template = _common_repeated_template(context.invocation)
    if template is None:
        raise NotImplementedError(
            "localized map iterations have different body layouts/routes; "
            "a single lax.map template cannot represent them"
        )

    logical_indices = np.asarray(
        [
            child.logical_index
            for child in context.invocation.children()
            if child.logical_index is not None
        ],
        dtype=np.int64,
    )
    n_iterations = logical_indices.size

    if n_iterations == 0:
        raise RuntimeError("live local map has no surviving iterations")

    if len(inputs) != len(template.input_layouts):
        raise RuntimeError(
            "map outer/body input arity mismatch: "
            f"{len(inputs)} != "
            f"{len(template.input_layouts)}"
        )

    if context.spec.num_consts > len(inputs):
        raise RuntimeError("map num_consts exceeds input arity")

    body_constants = _project_repeated_constants(
        plan,
        template,
        inputs,
        num_consts=context.spec.num_consts,
        label="map",
    )

    mapped_input_indices, mapped_values = _stack_repeated_inputs(
        plan,
        template,
        inputs,
        start=context.spec.num_consts,
        logical_indices=logical_indices,
        label="map",
    )

    mapped_slot = {
        input_index: slot for slot, input_index in enumerate(mapped_input_indices)
    }

    # Determine live body outputs.
    live_output_indices = tuple(
        index
        for index, layout in enumerate(template.output_layouts)
        if layout is not None
    )

    if not live_output_indices:
        raise RuntimeError("live map has no live body outputs")

    # Every live outer output should correspond to a live body output.
    for output_index, (outer_layout, body_layout) in enumerate(
        zip(plan.output_layouts, template.output_layouts)
    ):
        if outer_layout is None:
            if body_layout is not None:
                raise RuntimeError(
                    f"map output {output_index} is dead outside "
                    "but live in body template"
                )

            continue

        if body_layout is None:
            raise RuntimeError(
                f"map output {output_index} is live outside but dead in body template"
            )

        _validate_repeated_output_layout(
            outer_layout=outer_layout,
            body_layout=body_layout,
            logical_indices=logical_indices,
            label="map",
        )

    # One compiled iteration body.
    def body(
        mapped_slices,
    ):
        # lax.map passes one slice from every leaf in the xs pytree.
        if mapped_values:
            slices = mapped_slices
        else:
            slices = ()

        local_inputs: list[Any] = []

        for input_index, body_layout in enumerate(template.input_layouts):
            if body_layout is None:
                continue

            if input_index < context.spec.num_consts:
                local_inputs.append(body_constants[input_index])

            else:
                slot = mapped_slot[input_index]
                local_inputs.append(slices[slot])

        env = _execute_frame(template, tuple(local_inputs))
        outputs = _frame_outputs(template, env)

        return tuple(outputs[index] for index in live_output_indices)

    # Execute the map.
    if mapped_values:
        xs = tuple(mapped_values)
    else:
        # lax.map needs a leading iteration axis. The values are ignored.
        xs = jnp.arange(n_iterations, dtype=jnp.int32)

    mapped_outputs = lax.map(body, xs)

    # Body always returns a tuple, so mapped_outputs is a tuple too.
    if not isinstance(mapped_outputs, tuple):
        mapped_outputs = (mapped_outputs,)

    if len(mapped_outputs) != len(live_output_indices):
        raise RuntimeError("lax.map output arity mismatch")

    by_output_index = {
        output_index: value
        for output_index, value in zip(live_output_indices, mapped_outputs)
    }

    result: list[Any | None] = []

    for output_index, layout in enumerate(plan.output_layouts):
        if layout is None:
            result.append(None)
            continue

        value = by_output_index[output_index]

        if tuple(value.shape) != layout.local_shape:
            raise RuntimeError(
                f"localized map output {output_index} "
                f"has shape {value.shape}; "
                f"expected {layout.local_shape}"
            )

        result.append(value)

    return tuple(result)


def _lower_scan(
    plan: LocalEqnPlan,
    context: ScanContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    invocation = context.invocation
    spec = context.spec
    template = _common_repeated_template(invocation)
    if template is None:
        return _lower_scan_unrolled(plan, context, inputs)
    _validate_scan_iteration_subset(context, template)

    logical_indices = np.asarray(
        tuple(
            child.logical_index
            for child in invocation.children(TraversalOrder.LOGICAL)
            if child.logical_index is not None
        ),
        dtype=np.int64,
    )
    n_iterations = logical_indices.size

    num_consts = spec.num_consts
    num_carry = spec.num_carry

    if len(inputs) != len(template.input_layouts):
        raise RuntimeError("scan parent/body input arity mismatch")

    # Constants
    body_constants = _project_repeated_constants(
        plan,
        template,
        inputs,
        num_consts=num_consts,
        label="scan",
    )

    # Carry
    live_carry_indices = _scan_live_carry_indices(spec, template)
    carry_slots = {
        carry_index: slot for slot, carry_index in enumerate(live_carry_indices)
    }
    initial_carry = []

    for carry_index in live_carry_indices:
        parent_input_index = num_consts + carry_index
        value = inputs[parent_input_index]
        outer_layout = plan.input_layouts[parent_input_index]
        body_layout = template.input_layouts[parent_input_index]

        if value is None or body_layout is None:
            raise RuntimeError(f"live scan carry {carry_index} is unavailable")

        initial_carry.append(
            _project_repeated_value(
                plan,
                parent_input_index,
                value,
                outer_layout=outer_layout,
                body_layout=body_layout,
                label="scan carry",
            )
        )

    initial_carry = tuple(initial_carry)

    # Scanned inputs xs
    xs_start = num_consts + num_carry
    mapped_input_indices, stacked_xs = _stack_repeated_inputs(
        plan,
        template,
        inputs,
        start=xs_start,
        logical_indices=logical_indices,
        label="scan xs",
    )

    x_slots = {
        input_index: slot for slot, input_index in enumerate(mapped_input_indices)
    }

    # Live stacked y outputs
    body_y_start = num_carry
    live_y_indices = []

    for parent_output_index in range(num_carry, len(plan.output_layouts)):
        y_index = parent_output_index - num_carry
        body_output_index = body_y_start + y_index
        outer_layout = plan.output_layouts[parent_output_index]
        body_layout = template.output_layouts[body_output_index]

        if outer_layout is None and body_layout is None:
            continue

        if outer_layout is None or body_layout is None:
            raise RuntimeError(
                f"scan y output {y_index} has inconsistent parent/body liveness"
            )

        _validate_repeated_output_layout(
            outer_layout=outer_layout,
            body_layout=body_layout,
            logical_indices=logical_indices,
            label="scan",
        )
        live_y_indices.append(y_index)

    live_y_indices = tuple(live_y_indices)

    # Compiled scan body
    def body(packed_carry, packed_xs):
        body_inputs: list[Any] = []

        for input_index, body_layout in enumerate(template.input_layouts):
            if body_layout is None:
                continue

            # Constants
            if input_index < num_consts:
                body_inputs.append(body_constants[input_index])
                continue

            # Carries
            if input_index < (num_consts + num_carry):
                carry_index = input_index - num_consts
                slot = carry_slots[carry_index]
                body_inputs.append(packed_carry[slot])
                continue

            # xs
            slot = x_slots[input_index]
            body_inputs.append(packed_xs[slot])

        env = _execute_frame(template, tuple(body_inputs))
        outputs = _frame_outputs(template, env)
        next_carry = tuple(outputs[carry_index] for carry_index in live_carry_indices)
        ys = tuple(outputs[num_carry + y_index] for y_index in live_y_indices)

        return (next_carry, ys)

    # Run scan
    xs = tuple(stacked_xs) if stacked_xs else None

    final_carry, stacked_ys = jax.lax.scan(
        body,
        initial_carry,
        xs,
        length=n_iterations,
        reverse=spec.reverse,
    )

    # Reconstruct parent outputs
    carry_by_index = {
        carry_index: value
        for carry_index, value in zip(live_carry_indices, final_carry)
    }
    y_by_index = {y_index: value for y_index, value in zip(live_y_indices, stacked_ys)}

    result: list[Any | None] = []

    for output_index, outer_layout in enumerate(plan.output_layouts):
        if outer_layout is None:
            result.append(None)
            continue

        # Final carry
        if output_index < num_carry:
            carry_index = output_index
            value = carry_by_index[carry_index]
            body_layout = template.output_layouts[carry_index]
            assert body_layout is not None

            # Usually identical, but projecting makes the boundary robust
            # against a larger body hull.
            value = _project_local_value(
                value,
                source_layout=body_layout,
                target_layout=outer_layout,
            )

            result.append(value)
            continue

        # Stacked ys
        y_index = output_index - num_carry
        result.append(y_by_index[y_index])

    return tuple(result)


def _lower_scan_unrolled(
    plan: LocalEqnPlan,
    context: ScanContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    """Lower a scan whose localized iterations do not share one signature.

    The Python loop runs while JAX traces the local function, so this still emits
    an ordinary JAX program; it does not interpret the scan at runtime.
    """
    spec = context.spec
    _validate_scan_iteration_subset(context, context.invocation.iterations[0].body)
    carry: dict[int, tuple[Any, TensorLayout | None]] = {}
    ys: dict[tuple[int, int], tuple[Any, TensorLayout]] = {}

    for carry_index in range(spec.num_carry):
        input_index = spec.num_consts + carry_index
        value = inputs[input_index]
        layout = plan.input_layouts[input_index]
        if value is not None:
            carry[carry_index] = (value, layout)

    for child in context.invocation.children(TraversalOrder.EXECUTION):
        logical_index = child.logical_index
        assert logical_index is not None
        body_plan = child.payload
        body_inputs: list[Any] = []

        for input_index, body_layout in enumerate(body_plan.input_layouts):
            if body_layout is None:
                continue
            if input_index < spec.num_consts:
                value = inputs[input_index]
                if value is None:
                    raise RuntimeError(
                        f"live scan constant {input_index} is unavailable"
                    )
                body_inputs.append(
                    _project_repeated_value(
                        plan,
                        input_index,
                        value,
                        outer_layout=plan.input_layouts[input_index],
                        body_layout=body_layout,
                        label="scan constant",
                    )
                )
                continue

            if input_index < spec.num_consts + spec.num_carry:
                carry_index = input_index - spec.num_consts
                try:
                    value, source_layout = carry[carry_index]
                except KeyError as exc:
                    raise RuntimeError(
                        f"scan carry {carry_index} is unavailable at iteration "
                        f"{logical_index}"
                    ) from exc
                if source_layout is None:
                    body_inputs.append(
                        _project_repeated_value(
                            plan,
                            spec.num_consts + carry_index,
                            value,
                            outer_layout=None,
                            body_layout=body_layout,
                            label="scan carry",
                        )
                    )
                else:
                    body_inputs.append(
                        _project_local_value(
                            value,
                            source_layout=source_layout,
                            target_layout=body_layout,
                        )
                    )
                continue

            value = inputs[input_index]
            outer_layout = plan.input_layouts[input_index]
            if value is None or outer_layout is None:
                raise RuntimeError(f"live scan xs input {input_index} is unavailable")
            body_inputs.append(
                _stack_repeated_input(
                    value,
                    outer_layout=outer_layout,
                    body_layout=body_layout,
                    logical_indices=np.asarray([logical_index], dtype=np.int64),
                )[0]
            )

        outputs = _frame_outputs(
            body_plan, _execute_frame(body_plan, tuple(body_inputs))
        )
        for carry_index in range(spec.num_carry):
            value = outputs[carry_index]
            layout = body_plan.output_layouts[carry_index]
            if value is not None and layout is not None:
                carry[carry_index] = (value, layout)

        for y_index in range(len(outputs) - spec.num_carry):
            output_index = spec.num_carry + y_index
            value = outputs[output_index]
            layout = body_plan.output_layouts[output_index]
            if value is not None and layout is not None:
                ys[logical_index, y_index] = value, layout

    result: list[Any | None] = []
    for output_index, outer_layout in enumerate(plan.output_layouts):
        if outer_layout is None:
            result.append(None)
            continue
        if output_index < spec.num_carry:
            value, source_layout = carry[output_index]
            if source_layout is None:
                raise RuntimeError(
                    f"scan final carry {output_index} has no localized layout"
                )
            result.append(
                _project_local_value(
                    value,
                    source_layout=source_layout,
                    target_layout=outer_layout,
                )
            )
            continue

        y_index = output_index - spec.num_carry
        values = []
        for logical_index in outer_layout.global_axis_indices(0):
            try:
                value, body_layout = ys[int(logical_index), y_index]
            except KeyError as exc:
                raise RuntimeError(
                    f"scan y output {y_index} is unavailable at iteration "
                    f"{logical_index}"
                ) from exc
            _validate_repeated_output_layout(
                outer_layout=outer_layout,
                body_layout=body_layout,
                logical_indices=outer_layout.global_axis_indices(0),
                label="scan",
            )
            values.append(value)
        result.append(jnp.stack(values, axis=0))

    return tuple(result)


def _lower_nested(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    if plan.nested is None:
        raise TypeError("expected a nested local plan")
    return dispatch_nested(
        plan.nested.spec,
        plan.nested.invocation,
        _LowerNestedHandler(plan, inputs),
    )


def _lower_linear_solve(
    plan: LocalEqnPlan,
    context: LinearSolveContext[LocalJaxprPlan],
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    """Reconstruct ``custom_linear_solve`` instead of executing ``solve`` directly.

    JAX owns the implicit differentiation rule of the reconstructed primitive. This is
    essential because the primal ``solve`` callback can be correct while being
    deliberately unsuitable for algorithmic differentiation (for example, by containing
    ``stop_gradient``).  Each localized callback closes over its captured parent values,
    projects them into its own local layout, and uses its final runtime argument as the
    RHS; JAX passes ``(matvec, rhs)`` to the solve callbacks and just ``rhs`` to matvec.

    The current local ABI supports one RHS/result, ``has_aux=False``, and an explicit
    transpose solve.  Unsupported solve layouts are handled earlier by liveness with
    conservative full demands; captures must still be live in the parent local plan when
    this function is traced.
    """

    def callback(body, bindings):
        def fn(*runtime_args):
            rhs = runtime_args[-1]
            values = []
            for index, binding in enumerate(bindings):
                if binding.runtime:
                    values.append(rhs)
                else:
                    value = inputs[binding.outer_input_index]
                    if value is None:
                        raise RuntimeError(
                            "live custom_linear_solve capture is unavailable"
                        )
                    source = plan.input_layouts[binding.outer_input_index]
                    target = body.input_layouts[index]
                    values.append(
                        value
                        if source is None or target is None
                        else _project_local_value(
                            value, source_layout=source, target_layout=target
                        )
                    )
            # Child local plans may omit dead captured values, so pass precisely
            # the live inputs in JAXPR order.
            local = tuple(
                v
                for v, layout in zip(values, body.input_layouts, strict=True)
                if layout is not None
            )
            return _frame_outputs(body, _execute_frame(body, local))[0]

        return fn

    spec = context.spec
    rhs = inputs[spec.rhs_indices[0]]
    if rhs is None:
        return (None,)

    result = lax.custom_linear_solve(
        callback(context.invocation.matvec, spec.matvec.inputs),
        rhs,
        solve=callback(context.invocation.solve, spec.solve.inputs),
        transpose_solve=callback(
            context.invocation.transpose_solve, spec.transpose_solve.inputs
        ),
        symmetric=False,
        has_aux=False,
    )

    source_layout = context.invocation.solve.output_layouts[0]
    target_layout = plan.output_layouts[0]

    if source_layout is not None and target_layout is not None:
        result = _project_local_value(
            result, source_layout=source_layout, target_layout=target_layout
        )

    return (result,)


@dataclass(frozen=True)
class _LowerNestedHandler:
    plan: LocalEqnPlan
    inputs: tuple[Any | None, ...]

    def call(self, context: CallContext[LocalJaxprPlan]):
        return _lower_call(self.plan, context, self.inputs)

    def map(self, context: MapContext[LocalJaxprPlan]):
        return _lower_map(self.plan, context, self.inputs)

    def scan(self, context: ScanContext[LocalJaxprPlan]):
        return _lower_scan(self.plan, context, self.inputs)

    def cond(self, context: CondContext[LocalJaxprPlan]):
        return _lower_cond(self.plan, context, self.inputs)

    def linear_solve(self, context: LinearSolveContext[LocalJaxprPlan]):
        return _lower_linear_solve(self.plan, context, self.inputs)


def _lower_eqn(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    if plan.nested is not None:
        result = _lower_nested(plan, inputs)

    else:
        semantics = SEMANTICS.get_ordinary(plan.eqn.primitive)
        rule = semantics.lowering or lower_bind

        result = rule(LoweringContext(plan=plan, inputs=inputs))

    _validate_outputs(plan, result)

    return result


def _execute_frame(
    plan: LocalJaxprPlan,
    local_inputs: tuple[Any, ...],
) -> dict[Var, Any]:
    jaxpr = plan.plan.jaxpr
    expected_inputs = sum(layout is not None for layout in plan.input_layouts)

    if len(local_inputs) != expected_inputs:
        raise ValueError(
            f"local frame expected {expected_inputs} inputs, got {len(local_inputs)}"
        )

    env: dict[Var, Any] = {}

    # Runtime inputs.
    iterator = iter(local_inputs)

    for var, layout in zip(jaxpr.invars, plan.input_layouts):
        if layout is None:
            continue

        value = jnp.asarray(next(iterator))
        if tuple(value.shape) != layout.local_shape:
            raise ValueError(
                f"input {var} has shape {value.shape}; expected {layout.local_shape}"
            )

        env[var] = value

    # Closed JAXPR constants.
    for var, layout, value in zip(
        jaxpr.constvars, plan.const_layouts, plan.const_values, strict=True
    ):
        if layout is None:
            continue

        if value is None:
            raise RuntimeError(
                f"live constvar {var} is not available in the local plan"
            )
        env[var] = jnp.asarray(value)

    # Surviving local equations.
    for eqn_plan in plan.eqns:
        inputs = tuple(
            _read_atom(atom, layout, env)
            for atom, layout in zip(eqn_plan.eqn.invars, eqn_plan.input_layouts)
        )
        outputs = _lower_eqn(eqn_plan, inputs)

        for outvar, layout, value in zip(
            eqn_plan.eqn.outvars, eqn_plan.output_layouts, outputs
        ):
            if layout is None:
                continue

            if isinstance(outvar, Var):
                env[outvar] = value

    return env


def _stack_repeated_input(
    value,
    *,
    outer_layout: TensorLayout,
    body_layout: TensorLayout,
    logical_indices: np.ndarray,
):
    """Extract body-local tensors for multiple logical repeated iterations.

    outer tensor:
        (global_iteration_count, *body_global_shape)

    result:
        (n_local_iterations, *body_layout.local_shape)

    This uses exact global scalar coordinates, so it remains correct when the
    outer TensorLayout is a structured hull larger than this map's exact
    requirements.
    """
    if outer_layout.ndim < 1:
        raise ValueError("repeated input must have a leading iteration axis")

    if outer_layout.global_shape[1:] != body_layout.global_shape:
        raise ValueError(
            "repeated input/body global shape mismatch: "
            f"{outer_layout.global_shape[1:]} != "
            f"{body_layout.global_shape}"
        )

    logical_indices = np.asarray(logical_indices, dtype=np.int64).ravel()

    n_iterations = logical_indices.size
    body_rows = np.arange(body_layout.local_size, dtype=np.int64)
    body_global_rows = body_layout.local_rows_to_global_rows(body_rows)

    # Convert body scalar rows to body global coordinates.
    if body_layout.ndim == 0:
        # Outer tensor is simply shape (map_length,).
        outer_global_rows = logical_indices.copy()

    else:
        body_coords = np.unravel_index(body_global_rows, body_layout.global_shape)
        leading = np.repeat(logical_indices, body_layout.local_size)
        trailing = tuple(np.tile(coords, n_iterations) for coords in body_coords)

        outer_global_rows = np.ravel_multi_index(
            (leading,) + trailing,
            outer_layout.global_shape,
        )

    # Those global rows must be present in the enclosing local layout.
    try:
        outer_local_rows = outer_layout.global_rows_to_local_rows(outer_global_rows)

    except ValueError as exc:
        raise ValueError(
            "repeated body requires values not present in the outer input layout"
        ) from exc

    selected = jnp.ravel(value)[jnp.asarray(outer_local_rows)]

    return jnp.reshape(
        selected,
        (n_iterations, *body_layout.local_shape),
    )


def _project_repeated_constants(
    plan: LocalEqnPlan,
    template: LocalJaxprPlan,
    inputs: tuple[Any | None, ...],
    *,
    num_consts: int,
    label: str,
) -> dict[int, Any]:
    constants: dict[int, Any] = {}

    for input_index in range(num_consts):
        body_layout = template.input_layouts[input_index]
        if body_layout is None:
            continue

        value = inputs[input_index]
        outer_layout = plan.input_layouts[input_index]
        if value is None:
            raise RuntimeError(f"live {label} constant {input_index} is unavailable")

        constants[input_index] = _project_repeated_value(
            plan,
            input_index,
            value,
            outer_layout=outer_layout,
            body_layout=body_layout,
            label=f"{label} constant",
        )

    return constants


def _stack_repeated_inputs(
    plan: LocalEqnPlan,
    template: LocalJaxprPlan,
    inputs: tuple[Any | None, ...],
    *,
    start: int,
    logical_indices: np.ndarray,
    label: str,
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    indices: list[int] = []
    values: list[Any] = []

    for input_index in range(start, len(inputs)):
        body_layout = template.input_layouts[input_index]
        if body_layout is None:
            continue

        value = inputs[input_index]
        outer_layout = plan.input_layouts[input_index]
        if value is None:
            raise RuntimeError(f"live {label} input {input_index} is unavailable")

        if outer_layout is None:
            atom = plan.eqn.invars[input_index]
            if not isinstance(atom, Literal):
                raise RuntimeError(f"live {label} input {input_index} has no layout")

            demand = TensorDemand.full(tuple(value.shape))
            assert demand is not None
            outer_layout = TensorLayout.from_demand(demand)

        indices.append(input_index)
        values.append(
            _stack_repeated_input(
                value,
                outer_layout=outer_layout,
                body_layout=body_layout,
                logical_indices=logical_indices,
            )
        )

    return tuple(indices), tuple(values)


def _project_repeated_value(
    plan: LocalEqnPlan,
    input_index: int,
    value,
    *,
    outer_layout: TensorLayout | None,
    body_layout: TensorLayout,
    label: str,
):
    if outer_layout is not None:
        return _project_local_value(
            value,
            source_layout=outer_layout,
            target_layout=body_layout,
        )

    if not isinstance(plan.eqn.invars[input_index], Literal):
        raise TypeError(f"live {label} {input_index} has no outer layout")

    return extract_local_value(value, body_layout)


def _same_layout(
    lhs: TensorLayout | None,
    rhs: TensorLayout | None,
) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs

    if lhs.global_shape != rhs.global_shape:
        return False

    if lhs.local_shape != rhs.local_shape:
        return False

    return all(
        np.array_equal(
            lhs.global_axis_indices(axis),
            rhs.global_axis_indices(axis),
        )
        for axis in range(lhs.ndim)
    )


def _same_layouts(
    lhs,
    rhs,
) -> bool:
    return len(lhs) == len(rhs) and all(_same_layout(a, b) for a, b in zip(lhs, rhs))


def _same_local_route(
    lhs,
    rhs,
) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs

    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, LocalGatherRoute):
        return lhs.output_shape == rhs.output_shape and np.array_equal(
            lhs.source_rows, rhs.source_rows
        )

    if isinstance(lhs, LocalScatterRoute):
        return (
            lhs.operand_shape == rhs.operand_shape
            and lhs.update_shape == rhs.update_shape
            and lhs.output_shape == rhs.output_shape
            and np.array_equal(lhs.operand_rows, rhs.operand_rows)
            and np.array_equal(lhs.operand_output_rows, rhs.operand_output_rows)
            and np.array_equal(lhs.update_rows, rhs.update_rows)
            and np.array_equal(lhs.target_rows, rhs.target_rows)
        )

    if isinstance(lhs, LocalDynamicSliceRoute):
        return lhs.output_shape == rhs.output_shape and np.array_equal(
            lhs.source_rows, rhs.source_rows
        )

    if isinstance(lhs, LocalSelectNRoute):
        if lhs.output_shape != rhs.output_shape or len(lhs.cases) != len(rhs.cases):
            return False

        return all(
            np.array_equal(a.output_rows, b.output_rows)
            and np.array_equal(a.source_rows, b.source_rows)
            for a, b in zip(lhs.cases, rhs.cases)
        )

    # Unknown localized route => don't assume equivalence.
    return False


def _same_nested_local_plan(
    lhs: LocalNestedPlan | None,
    rhs: LocalNestedPlan | None,
) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs

    if not isinstance(lhs, LocalNestedPlan) or not isinstance(rhs, LocalNestedPlan):
        return False
    if lhs.spec != rhs.spec:
        return False
    left = lhs.invocation
    right = rhs.invocation
    if left.kind is not right.kind or left.eqn_index != right.eqn_index:
        return False
    left_children = left.children()
    right_children = right.children()
    if len(left_children) != len(right_children):
        return False
    return all(
        a.logical_index == b.logical_index
        and _same_local_jaxpr_plan(a.payload, b.payload)
        for a, b in zip(left_children, right_children)
    )


def _same_local_jaxpr_plan(
    lhs: LocalJaxprPlan,
    rhs: LocalJaxprPlan,
) -> bool:
    # All map iterations should originate from the same analyzed body.
    if lhs.plan.jaxpr is not rhs.plan.jaxpr:
        return False

    if not _same_layouts(lhs.input_layouts, rhs.input_layouts):
        return False

    if not _same_layouts(lhs.output_layouts, rhs.output_layouts):
        return False

    if not _same_layouts(lhs.const_layouts, rhs.const_layouts):
        return False

    if len(lhs.eqns) != len(rhs.eqns):
        return False

    for left, right in zip(lhs.eqns, rhs.eqns):
        if left.index != right.index:
            return False

        if left.eqn.primitive is not right.eqn.primitive:
            return False

        if not _same_layouts(left.input_layouts, right.input_layouts):
            return False

        if not _same_layouts(left.output_layouts, right.output_layouts):
            return False

        left_route = None if left.route is None else left.route.local
        right_route = None if right.route is None else right.route.local

        if not _same_local_route(left_route, right_route):
            return False

        if not _same_nested_local_plan(left.nested, right.nested):
            return False

    return True


def _common_repeated_template(
    invocation: RepeatedInvocation[LocalJaxprPlan],
) -> LocalJaxprPlan | None:
    # This check iterates over all iterations, to check if the body is the same for all of them.
    # Potentially slow, maybe skip if we know this from somewhere already.
    if not invocation.iterations:
        return None

    template = invocation.iterations[0].body

    for iteration in invocation.iterations[1:]:
        if not _same_local_jaxpr_plan(template, iteration.body):
            return None

    return template


def _validate_repeated_output_layout(
    *,
    outer_layout: TensorLayout,
    body_layout: TensorLayout,
    logical_indices: np.ndarray,
    label: str,
) -> None:
    if outer_layout.ndim < 1:
        raise RuntimeError(f"{label} output must have a leading iteration axis")

    if outer_layout.global_shape[1:] != body_layout.global_shape:
        raise RuntimeError(
            f"{label} body/output global shape mismatch: "
            f"{outer_layout.global_shape[1:]} != "
            f"{body_layout.global_shape}"
        )

    for axis in range(body_layout.ndim):
        if not np.array_equal(
            outer_layout.global_axis_indices(axis + 1),
            body_layout.global_axis_indices(axis),
        ):
            raise RuntimeError(
                f"{label} body/output selection differs along body axis {axis}"
            )

    stored_indices = outer_layout.global_axis_indices(0)
    if not np.array_equal(stored_indices, logical_indices):
        raise RuntimeError(
            f"{label} iterations do not match the output layout leading axis"
        )

    expected_local_shape = (
        logical_indices.size,
        *body_layout.local_shape,
    )

    if outer_layout.local_shape != expected_local_shape:
        raise RuntimeError(
            f"{label} output local shape mismatch: "
            f"{outer_layout.local_shape} != "
            f"{expected_local_shape}"
        )


def _validate_scan_iteration_subset(
    context: ScanContext[LocalJaxprPlan],
    template: LocalJaxprPlan,
) -> None:
    """If any carry component is runtime-live, surviving iterations must form
    a prefix of the original execution order.

    Otherwise dropping an intermediate iteration would incorrectly skip a
    carry transition.

    If all carry components are dead, arbitrary selected iterations are safe:
    the scan has effectively become map-like for local runtime purposes.
    """
    spec = context.spec
    selected = tuple(child.logical_index for child in context.invocation.children())
    if not selected:
        raise RuntimeError("live scan has no surviving iterations")

    carry_live = False

    for carry_index in range(spec.num_carry):
        body_input_index = spec.num_consts + carry_index

        if (
            template.input_layouts[body_input_index] is not None
            or template.output_layouts[carry_index] is not None
        ):
            carry_live = True
            break

    if not carry_live:
        return

    expected = spec.execution_indices()[: len(selected)]

    if selected != expected:
        raise NotImplementedError(
            "localized recurrent scan iterations do not form "
            "a prefix of execution order; skipping an intermediate "
            "carry transition would change semantics"
        )


def _scan_live_carry_indices(
    spec: ScanSpec,
    template: LocalJaxprPlan,
) -> tuple[int, ...]:
    result = []

    for carry_index in range(spec.num_carry):
        input_index = spec.num_consts + carry_index
        input_layout = template.input_layouts[input_index]
        output_layout = template.output_layouts[carry_index]

        if input_layout is None and output_layout is None:
            continue

        if input_layout is None or output_layout is None:
            raise NotImplementedError(
                f"scan carry {carry_index} is live on only "
                "one side of the body boundary"
            )

        if not _same_layout(input_layout, output_layout):
            raise NotImplementedError(
                f"localized scan carry {carry_index} changes layout between iterations"
            )

        result.append(carry_index)

    return tuple(result)


def _compile_contribution_terms(
    plan: LocalJaxprPlan,
    contributions: ContributionTrace,
    owned: tuple[OwnedContribution, ...],
) -> tuple[LocalContributionTerm, ...]:
    roots_by_id = {}

    for root in contributions.roots:
        if root.id in roots_by_id:
            raise ValueError(f"duplicate contribution root id {root.id}")

        roots_by_id[root.id] = root

    terms: list[LocalContributionTerm] = []

    for ownership in owned:
        try:
            root = roots_by_id[ownership.root_id]
        except KeyError as exc:
            raise ValueError(
                f"ownership references unknown contribution root {ownership.root_id}"
            ) from exc

        # First executable milestone:
        #
        # Contribution detection may eventually place roots inside a
        # transparent call/remat FramePath. Our current executable environment
        # exposes root-frame variables only.
        #
        # Map/scan contribution roots are deliberately kept outside those
        # frames by contribution detection.
        if root.value.path:
            raise NotImplementedError(
                "local scalar reconstruction for contribution "
                f"root {root.id} inside FramePath "
                f"{root.value.path} is not implemented yet"
            )

        var = root.value.var

        try:
            stored_layout = plan.layouts[var]
        except KeyError as exc:
            raise RuntimeError(
                f"contribution root {root.id} variable {var} has no local layout"
            ) from exc

        owned_layout = TensorLayout.from_demand(ownership.demand)
        if stored_layout.global_shape != owned_layout.global_shape:
            raise RuntimeError(
                f"contribution root {root.id} shape mismatch: "
                f"stored {stored_layout.global_shape}, "
                f"owned {owned_layout.global_shape}"
            )

        # Exact owned contribution rows in global coordinates.
        owned_local_rows = np.arange(owned_layout.local_size, dtype=np.int64)
        owned_global_rows = owned_layout.local_rows_to_global_rows(owned_local_rows)

        # Rewrite them into rows of the actual stored local tensor.
        #
        # This also verifies that liveness included every owned
        # contribution entry.
        try:
            source_rows = stored_layout.global_rows_to_local_rows(owned_global_rows)

        except ValueError as exc:
            raise RuntimeError(
                f"owned contribution root {root.id} "
                "is not contained in its finalized local layout"
            ) from exc

        terms.append(
            LocalContributionTerm(
                root_id=root.id,
                part=ownership.part,
                var=var,
                coefficient=root.coefficient,
                source_rows=source_rows,
                owned_shape=owned_layout.local_shape,
            )
        )

    return tuple(terms)


def _reconstruct_local_scalar(
    env: dict[Var, Any],
    terms: tuple[LocalContributionTerm, ...],
):
    total = None

    for term in terms:
        try:
            value = env[term.var]
        except KeyError as exc:
            raise RuntimeError(
                f"contribution root {term.root_id} variable {term.var} was not produced"
            ) from exc

        flat = jnp.ravel(value)
        selected = flat[jnp.asarray(term.source_rows)]
        contribution = jnp.asarray(term.coefficient) * jnp.sum(selected)
        total = contribution if total is None else total + contribution

    if total is None:
        # A rank can legitimately own no contributions.
        return jnp.asarray(0.0)

    return total


@dataclass(frozen=True, slots=True, eq=False)
class LocalContributionTerm:
    """One rank-owned contribution to the local scalar objective.

    `source_rows` indexes the flattened tensor stored for `var`.

    The rows correspond exactly to OwnedContribution.demand, so the stored
    TensorLayout may be a larger structured hull without causing overcounting.
    """

    root_id: int
    part: int

    var: Var
    coefficient: Any

    source_rows: NDArray[np.int64]
    owned_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(self.source_rows, dtype=np.int64).ravel()
        if np.any(rows < 0):
            raise ValueError("contribution source rows must be nonnegative")

        rows = rows.copy()
        rows.flags.writeable = False

        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(self, "owned_shape", tuple(int(x) for x in self.owned_shape))


@dataclass(frozen=True)
class LocalExecutable:
    plan: LocalJaxprPlan
    input_indices: tuple[int, ...]
    contribution_terms: tuple[LocalContributionTerm, ...]
    function: Callable

    # unused, only used in a test
    def pack_global_inputs(
        self,
        *global_inputs,
    ) -> tuple[Any, ...]:
        if len(global_inputs) != len(self.plan.input_layouts):
            raise ValueError(
                f"expected {len(self.plan.input_layouts)} "
                f"global inputs, got {len(global_inputs)}"
            )

        result = []

        for index in self.input_indices:
            layout = self.plan.input_layouts[index]
            assert layout is not None
            result.append(extract_local_value(global_inputs[index], layout))

        return tuple(result)

    def __call__(
        self,
        *local_inputs,
    ):
        return self.function(*local_inputs)


def build_local_executable(
    plan: LocalJaxprPlan,
    *,
    contributions: ContributionTrace,
    owned: tuple[OwnedContribution, ...],
) -> LocalExecutable:
    input_indices = tuple(
        index for index, layout in enumerate(plan.input_layouts) if layout is not None
    )
    contribution_terms = _compile_contribution_terms(plan, contributions, owned)

    def function(
        *local_inputs,
    ):
        env = _execute_frame(plan, tuple(local_inputs))
        return _reconstruct_local_scalar(env, contribution_terms)

    return LocalExecutable(
        plan=plan,
        input_indices=input_indices,
        contribution_terms=contribution_terms,
        function=function,
    )
