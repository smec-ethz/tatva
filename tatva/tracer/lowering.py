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
from jax.extend.core import Literal, Var

from tatva.tracer.demand import (
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
)
from tatva.tracer.layout import TensorLayout
from tatva.tracer.local_plan import (
    LocalCallPlan,
    LocalEqnPlan,
    LocalJaxprPlan,
    LocalMapPlan,
    LocalPlan,
    LocalScanPlan,
)
from tatva.tracer.lowerings import (
    LOWERINGS,
    LoweringContext,
    lower_default,
)


def extract_local_value(
    value,
    layout: TensorLayout,
):
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


def _lower_call(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    nested = plan.nested

    if not isinstance(nested, LocalCallPlan):
        raise TypeError("expected LocalCallPlan")

    body = nested.body

    if len(inputs) != len(body.input_layouts):
        raise RuntimeError("call/body input arity mismatch")

    child_inputs = tuple(
        value for value, layout in zip(inputs, body.input_layouts) if layout is not None
    )

    env = _execute_frame(body, child_inputs)
    results: list[Any | None] = []

    for atom, layout in zip(
        body.instance.plan.jaxpr.outvars,
        body.output_layouts,
    ):
        if layout is None:
            results.append(None)

        elif isinstance(atom, Literal):
            results.append(atom.val)

        else:
            results.append(env[atom])

    return tuple(results)


def _lower_nested(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    if isinstance(plan.nested, LocalCallPlan):
        return _lower_call(plan, inputs)

    if isinstance(plan.nested, LocalMapPlan):
        raise NotImplementedError("local map lowering is the next nested milestone")

    if isinstance(plan.nested, LocalScanPlan):
        raise NotImplementedError("local scan lowering is not implemented")

    raise TypeError(f"unsupported nested plan {type(plan.nested)!r}")


def _lower_eqn(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    if plan.nested is not None:
        result = _lower_nested(plan, inputs)

    else:
        rule = LOWERINGS.get(plan.eqn.primitive, lower_default)
        result = rule(LoweringContext(plan=plan, inputs=inputs))

    _validate_outputs(plan, result)

    return result


def _execute_frame(
    plan: LocalJaxprPlan,
    local_inputs: tuple[Any, ...],
) -> dict[Var, Any]:
    jaxpr = plan.instance.plan.jaxpr
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

        value = next(iterator)
        if tuple(value.shape) != layout.local_shape:
            raise ValueError(
                f"input {var} has shape {value.shape}; expected {layout.local_shape}"
            )

        env[var] = value

    # Closed JAXPR constants.
    for var, layout in zip(jaxpr.constvars, plan.const_layouts):
        if layout is None:
            continue

        global_value = plan.instance.concrete.get(var)
        if global_value is None:
            raise RuntimeError(
                f"live constvar {var} is not available in materialization"
            )

        env[var] = extract_local_value(global_value, layout)

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


@dataclass(frozen=True)
class LocalExecutable:
    plan: LocalJaxprPlan

    # Original root JAXPR input positions that survive.
    input_indices: tuple[int, ...]
    output_vars: tuple[Var, ...]
    function: Callable

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
    plan: LocalPlan | LocalJaxprPlan,
    *,
    output_vars: tuple[Var, ...] | None = None,
    jit: bool = True,
) -> LocalExecutable:
    if isinstance(plan, LocalPlan):
        root = plan.root
    else:
        root = plan

    jaxpr = root.instance.plan.jaxpr

    input_indices = tuple(
        index for index, layout in enumerate(root.input_layouts) if layout is not None
    )

    if output_vars is None:
        output_vars = tuple(
            atom
            for atom, layout in zip(jaxpr.outvars, root.output_layouts)
            if (layout is not None and isinstance(atom, Var))
        )

    if not output_vars:
        raise ValueError(
            "no root outputs are live; pass output_vars explicitly, "
            "for example the rank-owned ContributionRoot variables"
        )

    for var in output_vars:
        if var not in root.layouts:
            raise ValueError(f"requested output {var} has no local layout")

    def function(
        *local_inputs,
    ):
        env = _execute_frame(root, tuple(local_inputs))
        return tuple(env[var] for var in output_vars)

    lowered = jax.jit(function) if jit else function

    return LocalExecutable(
        plan=root,
        input_indices=input_indices,
        output_vars=output_vars,
        function=lowered,
    )
