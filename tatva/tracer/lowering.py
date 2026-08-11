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
from jax import lax
from jax.extend.core import Literal, Var
from numpy.typing import NDArray

from tatva.tracer.contributions import ContributionTrace
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
from tatva.tracer.localize import (
    LocalDynamicSliceRoute,
    LocalGatherRoute,
    LocalScatterRoute,
    LocalSelectNRoute,
)
from tatva.tracer.lowerings import (
    LOWERINGS,
    LoweringContext,
    lower_default,
)
from tatva.tracer.partition import OwnedContribution


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

    for atom, layout in zip(plan.instance.plan.jaxpr.outvars, plan.output_layouts):
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
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    nested = plan.nested

    if not isinstance(nested, LocalCallPlan):
        raise TypeError("expected LocalCallPlan")

    body = nested.body
    child_inputs = tuple(
        value for value, layout in zip(inputs, body.input_layouts) if layout is not None
    )

    env = _execute_frame(body, child_inputs)
    return _frame_outputs(body, env)


def _lower_map(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    nested = plan.nested

    if not isinstance(nested, LocalMapPlan):
        raise TypeError("expected LocalMapPlan")

    template = _map_template(nested)

    logical_indices = np.asarray(nested.indices, dtype=np.int64)
    n_iterations = logical_indices.size

    if n_iterations == 0:
        raise RuntimeError("live local map has no surviving iterations")

    if len(inputs) != len(template.input_layouts):
        raise RuntimeError(
            "map outer/body input arity mismatch: "
            f"{len(inputs)} != "
            f"{len(template.input_layouts)}"
        )

    if nested.num_consts > len(inputs):
        raise RuntimeError("map num_consts exceeds input arity")

    # Prepare body constants.
    body_constants: dict[int, Any] = {}

    for input_index in range(nested.num_consts):
        body_layout = template.input_layouts[input_index]
        if body_layout is None:
            continue

        value = inputs[input_index]
        outer_layout = plan.input_layouts[input_index]

        if value is None:
            raise RuntimeError(
                f"map constant {input_index} is live in body but dead in outer frame"
            )

        if outer_layout is None:
            raise RuntimeError(f"map constant {input_index} has no outer layout")

        body_constants[input_index] = _project_local_value(
            value,
            source_layout=outer_layout,
            target_layout=body_layout,
        )

    # Prepare stacked mapped inputs.
    mapped_input_indices: list[int] = []
    mapped_values: list[Any] = []

    for input_index in range(nested.num_consts, len(inputs)):
        body_layout = template.input_layouts[input_index]

        # This map input exists globally but is not required by
        # the local body.
        if body_layout is None:
            continue

        value = inputs[input_index]
        outer_layout = plan.input_layouts[input_index]

        if value is None:
            raise RuntimeError(
                f"mapped input {input_index} is live in body but unavailable"
            )

        if outer_layout is None:
            raise RuntimeError(f"mapped input {input_index} has no outer layout")

        stacked = _stack_map_input(
            value,
            outer_layout=outer_layout,
            body_layout=body_layout,
            logical_indices=logical_indices,
        )

        mapped_input_indices.append(input_index)
        mapped_values.append(stacked)

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

        _validate_map_output_layout(
            outer_layout=outer_layout,
            body_layout=body_layout,
            logical_indices=logical_indices,
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

            if input_index < nested.num_consts:
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


def _lower_nested(
    plan: LocalEqnPlan,
    inputs: tuple[Any | None, ...],
) -> tuple[Any | None, ...]:
    if isinstance(plan.nested, LocalCallPlan):
        return _lower_call(plan, inputs)

    if isinstance(plan.nested, LocalMapPlan):
        return _lower_map(plan, inputs)

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


def _stack_map_input(
    value,
    *,
    outer_layout: TensorLayout,
    body_layout: TensorLayout,
    logical_indices: np.ndarray,
):
    """Extract body-local tensors for multiple logical map iterations.

    outer tensor:
        (global_map_length, *body_global_shape)

    result:
        (n_local_iterations, *body_layout.local_shape)

    This uses exact global scalar coordinates, so it remains correct when the
    outer TensorLayout is a structured hull larger than this map's exact
    requirements.
    """
    if outer_layout.ndim < 1:
        raise ValueError("mapped input must have a leading map axis")

    if outer_layout.global_shape[1:] != body_layout.global_shape:
        raise ValueError(
            "mapped input/body global shape mismatch: "
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
            "mapped body requires values not present in the outer input layout"
        ) from exc

    selected = jnp.ravel(value)[jnp.asarray(outer_local_rows)]

    return jnp.reshape(
        selected,
        (n_iterations, *body_layout.local_shape),
    )


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
    lhs,
    rhs,
) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs

    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, LocalCallPlan):
        return _same_local_jaxpr_plan(lhs.body, rhs.body)

    if isinstance(lhs, LocalMapPlan):
        if lhs.num_consts != rhs.num_consts:
            return False

        if lhs.indices != rhs.indices:
            return False

        if len(lhs.iterations) != len(rhs.iterations):
            return False

        return all(
            _same_local_jaxpr_plan(a.body, b.body)
            for a, b in zip(lhs.iterations, rhs.iterations)
        )

    if isinstance(lhs, LocalScanPlan):
        if len(lhs.iterations) != len(rhs.iterations):
            return False

        return all(
            (a.index == b.index and _same_local_jaxpr_plan(a.body, b.body))
            for a, b in zip(lhs.iterations, rhs.iterations)
        )

    return False


def _same_local_jaxpr_plan(
    lhs: LocalJaxprPlan,
    rhs: LocalJaxprPlan,
) -> bool:
    # All map iterations should originate from the same analyzed body.
    if lhs.instance.plan.jaxpr is not rhs.instance.plan.jaxpr:
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


def _map_template(
    plan: LocalMapPlan,
) -> LocalJaxprPlan:
    if not plan.iterations:
        raise RuntimeError("cannot lower empty LocalMapPlan")

    template = plan.iterations[0].body

    for iteration in plan.iterations[1:]:
        if not _same_local_jaxpr_plan(template, iteration.body):
            raise NotImplementedError(
                "local map iterations have different localized "
                "program structures; template map lowering is "
                "not valid for this map"
            )

    return template


def _validate_map_output_layout(
    *,
    outer_layout: TensorLayout,
    body_layout: TensorLayout,
    logical_indices: np.ndarray,
) -> None:
    if outer_layout.ndim < 1:
        raise RuntimeError("map output must have leading map axis")

    if outer_layout.global_shape[1:] != body_layout.global_shape:
        raise RuntimeError(
            "map body/output global shape mismatch: "
            f"{outer_layout.global_shape[1:]} != "
            f"{body_layout.global_shape}"
        )

    stored_indices = outer_layout.global_axis_indices(0)
    if not np.array_equal(stored_indices, logical_indices):
        raise RuntimeError(
            "map iteration set does not match outer output layout leading axis"
        )

    expected_local_shape = (
        logical_indices.size,
        *body_layout.local_shape,
    )

    if outer_layout.local_shape != expected_local_shape:
        raise RuntimeError(
            "map output local shape mismatch: "
            f"{outer_layout.local_shape} != "
            f"{expected_local_shape}"
        )


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
    contributions: ContributionTrace,
    owned: tuple[OwnedContribution, ...],
    jit: bool = True,
) -> LocalExecutable:
    if isinstance(plan, LocalPlan):
        root = plan.root
    else:
        root = plan

    input_indices = tuple(
        index for index, layout in enumerate(root.input_layouts) if layout is not None
    )
    contribution_terms = _compile_contribution_terms(root, contributions, owned)

    def function(
        *local_inputs,
    ):
        env = _execute_frame(root, tuple(local_inputs))
        return _reconstruct_local_scalar(env, contribution_terms)

    lowered = jax.jit(function) if jit else function

    return LocalExecutable(
        plan=root,
        input_indices=input_indices,
        contribution_terms=contribution_terms,
        function=lowered,
    )
