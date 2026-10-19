from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from tatva.tracer.capture import CallABI, CapturedJaxpr
from tatva.tracer.local.dof_plan import LocalDofPlan
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.plan import LocalJaxprPlan


@dataclass(frozen=True, slots=True)
class LocalInput:
    """Rank-local representation of one original flat input."""

    layout: TensorLayout | None

    # Global rows represented by the caller-facing local value.
    # Used for the canonical DOF storage input.
    storage_global_rows: NDArray[np.int64] | None = None

    # Caller-facing local storage -> compact executable value.
    storage_to_compute: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        for name in ("storage_global_rows", "storage_to_compute"):
            value = getattr(self, name)
            if value is None:
                continue

            rows = np.asarray(value, dtype=np.int64).ravel().copy()
            if np.any(rows < 0):
                raise ValueError(f"{name} rows must be nonnegative")

            rows.flags.writeable = False
            object.__setattr__(self, name, rows)

    def localize(self, value: Any, *, preserve_dead: bool = False) -> Any:
        """Global value -> caller-facing local value."""
        if self.layout is None:
            return value if preserve_dead else None

        if self.storage_global_rows is not None:
            return jnp.asarray(value)[jnp.asarray(self.storage_global_rows)]

        return self.layout.extract(value)

    def executable_value(self, local_value: Any) -> Any:
        """Caller-facing local value -> compact compute value."""
        if self.storage_to_compute is not None:
            return local_value[jnp.asarray(self.storage_to_compute)]

        return local_value

    def global_rows(self) -> NDArray[np.int64] | None:
        """Global coordinates represented by the local storage value."""
        if self.layout is None:
            return None

        if self.storage_global_rows is not None:
            return self.storage_global_rows

        return self.layout.local_rows_to_global_rows(
            np.arange(self.layout.local_size, dtype=np.int64)
        ).ravel()


@dataclass(frozen=True, slots=True)
class LocalInputPlan:
    abi: CallABI

    # indexed exactly by original flat input index
    inputs: tuple[LocalInput, ...]
    # ordered exactly like LocalJaxprPlan's compact executable abi
    live_indices: tuple[int, ...]

    global_examples: tuple[Any, ...]
    specializers: LocalizeOverrides = field(default_factory=dict)

    def localize_flat(
        self,
        flat: tuple[Any, ...],
        *,
        preserve_dead: bool = False,
    ) -> tuple[Any, ...]:
        if len(flat) != len(self.inputs):
            raise ValueError(
                f"flat input count {len(flat)} does not match plan input count "
                f"{len(self.inputs)}"
            )

        return tuple(
            input_.localize(value, preserve_dead=preserve_dead)
            for input_, value in zip(self.inputs, flat, strict=True)
        )

    def compute_inputs(self, local_flat: tuple[Any, ...]) -> tuple[Any, ...]:
        """Build the compact executable ABI."""
        if len(local_flat) != len(self.inputs):
            raise ValueError(
                f"flat input count {len(local_flat)} does not match plan input count "
                f"{len(self.inputs)}"
            )

        return tuple(
            self.inputs[index].executable_value(local_flat[index])
            for index in self.live_indices
        )

    def global_rows(self, index: int) -> NDArray[np.int64] | None:
        return self.inputs[index].global_rows()

    def localize(self, *args, **kwargs) -> inspect.BoundArguments:
        bound = self.abi.bind(*args, **kwargs)
        global_flat = self.abi.flatten_bound(bound)
        local_flat = self.localize_flat(global_flat)

        return _reconstruct_local_call(
            self.abi,
            bound,
            local_flat,
            self.inputs,
            specializers=self.specializers,
        )

    def example_flat(self, *, preserve_dead: bool = True) -> tuple[Any, ...]:
        return self.localize_flat(self.global_examples, preserve_dead=preserve_dead)

    def example_call(self, *, preserve_dead: bool = True) -> inspect.BoundArguments:
        """Return the captured example call in rank-local representation."""
        # Reconstruct the original valid global call first.
        global_bound = self.abi.unflatten(self.global_examples)
        local_flat = self.localize_flat(
            self.global_examples, preserve_dead=preserve_dead
        )

        return _reconstruct_local_call(
            self.abi,
            global_bound,
            local_flat,
            self.inputs,
            specializers=self.specializers,
        )


@dataclass(frozen=True, slots=True)
class LocalizationContext:
    """Compiler facts exposed while reconstructing one PyTree node."""

    # One tuple per immediate child of the node being reconstructed.
    # Each tuple contains all compiler layouts below that child.
    child_layouts: tuple[tuple[TensorLayout | None, ...], ...]

    @property
    def layouts(
        self,
    ) -> tuple[TensorLayout | None, ...]:
        return tuple(chain.from_iterable(self.child_layouts))

    def child_layout(
        self,
        index: int,
    ) -> TensorLayout | None:
        """Layout of an immediate child that is one JAX leaf."""

        layouts = self.child_layouts[index]

        if len(layouts) != 1:
            raise ValueError(
                f"child {index} contains {len(layouts)} JAX leaves, not one"
            )

        return layouts[0]


type LocalizeKey = str | type
type LocalizeOverrides = Mapping[LocalizeKey, InputLocalizer]


class InputLocalizer(Protocol):
    def __call__(
        self,
        global_value: Any,
        local_children: tuple[Any, ...],
        ctx: LocalizationContext,
        /,
    ) -> Any: ...


class SupportsTatvaLocalization(Protocol):
    def __tatva_localize__(
        self,
        local_children: tuple[Any, ...],
        ctx: LocalizationContext,
        /,
    ) -> Any: ...


def _one_level(value: Any) -> tuple[list[Any], jax.tree_util.PyTreeDef]:
    """Flatten exactly one PyTree node."""
    children, treedef = jax.tree_util.tree_flatten(
        value,
        is_leaf=lambda child: child is not value,
    )
    return children, treedef


def _reconstruct_local_call(
    abi: CallABI,
    bound: inspect.BoundArguments,
    local_flat: tuple[Any, ...],
    inputs: tuple[LocalInput, ...],
    *,
    specializers: LocalizeOverrides,
) -> inspect.BoundArguments:
    """Reconstruct caller-level PyTrees from localized flat JAX inputs."""

    if len(local_flat) != len(inputs):
        raise ValueError("localized input count does not match input binding count")

    layouts = tuple(input_.layout for input_ in inputs)

    arguments: dict[str, Any] = {}

    for param_name, _, flat_slice in abi.parameter_trees():
        global_value = bound.arguments[param_name]

        leaves = iter(zip(local_flat[flat_slice], layouts[flat_slice], strict=True))
        local_value, _ = _reconstruct_node(
            global_value,
            leaves,
            specializers=specializers,
            param_name=param_name,
        )
        arguments[param_name] = local_value

    return inspect.BoundArguments(abi.signature, arguments)


def _reconstruct_node(
    global_node: Any,
    leaves: Iterator[tuple[Any | None, TensorLayout | None]],
    specializers: LocalizeOverrides,
    param_name: str | None = None,
) -> tuple[Any, tuple[TensorLayout | None, ...]]:
    """Reconstruct one PyTree node with semantic localization support."""
    # 1-level pytree unroll
    children, treedef = _one_level(global_node)

    # leaf case
    if len(children) == 1 and children[0] is global_node:
        local_val, layout = next(leaves)
        child_vals = (local_val,)
        child_layouts = ((layout,),)
        default_val = local_val
    else:
        # internal node: recurse on children
        sub_results = [
            _reconstruct_node(child, leaves, specializers) for child in children
        ]
        child_vals = tuple(val for val, _ in sub_results)
        child_layouts = tuple(layouts for _, layouts in sub_results)
        default_val = treedef.unflatten(child_vals)

    all_layouts = tuple(chain.from_iterable(child_layouts))
    ctx = LocalizationContext(child_layouts=child_layouts)

    # unified dispatch: parameter name > type > class protocol > default
    if param_name and param_name in specializers:
        val = specializers[param_name](global_node, child_vals, ctx)
    elif type(global_node) in specializers:
        val = specializers[type(global_node)](global_node, child_vals, ctx)
    elif (method := getattr(global_node, "__tatva_localize__", None)) is not None:
        val = method(child_vals, ctx)
    else:
        val = default_val

    return val, all_layouts


def build_local_input_plan(
    *,
    captured: CapturedJaxpr,
    local_plan: LocalJaxprPlan,
    dofs: LocalDofPlan,
    dof_input_index: int,
    specializers: LocalizeOverrides | None = None,
) -> LocalInputPlan:
    inputs: list[LocalInput] = []

    for index, layout in enumerate(local_plan.input_layouts):
        if index == dof_input_index:
            if layout is None:
                raise RuntimeError("canonical dof input is compiler-dead")
            inputs.append(
                LocalInput(
                    layout=layout,
                    storage_global_rows=dofs.storage.global_dofs,
                    storage_to_compute=dofs.compute_rows,
                )
            )
        else:
            inputs.append(LocalInput(layout=layout))

    return LocalInputPlan(
        abi=captured.call_abi,
        inputs=tuple(inputs),
        live_indices=local_plan.live_input_indices,
        global_examples=captured.flat_args,
        specializers=specializers or {},
    )
