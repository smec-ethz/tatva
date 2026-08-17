from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from tatva.tracer.capture import CallABI
from tatva.tracer.local.dof_plan import LocalDofPlan
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.lowering.executor import extract_local_value


@dataclass(frozen=True, slots=True)
class LocalizationContext:
    """Compiler facts exposed while reconstructing one PyTree node."""

    rank: int
    dof_plan: LocalDofPlan

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


type LocalizeKey = str | type
type LocalizeOverrides = Mapping[
    LocalizeKey,
    InputLocalizer,
]


@dataclass(frozen=True, slots=True)
class _Localized:
    value: Any
    layouts: tuple[TensorLayout | None, ...]


def _localize_flat_inputs(
    *,
    global_flat: tuple[Any, ...],
    input_layouts: tuple[TensorLayout | None, ...],
    dof_plan: LocalDofPlan,
) -> tuple[Any | None, ...]:
    if len(global_flat) != len(input_layouts):
        raise RuntimeError("call ABI and local input layouts disagree")

    local: list[Any | None] = []

    for flat_index, (value, layout) in enumerate(
        zip(global_flat, input_layouts, strict=True)
    ):
        if flat_index == 0:
            # Canonical DOF input:
            # user-facing local form is owned + ghost storage.
            local.append(jnp.asarray(value)[jnp.asarray(dof_plan.storage.global_dofs)])
            continue

        if layout is None:
            # Compiler-dead input.
            local.append(None)
            continue

        local.append(extract_local_value(value, layout))

    return tuple(local)


def _one_level(
    value: Any,
) -> tuple[
    tuple[Any, ...],
    jax.tree_util.PyTreeDef,
]:
    """Flatten exactly one PyTree node."""

    children, treedef = jax.tree_util.tree_flatten(
        value,
        is_leaf=lambda child: child is not value,
    )

    return tuple(children), treedef


def _localize_tree(
    *,
    global_value: Any,
    leaves: Iterator[
        tuple[
            Any | None,
            TensorLayout | None,
        ]
    ],
    rank: int,
    halo: LocalDofPlan,
    specializers: Mapping[
        LocalizeKey,
        InputLocalizer,
    ],
    parameter_name: str,
    is_parameter_root: bool = False,
) -> _Localized:
    children, node_treedef = _one_level(global_value)

    # JAX leaf
    # ``treedef_is_leaf`` also returns true for leafless PyTree nodes such as
    # ``{}``.  A JAX leaf is instead the one-level traversal whose sole child
    # is the original object itself.
    if len(children) == 1 and children[0] is global_value:
        try:
            local_value, layout = next(leaves)
        except StopIteration:
            raise RuntimeError(
                "local input reconstruction consumed more PyTree leaves than available"
            ) from None

        ctx = LocalizationContext(
            rank=rank,
            dof_plan=halo,
            child_layouts=((layout,),),
        )

        # Parameter-name override.
        if is_parameter_root and parameter_name in specializers:
            value = specializers[parameter_name](global_value, (local_value,), ctx)

        # Type override.
        elif type(global_value) in specializers:
            value = specializers[type(global_value)](global_value, (local_value,), ctx)

        # Class protocol.
        elif (method := getattr(global_value, "__tatva_localize__", None)) is not None:
            value = method((local_value,), ctx)

        else:
            value = local_value

        return _Localized(value=value, layouts=(layout,))

    # Internal PyTree node:
    # localize children first.
    localized_children = tuple(
        _localize_tree(
            global_value=child,
            leaves=leaves,
            rank=rank,
            halo=halo,
            specializers=specializers,
            parameter_name=parameter_name,
            is_parameter_root=False,
        )
        for child in children
    )

    child_values = tuple(child.value for child in localized_children)
    child_layouts = tuple(child.layouts for child in localized_children)

    ctx = LocalizationContext(
        rank=rank,
        dof_plan=halo,
        child_layouts=child_layouts,
    )

    # Explicit parameter specialization wins.
    if is_parameter_root and parameter_name in specializers:
        value = specializers[parameter_name](global_value, child_values, ctx)

    # Explicit type specialization next.
    elif type(global_value) in specializers:
        value = specializers[type(global_value)](global_value, child_values, ctx)

    # Then class-owned protocol.
    elif (method := getattr(global_value, "__tatva_localize__", None)) is not None:
        if not callable(method):
            raise TypeError(
                f"{type(global_value).__name__}.__tatva_localize__ is not callable"
            )

        value = method(child_values, ctx)

    # Ordinary node: rebuild exactly this one level.
    else:
        try:
            value = node_treedef.unflatten(child_values)
        except Exception as exc:
            raise TypeError(
                f"cannot reconstruct local "
                f"{type(global_value).__name__}; "
                "the type probably requires semantic "
                "localization. Implement "
                "__tatva_localize__() or register a "
                "localize specializer."
            ) from exc

    return _Localized(
        value=value,
        layouts=tuple(chain.from_iterable(child_layouts)),
    )


def localize_inputs(
    rank: int,
    call_abi: CallABI,
    dof_plan: LocalDofPlan,
    specializers: LocalizeOverrides,
    input_layouts: tuple[TensorLayout | None, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    global_bound = call_abi.bind(*args, **kwargs)
    global_flat = call_abi.flatten_bound(global_bound)
    local_flat = _localize_flat_inputs(
        global_flat=global_flat, input_layouts=input_layouts, dof_plan=dof_plan
    )

    localized_arguments: dict[str, Any] = {}

    for parameter_name, parameter_treedef, flat_slice in call_abi.parameter_trees():
        global_value = global_bound.arguments[parameter_name]
        local_values = local_flat[flat_slice]
        layouts = input_layouts[flat_slice]
        leaves = iter(zip(local_values, layouts, strict=True))

        localized = _localize_tree(
            global_value=global_value,
            leaves=leaves,
            rank=rank,
            halo=dof_plan,
            specializers=specializers,
            parameter_name=parameter_name,
            is_parameter_root=True,
        )

        # Ensure the recursion consumed exactly this parameter's leaves.
        try:
            next(leaves)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"localization of parameter "
                f"{parameter_name!r} did not consume "
                "all compiler leaves"
            )

        # Specializers may change values, but not the captured
        # PyTree structure.
        try:
            parameter_treedef.flatten_up_to(localized.value)
        except ValueError as exc:
            raise TypeError(
                f"localized parameter "
                f"{parameter_name!r} changed its "
                "captured PyTree structure"
            ) from exc

        localized_arguments[parameter_name] = localized.value

    bound = inspect.BoundArguments(call_abi.signature, localized_arguments)
    return bound.args, dict(bound.kwargs)
