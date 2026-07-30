# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

from math import prod
from typing import Any, Generator, Generic, Self, Sequence, TypeVar

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from tatva.compound.field import Field, FieldStackedView, _FieldBase, _FieldSpec

T_Compound = TypeVar("T_Compound", bound="Compound")

__all__ = ["Compound", "CompoundError", "CompoundStackError"]


class CompoundError(ValueError):
    """Base error class for Compound-related errors."""


class CompoundStackError(CompoundError):
    pass


class Compound:
    """A compound array whose fields are declared against function spaces.

    Packs several fields into one flat array and hands back views, so a solver can work on
    a single vector.

    Examples:

        >>> V = FunctionSpace(mesh, Tri3())
        >>> class State(Compound):
        ...     u = field(V, (2,))          # a displacement, 2 components per dof
        ...     p = field(V)                # a scalar pressure on the same space
        ...     load = field(shape=(3,))    # not backed by a space
        >>> state = State()

    `state.arr` is the flat array; `state.u` and `state.p` are views into it.

    By default each field gets a contiguous slice of its own, in declaration order. Asking
    for `stack` interleaves fields instead, so the values belonging to one dof sit together.
    Interleaving is a relation between fields, so it is declared on the class rather than on
    any one field::

        >>> class Blocked(Compound, stack=True):
        ...     u = field(V, (2,))  # u and p share one block: [u0x u0y p0 | u1x u1y p1 ...]
        ...     p = field(V)
        >>> class Partly(Compound, stack=(("u", "v"),)):
        ...     u = field(V, (2,))
        ...     v = field(V, (2,))
        ...     p = field(V)        # not named, so it keeps its own slice

    The components *within* a field are always interleaved regardless: the dof axis is the
    outermost one, so a dof's components are contiguous either way.
    """

    fields: tuple[tuple[str, Field], ...] = ()
    arr: Array
    size: int = 0

    def __init_subclass__(
        cls,
        *,
        stack: bool | Sequence[Sequence[str]] = False,
        **kwargs: Any,
    ) -> None:
        """Lays out the fields declared on `cls`.

        Args:
            stack: Which fields share an interleaved block, so that the values belonging to
                one dof sit together. Interleaving is a relation between fields, so it is
                stated here rather than on any one field. `False`, the default, gives every
                field a contiguous slice of its own. `True` interleaves whatever can be
                interleaved — every space-backed field, grouped by dof count. Explicit
                groups of names control it exactly; fields left unnamed stay contiguous.
        """
        super().__init_subclass__(**kwargs)

        all_fields = _inherited_fields(cls)
        current_offset = sum(int(prod(f.shape)) for _, f in all_fields)

        new_specs = [
            (name, val)
            for name, val in cls.__dict__.items()
            if isinstance(val, _FieldSpec)
        ]

        reserved = set(dir(Compound)) | {"arr"}
        for name, _ in new_specs:
            if name in reserved:
                raise CompoundError(
                    f"Field name {name!r} is reserved and cannot be used in a Compound."
                )

        groups, standalone = _plan_groups(new_specs, stack)

        name_to_descriptor: dict[str, Field] = {}
        for group in groups:
            descriptors, block_size = _stacked_descriptors(group, current_offset)
            name_to_descriptor.update(descriptors)
            current_offset += block_size

        descriptors, block_size = _standard_descriptors(standalone, current_offset)
        name_to_descriptor.update(descriptors)
        current_offset += block_size

        # Assemble cls.fields in the order the user declared them, not in layout order.
        for name, _ in new_specs:
            descriptor = name_to_descriptor[name]
            setattr(cls, name, descriptor)
            all_fields.append((name, descriptor))

        cls.fields = tuple(all_fields)
        cls.size = current_offset

        register_pytree_node_class(cls)

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

    def __init__(self, arr: Array | None = None, **kwargs: Any) -> None:
        """Initializes the compound, optionally from a flat array.

        Args:
            arr: The flat array to wrap. When omitted the compound is zero-filled and then
                populated from `kwargs` and from each field's `default_factory`.
            **kwargs: Initial values, by field name. Ignored when `arr` is given.
        """
        if arr is not None:
            if arr.size != self.size:
                raise CompoundError(
                    f"array size {arr.size} does not match the compound size {self.size}"
                )
            self.arr = arr
            return

        unknown = set(kwargs) - {name for name, _ in self.fields}
        if unknown:
            raise CompoundError(
                f"unknown field name(s) {sorted(unknown)}; this compound has "
                f"{[name for name, _ in self.fields]}"
            )

        self.arr = jnp.zeros(self.size, dtype=float)
        for name, field_obj in self.fields:
            if name in kwargs:
                self.arr = field_obj._set_in_array(self.arr, jnp.asarray(kwargs[name]))
            elif field_obj.default_factory is not None:
                self.arr = field_obj._set_in_array(
                    self.arr, jnp.asarray(field_obj.default_factory())
                )

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Generator[Array, None, None]:
        for name, _ in self.fields:
            yield getattr(self, name)

    def __repr__(self) -> str:
        field_reprs = [
            f"{name}={getattr(type(self), name).shape}" for name, _ in self.fields
        ]
        return f"{self.__class__.__name__}({', '.join(field_reprs)})"

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.arr + other.arr)

    def at(self, name: str) -> _CompoundAtHelper[Self]:
        """Returns a helper for a functional update, `state.at("u").set(value)`.

        Args:
            name: The field to update.
        """
        field_obj = dict(self.fields).get(name)
        if field_obj is None:
            raise AttributeError(f"Unknown field name: {name}")
        return _CompoundAtHelper(self, field_obj)

    def flatten(self) -> Array:
        """Returns the flat array. Same as `state.arr`."""
        return self.arr


class _CompoundAtHelper(Generic[T_Compound]):
    def __init__(self, state: T_Compound, field_obj: Field):
        self.state = state
        self.field_obj = field_obj

    def set(self, value: Array | float) -> T_Compound:
        return self.state.__class__(
            self.field_obj._set_in_array(self.state.arr, jnp.asarray(value))
        )


def _plan_groups(
    specs: Sequence[tuple[str, _FieldSpec]],
    stack: bool | Sequence[Sequence[str]],
) -> tuple[list[list[tuple[str, _FieldSpec]]], list[tuple[str, _FieldSpec]]]:
    """Decides which fields share an interleaved block.

    Args:
        specs: The `(name, spec)` pairs declared on the class, in declaration order.
        stack: `False` to interleave nothing, `True` to group by dof count, or explicit
            groups of field names.

    Returns:
        The interleaved groups, and the fields getting a contiguous slice of their own, the
        latter in declaration order.

    Raises:
        CompoundError: If an explicit group names an unknown field or repeats one.
    """
    by_name = dict(specs)
    candidates: list[list[tuple[str, _FieldSpec]]] = []

    # `stack` is checked for being a bool rather than for truthiness: `()` is an empty group
    # list, which asks for the same layout as False but by a different route.
    if stack is False:
        pass
    elif stack is True:
        # Fields interleave when they agree on a dof count. That is the only thing a block
        # requires, and it separates spaces of different degree — a P2 displacement and a P1
        # pressure land in different blocks — without comparing spaces at all. A field with
        # no space is left out: its leading axis is not a dof count and matching one would
        # be a coincidence.
        keyed: dict[int, list[tuple[str, _FieldSpec]]] = {}
        for name, spec in specs:
            if spec.space is not None:
                keyed.setdefault(spec.shape[0], []).append((name, spec))
        candidates = list(keyed.values())
    else:
        seen: dict[str, int] = {}
        for index, names in enumerate(stack):
            group: list[tuple[str, _FieldSpec]] = []
            for name in names:
                if name not in by_name:
                    raise CompoundError(
                        f"stack names unknown field {name!r}; this class declares "
                        f"{sorted(by_name)}"
                    )
                if name in seen:
                    raise CompoundError(
                        f"field {name!r} appears in stack group {seen[name]} and {index}; "
                        f"a field can only be interleaved into one block"
                    )
                seen[name] = index
                group.append((name, by_name[name]))
            if group:
                candidates.append(group)

    # A block of one is not a block: interleaving a lone field would pay for
    # FieldStackedView's reshape and slice on every access and buy nothing. Demoting here
    # rather than in the caller is what lets `standalone` be one declaration-order filter,
    # so a demoted field keeps its place among the others instead of being appended last.
    groups = [group for group in candidates if len(group) > 1]
    grouped = {name for group in groups for name, _ in group}
    return groups, [(name, spec) for name, spec in specs if name not in grouped]


def _inherited_fields(cls: type[Compound]) -> list[tuple[str, Field]]:
    """Returns the fields of the nearest Compound base, so subclasses extend a layout."""
    for base in cls.__mro__[1:]:
        if issubclass(base, Compound) and base is not Compound:
            return list(base.fields)
    return []


def _stacked_descriptors(
    items: Sequence[tuple[str, _FieldSpec]], offset: int
) -> tuple[dict[str, FieldStackedView], int]:
    """Interleaves fields sharing a space into one block, one dof's values contiguous.

    Every field in `items` is shaped `(n_scalar_dofs, *components)`, so the block is
    `(n_scalar_dofs, total_components)` and each field owns a column range of it.

    Args:
        items: The `(name, spec)` pairs to interleave. All must be on the same space.
        offset: Where the block starts in the flat array.

    Returns:
        The descriptors by name, and the number of entries the block occupies.
    """
    if not items:
        return {}, 0

    n_dofs = items[0][1].shape[0]
    layout: list[tuple[str, _FieldSpec, int, int]] = []
    width = 0
    for name, spec in items:
        if spec.shape[0] != n_dofs:
            raise CompoundStackError(
                f"field {name!r} has a leading dimension of {spec.shape[0]} but is "
                f"interleaved with fields of {n_dofs}; a block needs one dof axis, so only "
                f"fields agreeing on it can share one"
            )
        extent = int(prod(spec.shape[1:])) if len(spec.shape) > 1 else 1
        layout.append((name, spec, width, width + extent))
        width += extent

    block_shape = (n_dofs, width)
    block_size = int(prod(block_shape))
    block = _FieldBase(block_shape, _slice=slice(offset, offset + block_size))

    descriptors: dict[str, FieldStackedView] = {}
    for name, spec, start, end in layout:
        descriptors[name] = FieldStackedView(
            shape=spec.shape,
            default_factory=spec.default_factory,
            parent_field=block,
            parent_slice=(slice(None), slice(start, end)),
            field_type=spec.field_type,
        )
    return descriptors, block_size


def _standard_descriptors(
    specs: Sequence[tuple[str, _FieldSpec]], offset: int
) -> tuple[dict[str, Field], int]:
    """Gives each field its own contiguous slice, in declaration order.

    Args:
        specs: The `(name, spec)` pairs to lay out.
        offset: Where the first field starts in the flat array.

    Returns:
        The descriptors by name, and the number of entries they occupy in total.
    """
    descriptors: dict[str, Field] = {}
    current = offset
    for name, spec in specs:
        n = int(prod(spec.shape))
        descriptors[name] = Field(
            shape=spec.shape,
            default_factory=spec.default_factory,
            field_type=spec.field_type,
            _slice=slice(current, current + n),
        )
        current += n
    return descriptors, current - offset
