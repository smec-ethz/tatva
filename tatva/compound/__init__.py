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

import logging
from dataclasses import replace
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Literal,
    Self,
    Sequence,
    TypeVar,
    overload,
)

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class

from tatva.compound.field import (
    Field,
    FieldSize,
    FieldStackedView,
    FieldType,
    _FieldBase,
    _FieldSpec,
    field,
)
from tatva.compound.mpi import GlobalView

if TYPE_CHECKING:
    from mpi4py import MPI
    from scipy.sparse import csr_matrix

    from tatva.compound.mpi import _FieldGlobalInfo, _LocalLayout
    from tatva.mesh import Mesh, PartitionInfo

__all__ = ["Compound", "field", "stack_fields"]

log = logging.getLogger(__name__)

T_Compound = TypeVar("T_Compound", bound="Compound")


class CompoundError(ValueError):
    """Base error class for Compound-related errors."""


class CompoundStackError(CompoundError):
    pass


class Compound:
    """A compound array/state.

    A helper class to create a compound state with multiple fields. It simplifies packing
    and unpacking into and from a flat array. Useful to manage fields while working with a
    flat array for the solver.

    Args:
        arr: The flat data array. If None, initializes to zeros.

    Examples:

    Create a compound state with fields::

        class MyState(Compound):
            u = field(shape=(N, 3))
            phi = field(shape=(N,), default_factory=lambda: jnp.ones(N))

        state = MyState()

    Use `state.arr` to access the flat array, and `state.u`, `state.phi` to access the
    individual fields.

    You can use iterator unpacking to directly unpack the fields from the state::

        u, phi = MyState(arr)

    """

    fields: tuple[tuple[str, Field], ...] = ()
    arr: Array
    size: int = 0

    _global: GlobalView = GlobalView()
    _g = _global

    _layout: ClassVar[_LocalLayout | None] = None
    _global_field_info: ClassVar[dict[str, _FieldGlobalInfo] | None] = None
    _comm: ClassVar[MPI.Comm | None] = None
    _mesh: ClassVar[Mesh | None] = None

    def __init_subclass__(
        cls,
        *,
        mesh: Mesh | None = None,
        partition_info: PartitionInfo | None = None,
        comm: MPI.Comm | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__()

        # Collect fields from base classes first
        all_fields = _get_inherited_fields(cls)
        current_offset = sum(int(prod(f.shape)) for _, f in all_fields)

        # Collect all new field specs in order
        new_specs = _collect_field_specs(cls.__dict__)

        # validate all specs have valid names
        # we cannot allow field names to conflict with existing attributes/methods of the
        # base Compound class
        reserved_names = set(dir(Compound))
        reserved_names.add("arr")
        for name, _ in new_specs:
            if name in reserved_names:
                raise CompoundError(
                    f"Field name '{name}' is reserved and cannot be used in Compound class."
                )

        # Separate auto-nodal and other fields for layout calculation
        full_nodal_specs: list[tuple[str, _FieldSpec]] = []
        # full nodal means that field_type is Nodal with node_ids=None, i.e. the
        # field is defined on all nodes of the mesh and can be stacked together
        other_specs: list[tuple[str, _FieldSpec]] = []

        from tatva.compound.field_types import Nodal

        for name, spec in new_specs:
            ft_obj = spec.field_type.get()

            if _is_auto_nodal(spec):
                # if the first dimension is AUTO, we set it to Nodal, and resolve the
                # shape to be (num_items, *rest).
                if isinstance(ft_obj, Nodal) and ft_obj.node_ids is not None:
                    n_items_field = len(ft_obj.node_ids)
                else:
                    if mesh is None:
                        raise CompoundError(
                            f"Mesh must be provided to resolve AUTO size for field '{name}'."
                        )
                    n_items_field = mesh.coords.shape[0]

                spec = replace(
                    spec,
                    shape=(n_items_field, *spec.shape[1:]),
                    field_type=ft_obj if isinstance(ft_obj, Nodal) else FieldType.NODAL,
                )

            field_type_obj = spec.field_type.get()
            if (
                isinstance(field_type_obj, Nodal)
                and (field_type_obj.node_ids is None)
                and field_type_obj.stack
            ):
                full_nodal_specs.append((name, spec))
            else:
                other_specs.append((name, spec))

        # Create descriptors for all new fields
        name_to_descriptor: dict[str, Field] = {}

        # Optimization: If only one field is planned for stacking, don't stack it.
        # This avoids the overhead of FieldStackedView (reshaping/slicing on every access).
        if len(full_nodal_specs) == 1:
            other_specs = full_nodal_specs + other_specs
            full_nodal_specs = []

        # 1. Process auto-nodal fields first for memory layout (stacked block)
        if full_nodal_specs:
            descriptors, total_size = _create_stacked_descriptors(
                full_nodal_specs, current_offset, axis=1
            )
            name_to_descriptor.update(descriptors)
            current_offset += total_size

        # 2. Process other fields for memory layout
        descriptors, total_size = _create_standard_descriptors(
            other_specs, current_offset
        )
        name_to_descriptor.update(descriptors)
        current_offset += total_size

        # 3. Assemble cls.fields in the original user-defined order
        for name, _ in new_specs:
            f = name_to_descriptor[name]
            setattr(cls, name, f)
            all_fields.append((name, f))

        cls.fields = tuple(all_fields)
        cls.size = current_offset
        cls._mesh = mesh

        if (partition_info is not None) and (comm is not None):
            cls._set_mpi_context(partition_info, comm)

        # register as pytree node for JAX transformations
        register_pytree_node_class(cls)

    def tree_flatten(self) -> tuple[tuple[Array], Any]:
        return (self.arr,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Array]) -> Self:
        return cls(*children)

    @classmethod
    def _set_mpi_context(cls, partition_info: PartitionInfo, comm: MPI.Comm) -> None:
        if cls._mesh is None:
            raise CompoundError("Mesh must be set on Compound class to set mpi layout.")

        from tatva.compound.mpi import _layout_from_compound

        layout, global_field_info = _layout_from_compound(cls, partition_info, comm)

        cls._comm = comm
        cls._layout = layout
        cls._global_field_info = global_field_info

    @classmethod
    def get_layout(cls) -> _LocalLayout:
        """MPI only. Get the local layout information for this compound state. Requires
        that the class has been initialized with MPI context (mesh, partition_info,
        comm)."""
        if cls._layout is None:
            raise CompoundError("Layout not set on Compound class.")
        return cls._layout

    @overload
    @classmethod
    def get_sparsity(
        cls,
        couple_fields: tuple[Field, ...] = (),
        block_wise: Literal[False] = False,
    ) -> csr_matrix: ...
    @overload
    @classmethod
    def get_sparsity(
        cls,
        couple_fields: tuple[Field, ...] = (),
        block_wise: Literal[True] = True,
    ) -> list[list[csr_matrix]]: ...
    @classmethod
    def get_sparsity(
        cls,
        couple_fields: tuple[Field, ...] = (),
        block_wise: bool = False,
    ) -> csr_matrix | list[list[csr_matrix]]:
        """Create a sparsity pattern automatically from this Compound class and its
        attached mesh.

        Stacked nodal fields are fully coupled within elements. Other nodal fields can be
        coupled if wanted. All other fields (Local, Shared) are only connected to themselves
        (diagonal entries). For non-trivial couplings, you must create the sparsity
        pattern manually.

        Args:
            couple_fields: Tuple of fields to fully couple. Only nodal fields are supported.
            block_wise: If True, return the pattern as a list of lists of sparse matrices
                corresponding to the compound fields/blocks. Stacked fields are one block.
        """
        from tatva.sparse._extraction import create_sparsity_pattern_from_compound

        if cls._mesh is None:
            raise CompoundError(
                "Mesh must be set on Compound class to create sparsity pattern."
            )

        return create_sparsity_pattern_from_compound(
            cls,
            cls._mesh,
            couple_fields,
            block_wise=block_wise,
        )

    def __init__(self, arr: Array | None = None, **kwargs) -> None:
        """Initialize the state with given keyword arguments."""
        if arr is not None:
            assert arr.size == self.size, (
                f"Array size {arr.size} does not match expected size {self.size}."
            )
            self.arr = arr
        else:
            self.arr = jnp.zeros(self.size, dtype=float)
            # initialize fields with provided arrays OR default factories if available
            for name, field_obj in self.fields:
                if name in kwargs:
                    field_arr = kwargs[name]
                    self.arr = field_obj._set_in_array(self.arr, jnp.asarray(field_arr))
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
        # print shape of each field in the class
        field_reprs = [
            f"{name}={getattr(type(self), name).shape}" for name, _ in self.fields
        ]
        return f"{self.__class__.__name__}({', '.join(field_reprs)})"

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.arr + other.arr)

    def at(self, name: str) -> _CompoundAtHelper[Self]:
        """Helper for functional updates to fields. Usage: `new_state =
        state.at('field_name').set(new_value)`.

        Args:
            name: The name of the field to update.
        """
        field_obj = dict(self.fields).get(name)
        if field_obj is None:
            raise AttributeError(f"Unknown field name: {name}")
        return _CompoundAtHelper(self, field_obj)

    def flatten(self) -> Array:
        """Return the flat array representation of the state. Same as `state.arr`."""
        return self.arr


class _CompoundAtHelper(Generic[T_Compound]):
    def __init__(self, state: T_Compound, field_obj: Field):
        self.state = state
        self.field_obj = field_obj

    def set(self, value: Array | float) -> T_Compound:
        return self.state.__class__(
            self.field_obj._set_in_array(self.state.arr, jnp.asarray(value)),
        )


def _get_inherited_fields(cls: type[Compound]) -> list[tuple[str, Field]]:
    """Return fields inherited from the first Compound base class found in MRO."""
    for base in cls.__mro__[1:]:
        if issubclass(base, Compound) and base is not Compound:
            return list(base.fields)
    return []


def _collect_field_specs(cls_dict: dict[str, Any]) -> list[tuple[str, _FieldSpec]]:
    """Collect all _FieldSpec attributes from the class dictionary."""
    return [
        (name, val) for name, val in cls_dict.items() if isinstance(val, _FieldSpec)
    ]


def _is_auto_nodal(spec: _FieldSpec) -> bool:
    """Check if a field spec is an auto-sized nodal field."""
    return len(spec.shape) > 0 and spec.shape[0] == FieldSize.AUTO


def _create_stacked_descriptors(
    items: Sequence[tuple[str, _FieldSpec | Field]],
    offset: int,
    axis: int = -1,
) -> tuple[dict[str, FieldStackedView], int]:
    """Unified logic to create stacked descriptors and calculate layout."""
    if not items:
        return {}, 0

    # 1. Find stacking dimension and prefix
    first_shape = items[0][1].shape
    actual_axis = axis % len(first_shape) if len(first_shape) > 0 else 0
    base_prefix = first_shape[:actual_axis]

    # 2. Validate prefix compatibility and calculate layout
    stack_layout: list[tuple[str, _FieldSpec | Field, int, int]] = []
    stack_offset: int = 0
    for name, item in items:
        shape = item.shape
        if len(shape) < actual_axis:
            raise CompoundStackError(
                f"Field {name} rank {len(shape)} is not compatible with stacking axis {actual_axis}."
            )
        if shape[:actual_axis] != base_prefix:
            raise CompoundStackError(
                f"Field {name} shape {shape} prefix does not match base shape along axis {actual_axis}."
            )

        suffix = shape[actual_axis:]
        extent = int(prod(suffix)) if suffix else 1
        start = stack_offset
        end = stack_offset + extent
        stack_layout.append((name, item, start, end))
        stack_offset += extent

    stacked_shape_tuple = base_prefix + (stack_offset,)
    stacked_total_size = int(prod(stacked_shape_tuple))

    stacked_field_base = _FieldBase(
        stacked_shape_tuple, _slice=slice(offset, offset + stacked_total_size)
    )

    # 3. Create views
    descriptors = {}
    for name, item, start, end in stack_layout:
        parent_slice = [slice(None)] * len(base_prefix) + [slice(start, end)]

        descriptors[name] = FieldStackedView(
            shape=item.shape,
            default_factory=item.default_factory,
            parent_field=stacked_field_base,
            parent_slice=tuple(parent_slice),
            field_type=item.field_type,
        )

    return descriptors, stacked_total_size


def _create_standard_descriptors(
    specs: list[tuple[str, _FieldSpec]], offset: int
) -> tuple[dict[str, Field], int]:
    """Create descriptors for standard (non-auto-nodal) fields."""
    descriptors = {}
    current_offset = offset
    for name, spec in specs:
        n = int(prod(spec.shape))
        s = slice(current_offset, current_offset + n)

        descriptors[name] = Field(
            shape=spec.shape,
            default_factory=spec.default_factory,
            field_type=spec.field_type,
            _slice=s,
        )
        current_offset += n
    return descriptors, current_offset - offset


def _apply_stack_fields(
    cls: type[T_Compound], stack_names: tuple[str, ...], axis: int = -1
) -> type[T_Compound]:
    """Reorder and stack specified fields in a Compound class. Modifies class in place."""
    fields_map = {name: f for name, f in cls.fields}
    for name in stack_names:
        if name not in fields_map:
            raise CompoundStackError(f"Unknown field name in stack_fields: {name}")

    involved_names = set(stack_names)
    first_idx = min(i for i, (n, _) in enumerate(cls.fields) if n in involved_names)
    base_offset = cls.fields[first_idx][1]._base_offset
    if base_offset is None:
        raise RuntimeError("Base offset not set on involved fields.")

    items_to_stack = [(name, fields_map[name]) for name in stack_names]
    descriptors, stack_size = _create_stacked_descriptors(
        items_to_stack, base_offset, axis
    )

    new_fields: list[tuple[str, Field]] = []
    current_offset = base_offset + stack_size

    # Re-build cls.fields while preserving original order
    for i, (name, f) in enumerate(cls.fields):
        if i < first_idx:
            new_fields.append((name, f))
            continue

        if name in involved_names:
            new_f = descriptors[name]
            setattr(cls, name, new_f)
            new_fields.append((name, new_f))
        else:
            # Re-slice trailing field
            n = int(prod(f.shape))
            new_slice = slice(current_offset, current_offset + n)
            f._set_slice_and_strides(new_slice)
            new_fields.append((name, f))
            current_offset += n

    cls.fields = tuple(new_fields)
    cls.size = current_offset
    return cls


def stack_fields(
    *fields: str, axis: int = -1
) -> Callable[[type[T_Compound]], type[T_Compound]]:
    """Class decorator to stack compatible fields into a shared contiguous layout."""
    if not fields:
        raise CompoundError("At least one field name is required.")

    names = tuple(fields)

    def decorator(cls: type[T_Compound]) -> type[T_Compound]:
        return _apply_stack_fields(cls, names, axis=axis)

    return decorator
