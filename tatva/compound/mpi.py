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

import logging
from math import prod
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, cast, overload

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from tatva.compound import field_types
from tatva.compound.field import (
    FieldStackedView,
    _normalize_index,
    _reshape_affine_metadata,
    _row_major_strides,
)
from tatva.mesh import PartitionInfo

if TYPE_CHECKING:
    from mpi4py import MPI

    from tatva.compound import Compound
    from tatva.mpi import _LocalLayout

log = logging.getLogger(__name__)


class _FieldGlobalInfo(NamedTuple):
    global_shape: tuple[int, ...]
    global_base_offset: int
    global_strides: tuple[int, ...]
    global_subset: NDArray[np.int32] | None = None


T_Compound = TypeVar("T_Compound", bound="Compound")


class GlobalView:
    """Descriptor for the `.g` attribute on Compound instances and classes, providing
    access to global fields and indices."""

    @overload
    def __get__(
        self, instance: None, owner: type[T_Compound]
    ) -> GlobalIndicesView[T_Compound]: ...
    @overload
    def __get__(
        self, instance: T_Compound, owner: type[T_Compound]
    ) -> GlobalDataView[T_Compound]: ...
    def __get__(
        self, instance: None | T_Compound, owner: type[T_Compound]
    ) -> GlobalIndicesView[T_Compound] | GlobalDataView[T_Compound]:
        if owner._layout is None:
            raise ValueError(
                f"Compound class '{owner.__name__}' is missing a DOF layout. "
                "Global view requires a complete MPI layout."
            )

        if instance is None:
            return GlobalIndicesView(owner)

        return GlobalDataView(instance)


class GlobalIndicesView(Generic[T_Compound]):
    """Provides global indexing accessors for fields."""

    def __init__(self, compound: type[T_Compound]):
        self._compound = compound

    def __getattr__(self, name: str) -> _GlobalFieldIndices:
        field_info = self._compound._global_field_info
        if field_info is None or name not in field_info:
            raise AttributeError(
                f"'{self._compound.__class__.__name__}' has no field '{name}' "
                "or layout not initialized."
            )

        info = field_info[name]
        return _GlobalFieldIndices(
            info.global_shape,
            info.global_base_offset,
            info.global_strides,
            info.global_subset,
        )


class _GlobalFieldIndices:
    """Indexing logic for natural global DOF IDs of a field."""

    def __init__(
        self,
        shape: tuple[int, ...],
        base_offset: int,
        strides: tuple[int, ...],
        global_subset: NDArray[np.int32] | None = None,
    ):
        self.shape = shape
        self._base_offset = base_offset
        self._strides = strides
        self.global_subset = global_subset

    def __getitem__(self, arg) -> Array:
        arg = _normalize_index(arg, self.shape)

        # If global_subset is present, the first index is assumed to be global node IDs.
        # We must translate them to subset indices.
        if self.global_subset is not None:
            first_idx = arg[0]
            if isinstance(first_idx, (int, np.integer)):
                # Scalar lookup
                pos = np.searchsorted(self.global_subset, first_idx)
                if (
                    pos >= len(self.global_subset)
                    or self.global_subset[pos] != first_idx
                ):
                    raise IndexError(f"Global node ID {first_idx} not in field subset.")
                arg = (int(pos),) + arg[1:]
            elif isinstance(first_idx, slice):
                raise NotImplementedError("Slicing on global subset not supported yet.")
            else:
                # Array lookup
                first_idx_arr = np.asarray(first_idx)
                pos = np.searchsorted(self.global_subset, first_idx_arr)
                # Verify all matches
                valid = (pos < len(self.global_subset)) & (
                    self.global_subset[np.minimum(pos, len(self.global_subset) - 1)]
                    == first_idx_arr
                )
                if not np.all(valid):
                    missing = first_idx_arr[~valid]
                    raise IndexError(
                        f"Global node IDs {missing} not found in field subset."
                    )
                arg = (pos.astype(np.int32),) + arg[1:]

        # Perform standard multi-dimensional indexing math
        axis_indices: list[Array] = []
        for sub, extent in zip(arg, self.shape, strict=True):
            if isinstance(sub, (int, np.integer)):
                idx = sub if sub >= 0 else sub + extent
                axis_indices.append(jnp.asarray([idx], dtype=int))
            elif isinstance(sub, slice):
                start, stop, step = sub.indices(extent)
                axis_indices.append(jnp.arange(start, stop, step, dtype=int))
            else:
                values = jnp.asarray(sub, dtype=int).reshape(-1)
                axis_indices.append(jnp.where(values < 0, values + extent, values))

        if not axis_indices:
            return jnp.asarray([self._base_offset], dtype=int)

        grid_shape = tuple(len(values) for values in axis_indices)
        index = jnp.full(grid_shape, self._base_offset, dtype=int)
        for axis, (stride, values) in enumerate(
            zip(self._strides, axis_indices, strict=True)
        ):
            broadcast_shape = tuple(
                len(values) if i == axis else 1 for i in range(len(axis_indices))
            )
            index = index + jnp.reshape(
                values * stride,
                broadcast_shape,
            )
        return index.reshape(-1)


class GlobalDataView(Generic[T_Compound]):
    """A handle to access global fields and natural DOF indices.

    If accessed from a Compound instance, it can gather and return the full global
    state array. If accessed from a Compound class, it only provides access to
    global natural DOF indices via the `.indices` property.
    """

    def __init__(self, compound: T_Compound | type[T_Compound]):
        self._compound = compound
        self._cache: dict[str, Array] = {}

    def _gather(self) -> Array:
        """Gather the full natural global state array."""
        if isinstance(self._compound, type):
            raise TypeError(
                "Gathering global data requires a Compound instance, not the class."
            )

        if "full" not in self._cache:
            import mpi4jax
            from mpi4py import MPI

            layout = self._compound._layout
            if (
                layout is None
                or self._compound._comm is None
                or layout.natural_l2g is None
            ):
                raise ValueError(
                    "Compound layout or communicator not set. "
                    "Global view requires a complete MPI layout."
                )

            # Zero out non-owned DOFs to ensure correct sum during allreduce
            owned_arr = jnp.where(layout.owned_mask, self._compound.arr, 0.0)

            # Map to natural global indices and allreduce
            g_arr_local = (
                jnp.zeros(layout.n_global, dtype=self._compound.arr.dtype)
                .at[layout.natural_l2g]
                .set(owned_arr)
            )
            self._cache["full"] = mpi4jax.allreduce(
                g_arr_local, op=MPI.SUM, comm=self._compound._comm
            )

        return self._cache["full"]

    def __getattr__(self, name: str) -> Array:
        field_info = self._compound._global_field_info

        if name in dict(self._compound.fields):
            g_arr = self._gather()
            layout = self._compound._layout
            if layout is None or field_info is None:
                raise ValueError("Field global info not available in layout.")

            info = field_info[name]
            # Use advanced indexing to get the correct view of the global array
            indices = _GlobalFieldIndices(
                info.global_shape,
                info.global_base_offset,
                info.global_strides,
                info.global_subset,
            )
            # return g_arr[indices[:]].reshape(info.global_shape)
            # wait, g_arr is flat. indices[:] gives flat indices.
            return g_arr[indices[:]].reshape(info.global_shape)

        raise AttributeError(
            f"'{self._compound.__class__.__name__}' has no field '{name}'"
        )


def _dof_map_from_node_map(
    node_map: NDArray[np.int32], n_dofs_per_node: int
) -> NDArray[np.int32]:
    """Convert a node map to a degree of freedom (DOF) map."""
    dof_map = node_map[:, None] * n_dofs_per_node + np.arange(n_dofs_per_node)
    return dof_map.flatten().astype(np.int32)


def _layout_from_compound(
    compound_cls: type[Compound],
    partition_info: PartitionInfo,
    comm: MPI.Comm,
) -> tuple[_LocalLayout, dict[str, _FieldGlobalInfo]]:
    """Create a DOF layout from the compound class and mesh partition info.

    This function performs a collective initialization of the MPI layout for a Compound
    class. It determines how each field is distributed across ranks, computes global
    natural DOF indices, and identifies which DOFs are owned by each process.
    """
    from mpi4py import MPI

    from tatva.mpi import _create_dof_layout

    _rank = comm.Get_rank()
    natural_dof_map = np.full(compound_cls.size, -1, dtype=np.int32)
    owned_mask = np.zeros(compound_cls.size, dtype=bool)

    # STEP 1: Determine total global nodes
    # We find the maximum global node index across all ranks to compute the total count.
    local_max_node = (
        np.max(partition_info.nodes_local_to_global)
        if partition_info.nodes_local_to_global.size > 0
        else -1
    )
    n_nodes_global = comm.allreduce(local_max_node, op=MPI.MAX) + 1

    # Initialization of tracking state for natural global offsets
    current_natural_offset: int = 0
    processed_slices: set[slice] = set()
    # Cache for global metadata of root fields (base of stacked layouts)
    # root_slice -> (natural_offset, n_items_global, root_strides)
    root_global_info: dict[slice, tuple[int, int, tuple[int, ...]]] = {}
    field_global_info: dict[str, _FieldGlobalInfo] = {}

    # STEP 2: Iterate over each field to establish global metadata and DOF mappings
    for name, f in compound_cls.fields:
        # Resolve the root slice. For stacked fields, multiple fields share the same root.
        f_slice = getattr(f, "_root_slice", f._slice)
        f_slice = cast(slice, f_slice)

        field_type_obj = f.field_type.get()
        root_shape = getattr(f, "_root_shape", f.shape)

        # STEP 2a: Initialize global metadata for the root field if not already done
        if f_slice not in root_global_info:
            if isinstance(field_type_obj, field_types.Local):
                # Local fields: sum total size across all ranks
                f_size_local = f_slice.stop - f_slice.start
                f_size_global = comm.allreduce(f_size_local, op=MPI.SUM)
                n_items_global = f_size_global // int(prod(root_shape[1:]))
            elif isinstance(field_type_obj, field_types.Nodal):
                # Nodal fields: items correspond to unique global nodes
                if field_type_obj.node_ids is not None:
                    # Incomplete nodal subset: collective gathering of unique global node IDs
                    subset_global_nodes = np.asarray(field_type_obj.node_ids)
                    all_subset = comm.allgather(subset_global_nodes)
                    global_subset = np.unique(np.concatenate(all_subset))
                    n_items_global = len(global_subset)
                else:
                    n_items_global = n_nodes_global
            elif isinstance(field_type_obj, field_types.Shared):
                # Shared fields: replicated size, one owner (rank 0)
                f_size_global = f_slice.stop - f_slice.start
                n_items_global = f_size_global // int(prod(root_shape[1:]))
            else:
                raise TypeError(f"Unsupported field type: {type(field_type_obj)}")

            # Store the global natural start, item count, and row-major strides for the root field
            root_global_info[f_slice] = (
                current_natural_offset,
                n_items_global,
                _row_major_strides((n_items_global, *root_shape[1:])),
            )

        root_start, n_items_global, root_strides = root_global_info[f_slice]

        # STEP 2b: Calculate specific global metadata for this field (handles stacked views)
        # Nodal subset for incomplete fields
        g_subset = None
        if (
            isinstance(field_type_obj, field_types.Nodal)
            and field_type_obj.node_ids is not None
        ):
            subset_global_nodes = np.asarray(field_type_obj.node_ids)
            all_subset = comm.allgather(subset_global_nodes)
            g_subset = np.unique(np.concatenate(all_subset))

        if hasattr(f, "_view_slice"):
            # If the field is a view into a stacked layout, derive its global affine metadata
            # if it has _view_slice, it must be FieldStackedView
            f = cast(FieldStackedView, f)
            g_base_offset = root_start
            g_root_shape = (n_items_global, *root_shape[1:])
            g_view_shape = list(g_root_shape)
            g_strides = list(root_strides)

            # Map the local view slice to global coordinates
            for axis, sub in enumerate(f._view_slice):
                if isinstance(sub, int):
                    idx = sub % g_root_shape[axis]
                    g_base_offset += idx * g_strides[axis]
                    g_view_shape[axis] = 1
                else:  # sub is a slice
                    start, stop, step = sub.indices(g_root_shape[axis])
                    g_base_offset += start * g_strides[axis]
                    g_strides[axis] *= step
                    g_view_shape[axis] = len(range(start, stop, step))

            # Reshape affine metadata to match the user's requested field shape
            _, reshaped_strides = _reshape_affine_metadata(
                tuple(g_view_shape), tuple(g_strides), (n_items_global, *f.shape[1:])
            )
            field_global_info[name] = _FieldGlobalInfo(
                global_shape=(n_items_global, *f.shape[1:]),
                global_base_offset=g_base_offset,
                global_strides=reshaped_strides,
                global_subset=g_subset,
            )
        else:
            # Standard field: uses root global metadata directly
            field_global_info[name] = _FieldGlobalInfo(
                global_shape=(n_items_global, *f.shape[1:]),
                global_base_offset=root_start,
                global_strides=root_strides,
                global_subset=g_subset,
            )

        # STEP 2c: Assign local-to-global mappings (Natural DOF map)
        # Only process each root slice once to avoid redundant work in stacked fields.
        if f_slice in processed_slices:
            continue
        processed_slices.add(f_slice)

        f_size_local = f_slice.stop - f_slice.start
        f_size_global = n_items_global * int(prod(root_shape[1:]))

        if isinstance(field_type_obj, field_types.Local):
            # Local: assign natural indices using an exclusive scan for contiguous allocation
            rank_offset = comm.exscan(f_size_local, op=MPI.SUM)  # ty:ignore[unresolved-attribute]
            if _rank == 0:
                rank_offset = 0

            natural_dof_map[f_slice] = (
                current_natural_offset + rank_offset + np.arange(f_size_local)
            )
            owned_mask[f_slice] = True

        elif isinstance(field_type_obj, field_types.Nodal):
            # Nodal: map local node IDs to global IDs using partition information
            root_shape = getattr(f, "_root_shape", f.shape)
            n_dofs_per_node = int(np.prod(root_shape[1:]))

            if field_type_obj.node_ids is not None:
                # Incomplete Nodal: subset mapping
                subset_global_nodes = np.asarray(field_type_obj.node_ids)

                # Identify owned nodes within this subset
                owned_global_nodes = partition_info.nodes_local_to_global[
                    : partition_info.n_owned_nodes
                ]
                is_owned_node = np.isin(subset_global_nodes, owned_global_nodes)
                field_owned_mask = np.repeat(is_owned_node, n_dofs_per_node)
                owned_mask[f_slice.start : f_slice.stop] = field_owned_mask

                # Map subset node IDs to a contiguous global space to minimize directory fragmentation
                # if we reach this branch, we have set g_subset above. It is a ndarray at this point!
                subset_node_indices = np.searchsorted(g_subset, subset_global_nodes)  # ty:ignore[no-matching-overload]
                natural_dof_map[f_slice] = (
                    current_natural_offset
                    + np.repeat(subset_node_indices, n_dofs_per_node) * n_dofs_per_node
                    + np.tile(np.arange(n_dofs_per_node), len(subset_global_nodes))
                )
            else:
                # Full Nodal: map all local nodes
                field_dof_map = _dof_map_from_node_map(
                    partition_info.nodes_local_to_global, n_dofs_per_node
                )
                natural_dof_map[f_slice] = current_natural_offset + field_dof_map

                n_owned_in_field = partition_info.n_owned_nodes * n_dofs_per_node
                owned_mask[f_slice.start : f_slice.start + n_owned_in_field] = True

        elif isinstance(field_type_obj, field_types.Shared):
            # Shared: replicated across all ranks, owned by Rank 0
            natural_dof_map[f_slice] = current_natural_offset + np.arange(f_size_local)
            if _rank == 0:
                owned_mask[f_slice] = True

        # Advance global natural offset for the next root field
        current_natural_offset += f_size_global

    # STEP 3: Finalize the distributed DOF layout
    # Construct the global layout, determining routing tables for ghost exchange.
    layout = _create_dof_layout(
        natural_dof_map, owned_mask, current_natural_offset, comm
    )

    return layout, field_global_info
