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
from mpi4py import MPI
from numpy.typing import NDArray

from tatva.compound.field import FieldType, _normalize_index, _row_major_strides
from tatva.mesh import PartitionInfo
from tatva.mpi import _LocalLayout

if TYPE_CHECKING:
    from tatva.compound import Compound

log = logging.getLogger(__name__)


class _FieldGlobalInfo(NamedTuple):
    global_slice: slice
    global_shape: tuple[int, ...]
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
            info.global_shape, info.global_slice.start, info.global_subset
        )


class _GlobalFieldIndices:
    """Indexing logic for natural global DOF IDs of a field."""

    def __init__(
        self,
        shape: tuple[int, ...],
        base_offset: int,
        global_subset: NDArray[np.int32] | None = None,
    ):
        self.shape = shape
        self._base_offset = base_offset
        self._strides = _row_major_strides(shape)
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

            g_slice, g_shape, _ = field_info[name]
            return g_arr[g_slice].reshape(g_shape)

        raise AttributeError(
            f"'{self._compound.__class__.__name__}' has no field '{name}'"
        )


def _dof_map_from_node_map(
    node_map: NDArray[np.int32], n_dofs_per_node: int
) -> NDArray[np.int32]:
    """Convert a node map to a degree of freedom (DOF) map."""
    dof_map = node_map[:, None] * n_dofs_per_node + np.arange(n_dofs_per_node)
    return dof_map.flatten().astype(np.int32)


def _create_dof_layout(
    natural_dof_map: NDArray[np.int32],
    owned_mask: NDArray[np.bool_],
    n_natural_global: int,
    comm: MPI.Comm,
) -> _LocalLayout:
    """Create a global DOF layout for all DOFs across all processes."""
    _dtype = np.int32
    n_owned = int(np.sum(owned_mask))
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    n_per_rank = comm.allgather(n_owned)
    offset = (
        np.cumsum([0] + n_per_rank[: comm.rank], dtype=_dtype)[-1]
        if comm.rank > 0
        else 0
    )

    # Assign global indices to owned DOFs
    l2g = np.full(natural_dof_map.size, -1, dtype=_dtype)
    owned_indices = np.where(owned_mask)[0]
    l2g[owned_indices] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using the original global IDs (natural_dof_map)
    local_directory = np.full(n_natural_global, -1, dtype=_dtype)
    local_directory[natural_dof_map[owned_indices]] = l2g[owned_indices]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    ghost_indices = np.where(~owned_mask)[0]
    if ghost_indices.size > 0:
        l2g[ghost_indices] = global_directory[natural_dof_map[ghost_indices]]

    log.debug(
        f"Rank {comm.rank}: DOF layout - n_owned={n_owned}, "
        "n_total={natural_dof_map.size}, n_global={n_global}, offset={offset}"
    )

    return _LocalLayout(
        local_to_global=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=natural_dof_map.size,
        n_global=n_global,
        owned_mask=owned_mask,
        natural_l2g=natural_dof_map,
    )


def _layout_from_compound(
    compound_cls: type[Compound],
    partition_info: PartitionInfo,
    comm: MPI.Comm,
) -> tuple[_LocalLayout, dict[str, _FieldGlobalInfo]]:
    """Create a DOF layout from the compound class, partition info, and lifter."""
    _rank = comm.Get_rank()
    natural_dof_map = np.full(compound_cls.size, -1, dtype=np.int32)
    owned_mask = np.zeros(compound_cls.size, dtype=bool)

    # figure out the total number of nodes globally by finding the max global node
    # index across all ranks, then add 1 since node indices are zero-based
    local_max_node = (
        np.max(partition_info.nodes_local_to_global)
        if partition_info.nodes_local_to_global.size > 0
        else -1
    )
    n_nodes_global = comm.allreduce(local_max_node, op=MPI.MAX) + 1

    # loop over fields in the compound, assigning global DOF indices to owned DOFs and
    # building the natural DOF map for ghost resolution
    current_natural_offset: int = 0
    processed_slices: set[slice] = set()
    field_global_info: dict[str, _FieldGlobalInfo] = {}

    for name, f in compound_cls.fields:
        # if stacked fields: they have a _root_slice that identifies the original
        # slice of the base field; use that for DOF mapping
        # the cache processed_slices ensures, further fields stacked on the same base
        # field are skipped
        f_slice = getattr(f, "_root_slice", f._slice)
        f_slice = cast(slice, f_slice)

        # Calculate global shape and natural offset for this field
        # We do this for every field, even if it shares a root slice
        root_shape = getattr(f, "_root_shape", f.shape)
        g_subset = None
        if f.field_type == FieldType.LOCAL:
            f_size_local = f_slice.stop - f_slice.start
            f_size_global = comm.allreduce(f_size_local, op=MPI.SUM)
            n_items_global = f_size_global // int(prod(root_shape[1:]))
        elif f.field_type == FieldType.NODAL:
            # For incomplete fields, global size is more complex.
            # We need to know the total unique nodes in this subset across all ranks.
            if f.nodal_local_to_global is not None:
                subset_global_nodes = np.asarray(f.nodal_local_to_global)
                all_subset = comm.allgather(subset_global_nodes)
                global_subset = np.unique(np.concatenate(all_subset))
                n_items_global = len(global_subset)
                g_subset = global_subset
            else:
                n_items_global = n_nodes_global
            f_size_global = n_items_global * int(prod(root_shape[1:]))
        elif f.field_type == FieldType.SHARED:
            f_size_global = f_slice.stop - f_slice.start
            n_items_global = f_size_global // int(prod(root_shape[1:]))

        field_global_info[name] = _FieldGlobalInfo(
            global_slice=slice(
                current_natural_offset, current_natural_offset + f_size_global
            ),
            global_shape=(n_items_global, *f.shape[1:]),
            global_subset=g_subset,
        )

        if f_slice in processed_slices:
            continue
        processed_slices.add(f_slice)

        f_size_local = f_slice.stop - f_slice.start

        if f.field_type == FieldType.LOCAL:
            rank_offset = comm.exscan(f_size_local, op=MPI.SUM)  # ty:ignore[unresolved-attribute]
            if _rank == 0:
                rank_offset = 0

            natural_dof_map[f_slice] = (
                current_natural_offset + rank_offset + np.arange(f_size_local)
            )
            owned_mask[f_slice] = True

        elif f.field_type == FieldType.NODAL:
            root_shape = getattr(f, "_root_shape", f.shape)
            n_dofs_per_node = int(np.prod(root_shape[1:]))

            if f.nodal_local_to_global is not None:
                # Note: incomplete nodal fields could be lagrange multipliers attached
                # to a subset of nodes
                # incomplete subset of nodes, possibly different across ranks. Use the
                # provided local_to_global mapping to identify the global nodes in
                # this subset, then build the DOF map from that.
                subset_global_nodes = np.asarray(f.nodal_local_to_global)
                field_dof_map = _dof_map_from_node_map(
                    subset_global_nodes, n_dofs_per_node
                )
                natural_dof_map[f_slice] = current_natural_offset + field_dof_map

                owned_global_nodes = partition_info.nodes_local_to_global[
                    : partition_info.n_owned_nodes
                ]
                is_owned_node = np.isin(subset_global_nodes, owned_global_nodes)
                field_owned_mask = np.repeat(is_owned_node, n_dofs_per_node)
                owned_mask[f_slice.start : f_slice.stop] = field_owned_mask

                # Re-map natural indices to be contiguous within this field's global space
                # to avoid massive gaps in the directory array.
                subset_node_indices = np.searchsorted(
                    global_subset, subset_global_nodes
                )
                natural_dof_map[f_slice] = (
                    current_natural_offset
                    + np.repeat(subset_node_indices, n_dofs_per_node) * n_dofs_per_node
                    + np.tile(np.arange(n_dofs_per_node), len(subset_global_nodes))
                )
            else:
                # Full nodal field
                field_dof_map = _dof_map_from_node_map(
                    partition_info.nodes_local_to_global, n_dofs_per_node
                )
                natural_dof_map[f_slice] = current_natural_offset + field_dof_map

                n_owned_in_field = partition_info.n_owned_nodes * n_dofs_per_node
                owned_mask[f_slice.start : f_slice.start + n_owned_in_field] = True

        elif f.field_type == FieldType.SHARED:
            natural_dof_map[f_slice] = current_natural_offset + np.arange(f_size_local)
            if _rank == 0:
                owned_mask[f_slice] = True

        current_natural_offset += f_size_global

    # from the natural DOF map and owned mask, build the global DOF layout and routing
    # tables for ghost exchange
    # we only count DOFs as owned if they are both owned by the rank and free
    layout = _create_dof_layout(
        natural_dof_map, owned_mask, current_natural_offset, comm
    )

    return layout, field_global_info
