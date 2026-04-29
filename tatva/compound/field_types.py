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

from dataclasses import dataclass, replace
from enum import IntEnum
from math import prod
from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.sparse as sps
from jax import Array
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tatva.compound.field import Field, _FieldSpec
    from tatva.mesh import Mesh


class FieldSize(IntEnum):
    AUTO = -1


class _FieldType:
    def get(self) -> Self:
        return self

    def resolve_spec(
        self, spec: _FieldSpec, mesh: Mesh | None
    ) -> tuple[_FieldSpec, bool]:
        if len(spec.shape) > 0 and spec.shape[0] == FieldSize.AUTO:
            # Fallback: if AUTO is used but type is not specialized, we default to Nodal
            # This preserves the behavior of the existing tests.
            return Nodal().resolve_spec(spec, mesh)
        return spec, False

    def get_element_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        return None

    def get_diagonal_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        return None

    def get_coupling_block(
        self, this: Field, other: Field, mesh: Mesh
    ) -> sps.csr_matrix:
        """Returns the sparsity pattern of the coupling between two fields."""
        edofs_this = self.get_element_dofs(this, mesh)
        other_type = other.field_type.get()
        edofs_other = other_type.get_element_dofs(other, mesh)

        if edofs_this is not None and edofs_other is not None:
            row_indices = np.repeat(edofs_this, edofs_other.shape[1], axis=1).flatten()
            col_indices = np.tile(edofs_other, (1, edofs_this.shape[1])).flatten()

            # Filter out -1
            valid = (row_indices >= 0) & (col_indices >= 0)
            row_indices = row_indices[valid]
            col_indices = col_indices[valid]
        elif this is other:
            row_indices = self.get_diagonal_dofs(this, mesh)
            if row_indices is None:
                return sps.csr_matrix((this.size, other.size), dtype=np.int8)
            col_indices = row_indices
        else:
            return sps.csr_matrix((this.size, other.size), dtype=np.int8)

        # Make relative to field base
        row_indices = row_indices - this._base_offset
        col_indices = col_indices - other._base_offset

        data = np.ones(row_indices.shape[0], dtype=np.int8)
        return sps.csr_matrix(
            (data, (row_indices, col_indices)), shape=(this.size, other.size)
        )

    def get_sparsity_pattern(self, field: Field, mesh: Mesh) -> sps.csr_matrix:
        return self.get_coupling_block(field, field, mesh)


@dataclass
class Nodal(_FieldType):
    node_ids: Array | NDArray | None = None
    stack: bool = True

    def resolve_spec(
        self, spec: _FieldSpec, mesh: Mesh | None
    ) -> tuple[_FieldSpec, bool]:
        if len(spec.shape) > 0 and spec.shape[0] == FieldSize.AUTO:
            if mesh is None:
                raise ValueError(
                    "Mesh must be provided when using AUTO shape with Nodal field type."
                )
            if self.node_ids is not None:
                n_items = len(self.node_ids)
            else:
                n_items = mesh.coords.shape[0]
            shape = (n_items, *spec.shape[1:])
        else:
            shape = spec.shape

        spec = replace(spec, shape=shape, field_type=self)
        should_stack = self.stack and self.node_ids is None
        return spec, should_stack

    def get_element_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        n_nodes = mesh.coords.shape[0]
        num_elements = mesh.elements.shape[0]
        indices = np.asarray(field.indices(slice(None)))

        n_items = len(self.node_ids) if self.node_ids is not None else n_nodes

        if n_items == 0:
            return None

        dofs_per_item = indices.size // n_items
        node_dofs = np.full((n_nodes, dofs_per_item), -1, dtype=np.int32)
        if self.node_ids is None:
            node_dofs[:] = indices.reshape(n_nodes, dofs_per_item)
        else:
            node_ids = np.asarray(self.node_ids, dtype=np.int32)
            valid_nodes = (node_ids >= 0) & (node_ids < n_nodes)
            node_dofs[node_ids[valid_nodes]] = indices.reshape(-1, dofs_per_item)[
                valid_nodes
            ]

        # Map nodes to elements
        return node_dofs[mesh.elements].reshape(num_elements, -1)


@dataclass
class Local(_FieldType):
    def get_diagonal_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        return np.asarray(field.indices(slice(None))).flatten()


@dataclass
class Shared(_FieldType):
    def get_diagonal_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        return np.asarray(field.indices(slice(None))).flatten()


class FieldType(IntEnum):
    LOCAL = 0
    NODAL = 1
    SHARED = 2

    def get(self) -> _FieldType:
        if self not in _enum_to_class:
            raise ValueError(f"Cannot get corresponding class for {self}")
        _cls = _enum_to_class[self]
        return _cls()


_enum_to_class: dict[FieldType, type[_FieldType]] = {
    FieldType.LOCAL: Local,
    FieldType.SHARED: Shared,
    FieldType.NODAL: Nodal,
}


@dataclass
class CG1(Nodal):
    submesh: Mesh | None = None


@dataclass
class DG0(Local):
    def resolve_spec(
        self, spec: _FieldSpec, mesh: Mesh | None
    ) -> tuple[_FieldSpec, bool]:
        if len(spec.shape) > 0 and spec.shape[0] == FieldSize.AUTO:
            if mesh is None:
                raise ValueError(
                    "Mesh must be provided when using AUTO shape with DG0 field type."
                )
            shape = (mesh.elements.shape[0], *spec.shape[1:])
        else:
            shape = spec.shape

        spec = replace(spec, shape=shape, field_type=self)
        return spec, False

    def get_element_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        indices = np.asarray(field.indices(slice(None)))
        num_elements = mesh.elements.shape[0]
        return indices.reshape(num_elements, -1)

    def get_diagonal_dofs(self, field: Field, mesh: Mesh) -> NDArray | None:
        return None
