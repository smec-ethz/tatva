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

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from tatva.element import Element
from tatva.mesh import Mesh
from tatva.topology import vertex_permutation

ElementT = TypeVar("ElementT", bound=Element)

__all__ = ["FunctionSpace"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FunctionSpace(Generic[ElementT]):
    """Pairs a mesh with an element and owns the connectivity between them.

    A mesh records its cells in tatva's own vertex order; an element expects them in
    whatever order its basis was built for. `FunctionSpace` reconciles the two so that
    neither has to know about the other, and is the object that will grow the extra dofs
    for higher-degree spaces.

    A space covers a single cell type. Meshes mixing cell types are handled by splitting
    them and building one space — and one `Operator` — per type.

    Attributes:
        mesh: The mesh, with connectivity in tatva's vertex order.
        element: The element defining the basis. Static under jax transformations.
        dofmap: Global dof index of each element-local dof (shape: (n_elements, n_dofs)).
        geometry_dofmap: Global node index of each element-local geometry node (shape:
            (n_elements, n_geometry_nodes)). For a degree-1 space this is the same table as
            `dofmap`; the two diverge once the field carries dofs the geometry does not.
    """

    mesh: Mesh
    element: ElementT = field(metadata=dict(static=True))
    dofmap: Array | None = None
    geometry_dofmap: Array | None = None

    def __post_init__(self) -> None:
        if self.dofmap is not None and self.geometry_dofmap is not None:
            return

        elements = self.mesh.elements
        n_vertices = elements.shape[1]

        # permutation applied only to basix-backed elements
        # in future, tatva based elements will be replaced
        cell = getattr(self.element, "cell", None)
        if cell is None:
            perm = np.arange(
                n_vertices
            )  # to keep same order as defined in tatva.element
        else:
            perm = vertex_permutation(cell)

        if len(perm) != n_vertices:
            raise ValueError(
                f"element on cell {cell!r} has {len(perm)} vertices but the mesh "
                f"connectivity is {n_vertices} wide"
            )

        n_dofs = getattr(self.element, "n_dofs", n_vertices)
        if n_dofs != n_vertices:
            raise NotImplementedError(
                f"{self.element!r} has {n_dofs} dofs but the mesh supplies only "
                f"{n_vertices} per cell. Dofs on edges, faces and cell interiors are not "
                f"generated yet, so only degree-1 spaces are supported."
            )

        table = elements[:, perm]
        object.__setattr__(self, "dofmap", table)
        object.__setattr__(self, "geometry_dofmap", table)

    @property
    def n_global_dofs(self) -> int:
        """Number of dofs in the space.

        Equal to the node count while every dof sits on a vertex.
        """
        return self.mesh.coords.shape[0]

    @property
    def n_dofs_per_element(self) -> int:
        """Number of dofs each element gathers."""
        assert self.dofmap is not None
        return self.dofmap.shape[1]
