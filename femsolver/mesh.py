from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Mesh(NamedTuple):
    """A class used to represent a Mesh for finite element method (FEM) simulations.

    Attributes:
        nodes: The coordinates of the mesh nodes.
        elements: The connectivity of the mesh elements.
    """

    coords: jax.Array  # Shape (n_nodes, n_dim)
    elements: jax.Array  # Shape (n_elements, n_nodes_per_element)

    @classmethod
    def unit_square(cls, n_x: int, n_y: int) -> Mesh:
        """Generate a unit square mesh with n_x and n_y nodes in the x and y directions."""

        x = jnp.linspace(0, 1, n_x + 1)
        y = jnp.linspace(0, 1, n_y + 1)
        xv, yv = jnp.meshgrid(x, y, indexing="ij")
        coords = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)

        def node_id(i, j):
            return i * (n_y + 1) + j

        elements = []
        for i in range(n_x):
            for j in range(n_y):
                n0 = node_id(i, j)
                n1 = node_id(i + 1, j)
                n2 = node_id(i, j + 1)
                n3 = node_id(i + 1, j + 1)
                elements.append([n0, n1, n3])
                elements.append([n0, n3, n2])

        return cls(coords, jnp.array(elements, dtype=jnp.int32))
