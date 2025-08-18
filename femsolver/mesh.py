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
        return cls.rectangle((0.0, 1.0), (0.0, 1.0), n_x, n_y)

    @classmethod
    def rectangle(
        cls, x: tuple[float, float], y: tuple[float, float], n_x: int, n_y: int
    ) -> Mesh:
        """Generate a rectangular mesh with specified x and y ranges and number of nodes."""
        x_vals = jnp.linspace(x[0], x[1], n_x + 1)
        y_vals = jnp.linspace(y[0], y[1], n_y + 1)
        xv, yv = jnp.meshgrid(x_vals, y_vals, indexing="ij")
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


@jax.jit
def find_containing_polygons(
    points: jax.Array,
    polygons: jax.Array,
) -> jax.Array:
    """
    Finds the index of the containing polygon for each point.

    This function uses a vectorized Ray Casting algorithm and is JIT-compiled
    for maximum performance. It assumes polygons are non-overlapping.

    Args:
        points (Array): An array of points to test, shape (num_points, 2).
        polygons (Array): A 3D array of polygons, where each polygon is a list of
            vertices. Shape (num_polygons, num_vertices, 2).

    Returns:
        Array: An array of shape (num_points,) where each element is the index of the
            polygon containing the corresponding point. Returns -1 if a point is not in
            any polygon.
    """

    # --- Core function for a single point and a single polygon ---
    def is_inside(point, vertices):
        px, py = point

        # Get all edges of the polygon by pairing vertices with the next one
        p1s = vertices
        p2s = jnp.roll(vertices, -1, axis=0)  # Get p_{i+1} for each p_i

        # Conditions for a valid intersection of the horizontal ray from the point
        # 1. The point's y-coord must be between the edge's y-endpoints
        y_cond = (p1s[:, 1] <= py) & (p2s[:, 1] > py) | (p2s[:, 1] <= py) & (
            p1s[:, 1] > py
        )

        # 2. The point's x-coord must be to the left of the edge's x-intersection
        # Calculate the x-intersection of the ray with the edge
        x_intersect = (p2s[:, 0] - p1s[:, 0]) * (py - p1s[:, 1]) / (
            p2s[:, 1] - p1s[:, 1]
        ) + p1s[:, 0]
        x_cond = px < x_intersect

        # An intersection occurs if both conditions are met.
        intersections = jnp.sum(y_cond & x_cond)

        # The point is inside if the number of intersections is odd.
        return intersections % 2 == 1

    # --- Vectorize and apply the function ---
    # Create a boolean matrix: matrix[i, j] is True if point i is in polygon j
    # Vmap over points (axis 0) and polygons (axis 0)
    # in_axes=(0, None) -> maps over points, polygon is fixed
    # in_axes=(None, 0) -> maps over polygons, point is fixed
    # We vmap the second case over all points
    is_inside_matrix = jax.vmap(
        lambda p: jax.vmap(lambda poly: is_inside(p, poly))(polygons), in_axes=(0)
    )(points)

    # Find the index of the first 'True' value for each point (row).
    # This gives the index of the containing polygon.
    # We add a 'False' column to handle points outside all polygons.
    # jnp.argmax will then return the index of this last column.

    padded_matrix = jnp.pad(
        is_inside_matrix, ((0, 0), (0, 1)), "constant", constant_values=False
    )

    # If the point is not in any polygon, the all values in the row will be False.
    # We use this to map the index to -1.
    not_in_any_polygon = jnp.all(~is_inside_matrix, axis=1)

    indices = jnp.argmax(padded_matrix, axis=1)

    # we map the index to -1 if the point is not in any polygon.
    return jnp.where(not_in_any_polygon == True, -1, indices)
