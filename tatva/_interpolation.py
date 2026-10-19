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

"""Host-side point location and fixed interpolation stencils.

Only the finished stencil is needed for runtime JAX evaluation. Bounding boxes
are conservative for the linear simplex and multilinear tensor-product elements
supported here; higher-order geometry needs a different bounding strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from tatva.element import Element, Hexahedron8, Quad4, Tetrahedron4, Tri3

_BATCH_SIZE = 1024
_ELEMENT_SHAPES = {
    Tri3: (3, 2),
    Quad4: (4, 2),
    Tetrahedron4: (4, 3),
    Hexahedron8: (8, 3),
}


@dataclass
class _AABBTree:
    lower: np.ndarray
    upper: np.ndarray
    indices: np.ndarray | None = None
    left: _AABBTree | None = None
    right: _AABBTree | None = None

    @classmethod
    def build(
        cls, lower: np.ndarray, upper: np.ndarray, indices: np.ndarray | None = None
    ) -> _AABBTree:
        if indices is None:
            indices = np.arange(len(lower))
        node = cls(lower[indices].min(axis=0), upper[indices].max(axis=0))
        if len(indices) <= 16:
            node.indices = indices
        else:
            centers = lower[indices] / 2 + upper[indices] / 2
            axis = np.argmax(np.ptp(centers, axis=0))
            order = np.argsort(centers[:, axis], kind="stable")
            middle = len(indices) // 2
            node.left = cls.build(lower, upper, indices[order[:middle]])
            node.right = cls.build(lower, upper, indices[order[middle:]])
        return node

    def candidates(
        self, point: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> Iterator[int]:
        if np.any(point < self.lower) or np.any(point > self.upper):
            return
        if self.indices is not None:
            ids = self.indices
            inside = np.all((point >= lower[ids]) & (point <= upper[ids]), axis=1)
            yield from ids[inside]
        else:
            assert self.left is not None and self.right is not None
            yield from self.left.candidates(point, lower, upper)
            yield from self.right.candidates(point, lower, upper)


@partial(jax.jit, static_argnames=("element",))
def _candidate_weights(
    element: Element,
    points: jax.Array,
    coords: jax.Array,
    physical_tolerance: jax.Array,
    reference_tolerance: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Inverse-map a fixed-size batch without compiling for each candidate count."""
    simplex = type(element) in (Tri3, Tetrahedron4)

    def locate(point, nodes, tolerance):
        # Center coordinates to avoid cancellation in the physical map and its
        # derivative when the entire mesh is far from the origin.
        target = point - nodes[0]
        local_nodes = nodes - nodes[0]

        def residual(xi):
            return element.shape_function(xi) @ local_nodes - target

        if simplex:
            xi = jnp.linalg.solve(local_nodes[1:].T, target)
            inside = jnp.all(xi >= -reference_tolerance) & (
                jnp.sum(xi) <= 1 + reference_tolerance
            )
        else:
            xi0 = jnp.zeros(point.shape, dtype=nodes.dtype)

            def condition(state):
                iteration, xi = state
                error = jnp.linalg.norm(residual(xi))
                return (iteration < 30) & jnp.isfinite(error) & (error > tolerance)

            def step(state):
                iteration, xi = state
                rhs = residual(xi)
                jacobian = local_nodes.T @ element.shape_function_derivative(xi).T
                delta = jnp.linalg.solve(jacobian, rhs)
                initial_error = jnp.linalg.norm(rhs)

                def backtrack(_, trial):
                    alpha, candidate = trial
                    error = jnp.linalg.norm(residual(candidate))
                    decrease = jnp.isfinite(error) & (error < initial_error)
                    alpha = jnp.where(decrease, alpha, alpha / 2)
                    return alpha, xi - alpha * delta

                _, candidate = jax.lax.fori_loop(
                    0, 8, backtrack, (jnp.asarray(1, nodes.dtype), xi - delta)
                )
                # A failed line search must not discard an already better iterate.
                improved = jnp.linalg.norm(residual(candidate)) < initial_error
                return iteration + 1, jnp.where(improved, candidate, xi)

            _, xi = jax.lax.while_loop(condition, step, (0, xi0))
            inside = jnp.all(jnp.abs(xi) <= 1 + reference_tolerance)

        # Even a zero-residual starting point must have a nonsingular map.
        jacobian = local_nodes.T @ element.shape_function_derivative(xi).T
        nonsingular = jnp.linalg.slogdet(jacobian)[0] != 0
        accepted = (
            inside
            & nonsingular
            & jnp.all(jnp.isfinite(xi))
            & (jnp.linalg.norm(residual(xi)) <= tolerance)
        )
        return accepted, element.shape_function(xi)

    return jax.vmap(locate)(points, coords, physical_tolerance)


def make_interpolator(
    element: Element, coords: jax.Array, elements: jax.Array, points: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Construct a stencil on the host, returning a geometry-free JAX callable."""
    if type(element) not in _ELEMENT_SHAPES:
        raise NotImplementedError(
            "Static interpolation supports Tri3, Quad4, Tetrahedron4, and Hexahedron8."
        )
    if any(isinstance(x, jax.core.Tracer) for x in (coords, elements, points)):
        raise ValueError(
            "Interpolation construction requires concrete mesh coordinates and points; "
            "call make_interpolate outside JAX transformations, then transform its result."
        )
    nodes_per_element, dimension = _ELEMENT_SHAPES[type(element)]  # ty: ignore[invalid-argument-type]
    coords_np, elements_np, points_np = (
        np.asarray(coords),
        np.asarray(elements),
        np.asarray(points),
    )
    if coords_np.shape[1] != dimension:
        raise NotImplementedError(
            "Static interpolation does not support embedded elements."
        )
    if elements_np.shape[1] != nodes_per_element:
        raise ValueError(
            "Mesh connectivity width does not match the interpolation element."
        )
    if points_np.ndim != 2 or points_np.shape[1] != dimension:
        raise ValueError(
            f"Interpolation points must have shape (n_points, {dimension})."
        )
    if not np.isrealobj(coords_np) or not np.isrealobj(points_np):
        raise ValueError("Mesh coordinates and interpolation points must be real.")
    if not np.all(np.isfinite(coords_np)) or not np.all(np.isfinite(points_np)):
        raise ValueError("Mesh coordinates and interpolation points must be finite.")

    dtype = jax.dtypes.canonicalize_dtype(
        np.result_type(coords_np, points_np, np.float32)
    )
    coords_np, points_np = coords_np.astype(dtype), points_np.astype(dtype)
    element_coords = coords_np[elements_np]
    lower, upper = element_coords.min(axis=1), element_coords.max(axis=1)
    diameter = np.linalg.norm(upper - lower, axis=1)
    magnitude = np.max(np.abs(element_coords), axis=(1, 2))
    eps = np.finfo(dtype).eps
    reference_tolerance = max(1e-10, 32 * eps)
    tolerance = reference_tolerance * diameter + 8 * eps * magnitude
    lower, upper = lower - tolerance[:, None], upper + tolerance[:, None]
    tree = _AABBTree.build(lower, upper)
    selected = np.full(len(points_np), len(elements_np), dtype=np.int64)
    weights = np.empty((len(points_np), nodes_per_element), dtype=dtype)

    def process(point_ids: list[int], element_ids: list[int]) -> None:
        count = len(point_ids)
        # Padding keeps the numerical kernel's shape constant, including the
        # last batch. Only real entries participate in stencil selection.
        pids = np.pad(np.asarray(point_ids), (0, _BATCH_SIZE - count), mode="edge")
        eids = np.pad(np.asarray(element_ids), (0, _BATCH_SIZE - count), mode="edge")
        valid, batch_weights = _candidate_weights(
            element,
            jnp.asarray(points_np[pids]),
            jnp.asarray(element_coords[eids]),
            jnp.asarray(tolerance[eids]),
            jnp.asarray(reference_tolerance, dtype=dtype),
        )
        valid, batch_weights = np.asarray(valid), np.asarray(batch_weights)
        for i in np.flatnonzero(valid[:count]):
            pid, eid = pids[i], eids[i]
            if eid < selected[pid]:
                selected[pid] = eid
                weights[pid] = batch_weights[i]

    # Stream tree hits into bounded numerical batches; no point-by-element
    # array or unbounded candidate list is materialized.
    point_ids: list[int] = []
    element_ids: list[int] = []
    for pid, point in enumerate(points_np):
        for eid in tree.candidates(point, lower, upper):
            point_ids.append(pid)
            element_ids.append(eid)
            if len(point_ids) == _BATCH_SIZE:
                process(point_ids, element_ids)
                point_ids, element_ids = [], []
    if point_ids:
        process(point_ids, element_ids)

    missing = np.flatnonzero(selected == len(elements_np))
    if len(missing):
        raise RuntimeError(
            f"Interpolation points {missing.tolist()} are outside the mesh or their "
            "element inverse mapping failed (singular or unconverged)."
        )
    node_indices = jnp.asarray(elements_np[selected])
    fixed_weights = jnp.asarray(weights)
    n_nodes = len(coords_np)

    def interpolate(values: jax.Array) -> jax.Array:
        values = jnp.asarray(values)
        if values.ndim == 0 or values.shape[0] != n_nodes:
            raise ValueError(f"Nodal values must have shape ({n_nodes}, ...).")
        return jnp.einsum("pa,pa...->p...", fixed_weights, values[node_indices])

    return interpolate
