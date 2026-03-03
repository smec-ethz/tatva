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


from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array


class Element(ABC):
    """Abstract base class for all finite elements in tatva. Subclasses must implement
    methods to compute shape functions, their derivatives, and the Jacobian.

    Elements are to be used as static objects only in jax transformations! Considered
    immutable. Do not mutate.

    Currently, equality and hash are based on object identity, meaning two elements are
    only equal if they are the same object in memory. Even if two elements have the same
    type and quad rule.
    """

    quad_points: Array
    quad_weights: Array

    def __init__(
        self, quad_points: Array | None = None, quad_weights: Array | None = None
    ):
        """Initializes the element, optionally with custom quadrature points and weights.

        Args:
            quad_points: An array of shape (n_q, n_dim) containing the quadrature points
                in local coordinates.
            quad_weights: An array of shape (n_q,) containing the quadrature weights.
        """
        if (quad_points is not None) and (quad_weights is not None):
            self.quad_points = quad_points
            self.quad_weights = quad_weights
        else:
            self.quad_points, self.quad_weights = self._default_quadrature()

    @abstractmethod
    def _default_quadrature(self) -> tuple[Array, Array]:
        """Returns the default quadrature points and weights for this element."""
        raise NotImplementedError

    @abstractmethod
    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions and their derivatives at a point."""
        raise NotImplementedError

    @abstractmethod
    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates."""
        raise NotImplementedError

    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        dNdr = self.shape_function_derivative(xi)
        J = dNdr @ nodal_coords
        return J, jnp.linalg.det(J)

    def interpolate(self, xi: Array, nodal_values: Array, nodal_coords: Array) -> Array:
        N = self.shape_function(xi)
        return N @ nodal_values

    def gradient(self, xi: Array, nodal_values: Array, nodal_coords: Array) -> Array:
        dNdr = self.shape_function_derivative(xi)
        J, _ = self.get_jacobian(xi, nodal_coords)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array]:
        """Returns a tuple containing the interpolated value, gradient, and determinant of the Jacobian.

        Args:
            xi: Local coordinates (shape: (n_dim,)).
            nodal_values: Values at the nodes of the element (shape: (n_nodes, n_values)).
            nodal_coords: Coordinates of the nodes of the element (shape: (n_nodes, n_dim)).

        Returns:
            A tuple containing:
                - Interpolated value at the local coordinates (shape: (n_values,)).
                - Gradient of the nodal values at the local coordinates (shape: (n_dim, n_values)).
                - Determinant of the Jacobian (scalar).
        """
        N = self.shape_function(xi)
        dNdr = self.shape_function_derivative(xi)
        J, detJ = self.get_jacobian(xi, nodal_coords)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Line2(Element):
    """A 2-node linear interval element."""

    def _default_quadrature(self):
        quad_points = jnp.array([[0.0]])
        quad_weights = jnp.array([2.0])
        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        xi_val = xi[0]
        return jnp.array([0.5 * (1.0 - xi_val), 0.5 * (1.0 + xi_val)])

    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates."""
        return jnp.array([-0.5, 0.5])

    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        _N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J = dNdr @ nodal_coords
        t = jnp.asarray([J[0], J[1]]) / jnp.linalg.norm(J)
        return jnp.dot(J, t), jnp.dot(J, t)

    def gradient(self, xi: Array, nodal_values: Array, nodal_coords: Array) -> Array:
        _N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J, _ = self.get_jacobian(xi, nodal_coords)
        dNdX = dNdr / J
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array]:
        N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J, detJ = self.get_jacobian(xi, nodal_coords)
        dNdX = dNdr / J
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Line3(Element):
    """3-node quadratic line element."""

    def _default_quadrature(self):
        quad_points = jnp.array([[-jnp.sqrt(3.0 / 5.0)], [0.0], [jnp.sqrt(3.0 / 5.0)]])
        quad_weights = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        r = xi[0]

        N1 = 0.5 * r * (r - 1.0)
        N2 = 0.5 * r * (r + 1.0)
        N3 = 1.0 - r**2
        return jnp.array([N1, N2, N3])

    def shape_function_derivative(self, xi: Array) -> Array:
        r = xi[0]
        dN1 = r - 0.5
        dN2 = r + 0.5
        dN3 = -2.0 * r
        return jnp.array([dN1, dN2, dN3])

    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        dNdr = self.shape_function_derivative(xi)
        Jvec = dNdr @ nodal_coords

        t = Jvec / jnp.linalg.norm(Jvec)
        J = jnp.dot(Jvec, t)
        return J, J

    def gradient(self, xi: Array, nodal_values: Array, nodal_coords: Array) -> Array:
        dNdr = self.shape_function_derivative(xi)
        J, _ = self.get_jacobian(xi, nodal_coords)
        dNdS = dNdr / J
        return dNdS @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array]:

        N = self.shape_function(xi)
        dNdr = self.shape_function_derivative(xi)
        J, detJ = self.get_jacobian(xi, nodal_coords)
        dNdS = dNdr / J
        return N @ nodal_values, dNdS @ nodal_values, detJ


class Tri3(Element):
    """A 3-node linear triangular element."""

    def _default_quadrature(self) -> tuple[Array, Array]:
        quad_points = jnp.array([[1.0 / 3, 1.0 / 3]])
        quad_weights = jnp.array([1.0 / 2])
        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions evaluated at the local coordinates (xi, eta)."""
        xi1, xi2 = xi
        return jnp.array([1.0 - xi1 - xi2, xi1, xi2])

    def shape_function_derivative(self, *_args, **_kwargs) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta)."""
        # shape (n_q, 2, 3)
        return jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T


class Quad4(Element):
    """A 4-node bilinear quadrilateral element."""

    def _default_quadrature(self) -> tuple[Array, Array]:
        xi_vals = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
        w_vals = jnp.array([1.0, 1.0])
        quad_points = jnp.stack(jnp.meshgrid(xi_vals, xi_vals), axis=-1).reshape(-1, 2)
        weights = jnp.kron(w_vals, w_vals)

        return quad_points, weights

    def shape_function(self, xi: Array) -> Array:
        r, s = xi
        return 0.25 * jnp.array(
            [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)]
        )

    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta)."""
        r, s = xi
        return (
            0.25
            * jnp.array(
                [
                    [-(1 - s), -(1 - r)],
                    [(1 - s), -(1 + r)],
                    [(1 + s), (1 + r)],
                    [-(1 + s), (1 - r)],
                ]
            ).T
        )


class Quad8(Element):
    """An 8-node biquadratic quadrilateral element."""

    def _default_quadrature(self) -> tuple[Array, Array]:
        xi_1d = jnp.array([-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)])
        w_1d = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        rr, ss = jnp.meshgrid(xi_1d, xi_1d, indexing="xy")
        quad_points = jnp.stack([rr.ravel(), ss.ravel()], axis=-1).reshape(
            -1, 2
        )  # (9, 2)
        quad_weights = jnp.kron(w_1d, w_1d)  # (9,)
        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        r, s = xi

        N1 = 0.25 * (1 - r) * (1 - s) * (-r - s - 1)
        N2 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
        N3 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
        N4 = 0.25 * (1 - r) * (1 + s) * (-r + s - 1)
        N5 = 0.5 * (1 - r**2) * (1 - s)
        N6 = 0.5 * (1 + r) * (1 - s**2)
        N7 = 0.5 * (1 - r**2) * (1 + s)
        N8 = 0.5 * (1 - r) * (1 - s**2)

        return jnp.array([N1, N2, N3, N4, N5, N6, N7, N8])

    def shape_function_derivative(self, xi: Array) -> Array:
        """dN/d(r,s) as array of shape (2, 8)."""
        r, s = xi

        # dN/dr
        dN1_dr = 0.25 * (-2 * r - s) * (s - 1)
        dN2_dr = 0.25 * (-2 * r + s) * (s - 1)
        dN3_dr = 0.25 * (2 * r + s) * (s + 1)
        dN4_dr = 0.25 * (2 * r - s) * (s + 1)
        dN5_dr = r * (s - 1)
        dN6_dr = 0.5 - 0.5 * s**2
        dN7_dr = -r * (s + 1)
        dN8_dr = 0.5 * s**2 - 0.5

        dN_dr = jnp.array(
            [dN1_dr, dN2_dr, dN3_dr, dN4_dr, dN5_dr, dN6_dr, dN7_dr, dN8_dr]
        )

        # dN/ds
        dN1_ds = 0.25 * (-r - 2 * s) * (r - 1)
        dN2_ds = 0.25 * (-r + 2 * s) * (r + 1)
        dN3_ds = 0.25 * (r + 1) * (r + 2 * s)
        dN4_ds = 0.25 * (r - 1) * (r - 2 * s)
        dN5_ds = 0.5 * r**2 - 0.5
        dN6_ds = -s * (r + 1)
        dN7_ds = 0.5 - 0.5 * r**2
        dN8_ds = s * (r - 1)

        dN_ds = jnp.array(
            [dN1_ds, dN2_ds, dN3_ds, dN4_ds, dN5_ds, dN6_ds, dN7_ds, dN8_ds]
        )

        return jnp.vstack((dN_dr, dN_ds))


class Tetrahedron4(Element):
    """A 4-node linear tetrahedral element."""

    def _default_quadrature(self) -> tuple[Array, Array]:
        quad_points = jnp.array([[1.0 / 4, 1.0 / 4, 1.0 / 4]])
        quad_weights = jnp.array([1.0 / 6])
        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions evaluated at the local coordinates (xi, eta, zeta)."""
        xi, eta, zeta = xi
        return jnp.array([1.0 - xi - eta - zeta, xi, eta, zeta])

    def shape_function_derivative(self, *_args, **_kwargs) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta, zeta)."""
        # shape (n_q, 3, 4)
        return jnp.array(
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ).T


class Hexahedron8(Element):
    """A 8-node linear hexahedral element."""

    def _default_quadrature(self) -> tuple[Array, Array]:
        a = 1 / jnp.sqrt(3)

        # 2x2x2 Gauss Quadrature Rule
        quad_points = jnp.array(
            [
                [-a, -a, -a],
                [a, -a, -a],
                [a, a, -a],
                [-a, a, -a],
                [-a, -a, a],
                [a, -a, a],
                [a, a, a],
                [-a, a, a],
            ]
        )

        # Weights are all 1.0 for this rule (since interval is [-1, 1])
        quad_weights = jnp.ones(8)

        return quad_points, quad_weights

    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions evaluated at the local coordinates (xi, eta, zeta)."""
        xi, eta, zeta = xi
        return (1 / 8) * jnp.array(
            [
                (1 - xi) * (1 - eta) * (1 - zeta),
                (1 + xi) * (1 - eta) * (1 - zeta),
                (1 + xi) * (1 + eta) * (1 - zeta),
                (1 - xi) * (1 + eta) * (1 - zeta),
                (1 - xi) * (1 - eta) * (1 + zeta),
                (1 + xi) * (1 - eta) * (1 + zeta),
                (1 + xi) * (1 + eta) * (1 + zeta),
                (1 - xi) * (1 + eta) * (1 + zeta),
            ]
        )

    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the derivative of the shape functions."""
        # shape (3, 8) -> (dim, n_nodes)
        xi, eta, zeta = xi
        return (1 / 8) * jnp.array(
            [
                [
                    -(1 - eta) * (1 - zeta),
                    (1 - eta) * (1 - zeta),
                    (1 + eta) * (1 - zeta),
                    -(1 + eta) * (1 - zeta),
                    -(1 - eta) * (1 + zeta),
                    (1 - eta) * (1 + zeta),
                    (1 + eta) * (1 + zeta),
                    -(1 + eta) * (1 + zeta),
                ],
                [
                    -(1 - xi) * (1 - zeta),
                    -(1 + xi) * (1 - zeta),
                    (1 + xi) * (1 - zeta),
                    (1 - xi) * (1 - zeta),
                    -(1 - xi) * (1 + zeta),
                    -(1 + xi) * (1 + zeta),
                    (1 + xi) * (1 + zeta),
                    (1 - xi) * (1 + zeta),
                ],
                [
                    -(1 - xi) * (1 - eta),
                    -(1 + xi) * (1 - eta),
                    -(1 + xi) * (1 + eta),
                    -(1 - xi) * (1 + eta),
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ],
            ]
        )
