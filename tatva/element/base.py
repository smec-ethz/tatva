# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
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


from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class Element(eqx.Module):
    """Base Module for all finite elements, compatible with JAX."""

    quad_points: AbstractClassVar[Array]
    quad_weights: AbstractClassVar[Array]

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

    def interpolate(self, xi: Array, nodal_values: Array) -> Array:
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

    quad_points = jnp.array([[0.0]])
    quad_weights = jnp.array([2.0])

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


class Tri3(Element):
    """A 3-node linear triangular element."""

    quad_points = jnp.array([[1.0 / 3, 1.0 / 3]])
    quad_weights = jnp.array([1.0 / 2])

    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions evaluated at the local coordinates (xi, eta)."""
        xi1, xi2 = xi
        return jnp.array([1.0 - xi1 - xi2, xi1, xi2])

    def shape_function_derivative(self, *_args, **_kwargs) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta)."""
        # shape (n_q, 2, 3)
        return jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T


def _get_quad4_quadrature() -> tuple[Array, Array]:
    xi_vals = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
    w_vals = jnp.array([1.0, 1.0])
    quad_points = jnp.stack(jnp.meshgrid(xi_vals, xi_vals), axis=-1).reshape(-1, 2)
    weights = jnp.kron(w_vals, w_vals)
    return quad_points, weights


_quad4_qp, _quad4_w = _get_quad4_quadrature()


class Quad4(Element):
    """A 4-node bilinear quadrilateral element."""

    quad_points = _quad4_qp
    quad_weights = _quad4_w

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


class Tetrahedron4(Element):
    """A 4-node linear tetrahedral element."""

    quad_points = jnp.array([[1.0 / 4, 1.0 / 4, 1.0 / 4]])
    quad_weights = jnp.array([1.0 / 6])

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


def _get_hex8_quadrature() -> tuple[Array, Array]:
    xi_vals = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
    quad_points = jnp.stack(jnp.meshgrid(xi_vals, xi_vals, xi_vals), axis=-1).reshape(
        -1, 3
    )
    weights = jnp.full(quad_points.shape[0], fill_value=1.0)
    return quad_points, weights


_hex8_qp, _hex8_w = _get_hex8_quadrature()


class Hexahedron8(Element):
    """A 8-node linear hexahedral element."""

    quad_points = _hex8_qp
    quad_weights = _hex8_w

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
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta, zeta)."""
        # shape (n_q, 3, 8)
        xi, eta, zeta = xi
        return (1 / 8) * jnp.array(
            [
                [
                    -(1 - eta) * (1 - zeta),
                    -(1 - xi) * (1 - zeta),
                    -(1 - xi) * (1 - eta),
                ],
                [(1 - eta) * (1 - zeta), -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)],
                [(1 + eta) * (1 - zeta), (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)],
                [-(1 + eta) * (1 - zeta), (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)],
                [-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)],
                [(1 - eta) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - eta)],
                [(1 + eta) * (1 + zeta), (1 + xi) * (1 + zeta), (1 + xi) * (1 + eta)],
                [-(1 + eta) * (1 + zeta), (1 - xi) * (1 + zeta), (1 - xi) * (1 + eta)],
            ]
        )

