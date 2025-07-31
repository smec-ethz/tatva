from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class Element(eqx.Module):
    """Base Module for all finite elements, compatible with JAX."""

    quad_points: AbstractClassVar[Array]
    quad_weights: AbstractClassVar[Array]

    @abstractmethod
    def shape_function(self, xi: Array) -> tuple[Array, Array]:
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

    def get_jacobian(self, xi: Array, nodes: Array) -> tuple[Array, Array]:
        _N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J = dNdr @ nodes
        t = jnp.asarray([J[0], J[1]]) / jnp.linalg.norm(J)
        return jnp.dot(J, t), jnp.dot(J, t)

    def gradient(self, xi: Array, nodal_values: Array, nodes: Array) -> Array:
        _N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J, _ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodes: Array
    ) -> tuple[Array, Array, Array]:
        N, dNdr = self.shape_function(xi), self.shape_function_derivative(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Tri3(Element):
    """A 3-node linear triangular element."""

    quad_points = jnp.array([[1 / 3, 1 / 3]])
    quad_weights = jnp.array([0.5])

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
