from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Element(eqx.Module):
    """Base Module for all finite elements, compatible with JAX."""

    @abstractmethod
    def get_quadrature(self) -> Tuple[Array, Array]:
        """Returns the element's quadrature points and weights."""
        raise NotImplementedError

    @abstractmethod
    def get_shape_functions(self, xi: Array) -> Tuple[Array, Array]:
        """Returns the shape functions and their derivatives at a point."""
        raise NotImplementedError

    def get_jacobian(self, xi: Array, nodes: Array) -> Tuple[Array, Array]:
        _N, dNdr = self.get_shape_functions(xi)
        J = dNdr @ nodes
        return J, jnp.linalg.det(J)

    def interpolate(self, xi: Array, nodal_values: Array) -> Array:
        N, _ = self.get_shape_functions(xi)
        return N @ nodal_values

    def gradient(self, xi: Array, nodal_values: Array, nodes: Array) -> Array:
        _N, dNdr = self.get_shape_functions(xi)
        J, _detJ = self.get_jacobian(xi, nodes)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodes: Array
    ) -> Tuple[Array, Array, Array]:
        N, dNdr = self.get_shape_functions(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Line2(Element):
    """A 2-node linear interval element."""

    def get_quadrature(self) -> Tuple[Array, Array]:
        qp = jnp.array([[0.0]])
        w = jnp.array([2.0])
        return qp, w

    def get_shape_functions(self, xi: Array) -> Tuple[Array, Array]:
        xi_val = xi[0]
        N = jnp.array([0.5 * (1.0 - xi_val), 0.5 * (1.0 + xi_val)])
        dNdxi = jnp.array([-0.5, 0.5])
        return N, dNdxi

    def get_jacobian(self, xi: Array, nodes: Array) -> tuple[Array, Array]:
        _N, dNdr = self.get_shape_functions(xi)
        J = dNdr @ nodes
        t = jnp.asarray([J[0], J[1]]) / jnp.linalg.norm(J)
        return jnp.dot(J, t), jnp.dot(J, t)

    def gradient(self, xi: Array, nodal_values: Array, nodes: Array) -> Array:
        _N, dNdr = self.get_shape_functions(xi)
        J, _ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: Array, nodal_values: Array, nodes: Array
    ) -> Tuple[Array, Array, Array]:
        N, dNdr = self.get_shape_functions(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Tri3(Element):
    """A 3-node linear triangular element."""

    def get_quadrature(self) -> Tuple[Array, Array]:
        qp = jnp.array([[1 / 3, 1 / 3]])
        w = jnp.array([0.5])
        return qp, w

    def get_shape_functions(self, xi: Array) -> Tuple[Array, Array]:
        xi1, xi2 = xi
        N = jnp.array([1.0 - xi1 - xi2, xi1, xi2])
        dNdxi = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
        return N, dNdxi


class Quad4(Element):
    """A 4-node bilinear quadrilateral element."""

    def get_quadrature(self) -> Tuple[Array, Array]:
        xi_vals = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
        w_vals = jnp.array([1.0, 1.0])
        quad_points = jnp.stack(jnp.meshgrid(xi_vals, xi_vals), axis=-1).reshape(-1, 2)
        weights = jnp.kron(w_vals, w_vals)
        return quad_points, weights

    def get_shape_functions(self, xi: Array) -> Tuple[Array, Array]:
        r, s = xi
        N = 0.25 * jnp.array(
            [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)]
        )
        dNdr = (
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
        return N, dNdr


# Dictionary mapping keywords to element classes
_element_map = {
    "line2": Line2,
    "tri3": Tri3,
    "quad4": Quad4,
}


def get_element(name: str) -> Element:
    """
    Factory to get an element object by its keyword.

    Args:
        name: The keyword for the element ('line2', 'tri3', 'quad4').

    Returns:
        An instance of the corresponding Equinox element module.
    """
    element_class = _element_map.get(name.lower())
    if element_class is None:
        raise ValueError(
            f"Unknown element type: '{name}'. Available: {list(_element_map.keys())}"
        )
    return element_class()

