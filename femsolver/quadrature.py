import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

# file: elements_eqx.py

import equinox as eqx
from typing import Tuple


# --- Base Module (defines the common interface) ---
class Element(eqx.Module):
    """Base Module for all finite elements, compatible with JAX."""

    # In Equinox, we don't use @abstractmethod. We can either leave this
    # class empty or raise NotImplementedError to enforce the interface.
    def get_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the element's quadrature points and weights."""
        raise NotImplementedError

    def get_shape_functions(self, xi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the shape functions and their derivatives at a point."""
        raise NotImplementedError

    def get_jacobian(self, xi: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J = dNdr @ nodes
        return J, jnp.linalg.det(J)

    def interpolate(self, xi: jnp.ndarray, nodal_values: jnp.ndarray) -> jnp.ndarray:
        N, _ = self.get_shape_functions(xi)
        return N @ nodal_values
    
    def gradient(
        self, xi: jnp.ndarray, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return dNdX @ nodal_values
    
    def get_local_values(
        self, xi: jnp.ndarray, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = jnp.linalg.inv(J) @ dNdr
        return N @ nodal_values, dNdX @ nodal_values, detJ




class Line2(Element):
    """A 2-node linear interval element."""

    def get_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        qp = jnp.array([[0.0]])
        w = jnp.array([2.0])
        return qp, w

    def get_shape_functions(self, xi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xi_val = xi[0]
        N = jnp.array([0.5 * (1.0 - xi_val), 0.5 * (1.0 + xi_val)])
        dNdxi = jnp.array([-0.5, 0.5])
        return N, dNdxi

    def get_jacobian(self, xi: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J = dNdr @ nodes
        t = jnp.asarray([J[0], J[1]]) / jnp.linalg.norm(J)
        return jnp.dot(J, t), jnp.dot(J, t)


    def gradient(
        self, xi: jnp.ndarray, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J, _ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return dNdX @ nodal_values

    def get_local_values(
        self, xi: jnp.ndarray, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        N, dNdr = self.get_shape_functions(xi)
        J, detJ = self.get_jacobian(xi, nodes)
        dNdX = dNdr / J
        return N @ nodal_values, dNdX @ nodal_values, detJ


class Tri3(Element):
    """A 3-node linear triangular element."""

    def get_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        qp = jnp.array([[1 / 3, 1 / 3]])
        w = jnp.array([0.5])
        return qp, w

    def get_shape_functions(self, xi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xi1, xi2 = xi
        N = jnp.array([1.0 - xi1 - xi2, xi1, xi2])
        dNdxi = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
        return N, dNdxi


class Quad4(Element):
    """A 4-node bilinear quadrilateral element."""

    def get_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xi_vals = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
        w_vals = jnp.array([1.0, 1.0])
        quad_points = jnp.stack(jnp.meshgrid(xi_vals, xi_vals), axis=-1).reshape(-1, 2)
        weights = jnp.kron(w_vals, w_vals)
        return quad_points, weights

    def get_shape_functions(self, xi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


''''
# --- Shape functions and quadrature ---
def quad_quad4() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadrature points and weights for a quadrilateral element.

    Returns
    -------
    quad_points : jnp.ndarray
        The quadrature points.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    weights : jnp.ndarray
        The quadrature weights.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    """
    xi = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
    w = jnp.array([1.0, 1.0])
    quad_points = jnp.stack(jnp.meshgrid(xi, xi), axis=-1).reshape(-1, 2)
    weights = jnp.kron(w, w)
    return quad_points, weights


def shape_fn_quad4(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shape functions and derivatives for a quadrilateral element.

    Parameters
    ----------
    xi : jnp.ndarray
        The local coordinates of the quadrature points.

    Returns
    -------
    N : jnp.ndarray
        The shape functions.
        The shape of the array is (nb_nodes_per_element,).
    dNdr : jnp.ndarray
        The derivatives of the shape functions with respect to the local coordinates.
        The shape of the array is (nb_axes_in_reference_space, nb_nodes_per_element).
    """
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


def quad_tri3() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadrature points and weights for a triangular element.

    Returns
    -------
    quad_points : jnp.ndarray
        The quadrature points.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    weights : jnp.ndarray
        The quadrature weights.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    """
    qp = jnp.array([[1 / 3, 1 / 3]])
    w = jnp.array([0.5])
    return qp, w


def shape_fn_tri3(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shape functions and derivatives for a triangular element.

    Parameters
    ----------
    xi : jnp.ndarray
        The local coordinates of the quadrature points.

    Returns
    -------
    N : jnp.ndarray
        The shape functions.
        The shape of the array is (nb_nodes_per_element,).
    dNdr : jnp.ndarray
        The derivatives of the shape functions with respect to the local coordinates.
        The shape of the array is (nb_axes_in_reference_space, nb_nodes_per_element).
    """
    xi1, xi2 = xi
    xi3 = 1.0 - xi1 - xi2
    N = jnp.array([xi3, xi1, xi2])
    dNdxi = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
    return N, dNdxi


def quad_line2() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadrature point and weight for a 2-node linear interval element.

    Uses a 1-point Gauss quadrature rule, which is exact for integrating
    linear polynomials over the reference interval [-1, 1].

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - quad_points (jnp.ndarray): Quadrature point(s), shape (1, 1).
            - weights (jnp.ndarray): Quadrature weight(s), shape (1,).
    """
    # 1-point Gauss quadrature rule
    qp = jnp.array([[0.0]])
    w = jnp.array([2.0])
    return qp, w


def shape_fn_line2(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shape functions and derivatives for a 2-node linear interval element.

    The reference element is defined on the interval xi in [-1, 1].

    Parameters:
        xi (jnp.ndarray): The local coordinate of a point, shape (1,).

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - N (jnp.ndarray): Shape functions evaluated at xi, shape (2,).
            - dNdxi (jnp.ndarray): Shape function derivatives, shape (1, 2).
    """
    # The input xi is a 1-element array, e.g., jnp.array([0.0])
    xi_val = xi[0]

    # Shape functions: N1 = (1 - xi) / 2,  N2 = (1 + xi) / 2
    N = jnp.array([0.5 * (1.0 - xi_val), 0.5 * (1.0 + xi_val)])

    # Derivatives: dN1/dxi = -0.5,  dN2/dxi = 0.5
    dNdxi = jnp.array([[-0.5, 0.5]])

    return N, dNdxi


@jax.tree_util.register_pytree_node_class
class Basis:
    """
    A class used to represent a Basis for finite element method (FEM) simulations.

    Attributes
    ----------
    N : jnp.ndarray
        Shape functions evaluated at quadrature points, with shape (nb_quads, nb_nodes_per_element).
    dNdξ : jnp.ndarray
        Derivatives of shape functions with respect to the reference coordinates, with shape (nb_quads, nb_nodes_per_element).
    wts : jnp.ndarray
        Quadrature weights, with shape (nb_quads).

    Methods
    -------
    tree_flatten():
        Flattens the Basis object into a tuple of children and auxiliary data for JAX transformations.
    tree_unflatten(aux_data, children):
        Reconstructs the Basis object from flattened children and auxiliary data.
    """

    def __init__(self, nb_quads, nb_nodes_per_element):
        self.N = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.dNdξ = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.d2Ndξ2 = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.wts = jnp.zeros(nb_quads)
        self.quad_pts = jnp.zeros(nb_quads)

    def tree_flatten(self):
        return ((self.N, self.dNdξ, self.wts, self.quad_pts), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        N, dNdξ, ω, quad_pts = children
        instance = cls(N.shape[0], N.shape[1])
        instance.N = N
        instance.dNdξ = dNdξ
        instance.wts = ω
        instance.quad_pts = quad_pts
        return instance
'''