from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ._element import Element


class _LocalFrame(NamedTuple):
    t1: Array  # first tangent vector
    t2: Array  # second tangent vector
    n: Array  # normal vector
    P: Array  # projection matrix to local frame


class Shell4(Element):
    """A 4-node bilinear shell element."""

    # nodal coords: (nnodes, 3)
    # nodal values: (nnodes, nvalues)

    quad_points = jnp.array(
        [
            [-1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
            [1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
            [1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)],
            [-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)],
        ]
    )
    quad_weights = jnp.array([1.0, 1.0, 1.0, 1.0])

    def shape_function(self, xi: Array) -> Array:
        """Returns the shape functions evaluated at the local coordinates (xi, eta)."""
        r, s = xi
        return jnp.array(
            [
                0.25 * (1.0 - r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 + s),
                0.25 * (1.0 - r) * (1.0 + s),
            ]
        )

    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the derivative of the shape functions with respect to the local coordinates (xi, eta)."""
        r, s = xi
        return jnp.array(
            [
                [-0.25 * (1.0 - s), -0.25 * (1.0 - r)],
                [0.25 * (1.0 - s), -0.25 * (1.0 + r)],
                [0.25 * (1.0 + s), 0.25 * (1.0 + r)],
                [-0.25 * (1.0 + s), 0.25 * (1.0 - r)],
            ]
        ).T

    def get_local_frame(self, J: Array) -> _LocalFrame:
        """Returns t1, t2, n, and the projection matrix P for the local frame defined by the Jacobian J."""
        t1 = J[:, 0] / jnp.linalg.norm(J[:, 0])
        t2_tmp = J[:, 2] - jnp.dot(J[:, 2], t1) * t1
        t2 = t2_tmp / jnp.linalg.norm(t2_tmp)
        n = jnp.cross(t1, t2)
        P = jnp.vstack((t1, t2))

        return _LocalFrame(t1, t2, n, P)

    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        """Returns the Jacobian matrix and its determinant at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).

        Returns:
            J: Jacobian matrix, shape (3, 2); columns: g1 = dx/dxi, g2 = dx/deta.
            J_area: Determinant of the Jacobian matrix (area scaling factor).
        """
        dNdr = self.shape_function_derivative(xi)
        J = nodal_coords.T @ dNdr.T  # (3, nnodes) @ (nodes, 2) -> (3, 2)
        G = J.T @ J  # (2, 2) first fundamental form
        J_area = jnp.sqrt(jnp.linalg.det(G))  # area scaling factor
        return J, J_area

    def dNdX(self, xi: Array, nodal_coords: Array) -> Array:
        """Returns the derivatives of the shape functions with respect to the global
        coordinates at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
        """
        dNdr = self.shape_function_derivative(xi)
        J, _ = self.get_jacobian(xi, nodal_coords)

        # (J^+)^T = J (J^T J)^-1 # pseudoinverse of J, maps parametric grads to 3D surface
        # grads
        GTinv = jnp.linalg.inv(J.T @ J)  # (2, 2)
        JTplus_T = J @ GTinv  # (3, 2)

        # all shape function 3d grads stacked: (3, nnodes)
        dNdX = JTplus_T @ dNdr
        return dNdX

    def directional_derivative(self, xi: Array, nodal_coords: Array) -> Array:
        """Returns the directional derivatives of the shape functions in the local frame.

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
        """
        P = self.get_local_frame(self.get_jacobian(xi, nodal_coords)[0]).P
        dNdX = self.dNdX(xi, nodal_coords)
        return P @ dNdX

    def gradient(self, xi: Array, nodal_values: Array, nodal_coords: Array) -> Array:
        """Returns the gradient of the nodal values at the given local coordinates (xi, eta).

        Surface gradient ∇_s of an interpolated field.

        Args:
            xi: Local coordinates (r, s).
            nodal_values: Nodal values of the element, shape (4, nvalues).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).

        Returns:
            grad: Gradient of the nodal values, shape (3, nvalues); rows: d/dx, d/dy, d/dz.
        """
        dNdX = self.dNdX(xi, nodal_coords)
        # apply to nodal values: (3, nvalues)
        return dNdX @ nodal_values
