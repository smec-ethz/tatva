from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, vmap

from femsolver.utils import auto_vmap

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
    shear_tying_points = jnp.array(
        [
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )

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
        g1, g2 = J[:, 0], J[:, 1]
        t1 = g1 / jnp.linalg.norm(g1)
        t2_tmp = g2 - jnp.dot(g2, t1) * t1
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

    def _get_geometrical_quantities(
        self, xi: Array, nodal_coords: Array
    ) -> tuple[Array, Array, _LocalFrame, Array]:
        """Helper function to bundle operations. Returns shape functions, directional
        derivatives, local frame, and Jacobian area at given local coords.
        """
        J, J_area = self.get_jacobian(xi, nodal_coords)  # (3, 2), ()
        N = self.shape_function(xi)  # (4,)
        directional_dN = self.directional_derivative(xi, nodal_coords)  # (2, 4)
        frame = self.get_local_frame(J)
        return N, directional_dN, frame, J_area

    def _interpolate_kinematics(
        self,
        N: Array,
        directional_dN: Array,
        frame: _LocalFrame,
        u: Array,
        theta: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Helper function to bundle operations. Returns displacements and rotations
        at a given Gauss point."""
        frame_basis = jnp.vstack((frame.t1, frame.t2, frame.n))  # (3, 3)
        u_gp = (N[:, None] * u).sum(axis=0)
        u_gp_local = frame_basis @ u_gp
        dudX = u.T @ directional_dN.T  # (3, 4) @ (4, 2) -> (3, 2)

        # directional derivatives of w (out-of-plane displacement) in tangential basis
        dw = jnp.einsum("ij,i->j", dudX, frame.n)
        # rotations about t1, t2 axes
        theta_gp = (N[:, None] * theta).sum(axis=0)
        # rotation derivatives in tangential basis
        dtheta = theta.T @ directional_dN.T

        return u_gp_local, dudX, dw, theta_gp, dtheta

    def _mitc4_shear_at_tying_points(
        self, nodal_coords: Array, u: Array, theta: Array
    ) -> Array:
        """Calculate the MITC4 shear strains at the shear tying points.

        Args:
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            gamma_mitc: Shear strains at the shear tying points, shape (4, 2).
        """

        def gamma_at_gp(xi_tie: Array) -> Array:
            N, directional_dN, frame, _ = self._get_geometrical_quantities(
                xi_tie, nodal_coords
            )
            # TODO: optimize by spliting _interpolate_kinematics such that the unnecessary
            # part is not computed
            *_, dw, theta_gp, dtheta = self._interpolate_kinematics(
                N, directional_dN, frame, u, theta
            )
            # shear strains in local frame
            return dw - theta_gp

        return vmap(gamma_at_gp)(self.shear_tying_points)

    def energy_density_gp(
        self, xi: Array, nodal_coords: Array, u: Array, theta: Array
    ) -> tuple[Array, Array, Array]:
        """Calculate the energy density at a given Gauss point.

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            energy_density: Energy density at the given Gauss point, shape ().
        """
        N, directional_dN, frame, J_area = self._get_geometrical_quantities(
            xi, nodal_coords
        )
        u_gp_local, dudX, dw, theta_gp, dtheta = self._interpolate_kinematics(
            N, directional_dN, frame, u, theta
        )
        dudX_u = dudX[:2, :]  # in-plane displacement gradients
        strain_membrane = 0.5 * (dudX_u + dudX_u.T + dudX_u.T @ dudX_u) + jnp.outer(
            dw, dw
        )
        # bending curvatures
        kappa = jnp.array(
            [
                dtheta[0, 0],
                dtheta[1, 1],
                dtheta[0, 1] + dtheta[1, 0],
            ]
        )
        # shear strains (MITC4)
        gamma = self.mitc4_shear(xi, nodal_coords, u, theta)

        return strain_membrane, kappa, gamma

    def get_membrane_strain(self, xi: Array, nodal_coords: Array, u: Array) -> Array:
        """Calculate the membrane strain at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).

        Returns:
            strain_membrane: Membrane strain at the given local coordinates, shape (2, 2).
        """
        dN_directional = self.directional_derivative(xi, nodal_coords)
        dudX = u.T @ dN_directional.T

        dw = jnp.einsum(
            "ij,i->j",
            dudX,
            self.get_local_frame(self.get_jacobian(xi, nodal_coords)[0]).n,
        )

        dudX_u = dudX[:2, :]  # in-plane displacement gradients
        strain_membrane = 0.5 * (dudX_u + dudX_u.T + dudX_u.T @ dudX_u) + jnp.outer(
            dw, dw
        )
        return strain_membrane

    def get_bending_curvature(
        self, xi: Array, nodal_coords: Array, theta: Array
    ) -> Array:
        """Calculate the bending curvature at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            kappa: Bending curvature at the given local coordinates, shape (3,).
        """
        dN_directional = self.directional_derivative(xi, nodal_coords)
        dtheta = theta.T @ dN_directional.T

        kappa = jnp.array(
            [
                dtheta[0, 0],
                dtheta[1, 1],
                dtheta[0, 1] + dtheta[1, 0],
            ]
        )
        return kappa

    def mitc4_shear(
        self, xi: Array, nodal_coords: Array, u: Array, theta: Array
    ) -> Array:
        """Calculate the MITC4 shear strains at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            gamma_mitc: Shear strains at the given local coordinates, shape (2,).
        """
        gamma_tying = self._mitc4_shear_at_tying_points(nodal_coords, u, theta)
        r, s = xi
        # TODO: check if this is correct
        N_tie = jnp.array(
            [
                0.5 * (1.0 - s),
                0.5 * (1.0 + s),
                0.5 * (1.0 - r),
                0.5 * (1.0 + r),
            ]
        )
        return N_tie @ gamma_tying
