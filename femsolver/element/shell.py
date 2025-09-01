import jax.numpy as jnp
from jax import Array, vmap

from femsolver.operator import MappableOverElements

from ._element import Element


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

    @staticmethod
    def _orthonormal_frame(A: Array) -> Array:
        """Build local orthonormal frame R=[e1, e2, e3] from A=[a1, a2].
        Shape: (3, 3)

        Args:
            A: surface tangent map, shape (3, 2)
        """
        a1, a2 = A[:, 0], A[:, 1]
        e1 = a1 / jnp.linalg.norm(a1)
        # remove e1 component from a2, then normalize
        a2_t = a2 - (e1 @ a2) * e1
        e2 = a2_t / jnp.linalg.norm(a2_t)
        e3 = jnp.cross(e1, e2)
        return jnp.stack((e1, e2, e3), axis=1)

    def surface_tangents(
        self, xi: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array]:
        """Surface tangents and metric at (r, s).

        Args:
            xi: (2,) parametric coords (r, s)
            nodal_coords: (4, 3) nodal coords of the element

        Returns:
            A: (3, 2) surface tangents, with columns a1=dX/dr, a2=dX/ds
            G: (2, 2) first fundamental form = A^T A (metric)
        """
        dN_dxi = self.shape_function_derivative(xi)  # (2, 4)
        A = nodal_coords.T @ dN_dxi.T  # (3, 4) @ (4, 2) -> (3, 2)
        G = A.T @ A  # (2, 2) first fundamental form
        J_area = jnp.sqrt(jnp.linalg.det(G))  # area scaling factor
        return A, G, J_area

    def shape_function_surface_grads(self, xi: Array, nodal_coords: Array) -> Array:
        """Surface (Cartesian) gradients ∇_s N_i at (r, s) in R^3.
        Uses Moore-Penrose for pseudoinverse 3x2 A: A^+ = A (A^T A)^-1.

        Args:
            xi: (2,) parametric coords (r, s)
            nodal_coords: (4, 3) nodal coords of the element

        Returns:
            dN_dX: (3, 4) column i is ∇_s N_i
        """
        dN_dxi = self.shape_function_derivative(xi)  # (2, 4)
        A, G, _ = self.surface_tangents(xi, nodal_coords)
        Y = jnp.linalg.solve(G, dN_dxi)  # (2, 4), solves G Y = dN_dxi
        dN_dX = A @ Y  # (3, 2) @ (2, 4) -> (3, 4)
        return dN_dX

    def local_frame(self, xi: Array, nodal_coords: Array) -> Array:
        """Local orthonormal frame R=[e1, e2, e3] at (r, s).

        Returns:
            R: (3, 3) world->local rotation is R^T; local->world is R
        """
        A, *_ = self.surface_tangents(xi, nodal_coords)
        return self._orthonormal_frame(A)

    def shape_function_local_grads(self, xi: Array, nodal_coords: Array) -> Array:
        """Shape function gradients in the local frame coordinates.

        Args:
            xi: (2,) parametric coords (r, s)
            nodal_coords: (4, 3) nodal coords of the element

        Returns:
            dN_dloc: (3, 4). Rows correspond to (e1, e2, e3) components. The third row
                should be ~zero for surface gradients.
        """
        dN_dX = self.shape_function_surface_grads(xi, nodal_coords)  # (3, 4)
        R = self.local_frame(xi, nodal_coords)  # (3, 3)
        dN_dloc = R.T @ dN_dX  # (3, 3) @ (3, 4) -> (3, 4)
        return dN_dloc

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
        dN_dX = self.shape_function_surface_grads(xi, nodal_coords)  # (3, 4)
        return dN_dX @ nodal_values  # (3, 4) @ (4, nvalues) -> (3, nvalues)

    def _geom_at(
        self, xi: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Helper function to bundle operations. Returns shape functions, directional
        derivatives, local frame, and Jacobian area at given local coords.

        Returns:
            N, dN/dt (tangential derivatives), R (local frame), J_area
        """
        N = self.shape_function(xi)  # (4,)
        R = self.local_frame(xi, nodal_coords)  # (3, 3)
        dN_dtan = self.shape_function_local_grads(xi, nodal_coords)[:2]  # (2, 4)
        _, _, J_area = self.surface_tangents(
            xi, nodal_coords
        )  # need area Jacobian only
        return N, dN_dtan, R, J_area

    def _interpolate_kinematics(
        self,
        N: Array,
        dN_dtan: Array,
        R: Array,
        u: Array,
        theta: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Helper function to bundle operations. Returns displacements and rotations
        at a given Gauss point.

        Returns:
            u_gp_local: Displacement at the Gauss point in local frame, shape (3,).
            U_tan: (3, 2) displacement tangential derivatives in world frame.
            dw: (2,) normal components of U_tan (dw/dr, dw/ds).
            theta_gp: (2,) rotation at the Gauss point (about t1, t2).
            dtheta_tan: (2, 2) rotation tangential derivatives in world frame.
        """
        u_gp = N @ u  # (3,)
        u_gp_local = R.T @ u_gp  # (3,)

        # tangential derivatives of displacement
        U_tan = u.T @ dN_dtan.T  # (3, 4) @ (4, 2) -> (3, 2)
        n = R[:, 2]  # (3,)
        dw = U_tan.T @ n  # (2,) dw/dr, dw/ds

        theta_gp = N @ theta  # (2,)
        dtheta_tan = theta.T @ dN_dtan.T  # (2, 4) @ (4, 2) -> (2, 2)

        return u_gp_local, U_tan, dw, theta_gp, dtheta_tan

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
            N, dN_dtan, R, _ = self._geom_at(xi_tie, nodal_coords)
            # TODO: optimize by spliting _interpolate_kinematics such that the unnecessary
            # part is not computed
            _, _, dw, theta_gp, _ = self._interpolate_kinematics(
                N, dN_dtan, R, u, theta
            )
            # shear strains in local frame
            return dw - theta_gp  # (2,)

        return vmap(gamma_at_gp)(self.shear_tying_points)  # (4, 2)

    @staticmethod
    def _constitutive_mats(E: float, nu: float, t: float, kappa_s: float = 5 / 6):
        """Return (A, D, Ks). Voigt order [xx, yy, xy], with γ_xy = 2 ε_xy.

                Args:
                    E: Young's modulus.
                    nu: Poisson's ratio.
        t: Thickness of the shell.
                    kappa_s: Shear correction factor, default is 5/6.
        """
        c = E / (1.0 - nu**2)
        M = jnp.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
        A = c * t * M  # membrane
        D = c * (t**3) / 12.0 * M  # bending
        G = E / (2.0 * (1.0 + nu))
        Ks = kappa_s * G * t * jnp.eye(2)  # transverse shear (2x2)
        return A, D, Ks

    def get_local_values(
        self, xi: Array, nodal_coords: Array, U: Array, theta: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Get local displacements, tangential displacement derivatives,
        normal displacement derivatives, rotations, and rotation derivatives at (r, s).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            U: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            u_gp_local: Displacement at the Gauss point in local frame, shape (3,).
            U_tan: (3, 2) displacement tangential derivatives in world frame.
            dw: (2,) normal components of U_tan (dw/dr, dw/ds).
            theta_gp: (2,) rotation at the Gauss point (about t1, t2).
            dtheta_tan: (2, 2) rotation tangential derivatives in world frame.
        """
        N, dN_dtan, R, _ = self._geom_at(xi, nodal_coords)
        return self._interpolate_kinematics(N, dN_dtan, R, U, theta)

    def energy_density(
        self,
        xi: Array,
        nodal_coords: Array,
        u: Array,
        theta: Array,
        E: float,
        nu: float,
        t: float,
        kappa_s: float = 5 / 6,
    ) -> Array:
        """Calculate the energy density at a given Gauss point.

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).
            E: Young's modulus.
            nu: Poisson's ratio.
            t: Thickness of the shell.
            kappa_s: Shear correction factor, default is 5/6.

        Returns:
            energy_density: Energy density at the given Gauss point, shape ().
        """
        A, D, Ks = self._constitutive_mats(E, nu, t, kappa_s)

        # membrane + bending
        N, dN_dtan, R, J_area = self._geom_at(xi, nodal_coords)
        u_gp_local, U_tan, dw, theta_gp, dtheta_tan = self._interpolate_kinematics(
            N, dN_dtan, R, u, theta
        )

        # membrane strains (Voigt [xx, yy, xy] with γ_xy = 2ε_xy) in LOCAL tangential frame
        U_tan_local = R.T @ U_tan  # (3, 2)
        eps_membrane = jnp.array(
            [
                U_tan_local[0, 0],
                U_tan_local[1, 1],
                U_tan_local[0, 1] + U_tan_local[1, 0],
            ]
        )
        energy_membrane = 0.5 * (eps_membrane @ (A @ eps_membrane))

        # bending curvatures from rotation gradients (engineering form)
        kappa_v = jnp.array(
            [dtheta_tan[0, 0], dtheta_tan[1, 1], dtheta_tan[0, 1] + dtheta_tan[1, 0]]
        )
        energy_bending = 0.5 * (kappa_v @ (D @ kappa_v))

        gamma = self.mitc4_shear(xi, nodal_coords, u, theta)  # (2,)
        energy_shear = 0.5 * (gamma @ (Ks @ gamma))

        return energy_membrane + energy_bending + energy_shear

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
