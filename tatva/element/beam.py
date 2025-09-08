from abc import abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker

import equinox as eqx


# Define Hermite shape functions for bending (cubic interpolation)
@jax.jit
def N1(xi):
    """Hermite shape function N1 for transverse displacement at node 1"""
    return (2 + xi) * (1 - xi) ** 2 / 4


@jax.jit
def N2(xi, L):
    """Hermite shape function N2 for rotation at node 1"""
    return L * (1 + xi) * (1 - xi) ** 2 / 8


@jax.jit
def N3(xi):
    """Hermite shape function N3 for transverse displacement at node 2"""
    return (2 - xi) * (1 + xi) ** 2 / 4


@jax.jit
def N4(xi, L):
    """Hermite shape function N4 for rotation at node 2"""
    return L * (1 + xi) ** 2 * (xi - 1) / 8


# Linear shape functions for axial displacement
@jax.jit
def N_axial_1(xi):
    """Linear shape function for axial displacement at node 1"""
    return (1 - xi) / 2


@jax.jit
def N_axial_2(xi):
    """Linear shape function for axial displacement at node 2"""
    return (1 + xi) / 2


@jax.jit
def compute_hermite_functions(xi, L):
    """Compute all Hermite shape functions for bending"""
    return N1(xi), N2(xi, L), N3(xi), N4(xi, L)


@jax.jit
def compute_axial_functions(xi):
    """Compute linear shape functions for axial displacement"""
    return N_axial_1(xi), N_axial_2(xi)


# Define local axial displacement (linear interpolation)
@jax.jit
def u(xi, u1, u2):
    """Axial displacement along the beam"""
    N1_ax, N2_ax = compute_axial_functions(xi)
    return N1_ax * u1 + N2_ax * u2


# Define local transverse displacement (Hermite interpolation)
@jax.jit
def w(xi, w1, theta1, w2, theta2, L):
    """Transverse displacement along the beam"""
    N1_val, N2_val, N3_val, N4_val = compute_hermite_functions(xi, L)
    return N1_val * w1 + N2_val * theta1 + N3_val * w2 + N4_val * theta2


@jax.jit
def v(xi, theta1, theta2, L):
    """Transverse displacement along the beam"""
    # N1_val, N2_val, N3_val, N4_val = compute_hermite_functions(xi, L)
    N1_ax, N2_ax = compute_axial_functions(xi)
    return N1_ax * theta1 + N2_ax * theta2


# Compute derivatives (strains)
du_dxi = jax.grad(u, argnums=0)  # axial strain (needs chain rule for dx)
dw_dxi = jax.grad(w, argnums=0)  # first derivative of transverse displacement
d2w_dxi2 = jax.grad(dw_dxi, argnums=0)  # second derivative (curvature)


@jaxtyped(typechecker=typechecker)
class BeamElement(eqx.Module):
    quad_points: Float[Array, "a"]
    quad_weights: Float[Array, "a"]
    dofs_per_node: int = 3

    @eqx.filter_jit
    def get_jacobian(
        self, xi: float, nodal_coords: Float[Array, "2 dim1"]
    ) -> Tuple[float, float]:
        J = jnp.linalg.norm(nodal_coords[1] - nodal_coords[0]) / 2
        return J, J

    @eqx.filter_jit
    def compute_length(self, coords: Float[Array, "2 dim1"]) -> float:
        return jnp.linalg.norm(coords[1] - coords[0])

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def interpolate(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodal_coords: Float[Array, "2 dim"],
    ) -> Float[Array, "3"]:
        L = self.compute_length(nodal_coords)
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodal_coords)

        # Displacements
        u_val = self.u(xi, local_dofs)
        w_val = self.w(xi, local_dofs, L)
        theta_val = self.v(xi, local_dofs, L)

        return jnp.array([u_val, w_val, theta_val])

    @eqx.filter_jit
    def get_local_values(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodal_coords: Float[Array, "2 dim"],
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Compute displacements and strains at natural coordinate xi

        Args:
            xi: Natural coordinate (-1 to 1)
            nodal_values: [u1, w1, theta1, u2, w2, theta2] - local DOFs
            nodal_coords: [x1, y1, x2, y2] - nodal coordinates

        Returns:
            u_val: Axial displacement
            w_val: Transverse displacement
            theta_val: Rotation
            du_dx: Axial strain
            dw_dx: Transverse strain
            d2w_dx2: Curvature
            detJ: Jacobian determinant
        """
        J, detJ = self.get_jacobian(xi, nodal_coords)

        # Compute the local DOFs
        L = self.compute_length(nodal_coords)
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodal_coords)

        # Displacements
        u_val = self.u(xi, local_dofs)
        w_val = self.w(xi, local_dofs, L)
        theta_val = self.v(xi, local_dofs, L)

        # Strains (need to transform from natural to physical coordinates)
        # Chain rule: du/dx = du/dxi * dxi/dx = du/dxi * 2/L
        du_dx = self.dudx(xi, local_dofs, L)  # axial strain

        dw_dx = self.dwdx(xi, local_dofs, L)

        # For bending: d²w/dx² = d²w/dxi² * (2/L)²
        d2w_dx2 = self.d2wdx2(xi, local_dofs, L)  # curvature

        return u_val, w_val, theta_val, du_dx, dw_dx, d2w_dx2, detJ

    @abstractmethod
    def u(self, xi: Array, local_dofs: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def w(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def v(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def dudx(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def dwdx(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def d2wdx2(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def compute_dofs_local_frame(self, dofs: Array, coords: Array) -> Array:
        raise NotImplementedError

    @eqx.filter_jit
    def axial_gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.dudx(xi, local_dofs, L)

    @eqx.filter_jit
    def transverse_gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.dwdx(xi, local_dofs, L)

    @eqx.filter_jit
    def curvature(
        self,
        xi: float,
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.d2wdx2(xi, local_dofs, L)

    @jaxtyped(typechecker=typechecker)
    def gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        return (
            self.axial_gradient(xi, nodal_values, nodes),
            self.transverse_gradient(xi, nodal_values, nodes),
            self.curvature(xi, nodal_values, nodes),
        )


# --------------------------------
# Linear approach
# --------------------------------


@jax.jit
def u_axial(xi, u1, u2):
    """Axial displacement along the beam"""
    N1_val, N2_val = compute_axial_functions(xi)
    return N1_val * u1 + N2_val * u2


# Define local transverse displacement (linear interpolation for Timoshenko)
@jax.jit
def w_axial(xi, w1, w2):
    """Transverse displacement along the beam"""
    N1_val, N2_val = compute_axial_functions(xi)
    return N1_val * w1 + N2_val * w2


# Define rotation (independent of slope in Timoshenko theory)
@jax.jit
def theta_axial(xi, theta1, theta2):
    """Rotation along the beam (independent DOF in Timoshenko)"""
    N1_val, N2_val = compute_axial_functions(xi)
    return N1_val * theta1 + N2_val * theta2


# Compute derivatives (strains and shear)
du_axial_dxi = jax.grad(u_axial, argnums=0)  # axial strain (needs chain rule for dx)
dw_axial_dxi = jax.grad(w_axial, argnums=0)  # slope of transverse displacement
dtheta_axial_dxi = jax.grad(theta_axial, argnums=0)  # curvature (change in rotation)


class LinearElement(BeamElement):

    @eqx.filter_jit
    def u(self, xi: Array, local_dofs: Array) -> Array:
        """Axial displacement along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return u_axial(xi, u1, u2)

    @eqx.filter_jit
    def w(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        """Transverse displacement along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return w_axial(xi, w1, w2)

    @eqx.filter_jit
    def v(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        """Rotation along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return theta_axial(xi, theta1, theta2)

    @eqx.filter_jit
    def dudx(self, xi: Array, local_dofs: Array, L: Float) -> Array:
        """Axial strain along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return du_axial_dxi(xi, u1, u2) * 2 / L

    @eqx.filter_jit
    def dwdx(self, xi: float, local_dofs: Array, L: float) -> float:
        """Transverse strain along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return dw_axial_dxi(xi, w1, w2) * 2 / L

    @eqx.filter_jit
    def d2wdx2(self, xi: float, local_dofs: Array, L: float) -> float:
        """Curvature along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return dtheta_axial_dxi(xi, theta1, theta2) * (2 / L)

    @eqx.filter_jit
    def compute_dofs_local_frame(
        self, dofs: Float[Array, "2 3"], coords: Float[Array, "2 dim"]
    ) -> Float[Array, "6"]:
        """
        Compute the dofs of the beam in the local frame.
        """
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        L = jnp.sqrt(dx**2 + dy**2)

        cos_theta = dx / L
        sin_theta = dy / L

        # Transformation matrix for each node
        T_node = jnp.array(
            [[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        # Transform DOFs for both nodes
        dofs_local_1 = T_node @ dofs[0]
        dofs_local_2 = T_node @ dofs[1]

        # Return as [u1, w1, theta1, u2, w2, theta2]
        return jnp.array(
            [
                dofs_local_1[0],
                dofs_local_1[1],
                dofs_local_1[2],
                dofs_local_2[0],
                dofs_local_2[1],
                dofs_local_2[2],
            ]
        )


# --------------------------------
# Hermite approach
# --------------------------------


@jaxtyped(typechecker=typechecker)
class HermiteElement(BeamElement):

    @eqx.filter_jit
    def u(self, xi: float, local_dofs: Float[Array, "6"]) -> float:
        """Axial displacement along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return u(xi, u1, u2)

    @eqx.filter_jit
    def w(self, xi: float, local_dofs: Float[Array, "6"], L: float) -> float:
        """Transverse displacement along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return w(xi, w1, theta1, w2, theta2, L)

    @eqx.filter_jit
    def v(self, xi: float, local_dofs: Float[Array, "6"], L: float) -> float:
        """Rotation along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return v(xi, theta1, theta2, L)

    @eqx.filter_jit
    def dudx(self, xi: float, local_dofs: Float[Array, "6"], L: float) -> float:
        """Axial strain along the beam"""
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return du_dxi(xi, u1, u2) * 2 / L

    @eqx.filter_jit
    def dwdx(self, xi: float, local_dofs: Float[Array, "6"], L: float) -> float:
        """
        First derivative of transverse displacement along the beam
        """
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return dw_dxi(xi, w1, theta1, w2, theta2, L) * 2 / L

    @eqx.filter_jit
    def d2wdx2(self, xi: float, local_dofs: Float[Array, "6"], L: float) -> float:
        """
        Second derivative of transverse displacement along the beam (curvature)
        """
        u1, w1, theta1, u2, w2, theta2 = local_dofs
        return d2w_dxi2(xi, w1, theta1, w2, theta2, L) * (2 / L) ** 2

    @eqx.filter_jit
    def compute_dofs_local_frame(
        self, dofs: Float[Array, "2 3"], coords: Float[Array, "2 dim"]
    ) -> Float[Array, "6"]:
        """
        Compute the dofs of the beam in the local frame.
        """
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        L = jnp.sqrt(dx**2 + dy**2)

        cos_theta = dx / L
        sin_theta = dy / L

        # Transformation matrix for each node
        T_node = jnp.array(
            [[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        # Transform DOFs for both nodes
        dofs_local_1 = T_node @ dofs[0]
        dofs_local_2 = T_node @ dofs[1]

        # Return as [u1, w1, theta1, u2, w2, theta2]
        return jnp.array(
            [
                dofs_local_1[0],
                dofs_local_1[1],
                dofs_local_1[2],
                dofs_local_2[0],
                dofs_local_2[1],
                dofs_local_2[2],
            ]
        )


# --------------------------------
# Corotational approach
# --------------------------------

@jax.jit
def phi_1(x, L, omega):
    mu = 1 / (1 + 12 * omega)
    return mu * x * ((1 - x / L) ** 2 + 6 * omega * (1 - x / L))


@jax.jit
def phi_2(x, L, omega):
    mu = 1 / (1 + 12 * omega)
    return mu * x * (-x / L + (x**2) / (L**2) + 6 * omega * (x / L - 1))


phi_3 = jax.jacrev(phi_2)
phi_4 = jax.jacrev(phi_3)


# Define local axial displacement
@jax.jit
def _u(x, u_s, L):
    return u_s * x / L


# Define local normal displacement
@jax.jit
def _w(x, tl1, tl2, L, omega):
    return phi_1(x, L, omega) * tl1 + phi_2(x, L, omega) * tl2


# Define local rotation
@jax.jit
def _v(x, tl1, tl2, L, omega):
    return phi_3(x, L, omega) * tl1 + phi_4(x, L, omega) * tl2


# Compute derivatives
dudx = jax.grad(_u, argnums=0)  # local axial strain
dwdx = jax.grad(_w, argnums=0)  # local rotation
d2wdx2 = jax.grad(dwdx, argnums=0)  # local curvature


@jaxtyped(typechecker=typechecker)
class CorotationalHermiteElement(eqx.Module):
    quad_points: Float[Array, "a"]
    quad_weights: Float[Array, "a"]
    omega: float
    dofs_per_node: int = 3

    @eqx.filter_jit
    def get_jacobian(
        self, xi: float, nodal_coords: Float[Array, "2 dim1"]
    ) -> Tuple[float, float]:
        J = jnp.linalg.norm(nodal_coords[1] - nodal_coords[0]) / 2
        return J, J

    @eqx.filter_jit
    def compute_length(self, coords: Float[Array, "2 dim1"]) -> float:
        return jnp.linalg.norm(coords[1] - coords[0])

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def interpolate(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodal_coords: Float[Array, "2 dim"],
    ) -> Float[Array, "3"]:
        L = self.compute_length(nodal_coords)
        xi = L * (xi + 1) * 0.5
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodal_coords)

        # Displacements
        u_val = self.u(xi, local_dofs, L)
        w_val = self.w(xi, local_dofs, L, self.omega)
        theta_val = self.v(xi, local_dofs, L, self.omega)

        u_rigid = (xi * nodal_values[1][0] + (L - xi) * nodal_values[0][0]) / L
        w_rigid = (xi * nodal_values[1][1] + (L - xi) * nodal_values[0][1]) / L
        theta_rigid = (xi * nodal_values[1][2] + (L - xi) * nodal_values[0][2]) / L

        return jnp.array([u_val + u_rigid, w_val + w_rigid, theta_val + theta_rigid])

    @eqx.filter_jit
    def get_local_values(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodal_coords: Float[Array, "2 dim"],
    ) -> Tuple[
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
    ]:
        """Compute displacements and strains at natural coordinate xi

        Args:
            xi: Natural coordinate (-1 to 1)
            nodal_values: [u1, theta1, theta2] - global DOFs
            nodal_coords: [x1, y1, x2, y2] - global coordinates

        Returns:
            u_val: Axial displacement
            w_val: Transverse displacement
            theta_val: Rotation
            du_dx: Axial strain
            dw_dx: Transverse strain
            d2w_dx2: Curvature
            detJ: Jacobian determinant
        """

        L = self.compute_length(nodal_coords)
        xi = L * (xi + 1) * 0.5

        J, detJ = self.get_jacobian(xi, nodal_coords)

        # Compute the local DOFs
        local_dofs = self.compute_dofs_local_frame(nodal_values, nodal_coords)

        # Displacements
        u_val = self.u(xi, local_dofs, L)
        w_val = self.w(xi, local_dofs, L, self.omega)
        theta_val = self.v(xi, local_dofs, L, self.omega)

        # Strains (need to transform from natural to physical coordinates)
        # Chain rule: du/dx = du/dxi * dxi/dx = du/dxi * 2/L
        du_dx = self.dudx(xi, local_dofs, L)  # axial strain

        dw_dx = self.dwdx(xi, local_dofs, L, self.omega)

        # For bending: d²w/dx² = d²w/dxi² * (2/L)²
        d2w_dx2 = self.d2wdx2(xi, local_dofs, L, self.omega)  # curvature

        return u_val, w_val, theta_val, du_dx, dw_dx, d2w_dx2, detJ

    @eqx.filter_jit
    def u(self, xi: float, local_dofs: Float[Array, "3"], L: float) -> float:
        """Axial displacement along the beam"""
        u1, theta1, theta2 = local_dofs
        return _u(xi, u1, L)

    @eqx.filter_jit
    def w(
        self,
        xi: float,
        local_dofs: Float[Array, "3"],
        L: float,
        omega: float,
    ) -> float:
        u1, theta1, theta2 = local_dofs
        return _w(xi, theta1, theta2, L, omega)

    @eqx.filter_jit
    def v(
        self,
        xi: float,
        local_dofs: Float[Array, "3"],
        L: float,
        omega: float,
    ) -> float:
        u1, theta1, theta2 = local_dofs
        return _v(xi, theta1, theta2, L, omega)

    @eqx.filter_jit
    def dudx(
        self,
        xi: float,
        local_dofs: Float[Array, "3"],
        L: float,
    ) -> float:
        u1, theta1, theta2 = local_dofs
        return dudx(xi, u1, L)

    @eqx.filter_jit
    def dwdx(
        self,
        xi: float,
        local_dofs: Float[Array, "3"],
        L: float,
        omega: float,
    ) -> Float[jnp.ndarray, "..."]:
        u1, theta1, theta2 = local_dofs
        return dwdx(xi, theta1, theta2, L, omega)

    @eqx.filter_jit
    def d2wdx2(
        self,
        xi: float,
        local_dofs: Float[Array, "3"],
        L: float,
        omega: float,
    ) -> float:
        u1, theta1, theta2 = local_dofs
        return d2wdx2(xi, theta1, theta2, L, omega)

    @eqx.filter_jit
    def compute_dofs_local_frame(
        self, dofs: Float[Array, "2 3"], coords: Float[Array, "2 dim"]
    ) -> Float[Array, "3"]:
        """
        Compute the dofs of the beam in the local frame.

        Args:
            dofs: The dofs of the beam per cell.
            coords: The coordinates of the beam per cell.

        Returns:
            The dofs of the beam in the local frame.
        """

        current_coords = coords + dofs[:, :2]
        l0 = self.compute_length(coords)
        ln = self.compute_length(current_coords)

        u_bar = (ln**2 - l0**2) / (ln + l0)
        beta0 = jnp.arctan2(
            (coords[1][1] - coords[0][1]), (coords[1][0] - coords[0][0])
        )

        beta = jnp.arctan2(
            (current_coords[1][1] - current_coords[0][1]),
            (current_coords[1][0] - current_coords[0][0]),
        )

        beta1 = dofs[0][self.dofs_per_node - 1] + beta0
        beta2 = dofs[1][self.dofs_per_node - 1] + beta0

        theta1_bar = jnp.arctan2(
            (jnp.cos(beta) * jnp.sin(beta1) - jnp.sin(beta) * jnp.cos(beta1)),
            (jnp.cos(beta) * jnp.cos(beta1) + jnp.sin(beta) * jnp.sin(beta1)),
        )

        theta2_bar = jnp.arctan2(
            (jnp.cos(beta) * jnp.sin(beta2) - jnp.sin(beta) * jnp.cos(beta2)),
            (jnp.cos(beta) * jnp.cos(beta2) + jnp.sin(beta) * jnp.sin(beta2)),
        )

        return jnp.array([u_bar, theta1_bar, theta2_bar])

    @eqx.filter_jit
    def axial_gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        xi = L * (xi + 1) * 0.5

        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.dudx(xi, local_dofs, L)

    @eqx.filter_jit
    def transverse_gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        xi = L * (xi + 1) * 0.5

        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.dwdx(xi, local_dofs, L, self.omega)

    @eqx.filter_jit
    def curvature(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Float[Array, "..."]:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.compute_length(nodes)
        xi = L * (xi + 1) * 0.5

        local_dofs = self.compute_dofs_local_frame(nodal_values, nodes)

        return self.d2wdx2(xi, local_dofs, L, self.omega)

    @jaxtyped(typechecker=typechecker)
    def gradient(
        self,
        xi: Float[Array, "..."],
        nodal_values: Float[Array, "2 3"],
        nodes: Float[Array, "2 dim"],
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        return (
            self.axial_gradient(xi, nodal_values, nodes),
            self.transverse_gradient(xi, nodal_values, nodes),
            self.curvature(xi, nodal_values, nodes),
        )
