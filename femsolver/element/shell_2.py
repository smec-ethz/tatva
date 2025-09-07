from typing import NamedTuple, override

import jax
import jax.numpy as jnp
from jax import Array, vmap

from femsolver.utils import auto_vmap

from ._element import Element


def skew(v: Array) -> Array:
    """Returns the skew-symmetric matrix of a vector."""
    x, y, z = v
    return jnp.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=float,
    )


def rodrigues(omega, eps=1e-12):
    """
    JAX-friendly exponential map R = exp([omega]_x).
    Uses a small-angle series branch; AD-safe via lax.cond.
    """
    I = jnp.eye(3, dtype=omega.dtype)
    K = skew(omega)
    theta2 = jnp.dot(omega, omega)

    def small(_):
        # 2nd-order series: I + (1 - θ²/6)K + (1/2 - θ²/24)K²
        return I + (1.0 - theta2 / 6.0) * K + (0.5 - theta2 / 24.0) * (K @ K)

    def large(_):
        theta = jnp.sqrt(theta2)
        A = jnp.sin(theta) / theta
        B = (1.0 - jnp.cos(theta)) / theta2
        return I + A * K + B * (K @ K)

    return jax.lax.cond(theta2 < eps, small, large, operand=None)


def normalize(v: Array) -> Array:
    """Returns the normalized vector."""
    norm = jnp.linalg.norm(v)
    return v / jnp.where(norm < 1e-12, 1.0, norm)


def normalize_interp_and_grad(N, dN_dxi, nodal_vecs, eps=1e-12):
    s = N @ nodal_vecs  # (3,)
    s_a = jnp.tensordot(dN_dxi, nodal_vecs, axes=((1,), (0,)))  # (2,3)
    nrm = jnp.linalg.norm(s)
    d = s / jnp.where(nrm > eps, nrm, 1.0)
    P = jnp.eye(3) - jnp.outer(d, d)
    dd = (s_a @ P.T) / jnp.where(nrm > eps, nrm, 1.0)  # (2,3)
    return d, dd


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
    _shear_tying_points = jnp.array(
        [
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    _edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])

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

    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        """Calculate the Jacobian matrix and its determinant at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).

        Returns:
            J: Jacobian matrix at the given local coordinates, shape (2, 3).
            detJ: Determinant of the Jacobian matrix at the given local coordinates, shape ().
        """
        dN_dxi = self.shape_function_derivative(xi)
        J = dN_dxi @ nodal_coords  # (2, 3)
        detJ = jnp.linalg.det(J @ J.T) ** 0.5
        return J, detJ

    def _1d_shape_function(self, s: Array | float) -> Array:
        """Returns the 1D shape functions evaluated at the local coordinate xi."""
        return jnp.array([0.5 * (1.0 - s), 0.5 * (1.0 + s)])

    def _1d_shape_function_derivative(self, s: Array | float) -> Array:
        """Returns the derivative of the 1D shape functions with respect to the local coordinate xi."""
        return jnp.array([-0.5, 0.5])

    def get_triad(self, xi: Array, nodal_coords: Array) -> Array:
        """Calculate the triad (local coordinate system) at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).

        Returns:
            triad: Triad at the given local coordinates, shape (3, 3).
        """
        dN_dxi = self.shape_function_derivative(xi)
        A = dN_dxi @ nodal_coords  # (2, 3)
        e3 = normalize(jnp.cross(A[0], A[1]))
        e1 = normalize(A[0])
        e2 = normalize(jnp.cross(e3, e1))
        return jnp.stack([e1, e2, e3], axis=1)

    def _interpolate(self, xi: Array, X: Array, U: Array, TH: Array):
        N = self.shape_function(xi)  # (4,)
        dN_dxi = self.shape_function_derivative(xi)  # (2, 4)

        # initial and current nodal positions
        Xn = X  # (4, 3)
        xn = X + U  # (4, 3)

        # interpolate fields
        R0 = N @ Xn  # (3,)
        r = N @ xn  # (3,)

        # covariant tangent bases (parametric derivatives)
        A = dN_dxi @ Xn  # (2, 3)
        a = dN_dxi @ xn  # (2, 3)

        # rotated directors at nodes (large rotation)
        triad = self.get_triad(xi, Xn)  # (3, 3)
        dn0 = triad[:, 2]  # (3,) initial director at gp
        # stack for times dn0
        dn0 = jnp.tile(dn0, (4, 1))  # (4, 3)

        # omega_local = jnp.array([theta_gp[0], theta_gp[1], 0.0])
        omega_local = jnp.hstack((TH, jnp.zeros((4, 1))))  # (4, 3)
        omega = vmap(lambda w: triad @ w)(omega_local)  # (4, 3)
        Rn = vmap(rodrigues, 0)(omega)  # (4, 3, 3)
        dn = vmap(lambda R, d0: normalize(R @ d0))(Rn, dn0)  # (4, 3)

        # director derivatives (needed for curvature)
        d, dd_dxi = normalize_interp_and_grad(N, dN_dxi, dn)  # (3,), (2, 3)
        D0, dD0_dxi = normalize_interp_and_grad(N, dN_dxi, dn0)  # = (0,0,1), zero grads

        return _InterpolateResult(N, dN_dxi, R0, r, A, a, D0, d, dD0_dxi, dd_dxi)

    def membrane_strain(
        self, xi: Array, nodal_coords: Array, U: Array, TH: Array
    ) -> Array:
        """Calculate the membrane strains at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            epsilon: Membrane strains at the given local coordinates, shape (3,).
        """
        val = self._interpolate(xi, nodal_coords, U, TH)
        E = 0.5 * (val.a @ val.a.T - val.A @ val.A.T)  # (2, 2)
        return jnp.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]])  # (3,) Voigt notation

    def bending_curvature(
        self, xi: Array, nodal_coords: Array, U: Array, TH: Array
    ) -> Array:
        """Calculate the bending curvatures at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            kappa: Bending curvatures at the given local coordinates, shape (3,).
        """
        val = self._interpolate(xi, nodal_coords, U, TH)
        B0 = -val.dD0_dxi @ val.A.T  # (2, 2)
        b = -val.dd_dxi @ val.a.T  # (2, 2)
        kappa = b - B0  # (2, 2)
        kappa = 0.5 * (kappa + kappa.T)  # make symmetric
        return jnp.array(
            [kappa[0, 0], kappa[1, 1], 2.0 * kappa[0, 1]]
        )  # (3,) Voigt notation

    def mitc4_shear(self, xi: Array, nodal_coords: Array, U: Array, TH: Array) -> Array:
        """Calculate the MITC4 shear strains at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            gamma_mitc: Shear strains at the given local coordinates, shape (2,).
        """
        gamma_tying = self._mitc4_shear_at_tying_points(nodal_coords, U, TH)
        r, s = xi
        # TODO: check if this is correct
        N_tie = jnp.array(
            [
                [0.0, 0.5 * (1.0 - r)],
                [0.5 * (1.0 + s), 0.0],
                [0.0, 0.5 * (1.0 + r)],
                [0.5 * (1.0 - s), 0.0],
            ]
        )
        return N_tie.T @ gamma_tying  # (2,4) @ (4,) = (2,)

    def _mitc4_shear_at_tying_points(
        self, nodal_coords: Array, U: Array, TH: Array
    ) -> Array:
        """Calculate the MITC4 shear strains at the shear tying points.

        Args:
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            gamma_mitc: Shear strains at the shear tying points, shape (4, 2).
        """
        triad = self.get_triad(jnp.array([0, 0]), nodal_coords)  # (3, 3)
        dn0 = triad[:, 2]  # (3,) initial director at gp
        dn0 = jnp.tile(dn0, (4, 1))  # (4, 3)
        omega_local = jnp.hstack((TH, jnp.zeros((4, 1))))  # (4, 3)
        omega = vmap(lambda w: triad @ w)(omega_local)  # (4, 3)
        Rn = vmap(rodrigues, 0)(omega)  # (4, 3, 3)
        dn = vmap(lambda R, d0: normalize(R @ d0))(Rn, dn0)  # (4, 3)

        def shear_at_edge(edge: Array) -> Array:
            x_e = nodal_coords[edge] + U[edge]  # (2, 3)
            N1d = self._1d_shape_function(0)  # (2,)
            dN1d = self._1d_shape_function_derivative(0)  # (2,)
            t_e = dN1d @ x_e  # (3,)

            d = dn[edge]  # (2, 3)
            d = N1d @ d  # (3,)
            g3 = normalize(d)
            return jnp.dot(t_e, g3)  # scalar shear along edge

        gamma = vmap(shear_at_edge)(self._edges)  # (4,)
        return gamma

    def drill_penalty(
        self,
        xi: Array,
        nodal_coords: Array,
        U: Array,
        TH: Array,
        *,
        E: float = 1.0,
        t: float = 1.0,
        beta=1e-3,
    ) -> Array:
        """Calculate the drilling penalty at the given local coordinates (xi, eta).

        Args:
            xi: Local coordinates (r, s).
            nodal_coords: Nodal coordinates of the element, shape (4, 3).
            u: Nodal displacements of the element, shape (4, 3).
            theta: Nodal rotations of the element, shape (4, 2).

        Returns:
            penalty: Drilling penalty at the given local coordinates, shape ().
        """
        val = self._interpolate(xi, nodal_coords, U, TH)
        a1 = normalize(val.a[0])
        a2 = normalize(jnp.cross(val.d, a1))
        theta_gp = val.N @ TH
        R_gp = rodrigues(theta_gp)
        e1 = jnp.array([1.0, 0.0, 0.0])
        e2 = jnp.array([0.0, 1.0, 0.0])
        m1 = R_gp @ e1
        m2 = R_gp @ e2

        mis = jnp.sum((a1 - m1) ** 2) + jnp.sum((a2 - m2) ** 2)
        return beta * E * t * mis


class _InterpolateResult(NamedTuple):
    N: Array  # (4,) shape functions at xi
    dN_dxi: Array  # (2, 4) shape function derivatives at xi
    R0: Array  # (3,) initial position at xi
    r: Array  # (3,) current position at xi
    A: Array  # (2, 3) initial covariant basis
    a: Array  # (2, 3) current covariant basis
    D0: Array  # (3,) initial director at xi
    d: Array  # (3,) current directors at xi
    dD0_dxi: Array  # (2, 3) initial director derivatives at xi
    dd_dxi: Array  # (2, 3) current director derivatives at xi
