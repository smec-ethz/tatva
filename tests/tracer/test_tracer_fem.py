from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator
from tatva.compound import Compound, field
from tatva.element.base import Tri3
from tatva.lifter import Fixed, Lifter, Periodic
from tatva.tracer.api import CapturedJaxpr, trace

jax.config.update("jax_enable_x64", True)


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  # Diffusion coefficient
    lmbda: float  # Diffusion coefficient

    @classmethod
    def from_youngs_poisson_2d(
        cls, E: float, nu: float, plane_stress: bool = False
    ) -> "Material":
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * nu * mu / (1 - nu)
        else:
            lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
        return cls(mu=mu, lmbda=lmbda)


@autovmap(grad_u=2)
def compute_deformation_gradient(grad_u: Array) -> Array:
    return jnp.eye(2) + grad_u


@autovmap(grad_u=2, mat=None)
def strain_energy_density(grad_u: Array, mat: Material) -> Array:
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)
    return (
        mat.mu / 2 * (jnp.trace(C) - 2)  # 2D case
        - mat.mu * jnp.log(J)
        + (mat.lmbda / 2) * (jnp.log(J)) ** 2
    )


@autovmap(grad_u=2, mat=None)
def get_cauchy_stress(grad_u: Array, mat: Material) -> Array:
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)

    C_inv = jnp.linalg.inv(C)
    S = mat.mu * (jnp.eye(2) - C_inv) + mat.lmbda * jnp.log(J) * C_inv  # 2nd PK
    P = F @ S  # 1st PK

    sigma = (P @ F.T) / J  # Cauchy
    return sigma


@autovmap(grad_u=2, mat=None)
def get_stress(grad_u: Array, mat: Material) -> Array:
    # 2nd Piola-Kirchhoff stress tensor
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)
    C_inv = jnp.linalg.inv(C)
    S = mat.mu * (jnp.eye(2) - C_inv) + mat.lmbda * jnp.log(J) * C_inv  # 2nd PK
    return S


def von_mises_stress(sig):
    s_xx, s_yy, s_xy = sig[..., 0, 0], sig[..., 1, 1], sig[..., 0, 1]
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)


def test_tracer_fem():
    n = 20
    mesh = Mesh.unit_square(n, n)

    class Solution(Compound, mesh=mesh):
        u = field((-1, 2))

    bottom = np.where(np.isclose(mesh.coords[:, 1], 0))[0]
    top = np.where(np.isclose(mesh.coords[:, 1], 1))[0]
    right = np.where(mesh.coords[:, 0] == 1)[0]
    left = np.where(mesh.coords[:, 0] == 0)[0]
    corner_0 = np.where((mesh.coords[:, 0] == 0) & (mesh.coords[:, 1] == 0))[0]

    lifter = Lifter(
        mesh.coords.shape[0] * 2,
        Fixed(Solution.u[corner_0]),
        Periodic(Solution.u[right, :], Solution.u[left, :]),
        Periodic(Solution.u[top, :], Solution.u[bottom, :]),
    )

    def energy_functional(
        z: Array,  # flat array of reduced dofs
        op: Operator,  # fem operator
        lifter: Lifter,  # lifting operator
        mat: Material,
    ) -> Array:
        z_full = lifter.lift_from_zeros(z)  # lift operation
        (u,) = Solution(z_full)  # reshape flat array into fields
        grad_u = op.grad(u)
        psi = strain_energy_density(grad_u, mat)
        return op.integrate(psi)

    op = Operator(mesh, Tri3())
    mat = Material.from_youngs_poisson_2d(E=2e3, nu=0.3)

    cap = CapturedJaxpr.from_fn(
        energy_functional,
        jnp.zeros(lifter.size_reduced),
        op=op,
        lifter=lifter,
        mat=mat,
    )

    # profile this, and must run
    result = trace(cap)
    # inspect hessian
    print(result.hessian.nnz)
