import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax import Array
from jax_autovmap import autovmap

from tatva import Operator, element, Mesh, sparse

jax.config.update("jax_enable_x64", True)


from typing import NamedTuple


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  # Shear modulus
    lmbda: float  # First Lamé parameter


mat = Material(mu=0.5, lmbda=1.0)


@autovmap(grad_u=2)
def compute_strain(grad_u: Array) -> Array:
    """Compute the strain tensor from the gradient of the displacement."""
    return 0.5 * (grad_u + grad_u.T)


@autovmap(eps=2, mu=0, lmbda=0)
def compute_stress(eps: Array, mu: float, lmbda: float) -> Array:
    """Compute the stress tensor from the strain tensor."""
    I = jnp.eye(2)
    return 2 * mu * eps + lmbda * jnp.trace(eps) * I


@autovmap(grad_u=2, mu=0, lmbda=0)
def strain_energy(grad_u: Array, mu: float, lmbda: float) -> Array:
    """Compute the strain energy density."""
    eps = compute_strain(grad_u)
    sig = compute_stress(eps, mat.mu, mat.lmbda)
    return 0.5 * jnp.einsum("ij,ij->", sig, eps)


mesh = Mesh.unit_square(8, 8)
tri = element.Tri3()
op = Operator(mesh, tri)


@jax.jit
def total_energy(u_flat):
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    energy_density = strain_energy(u_grad, 1.0, 0.0)
    return op.integrate(energy_density)


def test_sparse_matrix():

    K = jax.jacfwd(jax.jacrev(total_energy))(jnp.zeros(mesh.coords.shape[0] * 2))

    sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=2)


    K_sparse = sparse.jacfwd(jax.jacrev(total_energy), sparsity_pattern=sparsity_pattern)(
        jnp.zeros(mesh.coords.shape[0] * 2)
    )

    np.testing.assert_allclose(K, K_sparse.todense())