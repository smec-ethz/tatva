import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element, sparse

jax.config.update("jax_enable_x64", True)


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
    sig = compute_stress(eps, mu, lmbda)
    return 0.5 * jnp.einsum("ij,ij->", sig, eps)


@pytest.fixture(scope="module")
def op():
    mesh = Mesh.unit_square(8, 8)
    tri = element.Tri3()
    return Operator(mesh, tri)


def test_sparse_matrix(op: Operator):
    @jax.jit
    def total_energy(u_flat):
        u = u_flat.reshape(-1, 2)
        u_grad = op.grad(u)
        energy_density = strain_energy(u_grad, 1.0, 0.0)
        return op.integrate(energy_density)

    K = jax.jacfwd(jax.jacrev(total_energy))(jnp.zeros(op.mesh.coords.shape[0] * 2))

    sparsity_pattern = sparse.create_sparsity_pattern(op.mesh, n_dofs_per_node=2)

    K_sparse = sparse.jacfwd(
        jax.jacrev(total_energy), sparsity_pattern=sparsity_pattern
    )(jnp.zeros(op.mesh.coords.shape[0] * 2))

    np.testing.assert_allclose(K, K_sparse.todense())
