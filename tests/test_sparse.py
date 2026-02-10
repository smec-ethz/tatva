import jax

jax.config.update("jax_enable_x64", True)  # use double-precision

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax import Array
from jax_autovmap import autovmap
from tatva_coloring import distance2_color_and_seeds

from tatva import Mesh, Operator, element, sparse

jax.config.update("jax_enable_x64", True)


COLORING_VARIANTS = [
    distance2_color_and_seeds,
    # Add other coloring functions here if needed
]


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


@pytest.mark.parametrize("coloring_func", COLORING_VARIANTS)
def test_sparse_matrix(op: Operator, coloring_func):
    @jax.jit
    def total_energy(u_flat):
        u = u_flat.reshape(-1, 2)
        u_grad = op.grad(u)
        energy_density = strain_energy(u_grad, 1.0, 0.0)
        return op.integrate(energy_density)

    n_dofs = op.mesh.coords.shape[0] * 2

    K = jax.jacfwd(jax.jacrev(total_energy))(jnp.zeros(op.mesh.coords.shape[0] * 2))

    sparsity_pattern = sparse.create_sparsity_pattern(op.mesh, n_dofs_per_node=2)
    sparsity_pattern_csr = sp.csr_matrix(
        (
            sparsity_pattern.data,
            (sparsity_pattern.indices[:, 0], sparsity_pattern.indices[:, 1]),
        )
    )
    indptr = sparsity_pattern_csr.indptr
    indices = sparsity_pattern_csr.indices

    colors, seeds = coloring_func(row_ptr=indptr, col_idx=indices, n_dofs=n_dofs)

    K_sparse = sparse.jacfwd(
        gradient=jax.jacrev(total_energy),
        row_ptr=indptr,
        col_indices=indices,
        colors=colors,
    )(jnp.zeros(op.mesh.coords.shape[0] * 2))

    np.testing.assert_allclose(K, K_sparse.todense())
