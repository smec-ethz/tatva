import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element, sparse
from tatva.experimental.assembler import TotalEnergyCallable, assemble


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


def test_assembler_single_operator(op: Operator):
    @jax.jit
    def total_energy(u_flat: Array, op: Operator):
        u = u_flat.reshape(-1, 2)
        u_grad = op.grad(u)
        energy_density = strain_energy(u_grad, 1.0, 0.0)
        return op.integrate(energy_density)

    u = jnp.zeros(op.mesh.coords.shape[0] * 2)

    K = jax.jacfwd(jax.jacrev(total_energy))(u, op)

    K_sparse = assemble(
        total_energy_fn=total_energy,
        operators={"op": op},
        nodal_values_flat=u,
    )

    np.testing.assert_allclose(K, K_sparse.todense())


@pytest.fixture(scope="module")
def ops():
    mesh = Mesh.unit_square(8, 8)
    tri = element.Tri3()
    right_element_indices = jnp.where(
        jnp.mean(mesh.coords[mesh.elements], axis=1)[:, 0] > 0.5
    )[0]
    left_element_indices = jnp.setdiff1d(
        jnp.arange(0, mesh.elements.shape[0]), right_element_indices
    )

    left_elements = mesh.elements[left_element_indices]
    right_elements = mesh.elements[right_element_indices]

    op_left = Operator(Mesh(coords=mesh.coords, elements=left_elements), tri)
    op_right = Operator(Mesh(coords=mesh.coords, elements=right_elements), tri)

    return {"op_left": op_left, "op_right": op_right}


def test_assembler_multi_operator(ops: dict[str, Operator]):
    op_left = ops["op_left"]
    op_right = ops["op_right"]

    @jax.jit
    def total_energy(u_flat: Array, op_left: Operator, op_right: Operator):
        u = u_flat.reshape(-1, 2)
        u_grad_left = op_left.grad(u)
        energy_density_left = strain_energy(u_grad_left, 1.0, 0.0)
        u_grad_right = op_right.grad(u)
        energy_density_right = strain_energy(u_grad_right, 1.0, 0.0)
        _total_energy = op_left.integrate(energy_density_left) + op_right.integrate(
            energy_density_right
        )
        return _total_energy

    u = jnp.zeros(op_left.mesh.coords.shape[0] * 2)

    K = jax.jacfwd(jax.jacrev(total_energy))(u, op_left, op_right)

    K_sparse = assemble(
        total_energy_fn=total_energy,
        operators={"op_left": op_left, "op_right": op_right},
        nodal_values_flat=u,
    )

    np.testing.assert_allclose(K, K_sparse.todense())


@autovmap(grad_u=2, phi_val=0)
def coupled_energy_density(grad_u, phi_val):
    """
    Energy depending on displacement u and phase-field phi.
    E = 0.5 * (phi^2 + epsilon) * strain : strain
    This creates a coupling where stiffness depends on phi.
    """
    # Linear Strain
    eps = 0.5 * (grad_u + grad_u.T)
    trace_eps = jnp.trace(eps)
    strain_energy = 1.0 * jnp.trace(eps @ eps) + 0.5 * 0.0 * trace_eps**2

    # Phase field degradation function
    degradation = phi_val**2 + 0.001

    return strain_energy * degradation


@pytest.fixture(scope="module")
def coupled_ops():
    mesh = Mesh.unit_square(8, 8)
    tri = element.Tri3()
    return Operator(mesh, tri), Operator(mesh, tri)


def test_assembler_coupled_operators(coupled_ops: tuple[Operator, Operator]):
    op_u, op_phi = coupled_ops

    n_dofs_u = op_u.mesh.coords.shape[0] * 2
    n_dofs_phi = op_phi.mesh.coords.shape[0] * 1

    # Create a random field to make gradients non-zero
    key = jax.random.PRNGKey(0)
    phi_fixed = jax.random.uniform(key, (n_dofs_phi,)).flatten()

    # Note: 'u' is active (local element vector), 'phi' is fixed (global vector)
    def total_elastic_energy(u_flat: Array, op_u: Operator, op_phi: Operator):
        # active Operator (u): Computes gradients locally
        u_local = u_flat.reshape(-1, 2)
        grad_u = op_u.grad(u_local)

        val_phi = op_phi.eval(phi_fixed)

        densities = coupled_energy_density(grad_u, val_phi)

        return op_u.integrate(densities)

    u_zero = jnp.zeros(n_dofs_u)

    operators = {"op_u": op_u, "op_phi": op_phi}

    K_sparse = assemble(total_elastic_energy, operators, u_zero)

    K_dense = jax.jacfwd(jax.jacrev(total_elastic_energy))(u_zero, op_u, op_phi)

    np.testing.assert_allclose(K_sparse.todense(), K_dense, atol=1e-12)
