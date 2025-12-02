# %% [markdown]
# ## Sparse solvers {#sec-sparse-solvers}
#

# %%
# | code-fold: true
# | code-summary: "Code: Define mesh for square domain"
# | fig-align: center
# | fig-cap: "Sparsity pattern of the stiffness matrix for a 5 $\\times$ 5 mesh. Blue dots indicate the non-zero entries of the stiffness matrix. The region in white indicates the zero entries of the stiffness matrix."
# | label: fig-sparsity-pattern

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_platforms", "cpu")
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element, sparse
from tatva.experimental.assembler import assemble


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


mesh = Mesh.unit_square(5, 5)
tri = element.Tri3()
op = Operator(mesh, tri)


@jax.jit
def total_energy(u_flat: Array, op: Operator) -> Array:
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    energy_density = strain_energy(u_grad, 1.0, 0.0)
    return op.integrate(energy_density)


K = jax.jacfwd(jax.jacrev(total_energy))(jnp.zeros(mesh.coords.shape[0] * 2), op)


# %%
u = jnp.zeros(mesh.coords.shape[0] * 2)
op.grad(u)

# %%
K_sparse = assemble(
    total_energy_fn=total_energy,
    operators={"op": op},
    nodal_values_flat=jnp.zeros(mesh.coords.shape[0] * 2),
)
# %%

K

# %%

jnp.allclose(K_sparse.todense(), K)
# %%

mesh = Mesh.unit_square(5, 5)
right_element_indices = jnp.where(
    jnp.mean(mesh.coords[mesh.elements], axis=1)[:, 0] > 0.5
)[0]
left_element_indices = jnp.setdiff1d(
    jnp.arange(0, mesh.elements.shape[0]), right_element_indices
)

left_elements = mesh.elements[left_element_indices]
right_elements = mesh.elements[right_element_indices]

left_nodes = jnp.unique(left_elements.flatten())
right_nodes = jnp.unique(right_elements.flatten())

op_left = Operator(Mesh(coords=mesh.coords, elements=left_elements), tri)
op_right = Operator(Mesh(coords=mesh.coords, elements=right_elements), tri)

print("Left elements:", left_nodes)
print("Right elements:", right_nodes)
# %%


@jax.jit
def total_energy_regions(u_flat, op_left, op_right):
    u = u_flat.reshape(-1, 2)
    u_grad_left = op_left.grad(u)
    energy_density_left = strain_energy(u_grad_left, 1.0, 0.0)
    u_grad_right = op_right.grad(u)
    energy_density_right = strain_energy(u_grad_right, 1.0, 0.0)
    _total_energy = op_left.integrate(energy_density_left) + op_right.integrate(
        energy_density_right
    )
    return _total_energy


# %%
K_regions = jax.jacfwd(jax.jacrev(total_energy_regions))(
    jnp.zeros(mesh.coords.shape[0] * 2), op_left, op_right
)
# %%

K_sparse_regions = assemble(
    total_energy_fn=total_energy_regions,
    operators={"op_left": op_left, "op_right": op_right},
    nodal_values_flat=jnp.zeros(mesh.coords.shape[0] * 2),
)


# %%

jnp.allclose(K_regions, K_sparse_regions.todense())

# %%


#
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


# %%


def test_staggered_assembly_fixed_field():
    """
    Tests assembly of K_uu where T (or phi) is a FIXED field passed to energy.
    Verifies that the assembler correctly injects operators to handle the fixed field.
    """
    # 1. Setup
    mesh = Mesh.unit_square(4, 4)
    tri = element.Tri3()

    op_u = Operator(mesh, tri)
    op_phi = Operator(mesh, tri)  # Same mesh

    n_dofs_u = mesh.coords.shape[0] * 2
    n_dofs_phi = mesh.coords.shape[0] * 1

    # 2. Fixed Field (Phi)
    # Create a random field to make gradients non-zero
    key = jax.random.PRNGKey(0)
    phi_fixed = jax.random.uniform(key, (n_dofs_phi,)).flatten()

    # 3. Energy Function
    # Note: 'u' is active (local element vector), 'phi' is fixed (global vector)
    def total_energy(u_flat, op_u, op_phi):
        # Active Operator (u): Computes gradients locally
        u_local = u_flat.reshape(-1, 2)
        grad_u = op_u.grad(u_local)

        # Passive Operator (phi): Projects/Slices fixed global field
        # We pass the GLOBAL phi vector here.
        # op_phi must be smart enough to slice it using the context index.
        # grad_phi = op_phi.grad(phi_fixed)
        val_phi = op_phi.eval(phi_fixed)

        # Map kernel
        densities = coupled_energy_density(grad_u, val_phi)

        return op_u.integrate(densities)

    # 4. Partial Application
    # We fix 'phi_fixed' via the closure or explicit partial, but here
    # we pass it inside the function.

    # 5. Assemble K_uu
    u_zero = jnp.zeros(n_dofs_u)

    # We ONLY pass op_u in the dict because we only want to differentiate w.r.t u.
    # However, total_energy needs op_phi.
    # So we MUST pass op_phi in the dict so it gets injected.
    # Does this mean assemble will try to differentiate w.r.t u using op_phi?
    # Yes, but op_phi loop will compute derivatives of "op_phi.integrate(...)".
    # Our energy function DOES NOT return op_phi.integrate(...).
    # So op_phi loop will return a zero matrix (Hessian of 0 is 0).
    # This is safe!

    operators = {"op_u": op_u, "op_phi": op_phi}

    K_sparse = assemble(total_energy, operators, u_zero)

    print("K_sparse:", K_sparse.todense())

    K_dense = jax.jacfwd(jax.jacrev(total_energy))(u_zero, op_u, op_phi)

    print("K_dense:", K_dense)

    print(jnp.allclose(K_sparse.todense(), K_dense, atol=1e-12))


# %%

test_staggered_assembly_fixed_field()

# and %% [markdown]
# ::: {.callout-note}
# The non-zero entries of the stiffness matrix are concentrated around a diagonal band is indicated by blue dots in the figure above. The zero entries are indicated by white dots.
# :::
#
# %%
