# %%

import jax
from jax._src.interpreters.batching import MeshAxis

jax.config.update("jax_enable_x64", True)  # use double-precision
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element, sparse
from tatva.experimental.assembler import assemble


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  # Shear modulus
    lmbda: float  # First Lamé parameter


mat = Material(mu=0.5, lmbda=1.0)

# %%
# Define Single Operator and Energy Function


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
op = Operator(mesh, tri, nb_local_dofs=2)


@jax.jit
def total_energy(u_flat: Array, op: Operator) -> Array:
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    energy_density = strain_energy(u_grad, 1.0, 0.0)
    return op.integrate(energy_density)


key = jax.random.PRNGKey(0)
n_dofs_u = mesh.coords.shape[0] * 2
u_fixed = jax.random.uniform(key, (n_dofs_u,)).flatten()


K = jax.jacfwd(jax.jacrev(total_energy))(u_fixed, op)

K_sparse = assemble(
    total_energy_fn=total_energy,
    operators={"op": op},
    nodal_values_flat=u_fixed,
)

jnp.allclose(K_sparse.todense(), K)


# %%
# Example: Two regions with different operators

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

op_left = Operator(
    Mesh(coords=mesh.coords, elements=left_elements), tri, nb_local_dofs=2
)
op_right = Operator(
    Mesh(coords=mesh.coords, elements=right_elements), tri, nb_local_dofs=2
)

print("Left elements:", left_nodes)
print("Right elements:", right_nodes)


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


K_sparse_regions = assemble(
    total_energy_fn=total_energy_regions,
    operators={"op_left": op_left, "op_right": op_right},
    nodal_values_flat=jnp.zeros(mesh.coords.shape[0] * 2),
)


jnp.allclose(K_regions, K_sparse_regions.todense())


# %%
# Example: Coupled energy with fixed phase-field


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
    strain_energy = 1.0 * jnp.trace(eps @ eps) + 0.5 * 1.0 * trace_eps**2

    # Phase field degradation function
    degradation = phi_val**2 + 0.001

    return strain_energy * degradation


def test_staggered_assembly_fixed_field():
    """
    Tests assembly of K_uu where T (or phi) is a FIXED field passed to energy.
    Verifies that the assembler correctly injects operators to handle the fixed field.
    """
    # 1. Setup
    mesh = Mesh.unit_square(4, 4)
    tri = element.Tri3()

    op_u = Operator(mesh, tri, nb_local_dofs=2)
    op_phi = Operator(mesh, tri, nb_local_dofs=1)  # Same mesh

    n_dofs_u = mesh.coords.shape[0] * 2
    n_dofs_phi = mesh.coords.shape[0] * 1

    # 2. Fixed Field (Phi)
    # Create a random field to make gradients non-zero
    key = jax.random.PRNGKey(0)
    phi_fixed = jax.random.uniform(key, (n_dofs_phi,)).flatten()

    u_zero = jax.random.uniform(key, (n_dofs_u,)).flatten()

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
        # jax.debug.print("phi_fixed: {phi_fixed}", phi_fixed=phi_fixed)
        val_phi = op_phi.eval(phi_fixed)
        # jax.debug.print("op_phi context {x}", x=op_phi._ctx_mode)
        # jax.debug.print("op_u context {x}", x=op_u._ctx_mode)
        # jax.debug.print("val_phi: {val_phi}", val_phi=val_phi)

        # Map kernel
        densities = coupled_energy_density(grad_u, val_phi)

        return op_u.integrate(densities)

    # 4. Partial Application
    # We fix 'phi_fixed' via the closure or explicit partial, but here
    # we pass it inside the function.

    # 5. Assemble K_uu

    # We ONLY pass op_u in the dict because we only want to differentiate w.r.t u.
    # However, total_energy needs op_phi.
    # So we MUST pass op_phi in the dict so it gets injected.
    # Does this mean assemble will try to differentiate w.r.t u using op_phi?
    # Yes, but op_phi loop will compute derivatives of "op_phi.integrate(...)".
    # Our energy function DOES NOT return op_phi.integrate(...).
    # So op_phi loop will return a zero matrix (Hessian of 0 is 0).
    # This is safe!

    operators = {"op_u": op_u, "op_phi": op_phi}

    K_sparse = assemble(total_energy, operators=operators, nodal_values_flat=u_zero)

    # print("K_sparse:", K_sparse.todense())

    K_dense = jax.jacfwd(jax.jacrev(total_energy))(u_zero, op_u, op_phi)

    # print("K_dense:", K_dense)

    print(jnp.allclose(K_sparse.todense(), K_dense, atol=1e-12))
    #
    # print(op_phi.eval(phi_fixed))


# %%

test_staggered_assembly_fixed_field()

# %%
# Example: Assembling by manually extracting indices for global DOFs
mesh = Mesh.unit_square(5, 5)
coords = mesh.coords.at[:, 1].add(0.2)
mesh = Mesh(coords, mesh.elements)
nb_dofs_per_node = 2
n_dofs_u = nb_dofs_per_node * mesh.coords.shape[0]

nodal_values = mesh.coords
y_max = jnp.max(nodal_values[:, 1])
y_min = jnp.min(nodal_values[:, 1])
top_nodes = jnp.where(nodal_values[:, 1] == y_max)[0]
bottom_nodes = jnp.where(nodal_values[:, 1] == y_min)[0]

k_pen = 1e3


@jax.jit
def macaulay_bracket(x):
    return jnp.where(x > 0, 0, x)


@jax.jit
def compute_contact_energy(
    u: Array,
    coords: Array,
    contact_nodes: Array,
    op: Operator,
) -> Array:
    """Compute the contact energy for a given displacement field.
    Args:
        u: Displacement field.
        coords: Coordinates of the nodes.
        contact_nodes: Indices of the nodes on the contact surface.
    Returns:
        Contact energy.
    """
    u_nodes = op.slice(u, global_indices=contact_nodes)
    x_nodes = op.slice(coords, global_indices=contact_nodes)

    # jax.debug.print("u_nodes: {u_nodes}", u_nodes=u_nodes)
    # jax.debug.print("x_nodes: {x_nodes}", x_nodes=x_nodes)

    # Loop over nodes on the potential contact surface
    def _contact_energy_node(
        u_node: tuple[float, float], x_node: tuple[float, float]
    ) -> float:
        gap = (x_node[1] + u_node[1]) - 0.0
        penetration = macaulay_bracket(gap)
        return 0.5 * k_pen * (penetration**2)

    contact_energy_node = jax.vmap(_contact_energy_node, in_axes=(0, 0))

    return jnp.sum(contact_energy_node(u_nodes, x_nodes))


@autovmap(grad_u=2, mu=0, lmbda=0)
def compute_strain_energy_density(grad_u: Array, mu: float, lmbda: float) -> Array:
    """Compute the strain energy density."""
    eps = compute_strain(grad_u)
    sig = compute_stress(eps, mu, lmbda)
    return 0.5 * jnp.einsum("ij,ij->", sig, eps)


@jax.jit
def _total_energy(
    u_flat: Array,
    op: Operator,
    coords: Array,
    contact_nodes: Array,
) -> Array:
    """Compute the total energy for a given displacement field.
    Args:
        u_flat: Flattened displacement field.
        coords: Coordinates of the nodes.
        contact_nodes: Indices of the nodes on the contact surface.
        nodes_area: Area associated with all the nodes in the mesh
    Returns:
        Total energy.
    """
    u = u_flat.reshape(-1, nb_dofs_per_node)
    contact_energy = compute_contact_energy(u, coords, contact_nodes, op)
    u_grad = op.grad(u)
    strain_energy_density = compute_strain_energy_density(u_grad, 1.0, 0.5)
    strain_energy = op.integrate(strain_energy_density)
    return strain_energy + contact_energy


partial_total_energy = partial(
    _total_energy,
    coords=mesh.coords,
    contact_nodes=top_nodes,
)

operators = {"op": op}

key = jax.random.PRNGKey(0)
u_zero = jax.random.uniform(key, (n_dofs_u,)).flatten()


K_dense = jax.jacfwd(jax.jacrev(partial_total_energy))(u_zero, op)


# %%
K_sparse = assemble(partial_total_energy, operators=operators, nodal_values_flat=u_zero)
print(jnp.allclose(K_sparse.todense(), K_dense, atol=1e-12))
