import os

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_platforms", "cpu")

from femsolver.quadrature import quad_tri3, shape_fn_tri3
from femsolver.operator import FemOperator
import jax.numpy as jnp
import jax.profiler
import sparsejac
import numpy as np

import time

# --- Mesh generation ---


def generate_unit_square_mesh_tri(nx, ny):
    x = jnp.linspace(0, 1, nx + 1)
    y = jnp.linspace(0, 1, ny + 1)
    xv, yv = jnp.meshgrid(x, y, indexing="ij")
    coords = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)

    def node_id(i, j):
        return i * (ny + 1) + j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    return coords, jnp.array(elements)


# --- Material model (linear elasticity: plane strain) ---
def compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


def compute_stress(eps, mu=1.0, lmbda=1.0):
    I = jnp.eye(2)
    return 2 * mu * eps + lmbda * jnp.trace(eps) * I


def linear_elasticity_energy(grad_u, mu=1.0, lmbda=1.0):
    eps = compute_strain(grad_u)
    sigma = compute_stress(eps, mu, lmbda)
    return 0.5 * jnp.sum(sigma * eps)


start_time = time.time()
fem = FemOperator(quad_tri3, shape_fn_tri3, linear_elasticity_energy)
print(time.time() - start_time)
# --- Mesh ---
start_time = time.time()
coords, elements = generate_unit_square_mesh_tri(100, 100)
n_nodes = coords.shape[0]
n_dofs_per_node = 2
n_dofs = n_dofs_per_node * n_nodes
u = jnp.zeros(n_dofs)
print(time.time() - start_time)


# --- Total energy ---
def total_energy(u_flat, coords, elements, fem):
    u = u_flat.reshape(-1, n_dofs_per_node)
    u_cell = u[elements]
    x_cell = coords[elements]
    return jnp.sum(fem.integrate(u_cell, x_cell))


# creating functions to compute the gradient of total energy using jax
grad_E = jax.grad(total_energy)
# --- Apply Dirichlet BCs ---
left_nodes = jnp.where(jnp.isclose(coords[:, 0], 0.0))[0]
right_nodes = jnp.where(jnp.isclose(coords[:, 0], 1.0))[0]
fixed_dofs = jnp.concatenate(
    [
        2 * left_nodes,
        2 * left_nodes + 1,
        2 * right_nodes,
    ]
)
prescribed_dofs = jnp.concatenate(
    [
        2 * right_nodes,
    ]
)
prescibed_disp = 0.3
prescribed_values = jnp.zeros(n_dofs).at[prescribed_dofs].set(prescibed_disp)
free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), fixed_dofs)


def create_sparse_structure(elements, nstate, K_shape):
    # elements: (num_elements, nodes_per_element)
    elements = jnp.array(elements)
    num_elements, nodes_per_element = elements.shape
    # Compute all (i, j, k, l) combinations for each element
    i_idx = jnp.repeat(
        elements, nodes_per_element, axis=1
    )  # (num_elements, nodes_per_element^2)
    j_idx = jnp.tile(
        elements, (1, nodes_per_element)
    )  # (num_elements, nodes_per_element^2)
    # Expand for nstate
    k_idx = jnp.arange(nstate)
    l_idx = jnp.arange(nstate)
    k_idx, l_idx = jnp.meshgrid(k_idx, l_idx, indexing="ij")
    k_idx = k_idx.flatten()
    l_idx = l_idx.flatten()

    # For each element, get all (row, col) indices
    def element_indices(i, j):
        row = nstate * i + k_idx
        col = nstate * j + l_idx
        return row, col

    # Vectorize over all (i, j) pairs for all elements
    row_idx, col_idx = jax.vmap(element_indices)(i_idx.flatten(), j_idx.flatten())
    # Flatten and clip to matrix size
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    mask = (row_idx < K_shape[0]) & (col_idx < K_shape[1])
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    # Create the sparse structure
    K = jnp.zeros(K_shape, dtype=jnp.int32)
    K = K.at[row_idx, col_idx].set(1)
    indices = np.unique(np.vstack((row_idx, col_idx)).T, axis=0)
    return np.ones(indices.shape[0]), indices


start_time = time.time()
data, indices = create_sparse_structure(elements, n_dofs_per_node, (n_dofs, n_dofs))
sparsity_pattern = jax.experimental.sparse.BCOO((data, indices), shape=(n_dofs, n_dofs))
hess_E_sparse = sparsejac.jacfwd(grad_E, sparsity=sparsity_pattern)
print(time.time() - start_time)
start_time = time.time()
K_sparse = hess_E_sparse(u, coords, elements, fem)
print(time.time() - start_time)
start_time = time.time()
K_sparse = hess_E_sparse(u, coords, elements, fem)
print(time.time() - start_time)
K_bc_data = K_sparse.data
for dof in fixed_dofs:
    indexes = np.where(indices[:, 0] == dof)[0]
    for idx in indexes:
        K_bc_data = K_bc_data.at[idx].set(0)
    idx = np.where(np.all(indices == np.array([dof, dof]), axis=1))[0]
    K_bc_data = K_bc_data.at[idx].set(1)
for dof in prescribed_dofs:
    for idx in np.where(indices[:, 0] == dof)[0]:
        K_bc_data = K_bc_data.at[idx].set(0)
    idx = np.where(np.all(indices == np.array([dof, dof]), axis=1))[0]
    K_bc_data = K_bc_data.at[idx].set(1)

jax.profiler.save_device_memory_profile("memory-100.prof")
