# femsolver
Functional programming and differentiable framework for finite element method (FEM) simulations.


## Overview

femsolver is a Python library for finite element method (FEM) simulations. It is built on top of JAX and Equinox, making it easy to use FEM in a differentiable way.


## Features

- Functional programming interface for FEM simulations
- Differentiable operations using JAX
- Support for linear, nonlinear, and mixed FEM simulations
- Easy integration with machine learning frameworks


## Installation

```bash
pip install -e .
```

## Usage

```python
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_platforms", "cpu")

from femsolver.quadrature import quad_tri3, shape_fn_tri3
from femsolver.operator import FemOperator
import matplotlib.pyplot as plt


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


def von_mises_stress(stress):
    s_xx, s_yy = stress[0, 0], stress[1, 1]
    s_xy = stress[0, 1]
    return jnp.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)


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


# --- Solver ---
def solve_fem():
    # --- Mesh ---
    coords, elements = generate_unit_square_mesh_tri(4, 2)
    n_nodes = coords.shape[0]
    n_dofs = 2 * n_nodes
    u = jnp.zeros(n_dofs)

    fem = FemOperator(
        compute_quads=quad_tri3,
        compute_shape_fn=shape_fn_tri3,
        compute_energy=linear_elasticity_energy,
    )


    # --- Total energy ---
    def total_energy(u_flat, coords, elements, fem):
        u = u_flat.reshape(-1, 2)
        u_cell = u[elements]
        x_cell = coords[elements]
        return jnp.sum(fem.integrate(u_cell, x_cell))

    grad_E = jax.grad(total_energy)
    hess_E = jax.jacfwd(jax.grad(total_energy))
    K = hess_E(u, coords, elements, fem)
    f_int = grad_E(u, coords, elements, fem)

    # Apply Dirichlet BCs
    left_nodes = jnp.where(jnp.isclose(coords[:, 0], 0.0))[0]
    right_nodes = jnp.where(jnp.isclose(coords[:, 0], 1.0))[0]
    fixed_dofs = jnp.concatenate(
        [
            2 * left_nodes,
            2 * left_nodes + 1,
            2 * right_nodes,
        ]
    )
    prescribed_values = jnp.zeros(n_dofs).at[2 * right_nodes].set(0.1)
    free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), fixed_dofs)

    f_ext = -f_int - K @ prescribed_values
    f_reduced = f_ext[free_dofs]
    K_reduced = K[jnp.ix_(free_dofs, free_dofs)]
    u_free = jnp.linalg.solve(K_reduced, f_reduced)
    u_full = prescribed_values.at[free_dofs].set(u_free)

    return coords, u_full.reshape(-1, 2), elements


# --- Compute von Mises stress per element ---
def compute_element_stress(coords, u, elements, fem):
    u_cells = u.reshape(-1, 2)[elements]
    coords_cells = coords[elements]

    def element_von_mises(u_e, x_e):
        qp, _ = quad_tri3()
        xi = qp[0]  # just take one point per element
        grad_u = fem.gradient(xi, u_e, x_e)
        eps = compute_strain(grad_u)
        sigma = compute_stress(eps)
        return von_mises_stress(sigma)

    return jax.vmap(element_von_mises)(u_cells, coords_cells)


# --- Visualization ---
def plot_displacement_and_stress(coords, u, elements, stress, scale=1.0):
    displaced = coords + scale * u
    tri_elements = elements

    plt.figure(figsize=(10, 5))
    plt.tripcolor(
        displaced[:, 0],
        displaced[:, 1],
        tri_elements,
        facecolors=stress,
        shading="flat",
        cmap="viridis",
    )
    plt.colorbar(label="Von Mises Stress")
    plt.title("Von Mises Stress on Deformed Mesh")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


coords, u, elements = solve_fem()
fem = FemOperator(quad_tri3, shape_fn_tri3, linear_elasticity_energy)

stress_vm = compute_element_stress(coords, u, elements, fem)
plot_displacement_and_stress(coords, u, elements, stress_vm)

```
