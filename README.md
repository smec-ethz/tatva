# femsolver
Functional programming and differentiable framework for finite element method (FEM) simulations. `femsolver` is a Python library for finite element method (FEM) simulations. It is built on top of JAX and Equinox, making it easy to use FEM in a differentiable way.


## Features

- Functional programming interface for FEM simulations
- Differentiable operations using JAX
- Support for linear, nonlinear, and mixed FEM simulations


## Installation

We recommend to create a python virtual environment and then install the library.

```bash
pip install -e .
```

## Usage

The basic usage of the library is shown below for a linear elastic case. 

```python
import jax
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_platforms", "cpu")

from femsolver.quadrature import quad_tri3, shape_fn_tri3
from femsolver.operator import FemOperator
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
```

We define a function to generate mesh (coords and cells). Here we will use a square plate with triangular finite elements.

```python
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
```

Next, we define a function to compute the linear elastic energy density based on the displacement gradients $\nabla u$.

$$
\Psi(x) =  \sigma(x) : \epsilon(x) 
$$

where $\sigma$ is the stress tensor and $\epsilon$ is the strain tensor.

$$
\sigma = \lambda \text{tr}(\epsilon) I + 2\mu \epsilon
$$

and 

$$
\epsilon = \frac{1}{2} (\nabla u + \nabla u^T)
$$

```python
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
```

The femsolver provides a generic class `FemOperator` that can be used to solve FEM problems. This operator is the core of the library as it provide functions that can automatically integrate the energy density function defined above over the range of elements.

It takes three arguments:

- `compute_quads`: a function that returns the quadrature points and weights for the elements
- `compute_shape_fn`: a function that returns the shape functions for the elements
- `compute_energy`: a function that returns the energy density for the elements

```python
fem = FemOperator(quad_tri3, shape_fn_tri3, linear_elasticity_energy)
```

In the above definition of the ``FemOperator`` class, we have used the ``quad_tri3`` and ``shape_fn_tri3`` functions to compute the quadrature points and shape functions for the triangular elements.

One can simply replace these two functions with any other quadrature and shape function. Just look at the ``quad_tri3`` and ``shape_fn_tri3`` functions in ``femsolver/quadrature.py`` to see how to define your own.

For more complex problems, one can define their own implementation of the `FemOperator` class. One just have to inherit from the `FemOperator` class and override the functions that are needed.

For example, if we want to solve a problem with a history dependent material model, we can define a new class that inherits from the `FemOperator` class and overrides the integration functions.

```python
class HistoryDependentElasticityOperator(FemOperator):
    def integrate(self, nodal_values, nodes, history_variables):
        qp, w  = self.funcs["quads"]()

        def integrand(xi, wi, history):
            N, dNdr = self.funcs["shape_fn"](xi)
            J = dNdr @ nodes
            u_grad = self.gradient(xi, nodal_values, nodes)
            energy = self.funcs["energy"](u_grad, history)  
            return wi * energy * jnp.linalg.det(J)

        return jnp.sum(jax.vmap(integrand)(qp, w, history_variables))
```

For full implementation of such complex examples, please refere to the `examples/` directory.




```python
# --- Mesh ---
coords, elements = generate_unit_square_mesh_tri(10, 10)
n_nodes = coords.shape[0]
n_dofs_per_node = 2
n_dofs = n_dofs_per_node * n_nodes
u = jnp.zeros(n_dofs)


# --- Total energy ---
def total_energy(u_flat, coords, elements, fem):
    u = u_flat.reshape(-1, n_dofs_per_node)
    u_cell = u[elements]
    x_cell = coords[elements]
    return jnp.sum(fem.integrate(u_cell, x_cell))


# creating functions to compute the gradient and 
# Hessian of the total energy using jax
grad_E = jax.grad(total_energy)
hess_E = jax.jacfwd(jax.grad(total_energy))

# compute the hessian which is the stiffness matrix 
# and the gradient which is the internal force vector 
K = hess_E(u, coords, elements, fem)
f_int = grad_E(u, coords, elements, fem)

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
prescribed_values = jnp.zeros(n_dofs).at[2 * right_nodes].set(0.3)
free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), fixed_dofs)

# --- Solve for the displacement ---
f_ext = -f_int - K @ prescribed_values
f_reduced = f_ext[free_dofs]
K_reduced = K[jnp.ix_(free_dofs, free_dofs)]
u_free = jnp.linalg.solve(K_reduced, f_reduced)
u_full = prescribed_values.at[free_dofs].set(u_free)
```

For visualization, we can now compute the von Mises stress on the deformed mesh.

```python
def von_mises_stress(stress):
    s_xx, s_yy = stress[0, 0], stress[1, 1]
    s_xy = stress[0, 1]
    return jnp.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)


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


# --- Compute the stress ---    
u = u_full.reshape(-1, n_dofs_per_node)

stress_vm = compute_element_stress(coords, u, elements, fem)

# --- Plot the displacement and stress ---
plot_displacement_and_stress(coords, u, elements, stress_vm)

```

![Von Mises Stress on Deformed Mesh](examples/notebooks/linear_elasticity.png)


## Dense vs Sparse 

The example shown above creates a dense stiffness matrix which for understanding is great but is memory intensive. A sparse framework for the same example is provided in `examples/notebooks/sparse_linear_elasticity.ipynb`. For sparse representation of the stiffness matrix or the hessian of the total energy, we use the library `sparsejac` that allows automatic differentiation of a functional based on a sparsity pattern. This significantly reduces the memory consumption. For more details on how the automatic differentiation can be done using sparsity pattern, please check the link below:

![Paper](https://arxiv.org/html/2501.17737v1)</br>
![Github: sparsejac](https://github.com/mfschubert/sparsejac)</br>
![Github: Sparsediffax, python interface for the paper](https://github.com/gdalle/sparsediffax)</br>

## Profiling

### Time usage profiling

Below we provide the computational time for the assembly of the sparse stiffness matrix for linear elasticity problem. The code is available in `benchmarking/profiling_time.py`.


![Assembly Time](benchmarking/assembly_time_cpu.png)

The time above doesnot account for the compilation time  of the functions. In JAX, the first time a function is called, it is compiled and repeated calls are faster. This compilation time is not included in the time above. The time above is for a single core of a CPU. 



### Memory usage profiling

We use pprof to profile the memory usage. Please follow the instruction on JAX's documentation on profiling  ![Link to documentation](https://docs.jax.dev/en/latest/device_memory_profiling.html). Using `go` and `pprof`.

For 20000 degrees of freedom and a sparse linear elastic framework (`benchmarking/profiling_memory_usage.py`), a total of `15 MB` memory is used on `CPU`. The distribution of memory usage is as follows:

![](benchmarking/profile001.svg)

