# femsolver
Functional programming and differentiable framework for finite element method (FEM) simulations. `femsolver` is a Python library for finite element method (FEM) simulations. It is built on top of JAX and Equinox, making it easy to use FEM in a differentiable way.


## Features

- Functional programming interface for FEM simulations
- Differentiable operations using JAX
- Support for linear, nonlinear, and mixed FEM simulations


## Installation

Clone the repository and install the package with pip:
```bash
pip install path/to/femsolver
```

You can also use pip to install directly from the GitLab repository. Make sure
you have access to the repository and have set up SSH keys for authentication.
```bash
pip install git+ssh://git@gitlab.ethz.ch/compmechmat/research/mohit-pundir/femsolver.git
```

> [!note]
> We strongly recommend to always use a virtual environment. We further
> recommend using [uv](https://docs.astral.sh/uv/).

## Roadmap

- [x] Add example for linear elasticity (**Mohit**)
- [x] Add example for nonlinear elasticity (**Mohit**)
- [ ] Add example for Dirichlet BCs as constraints (**Mohit**)
- [ ] Add example for matrix-free solvers (with Dirichlet BCs) (**Mohit**)
- [x] Add example for contact problems with penalty method (**Mohit**)
- [ ] Add example for contact problems with Lagrange multipliers (**Flavio**)
- [ ] Add example for contact problems with augmented Lagrangian method (**Flavio**)
- [x] Add example for cohesive fracture problems (**Mohit**)
- [ ] Add example for cohesive fracture problems in dynamics (**Mohit**)
- [ ] Add example for thermal-mechanical coupled problems (**Mohit**)
- [ ] Add example for phase-field fracture coupled problems (**Mohit**)


## Usage

The basic usage of the library is shown below for a linear elastic case. 

```python
import jax
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_platforms", "cpu")

from femsolver.quadrature import get_element, Element
from femsolver.operator import Operator
from femsolver.jax_utils import auto_vmap 
import jax.numpy as jnp
import equinox as eqx

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

To ease the handling of the material paramters and later on ease the integration of the material parameters for computing energy density, we define a class `Material` that can be used to define the material parameters.

```python
class Material(eqx.Module):
    mu: eqx.AbstractVar[float] 
    lmbda: eqx.AbstractVar[float]
```
Now we can define the python functions to compute the strain, stress and energy density.

```python
# --- Material model (linear elasticity: plane strain) ---
@auto_vmap(grad_u=2)
def compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)

@auto_vmap(eps=2, mu=0, lmbda=0)
def compute_stress(eps, mu, lmbda):
    I = jnp.eye(2)
    return 2 * mu * eps + lmbda * jnp.trace(eps) * I

@auto_vmap(grad_u=2, mu=0, lmbda=0)
def linear_elasticity_energy(grad_u, mu, lmbda):
    eps = compute_strain(grad_u)
    sigma = compute_stress(eps, mu, lmbda)
    return 0.5 * jnp.sum(sigma * eps)

```
The femsolver provides a generic class `Operator` that can be used to solve FEM problems. This operator is the core of the library as it provide functions that can automatically integrate the energy density function defined above over the range of elements.

Below, we define a class `ElasticityOperator` that inherits from `Operator` and `Material`.

```python
class ElasticityOperator(Operator, Material):
    element: Element
    mu: float
    lmbda: float

    @auto_vmap(xi=1, wi=1, nodal_values=None, nodes=None)
    def integrand(self, xi, wi, nodal_values, nodes):
        u_quad, u_grad, detJ = self.element.get_local_values(
            xi, nodal_values, nodes
        )
        value = linear_elasticity_energy(u_grad, self.mu, self.lmbda)
        return wi * value * detJ
```

The `integrand` function is the key function that is used to integrate the energy density over the range of elements. It is a function that takes the quadrature points, weights, nodal values and nodes as input and returns the energy density.

The `Operator` class relies on the `Element` type. The `Element` class is a generic class that can be used to define the element type. The `Element` class knows what quadrature rule to apply and what shape functions to use.  It provides a `get_local_values` function that can be used to compute the local values of the displacement field (and its gradient) at the quadrature points.

To see in detail the functionalities of `Element` class or define a new `Element` type, please refer to the file `femsolver/quadrature.py`.

For this example, we are using the `tri3` element. The `tri3` element is a triangular element with 3 nodes. The `tri3` element is defined in the file `femsolver/quadrature.py`.

```python
tri3 = get_element("tri3")
fem = ElasticityOperator(tri3, mu=1.0, lmbda=1.0)
```

One can simply replace this element with any other element. Just look at the `tri3` element in `femsolver/quadrature.py` to see how to define your own.

For more complex problems, one can define their own implementation of the `Operator` class. One just have to inherit from the `Operator` class and override the functions that are needed.

For example, if we want to solve a problem with a history dependent material model, we can define a new class that inherits from the `Operator` class and overrides the integration functions.

```python
class HistoryDependentOperator(Operator):
    element: Element
    @auto_vmap(xi=1, wi=1, nodal_values=None, nodes=None, history_variables=1)
    def integrate(self, xi, wi, nodal_values, nodes, history_variables):
        u_quad, u_grad, detJ = self.element.get_local_values(
            xi, nodal_values, nodes
        )
        value = self.integrand(u_grad, history_variables)
        return wi * value * detJ
```

For full implementation of such complex examples, please refere to the `examples/` directory.

Now we can define the mesh and the operator.

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
@auto_vmap(stress=2)
def von_mises_stress(stress):
    s_xx, s_yy = stress[0, 0], stress[1, 1]
    s_xy = stress[0, 1]
    return jnp.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)
```

We can now call the functions defined above to compute the strains and stresses for each element at its quadrature points.

```python
grad_us = fem.gradient(u_full.reshape(-1, n_dofs_per_node)[elements], coords[elements])
strains = compute_strain(grad_us)
stresses = compute_stress(strains, fem.mu, fem.lmbda)
stress_vm = von_mises_stress(stresses)
```

In the above example, we have used the `gradient` function to compute the gradient of the displacement field at the quadrature points. The `gradient` function is a function that takes nodal values (per cell) and nodes (per cell) as input and returns the gradient of the displacement field (per cell per quadrature point).

Thus, the shape of the `grad_us` is `(n_elements, n_quadrature_points, n_dofs_per_node, n_dofs_per_node)`.


```python
# --- Visualization ---
from femsolver.plotting import STYLE_PATH
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_displacement_and_stress(coords, u, elements, stress, scale=1.0):
    displaced = coords + scale * u
    tri_elements = elements
    
    plt.style.use(STYLE_PATH)
    fig =plt.figure(figsize=(5, 4))
    ax = plt.axes()
    cb =ax.tripcolor(
        displaced[:, 0],
        displaced[:, 1],
        tri_elements,
        facecolors=stress,
        shading="flat",
        cmap=cmc.managua_r,
        edgecolors="black",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Von Mises Stress on Deformed Mesh")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cb, cax=cax)
    plt.show()

# --- Compute the stress ---    
u = u_full.reshape(-1, n_dofs_per_node)


# --- Plot the displacement and stress ---
plot_displacement_and_stress(coords, u, elements, stress_vm.flatten())
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

