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
def total_energy(u_flat):
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    energy_density = strain_energy(u_grad, 1.0, 0.0)
    return op.integrate(energy_density)


K = jax.jacfwd(jax.jacrev(total_energy))(jnp.zeros(mesh.coords.shape[0] * 2))

# %%
u = jnp.zeros(mesh.coords.shape[0] * 2)
op.grad(u)

# %%
K_sparse = assemble(
    total_energy_fn=total_energy,
    operators=[op],
    u_flat=jnp.zeros(mesh.coords.shape[0] * 2),
)
# %%

K

# %%

K_sparse.todense() == K


# %% [markdown]
# ::: {.callout-note}
# The non-zero entries of the stiffness matrix are concentrated around a diagonal band is indicated by blue dots in the figure above. The zero entries are indicated by white dots.
# :::
#

# %% [markdown]
# Therefore, we only need the non-zero entries of the stiffness matrix for solving a mechanical problem, which can save a lot of memory. Since the sparsity pattern (the location of the non-zero entries) is determined by the mesh connectivity (and the local nature of the shape functions), we can create a sparsity pattern for a given mesh and use it to construct a sparse stiffness matrix with only the non-zero entries.
#
# In the section below, we will discuss two aspects of sparse matrices:
#
# 1. How to store/construct a sparse matrix efficiently?
# 2. How to solve a sparse linear system of equations?
#

# %% [markdown]
# ## How to store a sparse matrix efficiently?
#
# There are several ways to store a sparse matrix, each designed to save the memory by only storing  the non-zero elements.
# Irrespective of the ways, two things required to store a sparse matrix are:
#
# - The row and column indices of the non-zero elements of a sparse matrix,  `(row, col)`
# - The values of the non-zero elements of a sparse matrix, `value`
#
#
# To illustrate the various ways, we will use a simple `4x4` sparse matrix $\mathbf{K}$ which has `6` non-zero values.
#
# $$
# \mathbf{K} = \begin{bmatrix}5 & 0 &0 &  1\\
#      0 & 7 & 2 &  0\\
#      0 & 0 & 0 & 0 \\
#      3 & 0 & 9 & 0 \\
#      \end{bmatrix}
# $$
#
# In this course, we will use two ways to sparse matrix based on the what we do with sparse matrices. The two different ways to represent a sparse matrices we will use are:
#
# - Coordinate Format (COO)
# - Compressed Sparse Row (CSR)
#
# ###  Coordinate Format (C00)
#
# In this format, sparse matrix is store as a simple list of triplets: `(row, col, value)`. It is the most straightforward way to represent a sparse matrix. For our matrix, the `COO` representation would be:
#
# - `row` : [0, 0, 1, 1, 3, 3]
# - `col` : [0, 3, 1, 2, 0, 2]
# - `value` : [5, 1, 7, 2, 3, 9]
#
# This format is good for **building a sparse matrix** as it is easy to add new `(row, col, value)` triplets.
#
#
# ### Compressed Sparse Row (CSR)
#
# This format is most often used for **performing calculations** with sparse matrix. It also uses three arrays to represent a sparse matrix but they have a different meaning:
#
# - `value`: All the non-zero values read from row by row.
# - `indices`: The column index for each corresponding value in the data array.
# - `indptr`: Any array of size `number of rows + 1`. The entry `indptr[i]` tells the index where the data for row `i` begins in the `value` array. The data for the row `i` is located in the slice `data[indptr[i]: indptr[i+1]]`
#
# For our matrix, the CSR representation is:
#
# - `value`: [5, 1, 7, 2, 3, 9]
# - `indices`: [0,3, 1, 2, 0, 2]
# - `indptr`: [0, 2, 4, 4, 6]
#
# From this we get that:
#
# - `Row 0` starts at index `0`. The next row starts at index `2`, so row 0's data is `value[0:2]` which is `[5, 1]`.
# - `Row 1` starts at index `2`. The next row starts at index `4`, so row 1's data is `value[2:4]` which is `[7, 2]`.
# - `Row 2` starts at index `4`. The next row starts at index `4`, so row 2's data is `value[4:4]` which is empty.
# - `Row 3` starts at index `4`. The next row starts at index `6`, so row 3's data is `value[4:6]` which is `[3, 9]`.
#
# ::: {.callout-note}
# We will both `COO` and `CSR` to represent sparse matrices. We will not be constructing these sparse representation ourselves, rather we will use libraries such as `scipy`  to handle this. The above description is just so that you know why we use a specific format for a specific operations.
# :::

# %% [markdown]
# ### Constructing sparse stiffness matrix in FEM
#

# %% [markdown]
# In order to construct a sparse stiffness matrix, we need to force the differentiation to be done only on the non-zero entries of the stiffness matrix.
#
#
# ::: {.callout-note}
# Remember we are using direct differentiation of internal force python  to compute the stiffness matrix, therefore, we need to force the differentiation to be done only on the non-zero entries of the stiffness matrix.
#
# $$
# \mathbf{K} = \dfrac{\partial \boldsymbol{f}_\text{int}}{\partial \boldsymbol{u}}
# $$
#
# And to do this, we were using the `jax.jacfwd` function.
# :::
#
# We can force the differentiation to be done only on the non-zero entries of the stiffness matrix if we know the which entries are non-zero and degree of  freedom affects the other degree of freedom.
#
# As mentioned earlier, that this information is already available in the mesh connectivity. Therefore, we can use the mesh connectivity to create a sparsity pattern for the matrix which will contain the following information:
#
# - `(row, col)` of the non-zero entries of the stiffness matrix
#
# In `tatva`, we use the sparse module to create a sparsity pattern for the matrix. The function takes the following arguments:
#
# - `mesh`: The mesh object.
# - `n_dofs_per_node`: The number of degrees of freedom per node.
# - `constraint_elements`: Optional array of constraint elements. If provided, the sparsity pattern will be created for the constraint elements.
#
# ::: {.callout-note}
# The `constraint_elements` is an optional argument. If provided, the sparsity pattern will be created for the constraint elements. We will see an example of this when we discuss fracture mechanics with cohesive elements.
# :::
#
# The function returns a `jax.experimental.sparse.BCOO` object  which basically `COO` representation as we discussed above.  The reason we used this format because here we are constructing  a sparse representation.
#

# %%
# | fig-align: center
# | fig-cap: "Sparsity created from `tatva.sparse` module based on mesh connectivity"
# | label: fig-sparsity-pattern-sparse


from tatva import sparse

sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=2)

plt.figure(figsize=(2, 2), layout="constrained")
plt.spy(sparsity_pattern.todense(), color=colors.blue, markersize=2)
plt.show()

# %% [markdown]
# You can access the `row, col` using the following code:
#
# - `sparsity_pattern.indices`
#
#

# %%
sparsity_pattern.indices

# %% [markdown]
# Now, we can use the sparsity pattern to create a sparse stiffness matrix. Based on this sparsity pattern, the automatic differentiation is restricted to the non-zero entries of the matrix. This considerably reduces the computational cost. We provided two functions to create a sparse stiffness matrix based on the sparsity pattern: `sparse.jacfwd` and `sparse.jacrev`. These functions are wrappers around the `jax.jacfwd` and `jax.jacrev` functions, but they take the sparsity pattern as an argument.
#

# %%
# | fig-align: center
# | fig-cap: "Stiffness matrix computed from sparsity pattern"
# | label: fig-sparsity-pattern-3


K_sparse = sparse.jacfwd(jax.jacrev(total_energy), sparsity_pattern=sparsity_pattern)(
    jnp.zeros(mesh.coords.shape[0] * 2)
)

plt.figure(figsize=(2, 2), layout="constrained")
plt.spy(K_sparse.todense(), color=colors.blue, markersize=2)
plt.show()

# %% [markdown]
# We can actually check if the stiffness matrix computed using our sparsity pattern is the same as the stiffness matrix computed using the full matrix.
#

# %%
np.allclose(K_sparse.todense(), K)

# %% [markdown]
# As a quick analyses let us check how much memory is saved by using a sparse representation of the stiffness matrix. Below we plot the ratio of the memory required for the sparse stiffness matrix to the memory required for the dense stiffness matrix.
#

# %%
# | echo: false
# | fig-align: center
# | fig-cap: "Comparing memory required for a dense stiffness matrix and a sparse stiffness matrix"
# | label: fig-memory-required-sparse-vs-dense

import matplotlib.pyplot as plt
import numpy as np

from tatva import Mesh
from tatva.plotting import STYLE_PATH, colors

memory_in_gb = []
memory_in_gb_sparse = []
n_nodes = []
nx_list = [1, 10, 20, 40, 80, 100, 160]

for nx in nx_list:
    ny = nx
    nb_nodes = (nx + 1) * (ny + 1)
    nb_dofs = 2 * nb_nodes
    nb_entries = nb_dofs * nb_dofs
    memory_required = nb_entries * 8 / 1024 / 1024 / 1024  # in GB
    n_nodes.append(nb_nodes)
    memory_in_gb.append(memory_required)

    _mesh = Mesh.unit_square(nx, ny)
    sparsity = sparse.create_sparsity_pattern(_mesh, n_dofs_per_node=2)
    nb_entries_sparse = sparsity.data.shape[0]
    memory_required_sparse = nb_entries_sparse * 8 / 1024 / 1024 / 1024  # in GB
    memory_in_gb_sparse.append(memory_required_sparse)


plt.figure(figsize=(6, 3), layout="constrained")
ax = plt.axes()
ax.semilogy(
    nx_list,
    np.array(memory_in_gb_sparse) / np.array(memory_in_gb),
    "o-",
    label="Sparse / Dense",
)


ax.set_xlabel(r"Number of elements in $x$-direction")
ax.set_ylabel(r"Memory required (GB) for sparse / dense")
ax.set_xlim(left=0)
plt.grid()
plt.legend(frameon=False)
plt.show()

# %% [markdown]
# ::: {.callout-note}
# We can clearly see that memory requirement for sparse stiffness matrix decreases tremendously with the number of elements. A reduction of 3 orders of magnitude is seen above.
# :::
#

# %% [markdown]
# ## How to solve a sparse linear system of equations?
#
# Now lets us solve the linear system of equations $\mathbf{K} \boldsymbol{u} = \boldsymbol{f}$ where $\mathbf{K}$ is a sparse matrix as we constructed above. For this example, we define a mesh of 50x50 elements and create a function that compute the sparse stiffness matrix for a given displacement field. We will use the functions defined above to construct the sparse stiffness matrix.
#

# %%
mesh = Mesh.unit_square(50, 50)
n_nodes = mesh.coords.shape[0]
n_dofs_per_node = 2
n_dofs = n_nodes * n_dofs_per_node


tri = element.Tri3()
op = Operator(mesh, tri)


@jax.jit
def total_energy(u_flat):
    u = u_flat.reshape(-1, 2)
    u_grad = op.grad(u)
    energy_density = strain_energy(u_grad, 1.0, 0.0)
    return op.integrate(energy_density)


sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=n_dofs_per_node)


gradient = jax.jacrev(total_energy)
hessian_sparse = sparse.jacfwd(
    jax.jacrev(total_energy), sparsity_pattern=sparsity_pattern
)

# %% [markdown]
# ### Applying Dirichlet boundary conditions to a sparse stiffness matrix
#

# %%
y_max = jnp.max(mesh.coords[:, 1])
y_min = jnp.min(mesh.coords[:, 1])
x_max = jnp.max(mesh.coords[:, 0])
x_min = jnp.min(mesh.coords[:, 0])

left_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_min))[0]
right_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_max))[0]
fixed_dofs = jnp.concatenate(
    [
        2 * left_nodes,
        2 * left_nodes + 1,
        2 * right_nodes,
    ]
)

prescribed_values = jnp.zeros(n_dofs).at[2 * right_nodes].set(0.3)

# %% [markdown]
# To apply the Dirichlet boundary conditions, we use the same lifting approach as we used in @sec-fem-with-ad. This is a constraint approach where the entry of the stiffness matrix corresponding to the constrained degrees of freedom are set to 1 and the corresponding rows and columns are set to 0.  In order to do this, we need to know the indices `(row, col)` of the non-zero entries of the stiffness matrix. Once we have the indices, we can set the corresponding entries to 1 and 0.
#
# We will use the function `sparse.get_bc_indices` to get the indices of the non-zero entries of the stiffness matrix. The function `get_bc_indices` takes the following arguments:
#
# - `sparsity_pattern`: The sparsity pattern created using the `create_sparsity_pattern` function.
# - `fixed_dofs`: The degrees of freedom where we apply the Dirichlet boundary conditions.
#
# The function returns two arrays:
#
# - `zero_indices`: The indices of the sparsity pattern that correspond to location where the stiffness matrix is set to 0.
# - `one_indices`: The indices of the sparsity pattern that correspond to location where the stiffness matrix is set to 1.
#

# %%
zero_indices, one_indices = sparse.get_bc_indices(sparsity_pattern, fixed_dofs)

# %% [markdown]
# Now, we have everything we need to solve the linear system of equations.

# %% [markdown]
# ### Sparse solvers (using SciPy)
#
# We will use `SciPy` to solve the sparse linear system of equations.

# %%
import scipy.sparse as sp

# %% [markdown]
# We define a newton solver that uses Scipy to solve the linear system of equations. We make use of the sparsity pattern to construct the linear system of equations. Notice that we have to convert the `BCOO` matrix to a `CSR` matrix. As we discussed `CSR` format is the more efficient format for performing mathematical operations on a sparse matrix, therefore, we convert the `COO` format to `CSR` format first and then use that `CSR` format to to solve the linear system.
#
# `Scipy` has a functionality to construct a `CSR` matrix from the list of triplets `(row, col, value)`. We use this to first construct the `CSR` matrix and then use the `scipy.linalg.spsolve` module to solve
#
# $$
# \mathbf{K}\boldsymbol{u} = \boldsymbol{f}_\text{ext} - \boldsymbol{f}_\text{int}
# $$
#
# The below function implements all these steps.


# %%
def newton_scipy_solver(
    u,
    fext,
    gradient,
    hessian_sparse,
    fixed_dofs,
    zero_indices,
    one_indices,
):
    fint = gradient(u)

    iiter = 0
    norm_res = 1.0

    tol = 1e-8
    max_iter = 10

    while norm_res > tol and iiter < max_iter:
        residual = fext - fint
        residual = residual.at[fixed_dofs].set(0.0)

        K_sparse = hessian_sparse(u)
        K_data_lifted = K_sparse.data.at[zero_indices].set(0)
        K_data_lifted = K_data_lifted.at[one_indices].set(1)

        K_csr = sp.csr_matrix(
            (K_data_lifted, (K_sparse.indices[:, 0], K_sparse.indices[:, 1]))
        )

        du = sp.linalg.spsolve(K_csr, residual)
        u = u.at[:].add(du)

        fint = gradient(u)
        residual = fext - fint
        residual = residual.at[fixed_dofs].set(0)
        norm_res = jnp.linalg.norm(residual)

        print(f"  Residual: {norm_res:.2e}")

        iiter += 1

    return u, norm_res


# %% [markdown]
# Now, we use the above defined function to solve the problem  in `10` loading steps.
#

# %%
# | output: false

u_prev = jnp.zeros(n_dofs)
fext = jnp.zeros(n_dofs)

n_steps = 10

applied_displacement = prescribed_values / n_steps  # displacement increment
for step in range(n_steps):
    print(f"Step {step + 1}/{n_steps}")
    u_prev = u_prev.at[fixed_dofs].add(applied_displacement[fixed_dofs])

    u_new, rnorm = newton_scipy_solver(
        u_prev,
        fext,
        gradient,
        hessian_sparse,
        fixed_dofs,
        zero_indices,
        one_indices,
    )

    u_prev = u_new

u_solution = u_prev.reshape(n_nodes, n_dofs_per_node)

# %% [markdown]
# ### Post-processing
#
# Now we can plot the stress distribution and the displacement.
#

# %%
# | code-fold: true
# | code-summary: "Code: Plotting the stress distribution"
# | fig-align: center
# | fig-cap: "Stress distribution after using sparse solver"
# | label: fig-stress-distribution

# squeeze to remove the quad point dimension (only 1 quad point)
grad_u = op.grad(u_solution).squeeze()
strains = compute_strain(grad_u)
stresses = compute_stress(strains, mat.mu, mat.lmbda)

plt.figure(figsize=(4, 3), layout="constrained")
ax = plt.axes()
plot_element_values(
    u=u_solution,
    mesh=mesh,
    values=stresses[:, 1, 1].flatten(),
    label=r"$\sigma_{yy}$",
    ax=ax,
)
ax.set_xlabel(r"x")
ax.set_ylabel(r"y")
ax.set_aspect("equal")
ax.margins(0, 0)
plt.show()

# %% [markdown]
# <!--
# ### Sparse solver (using PETSc)
#
# `PETSc` is a state-of-the-art library for solving partial differential equations. It is a collection of solvers for a wide range of problems in science and engineering. It is a very powerful library and is used in a wide range of applications.
#
# There are various types of solvers available in `PETSc`. The solvers that we will be using in this notebook and in other notebooks are:
#
# - `KSP`: Krylov subspace methods (for the linear system of equations)
# - `SNES`: Nonlinear solvers (for the Newton-Raphson method)
#
# The `KSP` solver is used to solve the linear system of equations that arises from the finite element method. The `SNES` solver is used to solve the nonlinear system of equations that arises from the Newton-Raphson method.
#
# In this notebook, we will use the `KSP` solver to solve the linear sparse system of equations. We will use this sparse linear solver within our custom Newton-Raphson solver, similar to the one we have use for the SciPy example.
#
# We define a newton solver that uses PETSc to solve the linear system of equations. It makes use of the sparse stiffness matrix and the sparsity pattern to construct the linear system of equations. Furthermore, the _Dirichlet boundary conditions_ are applied using the _lifting approach_.
#
# -->

# %%
# | echo: false

from petsc4py import PETSc


def newton_petsc_solver(
    u,
    fext,
    gradient,
    hessian_sparse,
    fixed_dofs,
    zero_indices,
    one_indices,
):
    fint = gradient(u)

    iiter = 0
    norm_res = 1.0

    tol = 1e-8
    max_iter = 10

    while norm_res > tol and iiter < max_iter:
        residual = fext - fint
        residual = residual.at[fixed_dofs].set(0)

        K_sparse = hessian_sparse(u)
        K_data_lifted = K_sparse.data.at[zero_indices].set(0)
        K_data_lifted = K_data_lifted.at[one_indices].set(1)

        K_csr = sp.csr_matrix(
            (K_data_lifted, (K_sparse.indices[:, 0], K_sparse.indices[:, 1]))
        )
        # creating PETSc matrix
        A = PETSc.Mat()
        A.createAIJWithArrays(
            size=K_csr.shape, csr=(K_csr.indptr, K_csr.indices, K_csr.data)
        )
        A.assemble()

        # creating PETSc KSP solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.getPC().setType("none")
        ksp.setConvergenceHistory()

        # creating PETSc vectors
        b = A.createVecLeft()
        du = A.createVecRight()

        # setting residual
        b.zeroEntries()
        b.setArray(residual)

        # solving the linear system
        du.zeroEntries()
        ksp.solve(b, du)

        # updating solution
        u = u.at[:].add(du.getArray(readonly=True))

        # computing residual
        fint = gradient(u)
        residual = fext - fint
        residual = residual.at[fixed_dofs].set(0)
        norm_res = jnp.linalg.norm(residual)

        print(f"  Residual: {norm_res:.2e}")

        # destroying PETSc objects
        A.destroy()
        b.destroy()
        du.destroy()
        ksp.destroy()

        iiter += 1

    return u, norm_res
