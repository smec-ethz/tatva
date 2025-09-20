import jax
import os

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")


from jax import Array
import jax.numpy as jnp

from jax_autovmap import autovmap
from tatva import Mesh, Operator, element
from tatva import sparse

from typing import NamedTuple
import timeit

import gmsh
import numpy as np
import meshio

def generate_mesh_with_gmsh(nx, ny):
    """
    Generates a structured triangular mesh of a unit square using the Gmsh Python API.

    This function replicates the output of the JAX-based meshing function,
    creating a grid with nx * ny cells, where each cell is split into two
    triangles with a consistent diagonal.

    Args:
        nx (int): The number of elements along the x-axis.
        ny (int): The number of elements along the y-axis.
        show_gui (bool): If True, opens the Gmsh GUI to visualize the mesh.

    Returns:
        tuple: A tuple containing:
            - coords (numpy.ndarray): Array of node coordinates, shape (N, 2).
            - elements (numpy.ndarray): Array of element connectivity, shape (M, 3),
                                       using 0-based indexing.
    """
    gmsh.initialize()
    gmsh.model.add("unit_square")
    ## 1. Define Geometry
    # Create the four corner points of the unit square
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0)
    p3 = gmsh.model.geo.addPoint(1, 1, 0)
    p4 = gmsh.model.geo.addPoint(0, 1, 0)
    # Create the four boundary lines
    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)
    # Create a surface from the boundary lines
    cl = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    surface = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    ## 2. Define Meshing Parameters
    # Use the "Transfinite" algorithm for a structured grid.
    # This requires specifying the number of nodes on each boundary curve.
    gmsh.model.mesh.setTransfiniteCurve(l_bottom, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l_top, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l_right, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(l_left, ny + 1)
    gmsh.model.mesh.setTransfiniteSurface(surface, "Left")

    gmsh.model.mesh.generate(2)  # Generate the 2D mesh

    gmsh.write("mesh.msh")
    print(f"Mesh successfully generated and saved to 'mesh.msh'")
    gmsh.finalize()
    _mesh = meshio.read("mesh.msh")
    mesh = Mesh(
        coords=_mesh.points[:, :2],
        elements=_mesh.cells_dict["triangle"],
    )

    return mesh

def generate_unit_square_mesh_tri_fast(nx, ny):
    # --- Coordinate generation is already efficient ---
    x = jnp.linspace(0, 1, nx + 1)
    y = jnp.linspace(0, 1, ny + 1)
    xv, yv = jnp.meshgrid(x, y, indexing="ij")
    coords = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)

    # --- Vectorized element creation ---
    # 1. Create 2D grids of indices for the bottom-left corner of each quad
    i_idx, j_idx = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")

    # 2. Calculate all node IDs for each corner of all quads at once
    n0 = (i_idx * (ny + 1) + j_idx).ravel()
    n1 = ((i_idx + 1) * (ny + 1) + j_idx).ravel()
    n2 = (i_idx * (ny + 1) + (j_idx + 1)).ravel()
    n3 = ((i_idx + 1) * (ny + 1) + (j_idx + 1)).ravel()

    # 3. Stack the node IDs to form the two triangles for each quad
    # 
    tri1 = jnp.stack([n0, n1, n3], axis=-1)
    tri2 = jnp.stack([n0, n3, n2], axis=-1)

    # 4. Concatenate the two sets of triangles into one array
    elements = jnp.concatenate([tri1, tri2], axis=0)
    
    return  Mesh(coords, elements)
  

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


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  # Shear modulus
    lmbda: float  # First Lamé parameter


mat = Material(mu=0.5, lmbda=1.0)

size_list = []
stiffness_compilation_time = []
stiffness_execution_time = []
# solve_compilation_time = []
# solve_execution_time = []

for nx in [10, 50, 100, 150, 200, 500, 700, 1000]:
    print(f"======= {nx} x {nx} ==============")
    mesh = generate_mesh_with_gmsh(nx, nx)

    n_nodes = mesh.coords.shape[0]
    n_dofs_per_node = 2
    n_dofs = n_dofs_per_node * n_nodes

    tri = element.Tri3()
    op = Operator(mesh, tri)

    @jax.jit
    def total_energy(u_flat):
        u = u_flat.reshape(-1, n_dofs_per_node)
        u_grad = op.grad(u)
        energy_density = strain_energy(u_grad, mat.mu, mat.lmbda)
        return op.integrate(energy_density)

    gradient = jax.jacrev(total_energy)
    hessian = jax.jacfwd(gradient)

    left_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], 0.0))[0]
    right_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], 1.0))[0]
    fixed_dofs = jnp.concatenate(
        [
            2 * left_nodes,
            2 * left_nodes + 1,
            2 * right_nodes,
        ]
    )

    free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), fixed_dofs)

    sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=2)
    hessian_sparse = sparse.jacfwd(gradient, sparsity_pattern=sparsity_pattern)

    prescribed_values = jnp.zeros(n_dofs).at[2 * right_nodes].set(0.3)

    u = jnp.zeros((n_dofs))

    start_time = timeit.default_timer()
    K = hessian_sparse(u)
    stiffness_compilation_time.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    K = hessian_sparse(u)
    stiffness_execution_time.append(timeit.default_timer() - start_time)

    """f_ext = jnp.zeros(n_dofs)
    K_lifted = K.at[jnp.ix_(free_dofs, fixed_dofs)].set(0.0)
    K_lifted = K_lifted.at[jnp.ix_(fixed_dofs, free_dofs)].set(0.0)
    K_lifted = K_lifted.at[jnp.ix_(fixed_dofs, fixed_dofs)].set(
        jnp.eye(len(fixed_dofs))
    )

    K_fc = K.at[jnp.ix_(free_dofs, fixed_dofs)].get()

    res_lifted = f_ext
    res_lifted = res_lifted.at[free_dofs].add(
        -K_fc @ prescribed_values.at[fixed_dofs].get()
    )
    res_lifted = res_lifted.at[fixed_dofs].set(prescribed_values.at[fixed_dofs].get())

    start_time = timeit.default_timer()
    u_flat = jnp.linalg.solve(K_lifted, res_lifted)
    u_solution = u_flat.reshape(n_nodes, n_dofs_per_node)
    solve_compilation_time.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    u_flat = jnp.linalg.solve(K_lifted, res_lifted)
    u_solution = u_flat.reshape(n_nodes, n_dofs_per_node)
    solve_execution_time.append(timeit.default_timer() - start_time)"""

    size_list.append(n_dofs)

import pandas as pd

# dictionary of lists
dict = {
    "size": size_list,
    "stiffness_comp_time": stiffness_compilation_time,
    "stiffness_exe_time": stiffness_execution_time,
    # "comp_time_solve": solve_compilation_time,
    # "exe_time_solve": solve_execution_time,
}

df = pd.DataFrame(dict)

print(df)

import subprocess

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True,
    text=True,
).stdout.strip("\n")

gpu_make = result.replace(" ", "_")
if os.environ["JAX_PLATFORM"] == "cpu":
    gpu_make = "cpu"

if os.environ["JAX_PLATFORM"] == "rocm-gpu":
    gpu_make = "mi300a"

df.to_csv(f"""benchmark-vmap_{gpu_make}.csv""")
