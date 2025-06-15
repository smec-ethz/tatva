import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp


from femsolver.jax_utils import auto_vmap


_registry = {
    "energy_density": compute_energy_density,
    "strain_measure": compute_strain_measure,
}


def register(namespace, val):
    """Register a dictionary of functions under a given namespace."""
    if namespace not in _registry:
        raise ValueError(f"Unknown namespace: {namespace}")
    _registry[namespace] = val


# compute strain energy at quadrature point x within a beam
@auto_vmap(x=0)
def compute_energy_density(x, dofs, nodes, mat_props):
    eps = _registry["strain_measure"](x, dofs, nodes)
    energy_density = mat_props["mu"] * jnp.sum(eps**2) + 0.5 * mat_props["lmbda"] * jnp.trace(eps)**2
    return energy_density

# compute strain energy for a finite element cell or a beam
@auto_vmap(
    dofs=2,
    nodes=2,
    mat_props=0,
    interpolation=0,
)
def compute_cell_strain_energy(dofs, nodes, mat_props, interpolation):
    # integration using Gauss-Quadrature
    points = interpolation["points"]
    weights = interpolation["weights"]
    energy_density = _registry["energy_density"](points, dofs, nodes, mat_props)
    return jnp.einsum("i...->...", energy_density, weights, optimize="optimal")


# compute total strain energy
@jax.jit
def compute_strain_energy(dofs, mesh, geometry, interpolation):
    cells = mesh["cells"]
    nodes = mesh["nodes"]
    total_comp_nodes = nodes.shape[0]

    dofs = dofs.reshape((total_comp_nodes, NB_DOFS_PER_NODE))

    # compute strain energy for each cell
    energies_cell = compute_cell_strain_energy(
        dofs[cells], nodes[cells], geometry, interpolation
    )

    # sum the local energies over all cells
    total_energy = jnp.einsum(
        "i->",
        energies_cell,
        optimize="optimal",
    )

    return total_energy