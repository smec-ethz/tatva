import os

import jax

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use double-precision
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")

import numpy as np
import timeit

from create_lattice_model import create_structure
from fem_tools import (
    create_finite_element,
    create_basis,
    integrate_over_cell,
    interpolate_at_quadrature,
    batched_jacobian,
)


def create_beam_structure(nb_elem, length=1):
    xi = np.linspace(0, length, nb_elem + 1)
    yi = np.zeros_like(xi)
    coordinates = np.vstack((xi.flatten(), yi.flatten())).T
    connectivity = list()
    for i in range(nb_elem):
        connectivity.append([i, i + 1])
    connectivity = np.unique(np.array(connectivity), axis=0)

    return coordinates, connectivity


@jax.jit
def compute_tangent(jacobian):
    return jnp.asarray([jacobian[0], jacobian[1]]) / jnp.linalg.norm(jacobian)


@jax.jit
def compute_tangential_displacement(dofs, tangent):
    return jnp.einsum("i..., i... -> i", dofs, jnp.asarray([tangent.T])).reshape(
        nb_nodes_per_cell, 1
    )


@jax.jit
def compute_product(a, b):
    return jnp.einsum("i..., i... -> i", a, b).reshape(nb_quads, -1)


# using jax.vmap to vectorize the functions, so as to perform operations
# over all cells simultaneously
batched_tangent = jax.vmap(compute_tangent, in_axes=(0))

batched_tangential_displacement = jax.vmap(
    compute_tangential_displacement, in_axes=(0, 0)
)


batched_interpolation = jax.vmap(interpolate_at_quadrature, in_axes=(0, None))

batched_integration = jax.vmap(integrate_over_cell, in_axes=(0, 0, None))

batched_product = jax.vmap(compute_product, in_axes=(0, 0))


# defining functions to compute axial and bending strain energy
@jax.jit
def compute_axial_strain_energy(dofs, cells, detJs, Js, tangents, wts, dNdξ):
    dofs = dofs.reshape((total_comp_nodes, nb_dofs_per_node))
    ut = batched_tangential_displacement(dofs[cells], tangents)

    grads_at_quad_ref = batched_interpolation(ut, dNdξ)

    J_dot_J = jnp.vecdot(Js, Js).reshape(ut.shape[0], -1)

    J_dot_t = jnp.vecdot(Js, tangents).reshape(ut.shape[0], -1)

    value_at_quad = batched_product(grads_at_quad_ref, J_dot_t / J_dot_J)

    integral_values_cell = batched_integration(0.5 * value_at_quad**2, detJs, wts)

    integral_value = jnp.einsum(
        "i...->...",
        integral_values_cell,
        optimize="optimal",
    )

    return integral_value[0]


def get_global_dofs_x(nodes, x, ndofs=2):
    idx = np.where(np.isclose(nodes[:, 0], x, atol=1e-8) == True)[0]
    return idx * ndofs, idx * ndofs + 1


def get_global_dofs_y(nodes, y, ndofs=2):
    idx = np.where(np.isclose(nodes[:, 1], y, atol=1e-8) == True)[0]
    return (idx * ndofs, idx * ndofs + 1)


def apply_boundary_conditions(u):
    applied_displacement = 0.5
    u = u.at[right_dofs[1]].set(applied_displacement)
    u = u.at[right_dofs[0]].set(0)
    u = u.at[left_dofs[0]].set(0.0)
    u = u.at[left_dofs[1]].set(0.0)
    return u


def apply_boundary_conditions_to_residual(res):
    res = res.at[right_dofs[0]].set(0.0)
    res = res.at[right_dofs[1]].set(0.0)
    res = res.at[left_dofs[0]].set(0.0)
    res = res.at[left_dofs[1]].set(0.0)
    return res


def gradient_descent_cache(
    dofs,
    new_connectivity,
    detJs,
    Js,
    tangents,
    wts,
    dNdξ,
    learning_rate=0.001,
    num_iterations=1000,
):

    iterations = jnp.arange(num_iterations)
    _, (positions_over_time, loss_over_time) = jax.lax.scan(
        grad_step_with_time_evolution,
        (
            dofs,
            new_connectivity,
            detJs,
            Js,
            tangents,
            wts,
            dNdξ,
            learning_rate,
        ),
        iterations,
    )

    return positions_over_time, loss_over_time


# defining function to compute internal forces
compute_internal_force = jax.jit(jax.jacrev(compute_axial_strain_energy))


def grad_step_with_time_evolution(carry, _):
    (
        dofs,
        new_connectivity,
        detJs,
        Js,
        tangents,
        wts,
        dNdξ,
        learning_rate,
    ) = carry
    loss_val = compute_axial_strain_energy(
        dofs,
        new_connectivity,
        detJs,
        Js,
        tangents,
        wts,
        dNdξ,
    )
    grad = compute_internal_force(
        dofs,
        new_connectivity,
        detJs,
        Js,
        tangents,
        wts,
        dNdξ,
    )
    dofs -= learning_rate * grad
    dofs = apply_boundary_conditions(dofs)

    return (
        dofs,
        new_connectivity,
        detJs,
        Js,
        tangents,
        wts,
        dNdξ,
        learning_rate,
    ), (dofs, loss_val)


gradient_compilation_time = []
gradient_executation_time = []
grad_descent_compilation_time = []
grad_descent_execution_time = []

# creating the lattice structure
sizes = [10, 100, 1000]
for size in sizes:

    print(f"======= {size} x {size} ==============")
    structure_size = (size, size)
    lengths = (1, 1)
    nodes, connectivity = create_structure(
        N=structure_size, cell_type="reg-triangle", lengths=lengths, notched=False
    )

    print(nodes.shape[0])

    # creating finite element
    degree = 1
    element = create_finite_element(
        "interval", element_type="Lagrange", degree=degree, variant="equispaced"
    )

    # creating quadrature scheme for interpolation and integration
    nb_quads = int(np.ceil((degree + 1) / 2))  # for exact integration
    quadrature_rule = "legendre"
    basis = create_basis(element, quadrature_rule, nb_quads=nb_quads)

    total_comp_nodes = nodes.shape[0]
    nb_nodes_per_cell = connectivity.shape[1]
    nb_dofs_per_node = 2

    # defining dofs
    dofs = np.zeros((total_comp_nodes, nb_dofs_per_node))
    dofs[:, 0] = nodes[:, 0] ** 1
    # dofs[:, 1] = nodes[:, 0] ** 2

    # we need to reshape the dofs
    dofs = dofs.reshape(total_comp_nodes * nb_dofs_per_node, -1)

    start_time = timeit.default_timer()

    dofs = jax.device_put(dofs)
    nodes = jax.device_put(nodes)
    connectivity = jax.device_put(connectivity)

    right_dofs = get_global_dofs_x(nodes=nodes, x=np.max(nodes[:, 0]))
    left_dofs = get_global_dofs_x(nodes=nodes, x=np.min(nodes[:, 0]))

    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()

    # computing determinant of Jacobian for each cell
    Js = batched_jacobian(0.5, nodes[connectivity])
    tangents = batched_tangent(Js)
    detJs = jnp.einsum("i..., i...->i", Js, tangents, optimize="optimal")

    print(timeit.default_timer() - start_time)

    # print(
    #    jax.make_jaxpr(compute_axial_strain_energy)(
    #        dofs, connectivity, detJs, Js, tangents, basis.wts, basis.dNdξ
    #    )
    # )

    # start_time = timeit.default_timer()

    print(
        "Axial energy",
        compute_axial_strain_energy(
            dofs, connectivity, detJs, Js, tangents, basis.wts, basis.dNdξ
        ),
    )

    # print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()

    compute_axial_strain_energy(
        dofs, connectivity, detJs, Js, tangents, basis.wts, basis.dNdξ
    )

    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()

    residual = compute_internal_force(
        dofs,
        connectivity,
        detJs,
        Js,
        tangents,
        basis.wts,
        basis.dNdξ,
    )

    gradient_compilation_time.append(timeit.default_timer() - start_time)
    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()

    residual = compute_internal_force(
        dofs,
        connectivity,
        detJs,
        Js,
        tangents,
        basis.wts,
        basis.dNdξ,
    )

    gradient_executation_time.append(timeit.default_timer() - start_time)

    print(timeit.default_timer() - start_time)

    u = jnp.zeros((total_comp_nodes, nb_dofs_per_node))
    u = u.reshape(total_comp_nodes * nb_dofs_per_node, -1)

    u = apply_boundary_conditions(u)

    start_time = timeit.default_timer()

    uu, xx = gradient_descent_cache(
        u,
        connectivity,
        detJs,
        Js,
        tangents,
        basis.wts,
        basis.dNdξ,
        learning_rate=0.02,
        num_iterations=100,
    )

    grad_descent_compilation_time.append(timeit.default_timer() - start_time)
    print(timeit.default_timer() - start_time)


    start_time = timeit.default_timer()

    uu, xx = gradient_descent_cache(
        u,
        connectivity,
        detJs,
        Js,
        tangents,
        basis.wts,
        basis.dNdξ,
        learning_rate=0.02,
        num_iterations=100,
    )

    grad_descent_execution_time.append(timeit.default_timer() - start_time)
    print(timeit.default_timer() - start_time)


import pandas as pd

# dictionary of lists
dict = {
    "size": sizes,
    "comp_time_grad": gradient_compilation_time,
    "exe_time_grad": gradient_executation_time,
    "comp_time_grad_desc": grad_descent_compilation_time,
    "exe_time_grad_desc": grad_descent_execution_time,
}

df = pd.DataFrame(dict)

print(df)

import subprocess

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip("\n")

    gpu_make = result.replace(" ", "_")

except:
    gpu_make = 'cpu'


df.to_csv(f"""benchmark-{gpu_make}.csv""")
