import os

import jax

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use double-precision
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")


import symfem
import sympy

import numpy as np


@jax.tree_util.register_pytree_node_class
class Basis:
    """
    A class used to represent a Basis for finite element method (FEM) simulations.

    Attributes
    ----------
    N : jnp.ndarray
        Shape functions evaluated at quadrature points, with shape (nb_quads, nb_nodes_per_element).
    dNdξ : jnp.ndarray
        Derivatives of shape functions with respect to the reference coordinates, with shape (nb_quads, nb_nodes_per_element).
    wts : jnp.ndarray
        Quadrature weights, with shape (nb_quads).

    Methods
    -------
    tree_flatten():
        Flattens the Basis object into a tuple of children and auxiliary data for JAX transformations.
    tree_unflatten(aux_data, children):
        Reconstructs the Basis object from flattened children and auxiliary data.
    """

    def __init__(self, nb_quads, nb_nodes_per_element):
        self.N = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.dNdξ = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.d2Ndξ2 = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.wts = jnp.zeros(nb_quads)
        self.quad_pts = jnp.zeros(nb_quads)

    def tree_flatten(self):
        return ((self.N, self.dNdξ, self.wts, self.quad_pts), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        N, dNdξ, ω, quad_pts = children
        instance = cls(N.shape[0], N.shape[1])
        instance.N = N
        instance.dNdξ = dNdξ
        instance.wts = ω
        instance.quad_pts = quad_pts
        return instance


def create_finite_element(cell, element_type, degree, variant):
    """
    Create a finite element.

    Parameters:
    cell (str): The type of simplex (e.g., 'triangle', 'tetrahedron').
    element_type (str): The type of finite element (e.g., 'Lagrange', 'Raviart-Thomas').
    degree (int): The degree of the finite element.
    variant (str): The variant type.

    Returns:
    symfem.Element: The created finite element.
    """
    return symfem.create_element(cell, element_type, order=degree, variant=variant)


def create_mapping_to_physical_cell(points, reference_element):
    x = sympy.symbols("x")

    mapping = sympy.lambdify(
        x,
        reference_element.reference.get_map_to(
            [
                points[0],
                points[1],
            ]
        ),
        "jax",
    )

    mapping_lamb = lambda x: jnp.array(mapping(x))
    map_to_physical = jax.jit(mapping_lamb)

    return map_to_physical


@jax.jit
def linear_interpolation(carry, iiter):
    x, v0, v1 = carry
    return v0[iiter] + (v1[iiter] - v0[iiter]) * x


@jax.jit
def map_to_interval_cell(x, cell_nodes):
    v0, v1 = cell_nodes[0], cell_nodes[1]
    mapped = jnp.zeros(2)
    mapped = mapped.at[0].set(linear_interpolation((x, v0, v1), 0))
    mapped = mapped.at[1].set(linear_interpolation((x, v0, v1), 1))
    return mapped


batched_mapping = jax.vmap(map_to_interval_cell, in_axes=(None, 0))
batched_jacobian = jax.vmap(
    jax.jit(jax.jacfwd(map_to_interval_cell)), in_axes=(None, 0)
)


def create_jacobian(points, reference_element):
    map_to_physical = create_mapping_to_physical_cell(points, reference_element)

    jacobian_fun = jax.jit(jax.jacrev(map_to_physical))  # returns [∂x/∂ξ ∂y/∂ξ].T

    return jacobian_fun


def create_tangent(points, reference_element):
    jacobian_fun = create_jacobian(points, reference_element)
    t = jacobian_fun(0.0)
    return jnp.asarray([t[0], t[1]]) / jnp.linalg.norm(t)


def create_basis(element, quadrature_rule, nb_quads=None):
    """
    Create a basis for a given reference element using quadrature points.

    Parameters:
    element (ReferenceElement): The reference element for which the basis is created.
    nb_quads (int): The number of quadrature points to use. If None, computes the number of quadrature required for exact integration.

    Returns:
    Basis: An object containing the shape functions and their derivatives evaluated at the quadrature points, as well as the quadrature weights.

    The function performs the following steps:
    1. Retrieves the basis functions from the reference element.
    2. Converts the basis functions to a numpy-compatible format.
    3. Uses JAX to JIT compile the shape functions and their derivatives.
    4. Evaluates the shape functions and their derivatives at the quadrature points.
    5. Stores the evaluated shape functions, their derivatives, and the quadrature weights in a Basis object.
    """

    x = sympy.symbols("x")

    f = sympy.lambdify(x, element.get_basis_functions(), "jax")

    # so that output is an array and not list
    shape_fun = lambda x: jnp.array(f(x))

    # using JAX JIT on the shape function
    shape_functions = jax.jit(shape_fun)

    # using forward-mode differentiation to compute the shape derivatives
    shape_derivatives = jax.jit(jax.jacfwd(shape_functions))

    # using forward-mode differentiation to compute the 2nd derivatives
    shape_double_derivatives = jax.jit(jax.jacfwd(shape_derivatives))

    # number of computational nodes (based on the polynomial degree)
    nb_nodes = len(element.dof_plot_positions())

    if nb_quads is None:
        nb_quads = int(
            np.ceil((element.order + 1) / 2)
        )  # for exact integration of polynomial degree

    basis = Basis(nb_quads, nb_nodes)
    for i, quad in enumerate(
        symfem.quadrature.get_quadrature(quadrature_rule, nb_quads)[0]
    ):
        basis.quad_pts = basis.quad_pts.at[i].set(float(quad))
        basis.N = basis.N.at[i, :].set(shape_functions(float(quad)))
        basis.dNdξ = basis.dNdξ.at[i, :].set(shape_derivatives(float(quad)))
        basis.d2Ndξ2 = basis.d2Ndξ2.at[i, :].set(shape_double_derivatives(float(quad)))

    for i, wts in enumerate(
        symfem.quadrature.get_quadrature(quadrature_rule, nb_quads)[1]
    ):
        basis.wts = basis.wts.at[i].set(float(wts))

    return basis


def compute_computational_nodes(nodes, cells, element):
    computational_nodes = nodes.tolist()
    cell_dof_map = []

    for cell in cells:
        dof_list = cell.tolist()

        # map_to_physical = create_mapping_to_physical_cell(nodes[cell], element)
        for dof_pos in element.dof_plot_positions():
            new_node = map_to_interval_cell(float(dof_pos[0]), nodes[cell]).tolist()
            # new_node = map_to_physical(float(dof_pos[0])).tolist()
            if new_node not in computational_nodes:
                dof_list.append(len(computational_nodes))
                computational_nodes.append(new_node)

        cell_dof_map.append(dof_list)

    computational_nodes = np.array(computational_nodes)
    return computational_nodes, np.array(cell_dof_map)


"""def compute_computational_nodes(nodes, cells, element):
    computational_nodes = nodes.tolist()
    node_index_map = {
        tuple(node): i for i, node in enumerate(computational_nodes)
    }  # Hashmap for fast lookup
    cell_dof_map = []

    dof_positions = [
        float(dof[0]) for dof in element.dof_plot_positions()
    ]  # Compute once

    for cell in cells:
        dof_list = cell.tolist()

        for dof_pos in dof_positions:
            new_node = tuple(
                map_to_interval_cell(dof_pos, nodes[cell])
            )  # Convert to tuple for hashing
            if new_node not in node_index_map:
                node_index_map[new_node] = len(computational_nodes)
                computational_nodes.append(new_node)
            dof_list.append(node_index_map[new_node])

        cell_dof_map.append(dof_list)

    return np.array(computational_nodes), np.array(cell_dof_map)"""


@jax.jit
def interpolate_at_quadrature(dofs, N):
    return jnp.einsum("ij, jk...->ik...", N, dofs, optimize="optimal")


@jax.jit
def integrate_over_cell(value_at_quad, detJ, wts):
    integral_value = jnp.einsum(
        "i...->...",
        jnp.einsum("i..., i...-> i...", value_at_quad, wts) * detJ,
        optimize="optimal",
    )

    return integral_value


'''@jax.jit
def integrate_per_cell(carry, iiter):
    value_at_quads, detJs, wts = carry
    integral_values = jnp.vecdot(value_at_quads[iiter], wts)*detJs[iiter]

    return (value_at_quads, detJ, wts), integral_values 


@jax.jit
def scanned_integration(value_at_quads, wts, detJs):
    num_iterations = value_at_quads.shape[0]
    iterations = jnp.arange(num_iterations)

    _, integral_values_cell = jax.lax.scan(
        integrate_per_cell,
        (value_at_quads, detJs, wts),
        iterations,
    )

    return integral_values_cell'''
