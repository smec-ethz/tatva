import os
import pytest
import numpy as np
import jax

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use double-precision
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")

import symfem
from fem_tools import (
    create_jacobian,
    create_tangent,
    create_basis,
    compute_computational_nodes,
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


def compute_cell_length(cells, nodes, element, basis):
    detJs = []
    for cell in cells:
        tangent = create_tangent(nodes[cell], element)
        J = create_jacobian(nodes[cell], element)
        detJs.append(tangent @ J(0.0))

    cell_lengths = []
    for detJ in detJs:
        length = sum(basis.wts[i] * detJ for i, _ in enumerate(basis.quad_pts))
        cell_lengths.append(length)

    return cell_lengths, detJs


@pytest.mark.parametrize("nb_elem", [1, 2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_gauss_integration(nb_elem, degree):
    nodes, cells = create_beam_structure(nb_elem)
    element = symfem.create_element(
        "interval", element_type="Lagrange", order=degree, variant="equispaced"
    )
    quadrature_rule = "legendre"
    nb_quads = int(np.ceil((degree + 1) / 2))
    basis = create_basis(element, quadrature_rule, nb_quads=nb_quads)

    cell_lengths, detJs = compute_cell_length(cells, nodes, element, basis)
    expected_length = 1.0 / nb_elem

    for length in cell_lengths:
        assert np.isclose(
            length, expected_length, atol=1e-6
        ), f"Incorrect cell length: {length}"


@pytest.mark.parametrize("nb_elem", [1, 2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fem_integration_scalar(nb_elem, degree):
    nodes, cells = create_beam_structure(nb_elem)
    element = symfem.create_element(
        "interval", element_type="Lagrange", order=degree, variant="equispaced"
    )
    quadrature_rule = "legendre"
    nb_quads = int(np.ceil((degree + 1) / 2))
    basis = create_basis(element, quadrature_rule, nb_quads=nb_quads)

    detJs = []
    for cell in cells:
        tangent = create_tangent(nodes[cell], element)
        J = create_jacobian(nodes[cell], element)
        detJs.append(tangent @ J(0.0))

    computational_nodes, cell_dof_map = compute_computational_nodes(
        nodes, cells, element
    )

    dofs = jnp.zeros((computational_nodes.shape[0], 1))

    polynomial_degree = degree
    dofs = dofs.at[:, 0].set(computational_nodes[:, 0] ** polynomial_degree)

    integral_value = 0
    for cell, detJ in zip(cell_dof_map, detJs):
        dofs_at_quad = jnp.einsum(
            "ij, jk...->ik...", basis.N, dofs.at[cell, :].get()
        ).reshape(nb_quads, -1)
        integral_value += jnp.einsum(
            "i...->...",
            jnp.einsum("i..., i... -> i...", dofs_at_quad, basis.wts) * detJ,
        )

    expected_integral = 1.0 / (polynomial_degree + 1)
    assert np.isclose(
        integral_value, expected_integral, atol=1e-6
    ), f"Incorrect integral value: {integral_value}"


@pytest.mark.parametrize("nb_elem", [1, 2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fem_integration_vector(nb_elem, degree):
    nodes, cells = create_beam_structure(nb_elem)
    element = symfem.create_element(
        "interval", element_type="Lagrange", order=degree, variant="equispaced"
    )
    quadrature_rule = "legendre"
    nb_quads = int(np.ceil((degree + 1) / 2))
    basis = create_basis(element, quadrature_rule, nb_quads=nb_quads)

    detJs = []
    for cell in cells:
        tangent = create_tangent(nodes[cell], element)
        J = create_jacobian(nodes[cell], element)
        detJs.append(tangent @ J(0.0))

    computational_nodes, cell_dof_map = compute_computational_nodes(
        nodes, cells, element
    )
    dofs = jnp.zeros((computational_nodes.shape[0], 2))

    polynomial_degree = degree
    dofs = dofs.at[:, 0].set(computational_nodes[:, 0] ** polynomial_degree)
    dofs = dofs.at[:, 1].set(1.0)

    integral_value = 0
    for cell, detJ in zip(cell_dof_map, detJs):
        dofs_at_quad = jnp.einsum(
            "ij, jk...->ik...", basis.N, dofs.at[cell, :].get()
        ).reshape(nb_quads, -1)

        integral_value += jnp.einsum(
            "i...->...",
            jnp.einsum("i..., i... -> i...", dofs_at_quad, basis.wts) * detJ,
        )

    expected_integral = jnp.array([1.0 / (polynomial_degree + 1), 1.0])
    assert np.allclose(
        integral_value, expected_integral, atol=1e-6
    ), f"Incorrect integral value: {integral_value}"


if __name__ == "__main__":
    pytest.main()
