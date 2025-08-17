'''import os
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


@pytest.mark.parametrize("nb_elem", [1, 2, 4, 8])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fem_integration_scalar(nb_elem, degree):
    nodes, cells = create_mesh(nb_elem)
    element = symfem.create_element(
        "quadrilateral", element_type="Lagrange", order=degree, variant="equispaced"
    )
    quadrature_rule = "legendre"
    nb_quads = int(np.ceil((degree + 1) / 2))
    basis = create_basis(element, quadrature_rule, nb_quads=nb_quads)

    detJs = []
    for cell in cells:
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
'''