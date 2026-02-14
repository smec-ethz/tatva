import jax

jax.config.update("jax_enable_x64", True)  # use double-precision

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tatva.element import Hexahedron8, Line2, Quad4, Tetrahedron4, Tri3


@pytest.mark.parametrize(
    "element_class, dim",
    [
        (Line2, 1),
        (Tri3, 2),
        (Quad4, 2),
        (Tetrahedron4, 3),
        (Hexahedron8, 3),
    ],
)
def test_element_patch_test(element_class, dim):
    """
    Verifies that the element can represent a constant strain field exactly.
    """
    element = element_class()
    key = jax.random.PRNGKey(42)

    if dim == 1:
        dummy_xi = (
            element.quad_points
        )  # because Line2 has only one quadrature point, we can use it directly
    else:
        dummy_xi = element.quad_points[
            0
        ]  # Use the first quadrature point to determine number of nodes
    n_nodes = len(element.shape_function(dummy_xi))

    nodal_coords = jax.random.uniform(key, (n_nodes, dim), minval=0.5, maxval=1.5)

    A = jax.random.normal(key, (dim, dim))
    if dim == 1:
        A = A[0]  # For 1D, we only need a single value to represent the gradient

    nodal_values = nodal_coords @ A.T

    for xi in element.quad_points:
        dNdr = element.shape_function_derivative(xi)

        J = dNdr @ nodal_coords

        detJ = jnp.linalg.det(J) if dim > 1 else J[0]
        assert jnp.abs(detJ) > 1e-10, (
            f"Element {element_class.__name__} has a singular Jacobian."
        )

        if dim == 1:
            invJ = 1.0 / J
            dNdX = invJ * dNdr
        else:
            invJ = jnp.linalg.inv(J)
            dNdX = invJ @ dNdr

        computed_grad = dNdX @ nodal_values

        np.testing.assert_allclose(
            computed_grad,
            A.T,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"Failed patch test for {element_class.__name__}",
        )
