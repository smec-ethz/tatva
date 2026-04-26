import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tatva.element import (
    Hexahedron8,
    Line2,
    Line3,
    Quad4,
    Quad8,
    Tetrahedron4,
    Tri3,
    Tri6,
)

# ---------------------------------------------------------------------------
# Physical coordinates used in tests.
#
# For Line2/Line3: get_jacobian hardcodes a 2-component tangent vector, so
# physical coords must be 2D. We embed the line along the x-axis.
#
# For all other elements: use _reference_nodes() as physical coords. This
# gives J = I for affine elements (Tri3, Tet4, Hex8) and a well-conditioned
# Jacobian for higher-order ones (Tri6, Quad8). Any linear field is
# represented exactly by all standard Lagrangian elements.
# ---------------------------------------------------------------------------

LINE2_COORDS = jnp.array([[0.0, 0.0], [1.0, 0.0]])          # x ∈ [0, 1]
LINE3_COORDS = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])  # ends then midpoint


def _ref_coords(element_class):
    return element_class()._reference_nodes()


# ---------------------------------------------------------------------------
# Scalar gradient: u = a · x  =>  ∇u = a  at every quadrature point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "element_class, coords, coeffs",
    [
        # 1-D elements embedded in 2D — arc-length derivative of u = 3x equals 3
        (Line2, LINE2_COORDS,            [3.0]),
        (Line3, LINE3_COORDS,            [3.0]),
        # 2-D elements — gradient of u = 2x + 3y equals [2, 3]
        (Tri3,  _ref_coords(Tri3),       [2.0, 3.0]),
        (Tri6,  _ref_coords(Tri6),       [2.0, 3.0]),
        (Quad4, _ref_coords(Quad4),      [2.0, 3.0]),
        (Quad8, _ref_coords(Quad8),      [2.0, 3.0]),
        # 3-D elements — gradient of u = 2x + 3y + 4z equals [2, 3, 4]
        (Tetrahedron4, _ref_coords(Tetrahedron4), [2.0, 3.0, 4.0]),
        (Hexahedron8,  _ref_coords(Hexahedron8),  [2.0, 3.0, 4.0]),
    ],
)
def test_scalar_gradient_is_exact(element_class, coords, coeffs):
    """element.gradient on a linear scalar field must return the exact coefficient vector."""
    element = element_class()
    a = jnp.array(coeffs)

    # For line elements: u = 3 * x-coordinate (arc-length along x-axis)
    # For 2-D/3-D elements: u = a · x
    if element_class in (Line2, Line3):
        u = 3.0 * coords[:, 0]      # shape (n_nodes,)
        expected = np.array([3.0])   # arc-length derivative
    else:
        u = coords @ a               # shape (n_nodes,)
        expected = np.array(coeffs)  # ∇u = a

    for xi in element.quad_points:
        grad = element.gradient(xi, u, coords)
        np.testing.assert_allclose(
            grad, expected, atol=1e-12,
            err_msg=f"Scalar gradient failed for {element_class.__name__}",
        )


# ---------------------------------------------------------------------------
# Vector gradient: u_i = A[i,j] * x_j  =>  ∂u_i/∂x_j = A[i,j]
# (component-first convention; line elements excluded — their gradient is
#  a 1-D arc-length derivative, not a full spatial Jacobian)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "element_class",
    [Tri3, Tri6, Quad4, Quad8, Tetrahedron4, Hexahedron8],
)
def test_vector_gradient_is_exact(element_class):
    """element.gradient on a linear vector field must return A with layout
    result[i,j] = ∂u_i/∂x_j (component-first)."""
    element = element_class()
    key = jax.random.PRNGKey(0)
    coords = element._reference_nodes()   # shape (n_nodes, dim)
    dim = coords.shape[1]

    A = jax.random.normal(key, (dim, dim))
    # u[k, i] = Σ_j A[i,j] * coords[k,j]  =>  ∂u_i/∂x_j = A[i,j]
    u = jnp.einsum("ij,kj->ki", A, coords)  # shape (n_nodes, dim)

    for xi in element.quad_points:
        grad = element.gradient(xi, u, coords)
        # grad[i,j] = ∂u_i/∂x_j = A[i,j]
        np.testing.assert_allclose(
            grad, np.array(A), atol=1e-12,
            err_msg=f"Vector gradient failed for {element_class.__name__}",
        )
