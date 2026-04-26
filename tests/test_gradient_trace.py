import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva import Mesh
from tatva.element import Hexahedron8, Line2, Quad4, Tetrahedron4, Tri3
from tatva.utils import make_gradient_trace

jax.config.update("jax_enable_x64", True)

# Two rows of triangles:
#   3---4---5
#   |\ |\ |
#   | \| \|
#   0---1---2
COORDS = jnp.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ],
    dtype=jnp.float64,
)

ELEMENTS = jnp.array(
    [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]],
    dtype=jnp.int32,
)

# Bottom edge: two Line2 faces, each belonging to a different element
TRACE_ELEMENTS = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)

N_TRACE_QP = len(Line2().quad_points)  # 1


@pytest.fixture(scope="module")
def mesh():
    return Mesh(coords=COORDS, elements=ELEMENTS)


@pytest.fixture(scope="module")
def grad_trace(mesh):
    return make_gradient_trace(mesh, TRACE_ELEMENTS, Tri3(), Line2())


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape_scalar_field(grad_trace):
    u = jnp.zeros(COORDS.shape[0], dtype=jnp.float64)
    result = grad_trace(u)
    assert result.shape == (len(TRACE_ELEMENTS) * N_TRACE_QP, COORDS.shape[1])


def test_output_shape_vector_field(grad_trace):
    u = jnp.zeros((COORDS.shape[0], 2), dtype=jnp.float64)
    result = grad_trace(u)
    assert result.shape == (len(TRACE_ELEMENTS) * N_TRACE_QP, COORDS.shape[1], 2)


# ---------------------------------------------------------------------------
# Correctness: gradient of linear fields must be exact
# ---------------------------------------------------------------------------


def test_scalar_linear_field_gradient_is_exact(grad_trace):
    # u = 2x + 3y  =>  grad u = [2, 3] everywhere
    u = 2.0 * COORDS[:, 0] + 3.0 * COORDS[:, 1]
    result = grad_trace(u)
    n_qp_total = len(TRACE_ELEMENTS) * N_TRACE_QP
    expected = np.tile([2.0, 3.0], (n_qp_total, 1))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_vector_linear_field_gradient_is_exact(grad_trace):
    # u = [2x + 3y, x - y]
    # result[i, j] = ∂u_i/∂x_j  =>  [[∂u0/∂x, ∂u0/∂y], [∂u1/∂x, ∂u1/∂y]] = [[2, 3], [1, -1]]
    u = jnp.stack(
        [2.0 * COORDS[:, 0] + 3.0 * COORDS[:, 1], COORDS[:, 0] - COORDS[:, 1]],
        axis=1,
    )
    result = grad_trace(u)
    n_qp_total = len(TRACE_ELEMENTS) * N_TRACE_QP
    expected = np.tile([[2.0, 3.0], [1.0, -1.0]], (n_qp_total, 1, 1))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_zero_field_gradient_is_zero(grad_trace):
    u = jnp.zeros(COORDS.shape[0], dtype=jnp.float64)
    result = grad_trace(u)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation: factory-time checks
# ---------------------------------------------------------------------------


def test_wrong_bulk_element_type_raises(mesh):
    with pytest.raises(ValueError, match="Line2"):
        make_gradient_trace(mesh, TRACE_ELEMENTS, Line2(), Line2())


def test_wrong_trace_element_type_raises(mesh):
    with pytest.raises(ValueError, match="Tri3"):
        make_gradient_trace(mesh, TRACE_ELEMENTS, Tri3(), Tri3())


def test_out_of_bounds_node_index_raises(mesh):
    bad_trace = jnp.array([[0, 999]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="references node"):
        make_gradient_trace(mesh, bad_trace, Tri3(), Line2())


def test_trace_element_without_parent_raises(mesh):
    # Nodes 2 and 3 are never co-located in any element
    bad_trace = jnp.array([[2, 3]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="no parent"):
        make_gradient_trace(mesh, bad_trace, Tri3(), Line2())


# ---------------------------------------------------------------------------
# Validation: call-time check (non-nodal field)
# ---------------------------------------------------------------------------


def test_non_nodal_field_raises(grad_trace):
    # Simulates accidentally passing bulk_op.grad(u) which has shape
    # (n_elements, n_qp, ...) instead of nodal shape (n_nodes, ...)
    bad_field = jnp.zeros((ELEMENTS.shape[0], len(Tri3().quad_points), 2))
    with pytest.raises(ValueError, match="Field has"):
        grad_trace(bad_field)


# ===========================================================================
# Quad4 bulk → Line2 trace  (2D)
# ===========================================================================

# Two quads:
#   3---4---5
#   |   |   |
#   0---1---2
# Quad4 node order: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)

COORDS_Q4 = jnp.array(
    [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ],
    dtype=jnp.float64,
)

ELEMENTS_Q4 = jnp.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=jnp.int32)
TRACE_Q4 = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)


@pytest.fixture(scope="module")
def mesh_q4():
    return Mesh(coords=COORDS_Q4, elements=ELEMENTS_Q4)


@pytest.fixture(scope="module")
def grad_trace_q4(mesh_q4):
    return make_gradient_trace(mesh_q4, TRACE_Q4, Quad4(), Line2())


def test_q4_output_shape_scalar(grad_trace_q4):
    u = jnp.zeros(COORDS_Q4.shape[0], dtype=jnp.float64)
    result = grad_trace_q4(u)
    n_qp = len(TRACE_Q4) * len(Line2().quad_points)
    assert result.shape == (n_qp, COORDS_Q4.shape[1])


def test_q4_output_shape_vector(grad_trace_q4):
    u = jnp.zeros((COORDS_Q4.shape[0], 2), dtype=jnp.float64)
    result = grad_trace_q4(u)
    n_qp = len(TRACE_Q4) * len(Line2().quad_points)
    assert result.shape == (n_qp, COORDS_Q4.shape[1], 2)


def test_q4_scalar_gradient_is_exact(grad_trace_q4):
    u = 2.0 * COORDS_Q4[:, 0] + 3.0 * COORDS_Q4[:, 1]
    result = grad_trace_q4(u)
    n_qp = len(TRACE_Q4) * len(Line2().quad_points)
    np.testing.assert_allclose(result, np.tile([2.0, 3.0], (n_qp, 1)), atol=1e-12)


def test_q4_vector_gradient_is_exact(grad_trace_q4):
    # result[i, j] = ∂u_i/∂x_j => [[2, 3], [1, -1]]
    u = jnp.stack(
        [2.0 * COORDS_Q4[:, 0] + 3.0 * COORDS_Q4[:, 1], COORDS_Q4[:, 0] - COORDS_Q4[:, 1]],
        axis=1,
    )
    result = grad_trace_q4(u)
    n_qp = len(TRACE_Q4) * len(Line2().quad_points)
    np.testing.assert_allclose(result, np.tile([[2.0, 3.0], [1.0, -1.0]], (n_qp, 1, 1)), atol=1e-12)


# ===========================================================================
# Tetrahedron4 bulk → Tri3 trace  (3D)
# ===========================================================================

# Two tets sharing an apex at (0.5, 0.5, 1):
#   bottom face nodes: 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
#   apex: 4:(0.5,0.5,1)
# Tet 0 [0,1,2,4] has bottom face [0,1,2]
# Tet 1 [0,2,3,4] has bottom face [0,2,3]

COORDS_TET = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ],
    dtype=jnp.float64,
)

ELEMENTS_TET = jnp.array([[0, 1, 2, 4], [0, 2, 3, 4]], dtype=jnp.int32)
TRACE_TET = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)


@pytest.fixture(scope="module")
def mesh_tet():
    return Mesh(coords=COORDS_TET, elements=ELEMENTS_TET)


@pytest.fixture(scope="module")
def grad_trace_tet(mesh_tet):
    return make_gradient_trace(mesh_tet, TRACE_TET, Tetrahedron4(), Tri3())


def test_tet_output_shape_scalar(grad_trace_tet):
    u = jnp.zeros(COORDS_TET.shape[0], dtype=jnp.float64)
    result = grad_trace_tet(u)
    n_qp = len(TRACE_TET) * len(Tri3().quad_points)
    assert result.shape == (n_qp, COORDS_TET.shape[1])


def test_tet_output_shape_vector(grad_trace_tet):
    u = jnp.zeros((COORDS_TET.shape[0], 3), dtype=jnp.float64)
    result = grad_trace_tet(u)
    n_qp = len(TRACE_TET) * len(Tri3().quad_points)
    assert result.shape == (n_qp, COORDS_TET.shape[1], 3)


def test_tet_scalar_gradient_is_exact(grad_trace_tet):
    # u = 2x + 3y + 4z  =>  grad = [2, 3, 4]
    u = 2.0 * COORDS_TET[:, 0] + 3.0 * COORDS_TET[:, 1] + 4.0 * COORDS_TET[:, 2]
    result = grad_trace_tet(u)
    n_qp = len(TRACE_TET) * len(Tri3().quad_points)
    np.testing.assert_allclose(result, np.tile([2.0, 3.0, 4.0], (n_qp, 1)), atol=1e-12)


def test_tet_vector_gradient_is_exact(grad_trace_tet):
    # u = [2x+3y+4z, x-y+2z, 3x+y-z]
    # result[i, j] = ∂u_i/∂x_j:
    #   row 0 (u0): [2, 3, 4]
    #   row 1 (u1): [1, -1, 2]
    #   row 2 (u2): [3, 1, -1]
    c = COORDS_TET
    u = jnp.stack(
        [2*c[:,0] + 3*c[:,1] + 4*c[:,2], c[:,0] - c[:,1] + 2*c[:,2], 3*c[:,0] + c[:,1] - c[:,2]],
        axis=1,
    )
    result = grad_trace_tet(u)
    n_qp = len(TRACE_TET) * len(Tri3().quad_points)
    expected_grad = [[2.0, 3.0, 4.0], [1.0, -1.0, 2.0], [3.0, 1.0, -1.0]]
    np.testing.assert_allclose(result, np.tile(expected_grad, (n_qp, 1, 1)), atol=1e-12)


# ===========================================================================
# Hexahedron8 bulk → Quad4 trace  (3D)
# ===========================================================================

# Two hexes side by side (x direction), bottom face at z=0:
#   bottom (z=0): 0:(0,0,0)  1:(1,0,0)  2:(2,0,0)
#                 3:(0,1,0)  4:(1,1,0)  5:(2,1,0)
#   top    (z=1): 6:(0,0,1)  7:(1,0,1)  8:(2,0,1)
#                 9:(0,1,1) 10:(1,1,1) 11:(2,1,1)
# Hex8 node order per element: bottom ring CCW then top ring CCW
#   Hex 0: [0,1,4,3, 6,7,10,9]
#   Hex 1: [1,2,5,4, 7,8,11,10]
# Quad4 bottom face (ζ=-1 in reference) of each hex:
#   Hex 0 bottom: [0,1,4,3]
#   Hex 1 bottom: [1,2,5,4]

COORDS_HEX = jnp.array(
    [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
    ],
    dtype=jnp.float64,
)

ELEMENTS_HEX = jnp.array(
    [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]],
    dtype=jnp.int32,
)

TRACE_HEX = jnp.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=jnp.int32)


@pytest.fixture(scope="module")
def mesh_hex():
    return Mesh(coords=COORDS_HEX, elements=ELEMENTS_HEX)


@pytest.fixture(scope="module")
def grad_trace_hex(mesh_hex):
    return make_gradient_trace(mesh_hex, TRACE_HEX, Hexahedron8(), Quad4())


def test_hex_output_shape_scalar(grad_trace_hex):
    u = jnp.zeros(COORDS_HEX.shape[0], dtype=jnp.float64)
    result = grad_trace_hex(u)
    n_qp = len(TRACE_HEX) * len(Quad4().quad_points)
    assert result.shape == (n_qp, COORDS_HEX.shape[1])


def test_hex_output_shape_vector(grad_trace_hex):
    u = jnp.zeros((COORDS_HEX.shape[0], 3), dtype=jnp.float64)
    result = grad_trace_hex(u)
    n_qp = len(TRACE_HEX) * len(Quad4().quad_points)
    assert result.shape == (n_qp, COORDS_HEX.shape[1], 3)


def test_hex_scalar_gradient_is_exact(grad_trace_hex):
    # u = 2x + 3y + 4z  =>  grad = [2, 3, 4] everywhere
    u = 2.0 * COORDS_HEX[:, 0] + 3.0 * COORDS_HEX[:, 1] + 4.0 * COORDS_HEX[:, 2]
    result = grad_trace_hex(u)
    n_qp = len(TRACE_HEX) * len(Quad4().quad_points)
    np.testing.assert_allclose(result, np.tile([2.0, 3.0, 4.0], (n_qp, 1)), atol=1e-12)


def test_hex_vector_gradient_is_exact(grad_trace_hex):
    # u = [2x+3y+4z, x-y+2z, 3x+y-z]
    # result[i, j] = ∂u_i/∂x_j:
    #   row 0 (u0): [2, 3, 4]
    #   row 1 (u1): [1, -1, 2]
    #   row 2 (u2): [3, 1, -1]
    c = COORDS_HEX
    u = jnp.stack(
        [2*c[:,0] + 3*c[:,1] + 4*c[:,2], c[:,0] - c[:,1] + 2*c[:,2], 3*c[:,0] + c[:,1] - c[:,2]],
        axis=1,
    )
    result = grad_trace_hex(u)
    n_qp = len(TRACE_HEX) * len(Quad4().quad_points)
    expected_grad = [[2.0, 3.0, 4.0], [1.0, -1.0, 2.0], [3.0, 1.0, -1.0]]
    np.testing.assert_allclose(result, np.tile(expected_grad, (n_qp, 1, 1)), atol=1e-12)
