import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tatva import Mesh, Operator, element, sparse

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def mesh():
    return Mesh.unit_square(2, 2, type="quad")

@pytest.fixture
def op(mesh):
    return Operator(mesh, element.Quad4())

def test_project_scalar_field(op, mesh):
    # Scalar field: s(x, y) = x + y
    quad_points = op.quads()
    field = quad_points[:, :, 0] + quad_points[:, :, 1]
    
    # Project with scalar matrix
    sp = sparse.create_sparsity_pattern(mesh, 1)
    cm = sparse.ColoredMatrix.from_csr(sp)
    
    projected = op.project(field, cm)
    
    expected = mesh.coords[:, 0] + mesh.coords[:, 1]
    assert projected.shape == (mesh.coords.shape[0],)
    np.testing.assert_allclose(projected, expected, atol=1e-7)

def test_project_vector_field(op, mesh):
    # Vector field: v(x, y) = [x, y]
    quad_points = op.quads()
    field = quad_points # (E, Q, 2)
    
    # Project with scalar matrix (Multiple RHS solve)
    sp_scalar = sparse.create_sparsity_pattern(mesh, 1)
    cm_scalar = sparse.ColoredMatrix.from_csr(sp_scalar)
    projected_scalar = op.project(field, cm_scalar)
    
    assert projected_scalar.shape == (mesh.coords.shape[0], 2)
    np.testing.assert_allclose(projected_scalar, mesh.coords, atol=1e-7)
    
    # Project with vector matrix (Coupled solve)
    sp_vector = sparse.create_sparsity_pattern(mesh, 2)
    cm_vector = sparse.ColoredMatrix.from_csr(sp_vector)
    projected_vector = op.project(field, cm_vector)
    
    assert projected_vector.shape == (mesh.coords.shape[0], 2)
    np.testing.assert_allclose(projected_vector, mesh.coords, atol=1e-7)

def test_project_tensor_field(op, mesh):
    # Tensor field: T(x, y) = [[x, 0], [0, y]]
    quad_points = op.quads()
    E, Q, _ = quad_points.shape
    field = jnp.zeros((E, Q, 2, 2))
    field = field.at[:, :, 0, 0].set(quad_points[:, :, 0])
    field = field.at[:, :, 1, 1].set(quad_points[:, :, 1])
    
    # Project with scalar matrix (4 independent RHS)
    sp = sparse.create_sparsity_pattern(mesh, 1)
    cm = sparse.ColoredMatrix.from_csr(sp)
    
    projected = op.project(field, cm)
    
    assert projected.shape == (mesh.coords.shape[0], 2, 2)
    np.testing.assert_allclose(projected[:, 0, 0], mesh.coords[:, 0], atol=1e-7)
    np.testing.assert_allclose(projected[:, 1, 1], mesh.coords[:, 1], atol=1e-7)
    np.testing.assert_allclose(projected[:, 0, 1], 0.0, atol=1e-7)

def test_project_default_matrix(op, mesh):
    # Test that project works without providing colored_matrix
    quad_points = op.quads()
    field = quad_points[:, :, 0]
    
    projected = op.project(field)
    assert projected.shape == (mesh.coords.shape[0],)
    np.testing.assert_allclose(projected, mesh.coords[:, 0], atol=1e-7)
