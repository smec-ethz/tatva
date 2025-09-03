import os
import pytest

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp
from jax import Array

from tatva import Mesh, Operator, element
from tatva.utils import auto_vmap


import sympy as sp
from sympy.vector import CoordSys3D, gradient, ParametricRegion, vector_integrate
from sympy.abc import x, y
import numpy as np


nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
tri_cells = np.array([[0, 1, 2], [0, 2, 3]])
quad_cells = np.array([[0, 1, 2, 3]])

n_nodes = nodes.shape[0]
n_dofs_per_node = 1
n_dofs = n_nodes * n_dofs_per_node

tri = element.Tri3()
tri_mesh = Mesh(nodes, tri_cells)
op_tri = Operator(tri_mesh, tri)

quad = element.Quad4()
quad_mesh = Mesh(nodes, quad_cells)
op_quad = Operator(quad_mesh, quad)


x_min = np.min(nodes[:, 0])
x_max = np.max(nodes[:, 0])
y_min = np.min(nodes[:, 1])
y_max = np.max(nodes[:, 1])
sp_region = ParametricRegion((x, y), (x, x_min, x_max), (y, y_min, y_max))
R = CoordSys3D("R")


# --- Test integration ---
def scalar_function(
    u: Array,
) -> Array:
    jax.debug.print("u: {}", u)
    return u


def total_area(u: Array, u_grad: Array, *_) -> Array:
    """Compute the total area of the system."""
    return scalar_function(u)


@pytest.mark.parametrize("operator", [op_tri, op_quad], ids=["tri", "quad"])
def test_integration(operator: Operator):
    val = 1.0
    u = jnp.full(fill_value=val, shape=(n_dofs,))
    integrate_func = operator.integrate(total_area)
    integral_value = integrate_func(u)
    print(integral_value)

    assert np.isclose(
        integral_value, float(vector_integrate(val, sp_region)), atol=1e-12
    ), f"Incorrect integration value is {integral_value}"


# --- Test gradient ---
@pytest.mark.parametrize("operator", [op_tri, op_quad], ids=["tri", "quad"])
def test_gradient(operator: Operator):
    f = nodes[:, 0] + nodes[:, 1]
    s1 = R.x + R.y

    quad_points = operator.eval(nodes)
    grad_values = operator.grad(f)

    for quad_point, grad_value in zip(quad_points, grad_values):
        for i in range(quad_point.shape[0]):
            dx = (
                gradient(s1)
                .evalf(subs={R.x: quad_point[i, 0], R.y: quad_point[i, 1]})
                .dot(R.i)
            )
            dy = (
                gradient(s1)
                .evalf(subs={R.x: quad_point[i, 0], R.y: quad_point[i, 1]})
                .dot(R.j)
            )

            assert np.isclose(grad_value[i, 0], float(dx), atol=1e-12)
            assert np.isclose(grad_value[i, 1], float(dy), atol=1e-12)


# --- Test interpolation ---
@pytest.mark.parametrize("operator", [op_tri, op_quad], ids=["tri", "quad"])
def test_interpolation(operator: Operator):
    points = jnp.array([[0.5, 0.25], [0.5, 0.75]])
    interpolated_values = operator.interpolate(nodes, points)

    assert np.isclose(interpolated_values[0, 0], 0.5, atol=1e-12)
    assert np.isclose(interpolated_values[0, 1], 0.25, atol=1e-12)
    assert np.isclose(interpolated_values[1, 0], 0.5, atol=1e-12)
    assert np.isclose(interpolated_values[1, 1], 0.75, atol=1e-12)


# --- Test stiffness matrix ---


# Define the vertived and triangles of the mesh
"""vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
triangles = [[0, 1, 2], [0, 2, 3]]

# Create a matrix of zeros with the correct shape
matrix = [[0 for i in range(4)] for j in range(4)]

# Create a Lagrange element
element = symfem.create_element("triangle", "Lagrange", 1)

for triangle in triangles:
    # Get the vertices of the triangle
    vs = tuple(vertices[i] for i in triangle)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("triangle", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)

    for test_i, test_f in zip(triangle, basis):
        for trial_i, trial_f in zip(triangle, basis):
            # Compute the integral of grad(u)-dot-grad(v) for each pair of basis
            # functions u and v. The second input (x) into `ref.integral` tells
            # symfem which variables to use in the integral.
            integrand = test_f.grad(2).dot(trial_f.grad(2))
            matrix[test_i][trial_i] += integrand.integral(ref, x)

print(np.array(matrix))"""

'''
def strain_energy(u: Array, u_grad: Array) -> Array:
    return u_grad @ u_grad.T


def total_energy(u: Array, u_grad: Array, *_) -> Array:
    """Compute the total energy of the system."""
    return strain_energy(u, u_grad)


@pytest.mark.parametrize("operator", [op_tri, op_quad])
def test_stiffness_matrix(operator: Operator):

    half = sp.Rational(1, 2)
    actual_matrix = np.array(
        [
            [1, -half, -half, 0],
            [-half, 1, 0, -half],
            [-half, 0, 1, -half],
            [0, -half, -half, 1],
        ]
    )
    u = jnp.full(fill_value=2.0, shape=(n_dofs,))
    K = jax.jacfwd(jax.jacrev(operator.integrate(total_energy)))(u)
    print(K)
    assert np.isclose(
        K, actual_matrix, atol=1e-12
    ), f"Incorrect stiffness matrix is {K}"
'''

# Create a matrix of zeros with the correct shape
"""quadrilateral = [[0, 1, 2, 3]]
matrix = [[0 for i in range(4)] for j in range(4)]

# Create a Lagrange element
element = symfem.create_element("quadrilateral", "Lagrange", 1)

for quad in quadrilateral:
    # Get the vertices of the quadrilateral
    vs = tuple(vertices[i] for i in quad)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the quadrilateral
    ref = symfem.create_reference("quadrilateral", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)

    for test_i, test_f in zip(quad, basis):
        for trial_i, trial_f in zip(quad, basis):
            # Compute the integral of grad(u)-dot-grad(v) for each pair of basis
            # functions u and v. The second input (x) into `ref.integral` tells
            # symfem which variables to use in the integral.
            integrand = test_f.grad(2).dot(trial_f.grad(2))
            matrix[test_i][trial_i] += integrand.integral(ref, x)

print(np.array(matrix))"""


if __name__ == "__main__":
    pytest.main()
