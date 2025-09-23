import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.element import Tri3
from tatva.mesh import Mesh
from tatva.operator import Operator

jax.config.update("jax_enable_x64", True)

NODES = jnp.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=jnp.float64,
)

ELEMENTS = jnp.array(
    [
        [0, 1, 2],
        [0, 2, 3],
    ],
    dtype=jnp.int32,
)

EXPECTED_ELEMENT_AREAS = np.array([0.5, 0.5], dtype=np.float64)


def test_operator_raises_for_dimension_mismatch():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    elements = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    mesh = Mesh(coords=coords, elements=elements)

    with pytest.raises(ValueError, match="expects 2D coordinates"):
        Operator(mesh, Tri3())


def test_operator_raises_for_node_count_mismatch():
    coords = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    elements = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    mesh = Mesh(coords=coords, elements=elements)

    with pytest.raises(ValueError, match="lists 4 nodes per element"):
        Operator(mesh, Tri3())


def test_operator_raises_for_invalid_connectivity():
    coords = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    elements = jnp.array([[0, 1, 3]], dtype=jnp.int32)
    mesh = Mesh(coords=coords, elements=elements)

    with pytest.raises(ValueError, match="outside the mesh coordinates"):
        Operator(mesh, Tri3())


@pytest.fixture(scope="module")
def tri_operator() -> Operator:
    mesh = Mesh(coords=NODES, elements=ELEMENTS)
    return Operator(mesh, Tri3())


def _manual_map_results(func, nodal_values: jnp.ndarray) -> np.ndarray:
    results = []
    for element in np.array(ELEMENTS):
        element_values = np.array(nodal_values)[element]
        element_results = [
            np.array(func(xi, element_values)) for xi in np.array(Tri3().quad_points)
        ]
        results.append(np.asarray(element_results))
    return np.asarray(results)


def test_map_applies_function_per_element_and_quad(tri_operator: Operator):
    """Test that the map method applies the function to each element and quadrature point
    correctly."""
    nodal_values = jnp.arange(NODES.shape[0], dtype=jnp.float64)

    def func(xi, el_values):
        return jnp.sum(el_values) + jnp.sum(xi)

    mapped = tri_operator.map(func)
    result = mapped(nodal_values)

    expected = _manual_map_results(
        lambda xi, el_vals: np.sum(el_vals) + np.sum(np.array(xi)),
        nodal_values,
    )
    np.testing.assert_allclose(result, expected)


def test_map_respects_element_quantity(tri_operator: Operator):
    """Test that the element_quantity argument correctly passes additional per-element
    data to the mapped function."""
    nodal_values = jnp.arange(NODES.shape[0], dtype=jnp.float64)
    element_bias = jnp.array([10.0, 20.0], dtype=jnp.float64)

    def func(xi, el_values, bias):
        return jnp.sum(el_values) + bias + xi[0]

    mapped = tri_operator.map(func, element_quantity=(1,))
    result = mapped(nodal_values, element_bias)

    def manual(xi, el_vals, bias):
        return np.sum(el_vals) + bias + np.array(xi)[0]

    expected = []
    for idx, element in enumerate(np.array(ELEMENTS)):
        element_values = np.array(nodal_values)[element]
        element_results = [
            manual(xi, element_values, float(element_bias[idx]))
            for xi in np.array(Tri3().quad_points)
        ]
        expected.append(np.asarray(element_results))

    np.testing.assert_allclose(result, np.asarray(expected))


def test_map_over_elements_matches_manual(tri_operator: Operator):
    """Test that map_over_elements applies the function to each element correctly."""
    nodal_values = jnp.arange(NODES.shape[0], dtype=jnp.float64)

    def func(el_values):
        return jnp.sum(el_values)

    mapped = tri_operator.map_over_elements(func)
    result = mapped(nodal_values)

    expected = np.array(
        [np.sum(np.array(nodal_values)[element]) for element in np.array(ELEMENTS)]
    )
    np.testing.assert_allclose(result, expected)


def test_eval_returns_weighted_average(tri_operator: Operator):
    nodal_values = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64)
    result = tri_operator.eval(nodal_values)
    expected = np.array([[1.0], [5.0 / 3.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_grad_of_linear_field_is_constant(tri_operator: Operator):
    coeffs = jnp.array([2.0, 3.0])
    nodal_values = NODES @ coeffs
    grad = tri_operator.grad(nodal_values)
    expected = np.array([[[2.0, 3.0]], [[2.0, 3.0]]], dtype=np.float64)
    np.testing.assert_allclose(grad, expected)


def test_integrate_nodal_field_matches_area(tri_operator: Operator):
    nodal_values = jnp.ones(NODES.shape[0], dtype=jnp.float64)
    per_element = tri_operator.integrate_per_element(nodal_values)
    total = tri_operator.integrate(nodal_values)

    np.testing.assert_allclose(per_element, EXPECTED_ELEMENT_AREAS)
    np.testing.assert_allclose(total, np.sum(EXPECTED_ELEMENT_AREAS))


def test_integrate_quad_values_matches_manual(tri_operator: Operator):
    quad_values = jnp.full(
        (ELEMENTS.shape[0], Tri3().quad_points.shape[0]), 4.0, dtype=jnp.float64
    )
    per_element = tri_operator.integrate_per_element(quad_values)
    expected = EXPECTED_ELEMENT_AREAS * 4.0
    np.testing.assert_allclose(per_element, expected)


def test_interpolate_recovers_linear_function(tri_operator: Operator):
    nodal_values = jnp.sum(NODES, axis=1)
    points = jnp.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.5, 0.5],
        ],
        dtype=jnp.float64,
    )
    interpolated = tri_operator.interpolate(nodal_values, points)
    expected = np.sum(np.array(points), axis=1)
    np.testing.assert_allclose(interpolated, expected)


def test_interpolate_raises_for_points_outside_mesh(tri_operator: Operator):
    nodal_values = jnp.ones(NODES.shape[0], dtype=jnp.float64)
    points = jnp.array([[1.5, 0.5]], dtype=jnp.float64)
    with pytest.raises(RuntimeError):
        tri_operator.interpolate(nodal_values, points)
