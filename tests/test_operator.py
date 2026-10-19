from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.element import Hexahedron8, Line2, Quad4, Quad8, Tetrahedron4, Tri3, Tri6
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


def test_hess_maps_nonzero_hessian_over_quadrature_points():
    element = Quad8()
    coords = element._reference_nodes()
    operator = Operator(Mesh(coords=coords, elements=jnp.arange(8)[None, :]), element)
    x, y = coords[:, 0], coords[:, 1]
    nodal_values = x**2 + 2.0 * x * y + 3.0 * y**2

    hess = operator.hess(nodal_values)
    expected = np.broadcast_to(
        np.array([[2.0, 2.0], [2.0, 6.0]]), (1, len(element.quad_points), 2, 2)
    )
    np.testing.assert_allclose(hess, expected, atol=1e-12)


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


def test_interpolate_returns_nan_for_points_outside_mesh(tri_operator: Operator):
    nodal_values = jnp.ones(NODES.shape[0], dtype=jnp.float64)
    points = jnp.array([[1.5, 0.5]], dtype=jnp.float64)
    assert bool(jnp.all(jnp.isnan(tri_operator.interpolate(nodal_values, points))))
    assert bool(
        jnp.all(jnp.isnan(jax.jit(tri_operator.interpolate)(nodal_values, points)))
    )
    with pytest.raises(RuntimeError):
        tri_operator.make_interpolate(points)


def test_operator_can_be_used_as_dynamic_jit_argument(tri_operator: Operator):
    nodal_values = jnp.ones(NODES.shape[0], dtype=jnp.float64)

    @jax.jit
    def integrated_area(op: Operator, values: jax.Array) -> jax.Array:
        return op.integrate(values)

    result = integrated_area(tri_operator, nodal_values)

    np.testing.assert_allclose(result, np.sum(EXPECTED_ELEMENT_AREAS))


@pytest.mark.skip(reason="Not supported")
def test_operator_can_be_used_as_static_jit_argument(tri_operator: Operator):
    nodal_values = jnp.ones(NODES.shape[0], dtype=jnp.float64)

    @partial(jax.jit, static_argnames=("op",))
    def integrated_area(op: Operator, values: jax.Array) -> jax.Array:
        return op.integrate(values)

    result = integrated_area(tri_operator, nodal_values)

    np.testing.assert_allclose(result, np.sum(EXPECTED_ELEMENT_AREAS))


@pytest.mark.parametrize("element_type", [Tri3, Quad4, Tetrahedron4, Hexahedron8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_static_interpolation_transformed_mesh(element_type, dtype):
    element = element_type()
    reference_nodes = element._reference_nodes().astype(dtype)
    dim = reference_nodes.shape[1]
    transform = jnp.eye(dim, dtype=dtype) * 1.7 + jnp.full(
        (dim, dim), 0.13, dtype=dtype
    )
    first = reference_nodes @ transform + jnp.arange(dim, dtype=dtype) + 2
    second = first + 10
    coords = jnp.concatenate([first, second])
    elements = jnp.arange(len(coords)).reshape(2, -1)
    op = Operator(Mesh(coords=coords, elements=elements), element)
    xi = jnp.full((dim,), 0.17, dtype=dtype)
    weights = element.shape_function(xi)
    points = jnp.stack([weights @ first, weights @ second])
    interpolate = op.make_interpolate(points)
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-10

    scalar = coords @ jnp.arange(1, dim + 1, dtype=dtype) + 3
    expected = points @ jnp.arange(1, dim + 1, dtype=dtype) + 3
    for values in [
        scalar,
        jnp.stack([scalar, 2 * scalar], axis=-1),
        jnp.broadcast_to(scalar[:, None, None], (len(coords), 2, 3)),
    ]:
        result = jax.jit(interpolate)(values)
        target = expected.reshape((2,) + (1,) * (values.ndim - 1))
        np.testing.assert_allclose(
            result,
            np.broadcast_to(target, result.shape)
            * (np.array([1, 2]) if values.ndim == 2 else 1),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize("element_type", [Quad4, Hexahedron8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_static_interpolation_distorted_tensor_product(element_type, dtype):
    element = element_type()
    coords = np.array(element._reference_nodes(), dtype=dtype)
    dim = coords.shape[1]
    # Bilinear/trilinear geometry: a single Newton step is not an inverse.
    coords[:, 0] += 0.35 * coords[:, 0] * coords[:, 1]
    coords[:, 1] += 0.2 * coords[:, 0] * coords[:, 1]
    if dim == 3:
        coords[:, 2] += 0.25 * coords[:, 0] * coords[:, 1]
    coords = jnp.asarray(coords)
    op = Operator(Mesh(coords=coords, elements=jnp.arange(len(coords))[None]), element)
    xi = jnp.asarray([[0.72, -0.63], [-0.55, 0.8], [0, 0]], dtype=dtype)
    if dim == 3:
        xi = jnp.concatenate(
            [xi, jnp.asarray([[0.51], [-0.4], [0]], dtype=dtype)], axis=1
        )
    weights = jax.vmap(element.shape_function)(xi)
    points = weights @ coords
    values = jnp.sin(jnp.arange(len(coords) * 6, dtype=dtype)).reshape(
        len(coords), 2, 3
    )
    expected = jnp.einsum("pa,aij->pij", weights, values)
    result = op.make_interpolate(points)(values)
    tolerance = 3e-5 if dtype == jnp.float32 else 1e-9
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        op.make_interpolate(points)(coords), points, atol=tolerance
    )


@pytest.mark.parametrize("element_type", [Tri3, Quad4, Tetrahedron4, Hexahedron8])
def test_static_interpolation_shared_boundaries_choose_first(element_type):
    element = element_type()
    first = element._reference_nodes()
    if element_type in (Tri3, Tetrahedron4):
        second = first.at[:, 0].multiply(-1)
        point = jnp.full((first.shape[1],), 0.2).at[0].set(0)
    else:
        second = first.at[:, 0].add(2)
        point = jnp.full((first.shape[1],), 0.2).at[0].set(1)
    coords = jnp.concatenate([first, second])
    elements = jnp.arange(len(coords)).reshape(2, -1)
    op = Operator(Mesh(coords=coords, elements=elements), element)
    values = jnp.concatenate([jnp.zeros(len(first)), jnp.ones(len(second))])
    # Distinct global nodes make the otherwise ambiguous boundary observable.
    np.testing.assert_allclose(op.make_interpolate(point[None])(values), 0, atol=1e-12)
    # Vertices and edges must be accepted too.
    np.testing.assert_allclose(op.make_interpolate(first)(coords), first, atol=1e-10)


@pytest.mark.parametrize("element_type", [Tri3, Quad4, Tetrahedron4, Hexahedron8])
def test_static_interpolation_rejects_aabb_false_positive(element_type):
    element = element_type()
    coords = element._reference_nodes()
    if element_type in (Tri3, Tetrahedron4):
        point = jnp.full((coords.shape[1],), 0.8)
    else:
        coords = coords.at[:, 0].add(0.7 * coords[:, 1])
        point = jnp.full((coords.shape[1],), 0.9).at[0].set(-1.5)
    assert bool(
        jnp.all(point >= coords.min(axis=0)) & jnp.all(point <= coords.max(axis=0))
    )
    op = Operator(Mesh(coords=coords, elements=jnp.arange(len(coords))[None]), element)
    with pytest.raises(RuntimeError, match=r"points \[0\].*outside the mesh"):
        op.make_interpolate(point[None])


@pytest.mark.parametrize("element_type", [Tri3, Quad4, Tetrahedron4, Hexahedron8])
def test_static_interpolation_skips_singular_candidate(element_type):
    element = element_type()
    regular = element._reference_nodes()
    singular = regular.at[:, 0].set(0)
    coords = jnp.concatenate([singular, regular])
    op = Operator(
        Mesh(coords=coords, elements=jnp.arange(len(coords)).reshape(2, -1)), element
    )
    point = jnp.full((regular.shape[1],), 0.2).at[0].set(0)
    values = jnp.concatenate([jnp.zeros(len(regular)), jnp.ones(len(regular))])
    np.testing.assert_allclose(op.make_interpolate(point[None])(values), 1, atol=1e-10)
    singular_op = Operator(
        Mesh(coords=singular, elements=jnp.arange(len(regular))[None]), element
    )
    with pytest.raises(RuntimeError, match="singular or unconverged"):
        singular_op.make_interpolate(point[None])


def test_static_interpolation_jax_runtime(tri_operator):
    points = jnp.array([[0.2, 0.1], [0.2, 0.8]])
    interpolate = tri_operator.make_interpolate(points)
    values = jnp.arange(4.0)
    expected_weights = np.array([[0.8, 0.1, 0.1, 0.0], [0.2, 0.0, 0.2, 0.6]])
    np.testing.assert_allclose(jax.jit(interpolate)(values), expected_weights @ values)
    np.testing.assert_allclose(
        jax.jit(interpolate)(2 * values), 2 * expected_weights @ values
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(interpolate))(jnp.stack([values, 2 * values])),
        np.stack([expected_weights @ values, 2 * expected_weights @ values]),
    )
    np.testing.assert_allclose(jax.jacrev(interpolate)(values), expected_weights)
    np.testing.assert_allclose(jax.jvp(interpolate, (values,), (jnp.ones(4),))[1], 1)
    program = jax.make_jaxpr(interpolate)(values)
    primitives = {eqn.primitive.name for eqn in program.jaxpr.eqns}
    assert "gather" in primitives
    assert primitives <= {
        "lt",
        "add",
        "select_n",
        "broadcast_in_dim",
        "gather",
        "dot_general",
        "mul",
        "reduce_sum",
        "convert_element_type",
    }
    # Runtime constants have no coordinates, reference points, or geometry tree.
    assert sorted(c.shape for c in program.consts) == [(2, 3), (2, 3)]


def test_static_interpolation_empty_points(tri_operator):
    interpolate = tri_operator.make_interpolate(jnp.empty((0, 2)))
    assert jax.jit(interpolate)(jnp.ones((4, 2, 3))).shape == (0, 2, 3)


@pytest.mark.parametrize(
    "points", [jnp.zeros(2), jnp.zeros((1, 3)), jnp.zeros((1, 2, 1))]
)
def test_static_interpolation_invalid_point_shape(tri_operator, points):
    with pytest.raises(ValueError, match="points must have shape"):
        tri_operator.make_interpolate(points)


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf])
def test_static_interpolation_nonfinite_geometry(tri_operator, bad_value):
    with pytest.raises(ValueError, match="must be finite"):
        tri_operator.make_interpolate(jnp.array([[bad_value, 0]]))
    op = tri_operator._replace(
        mesh=Mesh(coords=NODES.at[0, 0].set(bad_value), elements=ELEMENTS)
    )
    with pytest.raises(ValueError, match="must be finite"):
        op.make_interpolate(jnp.array([[0.2, 0.1]]))


@pytest.mark.parametrize("element_type", [Tri6, Quad8, Line2])
def test_static_interpolation_unsupported_element(element_type):
    element = element_type()
    coords = element._reference_nodes()
    op = Operator(Mesh(coords=coords, elements=jnp.arange(len(coords))[None]), element)
    with pytest.raises(NotImplementedError, match="Static interpolation supports"):
        op.make_interpolate(jnp.zeros((1, coords.shape[1])))


def test_static_interpolation_embedded_element(tri_operator):
    coords = jnp.pad(NODES, ((0, 0), (0, 1)))
    op = Operator(Mesh(coords=coords, elements=ELEMENTS), Tri3())
    with pytest.raises(NotImplementedError, match="embedded elements"):
        op.make_interpolate(jnp.zeros((1, 3)))


def test_static_interpolation_connectivity_width(tri_operator):
    op = Operator(tri_operator.mesh, Quad4())
    with pytest.raises(ValueError, match="connectivity width"):
        op.make_interpolate(jnp.zeros((1, 2)))


@pytest.mark.parametrize("shape", [(), (3,), (5, 2)])
def test_static_interpolation_invalid_values(tri_operator, shape):
    interpolate = tri_operator.make_interpolate(jnp.array([[0.2, 0.1]]))
    with pytest.raises(ValueError, match="Nodal values must have shape"):
        jax.jit(interpolate)(jnp.zeros(shape))


def test_static_interpolation_rejects_traced_setup(tri_operator):
    points = jnp.array([[0.2, 0.1]])
    with pytest.raises(ValueError, match="outside JAX transformations"):
        jax.jit(lambda p: tri_operator.make_interpolate(p)(jnp.ones(4)))(points)
    with pytest.raises(ValueError, match="outside JAX transformations"):
        jax.jit(lambda op: op.make_interpolate(points)(jnp.ones(4)))(tri_operator)


def test_static_interpolation_tree_prunes_candidates(monkeypatch):
    from tatva._interpolation import _AABBTree

    lower = np.stack(
        np.meshgrid(np.arange(20) * 2, np.arange(20) * 2), axis=-1
    ).reshape(-1, 2)
    upper = lower + 1
    tree = _AABBTree.build(lower, upper)
    visited = []
    original = _AABBTree.candidates

    def counted(self, *args):
        visited.append(self)
        yield from original(self, *args)

    monkeypatch.setattr(_AABBTree, "candidates", counted)
    for index in (0, 1, 37, 200, 399):
        visited.clear()
        assert list(tree.candidates(lower[index] + 0.5, lower, upper)) == [index]
        assert len(visited) < 20  # Only a tree path and its siblings, not all 63 nodes.
    assert list(tree.candidates(np.array([-1, -1]), lower, upper)) == []


def test_static_interpolation_multiple_candidate_batches():
    # All elements overlap, forcing more hits than a single numerical batch.
    count = 1030
    reference = Tri3()._reference_nodes()
    coords = jnp.tile(reference, (count, 1))
    op = Operator(
        Mesh(coords=coords, elements=jnp.arange(len(coords)).reshape(count, 3)), Tri3()
    )
    values = jnp.repeat(jnp.arange(count, dtype=float), 3)
    np.testing.assert_allclose(
        op.make_interpolate(jnp.array([[0.2, 0.3]]))(values), 0, atol=1e-12
    )


def test_static_interpolation_structured_mesh():
    mesh = Mesh.unit_square(25, 25)
    op = Operator(mesh, Tri3())
    points = mesh.coords[mesh.elements].mean(axis=1)
    assert len(points) > 1024
    np.testing.assert_allclose(
        op.make_interpolate(points)(mesh.coords), points, atol=1e-10
    )


def test_live_interpolation_dynamic_mesh_and_points(tri_operator):
    @jax.jit
    def interpolate(op, values, points):
        return op.interpolate(values, points)

    points = jnp.array([[0.2, 0.1], [0.2, 0.8], [0.5, 0.5]])
    for scale, shift in [(1.0, 0.0), (2.0, 5.0)]:
        mesh = Mesh(coords=NODES * scale + shift, elements=ELEMENTS)
        op = tri_operator._replace(mesh=mesh)
        query = points * scale + shift
        for offset in [0.0, 3.0]:
            values = jnp.stack([mesh.coords[:, 0] + offset, mesh.coords[:, 1]], axis=1)
            expected = np.asarray(query).copy()
            expected[:, 0] += offset
            np.testing.assert_allclose(
                interpolate(op, values, query), expected, atol=1e-12
            )
        np.testing.assert_allclose(
            interpolate(op, mesh.coords, query[::-1]), query[::-1], atol=1e-12
        )
    # Connectivity is also dynamic: the triangle with index zero can change.
    reversed_op = tri_operator._replace(
        mesh=Mesh(coords=NODES, elements=ELEMENTS[::-1])
    )
    np.testing.assert_allclose(
        interpolate(reversed_op, NODES, points), points, atol=1e-12
    )


def test_live_interpolation_vmap_and_derivatives(tri_operator):
    points = jnp.array([[0.2, 0.1], [0.2, 0.8]])
    values = jnp.stack([jnp.arange(4.0), 2 * jnp.arange(4.0)])
    vmapped = jax.jit(jax.vmap(tri_operator.interpolate, in_axes=(0, 0)))
    query_batches = jnp.stack([points, points[::-1]])
    expected = jnp.stack(
        [
            tri_operator.make_interpolate(points)(values[0]),
            tri_operator.make_interpolate(points[::-1])(values[1]),
        ]
    )
    np.testing.assert_allclose(vmapped(values, query_batches), expected, atol=1e-12)
    np.testing.assert_allclose(
        jax.jacrev(lambda v: tri_operator.interpolate(v, points))(values[0]),
        jax.jacrev(tri_operator.make_interpolate(points))(values[0]),
        atol=1e-12,
    )
    coeffs = jnp.array([2.0, 3.0])
    field = NODES @ coeffs
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda p: tri_operator.interpolate(field, p[None])[0]))(
            points[0]
        ),
        coeffs,
        atol=1e-12,
    )

    # Moving geometry and query points together leaves material interpolation unchanged.
    @jax.jit
    def translated_field(op, shift):
        moved = op._replace(
            mesh=Mesh(coords=op.mesh.coords + shift, elements=op.mesh.elements)
        )
        return moved.interpolate(field, points + shift).sum()

    np.testing.assert_allclose(
        jax.grad(lambda shift: translated_field(tri_operator, shift))(0.0),
        0,
        atol=1e-12,
    )


def test_live_interpolation_tensor_empty_and_outside(tri_operator):
    values = jnp.arange(24.0).reshape(4, 2, 3)
    points = jnp.array([[0.2, 0.1], [0.2, 0.8]])
    interpolate = jax.jit(tri_operator.interpolate)
    np.testing.assert_allclose(
        interpolate(values, points), tri_operator.make_interpolate(points)(values)
    )
    assert interpolate(values, jnp.empty((0, 2))).shape == (0, 2, 3)
    result = interpolate(values, jnp.concatenate([points, jnp.array([[1.5, 0.5]])]))
    np.testing.assert_allclose(
        result[:2], tri_operator.make_interpolate(points)(values)
    )
    assert np.isnan(result[2]).all()


@pytest.mark.parametrize(
    "element_type", [Quad4, Tetrahedron4, Hexahedron8, Tri6, Quad8, Line2]
)
def test_live_interpolation_only_supports_tri3(element_type):
    element = element_type()
    coords = element._reference_nodes()
    op = Operator(Mesh(coords=coords, elements=jnp.arange(len(coords))[None]), element)
    with pytest.raises(
        NotImplementedError, match="Live interpolation supports only Tri3"
    ):
        jax.jit(op.interpolate)(jnp.ones(len(coords)), jnp.zeros((1, coords.shape[1])))
