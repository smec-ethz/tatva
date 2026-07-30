import jax

jax.config.update("jax_enable_x64", True)

import basix
import jax.numpy as jnp
import numpy as np
import pytest

from tatva import Mesh
from tatva import element as tatva_element
from tatva.element import Quadrature, from_basix
from tatva.function_space import FunctionSpace
from tatva.operator import Operator
from tatva.topology import vertex_permutation

CASES = {
    "triangle": (
        tatva_element.Tri3(),
        lambda: Mesh(
            coords=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            elements=jnp.array([[0, 1, 2]]),
        ),
    ),
    "quadrilateral": (
        tatva_element.Quad4(),
        lambda: Mesh(
            coords=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            elements=jnp.array([[0, 1, 3, 2]]),
        ),
    ),
    "tetrahedron": (
        tatva_element.Tetrahedron4(),
        lambda: Mesh(
            coords=jnp.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            elements=jnp.array([[0, 1, 2, 3]]),
        ),
    ),
    "hexahedron": (
        tatva_element.Hexahedron8(),
        lambda: Mesh(
            coords=jnp.array(
                [
                    [-1.0, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
            ),
            elements=jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        ),
    ),
}


def p1_element(cell: str):
    return from_basix(
        "P",
        cell,
        1,
        Quadrature.rule(cell, quadrature_degree=1),
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )


@pytest.mark.parametrize("cell", list(CASES))
def test_expected_permutation(cell: str):
    """The permutation is a relabelling of the vertices, identity on simplices."""
    expected = {
        "triangle": [0, 1, 2],
        "quadrilateral": [0, 1, 3, 2],
        "tetrahedron": [0, 1, 2, 3],
        "hexahedron": [0, 1, 3, 2, 4, 5, 7, 6],
    }[cell]
    assert vertex_permutation(cell).tolist() == expected


def test_unknown_cell_is_rejected():
    with pytest.raises(KeyError, match="no tatva vertex order"):
        vertex_permutation("prism")


@pytest.mark.parametrize("cell", list(CASES))
def test_basix_p1_matches_handwritten_element(cell: str):
    """A basix P1 space integrates identically to tatva's own element on the same mesh.

    The hand-written element is only an oracle here: it has a different reference cell for
    the tensor cells, so the two agree in physical space and nowhere else.
    """
    native, make_mesh = CASES[cell]
    mesh = make_mesh()
    f = jnp.sin(mesh.coords[:, 0]) * mesh.coords[:, 1] ** 2 + 1.0

    reference = Operator(mesh, native)
    expected = reference.integrate(reference.eval(f))

    space = FunctionSpace(mesh, p1_element(cell))
    operator = Operator(mesh._replace(elements=space.geometry_dofmap), space.element)
    assert operator.integrate(operator.eval(f)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("cell", list(CASES))
def test_operator_functions(cell: str):
    native, make_mesh = CASES[cell]
    mesh = make_mesh()

    native_op = Operator(mesh, native)
    native_quad_points = native_op.eval(mesh.coords)

    V = FunctionSpace(mesh, p1_element(cell))
    op = Operator(mesh._replace(elements=V.dofmap), V.element)
    quad_points = op.eval(mesh.coords)
    print(quad_points)
    print(native_quad_points)


@pytest.mark.parametrize("cell", list(CASES))
def test_handwritten_element_needs_no_reordering(cell: str):
    """A hand-written element defines tatva's order, so its dofmap is the connectivity."""
    native, make_mesh = CASES[cell]
    mesh = make_mesh()
    space = FunctionSpace(mesh, native)
    assert np.array_equal(space.dofmap, mesh.elements)


@pytest.mark.parametrize("cell", ["quadrilateral", "hexahedron"])
def test_tensor_cells_are_actually_reordered(cell: str):
    """Guards against the permutation silently becoming the identity."""
    _, make_mesh = CASES[cell]
    mesh = make_mesh()
    space = FunctionSpace(mesh, p1_element(cell))
    assert not np.array_equal(space.dofmap, mesh.elements)
    # a permutation of columns preserves each row as a set
    assert np.array_equal(np.sort(space.dofmap, axis=1), np.sort(mesh.elements, axis=1))


def test_geometry_and_dof_tables_agree_at_degree_one():
    mesh = Mesh.unit_square(3, 3, type="quad")
    space = FunctionSpace(mesh, p1_element("quadrilateral"))
    assert np.array_equal(space.dofmap, space.geometry_dofmap)
    assert space.n_global_dofs == mesh.coords.shape[0]
    assert space.n_dofs_per_element == 4


def test_higher_degree_is_refused():
    """Edge/face/interior dofs are not generated yet; refuse rather than guess."""
    mesh = Mesh.unit_square(3, 3, type="triangle")
    element = from_basix("P", "triangle", 2, Quadrature.rule("triangle", 3))
    with pytest.raises(NotImplementedError, match="only degree-1 spaces"):
        FunctionSpace(mesh, element)


def test_pytree_round_trip_preserves_tables():
    mesh = Mesh.unit_square(3, 3, type="quad")
    space = FunctionSpace(mesh, p1_element("quadrilateral"))
    leaves, treedef = jax.tree_util.tree_flatten(space)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(restored.dofmap, space.dofmap)
    assert restored.element == space.element
