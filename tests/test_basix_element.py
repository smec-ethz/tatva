import jax

from tatva.operator import Operator

jax.config.update("jax_enable_x64", True)

import basix
import jax.numpy as jnp
import pytest

from tatva import Mesh
from tatva import element as tatva_element
from tatva.element import Quadrature, from_basix
from tatva.function_space import FunctionSpace


def get_cell_mesh(cell: str):
    if cell == "interval":
        return None
    elif cell == "quadrilateral":
        return Mesh.unit_square(1, 1, type="quad")
    elif cell == "hexahedron":
        return None
    else:
        raise ValueError(f"Unknown cell type: {cell}")


@pytest.mark.parametrize(
    "family, cell", [("P", "interval"), ("P", "quadrilateral"), ("P", "hexahedron")]
)
def test_lagrange_family(family: str, cell: str):
    element = from_basix(
        family=family,
        cell=cell,
        degree=1,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        quadrature_rule=Quadrature.rule(
            cell=cell, quadrature_degree=1, quadrature_type=basix.QuadratureType.gll
        ),
        # basix numbers quadrilateral vertices lexicographically; the mesh connectivity
        # follows tatva's counter-clockwise Quad4 order.
        dof_ordering=[0, 1, 3, 2] if cell == "quadrilateral" else None,
    )
    print(element.sobolev)
    print(element.quad_points)
    print(element.quad_weights)
    print(element.shape_function(element.quad_points[0]))
    print(element.shape_function_derivative(element.quad_points[0]))

    if cell == "quadrilateral":
        telement = tatva_element.Quad4()
        print(element.quad_points)
        print(telement.shape_function(element.quad_points[0]))
        print(telement.shape_function_derivative(element.quad_points[0]))

        mesh = get_cell_mesh(cell)
        op = Operator(FunctionSpace(mesh, element))
        print(op.eval(jnp.array([1])))
        print(op.integrate(op.eval(jnp.array([1]))))

        op = Operator(FunctionSpace(mesh, telement))
        print(op.eval(jnp.array([1])))
        print(op.integrate(op.eval(jnp.array([1]))))

    assert element is not None


@pytest.mark.parametrize(
    "family, cell",
    [("RT", "quadrilateral"), ("RT", "hexahedron")],
)
def test_RT_family(family: str, cell: str):
    element = from_basix(
        family=family,
        cell=cell,
        degree=1,
        quadrature_rule=Quadrature.rule(
            cell=cell, quadrature_degree=1, quadrature_type=basix.QuadratureType.gll
        ),
    )
    print(element.sobolev)
    print(element.quad_points)
    assert element is not None
