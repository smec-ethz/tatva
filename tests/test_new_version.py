import basix
import jax
import jax.numpy as jnp

from tatva import FunctionSpace, Mesh, Operator
from tatva.compound import field
from tatva.compound import space_compound as compound
from tatva.element import Quadrature, from_basix

jax.config.update("jax_enable_x64", True)


def p1_element(cell: str):
    return from_basix(
        "P",
        cell,
        1,
        Quadrature.rule(cell, quadrature_degree=1),
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )


mesh = Mesh.unit_square(1, 1, type="triangle")
space = FunctionSpace(
    mesh,
    p1_element("triangle"),
)

op = Operator(space)


class Solution(compound.Compound, stack=False):
    u = field(space, components=(2,))
    p = field(space)
    lag = field(shape=(10,))


print(Solution.size)
print(Solution.u[:])
print(Solution.p[:])
print(Solution.lag[:])


(u, p, lag) = Solution(jnp.zeros(Solution.size))
print(u)
print(p)
print(lag)
