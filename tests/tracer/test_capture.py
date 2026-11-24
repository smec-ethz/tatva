import jax.numpy as jnp

from tatva.element.base import Tri3
from tatva.mesh import Mesh
from tatva.operator import Operator
from tatva.tracer.capture import CapturedJaxpr


def test_capture_retains_parameter_pytree_ranges():
    mesh = Mesh.unit_square(1, 1)
    op = Operator(mesh, Tri3())

    def fn(u, op):
        return jnp.sum(op.mesh.coords[op.mesh.elements]) + jnp.sum(u)

    captured = CapturedJaxpr.from_fn(fn, jnp.zeros(mesh.coords.shape[0]), op=op)

    parameter_trees = captured.call_abi.parameter_trees()

    assert [(name, flat_slice) for name, _, flat_slice in parameter_trees] == [
        ("u", slice(0, 1)),
        ("op", slice(1, 3)),
    ]
    assert [tree.num_leaves for _, tree, _ in parameter_trees] == [1, 2]
