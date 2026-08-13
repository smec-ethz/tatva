import jax.numpy as jnp

from tatva.element.base import Tri3
from tatva.mesh import Mesh
from tatva.operator import Operator
from tatva.tracer.capture import CapturedJaxpr


def test_capture_retains_parameter_and_pytree_leaf_paths():
    mesh = Mesh.unit_square(1, 1)
    op = Operator(mesh, Tri3())

    def fn(u, op):
        return jnp.sum(op.mesh.coords[op.mesh.elements]) + jnp.sum(u)

    captured = CapturedJaxpr.from_fn(fn, jnp.zeros(mesh.coords.shape[0]), op=op)

    assert [origin.display_path for origin in captured.call_abi.input_origins] == [
        "u",
        "op.mesh.coords",
        "op.mesh.elements",
    ]
