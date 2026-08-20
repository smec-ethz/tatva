import jax
import jax.numpy as jnp
import numpy as np

from tatva.tracer.api import analyze
from tatva.tracer.rules.structural import tile_row_map


def _tile_energy(u):
    values = u.reshape(4, 2)
    tiled = jnp.tile(values, (3, 2))
    offsets = jnp.arange(tiled.shape[1], dtype=tiled.dtype)
    return jnp.sum((tiled + offsets) ** 2)


def test_tile_row_map_repeats_source_rows_in_jax_order():
    closed = jax.make_jaxpr(lambda x: jnp.tile(x, (2, 3)))(
        jnp.zeros((2, 2), dtype=jnp.float32)
    )
    eqn = next(eqn for eqn in closed.jaxpr.eqns if eqn.primitive.name == "tile")

    row_map = tile_row_map(eqn)

    source = np.arange(4).reshape(2, 2)
    np.testing.assert_array_equal(
        row_map.source_rows,
        np.tile(source, (2, 3)).ravel(),
    )
    assert row_map.output_shape == (4, 6)


def test_partitioned_tile_uses_local_shapes_and_preserves_value_and_gradient():
    u = jnp.arange(8, dtype=jnp.float32)
    distributed = analyze(_tile_energy, u).distribute(parts=3)

    value = 0.0
    gradient = np.zeros(u.shape, dtype=np.float32)

    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        inputs = local.localize(u)
        compiled = local.compile()

        value += float(compiled(*inputs.args, **inputs.kwargs))
        local_gradient = np.asarray(
            jax.grad(compiled)(*inputs.args, **inputs.kwargs)
        )
        gradient[local.dofs.storage.global_dofs] += local_gradient

        tile = next(
            eqn for eqn in local._plan.eqns if eqn.primitive_name == "tile"
        )
        assert tile.input_layouts[0] is not None
        assert tile.output_layouts[0] is not None
        assert tile.input_layouts[0].local_shape == (2, 2)
        assert tile.output_layouts[0].local_shape == (4, 4)
        assert tile.output_layouts[0].local_shape != (12, 4)

    np.testing.assert_allclose(value, _tile_energy(u), rtol=1e-6)
    np.testing.assert_allclose(gradient, jax.grad(_tile_energy)(u), rtol=1e-6)
