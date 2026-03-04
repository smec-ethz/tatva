import jax.numpy as jnp
import numpy as np

from tatva.mesh import find_containing_polygons


def test_find_containing_polygons_includes_boundary_points():
    polygons = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],
        ]
    )
    points = jnp.array(
        [
            [0.5, 0.5],
            [1.0, 0.5],  # shared boundary
            [1.5, 0.5],
        ]
    )

    indices = find_containing_polygons(points, polygons)

    np.testing.assert_array_equal(np.asarray(indices), np.array([0, 0, 1]))
