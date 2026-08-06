import jax.numpy as jnp
import numpy as np

from tatva.sparse.tracer.base import ghost_dofs_from_energy

PARTITION = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)


def test_reduce_sum_establishes_per_entry_contributions():
    def energy(u):
        return jnp.sum((u[1:] - u[:-1]) ** 2)

    ghosts = ghost_dofs_from_energy(energy, PARTITION)(jnp.zeros(6))

    np.testing.assert_array_equal(ghosts[0], [3])
    np.testing.assert_array_equal(ghosts[1], [])


def test_transparent_structural_operations_preserve_contributions():
    def energy(u):
        return jnp.sum(u.reshape(2, 3).T)

    ghosts = ghost_dofs_from_energy(energy, PARTITION)(jnp.zeros(6))

    np.testing.assert_array_equal(ghosts[0], [])
    np.testing.assert_array_equal(ghosts[1], [])


def test_constant_multiplication_preserves_additive_decomposition():
    def energy(u):
        return jnp.sum(jnp.arange(6.0) * u)

    ghosts = ghost_dofs_from_energy(energy, PARTITION)(jnp.zeros(6))

    np.testing.assert_array_equal(ghosts[0], [])
    np.testing.assert_array_equal(ghosts[1], [])


def test_nonlinear_scalar_aggregation_uses_conservative_root():
    def energy(u):
        return jnp.sin(jnp.sum(u))

    ghosts = ghost_dofs_from_energy(energy, PARTITION)(jnp.zeros(6))

    np.testing.assert_array_equal(ghosts[0], [3, 4, 5])
    np.testing.assert_array_equal(ghosts[1], [])
