import jax
import jax.numpy as jnp
import numpy as np

from tatva.lifter import DirichletBC, Lifter, PeriodicMap

jax.config.update("jax_enable_x64", True)


def test_lifter_without_constraints_roundtrips():
    lifter = Lifter(4)
    u_reduced = jnp.arange(lifter.size_reduced, dtype=jnp.float64)

    u_full = lifter.lift_from_zeros(u_reduced)
    np.testing.assert_array_equal(u_full, np.arange(4, dtype=np.float64))
    np.testing.assert_array_equal(lifter.reduce(u_full), u_reduced)
    np.testing.assert_array_equal(lifter.constrained_dofs, jnp.array([], dtype=jnp.int32))


def test_lifter_applies_dirichlet_and_periodic_constraints():
    lifter = Lifter(
        6,
        DirichletBC(jnp.array([0, 5], dtype=jnp.int32)),
        PeriodicMap(dofs=jnp.array([2], dtype=jnp.int32), master_dofs=jnp.array([1], dtype=jnp.int32)),
    )

    u_reduced = jnp.array([10.0, 20.0, 30.0])
    lifted = lifter.lift_from_zeros(u_reduced)

    expected = jnp.array([0.0, 10.0, 10.0, 20.0, 30.0, 0.0])
    np.testing.assert_array_equal(lifted, expected)
    np.testing.assert_array_equal(lifter.reduce(lifted), u_reduced)
