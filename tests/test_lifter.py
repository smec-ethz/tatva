from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from tatva.lifter import Fixed, Lifter, Periodic, RuntimeValue

jax.config.update("jax_enable_x64", True)


def test_lifter_without_constraints_roundtrips():
    lifter = Lifter(4)
    u_reduced = jnp.arange(lifter.size_reduced, dtype=jnp.float64)

    u_full = lifter.lift_from_zeros(u_reduced)
    np.testing.assert_array_equal(u_full, np.arange(4, dtype=np.float64))
    np.testing.assert_array_equal(lifter.reduce(u_full), u_reduced)
    np.testing.assert_array_equal(
        lifter.constrained_dofs, jnp.array([], dtype=jnp.int32)
    )


def test_lifter_applies_dirichlet_and_periodic_constraints():
    lifter = Lifter(
        6,
        Fixed(jnp.array([0, 5], dtype=jnp.int32)),
        Periodic(
            dofs=jnp.array([2], dtype=jnp.int32),
            master_dofs=jnp.array([1], dtype=jnp.int32),
        ),
    )

    u_reduced = jnp.array([10.0, 20.0, 30.0])
    lifted = lifter.lift_from_zeros(u_reduced)

    expected = jnp.array([0.0, 10.0, 10.0, 20.0, 30.0, 0.0])
    np.testing.assert_array_equal(lifted, expected)
    np.testing.assert_array_equal(lifter.reduce(lifted), u_reduced)


def test_constraints_and_lifter_are_hashable():
    periodic = Periodic(
        dofs=jnp.array([2], dtype=jnp.int32),
        master_dofs=jnp.array([1], dtype=jnp.int32),
    )
    dirichlet = Fixed(
        jnp.array([0, 5], dtype=jnp.int32),
        jnp.array([0.0, 1.0], dtype=jnp.float64),
    )
    lifter = Lifter(6, dirichlet, periodic)

    assert isinstance(hash(periodic), int)
    assert isinstance(hash(dirichlet), int)
    assert isinstance(hash(lifter), int)


def test_lifter_as_static_arg_in_jit():
    lifter = Lifter(
        6,
        Fixed(jnp.array([0, 5], dtype=jnp.int32)),
        Periodic(
            dofs=jnp.array([2], dtype=jnp.int32),
            master_dofs=jnp.array([1], dtype=jnp.int32),
        ),
    )
    u_reduced = jnp.array([10.0, 20.0, 30.0])

    @partial(jax.jit, static_argnames=("lifter",))
    def solve_step(u_reduced: Array, lifter: Lifter) -> Array:
        return lifter.lift_from_zeros(u_reduced)

    lifted = solve_step(u_reduced, lifter=lifter)
    expected = jnp.array([0.0, 10.0, 10.0, 20.0, 30.0, 0.0])
    np.testing.assert_array_equal(lifted, expected)


def test_lifter_eq_handles_array_runtime_values():
    lifter = Lifter(
        4,
        Fixed(jnp.array([0, 3], dtype=jnp.int32), RuntimeValue("top")),
    )
    lhs = lifter.with_values({"top": jnp.array([1.0, 2.0])})
    rhs = lifter.with_values({"top": jnp.array([1.0, 2.0])})
    diff = lifter.with_values({"top": jnp.array([1.0, 3.0])})

    assert lhs == rhs
    assert lhs != diff
