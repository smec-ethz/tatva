# Copyright (C) 2025 ETH Zurich (SMEC)
import jax.numpy as jnp
import numpy as np
from tatva.lifter import Lifter, Periodic, Fixed, lifted


def test_reduce_adjoint_periodic():
    # Size 4, dof 2 is periodic with master 1
    # Full vector: [0, 1, 2, 3]
    # Reduced vector will have 3 dofs (0, 1, 3)
    lifter = Lifter(4, Periodic(dofs=jnp.array([2]), master_dofs=jnp.array([1])))

    # Residual in full space
    r_full = jnp.array([10.0, 20.0, 30.0, 40.0])

    # Expected reduced residual:
    # r_red[0] = r_full[0] = 10.0
    # r_red[1] = r_full[1] + r_full[2] = 20.0 + 30.0 = 50.0
    # r_red[2] = r_full[3] = 40.0

    r_red = lifter.reduce_adjoint(r_full)
    np.testing.assert_array_equal(r_red, jnp.array([10.0, 50.0, 40.0]))


def test_reduce_adjoint_fixed():
    # Size 4, dof 0 and 3 are fixed
    lifter = Lifter(4, Fixed(jnp.array([0, 3]), 0.0))

    r_full = jnp.array([10.0, 20.0, 30.0, 40.0])

    # Expected reduced residual:
    # r_red[0] = r_full[1] = 20.0
    # r_red[1] = r_full[2] = 30.0

    r_red = lifter.reduce_adjoint(r_full)
    np.testing.assert_array_equal(r_red, jnp.array([20.0, 30.0]))


def test_lifted_dual_output():
    lifter = Lifter(4, Periodic(dofs=jnp.array([2]), master_dofs=jnp.array([1])))

    @lifted(argnums=0, output="dual")
    def compute_residual(u_full):
        # some dummy "gradient" operation
        return u_full * 2.0

    u_reduced = jnp.array([1.0, 2.0, 3.0])
    # lift_from_zeros -> [1.0, 2.0, 2.0, 3.0]
    # * 2.0 -> [2.0, 4.0, 4.0, 6.0]
    # reduce_adjoint:
    # r_red[0] = 2.0
    # r_red[1] = 4.0 + 4.0 = 8.0
    # r_red[2] = 6.0

    r_red = compute_residual(lifter, u_reduced)
    np.testing.assert_array_equal(r_red, jnp.array([2.0, 8.0, 6.0]))


def test_reduce_adjoint_multiple_constraints():
    # Complex case: 2 periodic with 1, 3 fixed
    lifter = Lifter(
        4,
        Periodic(dofs=jnp.array([2]), master_dofs=jnp.array([1])),
        Fixed(jnp.array([3]), 0.0),
    )
    # free dofs: [0, 1]

    r_full = jnp.array([10.0, 20.0, 30.0, 40.0])

    # Step 1 (Reverse order): apply Fixed transpose on 3
    # r -> [10, 20, 30, 0]
    # Step 2: apply Periodic transpose (2 -> 1)
    # r -> [10, 20+30, 0, 0] = [10, 50, 0, 0]
    # Step 3: extract [0, 1]
    # r_red -> [10, 50]

    r_red = lifter.reduce_adjoint(r_full)
    np.testing.assert_array_equal(r_red, jnp.array([10.0, 50.0]))
