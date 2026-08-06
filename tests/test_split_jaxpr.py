import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr

from tatva.lifter import Lifter, Periodic
from tatva.sparse.tracer import (
    ghost_dofs_from_energy,
    ghost_dofs_from_jaxpr,
    split_jaxpr_into_local,
)

jax.config.update("jax_enable_x64", True)


def test_ghost_dofs_are_energy_contribution_read_support():
    """The rank that owns a cut contribution receives its remote input."""

    def energy(u):
        return jnp.sum((u[1:] - u[:-1]) ** 2)

    u = jnp.zeros(6)
    part_map = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    ghosts = ghost_dofs_from_energy(energy, part_map)(u)

    np.testing.assert_array_equal(ghosts[0], [3])
    np.testing.assert_array_equal(ghosts[1], [])


def test_ghost_dofs_follow_concrete_gather_routing():
    """A non-geometric, static gather index is routed through dependency support."""

    def energy(u, pairs):
        return jnp.sum((u[pairs[:, 0]] - u[pairs[:, 1]]) ** 2)

    u = jnp.zeros(6)
    pairs = jnp.array([[0, 4], [1, 5], [2, 3]], dtype=jnp.int32)
    part_map = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    ghosts = ghost_dofs_from_energy(energy, part_map)(u, pairs)

    np.testing.assert_array_equal(ghosts[0], [3, 4, 5])
    np.testing.assert_array_equal(ghosts[1], [])


def test_ghost_dofs_include_affine_cross_partition_reads():
    """Affine energy terms have zero Hessian but still require the remote read."""

    def energy(u):
        return jnp.sum(u[:-1] - u[1:])

    u = jnp.zeros(6)
    part_map = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    ghosts = ghost_dofs_from_energy(energy, part_map)(u)

    np.testing.assert_array_equal(ghosts[0], [3])
    np.testing.assert_array_equal(ghosts[1], [])


def test_ghost_dofs_keep_scalar_branches_beside_reductions():
    """A scalar dot branch must not disappear when another branch is reduced."""

    def energy(u):
        local_terms = jnp.sum((u[1:] - u[:-1]) ** 2)
        return local_terms + jnp.dot(jnp.arange(u.size), u)

    u = jnp.zeros(4)
    part_map = np.array([0, 0, 1, 1], dtype=np.int32)
    ghosts = ghost_dofs_from_energy(energy, part_map)(u)

    # The local cut contribution reads 2; the unsplittable scalar dot conservatively
    # belongs to rank 0 and reads both remote DOFs.
    np.testing.assert_array_equal(ghosts[0], [2, 3])
    np.testing.assert_array_equal(ghosts[1], [])


def test_ghost_dofs_jaxpr_entry_point_validates_partition_size():
    closed = jax.make_jaxpr(lambda u: jnp.sum(u**2))(jnp.zeros(3))
    with pytest.raises(ValueError, match="part_map has 2 entries"):
        ghost_dofs_from_jaxpr(closed, np.array([0, 1]), [jnp.zeros(3)])


def test_split_simple_1d_energy():
    """Test splitting a simple nearest-neighbor energy functional across 2 ranks."""

    def energy(z):
        return jnp.sum(z[1:] * z[:-1])

    n_dofs = 6
    z_dummy = np.zeros(n_dofs)
    closed = jax.make_jaxpr(energy)(z_dummy)

    # Rank 0 owns DOFs [0, 1, 2], Rank 1 owns DOFs [3, 4, 5]
    part_map = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    local_results = split_jaxpr_into_local(closed, part_map, [z_dummy])

    assert len(local_results) == 2
    for lc, local_args in local_results:
        assert isinstance(lc, ClosedJaxpr)
        out = jax.core.eval_jaxpr(lc.jaxpr, lc.consts, *local_args)
        assert out is not None


def test_split_scatter_gather_dummy():
    """Test splitting a scatter/gather pipeline across 3 ranks."""

    def dummy(z, idx):
        # Scatter root DOFs into a zero buffer, then slice/multiply
        buf = jnp.zeros(20)
        z_full = buf.at[idx].set(z)
        return jnp.sum(z_full[5:] * z_full[:-5])

    n_dofs = 10
    z_dummy = np.zeros(n_dofs)
    idx = jnp.arange(10) * 2  # scatter to even positions [0, 2, ..., 18]

    closed = jax.make_jaxpr(dummy)(z_dummy, idx)
    part_map = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)

    local_results = split_jaxpr_into_local(closed, part_map, [z_dummy, idx])

    assert len(local_results) == 3
    for lc, local_args in local_results:
        assert isinstance(lc, ClosedJaxpr)
        out = jax.core.eval_jaxpr(lc.jaxpr, lc.consts, *local_args)
        assert out is not None


def test_split_scatter_gather_dummy_lifter():
    """Test splitting a scatter/gather pipeline across 3 ranks with Lifter."""

    n_dofs = 10
    lifter = Lifter.make(
        n_dofs,
        Periodic(jnp.array([0, 1]), jnp.array([8, 9])),
        Periodic(jnp.array([2]), jnp.array([7])),
    )

    def dummy(z, lifter: Lifter):
        z_full = lifter.lift_from_zeros(z)
        return jnp.sum(z_full[5:] * z_full[:-5])

    z_dummy = np.zeros(lifter.size_reduced)

    closed = jax.make_jaxpr(dummy)(z_dummy, lifter)
    part_map = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)

    concrete_vals, _ = jax.tree_util.tree_flatten((z_dummy, lifter))
    local_results = split_jaxpr_into_local(closed, part_map, concrete_vals)

    assert len(local_results) == 3
    for r, (lc, local_args) in enumerate(local_results):
        print(f"=== EVALUATING RANK {r} ===")
        print("consts:", lc.consts)
        print("local_args:", local_args)
        print("jaxpr:", lc.jaxpr)
        assert isinstance(lc, ClosedJaxpr)
        local_out = jax.core.eval_jaxpr(lc.jaxpr, lc.consts, *local_args)
        assert local_out is not None
