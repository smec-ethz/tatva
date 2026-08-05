import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr

from tatva.lifter import Lifter, Periodic
from tatva.sparse.tracer import split_jaxpr_into_local

jax.config.update("jax_enable_x64", True)


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
    )

    def dummy(z, lifter: Lifter):
        z_full = lifter.lift_from_zeros(z)
        return jnp.sum(z_full[5:] * z_full[:-5])

    z_dummy = np.zeros(lifter.size_reduced)

    closed = jax.make_jaxpr(dummy)(z_dummy, lifter)
    part_map = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)

    concrete_vals, _ = jax.tree_util.tree_flatten((z_dummy, lifter))
    local_results = split_jaxpr_into_local(closed, part_map, concrete_vals)

    assert len(local_results) == 3
    for lc, local_args in local_results:
        assert isinstance(lc, ClosedJaxpr)
        local_out = jax.core.eval_jaxpr(lc.jaxpr, lc.consts, *local_args)
        assert local_out is not None
