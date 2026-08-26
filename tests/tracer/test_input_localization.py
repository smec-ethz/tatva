import jax.numpy as jnp
import numpy as np

from tatva.lifter import Lifter
from tatva.tracer.api import analyze_captured
from tatva.tracer.capture import CapturedJaxpr
from tatva.tracer.local.inputs import _reconstruct_node


def test_reconstruct_node_preserves_leafless_pytree_children():
    lifter = Lifter(6)
    free_dofs = jnp.array([0, 1, 2])
    constrained_dofs = jnp.array([], dtype=jnp.int32)

    localized, _ = _reconstruct_node(
        lifter,
        leaves=iter(((free_dofs, None), (constrained_dofs, None))),
        specializers={},
        param_name="lifter",
    )

    assert localized._runtime_values == {}
    np.testing.assert_array_equal(localized.free_dofs, free_dofs)
    np.testing.assert_array_equal(localized.constrained_dofs, constrained_dofs)


def test_dead_gather_indices_do_not_prevent_local_execution():
    u = jnp.arange(6.0)
    coords = jnp.stack((u, u + 1), axis=1)
    connectivity = jnp.array(
        [[0, 1], [1, 2], [3, 4], [4, 5]],
        dtype=jnp.int32,
    )

    # Both gathers use the same connectivity input. The localization planner
    # should seed both index operands in one demand traversal for each rank.
    def energy(u, coords, connectivity):
        gathered_u = u[connectivity]
        gathered_coords = coords[connectivity]
        terms = jnp.sum(gathered_u**2, axis=1)
        terms += 0.01 * jnp.sum(gathered_coords, axis=(1, 2))
        return jnp.sum(terms)

    captured = CapturedJaxpr.from_fn(energy, u, coords, connectivity)
    distributed = analyze_captured(captured).distribute(parts=2)

    local_values = []
    for rank in range(2):
        local = distributed.rank(rank)
        args, kwargs = local.inputs(u, coords, connectivity)
        local_u, local_coords, local_connectivity = args

        assert not kwargs
        assert 0 < local_u.shape[0] < u.shape[0]
        assert local_coords.shape[1:] == coords.shape[1:]
        assert 0 < local_coords.shape[0] < coords.shape[0]
        assert local_connectivity is None

        local_values.append(local(*args, **kwargs))

    np.testing.assert_allclose(
        sum(local_values), energy(u, coords, connectivity), rtol=1e-6
    )


def test_distinct_index_inputs_may_use_distinct_operand_maps():
    u = jnp.arange(8.0)
    auxiliary = jnp.arange(10.0)
    u_indices = jnp.array(
        [[0, 1], [1, 2], [4, 5], [6, 7]],
        dtype=jnp.int32,
    )
    auxiliary_indices = jnp.array(
        [[8, 9], [7, 8], [2, 3], [1, 2]],
        dtype=jnp.int32,
    )

    def energy(u, auxiliary, u_indices, auxiliary_indices):
        u_terms = jnp.sum(u[u_indices] ** 2, axis=1)
        auxiliary_terms = jnp.sum(auxiliary[auxiliary_indices], axis=1)
        return jnp.sum(u_terms + 0.01 * auxiliary_terms)

    captured = CapturedJaxpr.from_fn(
        energy,
        u,
        auxiliary,
        u_indices,
        auxiliary_indices,
    )
    distributed = analyze_captured(captured).distribute(parts=2)

    local_values = []
    for rank in range(2):
        local = distributed.rank(rank)
        args, kwargs = local.inputs(
            u,
            auxiliary,
            u_indices,
            auxiliary_indices,
        )
        _, _, local_u_indices, local_auxiliary_indices = args

        assert not kwargs
        assert local_u_indices is None
        assert local_auxiliary_indices is None

        local_values.append(local(*args, **kwargs))

    np.testing.assert_allclose(
        sum(local_values),
        energy(u, auxiliary, u_indices, auxiliary_indices),
        rtol=1e-6,
    )
