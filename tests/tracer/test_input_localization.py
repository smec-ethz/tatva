import jax.numpy as jnp
import numpy as np

from tatva.tracer import input_localization
from tatva.tracer.api import trace
from tatva.tracer.capture import CapturedJaxpr


def test_input_localization_batches_gather_demand_per_rank(monkeypatch):
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

    calls = 0
    original = input_localization.backpropagate_demand

    def counted_backpropagate_demand(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        input_localization,
        "backpropagate_demand",
        counted_backpropagate_demand,
    )

    captured = CapturedJaxpr.from_fn(energy, u, coords, connectivity)
    distributed = trace(captured).partition(n_parts=2)

    # One auxiliary localization traversal per rank, not one per gather.
    assert calls == 2

    local_values = []
    for rank in range(2):
        local_args, local_kwargs = distributed.localize_inputs(
            rank, u, coords, connectivity
        )
        local_u, local_coords, local_connectivity = local_args

        assert not local_kwargs
        assert local_u.shape == (3,)
        assert local_coords.shape == (3, 2)
        assert local_connectivity.shape == (2, 2)
        assert int(local_connectivity.min()) >= 0
        assert int(local_connectivity.max()) < local_coords.shape[0]

        local_values.append(
            distributed.local_function(rank)(*local_args, **local_kwargs)
        )

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
    distributed = trace(captured).partition(n_parts=2)

    local_values = []
    for rank in range(2):
        local_args, local_kwargs = distributed.localize_inputs(
            rank,
            u,
            auxiliary,
            u_indices,
            auxiliary_indices,
        )
        local_u, local_auxiliary, local_u_indices, local_auxiliary_indices = local_args

        assert not local_kwargs
        assert int(local_u_indices.max()) < local_u.shape[0]
        assert int(local_auxiliary_indices.max()) < local_auxiliary.shape[0]

        local_values.append(
            distributed.local_function(rank)(*local_args, **local_kwargs)
        )

    np.testing.assert_allclose(
        sum(local_values),
        energy(u, auxiliary, u_indices, auxiliary_indices),
        rtol=1e-6,
    )
