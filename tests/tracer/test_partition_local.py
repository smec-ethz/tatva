import jax.numpy as jnp
import numpy as np

from tatva.tracer.api import CapturedJaxpr, trace


class _ReferenceComm:
    def __init__(self, reference, rank):
        self._reference = reference
        self._rank = rank

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return len(self._reference.halo_plans)

    def alltoall(self, sendobj):
        halo = self._reference.halo_plans[self._rank]
        expected = {exchange.peer: exchange.global_dofs for exchange in halo.recv}
        for peer, values in enumerate(sendobj):
            np.testing.assert_array_equal(
                values,
                expected.get(peer, np.empty(0, dtype=np.int64)),
            )

        incoming = [
            np.empty(0, dtype=np.int64) for _ in range(len(self._reference.halo_plans))
        ]
        for exchange in halo.send:
            incoming[exchange.peer] = exchange.global_dofs
        return incoming

    def bcast(self, obj, root=0):
        assert root == 0
        return obj


def test_partition_local_matches_all_rank_reference():
    u = jnp.arange(6.0)
    coords = jnp.stack((u, u + 1), axis=1)
    connectivity = jnp.array(
        [[0, 1], [1, 2], [3, 4], [4, 5]],
        dtype=jnp.int32,
    )

    def energy(u, coords, connectivity):
        gathered_u = u[connectivity]
        gathered_coords = coords[connectivity]
        terms = jnp.sum(gathered_u**2, axis=1)
        terms += 0.01 * jnp.sum(gathered_coords, axis=(1, 2))
        return jnp.sum(terms)

    traced = trace(CapturedJaxpr.from_fn(energy, u, coords, connectivity))
    reference = traced.partition(n_parts=2)
    assert reference.n_parts == 2
    local_values = []

    for rank in range(2):
        reference_rank = reference.for_rank(rank)
        assert reference_rank.rank == rank
        assert reference_rank.local_plan is reference.local_plans[rank]
        assert reference_rank.halo_plan is reference.halo_plans[rank]
        assert reference_rank.input_plan is reference.input_plans[rank]

        local = traced.partition_local(
            comm=_ReferenceComm(reference, rank),
        )

        assert local.rank == rank
        assert local.n_parts == 2
        assert local.local_plan is not reference.local_plans[rank]
        np.testing.assert_array_equal(local.dof_owner, reference.dof_owner)
        np.testing.assert_array_equal(
            local.halo_plan.compute_global,
            reference.halo_plans[rank].compute_global,
        )

        local_args, local_kwargs = local.localize_inputs(u, coords, connectivity)
        local_values.append(local.local_function()(*local_args, **local_kwargs))

        reference_args, reference_kwargs = reference_rank.localize_inputs(
            u, coords, connectivity
        )
        np.testing.assert_allclose(
            reference_rank.local_function()(*reference_args, **reference_kwargs),
            local_values[-1],
        )

    np.testing.assert_allclose(
        sum(local_values),
        energy(u, coords, connectivity),
        rtol=1e-6,
    )
