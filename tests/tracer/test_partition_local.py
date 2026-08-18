import jax
import jax.numpy as jnp
import numpy as np

import tatva.tracer.api as tracer_api
from tatva.tracer.api import CapturedJaxpr, trace
from tatva.tracer.core.nested import MapSpec
from tatva.tracer.local.liveness import DemandSeed, backpropagate_plan_demand
from tatva.tracer.local.plan import build_rank_local_plan
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.incidence import generate_contribution_blocks


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
        assert reference_rank.dof_plan is reference.dof_plans[rank]

        local = traced.partition_local(
            rank=rank,
            n_parts=2,
        )

        assert local.rank == rank
        assert local.n_parts == 2
        assert local.local_plan is not reference.local_plans[rank]
        np.testing.assert_array_equal(
            local.dof_plan.compute_global,
            reference.dof_plans[rank].compute_global,
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


def test_partition_paths_do_not_materialize_global_invocations(monkeypatch):
    def energy(u, connectivity):
        return jnp.sum(u[connectivity] ** 2)

    u = jnp.arange(8.0)
    connectivity = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    traced = trace(CapturedJaxpr.from_fn(energy, u, connectivity))

    def unexpected_materialization(*_args, **_kwargs):
        raise AssertionError("distributed planning materialized the global plan")

    monkeypatch.setattr(tracer_api, "materialize_plan", unexpected_materialization)
    distributed = traced.partition(n_parts=2, partitioning="incidence")
    local = traced.partition_local(rank=0, n_parts=2, partitioning="incidence")

    assert all(not hasattr(plan, "instance") for plan in distributed.local_plans)
    assert not hasattr(local.local_plan, "instance")


def test_partition_retains_only_rank_demanded_map_iterations():
    def energy(u):
        terms = jax.lax.map(lambda value: value**2, u)
        return jnp.sum(terms)

    u = jnp.arange(12.0)
    distributed = trace(CapturedJaxpr.from_fn(energy, u)).partition(n_parts=3)

    for plan in distributed.local_plans:
        mapped = [
            eqn.nested
            for eqn in plan.eqns
            if eqn.nested is not None and isinstance(eqn.nested.spec, MapSpec)
        ]
        assert len(mapped) == 1
        indices = mapped[0].indices
        assert 0 < len(indices) < u.size

    values = []
    for rank in range(distributed.n_parts):
        local = distributed.for_rank(rank)
        args, kwargs = local.localize_inputs(u)
        values.append(local.local_function()(*args, **kwargs))
    np.testing.assert_allclose(sum(values), energy(u))


def test_rank_planning_does_not_resolve_undemanded_map_iterations():
    def energy(u):
        return jnp.sum(jax.lax.map(lambda value: value**2, u))

    u = jnp.arange(32.0)
    traced = trace(CapturedJaxpr.from_fn(energy, u))
    block = generate_contribution_blocks(traced.contributions, block_size=1)[7]
    root = traced.contributions.root(block.root_id)
    resolver, frame = ConcreteResolver.root(
        traced.captured.closed_jaxpr,
        traced.captured.flat_args,
        traced.analysis,
    )
    demand = backpropagate_plan_demand(
        traced.analysis,
        frame,
        resolver,
        (DemandSeed(root.value, block.demand),),
    )

    assert resolver.stats.map_iterations == 1
    plan = build_rank_local_plan(traced.analysis, frame, resolver, demand)
    assert resolver.stats.map_iterations == 2
    assert resolver.stats.frames_created - resolver.stats.frames_released == 1
    mapped = next(
        eqn.nested
        for eqn in plan.eqns
        if eqn.nested is not None and isinstance(eqn.nested.spec, MapSpec)
    )
    assert mapped.indices == (7,)
