import jax
import jax.numpy as jnp
import numpy as np

import tatva.tracer.diagnostics as tracer_diagnostics
from tatva.tracer.api import analyze
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


def test_distribution_constructs_and_caches_ranks_lazily():
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

    traced = analyze(energy, u, coords, connectivity)
    reference = traced.distribute(parts=2)
    assert reference.parts == 2
    assert not reference._rank_cache
    local_values = []

    for rank in range(2):
        reference_rank = reference.rank(rank)
        assert reference.rank(rank) is reference_rank

        args, kwargs = reference_rank.inputs(u, coords, connectivity)
        local_values.append(reference_rank(*args, **kwargs))

    assert set(reference._rank_cache) == {0, 1}
    assert reference.all_ranks() == (reference.rank(0), reference.rank(1))

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
    traced = analyze(energy, u, connectivity)

    def unexpected_materialization(*_args, **_kwargs):
        raise AssertionError("distributed planning materialized the global plan")

    monkeypatch.setattr(
        tracer_diagnostics,
        "materialize_plan",
        unexpected_materialization,
    )
    distributed = traced.distribute(parts=2)
    assert not distributed._rank_cache
    local = distributed.rank(0)

    assert set(distributed._rank_cache) == {0}
    assert callable(local)


def test_partition_retains_only_rank_demanded_map_iterations():
    def energy(u):
        terms = jax.lax.map(lambda value: value**2, u)
        return jnp.sum(terms)

    u = jnp.arange(12.0)
    distributed = analyze(energy, u).distribute(parts=3)

    for local in distributed.all_ranks():
        args, _ = local.inputs(u)
        assert 0 < args[0].size < u.size

    values = []
    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        args, kwargs = local.inputs(u)
        values.append(local(*args, **kwargs))
    np.testing.assert_allclose(sum(values), energy(u))


def test_rank_planning_does_not_resolve_undemanded_map_iterations():
    def energy(u):
        return jnp.sum(jax.lax.map(lambda value: value**2, u))

    u = jnp.arange(32.0)
    traced = analyze(energy, u)
    block = generate_contribution_blocks(
        traced._contributions,
        blocks_per_root=u.size,
    )[7]
    root = traced._contributions.root(block.root_id)
    resolver, frame = ConcreteResolver.root(
        traced._captured.closed_jaxpr,
        traced._captured.flat_args,
        traced._plan,
    )
    demand = backpropagate_plan_demand(
        traced._plan,
        frame,
        resolver,
        (DemandSeed(root.value, block.demand),),
    )

    assert resolver.stats.map_iterations == 0
    plan = build_rank_local_plan(traced._plan, frame, resolver, demand)
    assert resolver.stats.map_iterations == 0
    assert resolver.stats.frames_created - resolver.stats.frames_released == 1
    mapped = next(
        eqn.nested
        for eqn in plan.eqns
        if eqn.nested is not None and isinstance(eqn.nested.spec, MapSpec)
    )
    assert mapped.indices == (7,)
