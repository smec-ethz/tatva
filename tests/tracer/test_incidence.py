from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

from tatva.tracer.api import trace_fn
from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.lowering.partition import (
    PartitionStrategy,
    dof_owner_from_incidence,
    partition_contribution_blocks,
)
from tatva.tracer.program.contributions import ContributionBlock
from tatva.tracer.program.incidence import (
    BlockDofIncidence,
    TaggedDemand,
    generate_contribution_blocks,
    merge_tagged,
    reference_block_dof_incidence,
    tagged_block_dof_incidence,
)


def _gather_energy(u, connectivity):
    gathered = u[connectivity]
    terms = jnp.sum(gathered**2, axis=1)
    return jnp.sum(terms)


def test_reference_incidence_tracks_each_contribution_block():
    u = jnp.arange(6.0)
    connectivity = jnp.array(
        [[0, 1], [1, 2], [3, 4], [4, 5]],
        dtype=jnp.int32,
    )

    traced = trace_fn(_gather_energy, u, connectivity)
    blocks = generate_contribution_blocks(traced.contributions, block_size=1)
    reference = reference_block_dof_incidence(
        traced.resolved, traced.contributions, blocks=blocks
    )
    tagged = tagged_block_dof_incidence(
        traced.resolved, traced.contributions, blocks=blocks
    )

    assert [block.id for block in tagged.blocks] == [0, 1, 2, 3]
    assert [block.root_id for block in tagged.blocks] == [0, 0, 0, 0]
    assert (reference.csr != tagged.csr).nnz == 0
    np.testing.assert_array_equal(
        tagged.csr.toarray(),
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(tagged.block_dof_counts, [2, 2, 2, 2])
    np.testing.assert_array_equal(tagged.blocks_for_dof(1), [0, 1])

    coarser = generate_contribution_blocks(traced.contributions, block_size=2)
    assert [block.id for block in coarser] == [0, 1]
    assert [block.demand.rows().tolist() for block in coarser] == [[0, 1], [2, 3]]


def test_incidence_partition_and_dof_ownership_are_derivative_free():
    shape = (4,)
    demands = [TensorDemand.axis_range(shape, 0, i, i + 1) for i in range(4)]
    assert all(demand is not None for demand in demands)
    blocks = tuple(
        ContributionBlock(id=i, root_id=0, demand=demand)
        for i, demand in enumerate(demands)
        if demand is not None
    )
    incidence = BlockDofIncidence(
        blocks=blocks,
        csr=sps.csr_matrix(
            np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1],
                ],
                dtype=bool,
            )
        ),
    )

    partition, block_to_part = partition_contribution_blocks(
        incidence,
        n_parts=2,
        strategy=PartitionStrategy.INCIDENCE,
    )
    np.testing.assert_array_equal(block_to_part, [0, 0, 1, 1])
    assert [item.demand.rows().tolist() for item in partition.owned] == [
        [0, 1],
        [2, 3],
    ]

    dof_owner = dof_owner_from_incidence(
        incidence,
        block_to_part=block_to_part,
        n_parts=2,
    )
    np.testing.assert_array_equal(dof_owner, [0, 0, 0, 1, 1, 1])


def test_incidence_partition_executes_same_functional():
    u = jnp.arange(21.0)
    connectivity = jnp.stack((jnp.arange(20), jnp.arange(1, 21)), axis=1).astype(
        jnp.int32
    )
    traced = trace_fn(_gather_energy, u, connectivity)
    distributed = traced.partition(n_parts=2, partitioning="incidence")

    values = []
    for rank in range(2):
        local = distributed.for_rank(rank)
        args, kwargs = local.localize_inputs(u, connectivity)
        values.append(local.local_function()(*args, **kwargs))

    np.testing.assert_allclose(sum(values), _gather_energy(u, connectivity))


def test_tagged_demand_preserves_overlapping_block_identity():
    first = TensorDemand.axis_range((20,), 0, 0, 10)
    second = TensorDemand.axis_range((20,), 0, 8, 20)
    assert first is not None and second is not None

    merged = merge_tagged(
        TaggedDemand(first.shape, first.rows(), np.full(first.size, 3)),
        TaggedDemand(second.shape, second.rows(), np.full(second.size, 7)),
    )
    assert merged is not None
    np.testing.assert_array_equal(merged.block_ids, [3, 7])
    np.testing.assert_array_equal(merged.rows[merged.blocks == 3], np.arange(10))
    np.testing.assert_array_equal(merged.rows[merged.blocks == 7], np.arange(8, 20))
    assert np.count_nonzero(merged.rows == 8) == 2
    assert np.count_nonzero(merged.rows == 9) == 2


def _assert_tagged_matches_reference(fn, *args):
    traced = trace_fn(fn, *args)
    blocks = generate_contribution_blocks(traced.contributions, block_size=1)
    reference = reference_block_dof_incidence(
        traced.resolved, traced.contributions, blocks=blocks
    )
    tagged = tagged_block_dof_incidence(
        traced.resolved, traced.contributions, blocks=blocks
    )
    assert (reference.csr != tagged.csr).nnz == 0


def test_tagged_incidence_matches_oracle_for_structural_and_nested_programs():
    u = jnp.arange(1.0, 7.0)

    _assert_tagged_matches_reference(
        lambda values, weights: jnp.sum(
            jnp.sum(values.reshape(3, 2) * weights, axis=1)
        ),
        u,
        jnp.array([2.0, 3.0]),
    )
    _assert_tagged_matches_reference(
        lambda values, matrix: jnp.sum(jnp.sum(values.reshape(3, 2) @ matrix, axis=1)),
        u,
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    _assert_tagged_matches_reference(
        lambda values, mask: jnp.sum(jnp.where(mask, values, values[::-1]) ** 2),
        u,
        jnp.array([True, False, True, False, True, False]),
    )
    _assert_tagged_matches_reference(
        lambda values, predicate: jnp.sum(
            jax.lax.cond(
                predicate,
                lambda operand: operand**2,
                lambda operand: 3 * operand,
                values,
            )
        ),
        u,
        jnp.array(True),
    )

    @jax.jit
    def called(values):
        return values**2 + 1

    _assert_tagged_matches_reference(lambda values: jnp.sum(called(values)), u)
    _assert_tagged_matches_reference(
        lambda values: jnp.sum(jax.lax.map(lambda value: value**2, values)),
        u,
    )
    connectivity = jnp.array([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=jnp.int32)
    _assert_tagged_matches_reference(
        lambda values, connectivity: jnp.sum(
            jax.lax.map(
                lambda at: jnp.sum(values[at] ** 2),
                connectivity,
            )
        ),
        u,
        connectivity,
    )

    def scanned(values):
        def body(carry, value):
            next_carry = carry + value
            return next_carry, next_carry**2

        _, outputs = jax.lax.scan(body, jnp.array(0.0), values)
        return jnp.sum(outputs)

    _assert_tagged_matches_reference(scanned, u)

    indices = jnp.array([0, 2, 4, 5], dtype=jnp.int32)
    _assert_tagged_matches_reference(
        lambda values, at: jnp.sum(jnp.zeros_like(values).at[at].add(values[:4]) ** 2),
        u,
        indices,
    )
    _assert_tagged_matches_reference(
        lambda values, start: jnp.sum(
            jax.lax.dynamic_slice(values, (start,), (3,)) ** 2
        ),
        u,
        jnp.array(2, dtype=jnp.int32),
    )
    _assert_tagged_matches_reference(
        lambda values: 2 * jnp.sum(values[:3] ** 2) - 3 * jnp.sum(values[3:] ** 3),
        u,
    )


def test_tagged_incidence_matches_oracle_for_linear_solve_callbacks():
    def implicit_scalar_solve(a, b):
        def matvec(value):
            return a * value

        def solve(_matvec, rhs):
            return jax.lax.stop_gradient(rhs / a)

        return jax.lax.custom_linear_solve(
            matvec,
            b,
            solve=solve,
            transpose_solve=solve,
        )

    def objective(values):
        solutions = jax.vmap(implicit_scalar_solve)(values[:2], values[2:])
        return jnp.sum(solutions)

    with pytest.warns(UserWarning, match="supported batched matrix layout"):
        _assert_tagged_matches_reference(
            objective,
            jnp.array([2.0, 4.0, 3.0, 5.0]),
        )


def test_tagged_primitive_rule_is_called_once_for_many_blocks(monkeypatch):
    n_blocks = 200
    values = jnp.arange(n_blocks + 1.0)
    connectivity = jnp.stack(
        (jnp.arange(n_blocks), jnp.arange(1, n_blocks + 1)), axis=1
    ).astype(jnp.int32)
    traced = trace_fn(_gather_energy, values, connectivity)
    blocks = generate_contribution_blocks(traced.contributions, block_size=1)
    integer_pow = next(
        resolved.plan.eqn.primitive
        for resolved in traced.resolved.eqns
        if resolved.plan.eqn.primitive.name == "integer_pow"
    )
    semantics = SEMANTICS.get_ordinary(integer_pow)
    calls = 0

    def counted_tagged_rule(ctx):
        nonlocal calls
        calls += 1
        return semantics.tagged_demand(ctx)

    def forbidden_legacy_rule(ctx):
        del ctx
        raise AssertionError("tagged propagation called the TensorDemand rule")

    monkeypatch.setitem(
        SEMANTICS._rules,
        integer_pow,
        replace(
            semantics,
            demand=forbidden_legacy_rule,
            tagged_demand=counted_tagged_rule,
        ),
    )

    tagged_block_dof_incidence(
        traced.resolved,
        traced.contributions,
        blocks=blocks,
    )
    assert calls == 1


def test_tagged_incidence_does_not_take_tensor_demand_cartesian_hulls():
    values = jnp.arange(9.0)
    diagonal = jnp.array([0, 1, 2], dtype=jnp.int32)

    def objective(values, diagonal):
        matrix = values.reshape(3, 3)
        return jnp.sum(matrix[diagonal, diagonal] ** 2)

    traced = trace_fn(objective, values, diagonal)
    blocks = generate_contribution_blocks(traced.contributions, block_size=2)
    reference = reference_block_dof_incidence(
        traced.resolved,
        traced.contributions,
        blocks=blocks,
    )
    tagged = tagged_block_dof_incidence(
        traced.resolved,
        traced.contributions,
        blocks=blocks,
    )

    # The old TensorDemand path closes {(0,0), (1,1)} into the Cartesian
    # product {0,1}×{0,1}. Colored row-label propagation remains exact.
    np.testing.assert_array_equal(reference.dofs_for_block(0), [0, 1, 3, 4])
    np.testing.assert_array_equal(tagged.dofs_for_block(0), [0, 4])
