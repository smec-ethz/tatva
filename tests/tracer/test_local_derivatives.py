import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

import tatva.tracer.diagnostics as tracer_diagnostics
from tatva.tracer import LocalDerivativeTrace, analyze
from tatva.tracer.diagnostics import global_derivatives


def _stencil_energy(dofs):
    return jnp.sum((dofs[:-1] + dofs[1:]) ** 2)


def _map_energy(dofs):
    return jnp.sum(jax.lax.map(lambda value: value**2, dofs))


def _scan_energy(dofs):
    def body(carry, value):
        carry = carry + value
        return carry, carry**2

    _, terms = jax.lax.scan(body, 0.0, dofs)
    return jnp.sum(terms)


def _boolean_union(matrices, shape):
    result = sps.csr_matrix(shape, dtype=bool)
    for matrix in matrices:
        result += matrix.astype(bool).tocsr()
    result.sum_duplicates()
    result.data[:] = True
    return result


def test_local_hessians_use_storage_coordinates_and_reconstruct_global_pattern():
    dofs = jnp.arange(6.0)
    traced = analyze(_stencil_energy, dofs)
    owners = np.array([1, 1, 0, 0, 1, 1], dtype=np.int64)
    distributed = traced.distribute(parts=2, dof_owner=owners)

    global_pattern = global_derivatives(traced).hessian.astype(bool)
    translated = []

    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        derivatives = local.derivatives()

        assert isinstance(derivatives, LocalDerivativeTrace)
        assert local.derivatives() is derivatives
        assert derivatives.hessian.shape == (
            local.dofs.storage.local_size,
            local.dofs.storage.local_size,
        )
        np.testing.assert_array_equal(
            derivatives.storage_global_dofs,
            local.dofs.storage.global_dofs,
        )
        global_coo = derivatives.global_hessian_coo()
        assert sps.isspmatrix_coo(global_coo)
        translated.append(global_coo)

        storage_example = dofs[jnp.asarray(local.dofs.storage.global_dofs)]
        numerical = np.asarray(jax.hessian(local.compile())(storage_example))
        structural = derivatives.hessian.astype(bool).toarray()
        assert np.all((np.abs(numerical) > 0) <= structural)

    assert any(
        not np.array_equal(plan.compute_rows, np.arange(plan.compute_rows.size))
        for plan in (local.dofs for local in distributed.all_ranks())
    )
    union = _boolean_union(translated, global_pattern.shape)
    assert (union != global_pattern).nnz == 0


def test_partition_and_compilation_do_not_trace_global_derivatives(monkeypatch):
    def unexpected_global_derivatives(*_args, **_kwargs):
        raise AssertionError("distributed planning traced global derivatives")

    monkeypatch.setattr(
        tracer_diagnostics,
        "trace_derivatives",
        unexpected_global_derivatives,
    )
    dofs = jnp.arange(8.0)
    traced = analyze(_stencil_energy, dofs)
    distributed = traced.distribute(parts=2)

    values = []
    for rank in range(distributed.parts):
        local = distributed.rank(rank)
        storage = dofs[jnp.asarray(local.dofs.storage.global_dofs)]
        values.append(local.compile()(storage))

    np.testing.assert_allclose(sum(values), _stencil_energy(dofs))
    assert not hasattr(traced, "derivatives")
    assert not hasattr(traced, "hessian")
    assert not hasattr(distributed.rank(0), "hessian")


@pytest.mark.parametrize("energy", [_map_energy, _scan_energy])
def test_nested_local_hessian_union_matches_global_pattern(energy):
    dofs = jnp.arange(1.0, 7.0)
    traced = analyze(energy, dofs)
    distributed = traced.distribute(parts=2)

    translated = [
        distributed.rank(rank).derivatives().global_hessian_coo()
        for rank in range(distributed.parts)
    ]
    global_pattern = global_derivatives(traced).hessian.astype(bool)
    union = _boolean_union(translated, global_pattern.shape)

    assert (union != global_pattern).nnz == 0


@pytest.mark.parametrize("local", [False, True])
def test_distribution_rejects_nonpositive_blocks_per_part(local):
    dofs = jnp.arange(4.0)
    traced = analyze(lambda z: jnp.sum(z**2), dofs)

    with pytest.raises(ValueError, match="blocks_per_part must be positive"):
        if local:
            traced.distribute(parts=2, blocks_per_part=0).rank(0)
        else:
            traced.distribute(parts=2, blocks_per_part=0)
