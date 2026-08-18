import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

import tatva.tracer.api as tracer_api
from tatva.tracer import CapturedJaxpr, LocalDerivativeTrace, trace


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
    traced = trace(CapturedJaxpr.from_fn(_stencil_energy, dofs))
    owners = np.array([1, 1, 0, 0, 1, 1], dtype=np.int64)
    distributed = traced.partition(n_parts=2, dof_owner=owners)

    global_pattern = traced.global_hessian_sparsity().astype(bool)
    translated = []

    for rank in range(distributed.n_parts):
        local = distributed.for_rank(rank)
        derivatives = local.analyze_derivatives()

        assert isinstance(derivatives, LocalDerivativeTrace)
        assert local.analyze_derivatives() is derivatives
        assert derivatives.hessian.shape == (
            local.dof_plan.storage.local_size,
            local.dof_plan.storage.local_size,
        )
        np.testing.assert_array_equal(
            derivatives.storage_global_dofs,
            local.dof_plan.storage.global_dofs,
        )
        np.testing.assert_array_equal(
            local.hessian_sparsity().toarray(),
            derivatives.hessian.toarray(),
        )

        global_coo = derivatives.global_hessian_coo()
        assert sps.isspmatrix_coo(global_coo)
        translated.append(global_coo)

        storage_example = dofs[jnp.asarray(local.dof_plan.storage.global_dofs)]
        numerical = np.asarray(jax.hessian(local.local_function())(storage_example))
        structural = derivatives.hessian.astype(bool).toarray()
        assert np.all((np.abs(numerical) > 0) <= structural)

    assert any(
        not np.array_equal(plan.compute_rows, np.arange(plan.compute_rows.size))
        for plan in distributed.dof_plans
    )
    union = _boolean_union(translated, global_pattern.shape)
    assert (union != global_pattern).nnz == 0


def test_partition_and_compilation_do_not_trace_global_derivatives(monkeypatch):
    def unexpected_global_derivatives(*_args, **_kwargs):
        raise AssertionError("distributed planning traced global derivatives")

    monkeypatch.setattr(tracer_api, "trace_derivatives", unexpected_global_derivatives)
    dofs = jnp.arange(8.0)
    traced = trace(CapturedJaxpr.from_fn(_stencil_energy, dofs))
    distributed = traced.partition(n_parts=2)

    values = []
    for rank in range(distributed.n_parts):
        local = distributed.for_rank(rank)
        storage = dofs[jnp.asarray(local.dof_plan.storage.global_dofs)]
        values.append(local.local_function()(storage))

    np.testing.assert_allclose(sum(values), _stencil_energy(dofs))
    assert not hasattr(traced, "derivatives")
    assert not hasattr(traced, "hessian")
    assert not hasattr(distributed.for_rank(0), "hessian")


@pytest.mark.parametrize("energy", [_map_energy, _scan_energy])
def test_nested_local_hessian_union_matches_global_pattern(energy):
    dofs = jnp.arange(1.0, 7.0)
    traced = trace(CapturedJaxpr.from_fn(energy, dofs))
    distributed = traced.partition(n_parts=2)

    translated = [
        distributed.for_rank(rank).analyze_derivatives().global_hessian_coo()
        for rank in range(distributed.n_parts)
    ]
    global_pattern = traced.global_hessian_sparsity().astype(bool)
    union = _boolean_union(translated, global_pattern.shape)

    assert (union != global_pattern).nnz == 0


@pytest.mark.parametrize("local", [False, True])
def test_partition_rejects_nonpositive_block_factor(local):
    dofs = jnp.arange(4.0)
    traced = trace(CapturedJaxpr.from_fn(lambda z: jnp.sum(z**2), dofs))

    with pytest.raises(ValueError, match="block_factor must be positive"):
        if local:
            traced.partition_local(rank=0, n_parts=2, block_factor=0)
        else:
            traced.partition(n_parts=2, block_factor=0)
