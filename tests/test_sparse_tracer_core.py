"""Implementation-level tests for ``tatva.sparse.tracer``.

These tests target the tracer's internal machinery directly — the primitive handlers in
``Handlers``, the nonlinear classification, the JAXpr traversal, the higher-order handlers
(``scan``/``map``/``cond``/``pjit``), the trial/test split logic and the small data
structures — using tiny hand-written functions whose exact Hessian / Jacobian pattern is
knowable via ``jax.hessian`` / ``jax.jacobian``. This complements ``test_sparse_tracer.py``,
which exercises the tracer end-to-end through the FEM stack (Operator/Compound/Lifter).

The core contract under test:
  * **no false negatives** — the traced pattern is always a superset of the true pattern
    (the property that makes it safe for graph-coloring based sparse AD), and
  * **structural symmetry** — an energy Hessian pattern is always symmetric.

``cond``/``switch`` branches are traversed (couplings created inside a branch are
captured). One genuine soundness boundary remains documented as ``xfail``:
  * a data-dependent *carry* coupling in ``lax.scan``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

from tatva.sparse import trace_energy_sparsity, trace_virtual_work_sparsity
from tatva.sparse.tracer import (
    CouplingAccumulator,
    SparseDepSet,
    _unwrap_jit,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def nz_set(m) -> set[tuple[int, int]]:
    """CSR/array → set of ``(row, col)`` nonzero index tuples."""
    m = sps.csr_matrix(m)
    r, c = m.nonzero()
    return set(zip(r.tolist(), c.tolist()))


def dense_hessian_pattern(f, n: int, n_samples: int = 6, seed: int = 0) -> set:
    """True structural Hessian pattern, unioned over several random evaluation points.

    Unioning over points recovers the point-independent structural pattern while avoiding
    accidental zeros at any single point. The union is always a subset of the structural
    pattern, so ``dense ⊆ traced`` is the faithful no-false-negative check.
    """
    rng = np.random.default_rng(seed)
    pattern: set = set()
    for _ in range(n_samples):
        x = jnp.asarray(rng.normal(size=n))
        H = np.asarray(jax.hessian(f)(x))
        rows, cols = np.where(np.abs(H) > 1e-9)
        pattern |= set(zip(rows.tolist(), cols.tolist()))
    return pattern


def dense_vw_pattern(g, n: int, n_samples: int = 6, seed: int = 0) -> set:
    """True tangent-stiffness pattern K_ij = d/du_j (dG/dw_i), unioned over points."""

    def R_fn(u):
        return jax.grad(g, argnums=0)(jnp.zeros(n), u)

    rng = np.random.default_rng(seed)
    pattern: set = set()
    for _ in range(n_samples):
        x = jnp.asarray(rng.normal(size=n))
        K = np.asarray(jax.jacobian(R_fn)(x))
        rows, cols = np.where(np.abs(K) > 1e-9)
        pattern |= set(zip(rows.tolist(), cols.tolist()))
    return pattern


# ---------------------------------------------------------------------------
# A. property battery: one minimal energy per handler family
# ---------------------------------------------------------------------------

_A = np.random.default_rng(1).normal(size=(6, 6))
_A = jnp.asarray(_A + _A.T)  # symmetric → dense quadratic form
_PERM = jnp.array([2, 0, 1, 1, 3, 5])
_LO = jnp.array([0, 1, 2])
_HI = jnp.array([3, 4, 5])
_SCATTER_IDX = jnp.array([0, 1, 2, 0, 1, 2])


# each case: (id, fn, n_dofs, exact) where ``exact`` asserts the traced pattern equals
# the true structural pattern (otherwise only the no-false-negative superset is required)
ENERGY_CASES = [
    # --- scalar nonlinear handlers ---
    ("unary_sin_diagonal", lambda u: jnp.sum(jnp.sin(u)), 6, True),
    ("unary_dense_via_reduce", lambda u: jnp.sin(jnp.sum(u)), 6, True),
    ("unary_exp", lambda u: jnp.sum(jnp.exp(u)), 6, True),
    ("unary_log_safe", lambda u: jnp.sum(jnp.log(u**2 + 1.0)), 6, False),
    ("unary_tanh", lambda u: jnp.sum(jnp.tanh(u)), 6, True),
    ("binary_mul_adjacent", lambda u: jnp.sum(u[:-1] * u[1:]), 6, False),
    ("binary_div_adjacent", lambda u: jnp.sum(u[:-1] / (u[1:] ** 2 + 1.0)), 6, False),
    ("integer_pow2_diagonal", lambda u: jnp.sum(u**2), 6, True),
    ("integer_pow3_diagonal", lambda u: jnp.sum(u**3), 6, True),
    # --- contraction / reduction ---
    ("dot_general_quadratic_form", lambda u: u @ _A @ u, 6, True),
    ("reduce_sum_then_square", lambda u: jnp.sum(u.reshape(2, 3).sum(0) ** 2), 6, True),
    # --- index-routing structural handlers ---
    (
        "reshape_transpose",
        lambda u: jnp.sum(jnp.sin(u.reshape(2, 3).T.reshape(-1))),
        6,
        True,
    ),
    ("pad", lambda u: jnp.sum(jnp.sin(jnp.pad(u, (1, 1)))), 6, True),
    (
        "broadcast_in_dim",
        lambda u: jnp.sum(jnp.sin(jnp.broadcast_to(u[:, None], (6, 3)))),
        6,
        True,
    ),
    ("concatenate_duplicate", lambda u: jnp.sum(jnp.sin(jnp.concatenate([u, u]))), 6, True),
    ("squeeze", lambda u: jnp.sum(jnp.sin(u[:, None]).squeeze(-1)), 6, True),
    ("slice_product", lambda u: jnp.sum(u[:3] * u[3:]), 6, False),
    # --- gather / scatter ---
    ("gather_permutation", lambda u: jnp.sum(jnp.sin(u[_PERM])), 6, True),
    ("gather_product", lambda u: jnp.sum(u[_LO] * u[_HI]), 6, False),
    (
        "scatter_add_then_square",
        lambda u: jnp.sum(jnp.zeros(3).at[_SCATTER_IDX].add(u) ** 2),
        6,
        True,
    ),
    # --- select / where ---
    ("where_two_branches", lambda u: jnp.sum(jnp.where(u > 0, u**2, u**3)), 6, True),
    # --- conservative fallbacks (no false negatives, not exact) ---
    (
        "dynamic_slice_fallback",
        lambda u: jnp.sum(jax.lax.dynamic_slice(u, (1,), (3,)) ** 2),
        6,
        False,
    ),
    # --- higher-order: jit / map / scan(map-like) ---
    ("nested_jit", jax.jit(lambda u: jnp.sum(u[:-1] * u[1:])), 6, False),
    ("lax_map_independent", lambda u: jnp.sum(jax.lax.map(lambda x: x**2, u)), 6, True),
    (
        "lax_map_batched",
        lambda u: jnp.sum(jax.lax.map(lambda x: x**2, u, batch_size=2)),
        6,
        True,
    ),
    (
        "scan_map_like",
        lambda u: jnp.sum(jax.lax.scan(lambda c, x: (c, x**2), 0.0, u)[1]),
        6,
        True,
    ),
]

_ENERGY_IDS = [c[0] for c in ENERGY_CASES]


@pytest.mark.parametrize("fn,n,exact", [c[1:] for c in ENERGY_CASES], ids=_ENERGY_IDS)
def test_energy_no_false_negatives(fn, n, exact):
    """The traced Hessian pattern must contain every entry of the true pattern."""
    pat = trace_energy_sparsity(fn, n)
    assert dense_hessian_pattern(fn, n) <= nz_set(pat)


@pytest.mark.parametrize("fn,n,exact", [c[1:] for c in ENERGY_CASES], ids=_ENERGY_IDS)
def test_energy_hessian_structurally_symmetric(fn, n, exact):
    """An energy Hessian pattern is symmetric: the tracer records both (i,j) and (j,i)."""
    pat = trace_energy_sparsity(fn, n)
    assert nz_set(pat) == nz_set(pat.T)


@pytest.mark.parametrize(
    "fn,n",
    [c[1:3] for c in ENERGY_CASES if c[3]],
    ids=[c[0] for c in ENERGY_CASES if c[3]],
)
def test_energy_exact_pattern(fn, n):
    """For tight cases the traced pattern equals the true structural pattern exactly."""
    pat = trace_energy_sparsity(fn, n)
    assert nz_set(pat) == dense_hessian_pattern(fn, n)


# ---------------------------------------------------------------------------
# A. fallbacks: linear / constant / zero-dependency → identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda u: 3.0 * jnp.sum(u),  # linear
        lambda u: 5.0 + 0.0 * u[0],  # constant
        lambda u: jnp.sum(jnp.floor(u)),  # zero-dependency primitive (floor)
        lambda u: jnp.sum(u > 0.0),  # comparison → no deps
    ],
    ids=["linear", "constant", "floor", "comparison"],
)
def test_zero_hessian_returns_identity(fn):
    """When nothing couples, ``_trace_hessian_sparsity`` falls back to the identity."""
    n = 4
    pat = trace_energy_sparsity(fn, n)
    assert nz_set(pat) == nz_set(sps.eye(n))


# ---------------------------------------------------------------------------
# B. nonlinear classification (_NONLINEAR_*, integer_pow, linearity exception)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exponent,records",
    [(-1, True), (0, False), (1, False), (2, True), (3, True)],
)
def test_integer_pow_exponent_classification(exponent, records):
    """``integer_pow`` only records couplings for exponents >= 2 or <= -1."""
    # the +2.0 offset keeps the base away from 0 so the reciprocal (y=-1) is well defined
    fn = lambda u: jnp.sum((u + 2.0) ** exponent)
    pat = trace_energy_sparsity(fn, 4)
    if records:
        # y in {-1, 2, 3}: genuine diagonal curvature is recorded
        assert nz_set(pat) == {(i, i) for i in range(4)}
        assert dense_hessian_pattern(fn, 4) <= nz_set(pat)
    else:
        # y in {0, 1}: linear/constant → identity fallback only
        assert nz_set(pat) == nz_set(sps.eye(4))


@pytest.mark.parametrize(
    "fn",
    [
        lambda u: jnp.sum(3.0 * u),  # mul by constant
        lambda u: jnp.sum(u * 2.0),  # mul by constant (other operand)
        lambda u: jnp.sum(u / 4.0),  # div by constant
    ],
    ids=["const_mul_lhs", "const_mul_rhs", "const_div"],
)
def test_linear_scaling_not_recorded(fn):
    """``mul``/``div`` by a constant is linear and must record no couplings."""
    pat = trace_energy_sparsity(fn, 5)
    assert nz_set(pat) == nz_set(sps.eye(5))


def test_unary_nonlinear_is_separable_diagonal():
    """An element-wise unary nonlinearity yields a purely diagonal Hessian pattern."""
    pat = trace_energy_sparsity(lambda u: jnp.sum(jnp.sin(u)), 5)
    assert nz_set(pat) == {(i, i) for i in range(5)}


def test_binary_product_couples_operands():
    """``u[:-1] * u[1:]`` couples adjacent DOFs (a superset of the off-diagonal pattern)."""
    n = 5
    pat = trace_energy_sparsity(lambda u: jnp.sum(u[:-1] * u[1:]), n)
    # every true adjacency must be captured
    expected_adjacent = {(i, i + 1) for i in range(n - 1)} | {
        (i + 1, i) for i in range(n - 1)
    }
    assert expected_adjacent <= nz_set(pat)


# ---------------------------------------------------------------------------
# D. higher-order handlers: equivalences and known soundness boundaries
# ---------------------------------------------------------------------------


def test_nested_jit_matches_unjitted():
    """``@jax.jit`` (and nesting) must not change the traced pattern."""
    e = lambda u: jnp.sum(u[:-1] * u[1:]) + jnp.sum(jnp.sin(u))
    base = nz_set(trace_energy_sparsity(e, 6))
    assert nz_set(trace_energy_sparsity(jax.jit(e), 6)) == base
    assert nz_set(trace_energy_sparsity(jax.jit(jax.jit(e)), 6)) == base


def test_cond_internal_nonlinearity_captured():
    """The ``cond`` handler traverses branch jaxprs, so a nonlinearity created *inside* a
    branch and consumed linearly is captured (union of both branches)."""
    n = 6
    fn = lambda u: jax.lax.cond(
        u[0] > 0,
        lambda v: jnp.sum(v[:-1] * v[1:]),  # tridiagonal coupling
        lambda v: jnp.sum(v**2),  # diagonal
        u,
    )
    pat = trace_energy_sparsity(fn, n)
    assert dense_hessian_pattern(fn, n) <= nz_set(pat)
    # union of a tridiagonal and a diagonal branch is exactly the tridiagonal pattern
    assert nz_set(pat) == dense_hessian_pattern(fn, n)


def test_cond_nonlinearity_after_branch_is_sound():
    """A nonlinearity applied *after* the cond (affine branches) is also captured."""
    n = 6
    fn = lambda u: jnp.sum(
        jax.lax.cond(u[0] > 0, lambda v: v * 2.0, lambda v: v[::-1] * 3.0, u) ** 2
    )
    assert dense_hessian_pattern(fn, n) <= nz_set(trace_energy_sparsity(fn, n))


def test_switch_multibranch_captured():
    """``switch`` lowers to a multi-branch ``cond``; every branch is traced and unioned."""
    n = 6
    fn = lambda u: jax.lax.switch(
        jnp.int32(jnp.clip(u[0], 0, 2)),
        [
            lambda v: jnp.sum(v**2),
            lambda v: jnp.sum(v[:-1] * v[1:]),
            lambda v: jnp.sum(jnp.sin(v)),
        ],
        u,
    )
    assert dense_hessian_pattern(fn, n) <= nz_set(trace_energy_sparsity(fn, n))


@pytest.mark.xfail(
    strict=True,
    reason="scan_map seeds the carry with empty deps: a data-dependent carry coupling "
    "across iterations is not tracked (false negatives). Map-style scans are sound.",
)
def test_scan_carry_coupling_is_unsound():
    """Documents a soundness boundary: cross-iteration coupling via a scan carry."""
    n = 6
    fn = lambda u: jnp.sum(jax.lax.scan(lambda c, x: (x, c * x), 1.0, u)[1])
    assert dense_hessian_pattern(fn, n) <= nz_set(trace_energy_sparsity(fn, n))


# ---------------------------------------------------------------------------
# E. virtual-work trial/test split machinery
# ---------------------------------------------------------------------------


def test_virtual_work_bad_argument_name_raises():
    """An unknown ``trial_arg`` / ``test_arg`` name is rejected up front."""
    g = lambda w, u: jnp.sum(w * u)
    with pytest.raises(ValueError, match="Trial argument 'x' not found"):
        trace_virtual_work_sparsity(g, 3, trial_arg="x", test_arg="w")
    with pytest.raises(ValueError, match="Test argument 'y' not found"):
        trace_virtual_work_sparsity(g, 3, trial_arg="u", test_arg="y")


def test_virtual_work_excludes_trial_only_term():
    """A trial-only nonlinear term must not leak into the cross-block tangent K."""
    n = 4
    # cross term w*u (→ diagonal K) plus a trial-only nonlinearity u**2 that must be masked
    g = lambda w, u: jnp.sum(w * u) + jnp.sum(u**2)
    K = trace_virtual_work_sparsity(g, n, trial_arg="u", test_arg="w")
    assert nz_set(K) == {(i, i) for i in range(n)}


def test_virtual_work_static_arg_forwarded():
    """Extra ``*static_args`` are forwarded positionally to the virtual-work function."""
    n = 4
    g = lambda w, u, kappa: jnp.sum(kappa * w[:-1] * u[1:])
    K = trace_virtual_work_sparsity(g, n, "u", "w", 2.0)
    assert dense_vw_pattern(lambda w, u: g(w, u, 2.0), n) <= nz_set(K)
    assert K.nnz > 0  # the static coefficient does not zero-out the coupling


def test_virtual_work_default_arg_used():
    """A parameter with a default is filled from the signature when not supplied."""
    n = 4
    g = lambda w, u, kappa=3.0: jnp.sum(kappa * w[:-1] * u[1:])
    K = trace_virtual_work_sparsity(g, n, "u", "w")
    assert dense_vw_pattern(lambda w, u: g(w, u), n) <= nz_set(K)


def test_virtual_work_missing_static_arg_raises():
    """A non-defaulted extra parameter with no supplied static arg is an error."""
    n = 3
    g = lambda w, u, kappa: jnp.sum(kappa * w * u)
    with pytest.raises(ValueError, match="Missing static argument"):
        trace_virtual_work_sparsity(g, n, "u", "w")


def test_virtual_work_unsymmetric_pattern_captured():
    """A one-way cross coupling yields a genuinely unsymmetric K, captured with no
    false negatives (the path that distinguishes the VW tracer from the energy tracer)."""
    n = 4
    # w_i couples to u_{i+1} but not the reverse → strictly upper off-diagonal block
    g = lambda w, u: jnp.sum(w[:-1] * u[1:])
    K = trace_virtual_work_sparsity(g, n, trial_arg="u", test_arg="w")
    ref = dense_vw_pattern(g, n)
    assert ref <= nz_set(K)
    assert nz_set(K) != {(c, r) for (r, c) in nz_set(K)}  # not symmetric


# ---------------------------------------------------------------------------
# F. small unit tests of the data structures and helpers
# ---------------------------------------------------------------------------


def test_unwrap_jit_unwraps_jit_but_preserves_other_transforms():
    """``_unwrap_jit`` strips ``jax.jit`` wrappers but leaves ``grad``/``vmap`` intact."""
    e = lambda u: jnp.sum(u**2)
    assert _unwrap_jit(jax.jit(e)) is e
    assert _unwrap_jit(jax.jit(jax.jit(e))) is e
    assert _unwrap_jit(e) is e

    g = jax.grad(e)
    assert _unwrap_jit(g) is g  # grad must NOT be unwrapped to its primal


def test_sparse_dep_set_singletons_and_empty():
    s = SparseDepSet.singletons(3)
    assert s.shape == (3,)
    assert s.dep.shape == (3, 3)
    assert s.dep.nnz == 3
    np.testing.assert_array_equal(s.dep.toarray(), np.eye(3, dtype=bool))

    e = SparseDepSet.empty((2, 2), 3)
    assert e.shape == (2, 2)
    assert e.dep.shape == (4, 3)
    assert e.dep.nnz == 0


def test_sparse_dep_set_total_union_and_reshape():
    s = SparseDepSet.singletons(4)
    union = s.total_union()
    assert union.shape == ()
    assert union.dep.nnz == 4  # all four DOFs active after OR-reduction

    reshaped = SparseDepSet.singletons(6).reshape(2, 3)
    assert reshaped.shape == (2, 3)
    assert reshaped.dep.shape == (6, 6)  # underlying dep is unchanged by reshape


def test_sparse_dep_set_broadcast_to_replicates_rows():
    one_row = SparseDepSet(sps.csr_matrix(np.array([[1, 0, 1]], dtype=bool)), (1,))
    out = one_row.broadcast_to((4,))
    assert out.shape == (4,)
    assert out.dep.shape == (4, 3)
    np.testing.assert_array_equal(
        out.dep.toarray(), np.tile(np.array([[1, 0, 1]], dtype=bool), (4, 1))
    )


def test_coupling_accumulator_fingerprint_dedup():
    """Recording an identical dependency structure twice must not duplicate coordinates."""
    acc = CouplingAccumulator(3)
    dep = sps.csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=bool))
    acc.record_dep(dep)
    acc.record_dep(dep)  # same fingerprint → skipped
    pat = acc.finalize()
    assert pat.dtype == np.int8
    assert nz_set(pat) == {(0, 0), (1, 1)}
    assert set(pat.data.tolist()) == {1}  # binarized


def test_coupling_accumulator_trial_test_split_mask():
    """With a split, only cross pairs (one DOF on each side) are retained."""
    acc = CouplingAccumulator(4)
    full_row = sps.csr_matrix(np.ones((1, 4), dtype=bool))
    acc.record_dep(full_row, trial_test_split=2)
    pat = acc.finalize()
    expected = {(r, c) for r in range(4) for c in range(4) if (r < 2) != (c < 2)}
    assert nz_set(pat) == expected


def test_coupling_accumulator_empty_finalize():
    acc = CouplingAccumulator(3)
    pat = acc.finalize()
    assert pat.shape == (3, 3)
    assert pat.nnz == 0
    assert pat.dtype == np.int8
