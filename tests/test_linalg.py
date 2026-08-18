# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.


import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva import Mesh, Operator, linalg
from tatva.element.base import Hexahedron8, Line2, Line3, Quad4, Tetrahedron4, Tri3

jax.config.update("jax_enable_x64", True)


def _well_conditioned(d: int, seed: int = 0) -> jnp.ndarray:
    """Identity plus a small perturbation, so the inverse is stable."""
    rng = np.random.default_rng(seed)
    return jnp.eye(d) + 0.1 * jnp.asarray(rng.standard_normal((d, d)))


# --------------------------------------------------------------------------------
# inv
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inv_matches_jnp_linalg_inv(d):
    J = _well_conditioned(d)
    assert jnp.allclose(linalg.inv(J), jnp.linalg.inv(J))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inv_round_trips(d):
    J = _well_conditioned(d)
    assert jnp.allclose(linalg.inv(J) @ J, jnp.eye(d), atol=1e-12)


@pytest.mark.parametrize("d", [2, 3])
def test_inv_vmaps_over_a_batch(d):
    """Batching is vmap's job -- the kernel itself stays unbatched."""
    rng = np.random.default_rng(1)
    stack = jnp.eye(d) + 0.1 * jnp.asarray(rng.standard_normal((16, d, d)))
    assert jnp.allclose(jax.vmap(linalg.inv)(stack), jnp.linalg.inv(stack))


@pytest.mark.parametrize("shape", [(4, 2, 2), (3, 3, 3), (2,)])
def test_inv_rejects_anything_that_is_not_a_single_matrix(shape):
    with pytest.raises(ValueError, match="single matrix"):
        linalg.inv(jnp.zeros(shape))


# --------------------------------------------------------------------------------
# det
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
def test_det_matches_jnp_linalg_det(d):
    J = _well_conditioned(d)
    assert jnp.allclose(linalg.det(J), jnp.linalg.det(J))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_det_returns_a_scalar(d):
    """`jnp.linalg.det` reduces the matrix axes away; the closed form must too, or a
    (1, 1) determinant would broadcast against everything downstream."""
    assert linalg.det(_well_conditioned(d)).shape == ()


@pytest.mark.parametrize("d", [1, 2, 3])
def test_det_is_consistent_with_inv(d):
    """det(J) * det(inv(J)) == 1 -- ties the two closed forms together, so a sign or
    cofactor slip in either one shows up here."""
    J = _well_conditioned(d)
    assert jnp.allclose(linalg.det(J) * linalg.det(linalg.inv(J)), 1.0)


@pytest.mark.parametrize("d", [2, 3])
def test_det_vmaps_over_a_batch(d):
    """Batching is vmap's job -- the kernel itself stays unbatched."""
    rng = np.random.default_rng(1)
    stack = jnp.eye(d) + 0.1 * jnp.asarray(rng.standard_normal((16, d, d)))
    assert jnp.allclose(jax.vmap(linalg.det)(stack), jnp.linalg.det(stack))


@pytest.mark.parametrize("shape", [(4, 2, 2), (3, 3, 3), (2,)])
def test_det_rejects_anything_that_is_not_a_single_matrix(shape):
    with pytest.raises(ValueError, match="single matrix"):
        linalg.det(jnp.zeros(shape))


# --------------------------------------------------------------------------------
# tensordot
# --------------------------------------------------------------------------------

# Every shape the kernel accepts: a 1-D or 2-D first operand against a second operand
# of any rank. The 1-D rows are the shape-function contraction; the 2-D rows are the
# matrix products inside Element.gradient.
ACCEPTED_SHAPES = [
    ((3,), (3,)),
    ((3,), (3, 2)),
    ((3,), (3, 2, 2)),
    ((2, 3), (3,)),
    ((2, 3), (3, 2)),
    ((2, 3), (3, 2, 2)),
    ((2, 3), (3, 4, 2, 2)),
]


@pytest.mark.parametrize("a_shape, b_shape", ACCEPTED_SHAPES)
def test_tensordot_matches_jnp_tensordot(a_shape, b_shape):
    """This is the whole contract: a drop-in for jnp.tensordot(A, B, axes=1) that
    happens to compile without a dot. If this fails the kernel is not what it says.
    """
    rng = np.random.default_rng(2)
    A = jnp.asarray(rng.random(a_shape))
    B = jnp.asarray(rng.random(b_shape))
    assert jnp.allclose(linalg._dot_broadcast(A, B), jnp.tensordot(A, B, axes=1))


def test_tensordot_is_not_jnp_matmul_for_rank3_operands():
    """Pinning the reason this is not called `matmul`.

    For a rank-3+ second operand jnp.matmul reads it as a stack of matrices and
    contracts against its second-to-last axis, so it returns a *different array* --
    no error, just a different answer. Anyone "simplifying" this to `@` changes
    results, not only speed.
    """
    rng = np.random.default_rng(6)
    A = jnp.asarray(rng.random((2, 3)))
    B = jnp.asarray(rng.random((3, 3, 4)))

    assert linalg._dot_broadcast(A, B).shape == (2, 3, 4)
    assert jnp.matmul(A, B).shape == (3, 2, 4)
    assert jnp.allclose(linalg._dot_broadcast(A, B), jnp.tensordot(A, B, axes=1))


def test_tensordot_vmaps_over_a_batch():
    rng = np.random.default_rng(3)
    A = jnp.asarray(rng.random((16, 2, 3)))
    B = jnp.asarray(rng.random((16, 3, 4)))
    assert jnp.allclose(
        jax.vmap(linalg._dot_broadcast)(A, B), jnp.einsum("epq,eqr->epr", A, B)
    )


def test_tensordot_rejects_a_stack_of_matrices():
    """Legal tensordot, but it means someone is hand-batching: this materialises the
    (p, q, r) product, so a batch axis allocates it in full. Use jax.vmap.
    """
    with pytest.raises(ValueError, match="single vector or matrix"):
        linalg._dot_broadcast(jnp.zeros((5, 2, 3)), jnp.zeros((3, 4)))


# --------------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "a_shape", "b_shape", "einsum"),
    [
        ("in,nj->ij", (2, 4), (4, 2), "in,nj->ij"),
        ("ij,jn->in", (3, 3), (3, 8), "ij,jn->in"),
        ("in,n...->i...", (2, 4), (4,), "in,n->i"),
        ("in,n...->i...", (2, 4), (4, 3), "in,nc->ic"),
        ("in,n...->i...", (2, 4), (4, 3, 3), "in,ncd->icd"),
        ("n,n...->...", (4,), (4,), "n,n->"),
        ("n,n...->...", (4,), (4, 3), "n,nc->c"),
        ("n,nj->j", (4,), (4, 2), "n,nj->j"),
    ],
)
def test_contract_computes_what_the_spec_says(spec, a_shape, b_shape, einsum):
    """The spec is a name, not an instruction -- but the name has to be honest. Every
    supported contraction is checked against the einsum it claims to be.
    """
    rng = np.random.default_rng(9)
    A = jnp.asarray(rng.random(a_shape))
    B = jnp.asarray(rng.random(b_shape))
    assert jnp.allclose(linalg.contract(spec, A, B), jnp.einsum(einsum, A, B))


def test_contract_is_indifferent_to_the_index_letters():
    """Specs are canonicalised by order of first appearance, so a call site may use
    letters that mean something without adding a table entry.
    """
    rng = np.random.default_rng(10)
    A = jnp.asarray(rng.random((2, 4)))
    B = jnp.asarray(rng.random((4, 3)))
    assert jnp.allclose(
        linalg.contract("in,nj->ij", A, B), linalg.contract("pq,qr->pr", A, B)
    )
    assert jnp.allclose(linalg.contract("ab,bc->ac", A, B), A @ B)


def test_contract_refuses_an_unmeasured_contraction():
    """Not a gap to be filled with a general einsum fallback. Each entry is a measured
    decision about how to spell the contraction so it does not lower to a `dot`; a spec
    with no entry has had no such decision made.
    """
    A = jnp.zeros((2, 3))
    B = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match="no implementation"):
        linalg.contract("ij,jk->ijk", A, B)  # no contraction at all
    with pytest.raises(ValueError, match="no implementation"):
        linalg.contract("ij,jk->ki", A, B)  # transposed output


@pytest.mark.parametrize("spec", ["in,nj->ij", "n,n...->...", "in,n...->i..."])
def test_contract_rejects_a_batch_axis(spec):
    """These take a single tensor each. The broadcast spelling materialises its
    intermediate, so a hand-rolled batch axis allocates it in full -- use jax.vmap.
    """
    with pytest.raises(ValueError, match="ranks"):
        linalg.contract(spec, jnp.zeros((5, 2, 3)), jnp.zeros((3, 4)))


def test_contract_rejects_a_mismatched_contraction_axis():
    with pytest.raises(ValueError, match="contracts A's last axis"):
        linalg.contract("in,nj->ij", jnp.zeros((2, 3)), jnp.zeros((4, 2)))


def test_contract_rejects_an_unparseable_spec():
    A = jnp.zeros((2, 3))
    B = jnp.zeros((3, 4))
    for spec in ["ij->i", "ij,jk", "ij,jk->ik->ik", "i1,1k->ik"]:
        with pytest.raises(ValueError, match="could not parse|no implementation"):
            linalg.contract(spec, A, B)


def test_contract_dispatches_the_matmul_spec_on_the_lowering_platform():
    """The one spec that consults the backend, and the reason it is
    `jax.lax.platform_dependent`.
    """
    A = jnp.ones((64, 2, 4))
    B = jnp.ones((64, 4, 2))
    fn = jax.jit(jax.vmap(lambda a, b: linalg.contract("in,nj->ij", a, b)))
    traced = fn.trace(A, B)

    cpu = traced.lower(lowering_platforms=("cpu",)).as_text()
    assert cpu.count("dot_general") == 1  # `@`
    assert cpu.count("stablehlo.multiply") == 0

    gpu = traced.lower(lowering_platforms=("cuda",)).as_text()
    assert gpu.count("dot_general") == 0  # broadcast
    assert gpu.count("stablehlo.multiply") == 1

    for text in (cpu, gpu):
        assert "conditional" not in text  # pruned, not selected at runtime


@pytest.mark.parametrize("spec", ["in,n...->i...", "n,n...->..."])
def test_contract_does_not_dispatch_the_node_contractions(spec):
    """Broadcast on *both* backends, deliberately. On CPU the faster spelling at this
    site reverses with the element type and the component count (see `contract`), and a
    rule that reverses is not a rule -- so this spec does not get a platform branch.
    """
    A = jnp.ones((64, 2, 4)) if spec.startswith("in") else jnp.ones((64, 4))
    B = jnp.ones((64, 4, 3))
    traced = jax.jit(jax.vmap(lambda a, b: linalg.contract(spec, a, b))).trace(A, B)

    for platform in ("cpu", "cuda"):
        text = traced.lower(lowering_platforms=(platform,)).as_text()
        assert text.count("dot_general") == 0
        assert text.count("stablehlo.multiply") == 1


def test_contract_matmul_branches_agree_where_it_dispatches():
    """A backend-dependent spelling is only safe if the spellings agree. They do at the
    one spec that dispatches -- and would not for the node contractions, which is a
    second reason those are pinned to one spelling.
    """
    rng = np.random.default_rng(11)
    A = jnp.asarray(rng.random((2, 4)))
    B = jnp.asarray(rng.random((4, 3)))
    assert jnp.allclose(A @ B, linalg._dot_broadcast(A, B))


def test_every_contract_spec_in_the_library_is_in_the_table():
    """A spec is a string, so a typo in one is a runtime error on a path that may only
    run for one element type in one dimension. This reads every ``linalg.contract`` literal
    in the package and resolves it, so a bad spec fails here rather than the first time
    someone meshes with a Line3.
    """
    package = pathlib.Path(linalg.__file__).parent
    literals = {
        match.group(1)
        for path in package.rglob("*.py")
        for match in re.finditer(
            r"""linalg\.contract\(\s*["']([^"']+)["']""", path.read_text()
        )
    }
    assert literals, "found no linalg.contract call sites -- has the API been renamed?"

    for spec in sorted(literals):
        canonical = linalg._canonicalise(spec)
        assert canonical in linalg._known_implementations, (
            f"{spec!r} is used in the package but canonicalises to {canonical!r}, which "
            "has no entry in _known_implementations"
        )


@pytest.mark.parametrize(
    ("spec", "a_shape", "b_shape", "einsum"),
    [
        ("in,nj->ij", (2, 4), (4, 2), "in,nj->ij"),
        ("in,n...->i...", (2, 4), (4, 3), "in,nc->ic"),
        ("n,n...->...", (4,), (4, 3), "n,nc->c"),
        ("n,nj->j", (4,), (4, 2), "n,nj->j"),
    ],
)
def test_contract_differentiates_like_the_einsum_it_names(
    spec, a_shape, b_shape, einsum
):
    """The whole library reaches these through jax.grad, so agreeing on the primal is
    only half the contract. `platform_dependent` in particular puts a branch between the
    caller and the arithmetic, and that has to be transparent to AD.
    """
    rng = np.random.default_rng(12)
    A = jnp.asarray(rng.random(a_shape))
    B = jnp.asarray(rng.random(b_shape))

    mine = jax.grad(lambda a, b: linalg.contract(spec, a, b).sum(), argnums=(0, 1))(
        A, B
    )
    reference = jax.grad(lambda a, b: jnp.einsum(einsum, a, b).sum(), argnums=(0, 1))(
        A, B
    )
    for got, want in zip(mine, reference):
        assert jnp.allclose(got, want)


@pytest.mark.parametrize(
    ("spec", "a_shape", "b_shape", "einsum"),
    [
        ("in,nj->ij", (2, 4), (4, 2), "ein,enj->eij"),
        ("in,n...->i...", (2, 4), (4, 3), "ein,enc->eic"),
        ("n,n...->...", (4,), (4, 3), "en,enc->ec"),
        ("n,nj->j", (4,), (4, 2), "en,enj->ej"),
    ],
)
def test_contract_vmaps_over_elements(spec, a_shape, b_shape, einsum):
    """How every call site actually reaches it: one element's worth of arithmetic,
    batched by vmap rather than by hand.
    """
    rng = np.random.default_rng(13)
    A = jnp.asarray(rng.random((16, *a_shape)))
    B = jnp.asarray(rng.random((16, *b_shape)))
    batched = jax.vmap(lambda a, b: linalg.contract(spec, a, b))(A, B)
    assert jnp.allclose(batched, jnp.einsum(einsum, A, B))


# --------------------------------------------------------------------------------
# einsum: the drop-in wrapper over contract
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "shapes", "path"),
    [
        # in the table: must reach the measured implementation
        ("in,nj->ij", [(2, 4), (4, 2)], "contract"),
        ("pq,qr->pr", [(2, 4), (4, 2)], "contract"),  # same contraction, other letters
        ("in,n...->i...", [(2, 4), (4, 3)], "contract"),
        ("n,n...->...", [(4,), (4, 3)], "contract"),
        # not in the table: falls back rather than raising, which is what makes this a
        # drop-in replacement
        ("eq,q->eq", [(64, 4), (4,)], "jnp.einsum"),  # sums nothing
        ("ij,jk->ki", [(2, 3), (3, 4)], "jnp.einsum"),  # transposed output
        ("ij->ji", [(2, 3)], "jnp.einsum"),  # one operand, contract cannot parse it
        ("ij,jk", [(2, 3), (3, 4)], "jnp.einsum"),  # implicit output
        ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)], "jnp.einsum"),  # three operands
    ],
)
def test_einsum_takes_the_contract_path_only_for_a_measured_spec(
    spec, shapes, path, monkeypatch
):
    """Which path a call takes is the whole point of the wrapper and is invisible in the
    result -- both paths compute the same numbers, only the HLO differs. A table spec
    that falls through to jnp.einsum silently reintroduces the `dot` this module exists
    to avoid, so the path is asserted directly.
    """
    rng = np.random.default_rng(14)
    arrays = [jnp.asarray(rng.random(shape)) for shape in shapes]
    reference = jnp.einsum(spec, *arrays)

    taken: list[str] = []
    real_contract, real_einsum = linalg.contract, jnp.einsum
    monkeypatch.setattr(
        linalg,
        "contract",
        lambda s, *o: (taken.append("contract"), real_contract(s, *o))[1],
    )
    monkeypatch.setattr(
        jnp,
        "einsum",
        lambda s, *o, **k: (taken.append("jnp.einsum"), real_einsum(s, *o, **k))[1],
    )

    result = linalg.einsum(spec, *arrays)

    assert taken == [path]
    assert jnp.allclose(result, reference)


def test_every_einsum_spec_in_the_library_stays_off_the_dot_path():
    """The counterpart to `test_every_contract_spec_in_the_library_is_in_the_table`.

    `linalg.einsum` falls back rather than refusing, so an unmeasured spec written at a
    call site is accepted quietly and lowers to the `dot` this module exists to avoid --
    the silent regression the fallback makes possible. Every literal in the package must
    therefore either resolve to a table entry or sum nothing at all.
    """
    package = pathlib.Path(linalg.__file__).parent
    literals = {
        match.group(1)
        for path in package.rglob("*.py")
        for match in re.finditer(
            r"""linalg\.einsum\(\s*["']([^"']+)["']""", path.read_text()
        )
    }

    for spec in sorted(literals):
        canonical = linalg._canonicalise(spec)
        assert canonical in linalg._known_implementations, (
            f"{spec!r} is used in the package and sums over a shared index, but "
            f"canonicalises to {canonical!r}, which has no entry in "
            "_known_implementations -- linalg.einsum will hand it to jnp.einsum and it "
            "will lower to a dot. Add a measured entry, or use jnp.einsum knowingly."
        )


# --------------------------------------------------------------------------------
# the regression that numerical tests cannot catch
# --------------------------------------------------------------------------------


def _compiled_hlo(fn, *args) -> str:
    return jax.jit(fn).lower(*args).compile().as_text()


def _gemm_ops(hlo: str) -> dict[str, int]:
    return {
        "dot": hlo.count(" dot("),
        "cublas": hlo.count('custom_call_target="__cublas'),
        "getrf": hlo.count("getrf"),
        "triangular_solve": hlo.count("triangular-solve"),
    }


@pytest.mark.parametrize(
    "element, dim",
    [(Tri3(), 2), (Quad4(), 2), (Tetrahedron4(), 3), (Hexahedron8(), 3)],
)
def test_element_kernels_avoid_lapack_and_cublas(element, dim):
    """`Element.gradient` and `get_jacobian` must never reach LAPACK or cuBLAS.

    If this fails, someone has put `jnp.linalg.inv` back in place of `linalg.inv` -- an LU
    factorisation plus a triangular solve, per element, measured 77x slower than the
    closed form. The numbers stay correct, only the speed changes.

    Note what is deliberately NOT asserted here: the `dot` count. Whether a given
    contraction becomes a `dot_general` depends on the *call context*, not just the
    source line. Vmapped directly over a contiguous array, as below, `dNdr @ coords`
    compiles to a dot; reached through `Operator`, where the operands arrive from a
    gather, the identical line fuses to none. The dot-free guarantee is therefore
    asserted at the Operator level, in the test below, which is the path tatva uses.
    """
    rng = np.random.default_rng(4)
    n_nodes = element._reference_nodes().shape[0]
    xi = element.quad_points[0]
    ref = jnp.asarray(element._reference_nodes())
    coords = ref + 0.05 * jnp.asarray(rng.standard_normal((64, n_nodes, dim)))
    values = jnp.asarray(rng.random((64, n_nodes, dim)))

    grad_hlo = _compiled_hlo(
        lambda v, c: jax.vmap(lambda a, b: element.gradient(xi, a, b))(v, c),
        values,
        coords,
    )
    jac_hlo = _compiled_hlo(
        lambda c: jax.vmap(lambda b: element.get_jacobian(xi, b))(c), coords
    )

    for hlo in (grad_hlo, jac_hlo):
        ops = _gemm_ops(hlo)
        assert ops["cublas"] == 0
        assert ops["getrf"] == 0
        assert ops["triangular_solve"] == 0


@pytest.mark.parametrize("kind, element", [("quad", Quad4()), ("triangle", Tri3())])
def test_operator_entry_points_avoid_the_gemm_path(kind, element):
    """The dot-free guarantee, asserted on the path tatva actually runs.

    Every `Operator` entry point must compile to fused loops. A `dot` here means a
    contraction slipped back to an einsum -- `Element.gradient`'s node contraction as
    `jnp.einsum("dn,n...->...d", ...)` measured 54-90 ms against 4.7 ms, and
    `_integrate_quad_array` as `jnp.einsum("eq...,eq->e...", ...)` cost 4x.
    """
    mesh = Mesh.unit_square(16, 16, type=kind)
    op = Operator(mesh, element, cache_weights=False)
    u = jnp.asarray(np.random.default_rng(5).random((mesh.coords.shape[0], 2)))

    for hlo in (
        _compiled_hlo(lambda x: op.grad(x), u),
        _compiled_hlo(lambda x: op.integrate(x), u),
        _compiled_hlo(lambda: op.get_integration_weights()),
    ):
        assert _gemm_ops(hlo) == dict(dot=0, cublas=0, getrf=0, triangular_solve=0)


# --------------------------------------------------------------------------------
# Line elements: get_jacobian deliberately keeps `@`
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("element, nodes_per_elem", [(Line2(), 2), (Line3(), 3)])
@pytest.mark.parametrize("dim", [2, 3])
def test_line_elements_measure_arc_length_in_2d_and_3d(element, nodes_per_elem, dim):
    """A helix has an analytic arc length. This also pins the fix that let Line2 work
    with 3-D coordinates at all -- it used to build its unit tangent as [J[0], J[1]].
    """
    n_elem, turns, radius, pitch = 400, 1.0, 2.0, 3.0
    n_nodes = n_elem * (nodes_per_elem - 1) + 1
    s = np.linspace(0.0, 2 * np.pi * turns, n_nodes)
    cols = [radius * np.cos(s), radius * np.sin(s)]
    if dim == 3:
        cols.append(pitch * s / (2 * np.pi * turns))
    coords = jnp.asarray(np.stack(cols, axis=-1))

    if nodes_per_elem == 2:
        elements = np.stack([np.arange(n_elem), np.arange(n_elem) + 1], axis=-1)
    else:  # Line3 stores (end, end, mid)
        first = np.arange(n_elem) * 2
        elements = np.stack([first, first + 2, first + 1], axis=-1)

    mesh = Mesh(coords=coords, elements=jnp.asarray(elements))
    length = (
        Operator(mesh, element, cache_weights=False).get_integration_weights().sum()
    )

    exact = turns * np.sqrt(
        (2 * np.pi * radius) ** 2 + (pitch if dim == 3 else 0.0) ** 2
    )
    assert float(abs(length - exact) / exact) < 1e-4
