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


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tatva.tensor_kernels as tk
from tatva import Mesh, Operator
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
    assert jnp.allclose(tk.inv(J), jnp.linalg.inv(J))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inv_round_trips(d):
    J = _well_conditioned(d)
    assert jnp.allclose(tk.inv(J) @ J, jnp.eye(d), atol=1e-12)


@pytest.mark.parametrize("d", [2, 3])
def test_inv_vmaps_over_a_batch(d):
    """Batching is vmap's job -- the kernel itself stays unbatched."""
    rng = np.random.default_rng(1)
    stack = jnp.eye(d) + 0.1 * jnp.asarray(rng.standard_normal((16, d, d)))
    assert jnp.allclose(jax.vmap(tk.inv)(stack), jnp.linalg.inv(stack))


@pytest.mark.parametrize("shape", [(4, 2, 2), (3, 3, 3), (2,)])
def test_inv_rejects_anything_that_is_not_a_single_matrix(shape):
    with pytest.raises(ValueError, match="single matrix"):
        tk.inv(jnp.zeros(shape))


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
    assert jnp.allclose(tk.tensordot(A, B), jnp.tensordot(A, B, axes=1))


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

    assert tk.tensordot(A, B).shape == (2, 3, 4)
    assert jnp.matmul(A, B).shape == (3, 2, 4)
    assert jnp.allclose(tk.tensordot(A, B), jnp.tensordot(A, B, axes=1))


def test_tensordot_vmaps_over_a_batch():
    rng = np.random.default_rng(3)
    A = jnp.asarray(rng.random((16, 2, 3)))
    B = jnp.asarray(rng.random((16, 3, 4)))
    assert jnp.allclose(jax.vmap(tk.tensordot)(A, B), jnp.einsum("epq,eqr->epr", A, B))


def test_tensordot_rejects_a_stack_of_matrices():
    """Legal tensordot, but it means someone is hand-batching: this materialises the
    (p, q, r) product, so a batch axis allocates it in full. Use jax.vmap.
    """
    with pytest.raises(ValueError, match="single vector or matrix"):
        tk.tensordot(jnp.zeros((5, 2, 3)), jnp.zeros((3, 4)))


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
def test_element_kernels_avoid_the_gemm_path(element, dim):
    """`Element.gradient` and `get_jacobian`, vmapped over elements, must compile to
    fused loops -- no dot_general, no cuBLAS, no LU.

    If this fails, someone has replaced a `tk.tensordot` / `tk.inv` with `@`,
    `jnp.einsum("dn,n...->...d", ...)` or `jnp.linalg.inv`. The numbers will still be
    right; the code will just be far slower. See the module docstring.
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

    assert _gemm_ops(grad_hlo) == dict(dot=0, cublas=0, getrf=0, triangular_solve=0)
    assert _gemm_ops(jac_hlo) == dict(dot=0, cublas=0, getrf=0, triangular_solve=0)


def test_operator_grad_and_integrate_avoid_the_gemm_path():
    """Same guarantee one level up, through the Operator entry points."""
    mesh = Mesh.unit_square(16, 16, type="quad")
    op = Operator(mesh, Quad4(), cache_weights=False)
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
