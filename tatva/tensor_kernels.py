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


"""Small dense tensor operations, written to avoid XLA's ``dot`` path on GPU.

In a tatva at element level, operations happen on tiny arrays: a 2x2 Jacobian, a
(2, 4) matrix of shape function derivatives. But the obvious way of writing tiny operations
as vmapped over millions of elements at once makes the code is dramatically slower
than it should be.

When we write ``A @ B`` or any einsum that sums over a shared index, XLA emits a
``dot_general`` and hands the work to the machinery built for matrix products: on
GPU, a batched GEMM kernel. That machinery is great for large matrices as it tiles them,
stages tiles into fast on-chip memory, and keeps the arithmetic units saturated.
All of that setup has a fixed cost per matrix.

Our matrices are 2x2. We pay the setup cost millions of times to do almost no
arithmetic. Worse, a GEMM is a hard boundary for the compiler which means it cannot
merge/fuse that step into the cheap operations on either side, so every intermediate result
is written out to main memory and read straight back in. The data movement, not
the arithmetic, becomes the whole cost.

``jnp.linalg.inv`` is the same story. It runs an LU factorisation and a triangular solve
which is correct and stable for a general matrix, but absurd for a 2x2 you can invert
with a division.

Writing the same arithmetic as a broadcast multiply followed by a sum gives
nothing to pattern-match. It stays an ordinary loop, gets fused into its
neighbours, and the data is touched once in a single pass.

How to spot it
--------------
The best way to spot it by reading the spec. Compile the function and count::

    txt = jax.jit(fn).lower(*args).compile().as_text()
    txt.count(" dot(")                            # any dot_general at all
    txt.count('custom_call_target="__cublas')     # handed to cuBLAS

If you must reason about a spec without compiling it, the rule is short: an einsum
becomes a ``dot`` if, and only if, it **sums over an index shared by both operands**.
Free indices have nothing to do with it. Every index in a two-operand spec is one of
four things, and XLA's ``dot_general`` is parameterised by the last two::

    index   in A   in B   in output   role
    p       yes    no     yes         free  (row)
    ...     no     yes    yes         free  (column)
    q       yes    yes    NO          contracted   -> dot_general
    e       yes    yes    yes         batch        -> dot_general, with batch dims

So all of ``"ij,jk->ik"``, ``"n,n...->..."``, ``"i,i->"`` and ``"eq...,eq->e..."``
are dots, however little they look like matrix products; while ``"eq,q->eq"`` and
``"ij,kl->ijkl"`` are not, because nothing is summed. Several specs in tatva were
assumed safe on the grounds that they "read like reductions" and every one of them
cost 2-5x. Note the last row. ``"eq...,eq->e..."`` is the *same* contraction as
``"q,q...->..."`` with a batch index bolted on -- it is what `jax.vmap` would build
for you. Writing that by hand is how `_integrate_quad_array` ended up emitting a
dot over the whole mesh.

What it cost us
---------------
Measured on 200k Quad4 elements, before and after this rewrite:

    Operator.grad     CPU  1117 ms  ->  43.8 ms
                      GPU  8.17 ms  ->  0.27 ms

The inverse dominated on CPU (~450x on its own); the matmuls and the einsum
dominated on GPU (~230x). Note that the two backends fail differently, so
measuring only one of them points at the wrong culprit.

Please do not "simplify" these backends
--------------------------------------
Reverting to ``@``, ``jnp.einsum`` or ``jnp.linalg.inv`` leaves every numerical
test passing and the results correct to the last digit. The only thing that changes
is that the code becomes far slower, silently. That is what
``tests/test_tensor_kernels.py::test_element_kernels_avoid_the_gemm_path`` is for:
it compiles the element kernels and asserts the HLO holds no ``dot``, no cuBLAS
call and no LU, so the regression fails loudly instead.

One exception is not a simplification but a correctness change: ``tk.tensordot`` is
``jnp.tensordot(A, B, axes=1)``, *not* ``jnp.matmul``. The two disagree once ``B``
is rank 3 or more, silently and with no error.

Say what you mean: ``tk.contract``
---------------------------------
The entry point is ``tk.contract(spec, A, B)``, and the spec is an einsum-style
string naming the contraction::

    J    = tk.contract("in,nj->ij",    dNdr,        nodal_coords)
    dNdX = tk.contract("ij,jn->in",    tk.inv(J),   dNdr)
    grad = tk.contract("in,n...->i...", dNdX,       nodal_values)
    val  = tk.contract("n,n...->...",  N,           nodal_values)

The spec is a *name*, not an instruction. It is never handed to ``jnp.einsum`` --
every one of these sums over a shared index, so einsum would lower all four to a
``dot``, which is the thing this module exists to avoid. It is resolved at trace
time against ``_known_implementationss``, a four-line table holding how each contraction
is spelled and why.

Why a spec rather than differently-named functions: how a contraction should be
spelled does not line up with anything visible at the call site. At rank 2,
``"in,nj->ij"`` and ``"in,n...->i..."`` are the same kind of product and compute
the same thing, yet they are spelled differently, for reasons that come out of
measurement rather than shape. A name like ``matmul`` cannot carry that, and a
reader cannot infer it. The spec says *what*; the table says *how*.

Only one entry consults the backend, ``_MATMUL``: ``@`` on CPU, broadcast on GPU.
It earns the dispatch because ``@`` wins 5-7x on CPU across every element type with
no reversal. The node contraction is broadcast on both backends even though the CPU
picture there is mixed -- ``@`` wins on Tri3/Quad4, loses 1.6x on Hex8 with a scalar
field, ties on Quad4 with a vector field -- because a rule that reverses with the
element type and the component count is not a rule. See `contract` for the numbers
and what always-broadcast costs.

When not to use these
---------------------
These take one tensor, never a stack of tensors -- batching is ``jax.vmap``'s job,
and both functions reject a batch axis rather than quietly returning nonsense. That
is not just a naming preference: ``tensordot`` materialises its intermediate (see
below), so hand-batching it defeats the entire purpose of the module.

These routines win *because* the arrays are small. ``tensordot`` builds a
(p, q, r) intermediate before reducing it, which is free at the sizes one element
produces and ruinous once a dimension reaches the hundreds. If a contracted
dimension can grow large, use ``@`` and let XLA call the GEMM -- large matrices are
the case that machinery was built for, and there it wins.
"""

import jax
import jax.numpy as jnp
from jax import Array


def inv(J: Array) -> Array:
    """Closed-form inverse of a 1x1, 2x2 or 3x3 matrix.

    `jnp.linalg.inv` goes through an LU factorization plus a triangular solve.
    That is right for one large matrix and ruinous when vmapped over millions of
    tiny element Jacobians. The branch is on a static shape, so it is resolved at
    trace time -- there is no runtime dispatch.

    For 1x1, 2x2 and 3x3 matrices, the inverse is computed in closed form. For larger
    matrices, it falls back to `jnp.linalg.inv`.

    Args:
        J: a single matrix, shape (d, d) with d in {1, 2, 3}.

    Returns:
        The inverse of J, shape (d, d).

    """
    if J.ndim != 2:
        raise ValueError(
            f"tensor_kernels.inv takes a single matrix, got shape {J.shape}. Batch with jax.vmap "
            "rather than passing a stack"
        )

    d = J.shape[-1]
    if J.shape[-2] != d:
        return jnp.linalg.inv(J)  # not square: let JAX raise as before
    if d == 1:
        return 1.0 / J
    if d == 2:
        a, b = J[0, 0], J[0, 1]
        c, e = J[1, 0], J[1, 1]
        det = a * e - b * c
        return jnp.stack([jnp.stack([e, -b]), jnp.stack([-c, a])]) / det
    if d == 3:
        m = J
        c00 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
        c01 = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
        c02 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
        c10 = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
        c11 = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
        c12 = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]
        c20 = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
        c21 = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
        c22 = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        det = m[0, 0] * c00 + m[0, 1] * c01 + m[0, 2] * c02
        return (
            jnp.stack(
                [
                    jnp.stack([c00, c10, c20]),
                    jnp.stack([c01, c11, c21]),
                    jnp.stack([c02, c12, c22]),
                ]
            )
            / det
        )
    return jnp.linalg.inv(J)


def _tensordot(A: Array, B: Array) -> Array:
    """
    Contracts the last axis of `A` with the first axis of `B` and leaves every other
    axis free: (q,) or (p, q), against (q, ...). That single rule covers both shapes
    tatva needs, the shape-function contraction ``N . nodal_values``, and the
    matrix products inside `Element.gradient`.

    Both operands are padded with singleton axes so broadcasting lines the contracted
    axis up: `A` gains one trailing axis per free axis of `B`, `B` one leading axis
    per free axis of `A`. Both counts are derived from the operand ranks.

    We cannot use `jnp.tensordot` as it lowers to a `dot`, which is the entire
    problem. A rank-3+ `A` is refused. That is legal `tensordot`, but it means someone is
    hand-batching: this materialises the (p, q, r) product before reducing it, which
    is free at the sizes one element produces and ruinous once a dimension is large.
    Batch with `jax.vmap`, which fuses the reduction into the surrounding work.

    Args:
        A: a single vector or matrix, shape (q,) or (p, q).
        B: a single tensor, shape (q, ...).

    Returns:
        The contraction, shape (p, ...) or (...), with axes in the order they appear in `B`
        after the contracted axis.
    """
    if A.ndim > 2:
        raise ValueError(
            f"tensor_kernels.tensordot takes a single vector or matrix as A, got shape {A.shape}. "
            "Batch with jax.vmap rather than passing a stack: this builds the full "
            "outer product before reducing, so a batch axis materialises it in full."
        )

    Ae = A.reshape(A.shape + (1,) * (B.ndim - 1))
    Be = B.reshape((1,) * (A.ndim - 1) + B.shape)
    return jnp.sum(Ae * Be, axis=A.ndim - 1)


# --------------------------------------------------------------------------------
# contract is a drop-in replacement for einsum with a fixed set of contractions
# --------------------------------------------------------------------------------
#


def _dot(A: Array, B: Array) -> Array:
    """The contraction as XLA's `dot`. Fast where XLA can fuse or fold it away.

    Args:
        A: a single vector or matrix, shape (q,) or (p, q).
        B: a single tensor, shape (q, ...) or (q, r).

    Returns:
        The contraction, shape (p, ...) or (...), with axes in the order they appear in `B`
        after the contracted axis.
    """

    return A @ B


def _broadcast(A: Array, B: Array) -> Array:
    """The contraction as broadcast-multiply-reduce. Never lowers to a `dot`.
    Hence never cross the boundary with cublas for GEMM on GPU.

    Args:
        A: a single vector or matrix, shape (q,) or (p, q).
        B: a single tensor, shape (q, ...) or (q, r).

    Returns:
        The contraction, shape (p, ...) or (...), with axes in the order they appear in `B` after the contracted axis.
    """
    return _tensordot(A, B)


def _dot_on_cpu_broadcast_elsewhere(A: Array, B: Array) -> Array:
    """
    The only spec-dependent dispatch, and the only place a backend is consulted.
    On CPU, `dot` is fused away by XLA, so we use `@` to avoid the broadcast form.
    But only for matrix-matrix products: `dot` is not fused for vector-vector or matrix-vector.
    On GPU, we avoid `dot` altogether and use use _broadcast form instead.
    """
    return jax.lax.platform_dependent(A, B, cpu=_dot, default=_broadcast)


# Currently in tatva we have four contractions: matmul, vecmat, row_contract, and vec_contract:
_MATMUL = "ab,bc->ac"  # (p, q) x (q, r): the element Jacobian and the product after it
_VECMAT = "a,ab->b"  # (q,) x (q, r): the Line2/Line3 tangent vector
_ROW_CONTRACT = "ab,b...->a..."  # (p, q) x (q, ...): dNdX against nodal values
_VEC_CONTRACT = "a,a...->..."  # (q,) x (q, ...): N against nodal values


_known_implementations = {
    _MATMUL: _dot_on_cpu_broadcast_elsewhere,  # matrix-matrix: dot on CPU, broadcast otherwise
    _VECMAT: _dot,  # vector-matrix: dot everywhere
    _ROW_CONTRACT: _broadcast,  # row-contract: broadcast everywhere
    _VEC_CONTRACT: _broadcast,  # vec-contract: broadcast everywhere
}

# (A.ndim, B.ndim); None means "any rank >= 1", i.e. the spec has an ellipsis there.
_known_ranks = {
    _MATMUL: (2, 2),
    _VECMAT: (1, 2),
    _ROW_CONTRACT: (2, None),
    _VEC_CONTRACT: (1, None),
}


def _tokenize(term: str) -> list[str]:
    """Split a spec term into index letters, with ``...`` as a single token.
    For example, ``"in,n...->i..."`` is split into ``["in", "...", "->", "i..."]``.

    Args:
        term: The spec term to tokenize.

    Returns:
        The list of tokens.
    """
    tokens, i = [], 0
    while i < len(term):
        if term.startswith("...", i):
            tokens.append("...")
            i += 3
        else:
            tokens.append(term[i])
            i += 1
    return tokens


def _canonicalise(spec: str) -> str:
    """Rename index letters in order of first appearance.

    So a call may use letters that mean something -- ``"in,nj->ij"`` for the
    Jacobian, ``"in,n...->i..."`` for the node contraction -- while the lookup table
    holds one entry per *contraction*, not one per spelling.
    """
    cleaned = spec.replace(" ", "")
    if cleaned.count("->") != 1 or cleaned.count(",") != 1:
        raise ValueError(
            f"tk.contract could not parse the spec {spec!r}. Expected exactly one ',' "
            "and one '->', as in 'in,nj->ij'."
        )
    lhs, out = cleaned.split("->")
    a_term, b_term = lhs.split(",")

    renaming: dict[str, str] = {}

    def rename(term: str) -> str:
        letters = []
        for token in _tokenize(term):
            if token == "...":
                letters.append(token)
                continue
            if not token.isalpha():
                raise ValueError(
                    f"tk.contract could not parse the spec {spec!r}: {token!r} is not an "
                    "index letter."
                )
            if token not in renaming:
                renaming[token] = chr(ord("a") + len(renaming))
            letters.append(renaming[token])
        return "".join(letters)

    return f"{rename(a_term)},{rename(b_term)}->{rename(out)}"


def contract(spec: str, A: Array, B: Array) -> Array:
    """Contract two element-sized tensors, named by an einsum-style spec.

        J    = tk.contract("in,nj->ij", dNdr, nodal_coords)     # (d, n) x (n, d)
        dNdX = tk.contract("ij,jn->in", tk.inv(J), dNdr)        # (d, d) x (d, n)
        grad = tk.contract("in,n...->i...", dNdX, nodal_values)  # over the node axis
        val  = tk.contract("n,n...->...", N, nodal_values)      # over the node axis

    Index letters are free. They are canonicalised by order of first appearance, so
    ``"in,nj->ij"`` and ``"pq,qr->pr"`` are the same entry, and a call site may use
    letters that carry meaning. An unrecognised contraction is refused rather than
    given a general fallback: a new spec needs a measured decision about how to spell
    it, which is exactly what this function exists to record.

    Args:
        spec: einsum-style contraction, e.g. ``"in,nj->ij"``. Must name one of the four
            supported contractions; letters are arbitrary.
        A: first operand, a single tensor.
        B: second operand, a single tensor.

    Returns:
        The contraction, with axes in the order the spec's output term gives.

    Raises:
        ValueError: if the spec is unparseable, names a contraction with no measured
            implementation, or disagrees with the operand shapes.
    """
    canonical = _canonicalise(spec)
    implementation = _known_implementations.get(canonical)
    if implementation is None:
        supported = ", ".join(repr(s) for s in _known_implementations)
        raise ValueError(
            f"tk.contract has no implementation for {spec!r} (canonically {canonical!r})."
            f" Supported contractions are {supported}. This is deliberate rather than a "
            "gap: each entry is a measured decision about how to spell the contraction "
            "so it does not lower to a `dot`. Add an entry, with the measurement, rather "
            "than reaching for jnp.einsum -- see the module docstring."
        )

    expected_a, expected_b = _known_ranks[canonical]
    got = (A.ndim, B.ndim)
    if A.ndim != expected_a or (expected_b is not None and B.ndim != expected_b):
        wanted = f"({expected_a}, {expected_b if expected_b is not None else '>=1'})"
        raise ValueError(
            f"tk.contract({spec!r}, ...) wants operand ranks {wanted}, got {got} from "
            f"shapes {A.shape} and {B.shape}. These take a single tensor each -- batch "
            "with jax.vmap, which fuses the contraction into the surrounding work "
            "instead of materialising it."
        )
    if A.shape[-1] != B.shape[0]:
        raise ValueError(
            f"tk.contract({spec!r}, ...) contracts A's last axis with B's first, but "
            f"they are {A.shape[-1]} and {B.shape[0]} from shapes {A.shape} and "
            f"{B.shape}."
        )

    return implementation(A, B)
