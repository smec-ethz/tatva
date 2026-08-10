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


"""Small dense tensor operations, written to avoid XLA's ``dot`` path.

Everything in a tatva happens on tiny arrays: a 2x2 Jacobian, a
(2, 4) matrix of shape function derivatives. But those tiny operations are
vmapped over millions of elements at once. That combination of very small
matrices but very large batch is the one case where the obvious way to write the
code is dramatically slower than it should be.

When we write ``A @ B`` or any einsum that sums over a shared index, XLA emits a
``dot_general`` and hands the work to the machinery built for matrix products: on
GPU, a batched GEMM kernel. "Matrix product" is the machinery's idea, not ours --
``"n,n...->..."`` gets the same treatment, see "How to spot it" below.

That machinery is great for large matrices as it tiles them, stages tiles into fast
on-chip memory, and keeps the arithmetic units saturated. All of that setup has a
fixed cost per matrix.

Our matrices are 2x2. We pay the setup cost millions of times to do almost no
arithmetic. Worse, a GEMM is a hard boundary for the compiler which means it cannot merge
that step into the cheap operations on either side, so every intermediate result
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
cost 2-5x.

Note the last row. ``"eq...,eq->e..."`` is the *same* contraction as
``"q,q...->..."`` with a batch index bolted on -- it is what `jax.vmap` would build
for you. Writing that by hand is how `_integrate_quad_array` ended up emitting a
dot over the whole mesh.

Two things this rule does NOT tell you, both of which cost real time here:

A dot is not always a batched tiny GEMM. XLA may hoist a constant operand out of the
vmap and emit one large matrix product instead -- assumed to be "fine, that is what
GEMM is for", but a skinny k=4 GEMM is memory-bound and still lost 2.5-3.7x to a
fused broadcast.

And the converse: ``Line2/Line3.get_jacobian`` writes ``dNdr @ coords`` and XLA emits
no ``dot`` at all, fusing it perfectly; rewriting *that* by hand was 7x slower.

The rule that survived every measurement: rewrite a site if, and only if, its
compiled HLO contains a ``dot``. `tests/test_tensor_kernels.py` asserts this for the
element kernels and the `Operator` entry points.

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

import jax.numpy as jnp
from jax import Array


def inv(J: Array) -> Array:
    """Closed-form inverse of a 1x1, 2x2 or 3x3 matrix.

    `jnp.linalg.inv` goes through an LU factorization plus a triangular solve.
    That is right for one large matrix and ruinous when vmapped over millions of
    tiny element Jacobians. The branch is on a static shape, so it is resolved at
    trace time -- there is no runtime dispatch.
    """
    if J.ndim != 2:
        raise ValueError(
            f"tk.inv takes a single matrix, got shape {J.shape}. Batch with jax.vmap "
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


def tensordot(A: Array, B: Array) -> Array:
    """``jnp.tensordot(A, B, axes=1)`` without the dot path.

    Contracts the last axis of `A` with the first axis of `B` and leaves every other
    axis free: (q,) or (p, q), against (q, ...). That single rule covers both shapes
    tatva needs -- the shape-function contraction ``N . nodal_values``, and the
    matrix products inside `Element.gradient` -- so they are one function, not two.

    Both operands are padded with singleton axes so broadcasting lines the contracted
    axis up: `A` gains one trailing axis per free axis of `B`, `B` one leading axis
    per free axis of `A`. Both counts are derived from the operand ranks.

    We cannot `jnp.tensordot` as it lowers to a `dot`, which is the entire
    problem. Vmapped over millions of elements at quad shapes, `jnp.tensordot` and
    `jnp.matmul` both compile to ``dot=1``, this to ``dot=0``. Measured as a
    replacement for the equivalent einsums, per `Operator` call at 90k-400k elements:

        Element.interpolate  Tri3 3.7x   Quad4 2.5x   Line2 1.7x   Line3 2.7x
        Line gradients       Line2 4.9x  Line3 4.2x

    A rank-3+ `A` is refused. That is legal `tensordot`, but it means someone is
    hand-batching: this materialises the (p, q, r) product before reducing it, which
    is free at the sizes one element produces and ruinous once a dimension is large.
    Batch with `jax.vmap`, which fuses the reduction into the surrounding work.

    Finally, this is not a blanket replacement for ``@``. `Line2/Line3.get_jacobian`
    keeps plain ``@`` because XLA emits no `dot` there at all, and routing it through
    this function measured 7x *slower*. Rewrite a site only when its compiled HLO
    actually contains a `dot` -- see the module docstring.
    """
    if A.ndim > 2:
        raise ValueError(
            f"tk.tensordot takes a single vector or matrix as A, got shape {A.shape}. "
            "Batch with jax.vmap rather than passing a stack: this builds the full "
            "outer product before reducing, so a batch axis materialises it in full."
        )

    Ae = A.reshape(A.shape + (1,) * (B.ndim - 1))
    Be = B.reshape((1,) * (A.ndim - 1) + B.shape)
    return jnp.sum(Ae * Be, axis=A.ndim - 1)
