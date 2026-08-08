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


"""Small dense tensor operations, written to avoid XLA's matmul path.

Everything in a tatva happens on tiny arrays: a 2x2 Jacobian, a
(2, 4) matrix of shape function derivatives. But those tiny operations are
vmapped over millions of elements at once. That combination of very small
matrices but very large batch is the one case where the obvious way to write the
code is dramatically slower than it should be.

The problem
-----------
When we write ``A @ B`` or ``jnp.einsum("ij,jk->ik", A, B)``, XLA recognises the
shape of a matrix product and hands the work to the machinery built for matrix
products: on GPU, a batched GEMM kernel. That machinery is great for large
matrices as it tiles them, stages tiles into fast on-chip memory, and keeps the
arithmetic units saturated. All of that setup has a fixed cost per matrix.

Our matrices are 2x2. We pay the setup cost millions of times to do almost no
arithmetic. Worse, a GEMM is a hard boundary for the compiler: it cannot merge
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
An einsum turns into a real matrix product only when all three of these

1. an index that appears only in the first operand,
2. an index that appears only in the second operand,
3. an index that is summed over.

``"ij,jk->ik"`` has all three and is a matmul. ``"ij,ij->i"`` has none o
first two -- it is a multiply and a sum, and is already fine. ``"ij,kl->
has no summed index -- it is an outer product, also fine.

Becoming a matmul is not automatically bad. One large matrix product is
what the GEMM path is for. It is bad only when the batch is huge and each
element is tiny, which is precisely our situation in tatva.

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
Reverting to ``@``, ``jnp.einsum`` or ``jnp.linalg.inv`` leaves every test
passing and the results correct to the last digit. The only thing that changes
is that the code becomes ~30x slower, silently. ``tests/test_tensor_kern
guards against this by asserting that the compiled code contains no ``dot`` and
no cuBLAS call.

When not to use these
---------------------
These routines win *because* the arrays are small. ``matmul`` builds a
(p, q, r) intermediate before reducing it, which is free when p, q, r <=
ruinous when a dimension is in the hundreds. If a contracted dimension can grow
large, use ``@`` and let XLA call the GEMM -- that is the case it was bu
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


def matmul(A: Array, B: Array) -> Array:
    """(p, q) @ (q, ...) -> (p, ...) without the GEMM path."""

    Ae = A.reshape(A.shape + (1,) * (B.ndim - 1))
    return jnp.sum(Ae * B[None, ...], axis=1)
