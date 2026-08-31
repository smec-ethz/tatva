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

One exception is not a simplification but a correctness change: ``linalg.tensordot`` is
``jnp.tensordot(A, B, axes=1)``, *not* ``jnp.matmul``. The two disagree once ``B``
is rank 3 or more, silently and with no error.

Main entry point: ``linalg.contract``
---------------------------------
The entry point is ``linalg.contract(spec, A, B)``, and the spec is an einsum-style
string naming the contraction::

    J    = linalg.contract("in,nj->ij",    dNdr,        nodal_coords)
    dNdX = linalg.contract("ij,jn->in",    linalg.inv(J),   dNdr)
    grad = linalg.contract("in,n...->i...", dNdX,       nodal_values)
    val  = linalg.contract("n,n...->...",  N,           nodal_values)

The spec is a *name*, not an instruction. It is never handed to ``jnp.einsum`` --
every one of these sums over a shared index, so einsum would lower all of them to a
``dot``, which is the thing this module exists to avoid. It is resolved at trace
time against ``_known_implementations``, a small table holding how each contraction
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

from collections.abc import Callable
from functools import partial
from typing import Any

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
            f"tatva.linalg.inv takes a single matrix, got shape {J.shape}. Batch with jax.vmap "
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


def det(J: Array) -> Array:
    """Closed-form determinant of a 1x1, 2x2 or 3x3 matrix.

    Since JAX 0.11.1, `jnp.linalg.det` computes small determinants with a pivoted
    formula wrapped in `@custom_jvp`. The `custom_jvp` stops XLA fusing into its
    neighbours.

    In `float32+, pivoting is the more stable choice therefore use `jnp.linalg.det`
    then.

    Args:
        J: a single matrix, shape (d, d) with d in {1, 2, 3}.

    Returns:
        The determinant of J, a scalar.

    """
    if J.ndim != 2:
        raise ValueError(
            f"tatva.linalg.det takes a single matrix, got shape {J.shape}. Batch with jax.vmap "
            "rather than passing a stack"
        )

    d = J.shape[-1]
    if J.shape[-2] != d:
        return jnp.linalg.det(J)  # not square: let JAX raise as before
    if d == 1:
        return J[0, 0]
    if d == 2:
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if d == 3:
        m = J
        return (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            + m[0, 1] * (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        )
    return jnp.linalg.det(J)


# --------------------------------------------------------------------------------
# contract is a drop-in replacement for einsum with a fixed set of contractions
# --------------------------------------------------------------------------------


def _parse_einsum_subscripts(
    subscripts: str, operands: tuple[Array, ...]
) -> tuple[list[list[str]], list[str]]:
    """Parse string-form einsum subscripts into physical axis labels.

    Each ellipsis is expanded according to the corresponding operand rank. Ellipsis
    axes are right-aligned across operands, following NumPy/JAX broadcasting rules,
    and represented by synthetic labels such as ``"@0"`` that cannot collide with
    user labels. For an implicit-output spec, broadcast axes are emitted first and
    labels that occur exactly once follow in alphabetical order.

    Args:
        subscripts: Explicit or implicit string-form einsum specification, such as
            ``"...ij,...jk->...ik"`` or ``"ij,jk"``.
        operands: Operands used to determine how many physical axes each ellipsis
            represents and to validate each input term's rank.

    Returns:
        A pair ``(input_labels, output_labels)``. ``input_labels`` contains one list
        per operand and exactly one label per operand axis. ``output_labels`` gives
        the result-axis order. Neither result contains an unexpanded ellipsis.

    Raises:
        ValueError: If the number of input terms does not match the operands, a term
            does not match its operand rank, a label or ellipsis is malformed, or
            the output contains duplicate or unknown labels.
    """
    cleaned = subscripts.replace(" ", "")
    if cleaned.count("->") > 1:
        raise ValueError(f"invalid einsum subscripts {subscripts!r}")

    if "->" in cleaned:
        inputs, output = cleaned.split("->")
    else:
        inputs, output = cleaned, None

    input_terms = inputs.split(",")
    if len(input_terms) != len(operands):
        raise ValueError(
            f"einsum subscripts contain {len(input_terms)} input terms, but "
            f"{len(operands)} operands were passed"
        )

    tokenized = [_tokenize(term) for term in input_terms]
    ellipsis_ranks: list[int] = []
    for tokens, operand in zip(tokenized, operands):
        if tokens.count("...") > 1:
            raise ValueError("an einsum input term may contain at most one ellipsis")
        for label in tokens:
            if label != "..." and (
                len(label) != 1 or not label.isascii() or not label.isalpha()
            ):
                raise ValueError(f"invalid einsum label {label!r}")
        explicit_rank = len(tokens) - tokens.count("...")
        ellipsis_rank = operand.ndim - explicit_rank
        if ellipsis_rank < 0 or ("..." not in tokens and ellipsis_rank != 0):
            raise ValueError(
                f"einsum term {''.join(tokens)!r} does not match operand shape {operand.shape}"
            )
        ellipsis_ranks.append(ellipsis_rank)

    max_ellipsis_rank = max(ellipsis_ranks, default=0)
    # These cannot collide with user labels, which are restricted to ASCII letters.
    ellipsis_labels = [f"@{i}" for i in range(max_ellipsis_rank)]
    expanded_inputs: list[list[str]] = []
    for tokens, ellipsis_rank in zip(tokenized, ellipsis_ranks):
        expanded: list[str] = []
        for token in tokens:
            if token == "...":
                expanded.extend(ellipsis_labels[max_ellipsis_rank - ellipsis_rank :])
            else:
                expanded.append(token)
        expanded_inputs.append(expanded)

    counts: dict[str, int] = {}
    for labels in expanded_inputs:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    if output is None:
        # NumPy/JAX put broadcast axes first, then labels occurring exactly once in
        # alphabetical order for the implicit-output form.
        output_labels = ellipsis_labels + sorted(
            label
            for label, count in counts.items()
            if count == 1 and not label.startswith("@")
        )
    else:
        output_tokens = _tokenize(output)
        if output_tokens.count("...") > 1:
            raise ValueError("an einsum output term may contain at most one ellipsis")
        output_labels = []
        for token in output_tokens:
            if token == "...":
                output_labels.extend(ellipsis_labels)
            elif len(token) == 1 and token.isascii() and token.isalpha():
                output_labels.append(token)
            else:
                raise ValueError(f"invalid einsum label {token!r}")

    if len(set(output_labels)) != len(output_labels):
        raise ValueError("einsum output labels must be unique")
    missing = [label for label in output_labels if label not in counts]
    if missing:
        raise ValueError(
            f"einsum output label {missing[0]!r} does not appear in an input"
        )
    return expanded_inputs, output_labels


def _diagonalize_repeated_axes(
    operand: Array, labels: list[str]
) -> tuple[Array, list[str]]:
    """Take diagonals for labels repeated within a single einsum operand.

    A repeated label denotes equal coordinates on its axes: for example, ``"ii"``
    selects a matrix diagonal rather than treating the two ``i`` axes independently.
    Each pair of repeated axes is collapsed with ``jnp.diagonal`` until every label
    is unique. Since ``jnp.diagonal`` appends its diagonal axis, the returned labels
    are reordered in the same way as the returned operand.

    Args:
        operand: Operand whose rank matches the number of labels.
        labels: Expanded axis labels for ``operand``. The input list is not mutated.

    Returns:
        The diagonalized operand and its updated, unique axis-label list.

    Raises:
        ValueError: If two axes carrying the same label have different sizes.
    """
    labels = list(labels)
    while len(set(labels)) != len(labels):
        for axis1, label in enumerate(labels):
            try:
                axis2 = labels.index(label, axis1 + 1)
            except ValueError:
                continue
            if operand.shape[axis1] != operand.shape[axis2]:
                raise ValueError(
                    f"dimensions for repeated einsum label {label!r} must match, got "
                    f"{operand.shape[axis1]} and {operand.shape[axis2]}"
                )
            operand = jnp.diagonal(operand, axis1=axis1, axis2=axis2)
            labels = [
                current
                for axis, current in enumerate(labels)
                if axis not in (axis1, axis2)
            ] + [label]
            break
    return operand, labels


def _einsum_broadcast(
    subscripts: str, *operands: Any, preferred_element_type: Any | None = None
) -> Array:
    """General einsum implemented as aligned broadcasting, multiplication and sum.

    This supports explicit and implicit string specs, ellipses, diagonals, output
    permutations, and any number of operands.  It deliberately does not call
    ``jnp.einsum`` or ``lax.dot_general``, so small contractions remain ordinary,
    fusible elementwise/reduction work on GPU.

    The tradeoff is the same as for any outer-product spelling: the conceptual
    intermediate contains every free and contracted axis.  Use this for tiny tensor
    contractions, not large matrices, and let ``jax.vmap`` introduce batch axes.

    ``precision`` and contraction-path options are intentionally absent because there
    is no dot or contraction path to configure. ``preferred_element_type`` casts the
    arithmetic and accumulation to the requested dtype.
    """
    if not isinstance(subscripts, str):
        raise TypeError("einsum_broadcast supports the string subscript form only")
    if not operands:
        raise ValueError("einsum_broadcast needs at least one operand")

    arrays = tuple(jnp.asarray(operand) for operand in operands)
    input_labels, output_labels = _parse_einsum_subscripts(subscripts, arrays)

    diagonalized = [
        _diagonalize_repeated_axes(operand, labels)
        for operand, labels in zip(arrays, input_labels)
    ]
    arrays = tuple(item[0] for item in diagonalized)
    input_labels = [item[1] for item in diagonalized]

    reduction_labels: list[str] = []
    for labels in input_labels:
        for label in labels:
            if label not in output_labels and label not in reduction_labels:
                reduction_labels.append(label)
    axis_labels = output_labels + reduction_labels

    aligned: list[Array] = []
    for operand, labels in zip(arrays, input_labels):
        present = [label for label in axis_labels if label in labels]
        permutation = tuple(labels.index(label) for label in present)
        if permutation != tuple(range(operand.ndim)):
            operand = jnp.transpose(operand, permutation)
        shape = tuple(
            operand.shape[present.index(label)] if label in present else 1
            for label in axis_labels
        )
        aligned.append(jnp.reshape(operand, shape))

    if preferred_element_type is not None:
        aligned = [operand.astype(preferred_element_type) for operand in aligned]

    product = aligned[0]
    for operand in aligned[1:]:
        product = product * operand
    if reduction_labels:
        product = jnp.sum(
            product,
            axis=tuple(range(len(output_labels), len(axis_labels))),
            dtype=product.dtype,
        )
    return product


def _dot_broadcast(A: Array, B: Array) -> Array:
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
            f"tatva.linalg._dot_broadcast takes a single vector or matrix as A, got shape {A.shape}. "
            "Batch with jax.vmap rather than passing a stack: this builds the full "
            "outer product before reducing, so a batch axis materialises it in full."
        )

    spec = "a,a...->..." if A.ndim == 1 else "ab,b...->a..."
    return _einsum_broadcast(spec, A, B)


_einsum = jnp.einsum

_POLICIES: dict[str, tuple[Callable, Callable]] = {
    # (p, q) x (q, r): the element Jacobian and the product after it
    "ab,bc->ac": (_einsum, _einsum_broadcast),
    # (q,) x (q, r): the Line2/Line3 tangent vector
    "a,ab->b": (_einsum, _einsum),
    # (p, q) x (q, ...): dNdX against nodal values
    "ab,b...->a...": (_einsum_broadcast, _einsum_broadcast),
    # (q,) x (q, ...): N against nodal values
    "a,a...->...": (_einsum_broadcast, _einsum_broadcast),
    # (..., i, r) x (j, r) -> (..., i, j)
    "...ab,cb->...ac": (_einsum, _einsum_broadcast),
}

# Compatibility for code that only inspects whether a canonical spec is registered.
_known_implementations = _POLICIES


def _tokenize(term: str) -> list[str]:
    """Split a spec term into index letters, with ``...`` as a single token.
    For example, ``"in,n...->i..."`` is split into  ["i", "n", "n", "...", "->", "i", "..."].

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
    """Rename index letters in order of first appearance i.e. the first letter encountered
    is renamed to 'a', the second to 'b', etc.

    So a call may use letters that mean something -- ``"in,nj->ij"`` for the
    Jacobian, ``"in,n...->i..."`` for the node contraction -- while the lookup table
    holds one entry per *contraction*, not one per spelling.
    """
    cleaned = spec.replace(" ", "")
    if cleaned.count("->") != 1 or cleaned.count(",") != 1:
        raise ValueError(
            f"tatva.linalg.contract could not parse the spec {spec!r}. Expected exactly one ',' "
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
                    f"tatva.linalg.contract could not parse the spec {spec!r}: {token!r} is not an "
                    "index letter."
                )
            if token not in renaming:
                renaming[token] = chr(ord("a") + len(renaming))  # assign a new letter
            letters.append(renaming[token])
        return "".join(letters)

    return f"{rename(a_term)},{rename(b_term)}->{rename(out)}"


def contract(spec: str, A: Array, B: Array) -> Array:
    """Contract two element-sized tensors, named by an einsum-style spec.

        J    = linalg.contract("in,nj->ij", dNdr, nodal_coords)     # (d, n) x (n, d)
        dNdX = linalg.contract("ij,jn->in", linalg.inv(J), dNdr)        # (d, d) x (d, n)
        grad = linalg.contract("in,n...->i...", dNdX, nodal_values)  # over the node axis
        val  = linalg.contract("n,n...->...", N, nodal_values)      # over the node axis

    Index letters are free. They are canonicalised by order of first appearance, so
    ``"in,nj->ij"`` and ``"pq,qr->pr"`` are the same entry, and a call site may use
    letters that carry meaning. An unrecognised contraction is refused rather than
    given a general fallback: a new spec needs a measured decision about how to spell
    it, which is exactly what this function exists to record.

    Args:
        spec: einsum-style contraction, e.g. ``"in,nj->ij"``. Must name one of the
            supported contractions; letters are arbitrary.
        A: first operand, a single tensor.
        B: second operand, a single tensor.

    Returns:
        The contraction, with axes in the order the spec's output term gives.

    Raises:
        ValueError: if the spec is unparseable, names a contraction with no measured
            implementation, or disagrees with the operand shapes.
    """
    # convert the spec to canonical form, to match keys in the known implementations
    canonical = _canonicalise(spec)
    # look up the implementation for the canonical spec
    policy = _POLICIES.get(canonical)

    if policy is None:
        supported = ", ".join(repr(s) for s in _POLICIES)
        raise ValueError(
            f"tatva.linalg.contract has no implementation policy for {spec!r} "
            f"(canonically {canonical!r})."
            f" Supported contractions are {supported}. This is deliberate rather than a "
            "gap: each entry is a measured decision about how to spell the contraction "
            "so it does not lower to a `dot`. Add an entry, with the measurement, rather "
            "than reaching for jnp.einsum -- see the module docstring."
        )

    operands = (jnp.asarray(A), jnp.asarray(B))
    try:
        input_labels, _ = _parse_einsum_subscripts(canonical, operands)
    except ValueError as error:
        raise ValueError(
            f"tatva.linalg.contract({spec!r}, ...) received operand ranks "
            f"{(A.ndim, B.ndim)} from shapes {A.shape} and {B.shape}, which do not "
            "match the registered spec. Batch with jax.vmap rather than adding an "
            "unregistered batch axis."
        ) from error

    for label in set(input_labels[0]).intersection(input_labels[1]):
        a_axis = input_labels[0].index(label)
        b_axis = input_labels[1].index(label)
        a_size, b_size = A.shape[a_axis], B.shape[b_axis]
        if a_size == b_size or a_size == 1 or b_size == 1:
            continue
        a_axis_name = (
            "first"
            if a_axis == 0
            else "last"
            if a_axis == A.ndim - 1
            else f"axis {a_axis}"
        )
        b_axis_name = (
            "first"
            if b_axis == 0
            else "last"
            if b_axis == B.ndim - 1
            else f"axis {b_axis}"
        )
        raise ValueError(
            f"tatva.linalg.contract({spec!r}, ...) contracts A's {a_axis_name} axis "
            f"with B's {b_axis_name} axis using label {label!r}, but they are "
            f"{a_size} and {b_size} from shapes {A.shape} and {B.shape}."
        )

    on_cpu, on_default = policy

    if on_cpu is on_default:
        return on_cpu(canonical, A, B)

    return jax.lax.platform_dependent(
        A,
        B,
        cpu=partial(on_cpu, canonical),
        default=partial(on_default, canonical),
    )


def einsum(spec: Any, *operands: Any, **kwargs: Any) -> Array:
    """
    Drop-in replacement for jnp.einsum that uses a custom implementation if available.
    This is necessary for internal optimization based on the hardware backend. Check `tatva.linalg.contract`
    for more details. If the custom implementation is not available, falls back to jnp.einsum.

    Only a two-operand spec naming one of the measured contractions takes the `contract`
    path; three or more operands, an implicit output, the interleaved calling convention
    and an unmeasured contraction all go to jnp.einsum. The `contract` path ignores the
    keyword arguments such as ``precision``, `preferred_element_type``  so call jnp.einsum
    directly when one of those has to hold.

    Args:
        spec: einsum-style subscripts, e.g. ``"in,nj->ij"``.
        operands: the tensors to contract.
        kwargs: forwarded verbatim to jnp.einsum on every path that reaches it.

    Returns:
        The contraction.
    """
    # the interleaved form, einsum(A, [0, 1], B, [1, 2], [0, 2]), passes an array first
    # and names no contraction to look up
    if not isinstance(spec, str):
        return jnp.einsum(spec, *operands, **kwargs)

    try:
        # convert the spec to canonical form, to match keys in the known implementations
        canonical = _canonicalise(spec)
    except ValueError:
        # a spec `contract` cannot name: implicit output, a single operand, or more than
        # two. There is nothing to look up, so fall back to jnp.einsum
        return jnp.einsum(spec, *operands, **kwargs)

    # look up the implementation for the canonical spec
    if canonical not in _POLICIES or len(operands) != 2:
        return jnp.einsum(spec, *operands, **kwargs)

    return contract(spec, *operands)
