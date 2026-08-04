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

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Jaxpr, Literal, Var

# FFI targets whose vmapped (batched) call is elementwise along the leading (vmap) axis,
# i.e. output[i] depends only on input[i] -- e.g. a per-quad-point external constitutive
# solver vmapped over quad points. Declaring a target here lets the tracer recover the
# sparse (block-diagonal) coupling from the single batched ``ffi_call`` instead of the
# conservative dense couple-all. See ``register_elementwise_ffi``.
_ELEMENTWISE_FFI_TARGETS: set[str] = set()


def register_elementwise_ffi(*target_names: str) -> None:
    """Declare one or more ``jax.ffi`` target names as elementwise along the vmap axis.

    A vmapped ``ffi_call`` is a *single* batched custom-call over the whole leading axis,
    so by default the tracer must treat it as a dense opaque block (it cannot tell the
    batch axis is independent). Registering the target tells the tracer that the call is a
    per-element map (output[i] depends only on input[i]) -- as it is for a per-quad-point
    external solver vmapped over quad points -- so it records conservative coupling *per
    slice* (block-diagonal across the leading axis), recovering the sparse pattern.

    Register by the FFI target name (the string passed to ``jax.ffi.ffi_call``) -- anyone
    building or using a ``jax.ffi`` solver knows this name. Then a per-quad-point external
    solver vmapped over quad points, ``jax.vmap(solver)(strain_field)``, traces sparse.

    Only register a target if this independence genuinely holds; otherwise real
    cross-element couplings would be silently missed.
    """
    _ELEMENTWISE_FFI_TARGETS.update(target_names)


def _get_shape(var: Var | Literal) -> tuple[int, ...]:
    """Helper to safely retrieve shape from a JAX Var/Literal abstract value (satisfying
    static type checkers)."""
    return getattr(var.aval, "shape", ())


def _subjaxpr_and_consts(eqn) -> tuple[Jaxpr, Sequence]:
    """Normalize the ``jaxpr`` param of a single-subgraph higher-order primitive.

    ``pjit``/``jit``/``scan``/``map`` store a ``ClosedJaxpr`` (``.jaxpr`` + ``.consts``),
    whereas ``remat2`` (``jax.checkpoint``) stores a bare ``Jaxpr`` with no consts.
    Both expose the same 1:1 invar/outvar correspondence with the parent equation, so
    this returns ``(jaxpr, consts)`` for either form.
    """
    sub = eqn.params["jaxpr"]
    if hasattr(sub, "jaxpr"):  # ClosedJaxpr
        return sub.jaxpr, sub.consts
    return sub, ()


def _unwrap_jit(fn):
    """Recursively unwrap `@jax.jit` / `@pjit` decorators only.

    `jax.grad`, `jax.vmap`, and `functools.wraps`-based wrappers also set `__wrapped__`,
    so blindly removing `__wrapped__` would strip semantic transforms (e.g., turning
    `jax.grad(E)` back into `E`). We only unwrap `PjitFunction`-class wrappers, which
    @jax.jit` produces — this preserves any `grad`/`vmap` layers underneath.
    """
    while type(fn).__name__ in ("PjitFunction", "JitWrapped") and hasattr(
        fn, "__wrapped__"
    ):
        fn = fn.__wrapped__
    return fn


def _broadcast_single_row(row: sps.csr_matrix, N: int) -> sps.csr_matrix:
    """Replicate a single-row CSR matrix N times, ~250x faster than sps.vstack([row]*N).

    Builds the result directly via a uniform indptr (each row has the same nnz),
    bypassing the O(N) Python overhead of sps.vstack.
    """
    if N <= 0:
        return sps.csr_matrix((0, row.shape[1]), dtype=row.dtype)
    m = row.nnz
    if m == 0:
        return sps.csr_matrix((N, row.shape[1]), dtype=row.dtype)
    indptr = np.arange(N + 1, dtype=row.indptr.dtype) * m
    indices = np.tile(row.indices, N)
    data = np.tile(row.data, N)
    return sps.csr_matrix((data, indices, indptr), shape=(N, row.shape[1]))


def _reduce_union_over_axes(
    dep: sps.csr_matrix, shape: tuple[int, ...], keep_axes: list[int]
) -> sps.csr_matrix:
    """OR-reduce a dep-array over all axes NOT in ``keep_axes``.

    ``dep`` is a (prod(shape), n_dofs) CSR whose rows are the row-major flattening of a
    tensor of logical ``shape``. Returns a (prod(shape[keep_axes]), n_dofs) CSR whose rows
    are the row-major flattening over ``keep_axes`` (in the given order), where each kept
    row is the union of the dependency sets of every original element that maps to it.

    This is the exact support map for a reduction (a summed-out axis makes the result
    depend on the union along it), so it only ever preserves or tightens support -- never
    drops a real dependency.
    """
    n_rows = dep.shape[0]
    if not keep_axes:
        # Reducing over everything -> single-row total union.
        return sps.csr_matrix(dep.sum(axis=0).astype(bool)) if n_rows else dep
    orig = np.arange(n_rows)
    multi = np.unravel_index(orig, shape) if shape else (np.zeros(n_rows, int),)
    keep_dims = tuple(shape[a] for a in keep_axes)
    key = np.ravel_multi_index(tuple(multi[a] for a in keep_axes), keep_dims)
    n_keys = int(np.prod(keep_dims))
    # Boolean aggregation matrix (n_keys x n_rows); (agg @ dep) unions the rows per key.
    agg = sps.csr_matrix(
        (np.ones(n_rows, dtype=np.int8), (key, orig)), shape=(n_keys, n_rows)
    )
    return (agg @ dep.astype(np.int8)).astype(bool).tocsr()
