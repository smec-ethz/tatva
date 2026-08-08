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
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tatva.sparse.tracer.partitioning import RowSet
    from tatva.sparse.tracer.state import SparseDepSet

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


def _dot_general_out_dep(
    lhs: SparseDepSet,
    rhs: SparseDepSet,
    lhs_c: Sequence[int],
    rhs_c: Sequence[int],
    lhs_b: Sequence[int],
    rhs_b: Sequence[int],
    oshp: tuple[int, ...],
    n_dofs: int,
) -> SparseDepSet:
    """Structure-preserving support propagation for a ``dot_general``.

    Output element ``out[b, fl, fr]`` (batch ``b``, lhs-free ``fl``, rhs-free ``fr``)
    depends on exactly ``(union_c lhs[b, fl, c]) | (union_c rhs[b, c, fr])``. We reduce
    each operand over its contracting axes, then broadcast the two reduced dep-tensors to
    the output layout ``[batch..., lhs_free..., rhs_free...]`` (JAX's ``dot_general`` output
    order) and union them. This keeps per-row (per-element) support instead of collapsing
    to the whole-array ``total_union`` -- the difference between a locally-supported field
    and one that appears to depend on every DOF.
    """
    from tatva.sparse.tracer.state import SparseDepSet

    La, Lb = lhs.shape, rhs.shape
    lhs_b, rhs_b = list(lhs_b), list(rhs_b)
    lhs_c, rhs_c = list(lhs_c), list(rhs_c)
    lhs_free = [a for a in range(len(La)) if a not in lhs_b and a not in lhs_c]
    rhs_free = [a for a in range(len(Lb)) if a not in rhs_b and a not in rhs_c]

    # Reduce out contracting axes, keeping (batch, free) in output order.
    lhs_red = _reduce_union_over_axes(lhs.dep, La, lhs_b + lhs_free)
    rhs_red = _reduce_union_over_axes(rhs.dep, Lb, rhs_b + rhs_free)

    batch_sizes = tuple(La[a] for a in lhs_b)
    lhs_free_sizes = tuple(La[a] for a in lhs_free)
    rhs_free_sizes = tuple(Lb[a] for a in rhs_free)
    out_dims = batch_sizes + lhs_free_sizes + rhs_free_sizes

    n_out = int(np.prod(oshp))
    if int(np.prod(out_dims)) != n_out:
        # Shape bookkeeping disagrees with the reported output shape; fall back to the
        # conservative whole-array union rather than risk a mismatch.
        combined = sps.csr_matrix((lhs.dep + rhs.dep).astype(bool))
        return SparseDepSet(
            _broadcast_single_row(
                sps.csr_matrix(combined.sum(axis=0).astype(bool)), n_out
            ),
            oshp,
        )

    nb, nlf = len(lhs_b), len(lhs_free)
    omulti = np.unravel_index(np.arange(n_out), out_dims) if out_dims else ()
    batch_m = omulti[:nb]
    lf_m = omulti[nb : nb + nlf]
    rf_m = omulti[nb + nlf :]

    lhs_keys = (
        np.ravel_multi_index(batch_m + lf_m, batch_sizes + lhs_free_sizes)
        if (batch_sizes + lhs_free_sizes)
        else np.zeros(n_out, int)
    )
    rhs_keys = (
        np.ravel_multi_index(batch_m + rf_m, batch_sizes + rhs_free_sizes)
        if (batch_sizes + rhs_free_sizes)
        else np.zeros(n_out, int)
    )
    out_dep = (
        (lhs_red[lhs_keys].astype(np.int8) + rhs_red[rhs_keys].astype(np.int8))
        .astype(bool)
        .tocsr()
    )
    return SparseDepSet(out_dep, oshp)


def _inverse_elementwise_rows(
    rows: NDArray[np.integer] | RowSet, in_shape: tuple, out_shape: tuple
) -> NDArray[np.int64]:
    """Map broadcasted output flat rows to the corresponding input flat rows."""
    if not in_shape:
        return np.zeros(len(rows), dtype=np.int64)
    out_coords = np.unravel_index(rows, out_shape)
    lead = len(out_shape) - len(in_shape)
    in_coords = []
    for axis, size in enumerate(in_shape):
        coord = out_coords[lead + axis]
        in_coords.append(np.zeros_like(coord) if size == 1 else coord)
    return np.ravel_multi_index(tuple(in_coords), in_shape).astype(np.int64)


def _inverse_broadcast_rows(
    rows: NDArray[np.integer] | RowSet,
    in_shape: tuple,
    out_shape: tuple,
    dimensions: Sequence[int],
) -> NDArray[np.int64]:
    if not in_shape:
        return np.zeros(len(rows), dtype=np.int64)
    out_coords = np.unravel_index(rows, out_shape)
    in_coords = [
        np.zeros_like(out_coords[axis]) if size == 1 else out_coords[axis]
        for size, axis in zip(in_shape, dimensions)
    ]
    return np.ravel_multi_index(tuple(in_coords), in_shape).astype(np.int64)


# NOTE: this function is unchecked code coming directly from GPT-5.6 Sol
def gather_routes(
    eqn: JaxprEqn,
    indices: NDArray,
    demanded_output_rows: NDArray | None,
) -> tuple[NDArray, NDArray] | None:
    """Map gather output entries to operand and index-array entries.

    Args:
        eqn:
            A JAX ``gather`` equation.
        indices:
            Concrete value of the gather index tensor.
        demanded_output_rows:
            Flat C-order output rows to route. If ``None``, route every
            output entry.

    Returns:
        ``(source_rows, index_rows)`` where:

        - ``source_rows[i]`` is the flat operand row used by
          ``demanded_output_rows[i]``. When all outputs are requested, it
          corresponds to output row ``i``.
        - A source row of ``-1`` means the output is filled because of an
          out-of-bounds ``FILL_OR_DROP`` gather.
        - ``index_rows`` contains the unique flattened index-tensor entries
          needed to compute the requested outputs.

        Returns ``None`` when the gather configuration cannot be routed
        exactly. Callers must then use a conservative fallback.
    """
    if len(eqn.invars) < 2 or not eqn.outvars:
        return None

    operand_var, indices_var = eqn.invars[:2]
    output_var = eqn.outvars[0]

    operand_shape = tuple(_get_shape(operand_var))
    indices_shape = tuple(_get_shape(indices_var))
    output_shape = tuple(_get_shape(output_var))

    indices = np.asarray(indices)

    if indices.ndim < 1 or tuple(indices.shape) != indices_shape:
        return None

    try:
        dnums = eqn.params["dimension_numbers"]
        slice_sizes = tuple(int(s) for s in eqn.params["slice_sizes"])

        offset_dims = tuple(int(d) for d in dnums.offset_dims)
        collapsed_dims = tuple(int(d) for d in dnums.collapsed_slice_dims)
        start_index_map = tuple(int(d) for d in dnums.start_index_map)
        operand_batching_dims = tuple(
            int(d) for d in getattr(dnums, "operand_batching_dims", ())
        )
        indices_batching_dims = tuple(
            int(d) for d in getattr(dnums, "start_indices_batching_dims", ())
        )
    except (KeyError, TypeError, AttributeError):
        return None

    operand_rank = len(operand_shape)
    indices_rank = len(indices_shape)
    output_rank = len(output_shape)

    if len(slice_sizes) != operand_rank:
        return None

    # In JAX, the index-vector dimension is always the last dimension.
    index_vector_size = indices_shape[-1]

    if index_vector_size != len(start_index_map):
        return None

    if len(offset_dims) + indices_rank - 1 != output_rank:
        return None

    if len(operand_batching_dims) != len(indices_batching_dims):
        return None

    if any(
        axis < 0 or axis >= operand_rank
        for axis in (
            *collapsed_dims,
            *start_index_map,
            *operand_batching_dims,
        )
    ):
        return None

    if any(axis < 0 or axis >= indices_rank - 1 for axis in indices_batching_dims):
        return None

    n_output = int(np.prod(output_shape, dtype=np.int64))

    if demanded_output_rows is None:
        output_rows = np.arange(n_output, dtype=np.int64)
    else:
        output_rows = np.asarray(demanded_output_rows, dtype=np.int64).ravel()

        if np.any(output_rows < 0) or np.any(output_rows >= n_output):
            return None

    if output_rows.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # Coordinates of each demanded output element.
    if output_shape:
        output_coords = np.stack(
            np.unravel_index(output_rows, output_shape),
            axis=1,
        ).astype(np.int64)
    else:
        output_coords = np.empty((output_rows.size, 0), dtype=np.int64)

    # Output dimensions not listed in offset_dims correspond, in order, to
    # indices.shape[:-1].
    offset_dim_set = set(offset_dims)
    output_batch_dims = tuple(
        axis for axis in range(output_rank) if axis not in offset_dim_set
    )

    if len(output_batch_dims) != indices_rank - 1:
        return None

    if output_batch_dims:
        indices_batch_coords = output_coords[:, output_batch_dims]
    else:
        indices_batch_coords = np.empty((output_rows.size, 0), dtype=np.int64)

    # Read the complete index vector associated with each demanded output.
    if index_vector_size:
        if indices_rank == 1:
            index_vectors = np.broadcast_to(
                indices.reshape(1, index_vector_size),
                (output_rows.size, index_vector_size),
            )
        else:
            index_key = tuple(
                indices_batch_coords[:, axis] for axis in range(indices_rank - 1)
            )
            index_vectors = np.asarray(indices[index_key]).reshape(
                output_rows.size, index_vector_size
            )

        index_vectors = np.asarray(index_vectors, dtype=np.int64)

        # Every component of the selected index vectors is numerically needed.
        repeated_batch_coords = np.repeat(
            indices_batch_coords, index_vector_size, axis=0
        )
        vector_components = np.tile(
            np.arange(index_vector_size, dtype=np.int64), output_rows.size
        )

        index_coords = tuple(
            repeated_batch_coords[:, axis] for axis in range(indices_rank - 1)
        ) + (vector_components,)

        index_rows = np.ravel_multi_index(index_coords, indices_shape).astype(np.int64)
    else:
        index_vectors = np.empty((output_rows.size, 0), dtype=np.int64)
        index_rows = np.empty(0, dtype=np.int64)

    # Build the starting operand coordinate for each output.
    starts = np.zeros((output_rows.size, operand_rank), dtype=np.int64)

    for component, operand_axis in enumerate(start_index_map):
        starts[:, operand_axis] = index_vectors[:, component]

    # Batching dimensions are not part of start_index_map. Their operand
    # coordinate comes directly from the corresponding index batch coordinate.
    for operand_axis, indices_axis in zip(operand_batching_dims, indices_batching_dims):
        starts[:, operand_axis] = indices_batch_coords[:, indices_axis]

    collapsed_set = set(collapsed_dims)
    operand_batching_set = set(operand_batching_dims)

    # The remaining operand axes correspond, in operand-axis order, to the
    # gather output's offset dimensions.
    window_operand_dims = tuple(
        axis
        for axis in range(operand_rank)
        if axis not in collapsed_set and axis not in operand_batching_set
    )

    if len(window_operand_dims) != len(offset_dims):
        return None

    offsets = np.zeros_like(starts)

    for output_axis, operand_axis in zip(offset_dims, window_operand_dims):
        offsets[:, operand_axis] = output_coords[:, output_axis]

    upper_starts = np.asarray(operand_shape, dtype=np.int64) - np.asarray(
        slice_sizes, dtype=np.int64
    )

    if np.any(upper_starts < 0):
        return None

    mode = eqn.params.get("mode")

    if mode is None:
        mode_name = "PROMISE_IN_BOUNDS"
    else:
        mode_name = getattr(mode, "name", str(mode))
        mode_name = mode_name.rsplit(".", 1)[-1].upper()

    if mode_name in {"FILL", "DROP"}:
        mode_name = "FILL_OR_DROP"

    valid = np.ones(output_rows.size, dtype=bool)

    if mode_name == "CLIP":
        starts = np.minimum(np.maximum(starts, 0), upper_starts)

    elif mode_name == "FILL_OR_DROP":
        # JAX considers the entire gathered slice invalid if any component
        # of its start-index vector is outside its valid start range.
        for component, operand_axis in enumerate(start_index_map):
            values = index_vectors[:, component]
            valid &= values >= 0
            valid &= values <= upper_starts[operand_axis]

    elif mode_name == "PROMISE_IN_BOUNDS":
        # 07.08.26 florez: current JAX clips out-of-bounds indices in PROMISE_IN_BOUNDS mode, so
        # we will do the same!
        starts = np.minimum(np.maximum(starts, 0), upper_starts)
        # The program promises these are valid. If the concrete values violate
        # that promise, fall back conservatively rather than routing incorrectly.
        # for component, operand_axis in enumerate(start_index_map):
        #     values = index_vectors[:, component]
        #     if np.any(values < 0) or np.any(values > upper_starts[operand_axis]):
        #         return None

    else:
        # For example ONE_HOT, which has different semantics.
        # TODO: implement ONE_HOT routing if needed.
        return None

    source_coords = starts + offsets
    source_rows = np.full(output_rows.size, -1, dtype=np.int64)

    if np.any(valid):
        valid_coords = source_coords[valid]
        operand_bounds = np.asarray(operand_shape, dtype=np.int64)

        if np.any(valid_coords < 0) or np.any(valid_coords >= operand_bounds):
            return None

        source_rows[valid] = np.ravel_multi_index(
            tuple(valid_coords.T), operand_shape
        ).astype(np.int64)

    return source_rows, np.unique(index_rows).astype(np.int64)


def scatter_routes(
    eqn: JaxprEqn,
    indices: NDArray,
    update_rows: NDArray | None = None,
    *,
    include_index_rows: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None] | None:
    """Map update entries to operand entries for a JAX scatter.

    ``target_rows`` is aligned with ``update_rows`` (or every flattened update
    entry when omitted); invalid, dropped routes are represented by ``-1``.
    The second result is the canonical set of flattened index-tensor entries
    needed for those routes.  Forward support propagation only needs targets,
    so it can disable index-row construction (and its costly canonicalization).
    This is deliberately shared by forward support propagation and reverse
    liveness so ScatterDimensionNumbers have one interpretation in the tracer.
    """
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None
    operand_shape = tuple(_get_shape(eqn.invars[0]))
    indices_shape = tuple(_get_shape(eqn.invars[1]))
    updates_shape = tuple(_get_shape(eqn.invars[2]))
    indices = np.asarray(indices)

    if indices.ndim < 1 or tuple(indices.shape) != indices_shape:
        return None

    try:
        dnums = eqn.params["dimension_numbers"]
        window_dims = tuple(int(x) for x in dnums.update_window_dims)
        inserted = tuple(int(x) for x in dnums.inserted_window_dims)
        scatter_to_operand = tuple(int(x) for x in dnums.scatter_dims_to_operand_dims)
        operand_batch = tuple(int(x) for x in dnums.operand_batching_dims)
        indices_batch = tuple(int(x) for x in dnums.scatter_indices_batching_dims)
    except (KeyError, TypeError, AttributeError):
        return None

    if len(indices_shape) < 1 or indices_shape[-1] != len(scatter_to_operand):
        return None
    if len(operand_batch) != len(indices_batch):
        return None

    n_updates = int(np.prod(updates_shape))
    rows = (
        np.arange(n_updates, dtype=np.int64)
        if update_rows is None
        else np.asarray(update_rows, dtype=np.int64)
    )
    if np.any((rows < 0) | (rows >= n_updates)):
        return None

    try:
        update_coords = np.unravel_index(rows, updates_shape)

        # Each non-window update dimension corresponds positionally to an
        # indices batch dimension.  This is JAX's shape-rule construction.
        scatter_dim_in_updates: list[int | None] = list(range(len(indices_shape) - 1))

        for axis in window_dims:
            scatter_dim_in_updates.insert(axis, None)

        if len(scatter_dim_in_updates) != len(updates_shape):
            return None

        index_coords = [
            update_coords[axis]
            for axis in range(len(updates_shape))
            if scatter_dim_in_updates[axis] is not None
        ]

        if len(index_coords) != len(indices_shape) - 1:
            return None

        index_base = tuple(index_coords)
        index_vector = indices[index_base]  # one vector for every update entry
        target_coords = [np.zeros(rows.size, dtype=np.int64) for _ in operand_shape]

        for operand_axis, indices_axis in zip(operand_batch, indices_batch):
            target_coords[operand_axis] = index_coords[indices_axis]

        for component, operand_axis in enumerate(scatter_to_operand):
            target_coords[operand_axis] = np.asarray(
                index_vector[..., component], dtype=np.int64
            )
        window_operand_axes = [
            axis
            for axis in range(len(operand_shape))
            if axis not in inserted and axis not in operand_batch
        ]
        if len(window_operand_axes) != len(window_dims):
            return None

        for update_axis, operand_axis in zip(window_dims, window_operand_axes):
            target_coords[operand_axis] += update_coords[update_axis]

        valid = np.ones(rows.size, dtype=bool)

        for coord, size in zip(target_coords, operand_shape):
            valid &= (coord >= 0) & (coord < size)

        target_rows = np.full(rows.size, -1, dtype=np.int64)
        if np.any(valid):
            target_rows[valid] = np.ravel_multi_index(
                tuple(coord[valid] for coord in target_coords), operand_shape
            )

        if not include_index_rows:
            return target_rows, None

        components = np.arange(indices_shape[-1], dtype=np.int64)

        # Give every batch coordinate an explicit trailing axis.  Without it,
        # NumPy broadcasts ``(n,)`` against ``(n, 1)`` to ``(n, n)`` instead of
        # the intended ``(n, index_vector_size)`` index-vector table.
        index_base_columns = tuple(
            np.asarray(coord, dtype=np.int64)[:, None] for coord in index_base
        )
        index_rows = np.ravel_multi_index(
            (*index_base_columns, components[None, :]),
            indices_shape,
        ).ravel()

        # ``np.unique`` takes a disproportionately expensive hash-based path
        # for these large integer vectors on supported NumPy versions.  Scatter
        # index rows are flat C-order integers, so sorting then de-duplicating
        # is both canonical and substantially faster.
        if not index_rows.size:
            return target_rows, np.empty(0, dtype=np.int64)
        index_rows = index_rows.astype(np.int64, copy=False)

        # Requested update rows are normally C-order, which makes their index
        # vectors monotonic as well (including repeated window entries).
        # Preserve that common case without a costly sort/copy.
        if not np.all(index_rows[1:] >= index_rows[:-1]):
            index_rows = np.sort(index_rows)

        index_rows = index_rows[
            np.concatenate(([True], index_rows[1:] != index_rows[:-1]))
        ]
        return target_rows, index_rows
    except (ValueError, IndexError, TypeError):
        return None
