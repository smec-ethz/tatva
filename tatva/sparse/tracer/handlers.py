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

# ---------------------------------------------------------------------------
# JAX primitive propagation handlers namespace class
# ---------------------------------------------------------------------------


import warnings
from abc import ABC, abstractmethod
from typing import Any

import jax
import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.sparse.tracer.common import (
    _DENSE_LINALG,
    _ELEMENTWISE_FFI_TARGETS,
    _NONLINEAR_BINARY,
    _NONLINEAR_UNARY,
    _broadcast_single_row,
    _dot_general_out_dep,
    _get_shape,
    _subjaxpr_and_consts,
    _try_concrete,
)
from tatva.sparse.tracer.state import CouplingAccumulator, SparseDepSet, TraceState


class TracerRegistry:
    """Registry for JAX primitive dependency propagation handlers."""

    def __init__(self):
        self._handlers = {}

    def register(self, *primitive_names: str):
        """Decorator to register a handler for one or more JAX primitives."""

        def decorator(func):
            for name in primitive_names:
                self._handlers[name] = func
            return func

        return decorator

    def get(self, primitive_name: str, default):
        """Get the registered handler, or return the default."""
        return self._handlers.get(primitive_name, default)


# Global registry instance
TRACER_REGISTRY = TracerRegistry()


def get_handler(primitive_name: str):
    """Get the registered handler for a JAX primitive, or return the fallback."""
    return TRACER_REGISTRY.get(primitive_name, fallback)


def fallback(
    eqn: JaxprEqn,
    state: TraceState,
    acc: CouplingAccumulator,
    trial_test_split: int | None,
) -> None:
    """Fallback handler for unrecognized primitives."""
    if not eqn.outvars:
        # Effect-only primitive (no array outputs), e.g. a debug/print callback.
        # Nothing to propagate and nothing couples through it.
        return
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

    cols_active = np.zeros(state.n_dofs, dtype=bool)
    has_active = False
    for d in in_d:
        if isinstance(d, SparseDepSet) and d.dep.shape[0] > 0:
            cols_active[d.dep.indices] = True
            has_active = True
    if not has_active:
        total = SparseDepSet.empty((), state.n_dofs)
    else:
        # This primitive carries a solution dependence but has no handler, so we can
        # only over-approximate its first-order support and cannot record its
        # second-order couplings -> possible false negatives if the result is not
        # re-coupled by a downstream nonlinear primitive. Flag it so an unhandled
        # primitive does not silently degrade the pattern.
        warnings.warn(
            f"Sparsity tracer has no handler for primitive "
            f"'{eqn.primitive.name}'; falling back to a conservative first-order "
            "approximation (second-order couplings not recorded, which can cause "
            "false negatives if the result is not re-coupled by a downstream "
            "nonlinear primitive). Register a handler for this primitive to make "
            "the pattern exact.",
            UserWarning,
        )
        reduced = sps.csr_matrix(cols_active.reshape(1, -1))
        total = SparseDepSet(reduced, ())
    stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
    state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))


@TRACER_REGISTRY.register("debug_print", "debug_callback")
def no_op(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Effect-only debug primitives (``jax.debug.print`` / ``jax.debug.callback``).

    They have no array outputs and contribute nothing to the Hessian, so the tracer
    simply skips them -- letting you sprinkle ``jax.debug.print`` into an energy or
    virtual-work form for debugging without breaking sparsity tracing.
    """
    return


@TRACER_REGISTRY.register("add", "add_any", "sub", "max", "min")
def add_sub_max_min(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    dep_out = (in_d[0].broadcast_to(oshp).dep + in_d[1].broadcast_to(oshp).dep).tocsr()
    dep_out.data[:] = 1
    state.set(eqn.outvars[0], SparseDepSet(dep_out, oshp))


@TRACER_REGISTRY.register(
    "neg",
    "abs",
    "convert_element_type",
    "copy",
    "stop_gradient",
    "device_put",
    "conj",
    "real",
    "imag",
)
def passthrough(
    eqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    state.set(eqn.outvars[0], in_d[0].copy())


@TRACER_REGISTRY.register("broadcast_in_dim")
def broadcast(
    eqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d = in_d[0]
    par = eqn.params
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    shape = par["shape"]
    bdims = par["broadcast_dimensions"]
    if not shape:
        state.set(eqn.outvars[0], d.copy())
        return
    src_indices = np.arange(int(np.prod(d.shape))).reshape(d.shape)
    new_shape = [1] * len(oshp)
    for i, b in enumerate(bdims):
        new_shape[b] = d.shape[i] if i < len(d.shape) else 1
    mapped_src_indices = np.broadcast_to(src_indices.reshape(new_shape), oshp).ravel()
    state.set(eqn.outvars[0], SparseDepSet(d.dep[mapped_src_indices], oshp))


@TRACER_REGISTRY.register("reshape")
def reshape(
    eqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    ns = eqn.params["new_sizes"]
    state.set(eqn.outvars[0], in_d[0].reshape(*ns))


@TRACER_REGISTRY.register("transpose")
def transpose(
    eqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    arr_indices = np.arange(int(np.prod(in_d[0].shape))).reshape(in_d[0].shape)
    perm = np.transpose(arr_indices, eqn.params["permutation"]).ravel()
    state.set(eqn.outvars[0], SparseDepSet(in_d[0].dep[perm], oshp))


@TRACER_REGISTRY.register("squeeze")
def squeeze(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    state.set(eqn.outvars[0], SparseDepSet(in_d[0].dep, oshp))


@TRACER_REGISTRY.register("slice")
def slice_handler(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    arr_indices = np.arange(int(np.prod(in_d[0].shape))).reshape(in_d[0].shape)
    ss, ls = eqn.params["start_indices"], eqn.params["limit_indices"]
    st = eqn.params["strides"] or [1] * len(ss)
    slc = tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))
    sliced_indices = arr_indices[slc].ravel()
    state.set(eqn.outvars[0], SparseDepSet(in_d[0].dep[sliced_indices], oshp))


@TRACER_REGISTRY.register("concatenate")
def concatenate(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    idx_arrays = []
    offset = 0
    for b in in_d:
        size = int(np.prod(b.shape))
        idx_arrays.append(np.arange(size).reshape(b.shape) + offset)
        offset += size
    stacked_dep = sps.vstack([b.dep for b in in_d], format="csr")
    concat_idx = np.concatenate(idx_arrays, axis=eqn.params["dimension"]).ravel()
    state.set(eqn.outvars[0], SparseDepSet(stacked_dep[concat_idx], oshp))


@TRACER_REGISTRY.register("stack")
def stack(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """``stack`` inserts a new axis and places input ``i`` at position ``i`` along it —
    purely structural, so each stacked slice keeps its own per-element support (no
    globalizing union, no couplings). Same construction as ``concatenate`` but the
    input blocks are laid out along a *new* axis rather than concatenated on an
    existing one.
    """
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    idx_arrays = []
    offset = 0
    for b in in_d:
        size = int(np.prod(b.shape))
        idx_arrays.append(np.arange(size).reshape(b.shape) + offset)
        offset += size
    stacked_dep = sps.vstack([b.dep for b in in_d], format="csr")
    stack_idx = np.stack(idx_arrays, axis=eqn.params["axis"]).ravel()
    state.set(eqn.outvars[0], SparseDepSet(stacked_dep[stack_idx], oshp))


@TRACER_REGISTRY.register("split")
def split(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """``split`` (``jnp.split``) cuts one array into several along ``axis`` -- the exact
    inverse of ``concatenate``, and just as structural: every output element *is* one
    input element, so it inherits that element's dependency row verbatim (no union, no
    couplings).

    Handled explicitly because it is a rare multi-output structural primitive:
    ``Handlers.fallback`` writes only ``outvars[0]``, which would leave every piece but
    the first with an empty dep-set (silently dropped couplings) while collapsing the
    first to a whole-array union (a spuriously dense pattern).
    """
    d = state.get(eqn.invars[0])
    axis = eqn.params["axis"]
    # Row-major positions of the input's elements, laid out in its logical shape, so a
    # slice of this index array names exactly the dep rows that piece is built from.
    src_indices = np.arange(int(np.prod(d.shape))).reshape(d.shape)
    offset = 0
    for outvar, size in zip(eqn.outvars, eqn.params["sizes"]):
        size = int(size)
        oshp = _get_shape(outvar)
        piece = np.take(
            src_indices, np.arange(offset, offset + size), axis=axis
        ).ravel()
        state.set(outvar, SparseDepSet(d.dep[piece], oshp))
        offset += size


@TRACER_REGISTRY.register("rev")
def rev(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """``rev`` (``jnp.flip``) reverses element order along the given axes — a pure
    permutation, so it just reorders the dependency rows and keeps each element's own
    support (structural, no couplings)."""
    d = state.get(eqn.invars[0])
    shape = d.shape
    dims = eqn.params["dimensions"]
    src = np.arange(int(np.prod(shape))).reshape(shape)
    slicer = tuple(
        slice(None, None, -1) if ax in dims else slice(None) for ax in range(len(shape))
    )
    rev_idx = src[slicer].ravel()
    state.set(eqn.outvars[0], SparseDepSet(d.dep[rev_idx], shape))


@TRACER_REGISTRY.register("pad")
def pad(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d = in_d[0]
    par = eqn.params
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    n_dofs = state.n_dofs
    result_dep = sps.lil_matrix((int(np.prod(oshp)), n_dofs), dtype=bool)
    if d.dep.shape[0] > 1:
        result_indices = np.arange(int(np.prod(oshp))).reshape(oshp)
        slc = tuple(
            slice(lo, lo + s) for (lo, _, _), s in zip(par["padding_config"], d.shape)
        )
        target_indices = result_indices[slc].ravel()
        result_dep[target_indices] = d.dep
    state.set(eqn.outvars[0], SparseDepSet(result_dep.tocsr(), oshp))


@TRACER_REGISTRY.register(
    "reduce_sum", "reduce_max", "reduce_min", "reduce_and", "reduce_or"
)
def reduce(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    axes = eqn.params["axes"]
    shape = in_d[0].shape
    out_indices = np.arange(int(np.prod(oshp))).reshape(oshp)
    broadcast_shape = list(shape)
    new_dims = list(oshp)
    for ax in sorted(axes):
        new_dims.insert(ax, 1)
    mapped_out_indices = np.broadcast_to(
        out_indices.reshape(new_dims), broadcast_shape
    ).ravel()

    rows = mapped_out_indices
    cols = np.arange(len(mapped_out_indices))
    data = np.ones(len(mapped_out_indices), dtype=bool)
    G = sps.csr_matrix(
        (data, (rows, cols)), shape=(int(np.prod(oshp)), int(np.prod(shape)))
    )

    dep_out = (G @ in_d[0].dep).tocsr()
    dep_out.data[:] = 1
    state.set(eqn.outvars[0], SparseDepSet(dep_out, oshp))


@TRACER_REGISTRY.register("select_n")
def select_n(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    cond_val = state.get_val(eqn.invars[0])
    branches = in_d[1:]

    if cond_val is not None and len(branches) == 2:
        cond_flat = np.asarray(cond_val).ravel().astype(bool)
        dep_0 = branches[0].dep.tocsr()
        dep_1 = branches[1].dep.tocsr()

        d0_data = dep_0.data.copy()
        d1_data = dep_1.data.copy()

        row_0 = np.repeat(
            np.arange(dep_0.shape[0]), dep_0.indptr[1:] - dep_0.indptr[:-1]
        )
        d0_data[cond_flat[row_0]] = 0

        row_1 = np.repeat(
            np.arange(dep_1.shape[0]), dep_1.indptr[1:] - dep_1.indptr[:-1]
        )
        d1_data[~cond_flat[row_1]] = 0

        dep0_masked = sps.csr_matrix(
            (d0_data, dep_0.indices, dep_0.indptr), shape=dep_0.shape
        )
        dep1_masked = sps.csr_matrix(
            (d1_data, dep_1.indices, dep_1.indptr), shape=dep_1.shape
        )
        dep0_masked.eliminate_zeros()
        dep1_masked.eliminate_zeros()

        dep_out = (dep0_masked + dep1_masked).astype(bool).tocsr()
    else:
        summed = sum(b.dep for b in branches)
        dep_out = sps.csr_matrix(summed.astype(bool))  # ty:ignore[unresolved-attribute]

    state.set(eqn.outvars[0], SparseDepSet(dep_out, oshp))


@TRACER_REGISTRY.register(
    "iota",
    "lt",
    "lt_to",
    "le",
    "le_to",
    "gt",
    "ge",
    "eq",
    "ne",
    "and",
    "or",
    "not",
    "xor",
    "shift_left",
    "shift_right_arithmetic",
    "is_finite",
    "is_nan",
    "argmax",
    "argmin",
    "floor",
    "ceil",
    "round",
    "sign",
)
def zero_dependency(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    state.set(eqn.outvars[0], SparseDepSet.empty(oshp, state.n_dofs))


@TRACER_REGISTRY.register(*_NONLINEAR_UNARY)
def nonlinear_unary(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d = in_d[0]
    state.set(eqn.outvars[0], d.copy())
    d.record_couplings(acc, trial_test_split)


@TRACER_REGISTRY.register(*_NONLINEAR_BINARY)
def nonlinear_binary(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    dep_out = (in_d[0].broadcast_to(oshp).dep + in_d[1].broadcast_to(oshp).dep).tocsr()
    dep_out.data[:] = 1
    combined = SparseDepSet(dep_out, oshp)
    state.set(eqn.outvars[0], combined)

    is_const0 = in_d[0].dep.nnz == 0
    is_const1 = in_d[1].dep.nnz == 0

    is_linear = False
    p = eqn.primitive.name
    if p == "mul" and (is_const0 or is_const1):
        is_linear = True
    elif p == "div" and is_const1:
        is_linear = True

    if not is_linear:
        combined.record_couplings(acc, trial_test_split)


@TRACER_REGISTRY.register(*_DENSE_LINALG)
def dense_linalg(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Dense nonlinear linear-algebra primitives (``lu`` / ``custom_linear_solve`` /
    ``triangular_solve`` / ``lu_solve`` / ``cholesky``) — the leaf ops behind
    ``jnp.linalg.inv`` / ``solve`` / ``cholesky``.

    Each is a *dense* nonlinear map: every output entry depends on the union of all
    solution-dependent input entries, and — being nonlinear — those input DOFs mutually
    couple (a full Hessian block over the union). We treat the op as one dense black
    box: union the solution-dependent inputs, record that union's self outer-product,
    and give every output element the union support. It is conservative, but these ops
    act on small element-local matrices whose DOFs already form a dense block in the
    element stiffness, so in practice it adds no real fill. Without a handler these
    primitives fell to the generic fallback, which records *no* couplings and can drop
    the op's curvature (false negatives) when its result is not re-coupled downstream.
    """
    in_d = [state.get(v) for v in eqn.invars]
    cols = np.zeros(state.n_dofs, dtype=bool)
    for d in in_d:
        if d.dep.shape[0] and d.dep.nnz:
            cols[d.dep.indices] = True
    union_row = sps.csr_matrix(cols.reshape(1, -1))

    # Dense self-coupling over the union: a real Hessian block for a nonlinear op.
    # record_dep computes union_rowᵀ @ union_row -> the full union×union block.
    if union_row.nnz:
        acc.record_dep(union_row, trial_test_split)

    # Every output element (all outputs, e.g. lu's LU/pivots/permutation) depends on
    # the whole union.
    for ov in eqn.outvars:
        oshp = _get_shape(ov)
        state.set(
            ov,
            SparseDepSet(_broadcast_single_row(union_row, int(np.prod(oshp))), oshp),
        )


@TRACER_REGISTRY.register("integer_pow")
def integer_pow(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d = in_d[0]
    state.set(eqn.outvars[0], d.copy())
    if eqn.params["y"] >= 2 or eqn.params["y"] <= -1:
        d.record_couplings(acc, trial_test_split)


@TRACER_REGISTRY.register("gather")
def gather(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d_src = in_d[0]
    idx = state.get_val(eqn.invars[1])
    par = eqn.params
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

    if idx is not None and d_src.dep.shape[0] >= 1:
        try:
            dnums = par["dimension_numbers"]
            ss = par["slice_sizes"]
            collapsed = dnums.collapsed_slice_dims
            sim = dnums.start_index_map

            arr_indices = np.arange(int(np.prod(d_src.shape))).reshape(d_src.shape)

            # Branch 1: 1D gather along dimension 0
            if collapsed == (0,) and sim == (0,) and ss[0] == 1:
                rows = np.clip(idx.ravel().astype(int), 0, d_src.shape[0] - 1)
                slc = (rows,) + tuple(slice(None, s) for s in ss[1:])
                flat_src_indices = arr_indices[slc].ravel()
                state.set(
                    eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                )
                return

            # Branch 2: 2D indexing with slice along axis 1
            if (
                collapsed == (0,)
                and sim == (0, 1)
                and ss[0] == 1
                and idx.ndim == 2
                and idx.shape[1] == 2
            ):
                r = np.clip(idx[:, 0].astype(int), 0, d_src.shape[0] - 1)
                c0 = np.clip(idx[:, 1].astype(int), 0, d_src.shape[1] - ss[1])

                # Vectorized slice index calculation
                col_offsets = np.arange(ss[1])
                cols = c0[:, None] + col_offsets[None, :]  # Shape: (n_gathered, ss[1])
                flat_src_indices = arr_indices[r[:, None], cols].ravel()

                state.set(
                    eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                )
                return

            # Branch 3: N-D point indexing with proper start_index_map mapping
            if (
                set(collapsed) == set(sim)
                and idx.ndim == 2
                and idx.shape[1] == len(sim)
                and all(ss[a] == 1 for a in sim)
            ):
                # Map index columns to proper operand axes according to sim
                coords = [None] * len(d_src.shape)
                for j, axis in enumerate(sim):
                    coords[axis] = np.clip(
                        idx[:, j].astype(int), 0, d_src.shape[axis] - 1
                    )

                flat_src_indices = arr_indices[tuple(coords)].ravel()
                state.set(
                    eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                )
                return
        except Exception:
            pass

    total = d_src.total_union()
    stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
    state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))


@TRACER_REGISTRY.register(
    "scatter",
    "scatter-add",
    "scatter-sub",
    "scatter-mul",
    "scatter-min",
    "scatter-max",
)
def scatter(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    d_tgt = in_d[0]
    d_vals = in_d[2]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    p = eqn.primitive.name
    nonlinear = p == "scatter-mul"

    u_vals = sps.csr_matrix(d_vals.dep.sum(axis=0).astype(bool))
    idx = state.get_val(eqn.invars[1])

    if idx is not None and d_tgt.dep.shape[0] > 1:
        try:
            idx_flat = idx.ravel().astype(int)
            if len(idx_flat) == d_vals.dep.shape[0]:
                # Vectorized precise element-wise scatter routing!
                coo_vals = d_vals.dep.tocoo()
                row_mapped = idx_flat[coo_vals.row]
                valid = (row_mapped >= 0) & (row_mapped < d_tgt.dep.shape[0])
                scattered_row = row_mapped[valid]
                scattered_col = coo_vals.col[valid]
                scattered_data = coo_vals.data[valid]

                scattered_mat = sps.csr_matrix(
                    (scattered_data, (scattered_row, scattered_col)),
                    shape=d_tgt.dep.shape,
                )
                res_dep = (d_tgt.dep + scattered_mat).tocsr()
            else:
                # Vectorized broadcast scatter routing!
                valid_idx = idx_flat[(idx_flat >= 0) & (idx_flat < d_tgt.dep.shape[0])]
                if len(valid_idx) > 0:
                    coo_u = u_vals.tocoo()
                    u_cols = coo_u.col
                    u_data = coo_u.data

                    scattered_row = np.repeat(valid_idx, len(u_cols))
                    scattered_col = np.tile(u_cols, len(valid_idx))
                    scattered_data = np.tile(u_data, len(valid_idx))

                    scattered_mat = sps.csr_matrix(
                        (scattered_data, (scattered_row, scattered_col)),
                        shape=d_tgt.dep.shape,
                    )
                    res_dep = (d_tgt.dep + scattered_mat).tocsr()
                else:
                    res_dep = d_tgt.dep.copy()

            res_dep.data[:] = 1
            res = SparseDepSet(res_dep, oshp)
            state.set(eqn.outvars[0], res)
            if nonlinear:
                res.record_couplings(acc, trial_test_split)
            return
        except Exception:
            pass

    result = d_tgt.dep + _broadcast_single_row(u_vals, int(np.prod(oshp)))
    res_dep = result.tocsr()
    res_dep.data[:] = 1
    res = SparseDepSet(res_dep, oshp)
    state.set(eqn.outvars[0], res)
    if nonlinear:
        res.record_couplings(acc, trial_test_split)


@TRACER_REGISTRY.register("dynamic_slice", "dynamic_update_slice")
def dynamic_slice(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
    cols_active = np.zeros(state.n_dofs, dtype=bool)
    has_active = False
    for d in in_d:
        if isinstance(d, SparseDepSet) and d.dep.shape[0] > 0:
            cols_active[d.dep.indices] = True
            has_active = True
    if not has_active:
        total = SparseDepSet.empty((), state.n_dofs)
    else:
        reduced = sps.csr_matrix(cols_active.reshape(1, -1))
        total = SparseDepSet(reduced, ())
    stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
    state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))


@TRACER_REGISTRY.register("dot_general")
def dot(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    in_d = [state.get(v) for v in eqn.invars]
    oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

    # Determine contracting and batch axes
    dn = eqn.params.get("dimension_numbers")
    lhs_batch = []
    rhs_batch = []
    lhs_contract = []
    rhs_contract = []
    if dn is not None:
        if hasattr(dn, "lhs_batch_dimensions"):
            lhs_batch = dn.lhs_batch_dimensions
            rhs_batch = dn.rhs_batch_dimensions
            lhs_contract = dn.lhs_contracting_dimensions
            rhs_contract = dn.rhs_contracting_dimensions
        elif len(dn) > 1:
            lhs_contract = dn[0][0]
            rhs_contract = dn[0][1]
            lhs_batch = dn[1][0]
            rhs_batch = dn[1][1]

    is_batched = False
    if lhs_batch and rhs_batch:
        if len(lhs_batch) > 0 and len(rhs_batch) > 0:
            if lhs_batch[0] == 0 and rhs_batch[0] == 0:
                is_batched = True

    if is_batched and len(in_d[0].shape) > 1 and len(in_d[1].shape) > 1:
        B = in_d[0].shape[0]
        S_a = int(np.prod(in_d[0].shape[1:]))
        S_b = int(np.prod(in_d[1].shape[1:]))
        S_out = int(np.prod(oshp[1:]))

        # Vectorized extraction of active DOFs per batch element
        coo_a = in_d[0].dep.tocoo()
        batch_a = coo_a.row // S_a
        dof_a = coo_a.col
        A_active = sps.csr_matrix(
            (np.ones(len(batch_a), dtype=bool), (batch_a, dof_a)),
            shape=(B, state.n_dofs),
            dtype=bool,
        )

        coo_b = in_d[1].dep.tocoo()
        batch_b = coo_b.row // S_b
        dof_b = coo_b.col
        B_active = sps.csr_matrix(
            (np.ones(len(batch_b), dtype=bool), (batch_b, dof_b)),
            shape=(B, state.n_dofs),
            dtype=bool,
        )

        # A batched contraction is bilinear: it couples only when BOTH
        # operands depend on the input. If either is constant (no active
        # DOFs) the contraction is linear -> record no couplings. Covers
        # both (const, var) and (var, const).
        if A_active.nnz and B_active.nnz:
            # Self-couplings (A.T@A / B.T@B) are real Hessian blocks only for an
            # operand that is itself a *nonlinear* function of u -- and in that case
            # they are already recorded upstream at the primitive that made it
            # nonlinear. An affine operand has zero second derivative, so its self
            # outer-product is spurious; record it only for nonlinear operands.
            if state.is_nonlinear(eqn.invars[0]):
                acc.record_dep(A_active, trial_test_split)
            if state.is_nonlinear(eqn.invars[1]):
                acc.record_dep(B_active, trial_test_split)

            # Cross-couplings between A and B (no fingerprint cache - structurally distinct)
            P_cross = (A_active.T @ B_active).tocsr()
            r_c, c_c = P_cross.nonzero()
            if trial_test_split is not None:
                mask_c = (r_c < trial_test_split) & (c_c >= trial_test_split)
                mask_c |= (r_c >= trial_test_split) & (c_c < trial_test_split)
                r_c, c_c = r_c[mask_c], c_c[mask_c]
            acc.add_coords(r_c, c_c)
            acc.add_coords(c_c, r_c)

        # Vectorized construction of stacked output dependencies
        C_active = (A_active + B_active).astype(bool).tocsr()
        repeat_indices = np.repeat(np.arange(B), S_out)
        stacked_dep = C_active[repeat_indices]

        state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))
    else:
        ua = in_d[0].total_union()
        ub = in_d[1].total_union()

        ia = ua.dep.indices
        ib = ub.dep.indices
        # A contraction dot(a, b) is bilinear, so it contributes second-order
        # coupling only when BOTH operands depend on the input. If either
        # operand is constant (empty dep-set) the contraction is linear in the
        # other and has zero Hessian -> record no couplings. This guard covers
        # both (const, var) and (var, const). The self outer-products ua.T@ua /
        # ub.T@ub are real Hessian blocks only when that operand is itself a
        # *nonlinear* function of u (and are then already recorded upstream at the
        # primitive that made it nonlinear); for an affine operand the second
        # derivative vanishes, so recording the self-product is spurious. Hence
        # gate each self-recording on operand nonlinearity and always record the
        # bilinear cross-term.
        if ia.size and ib.size:
            if state.is_nonlinear(eqn.invars[0]):
                ua.record_couplings(acc, trial_test_split)
            if state.is_nonlinear(eqn.invars[1]):
                ub.record_couplings(acc, trial_test_split)

            # Vectorized outer-product of active DOF indices for cross-couplings
            r_c = np.repeat(ia, ib.size)
            c_c = np.tile(ib, ia.size)
            if trial_test_split is not None:
                mask = ((r_c < trial_test_split) & (c_c >= trial_test_split)) | (
                    (r_c >= trial_test_split) & (c_c < trial_test_split)
                )
                r_c = r_c[mask]
                c_c = c_c[mask]
            acc.add_coords(r_c, c_c)
            acc.add_coords(c_c, r_c)

        # Propagate output support preserving per-element (free-dimension) structure
        # rather than broadcasting the whole-array union to every output element. This
        # keeps a locally-supported contraction (e.g. ``field @ normal`` with a constant
        # normal) local, instead of making every output component depend on all DOFs.
        out_dep = _dot_general_out_dep(
            in_d[0],
            in_d[1],
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
            oshp,
            state.n_dofs,
        )
        state.set(eqn.outvars[0], out_dep)


@TRACER_REGISTRY.register("cond", "switch", "while")
def control_flow(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Conservative fallback for control-flow primitives.

    Unions the input dependencies into every output element without descending
    into the carried jaxpr, so it propagates first-order dependencies but records
    no second-order couplings of its own. ``cond`` is handled precisely by the
    dedicated ``cond`` handler (which overrides this registration); ``switch``
    lowers to ``cond``, so in practice only ``while`` reaches here.

    The missing couplings are not reachable for the Hessian use case: a ``while``
    appearing inside an energy would have to be double-differentiable to contribute
    to the Hessian, but ``lax.while_loop`` is not reverse-mode differentiable in JAX
    (this is precisely why iterative solvers are wrapped in ``custom_vjp`` /
    implicit differentiation, which is handled by ``custom_vjp_jvp_call`` instead).
    """
    in_d = [state.get(v) for v in eqn.invars]
    cols_active = np.zeros(state.n_dofs, dtype=bool)
    has_active = False
    for d in in_d:
        if isinstance(d, SparseDepSet) and d.dep.shape[0] > 0:
            cols_active[d.dep.indices] = True
            has_active = True
    if not has_active:
        total = SparseDepSet.empty((), state.n_dofs)
    else:
        reduced = sps.csr_matrix(cols_active.reshape(1, -1))
        total = SparseDepSet(reduced, ())
    for ov in eqn.outvars:
        ov_shape = _get_shape(ov)
        ov_dep = (
            _broadcast_single_row(total.dep, int(np.prod(ov_shape)))
            if ov_shape
            else total.dep.copy()
        )
        state.set(ov, SparseDepSet(ov_dep, ov_shape))


@TRACER_REGISTRY.register("scan", "map")
def scan_map(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    closed_sub = eqn.params["jaxpr"]
    sub = closed_sub.jaxpr
    sub_consts = closed_sub.consts

    num_const = eqn.params.get("num_consts", 0)
    num_carry = eqn.params.get("num_carry", 0)
    num_xs = len(eqn.invars) - num_const - num_carry

    # Determine local shapes and sizes
    slice_shapes = []
    size_slices = []
    for k in range(num_xs):
        x = eqn.invars[num_const + num_carry + k]
        x_dep = state.get(x)
        slice_shapes.append(x_dep.shape[1:])
        size_slices.append(int(np.prod(x_dep.shape[1:])))

    # Determine if we have a batch dimension in the slices (e.g. from batched lax.map)
    batch_size = 1
    local_size_slices = []
    for k in range(num_xs):
        shp = slice_shapes[k]
        if len(shp) > 1:
            batch_size = shp[0]
            local_size_slices.append(int(np.prod(shp[1:])))
        else:
            local_size_slices.append(int(np.prod(shp)))

    n_local_dofs = sum(local_size_slices)
    length = eqn.params.get("length", 1)

    # Seed sub-state with symbolic dependencies for mapped inputs
    sub_active, sub_index_set, sub_bound_eqns = state.sub_info[id(eqn)]
    sub_state = TraceState(
        n_local_dofs, sub_active, {}, state.sub_info, state.nonlinear_ids
    )

    # Seed consts: empty deps
    for i in range(num_const):
        sub_state.set(
            sub.invars[i],
            SparseDepSet.empty(_get_shape(sub.invars[i]), n_local_dofs),
        )
        if state.is_nonlinear(eqn.invars[i]):
            sub_state.nonlinear_ids.add(id(sub.invars[i]))
        val = state.get_val(eqn.invars[i])
        if val is not None:
            sub_state.val_of[id(sub.invars[i])] = val

    # Seed consts of the sub-jaxpr
    for v, c in zip(sub.constvars, sub_consts):
        sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_local_dofs))
        sub_state.val_of[id(v)] = np.asarray(c)

    # Seed carry: empty deps
    for i in range(num_carry):
        sub_state.set(
            sub.invars[num_const + i],
            SparseDepSet.empty(_get_shape(sub.invars[num_const + i]), n_local_dofs),
        )
        val = state.get_val(eqn.invars[num_const + i])
        if val is not None:
            sub_state.val_of[id(sub.invars[num_const + i])] = val

    # Seed mapped inputs (xs) with symbolic singletons repeated for each batch index
    symbolic_seed = SparseDepSet.singletons(n_local_dofs)
    offset = 0
    for k in range(num_xs):
        sz = local_size_slices[k]
        shp = slice_shapes[k]
        local_dep = symbolic_seed.dep[offset : offset + sz]

        # Repeat local_dep vertically batch_size times to give each batch entry the same symbolic DOFs
        if batch_size > 1:
            # local_dep has sz rows, each with a single nnz; tile via direct CSR build
            local_indptr = local_dep.indptr
            local_indices = local_dep.indices
            local_data = local_dep.data
            # build (batch_size * sz, n_local_dofs) by tiling rows
            tiled_indices = np.tile(local_indices, batch_size)
            tiled_data = np.tile(local_data, batch_size)
            # Each tile has the same row-nnz layout; total rows = batch_size * sz.
            # Vectorized tiled indptr: row r of tile i -> local_indptr[r] + i*base_step
            # (avoids an O(batch_size) Python comprehension).
            base_step = local_indptr[-1]
            base = local_indptr[:-1]
            tiled_indptr = (
                base[None, :] + (np.arange(batch_size) * base_step)[:, None]
            ).ravel()
            tiled_indptr = np.concatenate(
                [tiled_indptr, np.array([batch_size * base_step])]
            ).astype(local_indptr.dtype)
            sub_dep = sps.csr_matrix(
                (tiled_data, tiled_indices, tiled_indptr),
                shape=(batch_size * sz, n_local_dofs),
            )
        else:
            sub_dep = local_dep

        sub_state.set(sub.invars[num_const + num_carry + k], SparseDepSet(sub_dep, shp))
        if state.is_nonlinear(eqn.invars[num_const + num_carry + k]):
            sub_state.nonlinear_ids.add(id(sub.invars[num_const + num_carry + k]))

        # Set a representative concrete value from element 0
        val = state.get_val(eqn.invars[num_const + num_carry + k])
        if val is not None and val.shape[0] > 0:
            sub_state.val_of[id(sub.invars[num_const + num_carry + k])] = val[0]

        offset += sz

    # Trace the sub-jaxpr symbolically using a child accumulator
    sub_acc = CouplingAccumulator(n_local_dofs)
    for sub_eqn, sub_handler, sub_is_active, sub_needs_concrete in sub_bound_eqns:
        sub_ovars = sub_eqn.outvars
        if sub_ovars and not sub_is_active:
            for v in sub_ovars:
                sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_local_dofs))
            if sub_needs_concrete:
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(sub_ovars[0])] = cv
            continue

        sub_handler(sub_eqn, sub_state, sub_acc, None)
        sub_state.mark_nonlinear(sub_eqn)

        if sub_ovars and sub_needs_concrete:
            in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
            cv = _try_concrete(sub_eqn.primitive, in_vals, sub_eqn.params)
            if cv is not None:
                sub_state.val_of[id(sub_ovars[0])] = cv

    # Extract unique local couplings from the sub-accumulator. sub_pat.nonzero()
    # already yields unique (r, c); canonicalize to (lo, hi) and dedup via a scipy
    # sparse round-trip (np.unique is slow on Python >= 3.13).
    sub_pat = sub_acc.finalize()
    sub_r, sub_c = sub_pat.nonzero()
    if sub_r.size:
        lo = np.minimum(sub_r, sub_c)
        hi = np.maximum(sub_r, sub_c)
        canon = sps.csr_matrix(
            (np.ones(lo.size, dtype=bool), (lo, hi)),
            shape=(n_local_dofs, n_local_dofs),
        )
        lo_arr, hi_arr = canon.nonzero()
    else:
        lo_arr = hi_arr = np.empty(0, dtype=np.intp)

    # Cumulative offsets to locate which mapped input owns a local index, plus a
    # per-local-index cache of strided column slices (the same local index recurs
    # across many coupling pairs, so slicing once avoids redundant CSR fancy-indexing).
    offsets = np.concatenate(([0], np.cumsum(local_size_slices)))
    slice_cache: dict[int, tuple[sps.csr_matrix, np.ndarray]] = {}

    def _get_col(local_idx: int) -> tuple[sps.csr_matrix, np.ndarray]:
        cached = slice_cache.get(local_idx)
        if cached is not None:
            return cached
        k_idx = int(np.searchsorted(offsets, local_idx, side="right")) - 1
        idx_in = local_idx - int(offsets[k_idx])
        n_k = local_size_slices[k_idx]
        dep_x = state.get(eqn.invars[num_const + num_carry + k_idx]).dep
        col = dep_x[idx_in::n_k]
        nnz_per_row = col.indptr[1:] - col.indptr[:-1]
        cached = (col, nnz_per_row)
        slice_cache[local_idx] = cached
        return cached

    # Build local sparse matrices for each unique (canonicalized) coupling pair
    for la, lb in zip(lo_arr.tolist(), hi_arr.tolist()):
        col_a, nnz_per_row_a = _get_col(la)
        col_b, nnz_per_row_b = _get_col(lb)

        # Check if we can use the fast direct index mapping (at most 1 nnz per row)
        if np.all(nnz_per_row_a <= 1) and np.all(nnz_per_row_b <= 1):
            # Direct index mapping (lightning fast, 0 ms!)
            active_a = nnz_per_row_a == 1
            active_b = nnz_per_row_b == 1
            active = active_a & active_b

            if np.any(active):
                c_a = col_a.indices[col_a.indptr[:-1][active]]
                c_b = col_b.indices[col_b.indptr[:-1][active]]

                if trial_test_split is not None:
                    mask = (c_a < trial_test_split) & (c_b >= trial_test_split)
                    mask |= (c_a >= trial_test_split) & (c_b < trial_test_split)
                    c_a = c_a[mask]
                    c_b = c_b[mask]

                acc.add_coords(c_a, c_b)
                acc.add_coords(c_b, c_a)
        else:
            # Fallback to general sparse matrix multiplication
            couplings = (col_a.T @ col_b).tocsr()
            r, c = couplings.nonzero()
            if trial_test_split is not None:
                mask = (r < trial_test_split) & (c >= trial_test_split)
                mask |= (r >= trial_test_split) & (c < trial_test_split)
                r = r[mask]
                c = c[mask]
            acc.add_coords(r, c)
            acc.add_coords(c, r)

    total_length = length * batch_size

    # Map outputs back to parent state
    # carry outputs
    for i in range(num_carry):
        state.set(
            eqn.outvars[i],
            SparseDepSet.empty(_get_shape(eqn.outvars[i]), state.n_dofs),
        )
        c_val = sub_state.get_val(sub.outvars[i])
        if c_val is not None:
            state.val_of[id(eqn.outvars[i])] = c_val

    # Propagate exact concrete values for scan outputs if invars have known values
    in_vals = [state.get_val(v) for v in eqn.invars]
    if not any(v is None for v in in_vals):
        try:
            v_casted = []
            for x, invar in zip(in_vals, sub.invars):
                target_dtype = getattr(invar.aval, "dtype", None)
                x_arr = np.asarray(x)
                if target_dtype is not None and x_arr.dtype != target_dtype:
                    x_arr = x_arr.astype(target_dtype)
                v_casted.append(x_arr)
            res = jax.core.eval_jaxpr(sub, sub_consts, *v_casted)
            for i in range(len(eqn.outvars)):
                state.val_of[id(eqn.outvars[i])] = np.asarray(res[i])
        except Exception:
            pass

    # mapped outputs (ys)
    if len(eqn.outvars) > num_carry:
        dep_inputs = [
            state.get(eqn.invars[num_const + num_carry + k]).dep for k in range(num_xs)
        ]
        dep_all = sps.vstack(dep_inputs, format="csr")

        offsets_all = [0]
        for k in range(num_xs - 1):
            offsets_all.append(offsets_all[-1] + total_length * local_size_slices[k])

        # Vectorized perm_idx construction using NumPy
        indices_list = []
        for k in range(num_xs):
            sz = local_size_slices[k]
            start_indices = offsets_all[k] + np.arange(total_length) * sz
            indices_list.append(start_indices[:, None] + np.arange(sz)[None, :])
        perm_idx = np.hstack(indices_list).ravel()

        In_all = dep_all[perm_idx]

        for i in range(len(eqn.outvars) - num_carry):
            sub_y = sub.outvars[num_carry + i]
            y = eqn.outvars[num_carry + i]

            sub_y_dep = sub_state.get(sub_y)
            slice_shape = sub_y_dep.shape

            sz_local_y = (
                int(np.prod(slice_shape[1:]))
                if len(slice_shape) > 1
                else int(np.prod(slice_shape))
            )

            # Vectorized sub_y_block construction using NumPy broadcasting instead of a Python loop
            Y_coo = sub_y_dep.dep.tocoo()
            if length > 1:
                l_arr = np.arange(length)
                r_block = (
                    Y_coo.row[None, :] + l_arr[:, None] * (batch_size * sz_local_y)
                ).ravel()

                base_col = (Y_coo.row // sz_local_y) * n_local_dofs + Y_coo.col
                c_block = (
                    base_col[None, :] + l_arr[:, None] * (batch_size * n_local_dofs)
                ).ravel()

                data_block = np.tile(Y_coo.data, length)
            else:
                r_block = Y_coo.row
                c_block = (Y_coo.row // sz_local_y) * n_local_dofs + Y_coo.col
                data_block = Y_coo.data

            sub_y_block = sps.csr_matrix(
                (data_block, (r_block, c_block)),
                shape=(total_length * sz_local_y, total_length * n_local_dofs),
            )

            expanded_dep = (sub_y_block @ In_all).tocsr()
            expanded_dep.data[:] = 1

            y_shape = (length,) + slice_shape
            state.set(y, SparseDepSet(expanded_dep, y_shape))
            if sub_state.is_nonlinear(sub_y):
                state.nonlinear_ids.add(id(y))
            sub_y_val = sub_state.get_val(sub_y)
            if sub_y_val is not None:
                try:
                    state.val_of[id(y)] = np.broadcast_to(sub_y_val, y_shape).copy()
                except Exception:
                    pass


@TRACER_REGISTRY.register("pjit", "jit", "remat2")
def subjaxpr(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Trace a single carried sub-jaxpr (``pjit``/``jit``/``remat2``).

    ``remat2`` (``jax.checkpoint``) is just a memory-recompute wrapper around an
    ordinary computation: descending into its jaxpr records the couplings created
    inside exactly as for ``pjit``. Treating it as opaque (fallback) would silently
    drop those couplings and under-count the Hessian.
    """
    sub, sub_consts = _subjaxpr_and_consts(eqn)
    in_d = [state.get(v) for v in eqn.invars]

    # Retrieve the pre-resolved active set and bound equations for this sub-jaxpr
    sub_active, sub_index_set, sub_bound_eqns = state.sub_info[id(eqn)]

    n_dofs = state.n_dofs
    sub_state = TraceState(
        n_dofs, sub_active, state.tags, state.sub_info, state.nonlinear_ids
    )

    for v, d in zip(sub.invars, in_d):
        sub_state.set(v, d)
    # Carry operand nonlinearity across the jit boundary onto the sub-invars.
    for pv, sv in zip(eqn.invars, sub.invars):
        if state.is_nonlinear(pv):
            sub_state.nonlinear_ids.add(id(sv))
        val = state.get_val(pv)
        if val is not None:
            sub_state.val_of[id(sv)] = val
    for v, c in zip(sub.constvars, sub_consts):
        sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
        sub_state.val_of[id(v)] = np.asarray(c)

    for sub_eqn, sub_handler, sub_is_active, sub_needs_concrete in sub_bound_eqns:
        ovars = sub_eqn.outvars
        if ovars and not sub_is_active:
            for v in ovars:
                sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
            # propagate concrete values (essential for gather/scatter routing indices)
            if sub_needs_concrete:
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(ovars[0])] = cv
            continue

        sub_handler(sub_eqn, sub_state, acc, trial_test_split)
        sub_state.mark_nonlinear(sub_eqn)

        # Propagate concrete value for executed equations as well
        if ovars and sub_needs_concrete:
            in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
            cv = _try_concrete(sub_eqn.primitive, in_vals, sub_eqn.params)
            if cv is not None:
                sub_state.val_of[id(ovars[0])] = cv

    for pv, sv in zip(eqn.outvars, sub.outvars):
        state.set(pv, sub_state.get(sv))
        if sub_state.is_nonlinear(sv):
            state.nonlinear_ids.add(id(pv))
        val = sub_state.get_val(sv)
        if val is not None:
            state.val_of[id(pv)] = val

    in_vals = [state.get_val(v) for v in eqn.invars]
    if not any(v is None for v in in_vals):
        try:
            v_casted = []
            for x, invar in zip(in_vals, sub.invars):
                target_dtype = getattr(invar.aval, "dtype", None)
                x_arr = np.asarray(x)
                if target_dtype is not None and x_arr.dtype != target_dtype:
                    x_arr = x_arr.astype(target_dtype)
                v_casted.append(x_arr)
            res = jax.core.eval_jaxpr(sub, sub_consts, *v_casted)
            for pv, r in zip(eqn.outvars, res):
                state.val_of[id(pv)] = np.asarray(r)
        except Exception:
            pass


@TRACER_REGISTRY.register("cond")
def cond(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Trace each branch of a ``cond``/``switch`` and union the outputs.

    Unlike the conservative ``control_flow`` fallback, this traverses every branch
    jaxpr — recording the couplings created *inside* a branch into the shared
    accumulator — and propagates the OR of all branches' output dependency sets. A
    piecewise function's derivative is one branch's, so unioning the branches is an
    AD-safe superset. The predicate (``invars[0]``) only selects and contributes no
    couplings, so it is ignored.
    """
    operands = eqn.invars[1:]
    in_d = [state.get(v) for v in operands]
    n_dofs = state.n_dofs
    branch_sub_list = state.sub_info[id(eqn)]

    out_deps: dict[int, SparseDepSet | None] = {id(ov): None for ov in eqn.outvars}
    for (sub_active, sub_index_set, sub_bound_eqns), branch in zip(
        branch_sub_list, eqn.params["branches"]
    ):
        sub = branch.jaxpr
        sub_consts = branch.consts
        sub_state = TraceState(
            n_dofs, sub_active, state.tags, state.sub_info, state.nonlinear_ids
        )

        # Seed branch invars with operand dep-sets and concrete values (the latter
        # are needed to route any gather/scatter indices inside the branch).
        for sv, d, ov in zip(sub.invars, in_d, operands):
            sub_state.set(sv, d)
            if state.is_nonlinear(ov):
                sub_state.nonlinear_ids.add(id(sv))
            val = state.get_val(ov)
            if val is not None:
                sub_state.val_of[id(sv)] = val
        for v, c in zip(sub.constvars, sub_consts):
            sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
            sub_state.val_of[id(v)] = np.asarray(c)

        for (
            sub_eqn,
            sub_handler,
            sub_is_active,
            sub_needs_concrete,
        ) in sub_bound_eqns:
            ovars = sub_eqn.outvars
            if ovars and not sub_is_active:
                for v in ovars:
                    sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
                if sub_needs_concrete:
                    in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                    cv = _try_concrete(
                        sub_eqn.primitive,
                        in_vals,
                        sub_eqn.params,
                    )
                    if cv is not None:
                        sub_state.val_of[id(ovars[0])] = cv
                continue

            sub_handler(sub_eqn, sub_state, acc, trial_test_split)
            sub_state.mark_nonlinear(sub_eqn)

            if ovars and sub_needs_concrete:
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(ovars[0])] = cv

        for ov, sv in zip(eqn.outvars, sub.outvars):
            d = sub_state.get(sv)
            if sub_state.is_nonlinear(sv):
                state.nonlinear_ids.add(id(ov))
            prev = out_deps[id(ov)]
            if prev is None:
                out_deps[id(ov)] = d
            else:
                merged = (prev.dep + d.dep).tocsr()
                merged.data[:] = 1
                out_deps[id(ov)] = SparseDepSet(merged, d.shape)

    for ov in eqn.outvars:
        d = out_deps[id(ov)]
        state.set(
            ov, d if d is not None else SparseDepSet.empty(_get_shape(ov), n_dofs)
        )


@TRACER_REGISTRY.register(
    "custom_vjp_call",
    "custom_jvp_call",
    "pure_callback",
    "io_callback",
)
def custom_vjp_jvp_call(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Handler for JAX custom_vjp/custom_jvp calls and host callbacks
    (``pure_callback`` / ``io_callback``).

    These all represent general nonlinear black-box functions whose internals are
    opaque to the tracer (a host callback has no traceable jaxpr at all; an external
    RVE solver typically lives inside one wrapped in ``custom_jvp``). We therefore
    propagate the union of input dependencies to each output element and record the
    self-couplings of the active inputs, coupling all inputs together even if the
    internals happen to be linear. This is conservative (over-approximate): it can
    never miss a coupling, which is the correct failure mode for an unseen black box.
    Note the plain ``fallback`` handler does NOT record couplings, so an external
    callback that landed there would silently drop them.
    """
    in_d = [state.get(v) for v in eqn.invars]

    # 1. Accumulate all input active columns to record couplings
    cols_active = np.zeros(state.n_dofs, dtype=bool)
    has_active = False
    for d in in_d:
        if isinstance(d, SparseDepSet) and d.dep.shape[0] > 0:
            cols_active[d.dep.indices] = True
            has_active = True

    if not has_active:
        total = SparseDepSet.empty((), state.n_dofs)
    else:
        reduced = sps.csr_matrix(cols_active.reshape(1, -1))
        total = SparseDepSet(reduced, ())
        # Record couplings for the active input DOFs
        acc.record_dep(total.dep, trial_test_split)

    # 2. Set the dependency set for all output variables
    for ovar in eqn.outvars:
        oshp = _get_shape(ovar)
        stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
        state.set(ovar, SparseDepSet(stacked_dep, oshp))


@TRACER_REGISTRY.register("ffi_call")
def ffi_call(
    eqn: JaxprEqn,
    state: TraceState,
    acc: "CouplingAccumulator",
    trial_test_split: int | None,
) -> None:
    """Handler for ``jax.ffi.ffi_call`` (external XLA custom-call solvers).

    Default: an opaque black box -> couple all active inputs (same conservative rule
    as ``custom_vjp_jvp_call``). Note a *vmapped* ffi_call is a SINGLE batched call
    over the whole leading axis, so this default over-approximates it to a dense block.

    Opt-in (``register_elementwise_ffi(target)``): treat the call as elementwise along
    the leading (vmap) axis -- couple inputs *per slice* (block-diagonal across that
    axis) so a per-quad-point external solver vmapped over quad points keeps its sparse
    pattern, exactly as it would via ``lax.map``/``scan``.
    """
    in_d = [state.get(v) for v in eqn.invars]
    target = eqn.params.get("target_name")

    # Decide whether the block-diagonal (elementwise) rule applies: the target must be
    # registered AND every operand must share a common leading (batch) axis.
    lead = None
    if target in _ELEMENTWISE_FFI_TARGETS:
        shapes = [d.shape for d in in_d] + [_get_shape(ov) for ov in eqn.outvars]
        if shapes and all(len(s) >= 1 for s in shapes):
            leads = {s[0] for s in shapes}
            if len(leads) == 1:
                lead = leads.pop()

    if not lead:
        # Conservative default: opaque couple-all over every active input.
        custom_vjp_jvp_call(eqn, state, acc, trial_test_split)
        return

    B = lead
    # Per-slice union of input columns, recorded as an independent coupling block.
    slice_rows: list[sps.csr_matrix] = []
    for b in range(B):
        cols_active = np.zeros(state.n_dofs, dtype=bool)
        for d in in_d:
            nrows = d.dep.shape[0]
            if nrows == 0:
                continue
            core = nrows // B
            blk = d.dep[b * core : (b + 1) * core]
            if blk.nnz:
                cols_active[blk.indices] = True
        row = sps.csr_matrix(cols_active.reshape(1, -1))
        if row.nnz:
            acc.record_dep(row, trial_test_split)
        slice_rows.append(row)

    # Assign each output: slice b broadcast over its core (per-slice) elements.
    for ovar in eqn.outvars:
        oshp = _get_shape(ovar)
        core_o = int(np.prod(oshp)) // B
        blocks = [_broadcast_single_row(slice_rows[b], core_o) for b in range(B)]
        stacked = sps.vstack(blocks).tocsr()
        state.set(ovar, SparseDepSet(stacked, oshp))
