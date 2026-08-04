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

from collections.abc import Callable
from typing import Any, ParamSpec

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn, Literal

from tatva.sparse.tracer.common import (
    _DENSE_LINALG,
    _NONLINEAR_BINARY,
    _NONLINEAR_UNARY,
    _get_shape,
    _prim_introduces_nonlinearity,
    _subjaxpr_and_consts,
)

P = ParamSpec("P")


class CouplingAccumulator:
    """Accumulates Hessian coupling pairs as numpy-array chunks; fingerprints dep matrices
    to skip redundant recordings."""

    def __init__(self, n_dofs: int):
        self.n_dofs = n_dofs
        self._row_chunks: list[np.ndarray] = []
        self._col_chunks: list[np.ndarray] = []
        self._seen_fingerprints: set[int] = set()

    def add_coords(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Append a chunk of coordinate pairs without converting to Python lists."""
        if rows.size == 0:
            return
        self._row_chunks.append(np.asarray(rows))
        self._col_chunks.append(np.asarray(cols))

    def record_dep(
        self, dep: sps.csr_matrix, trial_test_split: int | None = None
    ) -> None:
        """Compute dep.T @ dep couplings; skip if an identical dep structure has already
        been recorded."""
        if dep.nnz == 0:
            return
        # Fingerprint: identical (indptr, indices) + split → identical couplings
        fp = hash(
            (
                dep.indptr.tobytes(),
                dep.indices.tobytes(),
                trial_test_split,
                dep.shape[1],
            )
        )
        if fp in self._seen_fingerprints:
            return
        self._seen_fingerprints.add(fp)
        if trial_test_split is not None:
            # Only trial<->test cross couplings survive, so compute just that block
            # (trial_part.T @ test_part) instead of the full dep.T @ dep over all
            # columns followed by masking — avoids the discarded self-blocks.
            s = trial_test_split
            trial_part = dep[:, :s]
            test_part = dep[:, s:]
            cross = (trial_part.T @ test_part).tocsr()
            r, c = cross.nonzero()
            c = c + s
            self.add_coords(r, c)
            self.add_coords(c, r)
        else:
            P = (dep.T @ dep).tocsr()
            r, c = P.nonzero()
            self.add_coords(r, c)

    def finalize(self) -> sps.csr_matrix:
        """Build the final binary CSR sparsity pattern from the accumulated chunks."""
        if not self._row_chunks:
            return sps.csr_matrix((self.n_dofs, self.n_dofs), dtype=np.int8)
        rows = np.concatenate(self._row_chunks)
        cols = np.concatenate(self._col_chunks)
        data = np.ones(rows.shape[0], dtype=np.int8)
        pat = sps.csr_matrix((data, (rows, cols)), shape=(self.n_dofs, self.n_dofs))
        if hasattr(pat, "sum_duplicates"):
            pat.sum_duplicates()
        pat.data[:] = 1
        return pat


class SparseDepSet:
    def __init__(self, dep: sps.csr_matrix, shape: tuple):
        """
        Represents the dependency set of an array as a sparse boolean matrix.

        Args:
            dep: sps.csr_matrix of shape (prod(shape), n_dofs) with boolean data
            shape: logical shape of the array this dep-set corresponds to (e.g., (3,4) for
                a 3×4 array); the dep-array is always flattened in row-major order
        """
        self.dep = dep  # sps.csr_matrix of shape (prod(shape), n_dofs)
        self.shape = tuple(shape)

    def copy(self) -> "SparseDepSet":
        return SparseDepSet(self.dep.copy(), self.shape)

    def reshape(self, *ns) -> "SparseDepSet":
        return SparseDepSet(self.dep, ns)

    @classmethod
    def empty(cls, shape: tuple, n_dofs: int) -> "SparseDepSet":
        """Create a zero-dependency SparseDepSet of shape (*shape, n_dofs)."""
        size = int(np.prod(shape))
        dep = sps.csr_matrix((size, n_dofs), dtype=bool)
        return cls(dep, shape)

    @classmethod
    def singletons(cls, n_dofs: int) -> "SparseDepSet":
        """Create an identity-seeded SparseDepSet of shape (n_dofs,) where element i
        depends only on DOF i."""
        dep = sps.eye(n_dofs, format="csr", dtype=bool)
        return cls(dep, (n_dofs,))

    def total_union(self) -> "SparseDepSet":
        """OR all dependency sets in this array into a single 1D vector of shape ()."""
        if self.dep.shape[0] == 0:
            return SparseDepSet.empty((), self.dep.shape[1])
        reduced = sps.csr_matrix(self.dep.sum(axis=0).astype(bool))
        return SparseDepSet(reduced, ())

    def broadcast_to(self, S_out: tuple) -> "SparseDepSet":
        """Broadcast this dep-array to a new logical shape S_out."""
        S_in = self.shape
        if S_in == S_out:
            return self
        src_indices = np.arange(int(np.prod(S_in))).reshape(S_in)
        mapped_indices = np.broadcast_to(src_indices, S_out).ravel()
        return SparseDepSet(self.dep[mapped_indices], S_out)

    def record_couplings(
        self,
        acc: "CouplingAccumulator",
        trial_test_split: int | None = None,
    ) -> None:
        """Record all active self- and cross-coupling variable pairs via the global accumulator."""
        acc.record_dep(self.dep, trial_test_split)


class TraceState:
    """Encapsulates the state of dependency propagation and concrete value routing during tracing."""

    def __init__(
        self,
        n_dofs: int,
        active_ids: set[int],
        tags: dict | None = None,
        sub_info: dict | None = None,
        nonlinear_ids: set[int] | None = None,
    ):
        """
        Args:
            n_dofs: total number of DOFs (size of the input variable)
            active_ids: set of variable IDs that are currently active (feed into nonlinear
                primitives)
            tags: dict mapping variable IDs to their current tag (0=inactive,
                1=trial-only, 2=test-only, 3=both); used for trial/test splitting
            sub_info: dict to store sub-jaxpr analysis results for nested jits; maps eqn
                IDs to (active_set, resolved_eqns)
            nonlinear_ids: set of variable IDs that are a *non-affine* (nonlinear) function
                of the input ``u``. A variable enters this set when a coupling-recording
                (second-order) primitive touches its computation from ``u``. Shared across
                nested sub-states so operand nonlinearity propagates across jit/cond/scan
                boundaries. Used to gate the self-couplings recorded by bilinear
                contractions (``dot_general``): an affine operand has zero second
                derivative, so its self outer-product is not a real Hessian block.
        """
        self.n_dofs = n_dofs
        self.active_ids = active_ids
        self.tags = tags if tags is not None else {}
        self.dep_of: dict[int, SparseDepSet] = {}
        self.val_of: dict[int, np.ndarray] = {}
        self.sub_info = sub_info if sub_info is not None else {}
        self.nonlinear_ids = nonlinear_ids if nonlinear_ids is not None else set()

    def is_nonlinear(self, var) -> bool:
        """Whether ``var`` is a non-affine (nonlinear) function of the input ``u``."""
        return id(var) in self.nonlinear_ids

    def mark_nonlinear(self, eqn: JaxprEqn) -> None:
        """Flag ``eqn``'s outputs as nonlinear if a coupling-recording primitive touches
        their computation, or if any input is already nonlinear.

        Conservative: it only ever *adds* nonlinearity (over-flagging keeps a self-coupling
        that is at worst redundant), so it can never cause a bilinear contraction to drop a
        real second-order block. Higher-order primitives (jit/cond/scan) propagate their
        sub-jaxpr's nonlinearity explicitly in their handlers, so they are skipped here.
        """
        if not eqn.outvars or eqn.primitive.name in (
            "pjit",
            "jit",
            "scan",
            "map",
            "remat2",
            "cond",
            "switch",
            "while",
        ):
            return
        out_nl = any(id(v) in self.nonlinear_ids for v in eqn.invars)
        if not out_nl:
            invar_active = [self.get(v).dep.nnz > 0 for v in eqn.invars]
            out_nl = _prim_introduces_nonlinearity(eqn, invar_active)
        if out_nl:
            for ov in eqn.outvars:
                self.nonlinear_ids.add(id(ov))

    def set(self, var, d: SparseDepSet) -> None:
        """Associate dep-set with a JAX variable."""
        self.dep_of[id(var)] = d

    def get(self, var) -> SparseDepSet:
        """Get the dep-set of a JAX variable (or return an empty one for literals/unregistered)."""
        if isinstance(var, Literal):
            return SparseDepSet.empty(_get_shape(var), self.n_dofs)
        return self.dep_of.get(
            id(var), SparseDepSet.empty(_get_shape(var), self.n_dofs)
        )

    def get_val(self, var) -> np.ndarray | None:
        """Get the concrete value of a JAX variable if known."""
        if isinstance(var, Literal):
            return np.asarray(var.val)
        return self.val_of.get(id(var))


def _analyze_and_resolve_jaxpr(
    jaxpr,
    trial_test_split: int | None,
    tags: dict[int, int],
    main_input_id: int | None,
    sub_info: dict[int, Any],
) -> tuple[list[tuple[Any, Callable, bool, Any]], set[int], set[int]]:
    """
    Performs the forward pass of a unified JAXpr analysis traversal:
    - Propagates tags (forward)
    - Identifies nonlinear active equations (forward)
    - Resolves registered handlers (forward)
    - Collects index variable IDs for indexing operations (forward)

    Args:
        jaxpr: the JAXpr to analyze
        trial_test_split: if not None, the DOF index at which to split trial vs test
            variables for nonlinear interaction detection
        tags: a dict mapping variable IDs to their current tag (0=inactive, 1=trial-only,
            2=test-only, 3=both)
        main_input_id: the variable ID of the main input (e.g., the trial function) to
            seed with tags; used for trial/test splitting
        sub_info: a dict to store sub-jaxpr analysis results for nested jits; maps eqn IDs
            to (active_set, index_set, resolved_eqns)

    Returns:
        A list of forward data tuples: (eqn, handler, is_nonlinear, sub_res)
        The initial set of active variable IDs seeded from the outputs.
        The initial set of index variable IDs seeded from indexing operations.
    """
    from tatva.sparse.tracer.handlers import get_handler

    forward_data = []
    active_set = {id(v) for v in jaxpr.outvars}
    index_set = set()

    for eqn in jaxpr.eqns:
        p = eqn.primitive.name

        # Identify indexing primitives and seed index_set with their index operands
        if p in (
            "gather",
            "scatter",
            "scatter-add",
            "scatter-sub",
            "scatter-mul",
            "scatter-min",
            "scatter-max",
        ):
            if len(eqn.invars) > 1:
                index_set.add(id(eqn.invars[1]))
        elif p == "dynamic_slice":
            for v in eqn.invars[1:]:
                index_set.add(id(v))
        elif p == "dynamic_update_slice":
            for v in eqn.invars[2:]:
                index_set.add(id(v))
        elif p == "select_n" and len(eqn.invars) > 0:
            index_set.add(id(eqn.invars[0]))

        # propagate tags & JIT recursion
        sub_res = None
        if trial_test_split is not None:
            if (
                p == "slice"
                and main_input_id is not None
                and id(eqn.invars[0]) == main_input_id
            ):
                par = eqn.params
                start = par["start_indices"][0]
                limit = par["limit_indices"][0]
                if start == 0 and limit <= trial_test_split:
                    mask = 1
                elif start >= trial_test_split:
                    mask = 2
                else:
                    mask = 3
            elif p in ("pjit", "jit", "scan", "map", "remat2"):
                sub_jaxpr, _ = _subjaxpr_and_consts(eqn)
                # Map input tags to sub invars
                for pv, sv in zip(eqn.invars, sub_jaxpr.invars):
                    tags[id(sv)] = tags.get(id(pv), 0)

                # Recursively analyze sub-jaxpr
                sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                    sub_jaxpr,
                    trial_test_split,
                    tags,
                    None,
                    sub_info,
                )
                sub_res = (sub_active, sub_index_set, sub_eqns)

                for pv, sv in zip(eqn.invars, sub_jaxpr.invars):
                    if id(sv) in sub_index_set:
                        index_set.add(id(pv))

                # Map output tags back
                mask = 0
                for pv, sv in zip(eqn.outvars, sub_jaxpr.outvars):
                    tags[id(pv)] = tags.get(id(sv), 0)
                    mask |= tags[id(pv)]
            elif p == "cond":
                # cond carries one jaxpr per branch; invars[0] is the predicate and
                # invars[1:] are the operands passed to every branch.
                operands = eqn.invars[1:]
                sub_res = []
                mask = 0
                for branch in eqn.params["branches"]:
                    bj = branch.jaxpr
                    for pv, sv in zip(operands, bj.invars):
                        tags[id(sv)] = tags.get(id(pv), 0)
                    sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                        bj, trial_test_split, tags, None, sub_info
                    )
                    sub_res.append((sub_active, sub_index_set, sub_eqns))
                    for sv in bj.outvars:
                        mask |= tags.get(id(sv), 0)
                    for pv, sv in zip(operands, bj.invars):
                        if id(sv) in sub_index_set:
                            index_set.add(id(pv))
            else:
                mask = 0
                for v in eqn.invars:
                    mask |= tags.get(id(v), 0)

            for v in eqn.outvars:
                tags[id(v)] = mask
        else:
            if p in ("pjit", "jit", "scan", "map", "remat2"):
                sub_jaxpr, _ = _subjaxpr_and_consts(eqn)
                sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                    sub_jaxpr,
                    None,
                    tags,
                    None,
                    sub_info,
                )
                sub_res = (sub_active, sub_index_set, sub_eqns)
                for pv, sv in zip(eqn.invars, sub_jaxpr.invars):
                    if id(sv) in sub_index_set:
                        index_set.add(id(pv))
            elif p == "cond":
                sub_res = []
                operands = eqn.invars[1:]
                for branch in eqn.params["branches"]:
                    sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                        branch.jaxpr, None, tags, None, sub_info
                    )
                    sub_res.append((sub_active, sub_index_set, sub_eqns))
                    for pv, sv in zip(operands, branch.jaxpr.invars):
                        if id(sv) in sub_index_set:
                            index_set.add(id(pv))

        # check if the equation is a nonlinear primitive
        is_nonlinear = False
        if trial_test_split is not None:
            if p in _NONLINEAR_UNARY:
                if tags.get(id(eqn.invars[0]), 0) == 3:
                    is_nonlinear = True
            elif p in _NONLINEAR_BINARY:
                combined_mask = 0
                for v in eqn.invars:
                    combined_mask |= tags.get(id(v), 0)
                if combined_mask == 3:
                    is_nonlinear = True
            elif p == "integer_pow":
                exponent = eqn.params.get("y", 0)
                if exponent >= 2 or exponent <= -1:
                    if tags.get(id(eqn.invars[0]), 0) == 3:
                        is_nonlinear = True
            elif (
                p
                in (
                    "dot_general",
                    "scatter-mul",
                    "custom_vjp_call",
                    "custom_jvp_call",
                    "pure_callback",
                    "io_callback",
                    "ffi_call",
                )
                or p in _DENSE_LINALG
            ):
                combined_mask = 0
                for v in eqn.invars:
                    combined_mask |= tags.get(id(v), 0)
                if combined_mask == 3:
                    is_nonlinear = True
            elif p in ("pjit", "jit"):
                pass
        else:
            if p in _NONLINEAR_UNARY or p in _NONLINEAR_BINARY:
                is_nonlinear = True
            elif p == "integer_pow":
                exponent = eqn.params.get("y", 0)
                if exponent >= 2 or exponent <= -1:
                    is_nonlinear = True
            elif (
                p
                in (
                    "dot_general",
                    "scatter-mul",
                    "custom_vjp_call",
                    "custom_jvp_call",
                    "pure_callback",
                    "io_callback",
                    "ffi_call",
                )
                or p in _DENSE_LINALG
            ):
                is_nonlinear = True

        # Seed active variables if it is a nonlinear primitive
        if is_nonlinear:
            for v in eqn.invars:
                active_set.add(id(v))

        # resolve registered handler
        handler = get_handler(p)

        forward_data.append(
            (
                eqn,
                handler,
                is_nonlinear,
                sub_res
                if p in ("pjit", "jit", "scan", "map", "remat2", "cond")
                else None,
            )
        )

    return forward_data, active_set, index_set


def _propagate_active_backward(
    forward_data: list[tuple[Any, Callable, bool, Any]],
    active_set: set[int],
    index_set: set[int],
    sub_info: dict[int, Any],
) -> list[tuple[Any, Callable, bool, bool]]:
    """
    Performs the backward pass of a unified JAXpr analysis traversal:
    Propagates the active state and index state backwards through the resolved equations list.

    Args:
        forward_data: the list of forward data tuples (eqn, handler, is_nonlinear,
            sub_res) from the forward pass
        active_set: the initial set of active variable IDs seeded from the outputs
        index_set: the set of variable IDs required for concrete indexing operations
        sub_info: the dict storing sub-jaxpr analysis results for nested jits; maps eqn
            IDs to (active_set, index_set, resolved_eqns)

    Returns:
        A list of tuples (eqn, handler, is_active, needs_concrete) where is_active
        indicates whether this equation is on an active path, and needs_concrete
        indicates whether this equation's output must be evaluated concretely for indexing.
    """
    pruned_eqns = []
    for eqn, handler, is_nonlinear, sub_res in reversed(forward_data):
        p = eqn.primitive.name
        is_active = False
        needs_concrete = False

        if p in ("pjit", "jit", "scan", "map", "remat2"):
            outvars_active = any(id(v) in active_set for v in eqn.outvars)
            outvars_index = any(id(v) in index_set for v in eqn.outvars)
            if outvars_active or outvars_index:
                sub_active_set, sub_index_set, sub_eqns = sub_res
                sub, _ = _subjaxpr_and_consts(eqn)

                # Map active/index outvars to sub outvars
                for pv, sv in zip(eqn.outvars, sub.outvars):
                    if id(pv) in active_set:
                        sub_active_set.add(id(sv))
                    if id(pv) in index_set:
                        sub_index_set.add(id(sv))

                # Recursively propagate active state backward in sub-jaxpr
                sub_eqns_pruned = _propagate_active_backward(
                    sub_eqns, sub_active_set, sub_index_set, sub_info
                )

                if outvars_active:
                    is_active = True
                if outvars_index or any(nc for _, _, _, nc in sub_eqns_pruned):
                    needs_concrete = True

                # Map active sub invars to parent invars
                for pv, sv in zip(eqn.invars, sub.invars):
                    if id(sv) in sub_active_set:
                        active_set.add(id(pv))
                    if id(sv) in sub_index_set:
                        index_set.add(id(pv))

                # Store sub-info for this jit equation
                sub_info[id(eqn)] = (sub_active_set, sub_index_set, sub_eqns_pruned)
        elif p == "cond":
            outvars_active = any(id(v) in active_set for v in eqn.outvars)
            outvars_index = any(id(v) in index_set for v in eqn.outvars)
            if outvars_active or outvars_index:
                operands = eqn.invars[1:]
                pruned_branches = []
                branch_has_concrete = False
                for (sub_active_set, sub_index_set, sub_eqns), branch in zip(
                    sub_res, eqn.params["branches"]
                ):
                    sub = branch.jaxpr

                    # Map active outvars to each branch's outvars
                    for pv, sv in zip(eqn.outvars, sub.outvars):
                        if id(pv) in active_set:
                            sub_active_set.add(id(sv))
                        if id(pv) in index_set:
                            sub_index_set.add(id(sv))

                    sub_eqns_pruned = _propagate_active_backward(
                        sub_eqns, sub_active_set, sub_index_set, sub_info
                    )
                    if any(nc for _, _, _, nc in sub_eqns_pruned):
                        branch_has_concrete = True

                    # Map active branch invars back to the cond operands (invars[1:])
                    for pv, sv in zip(operands, sub.invars):
                        if id(sv) in sub_active_set:
                            active_set.add(id(pv))
                        if id(sv) in sub_index_set:
                            index_set.add(id(pv))

                    pruned_branches.append(
                        (sub_active_set, sub_index_set, sub_eqns_pruned)
                    )

                if outvars_active:
                    is_active = True
                if outvars_index or branch_has_concrete:
                    needs_concrete = True

                sub_info[id(eqn)] = pruned_branches
        else:
            if is_nonlinear or (
                eqn.outvars and any(id(v) in active_set for v in eqn.outvars)
            ):
                is_active = True
                for v in eqn.invars:
                    active_set.add(id(v))

            if eqn.outvars and any(id(v) in index_set for v in eqn.outvars):
                needs_concrete = True
                for v in eqn.invars:
                    index_set.add(id(v))

        pruned_eqns.append((eqn, handler, is_active, needs_concrete))

    pruned_eqns.reverse()
    return pruned_eqns
