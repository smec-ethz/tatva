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

import builtins
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeAlias

import jax
import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal
from numpy.typing import NDArray

from tatva.sparse.tracer.common import _get_shape

if TYPE_CHECKING:
    from tatva.sparse.tracer.handlers import PrimitiveHandler

P = ParamSpec("P")


@dataclass(frozen=True)
class TraceExecution:
    """Per-equation execution controls supplied to primitive handlers."""

    trial_test_split: int | None = None
    needs_concrete: bool = False


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

    def copy(self) -> SparseDepSet:
        return SparseDepSet(self.dep.copy(), self.shape)

    def reshape(self, *ns) -> SparseDepSet:
        return SparseDepSet(self.dep, ns)

    @classmethod
    def empty(cls, shape: tuple, n_dofs: int) -> SparseDepSet:
        """Create a zero-dependency SparseDepSet of shape (*shape, n_dofs)."""
        size = int(np.prod(shape))
        dep = sps.csr_matrix((size, n_dofs), dtype=bool)
        return cls(dep, shape)

    @classmethod
    def singletons(cls, n_dofs: int) -> SparseDepSet:
        """Create an identity-seeded SparseDepSet of shape (n_dofs,) where element i
        depends only on DOF i."""
        dep = sps.eye(n_dofs, format="csr", dtype=bool)
        return cls(dep, (n_dofs,))

    def total_union(self) -> SparseDepSet:
        """OR all dependency sets in this array into a single 1D vector of shape ()."""
        if self.dep.shape[0] == 0:
            return SparseDepSet.empty((), self.dep.shape[1])
        reduced = sps.csr_matrix(self.dep.sum(axis=0).astype(bool))
        return SparseDepSet(reduced, ())

    def broadcast_to(
        self,
        S_out: tuple,
        broadcast_dimensions: tuple[int, ...] | Sequence[int] | None = None,
    ) -> SparseDepSet:
        """Broadcast this dep-array to a new logical shape S_out."""
        S_in = self.shape
        if S_in == S_out:
            return self
        src_indices = np.arange(int(np.prod(S_in)))
        if broadcast_dimensions is not None:
            newshape = [1] * len(S_out)
            for i, b in enumerate(broadcast_dimensions):
                newshape[b] = S_in[i] if len(S_in) > 0 else 1
            src_indices = src_indices.reshape(newshape)
        else:
            src_indices = src_indices.reshape(S_in)
        mapped_indices = np.broadcast_to(src_indices, S_out).ravel()
        return SparseDepSet(self.dep[mapped_indices], S_out)

    def record_couplings(
        self,
        acc: CouplingAccumulator,
        trial_test_split: int | None = None,
    ) -> None:
        """Record all active self- and cross-coupling variable pairs via the global accumulator."""
        acc.record_dep(self.dep, trial_test_split)


@dataclass(frozen=True)
class ConcreteJaxpr:
    closed_jaxpr: ClosedJaxpr
    flat_args: list[Any]
    pytree_def: jax.tree_util.PyTreeDef

    @classmethod
    def from_fn(cls, fn: Callable[P, Array], *args: P.args, **kwargs: P.kwargs) -> Self:
        closed = jax.make_jaxpr(fn)(*args, **kwargs)
        flat_args, pytree_def = jax.tree_util.tree_flatten((args, kwargs))
        return cls(closed, flat_args, pytree_def)

    def tree_unflatten(self, flat_outs: Sequence[Any]) -> Any:
        return jax.tree_util.tree_unflatten(self.pytree_def, flat_outs)

    @property
    def jaxpr(self) -> Jaxpr:
        return self.closed_jaxpr.jaxpr

    @property
    def consts(self) -> tuple[Any, ...]:
        return self.closed_jaxpr.consts

    @property
    def constvars(self) -> list[Any]:
        return self.closed_jaxpr.constvars

    @property
    def invars(self) -> list[Any]:
        return self.closed_jaxpr.invars

    @property
    def outvars(self) -> list[Any]:
        return self.closed_jaxpr.outvars


# A bound equation tuple: (eqn, handler, is_active, needs_concrete)
BoundEqn: TypeAlias = tuple[JaxprEqn, "PrimitiveHandler", bool, bool]

# Sub-jaxpr analysis result: (sub_active_set, sub_index_set, sub_bound_eqns)
SubEqnInfo: TypeAlias = tuple[set[int], set[int], list[BoundEqn]]

# sub_info dictionary mapping id(eqn) to SubEqnInfo (for jit/scan/map) or list[SubEqnInfo] (for cond branches)
SubInfoDict: TypeAlias = dict[int, SubEqnInfo | list[SubEqnInfo]]


class TraceState:
    """Encapsulates the state of dependency propagation and concrete value routing during tracing."""

    def __init__(
        self,
        n_dofs: int,
        active_ids: builtins.set[int],
        sub_info: SubInfoDict | None = None,
        nonlinear_ids: builtins.set[int] | None = None,
    ):
        """
        Args:
            n_dofs: total number of DOFs (size of the input variable)
            active_ids: set of variable IDs that are currently active (feed into nonlinear
                primitives)
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
        self.dep_of: dict[int, SparseDepSet] = {}
        self.val_of: dict[int, np.ndarray] = {}
        self.sub_info: SubInfoDict = sub_info if sub_info is not None else {}
        self.nonlinear_ids = nonlinear_ids if nonlinear_ids is not None else set()

    def attach_concrete_values(
        self, jaxpr: ClosedJaxpr, concrete_vals: list[NDArray]
    ) -> None:
        """Store concrete input and constant values for routing operations."""
        for invar, arg_val in zip(jaxpr.invars, concrete_vals):
            self.val_of[id(invar)] = np.asarray(arg_val)

        for v, c in zip(jaxpr.constvars, jaxpr.consts):
            self.val_of[id(v)] = np.asarray(c)

    def seed_input_dependencies(self, jaxpr: ClosedJaxpr) -> None:
        """Seed dependency sets for JAXpr inputs and constants.

        The first input is the traced DOF vector and receives singleton dependencies;
        all other inputs and constants are independent of it.
        """
        u_seed = SparseDepSet.singletons(self.n_dofs)
        if jaxpr.invars:
            self.set(
                jaxpr.invars[0], u_seed
            )  # seed the input variable with singleton dep-sets
            for v in jaxpr.invars[1:]:
                self.set(
                    v, SparseDepSet.empty(_get_shape(v), self.n_dofs)
                )  # seed other input variables (e.g., static args) with empty dep-sets

        for v in jaxpr.constvars:
            self.set(v, SparseDepSet.empty(_get_shape(v), self.n_dofs))

    def is_nonlinear(self, var) -> bool:
        """Whether ``var`` is a non-affine (nonlinear) function of the input ``u``."""
        return id(var) in self.nonlinear_ids

    def is_inactive(self, var) -> bool:
        """Whether ``var`` is inactive (does not depend on the input ``u``)."""
        return self.get(var).dep.nnz == 0

    def is_scalar(self, var) -> bool:
        """Whether ``var`` is a scalar (shape () or (1,))."""
        return int(np.prod(_get_shape(var))) == 1

    def mark_nonlinear(self, eqn: JaxprEqn, handler: PrimitiveHandler) -> None:
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
            out_nl = handler.introduces_nonlinearity(eqn, invar_active)
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

    def run_bound_eqns(
        self,
        bound_eqns: list[tuple[JaxprEqn, PrimitiveHandler, bool, bool]],
        acc: CouplingAccumulator,
        trial_test_split: int | None = None,
    ) -> None:
        """Execute a resolved list of equations (used by main tracer and sub-jaxprs)."""
        for eqn, handler, is_active, needs_concrete in bound_eqns:
            ovars = eqn.outvars

            if ovars and not is_active:
                for v in ovars:
                    self.set(v, SparseDepSet.empty(_get_shape(v), self.n_dofs))
                if needs_concrete:
                    in_vals = [self.get_val(v) for v in eqn.invars]
                    cv = handler.safe_eval_concrete(eqn.primitive, in_vals, eqn.params)
                    if cv is not None:
                        self.val_of[id(ovars[0])] = cv
                continue

            handler.propagate_deps(
                eqn,
                self,
                acc,
                TraceExecution(
                    trial_test_split=trial_test_split,
                    needs_concrete=needs_concrete,
                ),
            )
            self.mark_nonlinear(eqn, handler)

            if ovars and needs_concrete:
                if eqn.primitive.name in ("pjit", "jit", "scan", "map", "remat2"):
                    continue  # sub-jaxpr concrete evaluation is handled in the handler
                in_vals = [self.get_val(v) for v in eqn.invars]
                cv = handler.safe_eval_concrete(eqn.primitive, in_vals, eqn.params)
                if cv is not None:
                    self.val_of[id(ovars[0])] = cv
