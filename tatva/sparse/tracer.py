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

import inspect
from typing import (
    Any,
    Callable,
    Concatenate,
    Dict,
    List,
    ParamSpec,
    Sequence,
    Set,
    Tuple,
)

import jax
import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

P = ParamSpec("P")


def _get_shape(var: Var | Literal) -> Tuple[int, ...]:
    """Helper to safely retrieve shape from a JAX Var/Literal abstract value (satisfying static type checkers)."""
    return getattr(var.aval, "shape", ())


def _subjaxpr_and_consts(eqn) -> Tuple[Jaxpr, Sequence]:
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


class CouplingAccumulator:
    """Accumulates Hessian coupling pairs as numpy-array chunks; fingerprints dep matrices to skip redundant recordings."""

    def __init__(self, n_dofs: int):
        self.n_dofs = n_dofs
        self._row_chunks: List[np.ndarray] = []
        self._col_chunks: List[np.ndarray] = []
        self._seen_fingerprints: Set[int] = set()

    def add_coords(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Append a chunk of coordinate pairs without converting to Python lists."""
        if rows.size == 0:
            return
        self._row_chunks.append(np.asarray(rows))
        self._col_chunks.append(np.asarray(cols))

    def record_dep(
        self, dep: sps.csr_matrix, trial_test_split: int | None = None
    ) -> None:
        """Compute dep.T @ dep couplings; skip if an identical dep structure has already been recorded."""
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


# ---------------------------------------------------------------------------
# primitive classification
# ---------------------------------------------------------------------------

_NONLINEAR_UNARY = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "log2",
        "sqrt",
        "rsqrt",
        "cbrt",
        "tanh",
        "sinh",
        "cosh",
        "atanh",
        "asinh",
        "acosh",
        "erf",
        "erfc",
        "erfinv",
        "lgamma",
        "digamma",
        "logistic",
    }
)

_NONLINEAR_BINARY = frozenset(
    {
        "mul",
        "div",
        "rem",
        "pow",
        "atan2",
        "igamma",
        "igammac",
        "nextafter",
        "complex",
    }
)


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


# FFI targets whose vmapped (batched) call is elementwise along the leading (vmap) axis,
# i.e. output[i] depends only on input[i] -- e.g. a per-quad-point external constitutive
# solver vmapped over quad points. Declaring a target here lets the tracer recover the
# sparse (block-diagonal) coupling from the single batched ``ffi_call`` instead of the
# conservative dense couple-all. See ``register_elementwise_ffi``.
_ELEMENTWISE_FFI_TARGETS: Set[str] = set()


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


class SparseDepSet:
    def __init__(self, dep: sps.csr_matrix, shape: tuple):
        """
        Represents the dependency set of an array as a sparse boolean matrix.

        Args:
            dep: sps.csr_matrix of shape (prod(shape), n_dofs) with boolean data
            shape: logical shape of the array this dep-set corresponds to (e.g., (3,4) for a 3×4 array);
                   the dep-array is always flattened in row-major order
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
        """Create an identity-seeded SparseDepSet of shape (n_dofs,) where element i depends only on DOF i."""
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
        active_ids: Set[int],
        tags: dict | None = None,
        sub_info: dict | None = None,
    ):
        """
        Args:
            n_dofs: total number of DOFs (size of the input variable)
            active_ids: set of variable IDs that are currently active (feed into nonlinear primitives)
            tags: dict mapping variable IDs to their current tag (0=inactive, 1=trial-only, 2=test-only, 3=both); used for trial/test splitting
            sub_info: dict to store sub-jaxpr analysis results for nested jits; maps eqn IDs to (active_set, resolved_eqns)
        """
        self.n_dofs = n_dofs
        self.active_ids = active_ids
        self.tags = tags if tags is not None else {}
        self.dep_of: Dict[int, SparseDepSet] = {}
        self.val_of: Dict[int, np.ndarray] = {}
        self.sub_info = sub_info if sub_info is not None else {}

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
    tags: Dict[int, int],
    main_input_id: int | None,
    sub_info: Dict[int, Any],
) -> Tuple[List[Tuple[Any, Callable, bool, Any]], Set[int]]:
    """
    Performs the forward pass of a unified JAXpr analysis traversal:
    - Propagates tags (forward)
    - Identifies nonlinear active equations (forward)
    - Resolves registered handlers (forward)

    Args:
        jaxpr: the JAXpr to analyze
        trial_test_split: if not None, the DOF index at which to split trial vs test variables for nonlinear interaction detection
        tags: a dict mapping variable IDs to their current tag (0=inactive, 1=trial-only, 2=test-only, 3=both)
        main_input_id: the variable ID of the main input (e.g., the trial function) to seed with tags; used for trial/test splitting
        sub_info: a dict to store sub-jaxpr analysis results for nested jits; maps eqn IDs to (active_set, resolved_eqns)

    Returns:
        A list of forward data tuples: (eqn, handler, is_nonlinear, sub_res)
        The initial set of active variable IDs seeded from the outputs.
    """
    forward_data = []
    active_set = {id(v) for v in jaxpr.outvars}

    for eqn in jaxpr.eqns:
        p = eqn.primitive.name

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
                sub_eqns, sub_active = _analyze_and_resolve_jaxpr(
                    sub_jaxpr,
                    trial_test_split,
                    tags,
                    None,
                    sub_info,
                )
                sub_res = (sub_active, sub_eqns)

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
                    sub_eqns, sub_active = _analyze_and_resolve_jaxpr(
                        bj, trial_test_split, tags, None, sub_info
                    )
                    sub_res.append((sub_active, sub_eqns))
                    for sv in bj.outvars:
                        mask |= tags.get(id(sv), 0)
            else:
                mask = 0
                for v in eqn.invars:
                    mask |= tags.get(id(v), 0)

            for v in eqn.outvars:
                tags[id(v)] = mask
        else:
            if p in ("pjit", "jit", "scan", "map", "remat2"):
                sub_jaxpr, _ = _subjaxpr_and_consts(eqn)
                sub_eqns, sub_active = _analyze_and_resolve_jaxpr(
                    sub_jaxpr,
                    None,
                    tags,
                    None,
                    sub_info,
                )
                sub_res = (sub_active, sub_eqns)
            elif p == "cond":
                sub_res = []
                for branch in eqn.params["branches"]:
                    sub_eqns, sub_active = _analyze_and_resolve_jaxpr(
                        branch.jaxpr, None, tags, None, sub_info
                    )
                    sub_res.append((sub_active, sub_eqns))

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
            elif p in (
                "dot_general",
                "scatter-mul",
                "custom_vjp_call",
                "custom_jvp_call",
                "pure_callback",
                "io_callback",
                "ffi_call",
            ):
                combined_mask = 0
                for v in eqn.invars:
                    combined_mask |= tags.get(id(v), 0)
                if combined_mask == 3:
                    is_nonlinear = True
            elif p in ("pjit", "jit"):
                pass
        else:
            if p in _NONLINEAR_UNARY:
                is_nonlinear = True
            elif p in _NONLINEAR_BINARY:
                is_nonlinear = True
            elif p == "integer_pow":
                exponent = eqn.params.get("y", 0)
                if exponent >= 2 or exponent <= -1:
                    is_nonlinear = True
            elif p in (
                "dot_general",
                "scatter-mul",
                "custom_vjp_call",
                "custom_jvp_call",
                "pure_callback",
                "io_callback",
                "ffi_call",
            ):
                is_nonlinear = True

        # Seed active variables if it is a nonlinear primitive
        if is_nonlinear:
            for v in eqn.invars:
                active_set.add(id(v))

        # resolve registered handler
        handler = TRACER_REGISTRY.get(p, Handlers.fallback)

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

    return forward_data, active_set


def _propagate_active_backward(
    forward_data: List[Tuple[Any, Callable, bool, Any]],
    active_set: Set[int],
    sub_info: Dict[int, Any],
) -> List[Tuple[Any, Callable, bool]]:
    """
    Performs the backward pass of a unified JAXpr analysis traversal:
    Propagates the active state backwards through the resolved equations list.

    Args:
        forward_data: the list of forward data tuples (eqn, handler, is_nonlinear, sub_res) from the forward pass
        active_set: the initial set of active variable IDs seeded from the outputs
        sub_info: the dict storing sub-jaxpr analysis results for nested jits; maps eqn IDs to (active_set, resolved_eqns)

    Returns:
        A list of tuples (eqn, handler, is_active) where is_active indicates whether this equation is on an active path.
    """
    pruned_eqns = []
    for eqn, handler, is_nonlinear, sub_res in reversed(forward_data):
        p = eqn.primitive.name
        is_active = False

        if p in ("pjit", "jit", "scan", "map", "remat2"):
            if any(id(v) in active_set for v in eqn.outvars):
                is_active = True
                sub_active_set, sub_eqns = sub_res
                sub, _ = _subjaxpr_and_consts(eqn)

                # Map active outvars to sub outvars
                for pv, sv in zip(eqn.outvars, sub.outvars):
                    if id(pv) in active_set:
                        sub_active_set.add(id(sv))

                # Recursively propagate active state backward in sub-jaxpr
                sub_eqns_pruned = _propagate_active_backward(
                    sub_eqns, sub_active_set, sub_info
                )

                # Map active sub invars to parent invars
                for pv, sv in zip(eqn.invars, sub.invars):
                    if id(sv) in sub_active_set:
                        active_set.add(id(pv))

                # Store sub-info for this jit equation
                sub_info[id(eqn)] = (sub_active_set, sub_eqns_pruned)
        elif p == "cond":
            if any(id(v) in active_set for v in eqn.outvars):
                is_active = True
                operands = eqn.invars[1:]
                pruned_branches = []
                for (sub_active_set, sub_eqns), branch in zip(
                    sub_res, eqn.params["branches"]
                ):
                    sub = branch.jaxpr

                    # Map active outvars to each branch's outvars
                    for pv, sv in zip(eqn.outvars, sub.outvars):
                        if id(pv) in active_set:
                            sub_active_set.add(id(sv))

                    sub_eqns_pruned = _propagate_active_backward(
                        sub_eqns, sub_active_set, sub_info
                    )

                    # Map active branch invars back to the cond operands (invars[1:])
                    for pv, sv in zip(operands, sub.invars):
                        if id(sv) in sub_active_set:
                            active_set.add(id(pv))

                    pruned_branches.append((sub_active_set, sub_eqns_pruned))

                sub_info[id(eqn)] = pruned_branches
        else:
            if is_nonlinear or (
                eqn.outvars and any(id(v) in active_set for v in eqn.outvars)
            ):
                is_active = True
                for v in eqn.invars:
                    active_set.add(id(v))

        pruned_eqns.append((eqn, handler, is_active))

    pruned_eqns.reverse()
    return pruned_eqns


def _trace_hessian_sparsity(
    fn: Callable[Concatenate[Array, P], Array],
    n_dofs: int,
    *static_args,
    trial_test_split: int | None = None,
) -> sps.csr_matrix:
    """
    Return the sparsity pattern of d²E/du² (or tangent stiffness matrix K for virtual work formulations)
    as a CSR matrix.
    """
    # Unwrap any outer @jax.jit so static slice indices stay static during tracing
    fn = _unwrap_jit(fn)

    closed = jax.make_jaxpr(fn)(np.zeros((n_dofs,)), *static_args)
    jaxpr: Jaxpr = closed.jaxpr
    consts: Sequence = closed.consts

    # Propagate tags, classify active primitives, and pre-resolve handlers in a single forward-backward pass
    tags = {}
    if trial_test_split is not None and jaxpr.invars:
        tags[id(jaxpr.invars[0])] = 3  # Seed main input with both

    sub_info = {}
    main_input_id = id(jaxpr.invars[0]) if jaxpr.invars else None
    forward_data, active_ids = _analyze_and_resolve_jaxpr(
        jaxpr, trial_test_split, tags, main_input_id, sub_info
    )
    bound_eqns = _propagate_active_backward(forward_data, active_ids, sub_info)

    # initialize tracing state
    state = TraceState(n_dofs, active_ids, tags, sub_info)

    # Seed concrete values of the input variables (essential for dynamic gather/scatter routing of static PyTree params)
    flat_args, _ = jax.tree_util.tree_flatten((np.zeros((n_dofs,)), *static_args))
    for invar, arg_val in zip(jaxpr.invars, flat_args):
        state.val_of[id(invar)] = np.asarray(arg_val)

    # seed: u gets singleton dep-sets; everything else gets empty sets
    u_seed = SparseDepSet.singletons(n_dofs)
    if jaxpr.invars:
        state.set(
            jaxpr.invars[0], u_seed
        )  # seed the input variable with singleton dep-sets
        for v in jaxpr.invars[1:]:
            state.set(
                v, SparseDepSet.empty(_get_shape(v), n_dofs)
            )  # seed other input variables (e.g., static args) with empty dep-sets

    # constants: empty deps but store concrete values for gather routing
    for v, c in zip(jaxpr.constvars, consts):
        state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
        state.val_of[id(v)] = np.asarray(c)

    acc = CouplingAccumulator(n_dofs)

    # forward pass: propagate dep-sets through the jaxpr, recording pairs at nonlinear primitives
    for eqn, handler, is_active in bound_eqns:
        ovars = eqn.outvars
        if ovars and not is_active:
            for v in ovars:
                state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
            # propagate concrete values (essential for gather/scatter routing indices)
            in_vals = [state.get_val(v) for v in eqn.invars]
            cv = _try_concrete(eqn.primitive.name, in_vals, eqn.params)
            if cv is not None:
                state.val_of[id(ovars[0])] = cv
            continue

        handler(eqn, state, acc, trial_test_split)

        # Propagate concrete value for executed equations as well (essential for active gather/scatter routing)
        if ovars:
            in_vals = [state.get_val(v) for v in eqn.invars]
            cv = _try_concrete(eqn.primitive.name, in_vals, eqn.params)
            if cv is not None:
                state.val_of[id(ovars[0])] = cv

    pat = acc.finalize()
    if pat.nnz == 0:
        return sps.eye(n_dofs, format="csr", dtype=np.int8)
    return pat


def pattern_from_energy(
    energy_fn: Callable[Concatenate[Array, P], Array],
    n_dofs: int,
    *static_args,
) -> sps.csr_matrix:
    """
    Return the sparsity pattern of d²E/du² as a symmetric CSR matrix for a scalar energy function E(u)
    where u has n_dofs degrees of freedom.

    Args:
        energy_fn: scalar JAX array energy function E(u, *static_args) as a function of input variable u and optional static arguments
        n_dofs: number of DOFs (integer size of flattened input array u)
        static_args: extra args passed to energy_fn, treated as constants

    Returns:
        A symmetric CSR matrix of shape (n_dofs, n_dofs) with binary entries indicating the sparsity pattern of the Hessian d²E/du².
    """
    return _trace_hessian_sparsity(
        energy_fn, n_dofs, *static_args, trial_test_split=None
    )


def pattern_from_virtual_work(
    virtual_work_fn: Callable[Concatenate[Array, P], Array],
    n_dofs: int,
    trial_arg: str,
    test_arg: str,
    *static_args,
) -> sps.csr_matrix:
    """
    Return the sparsity pattern of the tangent stiffness matrix K = dR/du = d²G/dvdu
    for a virtual work function virtual_work_fn(*args) as a CSR matrix.

    Args:
        virtual_work_fn : scalar JAX array (virtual work) as a function of trial and test variables (e.g., G(u, v, *static_args))
        n_dofs          : number of DOFs (integer size of flattened input arrays u and v)
        trial_arg       : parameter name of the trial function in virtual_work_fn
        test_arg        : parameter name of the test function in virtual_work_fn
        static_args     : extra arguments (e.g., mesh coordinates, parameters) passed to virtual_work_fn, treated as constants

    Returns:
        A CSR matrix of shape (n_dofs, n_dofs) with binary entries indicating the sparsity pattern of the tangent stiffness matrix K = dR/du = d²G/dvdu,
        where G is the virtual work and u,v are the trial and test functions respectively.
    """
    # Unwrap any outer @jax.jit so static slice indices stay static during tracing
    virtual_work_fn = _unwrap_jit(virtual_work_fn)

    combined_dofs = 2 * n_dofs  # combined size of trial and test variables

    # Inspect virtual_work_fn signature to locate trial and test parameter positions
    sig = inspect.signature(virtual_work_fn)
    params = list(sig.parameters.keys())

    if trial_arg not in params:
        raise ValueError(
            f"Trial argument '{trial_arg}' not found in signature of {virtual_work_fn.__name__}."
            f"Available parameters: {params}"
        )
    if test_arg not in params:
        raise ValueError(
            f"Test argument '{test_arg}' not found in signature of {virtual_work_fn.__name__}."
            f"Available parameters: {params}"
        )

    trial_idx = params.index(
        trial_arg
    )  # position of trial argument in virtual_work_fn signature
    test_idx = params.index(test_arg)

    def w_fn(w: Array) -> Array:
        u_val = w[:n_dofs]
        v_val = w[n_dofs:]

        # Reconstruct the arguments list for virtual_work_fn in the correct order
        args = []
        static_iter = iter(static_args)
        for k, param_name in enumerate(params):
            if k == trial_idx:
                args.append(u_val)
            elif k == test_idx:
                args.append(v_val)
            else:
                try:
                    args.append(next(static_iter))
                except StopIteration:
                    param = sig.parameters[param_name]
                    if param.default is not inspect.Parameter.empty:
                        args.append(param.default)
                    else:
                        raise ValueError(
                            f"Missing static argument for parameter '{param_name}' in {virtual_work_fn.__name__}"
                        )

        return virtual_work_fn(*args)

    # Trace the full Hessian of the combined function w_fn using private helper
    H_w: sps.csr_matrix = _trace_hessian_sparsity(
        w_fn, combined_dofs, trial_test_split=n_dofs
    )

    # Extract the cross-coupling block (v-derivatives vs u-derivatives)
    K_uv = H_w[n_dofs:, :n_dofs].tocsr()
    return K_uv


# ---------------------------------------------------------------------------
# concrete value propagation (needed to route gather indices exactly)
# ---------------------------------------------------------------------------


def _try_concrete(p: str, in_vals, par: dict):
    """Evaluate primitive on numpy values; return None if any input is unknown."""
    if any(v is None for v in in_vals):
        return None
    try:
        v = [np.asarray(x) for x in in_vals]
        if p == "add":
            return np.add(v[0], v[1])
        if p == "sub":
            return np.subtract(v[0], v[1])
        if p == "mul":
            return np.multiply(v[0], v[1])
        if p == "div":
            return np.true_divide(v[0], v[1])
        if p == "neg":
            return np.negative(v[0])
        if p == "abs":
            return np.abs(v[0])
        if p == "lt":
            return np.less(v[0], v[1])
        if p == "le":
            return np.less_equal(v[0], v[1])
        if p == "gt":
            return np.greater(v[0], v[1])
        if p == "ge":
            return np.greater_equal(v[0], v[1])
        if p == "eq":
            return np.equal(v[0], v[1])
        if p == "ne":
            return np.not_equal(v[0], v[1])
        if p == "min":
            return np.minimum(v[0], v[1])
        if p == "max":
            return np.maximum(v[0], v[1])
        if p == "floor":
            return np.floor(v[0])
        if p == "ceil":
            return np.ceil(v[0])
        if p == "round":
            return np.round(v[0])
        if p == "integer_pow":
            return v[0] ** par["y"]
        if p == "convert_element_type":
            return v[0].astype(par["new_dtype"])
        if p == "reshape":
            return v[0].reshape(par["new_sizes"])
        if p == "transpose":
            return np.transpose(v[0], par["permutation"])
        if p == "squeeze":
            return np.squeeze(v[0], axis=tuple(par["dimensions"]))
        if p == "slice":
            ss, ls = par["start_indices"], par["limit_indices"]
            st = par["strides"] or [1] * len(ss)
            return v[0][tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))]
        if p == "concatenate":
            return np.concatenate(v, axis=par["dimension"])
        if p == "iota":
            dim = par["dimension"]
            shp = par["shape"]
            newshp = [1] * len(shp)
            newshp[dim] = shp[dim]
            return np.broadcast_to(np.arange(shp[dim]).reshape(newshp), shp).copy()
        if p == "broadcast_in_dim":
            shape = par["shape"]
            bdims = par["broadcast_dimensions"]
            x = v[0]
            newshp = [1] * len(shape)
            for i, b in enumerate(bdims):
                newshp[b] = x.shape[i] if x.ndim > 0 else 1
            return np.broadcast_to(x.reshape(newshp), shape).copy()
        if p == "select_n":
            cond, cases = v[0], v[1:]
            if len(cases) == 2:
                return np.where(cond.astype(bool), cases[1], cases[0])
            result = cases[0].copy()
            for i, case in enumerate(cases[1:], 1):
                result = np.where(cond == i, case, result)
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# JAX primitive propagation handlers namespace class
# ---------------------------------------------------------------------------


class Handlers:
    """Namespace for primitive dependency propagation handlers."""

    @staticmethod
    def fallback(
        eqn: JaxprEqn,
        state: TraceState,
        acc: "CouplingAccumulator",
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
            reduced = sps.csr_matrix(cols_active.reshape(1, -1))
            total = SparseDepSet(reduced, ())
        stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
        state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))

    @staticmethod
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

    @staticmethod
    @TRACER_REGISTRY.register("add", "add_any", "sub", "max", "min")
    def add_sub_max_min(
        eqn: JaxprEqn,
        state: TraceState,
        acc: "CouplingAccumulator",
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        dep_out = (
            in_d[0].broadcast_to(oshp).dep + in_d[1].broadcast_to(oshp).dep
        ).tocsr()
        dep_out.data[:] = 1
        state.set(eqn.outvars[0], SparseDepSet(dep_out, oshp))

    @staticmethod
    @TRACER_REGISTRY.register(
        "neg",
        "abs",
        "convert_element_type",
        "copy",
        "stop_gradient",
        "device_put",
        "conj",
    )
    def passthrough(
        eqn,
        state: TraceState,
        acc: "CouplingAccumulator",
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        state.set(eqn.outvars[0], in_d[0].copy())

    @staticmethod
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
        mapped_src_indices = np.broadcast_to(
            src_indices.reshape(new_shape), oshp
        ).ravel()
        state.set(eqn.outvars[0], SparseDepSet(d.dep[mapped_src_indices], oshp))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    @TRACER_REGISTRY.register("slice")
    def slice(
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

    @staticmethod
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

    @staticmethod
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
                slice(lo, lo + s)
                for (lo, _, _), s in zip(par["padding_config"], d.shape)
            )
            target_indices = result_indices[slc].ravel()
            result_dep[target_indices] = d.dep
        state.set(eqn.outvars[0], SparseDepSet(result_dep.tocsr(), oshp))

    @staticmethod
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

    @staticmethod
    @TRACER_REGISTRY.register("select_n")
    def select_n(
        eqn: JaxprEqn,
        state: TraceState,
        acc: "CouplingAccumulator",
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        branches = in_d[1:]
        summed = sum(b.dep for b in branches)
        dep_out = sps.csr_matrix(summed.astype(bool))  # ty:ignore[unresolved-attribute]
        state.set(eqn.outvars[0], SparseDepSet(dep_out, oshp))

    @staticmethod
    @TRACER_REGISTRY.register(
        "iota",
        "lt",
        "le",
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

    @staticmethod
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

    @staticmethod
    @TRACER_REGISTRY.register(*_NONLINEAR_BINARY)
    def nonlinear_binary(
        eqn: JaxprEqn,
        state: TraceState,
        acc: "CouplingAccumulator",
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()
        dep_out = (
            in_d[0].broadcast_to(oshp).dep + in_d[1].broadcast_to(oshp).dep
        ).tocsr()
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

    @staticmethod
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

    @staticmethod
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

        if idx is not None and d_src.dep.shape[0] > 1:
            try:
                dnums = par["dimension_numbers"]
                ss = par["slice_sizes"]
                collapsed = dnums.collapsed_slice_dims
                sim = dnums.start_index_map

                if collapsed == (0,) and sim == (0,) and ss[0] == 1:
                    rows = idx.ravel().astype(int)
                    arr_indices = np.arange(int(np.prod(d_src.shape))).reshape(
                        d_src.shape
                    )
                    slc = (rows,) + tuple(slice(None, s) for s in ss[1:])
                    flat_src_indices = arr_indices[slc].ravel()
                    state.set(
                        eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                    )
                    return

                if (
                    collapsed == (0,)
                    and sim == (0, 1)
                    and ss[0] == 1
                    and idx.ndim == 2
                    and idx.shape[1] == 2
                ):
                    n_gathered = idx.shape[0]
                    n_cols = ss[1]
                    arr_indices = np.arange(int(np.prod(d_src.shape))).reshape(
                        d_src.shape
                    )
                    flat_src_indices = []
                    for k in range(n_gathered):
                        r = int(idx[k, 0])
                        c0 = int(idx[k, 1])
                        flat_src_indices.extend(
                            arr_indices[r, c0 : c0 + n_cols].tolist()
                        )
                    state.set(
                        eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                    )
                    return

                if (
                    set(collapsed) == set(sim)
                    and idx.ndim == 2
                    and idx.shape[1] == len(sim)
                    and all(ss[a] == 1 for a in sim)
                ):
                    n_gathered = idx.shape[0]
                    arr_indices = np.arange(int(np.prod(d_src.shape))).reshape(
                        d_src.shape
                    )
                    flat_src_indices = []
                    for k in range(n_gathered):
                        src_idx = tuple(int(idx[k, j]) for j in range(idx.shape[1]))
                        flat_src_indices.append(arr_indices[src_idx])
                    state.set(
                        eqn.outvars[0], SparseDepSet(d_src.dep[flat_src_indices], oshp)
                    )
                    return
            except Exception:
                pass

        total = d_src.total_union()
        stacked_dep = _broadcast_single_row(total.dep, int(np.prod(oshp)))
        state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))

    @staticmethod
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
                    valid_idx = idx_flat[
                        (idx_flat >= 0) & (idx_flat < d_tgt.dep.shape[0])
                    ]
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

    @staticmethod
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

    @staticmethod
    @TRACER_REGISTRY.register("dot_general")
    def dot(
        eqn: JaxprEqn,
        state: TraceState,
        acc: "CouplingAccumulator",
        trial_test_split: int | None,
    ) -> None:
        in_d = [state.get(v) for v in eqn.invars]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        # Determine if there is a batch axis
        dn = eqn.params.get("dimension_numbers")
        lhs_batch = []
        rhs_batch = []
        if dn is not None:
            if hasattr(dn, "lhs_batch_dimensions"):
                lhs_batch = dn.lhs_batch_dimensions
                rhs_batch = dn.rhs_batch_dimensions
            elif len(dn) > 1:
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

            # Record A and B self-couplings via accumulator (fingerprint-cached)
            acc.record_dep(A_active, trial_test_split)
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

            ua.record_couplings(acc, trial_test_split)
            ub.record_couplings(acc, trial_test_split)

            ia = ua.dep.indices
            ib = ub.dep.indices
            if ia.size and ib.size:
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

            combined = sps.csr_matrix((ua.dep + ub.dep).astype(bool))
            stacked_dep = _broadcast_single_row(combined, int(np.prod(oshp)))
            state.set(eqn.outvars[0], SparseDepSet(stacked_dep, oshp))

    @staticmethod
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

    @staticmethod
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
        sub_active, sub_bound_eqns = state.sub_info[id(eqn)]
        sub_state = TraceState(n_local_dofs, sub_active, {}, state.sub_info)

        # Seed consts: empty deps
        for i in range(num_const):
            sub_state.set(
                sub.invars[i],
                SparseDepSet.empty(_get_shape(sub.invars[i]), n_local_dofs),
            )
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

            sub_state.set(
                sub.invars[num_const + num_carry + k], SparseDepSet(sub_dep, shp)
            )

            # Set a representative concrete value from element 0
            val = state.get_val(eqn.invars[num_const + num_carry + k])
            if val is not None and val.shape[0] > 0:
                sub_state.val_of[id(sub.invars[num_const + num_carry + k])] = val[0]

            offset += sz

        # Trace the sub-jaxpr symbolically using a child accumulator
        sub_acc = CouplingAccumulator(n_local_dofs)
        for sub_eqn, sub_handler, sub_is_active in sub_bound_eqns:
            sub_ovars = sub_eqn.outvars
            if sub_ovars and not sub_is_active:
                for v in sub_ovars:
                    sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_local_dofs))
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(sub_ovars[0])] = cv
                continue

            sub_handler(sub_eqn, sub_state, sub_acc, None)

            if sub_ovars:
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
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
        slice_cache: Dict[int, Tuple[sps.csr_matrix, np.ndarray]] = {}

        def _get_col(local_idx: int) -> Tuple[sps.csr_matrix, np.ndarray]:
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

        # mapped outputs (ys)
        if len(eqn.outvars) > num_carry:
            dep_inputs = [
                state.get(eqn.invars[num_const + num_carry + k]).dep
                for k in range(num_xs)
            ]
            dep_all = sps.vstack(dep_inputs, format="csr")

            offsets_all = [0]
            for k in range(num_xs - 1):
                offsets_all.append(
                    offsets_all[-1] + total_length * local_size_slices[k]
                )

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

    @staticmethod
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
        sub_active, sub_bound_eqns = state.sub_info[id(eqn)]

        n_dofs = state.n_dofs
        sub_state = TraceState(n_dofs, sub_active, state.tags, state.sub_info)

        for v, d in zip(sub.invars, in_d):
            sub_state.set(v, d)
        for v, c in zip(sub.constvars, sub_consts):
            sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
            sub_state.val_of[id(v)] = np.asarray(c)

        for sub_eqn, sub_handler, sub_is_active in sub_bound_eqns:
            ovars = sub_eqn.outvars
            if ovars and not sub_is_active:
                for v in ovars:
                    sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
                # propagate concrete values (essential for gather/scatter routing indices)
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(ovars[0])] = cv
                continue

            sub_handler(sub_eqn, sub_state, acc, trial_test_split)

            # Propagate concrete value for executed equations as well
            if ovars:
                in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
                if cv is not None:
                    sub_state.val_of[id(ovars[0])] = cv

        for pv, sv in zip(eqn.outvars, sub.outvars):
            state.set(pv, sub_state.get(sv))

    @staticmethod
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

        out_deps: Dict[int, SparseDepSet | None] = {id(ov): None for ov in eqn.outvars}
        for (sub_active, sub_bound_eqns), branch in zip(
            branch_sub_list, eqn.params["branches"]
        ):
            sub = branch.jaxpr
            sub_consts = branch.consts
            sub_state = TraceState(n_dofs, sub_active, state.tags, state.sub_info)

            # Seed branch invars with operand dep-sets and concrete values (the latter
            # are needed to route any gather/scatter indices inside the branch).
            for sv, d, ov in zip(sub.invars, in_d, operands):
                sub_state.set(sv, d)
                val = state.get_val(ov)
                if val is not None:
                    sub_state.val_of[id(sv)] = val
            for v, c in zip(sub.constvars, sub_consts):
                sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
                sub_state.val_of[id(v)] = np.asarray(c)

            for sub_eqn, sub_handler, sub_is_active in sub_bound_eqns:
                ovars = sub_eqn.outvars
                if ovars and not sub_is_active:
                    for v in ovars:
                        sub_state.set(v, SparseDepSet.empty(_get_shape(v), n_dofs))
                    in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                    cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
                    if cv is not None:
                        sub_state.val_of[id(ovars[0])] = cv
                    continue

                sub_handler(sub_eqn, sub_state, acc, trial_test_split)

                if ovars:
                    in_vals = [sub_state.get_val(v) for v in sub_eqn.invars]
                    cv = _try_concrete(sub_eqn.primitive.name, in_vals, sub_eqn.params)
                    if cv is not None:
                        sub_state.val_of[id(ovars[0])] = cv

            for ov, sv in zip(eqn.outvars, sub.outvars):
                d = sub_state.get(sv)
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

    @staticmethod
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

    @staticmethod
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
            Handlers.custom_vjp_jvp_call(eqn, state, acc, trial_test_split)
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
