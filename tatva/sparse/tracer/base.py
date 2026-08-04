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
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, ParamSpec, cast

import jax
import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn

from tatva.sparse.tracer.cache import persistent_tracer_cache
from tatva.sparse.tracer.common import (
    _get_shape,
    _subjaxpr_and_consts,
    _unwrap_jit,
)
from tatva.sparse.tracer.handlers import PrimitiveHandler
from tatva.sparse.tracer.registry import TR
from tatva.sparse.tracer.state import (
    BoundEqn,
    CouplingAccumulator,
    SparseDepSet,
    SubInfoDict,
    TraceState,
)

P = ParamSpec("P")


def _analyze_and_resolve_jaxpr(
    jaxpr: Jaxpr,
    trial_test_split: int | None,
    tags: dict[int, int],
    main_input_id: int | None,
    sub_info: SubInfoDict,
) -> tuple[list[tuple[JaxprEqn, PrimitiveHandler, bool, Any]], set[int], set[int]]:
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
    forward_data = []
    active_set = {id(v) for v in jaxpr.outvars}
    index_set = set()

    for eqn in jaxpr.eqns:
        p = eqn.primitive.name
        handler = TR.get(p)

        # identify indexing variable IDs via handler
        for idx_pos in handler.get_index_invar_indices(eqn):
            index_set.add(id(eqn.invars[idx_pos]))

        # Main input slice tag seeding for trial/test split
        if (
            trial_test_split is not None
            and p == "slice"
            and main_input_id is not None
            and id(eqn.invars[0]) == main_input_id
        ):
            start, limit = (
                eqn.params["start_indices"][0],
                eqn.params["limit_indices"][0],
            )
            tags[id(eqn.outvars[0])] = (
                1
                if (start == 0 and limit <= trial_test_split)
                else (2 if start >= trial_test_split else 3)
            )

        # Sub-jaxpr recursion (unified for both trial_test_split modes)
        sub_res = None
        if p in ("pjit", "jit", "scan", "map", "remat2"):
            sub_jaxpr, _ = _subjaxpr_and_consts(eqn)
            for pv, sv in zip(eqn.invars, sub_jaxpr.invars):
                tags[id(sv)] = tags.get(id(pv), 0)
            sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                sub_jaxpr, trial_test_split, tags, None, sub_info
            )
            sub_res = (sub_active, sub_index_set, sub_eqns)
            for pv, sv in zip(eqn.invars, sub_jaxpr.invars):
                if id(sv) in sub_index_set:
                    index_set.add(id(pv))
            if trial_test_split is not None:
                mask = 0
                for pv, sv in zip(eqn.outvars, sub_jaxpr.outvars):
                    tags[id(pv)] = tags.get(id(sv), 0)
                    mask |= tags[id(pv)]
                for v in eqn.outvars:
                    tags[id(v)] = mask

        elif p == "cond":
            operands, sub_res = eqn.invars[1:], []
            for branch in eqn.params["branches"]:
                bj = branch.jaxpr
                for pv, sv in zip(operands, bj.invars):
                    tags[id(sv)] = tags.get(id(pv), 0)
                sub_eqns, sub_active, sub_index_set = _analyze_and_resolve_jaxpr(
                    bj, trial_test_split, tags, None, sub_info
                )
                sub_res.append((sub_active, sub_index_set, sub_eqns))
                for pv, sv in zip(operands, bj.invars):
                    if id(sv) in sub_index_set:
                        index_set.add(id(pv))
            if trial_test_split is not None:
                mask = 0
                for branch in eqn.params["branches"]:
                    for sv in branch.jaxpr.outvars:
                        mask |= tags.get(id(sv), 0)
                for v in eqn.outvars:
                    tags[id(v)] = mask

        elif trial_test_split is not None:
            mask = handler.propagate_tags(eqn, [tags.get(id(v), 0) for v in eqn.invars])
            for v in eqn.outvars:
                tags[id(v)] = mask

        # Polymorphic nonlinearity check & active set seeding
        if trial_test_split is not None:
            invar_active = [tags.get(id(v), 0) == 3 for v in eqn.invars]
        else:
            invar_active = [True for _ in eqn.invars]

        is_nonlinear = handler.introduces_nonlinearity(eqn, invar_active)
        if is_nonlinear:
            for v in eqn.invars:
                active_set.add(id(v))

        forward_data.append((eqn, handler, is_nonlinear, sub_res))

    return forward_data, active_set, index_set


def _propagate_active_backward(
    forward_data: list[tuple[JaxprEqn, PrimitiveHandler, bool, Any]],
    active_set: set[int],
    index_set: set[int],
    sub_info: SubInfoDict,
) -> list[BoundEqn]:
    """Performs the backward pass of a unified JAXpr analysis traversal:
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


def _trace_hessian_sparsity(
    jaxpr: ClosedJaxpr,
    tracer_shape: tuple[int, ...],
    concrete_vals: list[Any],
    trial_test_split: int | None = None,
) -> sps.csr_matrix:
    """Return the sparsity pattern of d²E/du² (or tangent stiffness matrix K for virtual
    work formulations) as a CSR matrix."""
    n_dofs = int(np.prod(tracer_shape))

    consts: Sequence = jaxpr.consts
    jaxpr: Jaxpr = jaxpr.jaxpr

    # Propagate tags, classify active primitives, and pre-resolve handlers in a single forward-backward pass
    tags = {}
    if trial_test_split is not None and jaxpr.invars:
        tags[id(jaxpr.invars[0])] = 3  # Seed main input with both

    sub_info: SubInfoDict = {}
    main_input_id = id(jaxpr.invars[0]) if jaxpr.invars else None
    forward_data, active_ids, index_ids = _analyze_and_resolve_jaxpr(
        jaxpr, trial_test_split, tags, main_input_id, sub_info
    )
    bound_eqns = _propagate_active_backward(
        forward_data, active_ids, index_ids, sub_info
    )

    # initialize tracing state
    state = TraceState(n_dofs, active_ids, tags, sub_info)

    # Seed concrete values of the input variables (essential for dynamic gather/scatter routing of static PyTree params)
    for invar, arg_val in zip(jaxpr.invars, concrete_vals):
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
    state.run_bound_eqns(bound_eqns, acc, trial_test_split)

    pat = acc.finalize()
    if pat.nnz == 0:
        return sps.eye(n_dofs, format="csr", dtype=np.int8)
    return pat


def pattern_from_energy(
    energy_fn: Callable[P, Array], skip_cache: bool = False
) -> Callable[P, sps.csr_matrix]:
    """Return a function that computes the sparsity pattern of d²E/du² as a symmetric CSR
    matrix for a scalar energy function E(u) where u has n_dofs degrees of freedom.

    Args:
        energy_fn: scalar JAX array energy function E(u, *static_args) as a function of
            input variable u and optional static arguments
        skip_cache: if True, skip the custom sparsity cache and recompute.
    """

    _tracer_fn = persistent_tracer_cache(skip_cache=skip_cache)(_trace_hessian_sparsity)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> sps.csr_matrix:
        # Unwrap any outer @jax.jit so static slice indices stay static during tracing
        fn = _unwrap_jit(energy_fn)

        u = cast(Array, args[0])
        assert isinstance(args[0], Array), (
            "First argument to energy_fn must be a JAX array (the input variable u)."
        )
        assert u.ndim == 1, (
            "Input variable u must be a 1D JAX array (flattened degrees of freedom)."
        )
        closed = jax.make_jaxpr(fn)(*args, **kwargs)
        flat_args, _ = jax.tree_util.tree_flatten((args, kwargs))

        return _tracer_fn(
            closed,
            tracer_shape=u.shape,
            concrete_vals=flat_args,
            trial_test_split=None,
        )

    return wrapper


def pattern_from_virtual_work(
    virtual_work_fn: Callable[Concatenate[Array, P], Array],
    n_dofs: int,
    trial_arg: str,
    test_arg: str,
    *static_args,
    skip_cache: bool = False,
) -> sps.csr_matrix:
    """
    Return the sparsity pattern of the tangent stiffness matrix K = dR/du = d²G/dvdu
    for a virtual work function virtual_work_fn(*args) as a CSR matrix.

    Args:
        virtual_work_fn: scalar JAX array (virtual work) as a function of trial and test
            variables (e.g., G(u, v, *static_args))
        n_dofs: number of DOFs (integer size of flattened input arrays u and v)
        trial_arg: parameter name of the trial function in virtual_work_fn
        test_arg: parameter name of the test function in virtual_work_fn
        static_args: extra arguments (e.g., mesh coordinates, parameters) passed to
            virtual_work_fn, treated as constants

    Returns:
        A CSR matrix of shape (n_dofs, n_dofs) with binary entries indicating the sparsity
        pattern of the tangent stiffness matrix K = dR/du = d²G/dvdu, where G is the
        virtual work and u,v are the trial and test functions respectively.
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

    dummy_w = np.zeros(combined_dofs)
    closed = jax.make_jaxpr(w_fn)(dummy_w)
    _tracer_fn = persistent_tracer_cache(skip_cache=skip_cache)(_trace_hessian_sparsity)
    H_w: sps.csr_matrix = _tracer_fn(
        closed, (combined_dofs,), (dummy_w,), trial_test_split=n_dofs
    )

    # Extract the cross-coupling block (v-derivatives vs u-derivatives)
    K_uv = H_w[n_dofs:, :n_dofs].tocsr()
    return K_uv
