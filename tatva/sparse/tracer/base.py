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
from jax.extend.core import ClosedJaxpr, Jaxpr

from tatva.sparse.tracer.cache import persistent_tracer_cache
from tatva.sparse.tracer.common import _get_shape, _unwrap_jit
from tatva.sparse.tracer.state import (
    _analyze_and_resolve_jaxpr,
    _propagate_active_backward,
)
from tatva.sparse.tracer.types import (
    CouplingAccumulator,
    SparseDepSet,
    SubInfoDict,
    TraceState,
)

P = ParamSpec("P")


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
