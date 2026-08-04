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

from typing import Any, ParamSpec

from jax.extend.core import Jaxpr, JaxprEqn

from tatva.sparse.tracer.common import (
    _subjaxpr_and_consts,
)
from tatva.sparse.tracer.handlers import PrimitiveHandler
from tatva.sparse.tracer.registry import TR
from tatva.sparse.tracer.types import BoundEqn, SubInfoDict

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
