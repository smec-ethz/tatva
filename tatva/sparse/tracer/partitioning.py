"""Backward decomposition of scalar JAX functionals into contribution roots."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from jax.extend.core import Jaxpr

from tatva.sparse.tracer.handlers import ContributionDemand, ContributionRoot
from tatva.sparse.tracer.state import BoundEqn, TraceState


def merge_demands(
    left: ContributionDemand | None, right: ContributionDemand | None
) -> ContributionDemand | None:
    """Union two flat-entry demands, retaining canonical sorted row IDs."""
    if left is None:
        return right
    if right is None:
        return left
    return ContributionDemand(np.union1d(left.rows, right.rows).astype(np.int64))


def merge_contribution_roots(
    roots: Iterable[ContributionRoot],
) -> list[ContributionRoot]:
    """Coalesce repeated roots reached through separate additive branches."""
    merged: dict[int, ContributionRoot] = {}
    for root in roots:
        previous = merged.get(id(root.var))
        if previous is None:
            merged[id(root.var)] = root
        else:
            demand = merge_demands(
                ContributionDemand(previous.rows), ContributionDemand(root.rows)
            )
            assert demand is not None
            merged[id(root.var)] = ContributionRoot(root.var, demand.rows)
    return list(merged.values())


def find_contribution_roots(
    jaxpr: Jaxpr,
    bound_eqns: list[BoundEqn],
    state: TraceState,
) -> list[ContributionRoot]:
    """Return additively separable roots for the scalar outputs of ``jaxpr``.

    Unsupported primitives deliberately become roots at their demanded outputs.  This
    fallback is conservative: later halo extraction uses the forward dependency rows of
    that output, so it cannot lose a data dependency merely because decomposition stops.
    """
    demand_of: dict[int, ContributionDemand | None] = {}
    for outvar in jaxpr.outvars:
        if state.is_scalar(outvar):
            demand_of[id(outvar)] = ContributionDemand(np.array([0], dtype=np.int64))

    if not demand_of:
        raise ValueError(
            "Contribution decomposition requires a scalar functional output."
        )

    roots: list[ContributionRoot] = []
    for eqn, handler, is_active, _needs_concrete in reversed(bound_eqns):
        if not is_active:
            continue

        out_demands = [demand_of.pop(id(outvar), None) for outvar in eqn.outvars]
        if not any(demand is not None for demand in out_demands):
            continue

        result = handler.propagate_contribution_demand(eqn, state, out_demands)
        if not result.valid:
            roots.extend(
                ContributionRoot(outvar, demand.rows)
                for outvar, demand in zip(eqn.outvars, out_demands)
                if demand is not None
            )
            continue

        roots.extend(result.roots)
        for invar, demand in zip(eqn.invars, result.in_demands):
            if demand is None or state.is_inactive(invar):
                continue
            demand_of[id(invar)] = merge_demands(demand_of.get(id(invar)), demand)

    for var in (*jaxpr.invars, *jaxpr.constvars):
        demand = demand_of.get(id(var))
        if demand is not None and not state.is_inactive(var):
            roots.append(ContributionRoot(var, demand.rows))

    return merge_contribution_roots(roots)
