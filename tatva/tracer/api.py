from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

import jax
import scipy.sparse as sps
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr

from tatva.tracer.analysis import JaxprPlan, analyze
from tatva.tracer.contributions import ContributionTrace, detect_contributions
from tatva.tracer.derivatives import DerivativeTrace, trace_derivatives
from tatva.tracer.helpers import _shape_of
from tatva.tracer.materialize import JaxprInstance, materialize_plan


@dataclass(frozen=True)
class CapturedJaxpr:
    closed_jaxpr: ClosedJaxpr
    flat_args: list[Any]
    pytree_def: jax.tree_util.PyTreeDef

    @classmethod
    def from_fn[**P](
        cls, fn: Callable[P, Array], *args: P.args, **kwargs: P.kwargs
    ) -> Self:
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


@dataclass(frozen=True)
class TraceResult:
    captured: CapturedJaxpr
    analysis: JaxprPlan
    resolved: JaxprInstance
    derivatives: DerivativeTrace
    contributions: ContributionTrace

    @property
    def hessian(self) -> sps.csr_matrix:
        return self.derivatives.hessian


def trace(captured: CapturedJaxpr) -> TraceResult:
    jaxpr = captured.jaxpr

    if not jaxpr.invars:
        raise ValueError("Functional JAXPR has no inputs")

    dof_shape = _shape_of(jaxpr.invars[0])
    if len(dof_shape) != 1:
        raise ValueError(
            f"First input must be a flat DOF vector, got shape {dof_shape}"
        )
    n_dofs = dof_shape[0]

    # 1. Static structural analysis
    analysis = analyze(jaxpr)

    # 2. recursive concrete evaluation + route materialization
    resolved = materialize_plan(
        captured.closed_jaxpr,
        captured.flat_args,
        analysis,
    )

    # 3. recursive derivative propagation
    derivatives = trace_derivatives(
        resolved,
        n_dofs=n_dofs,
    )

    contributions = detect_contributions(
        resolved,
    )

    return TraceResult(
        captured=captured,
        analysis=analysis,
        resolved=resolved,
        derivatives=derivatives,
        contributions=contributions,
    )
