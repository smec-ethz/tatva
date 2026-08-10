from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

import jax
import scipy.sparse as sps
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr

from tatva.tracer.analysis import (
    AnalysisPlan,
    analyze,
    dof_value_dependencies,
    validate_static_concrete_inputs,
)
from tatva.tracer.concrete import evaluate_concrete
from tatva.tracer.dependencies import DerivativeTrace, trace_derivatives
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import ConcreteEnv, RouteEnv
from tatva.tracer.routing import resolve_routes


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
    analysis: AnalysisPlan

    concrete: ConcreteEnv
    routes: RouteEnv

    derivatives: DerivativeTrace

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
    plan = analyze(jaxpr)

    # 2. Ensure anything baked into routing is independent of u
    value_dependencies = dof_value_dependencies(jaxpr)
    validate_static_concrete_inputs(plan, value_dependencies)

    # 3. Evaluate exactly the concrete subgraph needed by routing
    concrete = evaluate_concrete(
        captured.closed_jaxpr,
        captured.flat_args,
        plan,
    )

    # 4. Resolve all global-coordinate structural routes
    routes = resolve_routes(
        plan.eqns,
        concrete,
    )

    # 5. Forward derivative-structure propagation
    derivatives = trace_derivatives(
        jaxpr=jaxpr,
        eqns=plan.eqns,
        routes=routes,
        n_dofs=n_dofs,
    )

    return TraceResult(
        captured=captured,
        analysis=plan,
        concrete=concrete,
        routes=routes,
        derivatives=derivatives,
    )
