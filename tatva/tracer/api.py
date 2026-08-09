from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

import jax
from jax import Array
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from tatva.tracer.dependencies import DependencySet
from tatva.tracer.helpers import _shape_of
from tatva.tracer.hessian import HessianAccumulator
from tatva.tracer.routing import Route
from tatva.tracer.rules import SEMANTICS


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


def trace_dependencies(captured_jaxpr: CapturedJaxpr) -> Any:
    # this must seed the input vars with the correct dependency sets
    dependencies: dict[Var, DependencySet] = {}
    routes: dict[JaxprEqn, Route] = {}
    # assume it has a leading dimension of n_dofs
    n_dofs = _shape_of(captured_jaxpr.invars[0])[0]
    acc = HessianAccumulator(n_dofs)

    def dependency_of(atom: Atom) -> DependencySet:
        if type(atom) is Literal:
            return DependencySet.empty(_shape_of(atom), n_dofs)
        else:
            return dependencies[atom]

    jaxpr = captured_jaxpr.jaxpr

    for eqn in jaxpr.eqns:
        rule = SEMANTICS[eqn.primitive]
        # where is routes defined eventually? inside the pass?
        route = routes.get(eqn)

        input_deps = tuple(dependency_of(v) for v in eqn.invars)

        rule.hessian(eqn, input_deps, route, acc)

        output_deps = rule.dependencies(eqn, input_deps, route)

        for var, dep in zip(eqn.outvars, output_deps):
            dependencies[var] = dep
