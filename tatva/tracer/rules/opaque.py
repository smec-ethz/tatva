from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax.extend.core import Literal

from tatva.tracer.core.semantics import DemandContext, DerivativeRule, no_hessian
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import Demand, TensorDemand
from tatva.tracer.program.dependencies import DependencySet

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext
    from tatva.tracer.program.dependencies import InteractionGraph


@dataclass(frozen=True)
class OpaqueData:
    total: DependencySet


def prepare_opaque(ctx: RuleContext) -> OpaqueData:
    warnings.warn(
        f"Using conservative opaque derivative rule for {ctx.eqn.primitive.name}"
    )
    nonempty = [dep.total_union().csr for dep in ctx.input_deps if dep.csr.nnz]
    if not nonempty:
        total_csr = sps.csr_matrix((1, ctx.n_dofs), dtype=bool)
    else:
        total_csr = sps.vstack(nonempty, format="csr").max(axis=0)
        total_csr = sps.csr_matrix(total_csr, dtype=bool)

    return OpaqueData(DependencySet(total_csr, (1,)))


def opaque_dependencies(
    ctx: RuleContext,
    prepared: OpaqueData,
) -> tuple[DependencySet, ...]:
    outputs: list[DependencySet] = []

    for outvar in ctx.eqn.outvars:
        shape = _shape_of(outvar)
        n_rows = int(np.prod(shape))

        if n_rows == 0:
            csr = sps.csr_matrix(
                (0, prepared.total.csr.shape[1]),
                dtype=bool,
            )
        else:
            selection = sps.csr_matrix(
                (
                    np.ones(n_rows, dtype=bool),
                    (np.arange(n_rows), np.zeros(n_rows, dtype=np.int64)),
                ),
                shape=(n_rows, 1),
                dtype=bool,
            )
            csr = (selection @ prepared.total.csr).tocsr()

        outputs.append(DependencySet(csr, shape))

    return tuple(outputs)


def opaque_nonlinear_hessian(
    ctx: RuleContext,
    prepared: OpaqueData,
    acc: InteractionGraph,
) -> None:
    acc.add_self(prepared.total)


def full_operand_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    """Require every non-literal operand whenever any result is live."""
    if not any(demand is not None for demand in ctx.output_demands):
        return tuple(None for _ in ctx.eqn.invars)

    return tuple(
        None if isinstance(atom, Literal) else TensorDemand.full(_shape_of(atom))
        for atom in ctx.eqn.invars
    )


sort_demand = full_operand_demand


DERIVATIVES_OPAQUE_NONLINEAR = DerivativeRule(
    prepare_opaque,
    opaque_dependencies,
    opaque_nonlinear_hessian,
)


DERIVATIVES_OPAQUE_LINEAR = DerivativeRule(
    prepare_opaque,
    opaque_dependencies,
    no_hessian,
)
