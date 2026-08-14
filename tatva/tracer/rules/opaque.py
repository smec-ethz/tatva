from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps

from tatva.tracer.core.semantics import DerivativeRule, no_hessian
from tatva.tracer.helpers import _shape_of
from tatva.tracer.program.dependencies import DependencySet

if TYPE_CHECKING:
    from tatva.tracer.core.semantics import RuleContext
    from tatva.tracer.program.dependencies import HessianAccumulator


@dataclass(frozen=True)
class OpaqueData:
    total: DependencySet


def prepare_opaque(ctx: RuleContext) -> OpaqueData:
    warnings.warn(
        f"Using conservative opaque derivative rule for {ctx.eqn.primitive.name}"
    )
    active = np.zeros(ctx.n_dofs, dtype=bool)
    for dep in ctx.input_deps:
        if dep.csr.nnz:
            active[dep.csr.indices] = True

    cols = np.flatnonzero(active)
    total_csr = sps.csr_matrix(
        (
            np.ones(cols.size, dtype=bool),
            (
                np.zeros(cols.size, dtype=np.int64),
                cols,
            ),
        ),
        shape=(1, ctx.n_dofs),
        dtype=bool,
    )
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
    acc: HessianAccumulator,
) -> None:
    acc.add_self(prepared.total)


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
