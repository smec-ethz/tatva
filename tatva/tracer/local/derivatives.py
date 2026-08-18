"""Rank-local structural derivative analysis.

The local lowering plan has already resolved and localized every structural
route.  Derivative analysis therefore captures the lowered rank objective and
reuses the established JAXPR derivative tracer in storage-local DOF
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.local.dof_plan import LocalDofPlan
from tatva.tracer.lowering.executor import LocalExecutable
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.derivatives import (
    DerivativeTrace,
    JaxprDerivativeTrace,
    trace_derivatives,
)
from tatva.tracer.program.materialize import materialize_plan
from tatva.tracer.support import require_registered_operations


@dataclass(frozen=True, slots=True)
class LocalDerivativeTrace:
    """Derivative sparsity for one rank's contribution objective.

    Hessian rows and columns use the rank's runtime storage ordering:
    ``[owned DOFs | ghost DOFs]``.
    """

    trace: DerivativeTrace
    storage_global_dofs: NDArray[np.int64]
    global_size: int

    def __post_init__(self) -> None:
        global_dofs = (
            np.asarray(self.storage_global_dofs, dtype=np.int64).ravel().copy()
        )
        if self.global_size < 0:
            raise ValueError("global_size must be nonnegative")
        if np.any((global_dofs < 0) | (global_dofs >= self.global_size)):
            raise ValueError("storage_global_dofs contains out-of-range DOFs")
        if self.trace.hessian.shape != (global_dofs.size, global_dofs.size):
            raise ValueError(
                "local Hessian shape does not match storage coordinates: "
                f"{self.trace.hessian.shape} != {(global_dofs.size, global_dofs.size)}"
            )
        global_dofs.flags.writeable = False
        object.__setattr__(self, "storage_global_dofs", global_dofs)

    @property
    def root(self) -> JaxprDerivativeTrace:
        return self.trace.root

    @property
    def hessian(self) -> sps.csr_matrix:
        """Structural Hessian sparsity in storage-local coordinates."""
        return self.trace.hessian

    def global_hessian_coo(self) -> sps.coo_matrix:
        """Translate local nonzeros to global IDs without a global CSR indptr."""
        local = self.hessian.tocoo()
        rows = self.storage_global_dofs[local.row]
        cols = self.storage_global_dofs[local.col]
        return sps.coo_matrix(
            (local.data.copy(), (rows, cols)),
            shape=(self.global_size, self.global_size),
        )


def trace_local_derivatives(
    executable: LocalExecutable,
    dof_plan: LocalDofPlan,
    global_inputs: tuple[Any, ...],
) -> LocalDerivativeTrace:
    """Trace one lowered rank objective in storage-local DOF coordinates."""
    if not executable.input_indices or executable.input_indices[0] != 0:
        raise RuntimeError("local executable does not have a live first DOF input")
    if len(global_inputs) != len(executable.plan.input_layouts):
        raise ValueError(
            f"expected {len(executable.plan.input_layouts)} global inputs, "
            f"got {len(global_inputs)}"
        )

    packed = executable.pack_global_inputs(*global_inputs)
    storage_example = jnp.asarray(global_inputs[0])[
        jnp.asarray(dof_plan.storage.global_dofs)
    ]
    other_examples = packed[1:]
    compute_rows = jnp.asarray(dof_plan.compute_rows)

    def storage_objective(storage_dofs, *other_inputs):
        compute_dofs = storage_dofs[compute_rows]
        return executable(compute_dofs, *other_inputs)

    examples = (storage_example,) + other_examples
    closed_jaxpr = jax.make_jaxpr(storage_objective)(*examples)
    require_registered_operations(closed_jaxpr.jaxpr)
    plan = analyze(closed_jaxpr.jaxpr)
    instance = materialize_plan(closed_jaxpr, examples, plan)
    derivative_trace = trace_derivatives(instance, n_dofs=dof_plan.storage.local_size)

    return LocalDerivativeTrace(
        trace=derivative_trace,
        storage_global_dofs=dof_plan.storage.global_dofs,
        global_size=dof_plan.global_size,
    )
