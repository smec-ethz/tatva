from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal

from tatva.tracer.demand import (
    Demand,
    TensorDemand,
    _FullAxis,
    demand_axes,
    merge_demands,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.model import Shape
from tatva.tracer.semantics import (
    DemandContext,
    conservative_demand,
)


@dataclass(frozen=True)
class _CustomLinearSolveLayout:
    batch_shape: Shape
    # outer custom_linear_solve input indices
    solve_inputs: tuple[int, ...]
    rhs_inputs: tuple[int, ...]


def _as_jaxpr(value: Any) -> Jaxpr:
    if isinstance(value, ClosedJaxpr):
        return value.jaxpr
    if isinstance(value, Jaxpr):
        return value

    # Keeps this tolerant of wrapper objects while still failing loudly.
    jaxpr = getattr(value, "jaxpr", None)
    if isinstance(jaxpr, Jaxpr):
        return jaxpr

    raise TypeError(f"expected Jaxpr or ClosedJaxpr, got {type(value)!r}")


def _has_batch_prefix(
    atom,
    batch_shape: tuple[int, ...],
) -> bool:
    if isinstance(atom, Literal):
        return False

    shape = _shape_of(atom)
    return len(shape) >= len(batch_shape) and shape[: len(batch_shape)] == batch_shape


def _outputs_have_batch_prefix(
    eqn: JaxprEqn,
    batch_shape: tuple[int, ...],
) -> bool:
    return all(_has_batch_prefix(outvar, batch_shape) for outvar in eqn.outvars)


# Operations that are guaranteed to be batch-local when all inputs have the same
# leading batch shape. This is a conservative list of operations that are
# known to be batch-local
_BATCH_LOCAL_ELEMENTWISE = {
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "select_n",
    "convert_element_type",
    "copy",
    "stop_gradient",
}


def _jaxpr_preserves_batch(
    jaxpr: Jaxpr,
    batch_shape: tuple[int, ...],
) -> bool:
    """Return True only when every operation in `jaxpr` can be proven not to
    couple different entries of the leading batch dimensions.
    """
    batch_rank = len(batch_shape)

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name

        # Elementwise operations.
        if name in _BATCH_LOCAL_ELEMENTWISE:
            if any(
                _has_batch_prefix(atom, batch_shape) for atom in eqn.invars
            ) and not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # jit / pjit-style transparent wrapper.
        if name in {"jit", "pjit"}:
            nested = eqn.params.get("jaxpr")

            if nested is None:
                return False

            if not _jaxpr_preserves_batch(_as_jaxpr(nested), batch_shape):
                return False
            continue

        # broadcast_in_dim
        if name == "broadcast_in_dim":
            source = eqn.invars[0]

            if _has_batch_prefix(source, batch_shape):
                dimensions = tuple(
                    int(axis) for axis in eqn.params["broadcast_dimensions"]
                )

                # Existing batch dimensions must stay in the same
                # leading positions.
                if dimensions[:batch_rank] != tuple(range(batch_rank)):
                    return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # transpose
        if name == "transpose":
            permutation = tuple(int(axis) for axis in eqn.params["permutation"])

            # Batch axes must remain fixed.
            if permutation[:batch_rank] != tuple(range(batch_rank)):
                return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # reshape
        #
        # Keeping the exact same leading batch shape means reshape only
        # reorganizes entries within each independent batch item.
        if name == "reshape":
            source_shape = _shape_of(eqn.invars[0])
            output_shape = _shape_of(eqn.outvars[0])

            if source_shape[:batch_rank] != batch_shape:
                return False

            if output_shape[:batch_rank] != batch_shape:
                return False

            source_local_size = int(
                np.prod(
                    source_shape[batch_rank:],
                    dtype=np.int64,
                )
            )
            output_local_size = int(math.prod(output_shape[batch_rank:]))

            if source_local_size != output_local_size:
                return False

            continue

        # squeeze
        if name == "squeeze":
            dimensions = tuple(int(axis) for axis in eqn.params["dimensions"])

            # Never squeeze a batch dimension.
            if any(axis < batch_rank for axis in dimensions):
                return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # gather
        #
        # Only accept an explicitly batched gather whose batch dimensions
        # correspond one-to-one between operand and index tensor.
        if name == "gather":
            dnums = eqn.params["dimension_numbers"]
            operand_batching_dims = tuple(
                int(axis) for axis in getattr(dnums, "operand_batching_dims", ())
            )
            indices_batching_dims = tuple(
                int(axis) for axis in getattr(dnums, "start_indices_batching_dims", ())
            )
            expected = tuple(range(batch_rank))

            if operand_batching_dims != expected:
                return False

            if indices_batching_dims != expected:
                return False

            start_index_map = tuple(int(axis) for axis in dnums.start_index_map)

            # A batch axis must never be indexed through the ordinary
            # gather start vector.
            if any(axis < batch_rank for axis in start_index_map):
                return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # triangular_solve
        #
        # JAX triangular_solve is independent over its leading batch
        # dimensions.
        if name == "triangular_solve":
            if not all(
                _has_batch_prefix(atom, batch_shape)
                for atom in eqn.invars
                if not isinstance(atom, Literal)
            ):
                return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # Empty reductions are identity-like. This occurs in some JAX
        # generated linear algebra JAXPRs.
        if name == "reduce_sum":
            axes = tuple(int(axis) for axis in eqn.params["axes"])

            # Reducing a batch dimension would couple/remove it.
            if any(axis < batch_rank for axis in axes):
                return False

            if not _outputs_have_batch_prefix(eqn, batch_shape):
                return False

            continue

        # Unknown operation: do not guess.
        return False

    return True


def _recognize_batched_lu_solve(
    ctx: DemandContext,
) -> _CustomLinearSolveLayout | None:
    eqn = ctx.eqn

    try:
        lengths = eqn.params["const_lengths"]
        jaxprs = eqn.params["jaxprs"]
        n_matvec = int(lengths.matvec)
        n_vecmat = int(lengths.vecmat)
        n_solve = int(lengths.solve)
        n_transpose_solve = int(lengths.transpose_solve)

    except (KeyError, AttributeError, TypeError):
        return None

    # This rule specifically recognizes the LU solve layout:
    #
    # solve constants:
    #   0: LU factors [..., n, n]
    #   1: pivots     [..., n]
    if n_solve != 2:
        return None

    solve_start = n_matvec + n_vecmat
    transpose_start = solve_start + n_solve
    rhs_start = transpose_start + n_transpose_solve

    if rhs_start >= len(eqn.invars):
        return None

    solve_inputs = (solve_start, solve_start + 1)
    rhs_inputs = tuple(range(rhs_start, len(eqn.invars)))
    factors_shape = _shape_of(eqn.invars[solve_inputs[0]])
    pivots_shape = _shape_of(eqn.invars[solve_inputs[1]])

    if len(factors_shape) < 2:
        return None

    n = factors_shape[-1]

    if factors_shape[-2] != n:
        return None

    batch_shape = factors_shape[:-2]

    # LU pivot tensor is [..., n].
    if pivots_shape != (batch_shape + (n,)):
        return None

    # Every RHS and output must carry the same leading batch shape.
    for input_index in rhs_inputs:
        shape = _shape_of(eqn.invars[input_index])
        if len(shape) < len(batch_shape) or shape[: len(batch_shape)] != batch_shape:
            return None

    for outvar in eqn.outvars:
        shape = _shape_of(outvar)
        if len(shape) < len(batch_shape) or shape[: len(batch_shape)] != batch_shape:
            return None

    # Validate child solve input correspondence.
    try:
        solve_jaxpr = _as_jaxpr(jaxprs.solve)
    except TypeError:
        return None

    outer_primal_inputs = solve_inputs + rhs_inputs

    if len(solve_jaxpr.invars) != len(outer_primal_inputs):
        return None

    for child_var, outer_index in zip(solve_jaxpr.invars, outer_primal_inputs):
        if _shape_of(child_var) != _shape_of(eqn.invars[outer_index]):
            return None

    # Finally prove that the actual solve implementation does not mix
    # different batch entries.
    if not _jaxpr_preserves_batch(solve_jaxpr, batch_shape):
        return None

    return _CustomLinearSolveLayout(
        batch_shape=batch_shape,
        solve_inputs=solve_inputs,
        rhs_inputs=rhs_inputs,
    )


def _project_batch_demand(
    demand: Demand,
    *,
    batch_shape: tuple[int, ...],
) -> Demand:
    if demand is None:
        return None

    batch_rank = len(batch_shape)

    if batch_rank == 0:
        return TensorDemand.full(())

    if demand.shape[:batch_rank] != batch_shape:
        raise ValueError(
            f"demand shape {demand.shape} does not start with batch shape {batch_shape}"
        )

    return TensorDemand.from_axes(
        batch_shape,
        demand_axes(demand)[:batch_rank],
    )


def _expand_batch_demand(
    batch_demand: Demand,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> Demand:
    if batch_demand is None:
        return None

    batch_rank = len(batch_shape)

    if shape[:batch_rank] != batch_shape:
        raise ValueError(f"shape {shape} does not begin with batch shape {batch_shape}")

    return TensorDemand.from_axes(
        shape,
        demand_axes(batch_demand)
        + tuple(_FullAxis() for _ in range(len(shape) - batch_rank)),
    )


def custom_linear_solve_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    """Preserve leading batch structure for a recognized LU-backed custom_linear_solve.

    The primal solve uses only:
        - solve constants,
        - RHS inputs.

    matvec, vecmat, and transpose-solve closure constants are not primal-live.

    If the solve implementation cannot be proven batch-local, fall back to the
    conservative rule.
    """
    if not any(demand is not None for demand in ctx.output_demands):
        return tuple(None for _ in ctx.eqn.invars)

    layout = _recognize_batched_lu_solve(ctx)

    if layout is None:
        return conservative_demand(ctx)

    batch_demand: Demand = None

    for output_demand in ctx.output_demands:
        projected = _project_batch_demand(output_demand, batch_shape=layout.batch_shape)
        batch_demand = merge_demands(batch_demand, projected)

    if batch_demand is None:
        return tuple(None for _ in ctx.eqn.invars)

    result: list[Demand] = [None] * len(ctx.eqn.invars)

    # LU factors + pivots.
    for input_index in layout.solve_inputs:
        result[input_index] = _expand_batch_demand(
            batch_demand,
            shape=_shape_of(ctx.eqn.invars[input_index]),
            batch_shape=layout.batch_shape,
        )

    # RHS leaves.
    for input_index in layout.rhs_inputs:
        result[input_index] = _expand_batch_demand(
            batch_demand,
            shape=_shape_of(ctx.eqn.invars[input_index]),
            batch_shape=layout.batch_shape,
        )

    return tuple(result)


def lu_demand(
    ctx: DemandContext,
) -> tuple[Demand, ...]:
    if len(ctx.eqn.invars) != 1:
        raise ValueError(f"lu expected one input, got {len(ctx.eqn.invars)}")

    if not any(demand is not None for demand in ctx.output_demands):
        return (None,)

    input_shape = _shape_of(ctx.eqn.invars[0])

    if len(input_shape) < 2:
        return conservative_demand(ctx)

    batch_shape = input_shape[:-2]
    batch_rank = len(batch_shape)
    batch_demand: Demand = None

    for output_demand in ctx.output_demands:
        if output_demand is None:
            continue

        output_shape = output_demand.shape
        if output_shape[:batch_rank] != batch_shape:
            return conservative_demand(ctx)

        if batch_rank == 0:
            projected = TensorDemand.full(())
        else:
            projected = TensorDemand.from_axes(
                batch_shape,
                demand_axes(output_demand)[:batch_rank],
            )

        batch_demand = merge_demands(batch_demand, projected)

    if batch_demand is None:
        return (None,)

    return (
        _expand_batch_demand(
            batch_demand,
            shape=input_shape,
            batch_shape=batch_shape,
        ),
    )
