from __future__ import annotations

from jax.extend.core import JaxprEqn

from tatva.tracer.core.semantics import (
    CallTarget,
    DemandContext,
    conservative_demand,
)
from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.demand import (
    Demand,
    TensorDemand,
    _FullAxis,
    merge_demands,
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
        demand.axes[:batch_rank],
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
        batch_demand.axes + tuple(_FullAxis() for _ in range(len(shape) - batch_rank)),
    )


def custom_linear_solve_primal_inputs(eqn: JaxprEqn) -> tuple[int, ...]:
    """Outer operands consumed by the captured primal ``solve`` JAXPR."""
    lengths = eqn.params["const_lengths"]

    n_matvec = int(lengths.matvec)
    n_vecmat = int(lengths.vecmat)
    n_solve = int(lengths.solve)
    n_transpose_solve = int(lengths.transpose_solve)

    solve_start = n_matvec + n_vecmat
    rhs_start = solve_start + n_solve + n_transpose_solve

    solve_inputs = tuple(range(solve_start, solve_start + n_solve))
    rhs_inputs = tuple(range(rhs_start, len(eqn.invars)))

    return solve_inputs + rhs_inputs


def custom_linear_solve_call_target(eqn: JaxprEqn) -> CallTarget:
    """Expose the captured primal solve as an ordinary nested call body."""
    jaxpr = eqn.params["jaxprs"]

    return CallTarget(
        body=jaxpr.solve,
        input_indices=custom_linear_solve_primal_inputs(eqn),
    )


def triangular_solve_demand(ctx: DemandContext) -> tuple[Demand, ...]:
    """Preserve batch locality while conservatively keeping solve axes full.

    This intentionally supports only the non-broadcasted batch case for now.
    That is enough for the LU solve emitted by ``jnp.linalg.inv`` and avoids
    guessing about general batch-broadcast semantics.
    """
    if len(ctx.eqn.invars) != 2 or len(ctx.output_demands) != 1:
        return conservative_demand(ctx)

    output_demand = ctx.output_demands[0]
    if output_demand is None:
        return (None, None)

    a_shape = _shape_of(ctx.eqn.invars[0])
    b_shape = _shape_of(ctx.eqn.invars[1])
    out_shape = _shape_of(ctx.eqn.outvars[0])

    if len(a_shape) < 2 or len(b_shape) < 2 or len(out_shape) < 2:
        return conservative_demand(ctx)

    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    out_batch = out_shape[:-2]

    if a_batch != b_batch or b_batch != out_batch:
        return conservative_demand(ctx)

    batch_demand = _project_batch_demand(
        output_demand,
        batch_shape=out_batch,
    )

    return (
        _expand_batch_demand(
            batch_demand,
            shape=a_shape,
            batch_shape=a_batch,
        ),
        _expand_batch_demand(
            batch_demand,
            shape=b_shape,
            batch_shape=b_batch,
        ),
    )


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
                output_demand.axes[:batch_rank],
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
