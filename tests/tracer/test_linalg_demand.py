from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tatva.tracer.demand import (
    TensorDemand,
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
)
from tatva.tracer.rules.linalg import (
    _recognize_batched_lu_solve,
    custom_linear_solve_demand,
)
from tatva.tracer.semantics import DemandContext


def _find_eqn(jaxpr, primitive_name: str):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive_name:
            return eqn

        # jnp.linalg.solve is commonly wrapped in jit.
        nested = eqn.params.get("jaxpr")

        if nested is not None:
            child = getattr(nested, "jaxpr", nested)

            try:
                return _find_eqn(child, primitive_name)
            except LookupError:
                pass

    raise LookupError(f"primitive {primitive_name!r} not found")


def test_custom_linear_solve_preserves_batch_demand():
    batch = 8
    n = 2

    def f(a, b):
        return jnp.linalg.solve(a, b)

    # Full-rank matrices. Numerical values themselves don't matter for
    # demand analysis, but avoiding singular matrices makes tracing/testing
    # less surprising if evaluation happens anywhere in the helper stack.
    a = np.broadcast_to(np.eye(n), (batch, n, n)).copy() * 2.0
    b = np.arange(batch * n * n, dtype=np.float64).reshape(batch, n, n)

    closed = jax.make_jaxpr(f)(jnp.asarray(a), jnp.asarray(b))

    eqn = _find_eqn(closed.jaxpr, "custom_linear_solve")

    # Demand only batches 1, 2 and 6.
    output_demand = TensorDemand.axis_selection(
        shape=(batch, n, n),
        axis=0,
        indices=np.array(
            [1, 2, 6],
            dtype=np.int64,
        ),
    )

    assert output_demand is not None

    input_demands = custom_linear_solve_demand(
        DemandContext(
            eqn=eqn,
            output_demands=(output_demand,),
            route=None,
        )
    )

    # Current JAX custom_linear_solve layout:
    #
    # 0: matvec const
    # 1: vecmat const
    # 2: solve const 0 = LU factors
    # 3: solve const 1 = pivots
    # 4: transpose-solve const 0
    # 5: transpose-solve const 1
    # 6: RHS
    #
    assert len(input_demands) == 7

    assert input_demands[0] is None
    assert input_demands[1] is None

    assert input_demands[4] is None
    assert input_demands[5] is None

    factors = input_demands[2]
    pivots = input_demands[3]
    rhs = input_demands[6]

    assert factors is not None
    assert pivots is not None
    assert rhs is not None

    # The selected batch indices must survive rather than widening to all
    # eight systems.
    factors_batch = factors.axis_subset(0)
    pivots_batch = pivots.axis_subset(0)
    rhs_batch = rhs.axis_subset(0)

    expected = np.array([1, 2, 6], dtype=np.int64)

    assert isinstance(factors_batch, _IndexAxis)
    assert isinstance(pivots_batch, _IndexAxis)
    assert isinstance(rhs_batch, _IndexAxis)

    np.testing.assert_array_equal(factors_batch.indices, expected)
    np.testing.assert_array_equal(pivots_batch.indices, expected)
    np.testing.assert_array_equal(rhs_batch.indices, expected)

    # Local matrix/vector dimensions must be complete.
    assert isinstance(factors.axis_subset(1), _FullAxis)
    assert isinstance(factors.axis_subset(2), _FullAxis)

    assert isinstance(pivots.axis_subset(1), _FullAxis)

    assert isinstance(rhs.axis_subset(1), _FullAxis)
    assert isinstance(rhs.axis_subset(2), _FullAxis)


def test_custom_linear_solve_preserves_contiguous_batch_range():
    batch = 8
    n = 2

    def f(a, b):
        return jnp.linalg.solve(a, b)

    a = np.broadcast_to(2.0 * np.eye(n), (batch, n, n)).copy()
    b = np.ones((batch, n, n), dtype=np.float64)

    closed = jax.make_jaxpr(f)(jnp.asarray(a), jnp.asarray(b))
    eqn = _find_eqn(closed.jaxpr, "custom_linear_solve")

    output_demand = TensorDemand.axis_range(
        shape=(batch, n, n), axis=0, start=2, stop=6
    )

    assert output_demand is not None

    demands = custom_linear_solve_demand(
        DemandContext(eqn=eqn, output_demands=(output_demand,), route=None)
    )

    for index in (2, 3, 6):
        demand = demands[index]
        assert demand is not None
        assert demand.axis_subset(0) == _RangeAxis(start=2, stop=6)

    for index in (0, 1, 4, 5):
        assert demands[index] is None


def test_custom_linear_solve_recognizer_rejects_other_primitive():
    def f(x):
        return x + 1.0

    closed = jax.make_jaxpr(f)(jnp.ones((8, 2, 2)))

    eqn = closed.jaxpr.eqns[0]
    ctx = DemandContext(
        eqn=eqn,
        output_demands=(TensorDemand.full((8, 2, 2)),),
        route=None,
    )

    assert _recognize_batched_lu_solve(ctx) is None
