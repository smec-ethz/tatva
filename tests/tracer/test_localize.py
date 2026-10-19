import numpy as np
import pytest

from tatva.tracer.core.routes import DynamicSliceRoute, GatherRoute, ScatterRoute
from tatva.tracer.local.demand import TensorDemand
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.local.localize import (
    localize_dynamic_slice_route,
    localize_gather_route,
    localize_scatter_route,
)


def test_localize_gather_route():
    # --------------------------------------------------------------
    # Global operand:
    #
    # shape = (10, 2)
    #
    # Local rank stores global nodes:
    #     [2, 5, 7]
    #
    # therefore:
    #
    # global scalar rows   local scalar rows
    #   4,  5       ->       0, 1
    #  10, 11       ->       2, 3
    #  14, 15       ->       4, 5
    # --------------------------------------------------------------

    operand_demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=[2, 5, 7],
    )

    assert operand_demand is not None

    operand_layout = TensorLayout.from_demand(operand_demand)

    # --------------------------------------------------------------
    # Global gather output shape = (4, 2)
    #
    # output item 0 <- node 2
    # output item 1 <- node 5
    # output item 2 <- node 7
    # output item 3 <- node 2
    #
    # Flattened source relation:
    #
    # output rows 0,1 -> global operand rows 4,5
    # output rows 2,3 -> global operand rows 10,11
    # output rows 4,5 -> global operand rows 14,15
    # output rows 6,7 -> global operand rows 4,5
    # --------------------------------------------------------------

    route = GatherRoute(
        source_rows=np.array(
            [4, 5, 10, 11, 14, 15, 4, 5],
            dtype=np.int64,
        )
    )

    # This rank stores only output items 1 and 3.
    output_demand = TensorDemand.axis_selection(
        shape=(4, 2),
        axis=0,
        indices=[1, 3],
    )

    assert output_demand is not None

    output_layout = TensorLayout.from_demand(output_demand)
    local = localize_gather_route(
        route,
        operand_layout=operand_layout,
        output_layout=output_layout,
    )

    assert local.output_shape == (2, 2)

    # Global output rows:
    #
    # item 1 -> rows 2,3 -> source 10,11 -> local 2,3
    # item 3 -> rows 6,7 -> source  4, 5 -> local 0,1
    np.testing.assert_array_equal(
        local.source_rows,
        [2, 3, 0, 1],
    )


def test_localize_gather_route_preserves_invalid_entries():
    operand_demand = TensorDemand.axis_selection(
        shape=(6,),
        axis=0,
        indices=[1, 4],
    )
    assert operand_demand is not None
    operand_layout = TensorLayout.from_demand(operand_demand)

    # Global:
    #
    # output 0 -> operand 1
    # output 1 -> invalid/fill
    # output 2 -> operand 4
    route = GatherRoute(source_rows=np.array([1, -1, 4], dtype=np.int64))

    output_demand = TensorDemand.full((3,))
    assert output_demand is not None

    output_layout = TensorLayout.from_demand(output_demand)
    local = localize_gather_route(
        route,
        operand_layout=operand_layout,
        output_layout=output_layout,
    )

    np.testing.assert_array_equal(
        local.source_rows,
        [0, -1, 1],
    )


def test_localize_gather_route_rejects_missing_live_source():
    # Rank stores only global entries 1 and 4.
    operand_demand = TensorDemand.axis_selection(
        shape=(6,),
        axis=0,
        indices=[1, 4],
    )
    assert operand_demand is not None

    operand_layout = TensorLayout.from_demand(operand_demand)

    # Output 1 requires global operand entry 3,
    # which liveness failed to include.
    route = GatherRoute(source_rows=np.array([1, 3, 4], dtype=np.int64))
    output_demand = TensorDemand.axis_selection(shape=(3,), axis=0, indices=[1])

    assert output_demand is not None
    output_layout = TensorLayout.from_demand(output_demand)

    with pytest.raises(ValueError):
        localize_gather_route(
            route,
            operand_layout=operand_layout,
            output_layout=output_layout,
        )


def test_localize_gather_ignores_sources_of_dead_outputs():
    operand_demand = TensorDemand.axis_selection(
        shape=(6,),
        axis=0,
        indices=[1],
    )
    assert operand_demand is not None

    operand_layout = TensorLayout.from_demand(operand_demand)

    route = GatherRoute(
        source_rows=np.array(
            [
                1,  # locally demanded
                3,  # not available locally
                5,  # not available locally
            ],
            dtype=np.int64,
        )
    )

    # Only output 0 survives.
    output_demand = TensorDemand.axis_selection(
        shape=(3,),
        axis=0,
        indices=[0],
    )
    assert output_demand is not None

    output_layout = TensorLayout.from_demand(output_demand)
    local = localize_gather_route(
        route, operand_layout=operand_layout, output_layout=output_layout
    )

    assert local.output_shape == (1,)
    np.testing.assert_array_equal(local.source_rows, [0])


def test_localize_scatter_route():
    # --------------------------------------------------------------
    # Global operand/output:
    #     shape = (8,)
    #
    # Local output stores global rows:
    #     [1, 3, 5, 7]
    #
    # Operand contributes rows:
    #     [1, 5, 7]
    #
    # Updates globally:
    #
    # update 0 -> output 3
    # update 1 -> output 6
    # update 2 -> output 5
    # update 3 -> output 0
    #
    # Only updates 0 and 2 survive locally.
    # --------------------------------------------------------------

    operand_demand = TensorDemand.axis_selection(
        shape=(8,),
        axis=0,
        indices=[1, 5, 7],
    )

    update_demand = TensorDemand.axis_selection(
        shape=(4,),
        axis=0,
        indices=[0, 2],
    )

    output_demand = TensorDemand.axis_selection(
        shape=(8,),
        axis=0,
        indices=[1, 3, 5, 7],
    )

    assert operand_demand is not None
    assert update_demand is not None
    assert output_demand is not None

    operand_layout = TensorLayout.from_demand(operand_demand)
    update_layout = TensorLayout.from_demand(update_demand)
    output_layout = TensorLayout.from_demand(output_demand)
    route = ScatterRoute(target_rows=np.array([3, 6, 5, 0], dtype=np.int64))
    local = localize_scatter_route(
        route,
        operand_layout=operand_layout,
        update_layout=update_layout,
        output_layout=output_layout,
    )

    assert local.operand_shape == (3,)
    assert local.update_shape == (2,)
    assert local.output_shape == (4,)

    # output local mapping:
    #
    # global 1 -> local 0
    # global 3 -> local 1
    # global 5 -> local 2
    # global 7 -> local 3
    #
    # operand global [1,5,7] -> [0,2,3]
    np.testing.assert_array_equal(local.operand_output_rows, [0, 2, 3])

    # update global 0 -> target global 3 -> local 1
    # update global 2 -> target global 5 -> local 2
    np.testing.assert_array_equal(local.target_rows, [1, 2])


def test_localize_scatter_ignores_dead_updates():
    operand_demand = TensorDemand.axis_selection(
        shape=(8,),
        axis=0,
        indices=[1, 3],
    )

    update_demand = TensorDemand.axis_selection(
        shape=(4,),
        axis=0,
        indices=[0],
    )

    output_demand = TensorDemand.axis_selection(
        shape=(8,),
        axis=0,
        indices=[1, 3],
    )

    assert operand_demand is not None
    assert update_demand is not None
    assert output_demand is not None

    route = ScatterRoute(
        target_rows=np.array(
            [
                3,  # live
                6,  # dead on this rank
                7,  # dead
                0,  # dead
            ],
            dtype=np.int64,
        )
    )

    local = localize_scatter_route(
        route,
        operand_layout=TensorLayout.from_demand(operand_demand),
        update_layout=TensorLayout.from_demand(update_demand),
        output_layout=TensorLayout.from_demand(output_demand),
    )

    np.testing.assert_array_equal(local.target_rows, [1])


def test_localize_scatter_with_dead_operand():
    # Global output has six entries.
    #
    # This rank only wants output rows 1 and 4.
    # Both are completely supplied by scatter updates, so the original
    # operand is runtime-dead.
    update_demand = TensorDemand.axis_selection(
        shape=(4,),
        axis=0,
        indices=[0, 2],
    )

    output_demand = TensorDemand.axis_selection(
        shape=(6,),
        axis=0,
        indices=[1, 4],
    )

    assert update_demand is not None
    assert output_demand is not None

    route = ScatterRoute(
        target_rows=np.array(
            [
                1,  # live update
                3,
                4,  # live update
                5,
            ],
            dtype=np.int64,
        )
    )

    local = localize_scatter_route(
        route,
        operand_layout=None,
        update_layout=TensorLayout.from_demand(update_demand),
        output_layout=TensorLayout.from_demand(output_demand),
    )

    assert local.operand_shape is None
    assert local.operand_output_rows.size == 0
    assert local.update_shape == (2,)
    assert local.output_shape == (2,)

    # Global output:
    #   1 -> local 0
    #   4 -> local 1
    #
    # update 0 targets 1
    # update 2 targets 4
    np.testing.assert_array_equal(local.target_rows, [0, 1])


def test_dead_scatter_operand_requires_update_coverage():
    update_demand = TensorDemand.axis_selection(
        shape=(2,),
        axis=0,
        indices=[0],
    )

    output_demand = TensorDemand.axis_selection(
        shape=(6,),
        axis=0,
        indices=[1, 4],
    )

    assert update_demand is not None
    assert output_demand is not None

    route = ScatterRoute(target_rows=np.array([1, 3], dtype=np.int64))

    with pytest.raises(
        ValueError,
        match="cannot produce every local output row",
    ):
        localize_scatter_route(
            route,
            operand_layout=None,
            update_layout=TensorLayout.from_demand(update_demand),
            output_layout=TensorLayout.from_demand(output_demand),
        )


def test_scatter_operand_layout_may_be_larger_than_output_layout():
    operand_demand = TensorDemand.axis_range(
        shape=(10,),
        axis=0,
        start=0,
        stop=8,
    )

    output_demand = TensorDemand.axis_selection(
        shape=(10,),
        axis=0,
        indices=[4, 7],
    )

    update_demand = TensorDemand.axis_selection(
        shape=(2,),
        axis=0,
        indices=[0],
    )

    assert operand_demand is not None
    assert output_demand is not None
    assert update_demand is not None

    route = ScatterRoute(target_rows=np.array([7, 9], dtype=np.int64))
    local = localize_scatter_route(
        route,
        operand_layout=TensorLayout.from_demand(operand_demand),
        update_layout=TensorLayout.from_demand(update_demand),
        output_layout=TensorLayout.from_demand(output_demand),
    )

    # Local operand has rows representing global 0..7.
    #
    # Only global 4 and 7 intersect the local output:
    #
    # operand local row 4 -> output local row 0
    # operand local row 7 -> output local row 1
    np.testing.assert_array_equal(local.operand_rows, [4, 7])
    np.testing.assert_array_equal(local.operand_output_rows, [0, 1])
    # Update 0 targets global 7 -> local output row 1.
    np.testing.assert_array_equal(local.update_rows, [0])
    np.testing.assert_array_equal(local.target_rows, [1])


def test_localize_dynamic_slice_route():
    operand_demand = TensorDemand.axis_selection(
        shape=(10,),
        axis=0,
        indices=[2, 4, 5, 8],
    )

    output_demand = TensorDemand.axis_selection(
        shape=(4,),
        axis=0,
        indices=[1, 2],
    )

    assert operand_demand is not None
    assert output_demand is not None

    # Global dynamic-slice relation:
    #
    # output 0 -> operand 2
    # output 1 -> operand 4
    # output 2 -> operand 5
    # output 3 -> operand 8
    route = DynamicSliceRoute(source_rows=np.array([2, 4, 5, 8], dtype=np.int64))
    local = localize_dynamic_slice_route(
        route,
        operand_layout=TensorLayout.from_demand(operand_demand),
        output_layout=TensorLayout.from_demand(output_demand),
    )

    # Operand local coordinates:
    #
    # global [2,4,5,8]
    # local  [0,1,2,3]
    #
    # local outputs represent global outputs 1,2
    # -> sources global 4,5
    # -> local rows 1,2
    np.testing.assert_array_equal(local.source_rows, [1, 2])
    assert local.output_shape == (2,)
