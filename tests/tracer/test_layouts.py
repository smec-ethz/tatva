import numpy as np
import pytest

from tatva.tracer.local.demand import (
    TensorDemand,
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
)
from tatva.tracer.local.layout import (
    TensorLayout,
    finalize_layout,
)


def test_full_layout():
    demand = TensorDemand.full((5, 2))

    assert demand is not None

    layout = finalize_layout(demand)

    assert layout.global_shape == (5, 2)
    assert layout.local_shape == (5, 2)

    assert layout.global_size == 10
    assert layout.local_size == 10
    assert layout.is_full


def test_range_layout():
    demand = TensorDemand.axis_range(
        shape=(10, 2),
        axis=0,
        start=3,
        stop=7,
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    assert layout.local_shape == (4, 2)
    assert layout.axis_subset(0) == _RangeAxis(start=3, stop=7)
    assert isinstance(layout.axis_subset(1), _FullAxis)

    np.testing.assert_array_equal(
        layout.global_to_local_axis(0, [3, 4, 6]),
        [0, 1, 3],
    )

    np.testing.assert_array_equal(
        layout.local_to_global_axis(0, [0, 1, 3]),
        [3, 4, 6],
    )


def test_index_axis_layout():
    demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=np.array([2, 5, 7], dtype=np.int64),
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    assert layout.local_shape == (3, 2)

    node_axis = layout.axis_subset(0)

    assert isinstance(node_axis, _IndexAxis)

    np.testing.assert_array_equal(
        node_axis.indices,
        [2, 5, 7],
    )
    np.testing.assert_array_equal(
        layout.global_to_local_axis(0, [2, 5, 7]),
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        layout.local_to_global_axis(0, [0, 1, 2]),
        [2, 5, 7],
    )


def test_global_axis_coordinate_not_local():
    demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=[2, 5, 7],
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    np.testing.assert_array_equal(
        layout.global_to_local_axis_or_missing(0, [2, 3, 5, 9]),
        [0, -1, 1, -1],
    )

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        layout.global_to_local_axis(0, [2, 3])


def test_global_rows_to_local_rows():
    demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=[2, 5, 7],
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    global_rows = np.array([4, 5, 10, 11, 14, 15], dtype=np.int64)

    local_rows = layout.global_rows_to_local_rows(global_rows)

    np.testing.assert_array_equal(
        local_rows,
        [0, 1, 2, 3, 4, 5],
    )


def test_local_rows_to_global_rows():
    demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=[2, 5, 7],
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    np.testing.assert_array_equal(
        layout.local_rows_to_global_rows([0, 1, 2, 3, 4, 5]),
        [4, 5, 10, 11, 14, 15],
    )


def test_global_rows_missing_from_layout():
    demand = TensorDemand.axis_selection(
        shape=(10, 2),
        axis=0,
        indices=[2, 5, 7],
    )

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    # global row 6 = coordinate (3, 0),
    # and global node 3 is not stored.
    with pytest.raises(ValueError, match="not stored"):
        layout.global_rows_to_local_rows([4, 6])

    np.testing.assert_array_equal(
        layout.global_rows_to_local_rows(
            [4, 6, -1],
            allow_missing=True,
        ),
        [0, -1, -1],
    )


def test_scalar_layout():
    demand = TensorDemand.full(())

    assert demand is not None

    layout = TensorLayout.from_demand(demand)

    assert layout.global_shape == ()
    assert layout.local_shape == ()

    assert layout.global_size == 1
    assert layout.local_size == 1

    np.testing.assert_array_equal(
        layout.global_rows_to_local_rows([0]),
        [0],
    )

    np.testing.assert_array_equal(
        layout.local_rows_to_global_rows([0]),
        [0],
    )
