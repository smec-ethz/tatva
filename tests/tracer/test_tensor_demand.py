import numpy as np

from tatva.tracer.demand import (
    AxisProduct,
    TensorDemand,
    _FullAxis,
    _IndexAxis,
    _RangeAxis,
)


def test_axis_selection():
    demand = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=(7, 2, 7, 3),
    )

    assert demand is not None
    assert isinstance(demand.subset, AxisProduct)
    assert demand.subset.axes == (_IndexAxis(np.array((2, 3, 7))), _FullAxis())


def test_axis_selection_range():
    demand = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=np.array(
            [2, 3, 4, 5],
            dtype=np.int64,
        ),
    )

    assert demand is not None
    assert isinstance(demand.subset, AxisProduct)
    assert demand.subset.axes == (
        _RangeAxis(start=2, stop=6),
        _FullAxis(),
    )


def test_single_index_becomes_range():
    demand = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=np.array([7]),
    )

    assert demand is not None
    assert isinstance(demand.subset, AxisProduct)
    assert demand.subset.axes[0] == _RangeAxis(start=7, stop=8)


def test_irregular_selection_is_index_axis():
    demand = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=np.array([2, 3, 7], dtype=np.int64),
    )

    assert demand is not None
    assert isinstance(demand.subset, AxisProduct)
    axis = demand.subset.axes[0]

    assert isinstance(axis, _IndexAxis)
    np.testing.assert_array_equal(
        axis.indices,
        np.array([2, 3, 7], dtype=np.int64),
    )


def test_axis_selection_full():
    demand = TensorDemand.axis_selection(
        (3, 4),
        axis=0,
        indices=(0, 1, 2),
    )

    assert demand is not None
    assert demand.is_full


def test_axis_selection_empty():
    assert (
        TensorDemand.axis_selection(
            (3, 4),
            axis=0,
            indices=(),
        )
        is None
    )


def test_rows_hull_exact():
    # Entire row 1 in a (3, 4) tensor.
    demand = TensorDemand.from_rows_hull(
        (3, 4),
        rows=(4, 5, 6, 7),
    )

    assert demand is not None
    assert demand.subset == AxisProduct(
        shape=(3, 4),
        axes=(
            _RangeAxis(1, 2),
            _FullAxis(),
        ),
    )


def test_rows_hull_cartesian_closure():
    # Coordinates:
    #
    #   (0, 0)
    #   (1, 1)
    #
    # The smallest Cartesian hull is the complete 2x2 tensor.
    demand = TensorDemand.from_rows_hull(
        (2, 2),
        rows=(0, 3),
    )

    assert demand is not None
    assert demand.is_full


def test_merge_same_axis():
    lhs = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=(1, 2),
    )
    rhs = TensorDemand.axis_selection(
        (10, 4),
        axis=0,
        indices=(5,),
    )

    assert lhs is not None
    assert rhs is not None

    merged = lhs.merge(rhs)

    assert merged.subset == AxisProduct(
        shape=(10, 4),
        axes=(
            _IndexAxis(np.array((1, 2, 5))),
            _FullAxis(),
        ),
    )


def test_merge_cross_axis_becomes_full():
    lhs = TensorDemand.axis_selection(
        (3, 4),
        axis=0,
        indices=(0,),
    )

    rhs = TensorDemand.axis_selection(
        (3, 4),
        axis=1,
        indices=(2,),
    )

    assert lhs is not None
    assert rhs is not None

    merged = lhs.merge(rhs)

    assert merged.is_full


def test_scalar_full():
    demand = TensorDemand.full(())

    assert demand is not None
    assert demand.shape == ()
    assert demand.size == 1
    assert demand.is_full


def test_adjacent_ranges_merge():
    lhs = TensorDemand.axis_range(
        (10, 4),
        axis=0,
        start=2,
        stop=5,
    )

    rhs = TensorDemand.axis_range(
        (10, 4),
        axis=0,
        start=5,
        stop=8,
    )

    assert lhs is not None
    assert rhs is not None

    merged = lhs.merge(rhs)

    assert merged.subset.axes[0] == _RangeAxis(start=2, stop=8)


def test_disjoint_ranges_merge_to_index_axis():
    lhs = TensorDemand.axis_range((10,), axis=0, start=1, stop=3)
    rhs = TensorDemand.axis_range((10,), axis=0, start=7, stop=9)

    assert lhs is not None
    assert rhs is not None

    merged = lhs.merge(rhs)
    axis = merged.subset.axes[0]

    assert isinstance(axis, _IndexAxis)

    np.testing.assert_array_equal(
        axis.indices,
        np.array([1, 2, 7, 8], dtype=np.int64),
    )
