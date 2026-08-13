import numpy as np

from tatva.tracer.demand import TensorDemand
from tatva.tracer.halo import build_halo_plans, build_local_halo_plan
from tatva.tracer.layout import TensorLayout


class _ReferenceComm:
    def __init__(self, reference_plans, rank):
        self._plans = reference_plans
        self._rank = rank

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return len(self._plans)

    def alltoall(self, sendobj):
        reference = self._plans[self._rank]
        expected = {exchange.peer: exchange.global_dofs for exchange in reference.recv}

        for peer, values in enumerate(sendobj):
            np.testing.assert_array_equal(
                values,
                expected.get(peer, np.empty(0, dtype=np.int64)),
            )

        incoming = [np.empty(0, dtype=np.int64) for _ in range(len(self._plans))]
        for exchange in reference.send:
            incoming[exchange.peer] = exchange.global_dofs
        return incoming


def _layout(shape, indices):
    demand = TensorDemand.axis_selection(shape, axis=0, indices=indices)
    assert demand is not None
    return TensorLayout.from_demand(demand)


def test_local_halo_collective_matches_all_rank_reference():
    layouts = (
        _layout((8,), [0, 1, 4]),
        _layout((8,), [1, 2, 3, 6]),
        _layout((8,), [3, 5, 6, 7]),
    )
    owner = np.array([0, 0, 1, 1, 2, 2, 2, 0], dtype=np.int64)
    reference = build_halo_plans(layouts, owner)

    for rank, layout in enumerate(layouts):
        local = build_local_halo_plan(
            layout,
            owner,
            comm=_ReferenceComm(reference, rank),
        )
        expected = reference[rank]

        np.testing.assert_array_equal(local.compute_global, expected.compute_global)
        np.testing.assert_array_equal(local.compute_rows, expected.compute_rows)
        np.testing.assert_array_equal(local.owned_global, expected.owned_global)
        np.testing.assert_array_equal(local.ghost_global, expected.ghost_global)

        assert [exchange.peer for exchange in local.recv] == [
            exchange.peer for exchange in expected.recv
        ]
        assert [exchange.peer for exchange in local.send] == [
            exchange.peer for exchange in expected.send
        ]
        for actual, wanted in zip(local.recv, expected.recv, strict=True):
            np.testing.assert_array_equal(actual.global_dofs, wanted.global_dofs)
            np.testing.assert_array_equal(actual.local_rows, wanted.local_rows)
        for actual, wanted in zip(local.send, expected.send, strict=True):
            np.testing.assert_array_equal(actual.global_dofs, wanted.global_dofs)
            np.testing.assert_array_equal(actual.local_rows, wanted.local_rows)
