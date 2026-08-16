import numpy as np
import scipy.sparse as sps

from tatva.tracer.lowering.partition import (
    _contiguous_owners,
    dependency_partition_owners,
)
from tatva.tracer.program.dependencies import DependencySet


def test_contiguous_owners():
    owners = _contiguous_owners(10, 3)
    np.testing.assert_array_equal(
        owners,
        np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
    )


def test_dependency_partition_owners():
    csr = sps.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
    )
    dep = DependencySet(csr, (4,))

    owners = dependency_partition_owners(
        dep,
        axis=0,
        dof_to_part=np.array([0, 0, 0, 1, 1, 1]),
        n_parts=2,
    )

    np.testing.assert_array_equal(
        owners,
        np.array(
            [
                0,  # 2 vs 0
                1,  # 1 vs 2
                1,  # tie; contiguous preferred owner is 1
                1,  # no deps; contiguous owner
            ]
        ),
    )
