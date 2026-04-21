import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva.compound import Compound, FieldSize, FieldType, field
from tatva.mesh import Mesh, PartitionInfo

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py required")


def test_global_indices():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    if rank == 0:
        l2g = np.array([0, 1, 2], dtype=np.int32)
        n_owned = 2
    elif rank == 1:
        l2g = np.array([2, 3, 1], dtype=np.int32)
        n_owned = 2
    else:
        l2g = np.array([], dtype=np.int32)
        n_owned = 0

    p_info = PartitionInfo(nodes_local_to_global=l2g, n_owned_nodes=n_owned)
    mock_mesh = Mesh(
        coords=jnp.zeros((len(l2g), 1)), elements=jnp.zeros((0, 2), dtype=int)
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info):
        u = field(shape=(FieldSize.AUTO, 2), field_type=FieldType.NODAL)
        l = field(
            shape=(2, 1),
            field_type=FieldType.NODAL,
            nodal_local_to_global=jnp.array([1, 3]),
        )

    # Full field u (global shape 4x2)
    np.testing.assert_array_equal(MyState.g.u[0], [0, 1])
    np.testing.assert_array_equal(MyState.g.u[3, 1], [7])

    # Test class-level access
    np.testing.assert_array_equal(MyState.g.u[0], [0, 1])

    # Incomplete field l (global shape 2x1, subset [1, 3])
    # l starts after u (8 DOFs)
    np.testing.assert_array_equal(MyState.g.l[1], [8])
    np.testing.assert_array_equal(MyState.g.l[3], [9])

    with pytest.raises(IndexError):
        MyState.g.l[0]


if __name__ == "__main__":
    test_global_indices()
