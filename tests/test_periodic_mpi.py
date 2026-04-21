import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps
from tatva.lifter import Lifter, PeriodicMPI
from tatva.mpi import _LocalLayout
from tatva.compound import Compound, field, FieldType
from tatva.mesh import PartitionInfo, Mesh

jax.config.update("jax_enable_x64", True)

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py required")


def test_periodic_mpi_augment_sparsity_ghost_slave():
    """Test that sparsity augmentation works when the slave is a ghost on this rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Mock setup: 3 nodes total.
    # Rank 0: owns [0], ghosts [1, 2]
    # Rank 1: owns [1, 2], ghosts [0]
    # Periodic: 1 -> 2

    if rank == 0:
        l2g = np.array([0, 1, 2], dtype=np.int32)
        n_owned = 1
    else:
        l2g = np.array([1, 2, 0], dtype=np.int32)
        n_owned = 2

    p_info = PartitionInfo(nodes_local_to_global=l2g, n_owned_nodes=n_owned)
    mock_mesh = Mesh(
        coords=jnp.zeros((len(l2g), 1)), elements=jnp.zeros((0, 2), dtype=int)
    )

    class MyState(Compound, mesh=mock_mesh, partition_info=p_info, comm=comm):
        u = field(shape=(len(l2g), 1), field_type=FieldType.NODAL)

    layout = MyState.get_layout()

    # Periodic: global node 1 is slave of global node 2
    # Local indices on Rank 0: node 1 is index 1, node 2 is index 2
    # Node 1 is a GHOST on Rank 0.

    dofs = jnp.array([1], dtype=jnp.int32)  # Global node 1
    master_dofs = jnp.array([2], dtype=jnp.int32)  # Global node 2

    cond = PeriodicMPI(dofs, master_dofs, layout, comm=comm)
    lifter = Lifter(MyState.size, cond)

    # Augment layout
    layout_aug, lifter_aug = lifter.augment_layout(layout, comm)

    # Create a local sparsity pattern on Rank 0 connecting node 0 and node 1
    # On Rank 0, local index 0 (global 0) and local index 1 (global 1) are connected.
    if rank == 0:
        sparsity = sps.csr_matrix((3, 3), dtype=np.int8)
        sparsity[0, 1] = 1
        sparsity[1, 0] = 1
        sparsity[0, 0] = 1
        sparsity[1, 1] = 1
        sparsity = sparsity.tocsr()

        # Augment sparsity
        sparsity_aug = lifter_aug.augment_sparsity(sparsity)

        # We expect node 2 (master) to be connected to node 0 on Rank 0.
        # Local node 2 is global node 2.
        # So sparsity_aug[0, 2] should be 1.
        assert sparsity_aug[0, 2] != 0, (
            "Sparsity pattern missing coupling for ghost slave"
        )
        assert sparsity_aug[2, 0] != 0, (
            "Sparsity pattern missing symmetric coupling for ghost slave"
        )
    else:
        # Rank 1 owns 1 and 2.
        pass


if __name__ == "__main__":
    test_periodic_mpi_augment_sparsity_ghost_slave()
