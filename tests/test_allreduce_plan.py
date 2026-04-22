import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
    import mpi4jax  # noqa: F401
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from tatva import sparse
from tatva.mesh import Mesh
from tatva.mpi import (
    AllreducePlan,
)

pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py and mpi4jax required")

N_DOFS_PER_NODE = 2


def test_allreduce_plan_allgather():
    """make_allgather reconstructs the full vector from owned slices."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("requires at least 2 MPI ranks")

    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    elements = jnp.array([[0, 1], [1, 2]])  # 2 elements: (0-1) and (1-2)
    mesh = Mesh(coords=coords, elements=elements)
    sparsity_pattern = sparse.create_sparsity_pattern(
        mesh, n_dofs_per_node=N_DOFS_PER_NODE
    )

    plan = AllreducePlan(global_sparsity_pattern=sparsity_pattern, comm=comm)
    allgather = plan.make_allgather()

    if rank == 0:
        print(f"Rank 0 {plan.local_size}, {plan.global_size}")
        x_owned = jnp.ones(plan.local_size)
    else:
        print(f"Rank 1 {plan.local_size}, {plan.global_size}")
        x_owned = jnp.ones(plan.local_size) * 40.0

    u_full = allgather(x_owned)
    np.testing.assert_allclose(u_full, [1.0, 1.0, 1.0, 40.0, 40.0, 40.0])


def test_allreduce_plan_grad():
    """make_allreduce sums local contributions and returns the owned slice."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("requires at least 2 MPI ranks")

    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    elements = jnp.array([[0, 1], [1, 2]])  # 2 elements: (0-1) and (1-2)
    mesh = Mesh(coords=coords, elements=elements)

    sparsity_pattern = sparse.create_sparsity_pattern(
        mesh, n_dofs_per_node=N_DOFS_PER_NODE
    )

    plan = AllreducePlan(global_sparsity_pattern=sparsity_pattern, comm=comm)

    # local_fn returns u_full * (rank+1): rank 0 contributes *1, rank 1 *2
    # allreduce sum → u_full * 3, then each rank slices its owned rows

    def local_fn(u_full):
        return u_full

    grad_fn = plan.make_allreduce_owned(local_fn)
    u_full = None
    if rank == 0:
        u_full = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    if rank == 1:
        u_full = jnp.array([0.0, 0.0, 0.25, 0.25, 4.0, 5.0])
    result = grad_fn(u_full)
    print(f"Rank {rank} local_fn output: {local_fn(u_full)}")
    print(f"Rank {rank} result: {result}")

    expected_full = jnp.array([0.0, 1.0, 2.25, 3.25, 4.0, 5.0])
    if rank == 0:
        np.testing.assert_allclose(result, expected_full[0:3])
    elif rank == 1:
        np.testing.assert_allclose(result, expected_full[3:6])


def test_allreduce_plan_hessian():
    """make_allreduce sums local contributions and returns the owned slice."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("requires at least 2 MPI ranks")

    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    elements = jnp.array([[0, 1], [1, 2]])  # 2 elements: (0-1) and (1-2)
    mesh = Mesh(coords=coords, elements=elements)

    sparsity_pattern = sparse.create_sparsity_pattern(
        mesh, n_dofs_per_node=N_DOFS_PER_NODE
    )

    plan = AllreducePlan(global_sparsity_pattern=sparsity_pattern, comm=comm)

    def local_fn(K_full):
        return K_full

    hessian_fn = plan.make_allreduce_owned(local_fn)
    K_full = jnp.zeros((6, 6))
    if rank == 0:
        K_full = K_full.at[:3, :3].set(jnp.eye(3))
    if rank == 1:
        K_full = K_full.at[-3:, -3:].set(jnp.eye(3) * 0.25)
    result = hessian_fn(K_full)
    print(f"Rank {rank} local_fn output: {local_fn(K_full)}")
    print(f"Rank {rank} result: {result.shape}")

    expected_full = jnp.zeros((6, 6))
    expected_full_0 = expected_full.at[:3, :3].set(jnp.eye(3))
    expected_full_1 = expected_full.at[-3:, -3:].set(jnp.eye(3) * 0.25)
    expected_full = expected_full_0 + expected_full_1

    if rank == 0:
        np.testing.assert_allclose(result, expected_full[:3, :])
    elif rank == 1:
        np.testing.assert_allclose(result, expected_full[-3:, :])


if __name__ == "__main__":
    test_allreduce_plan_allgather()
    test_allreduce_plan_grad()
    test_allreduce_plan_hessian()
