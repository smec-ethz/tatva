import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
import sparsejac
from femsolver import Mesh
from typing import Optional
from jax import Array


def jacfwd(func, sparsity_pattern):
    return sparsejac.jacfwd(func, sparsity=sparsity_pattern)


def jacrev(func, sparsity_pattern):
    return sparsejac.jacrev(func, sparsity=sparsity_pattern)


def _create_sparse_structure(elements, n_dofs_per_node, K_shape):
    """
    Create a sparse structure for a given set of elements and constraints.
    Args:
        elements: (num_elements, nodes_per_element)
        n_dofs_per_node: Number of degrees of freedom per node
        K_shape: Shape of the matrix K
    Returns:
        data: (num_nonzero_elements,)
        indices: (num_nonzero_elements, 2)
    """

    elements = jnp.array(elements)
    num_elements, nodes_per_element = elements.shape

    # Compute all (i, j, k, l) combinations for each element
    i_idx = jnp.repeat(
        elements, nodes_per_element, axis=1
    )  # (num_elements, nodes_per_element^2)
    j_idx = jnp.tile(
        elements, (1, nodes_per_element)
    )  # (num_elements, nodes_per_element^2)

    # Expand for n_dofs_per_node
    k_idx = jnp.arange(n_dofs_per_node, dtype=jnp.int32)
    l_idx = jnp.arange(n_dofs_per_node, dtype=jnp.int32)
    k_idx, l_idx = jnp.meshgrid(k_idx, l_idx, indexing="ij")
    k_idx = k_idx.flatten()
    l_idx = l_idx.flatten()

    # For each element, get all (row, col) indices
    def element_indices(i, j):
        row = n_dofs_per_node * i + k_idx
        col = n_dofs_per_node * j + l_idx
        return row, col

    # Vectorize over all (i, j) pairs for all elements
    row_idx, col_idx = jax.vmap(element_indices)(i_idx.flatten(), j_idx.flatten())

    # Flatten and clip to matrix size
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    mask = (row_idx < K_shape[0]) & (col_idx < K_shape[1])
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]

    # Create the sparse structure
    indices = np.unique(np.vstack((row_idx, col_idx)).T, axis=0)

    data = np.ones(indices.shape[0], dtype=jnp.int32)
    sparsity_pattern = jax_sparse.BCOO((data, indices.astype(np.int32)), shape=K_shape)
    return sparsity_pattern


def get_bc_indices(sparsity_pattern: jax_sparse.BCOO, fixed_dofs: Array):
    """
    Get the indices of the fixed degrees of freedom.
    Args:
        sparsity_pattern: jax.experimental.sparse.BCOO
        fixed_dofs: (num_fixed_dofs,)
    Returns:
        zero_indices: (num_zero_indices,)
        one_indices: (num_one_indices,)
    """

    indices = sparsity_pattern.indices
    zero_indices = []
    one_indices = []

    for dof in fixed_dofs:
        indexes = np.where(indices[:, 0] == dof)[0]
        for idx in indexes:
            zero_indices.append(int(idx))

        idx = np.where(np.all(indices == np.array([dof, dof]), axis=1))[0][0]
        one_indices.append(int(idx))

    return np.array(zero_indices), np.array(one_indices)


def create_sparsity_pattern(
    mesh: Mesh, n_dofs_per_node: int, constraint_elements: Optional[Array] = None
):
    """
    Create a sparsity pattern for a given set of elements and constraints.
    Args:
        mesh: Mesh object
        n_dofs_per_node: Number of degrees of freedom per node
        constraint_elements: Optional array of constraint elements. If provided, the sparsity pattern will be created for the constraint elements.
    Returns:
        sparsity_pattern: jax.experimental.sparse.BCOO
    """

    elements = mesh.elements

    K_shape = (
        n_dofs_per_node * mesh.coords.shape[0],
        n_dofs_per_node * mesh.coords.shape[0],
    )

    sparsity_pattern = _create_sparse_structure(elements, n_dofs_per_node, K_shape)
    if constraint_elements is not None:
        sparsity_pattern_constraint = _create_sparse_structure(
            constraint_elements, n_dofs_per_node, K_shape
        )

        combined_data = np.concatenate(
            [sparsity_pattern.data, sparsity_pattern_constraint.data]
        )
        combined_indices = np.concatenate(
            [sparsity_pattern.indices, sparsity_pattern_constraint.indices]
        )
        sparsity_pattern = jax_sparse.BCOO(
            (combined_data, combined_indices), shape=K_shape
        )

    return sparsity_pattern


def create_sparsity_pattern_KKT(mesh: Mesh, n_dofs_per_node: int, B: Array):

    nb_cons = B.shape[0]

    K_sparsity_pattern = create_sparsity_pattern(mesh, n_dofs_per_node=n_dofs_per_node)
    B_sparsity_pattern = jax_sparse.BCOO.fromdense(B)

    sparsity_pattern_left = jax_sparse.bcoo_concatenate(
        [K_sparsity_pattern, B_sparsity_pattern], dimension=0
    )

    BT_sparsity_pattern = jax_sparse.BCOO.fromdense(B.T)
    C = jax_sparse.BCOO.fromdense(jnp.zeros((nb_cons, nb_cons), dtype=jnp.int32))
    sparsity_pattern_right = jax_sparse.bcoo_concatenate(
        [BT_sparsity_pattern, C], dimension=0
    )

    sparsity_pattern_KKT = jax_sparse.bcoo_concatenate(
        [sparsity_pattern_left, sparsity_pattern_right], dimension=1
    )

    return sparsity_pattern_KKT