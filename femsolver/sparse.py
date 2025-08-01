import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
 

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
    sparsity_pattern = jax_sparse.BCOO(
        (data, indices.astype(np.int32)), shape=K_shape
    )
    return sparsity_pattern


def get_bc_indices(indices, fixed_dofs):
    """
    Get the indices of the fixed degrees of freedom.
    Args:
        indices: (num_nonzero_elements, 2)
        fixed_dofs: (num_fixed_dofs,)
    Returns:
        zero_indices: (num_zero_indices,)
        one_indices: (num_one_indices,)
    """
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
    elements, n_dofs_per_node, K_shape, constraint_elements=None
):
    """
    Create a sparsity pattern for a given set of elements and constraints.
    Args:
        elements: (num_elements, nodes_per_element)
        n_dofs_per_node: Number of degrees of freedom per node
        K_shape: Shape of the matrix K
        constraint_elements: (num_constraint_elements, nodes_per_constraint_element)
    Returns:
        sparsity_pattern: jax_sparse.BCOO
    """
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
