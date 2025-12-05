# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import sparsejac
from jax import Array
from jax.experimental import sparse as jax_sparse
from jax.experimental.sparse.bcoo import BCOO

from tatva import Mesh


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
    sparsity_pattern = jax_sparse.BCOO((data, indices.astype(np.int32)), shape=K_shape)  # type: ignore
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
    mesh: Mesh,
    n_dofs_per_node: int,
    K_shape: Optional[Tuple[int, int]] = None,
    constraint_elements: Optional[Array] = None,
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

    if K_shape is None:
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
            (combined_data, combined_indices),  # type: ignore
            shape=K_shape,
        )

    return sparsity_pattern


def create_sparsity_pattern_KKT(mesh: Mesh, n_dofs_per_node: int, B: Array):
    """
    Create a sparsity pattern for the KKT system.
    Args:
        mesh: Mesh object
        n_dofs_per_node: Number of degrees of freedom per node
        B: Constraint matrix (nb_cons, n_dofs)
    Returns:
        sparsity_pattern_KKT: jax.experimental.sparse.BCOO
    """

    nb_cons = B.shape[0]

    K_sparsity_pattern = create_sparsity_pattern(mesh, n_dofs_per_node=n_dofs_per_node)
    B_sparsity_pattern = jax_sparse.BCOO.fromdense(B).astype(jnp.int32)

    sparsity_pattern_left = jax_sparse.bcoo_concatenate(
        [K_sparsity_pattern, B_sparsity_pattern], dimension=0
    )

    BT_sparsity_pattern = jax_sparse.BCOO.fromdense(B.T).astype(jnp.int32)
    C = jax_sparse.BCOO.fromdense(jnp.eye(nb_cons, nb_cons, dtype=jnp.int32))
    sparsity_pattern_right = jax_sparse.bcoo_concatenate(
        [BT_sparsity_pattern, C], dimension=0
    )

    sparsity_pattern_KKT = jax_sparse.bcoo_concatenate(
        [sparsity_pattern_left, sparsity_pattern_right], dimension=1
    )

    return sparsity_pattern_KKT


def reduce_sparsity_pattern(pattern: BCOO, free_dofs: Array) -> BCOO:
    """Reduce a sparse matrix pattern to only the free dofs (for K_ff).

    Args:
        pattern (BCOO): Sparse matrix pattern in BCOO format on the full
            set of dofs.
        free_dofs: Array of free dofs as integer indices.

    Returns:
        BCOO: Reduced sparse matrix pattern with rows and columns remapped
            to the reduced indexing of free dofs.
    """
    # Pull to host (avoid device OOM for big masks)
    I = np.asarray(pattern.indices[:, 0])  # noqa: E741
    J = np.asarray(pattern.indices[:, 1])
    D = np.asarray(pattern.data)

    n_full = int(pattern.shape[0])
    free = np.asarray(free_dofs, dtype=np.int64)

    # Membership mask: O(n_full) setup, O(nnz) index
    is_free = np.zeros(n_full, dtype=bool)
    is_free[free] = True
    mask = is_free[I] & is_free[J]

    I = I[mask]  # noqa: E741
    J = J[mask]
    D = D[mask]

    # Full -> reduced reindex
    index_map = -np.ones(n_full, dtype=np.int64)
    index_map[free] = np.arange(free.size, dtype=np.int64)
    I_red = index_map[I]
    J_red = index_map[J]

    # Deduplicate (sum data; for pure pattern set to 1)
    keys = I_red * free.size + J_red
    uniq, inv = np.unique(keys, return_inverse=True)
    # accumulate
    D_red = np.bincount(inv, weights=D, minlength=uniq.size)
    I_red = (uniq // free.size).astype(np.int32)
    J_red = (uniq % free.size).astype(np.int32)

    # Back to JAX
    indices_red = jnp.stack([jnp.asarray(I_red), jnp.asarray(J_red)], axis=1)
    data_red = jnp.asarray(D_red)
    shape = (free.size, free.size)

    return BCOO((data_red, indices_red), shape=shape)
