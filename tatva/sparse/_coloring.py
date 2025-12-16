# Copyright (C) 2025 ETH Zurich (SMEC)
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

from __future__ import annotations

from functools import partial
from multiprocessing import Array
from typing import Callable

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array


def get_distance2_adjacency_fast(n_dofs, row_ptr, col_idx):
    """
    Creates the adjacency list for the 'squared' graph.
    Input: Your standard CSR arrays.
    Output: Adjacency list where 'neighbors' include neighbors-of-neighbors.
    """
    # 1. Build Scipy Sparse Matrix (Structure only, boolean)
    # We use boolean data to save memory
    data = np.ones(len(col_idx), dtype=bool)
    A = sp.csr_matrix((data, col_idx, row_ptr), shape=(n_dofs, n_dofs))

    # 2. Compute A * A (Symbolic Squared Graph)
    # This finds all nodes reachable in 2 steps
    A2 = A @ A

    # 3. Convert to Adjacency List (Your preferred format)
    # We use the existing csr_to_adjacency logic but on A2
    return csr_to_adjacency_fast(n_dofs, A2.indptr, A2.indices, symmetric=False)


def csr_to_adjacency_fast(n_dofs, row_ptr, col_idx, symmetric=True):
    """
    Vectorized conversion of CSR to Adjacency List.
    Removes the Python loop entirely.
    """
    # 1. Expand row pointers to get row index for every non-zero entry
    # row_ptr: [0, 2, 5...] -> row_indices: [0, 0, 1, 1, 1...]
    diffs = np.diff(row_ptr)
    rows = np.repeat(np.arange(n_dofs), diffs)
    cols = col_idx

    if symmetric:
        # 2. Symmetrize by concatenating (rows, cols) with (cols, rows)
        # This creates the undirected graph edges
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
    else:
        all_rows = rows
        all_cols = cols

    # 3. Sort by Row, then by Column
    # This groups all neighbors of node 0, then node 1, etc.
    # We use a structured array or lexsort to sort pairs
    # Lexsort sorts by last key first (so sort by cols, then rows)
    sorter = np.lexsort((all_cols, all_rows))
    sorted_rows = all_rows[sorter]
    sorted_cols = all_cols[sorter]

    # 4. Remove Duplicates
    # Identify unique edges. An edge is unique if it's different from the previous one
    # We check both row and col changes.
    # (Fast boolean mask shift)
    mask = np.empty(len(sorted_rows), dtype=bool)
    mask[0] = True
    mask[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
        sorted_cols[1:] != sorted_cols[:-1]
    )

    unique_rows = sorted_rows[mask]
    unique_cols = sorted_cols[mask]

    # 5. Split into List of Arrays
    # We need to find where the row index changes to split the columns array
    # np.unique with return_counts or return_index is fast
    _, split_indices = np.unique(unique_rows, return_index=True)

    # np.split creates the list of arrays [ [neighs_0], [neighs_1], ... ]
    # We skip the first index (0) as split expects split points
    adjacency = np.split(unique_cols, split_indices[1:])

    # Handle case where some nodes might be isolated (empty lists)
    # The split method above works for contiguous nodes.
    # If strictly returning list of length n_dofs:
    final_adj = [np.array([], dtype=np.int32)] * n_dofs

    # We can assign easily if we iterate, but to stay fully vectorized involves
    # slightly more complex masking.
    # Given n_dofs is large, list comprehension over the splits is fine
    # provided the unique_rows covers all nodes.
    # If nodes are missing (isolated), we need to be careful.

    # Robust/Fastest way to ensure alignment with n_dofs:
    # Re-calculate a new row_ptr for the symmetric matrix
    # bincount gives number of neighbors per node
    new_row_counts = np.bincount(unique_rows, minlength=n_dofs)
    new_row_ptr = np.concatenate(([0], np.cumsum(new_row_counts)))

    # A generic loop to slice is actually faster than np.split for 1M items
    # because np.split creates a lot of view overhead in one function call.
    # But for simplicity, let's just return the list comprehension:

    final_adj = [
        unique_cols[new_row_ptr[i] : new_row_ptr[i + 1]] for i in range(n_dofs)
    ]

    return final_adj


def greedy_coloring(adjacency):
    n = len(adjacency)
    colors = -np.ones(n, dtype=np.int32)
    for i in range(n):
        # Check neighbors
        used = {colors[j] for j in adjacency[i] if colors[j] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    return colors


def seeds_from_coloring(colors):
    unique = jnp.unique(colors)
    seeds = []
    for c in unique:
        mask = colors == c
        seeds.append(jnp.array(mask, dtype=jnp.float64))
    return seeds


def distance2_color_and_seeds(row_ptr: Array, col_idx: Array, n_dofs: int):
    """
    Compute distance-2 coloring and corresponding seed vectors.

    Args:
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
        n_dofs: Number of degrees of freedom (size of the matrix)
    Returns:
        colors: Array of colors assigned to each DOF
        seeds: List of seed vectors for each color
    """
    adjacency = get_distance2_adjacency_fast(
        n_dofs, np.array(row_ptr), np.array(col_idx)
    )
    colors = greedy_coloring(adjacency)
    seeds = seeds_from_coloring(jnp.array(colors))
    return jnp.array(colors), jnp.stack(seeds, axis=0)


@partial(jax.jit, static_argnames=["F"])
def colored_jacobian(F: Callable, u: Array, seeds: Array) -> Array:
    """
    Compute the colored Jacobian of F at u using the provided seeds.
    Args:
        F: function to differentiate
        u: Point at which to evaluate the Jacobian, shape (N,)
        seeds: List of seed vectors, each of shape (N,). Length = n_colors
    Returns:

    """
    _, f_jvp = jax.linearize(F, u)

    def _jvp(s):
        return f_jvp(s)

    rows = jax.vmap(_jvp)(seeds)
    J = jnp.stack(rows, axis=1)
    return J


@partial(jax.jit, static_argnames=["n_dofs"])
def recover_stiffness_matrix(
    J_compressed: Array,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
    n_dofs: int,
) -> jsp.BCOO:
    """
    Recover the exact values from the compressed Jacobian and build BCOO.

    Args:
        J_compressed: Output from colored_jacobian, shape (N, n_colors)
        row_ptr, col_indices: The ORIGINAL sparsity pattern of the matrix
        colors: The array of colors used for compression

    Returns:
        K_bcoo: The recovered sparse Jacobian in BCOO format
    """

    # expand row pointers to get row indices for every non-zero
    diffs = jnp.diff(row_ptr)
    rows = jnp.repeat(jnp.arange(n_dofs), diffs, total_repeat_length=len(col_indices))
    cols = col_indices

    # find where the value for (i, j) is hiding in J_compressed
    # The value K_ij is stored at row 'i' and column 'color[j]'
    col_colors = colors[cols]

    # extract values using fancy indexing
    # values[k] = J_compressed[ rows[k], col_colors[k] ]
    values = J_compressed[rows, col_colors]

    # construct BCOO Matrix
    indices = jnp.stack([rows, cols], axis=1)

    # BCOO requires explicit shape
    K_bcoo = jsp.BCOO((values, indices), shape=(n_dofs, n_dofs))

    return K_bcoo


def jacfwd(
    gradient: Callable,
    seeds: Array,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
) -> Callable:
    """
    Compute the sparse Jacobian using forward-mode automatic differentiation
    and graph coloring.

    Args:
        gradient: Function whose Jacobian is to be computed
        seeds: List of seed vectors for coloring
        row_ptr, col_indices: The ORIGINAL sparsity pattern of the matrix
        colors: The array of colors used for compression
    Returns:
        A function that computes the sparse Jacobian in BCOO format

    """

    def _wraped_jacfwd(u: Array) -> jsp.BCOO:
        J_compressed = colored_jacobian(gradient, u, seeds)
        n_dofs = J_compressed.shape[0]

        K_bcoo = recover_stiffness_matrix(
            J_compressed, row_ptr, col_indices, colors, n_dofs
        )
        return K_bcoo

    return _wraped_jacfwd
