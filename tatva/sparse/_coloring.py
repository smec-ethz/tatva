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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array


def get_distance2_adjacency(n_dofs, row_ptr, col_idx):
    """
    Creates the adjacency list for the 'squared' graph.
    Args:
        n_dofs: Number of degrees of freedom (nodes)
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
    Returns:
        adjacency: Adjacency list where 'neighbors' include neighbors-of-neighbors.
    """
    # build Scipy Sparse Matrix (Structure only, boolean)
    # We use boolean data to save memory
    data = np.ones(len(col_idx), dtype=bool)
    A = sp.csr_matrix((data, col_idx, row_ptr), shape=(n_dofs, n_dofs))

    # compute A * A (Symbolic Squared Graph)
    # this finds all nodes reachable in 2 steps
    A2 = A @ A

    # convert to Adjacency List
    # we use the existing csr_to_adjacency logic but on A2
    return csr_to_adjacency(n_dofs, A2.indptr, A2.indices, symmetric=False)


def csr_to_adjacency(n_dofs, row_ptr, col_idx, symmetric=True):
    """
    Vectorized conversion of CSR to Adjacency List.
    Removes the Python loop entirely.
    Args:
        n_dofs: Number of degrees of freedom (nodes)
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
        symmetric: Whether to symmetrize the adjacency (undirected graph)
    Returns:
        adjacency: List of arrays, where each array contains the neighbors of the node
    """
    # expand row pointers to get row index for every non-zero entry
    # row_ptr: [0, 2, 5...] -> row_indices: [0, 0, 1, 1, 1...]
    diffs = np.diff(row_ptr)
    rows = np.repeat(np.arange(n_dofs), diffs)
    cols = col_idx

    if symmetric:
        # symmetrize by concatenating (rows, cols) with (cols, rows)
        # This creates the undirected graph edges
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
    else:
        all_rows = rows
        all_cols = cols

    # sort by Row, then by Column
    # This groups all neighbors of node 0, then node 1, etc.
    # We use a structured array or lexsort to sort pairs
    # Lexsort sorts by last key first (so sort by cols, then rows)
    sorter = np.lexsort((all_cols, all_rows))
    sorted_rows = all_rows[sorter]
    sorted_cols = all_cols[sorter]

    # remove Duplicates
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

    # split into List of Arrays
    # We need to find where the row index changes to split the columns array
    # np.unique with return_counts or return_index is fast
    _, split_indices = np.unique(unique_rows, return_index=True)

    # np.split creates the list of arrays [ [neighs_0], [neighs_1], ... ]
    # We skip the first index (0) as split expects split points
    adjacency = np.split(unique_cols, split_indices[1:])

    # handle case where some nodes might be isolated (empty lists)
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
    """
    Simple greedy coloring algorithm.
    Args:
        adjacency: Adjacency list where 'neighbors' include neighbors-of-neighbors.
    Returns:
        colors: Array of colors assigned to each node
    """
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


def greedy_coloring_ldf(adjacency):
    """
    Greedy coloring using the Largest Degree First (LDF) heuristic.
    Args:
        adjacency: Adjacency list where 'neighbors' include neighbors-of-neighbors.
    Returns:
        colors: Array of colors assigned to each node
    """
    n = len(adjacency)

    # compute degrees (number of distance-2 neighbors)
    # adjacency[i] contains all neighbors of node i in A^2
    degrees = np.array([len(adj) for adj in adjacency], dtype=np.int32)

    # sort nodes: highest degree first
    # argsort gives ascending, [::-1] makes it descending
    sorted_nodes = np.argsort(degrees)[::-1]

    colors = -np.ones(n, dtype=np.int32)

    for i in sorted_nodes:
        # Check colors of distance-2 neighbors
        # Using a set for 'used' is fast for lookups
        neighbor_colors = colors[adjacency[i]]
        used = {c for c in neighbor_colors if c >= 0}

        c = 0
        while c in used:
            c += 1
        colors[i] = c

    return jnp.array(colors)


def get_smallest_last_order(adjacency):
    """
    Computes the Smallest-Last (SL) ordering using a bucket-queue
    to handle millions of nodes efficiently.
    """
    n = len(adjacency)
    # initial degrees of all nodes
    degrees = np.array([len(adj) for adj in adjacency], dtype=np.int32)
    max_degree = np.max(degrees)

    # list of sets where buckets[d] contains nodes with degree d
    buckets = [set() for _ in range(max_degree + 1)]
    for i, d in enumerate(degrees):
        buckets[d].add(i)

    order = []
    is_removed = np.zeros(n, dtype=bool)
    min_deg_ptr = 0  # Pointer to the lowest non-empty bucket

    for _ in range(n):
        # Find the smallest non-empty bucket
        while min_deg_ptr <= max_degree and not buckets[min_deg_ptr]:
            min_deg_ptr += 1

        # Pick a node from the smallest bucket
        v = buckets[min_deg_ptr].pop()
        is_removed[v] = True
        order.append(v)

        # Update neighbors' degrees in the remaining graph
        for neighbor in adjacency[v]:
            if not is_removed[neighbor]:
                old_deg = degrees[neighbor]
                # Remove neighbor from its old bucket
                buckets[old_deg].remove(neighbor)

                # Update degree and move to a lower bucket
                new_deg = old_deg - 1
                degrees[neighbor] = new_deg
                buckets[new_deg].add(neighbor)

                # Adjust min_deg_ptr if necessary
                if new_deg < min_deg_ptr:
                    min_deg_ptr = new_deg

    # The SL coloring order is the REVERSE of the removal order
    return order[::-1]


def greedy_coloring_sl(adjacency):
    """
    Greedy coloring using the Smallest-Last (SL) ordering.
    Args:
        adjacency: Adjacency list where 'neighbors' include neighbors-of-neighbors.
    Returns:
        colors: Array of colors assigned to each node
    """
    n = len(adjacency)

    # 1. Get the SL sequence (O(N + E))
    print("Computing Smallest-Last ordering...")
    sl_order = get_smallest_last_order(adjacency)

    # 2. Greedy assignment
    colors = -np.ones(n, dtype=np.int32)

    print("Assigning colors...")
    for i in sl_order:
        # Check colors of distance-2 neighbors
        neighbor_colors = colors[adjacency[i]]
        used = {c for c in neighbor_colors if c >= 0}

        c = 0
        while c in used:
            c += 1
        colors[i] = c

    return colors


def distance2_colors(row_ptr: Array, col_idx: Array, n_dofs: int):
    """
    Compute distance-2 coloring based on greedy algorithm.

    Args:
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
        n_dofs: Number of degrees of freedom (size of the matrix)
    Returns:
        colors: Array of colors assigned to each DOF
    """
    adjacency = get_distance2_adjacency(n_dofs, np.array(row_ptr), np.array(col_idx))
    colors = greedy_coloring(adjacency)
    return jnp.array(colors)


def largest_degree_first_distance2_colors(
    row_ptr: Array, col_idx: Array, n_dofs: int
) -> Array:
    """
    Compute distance-2 coloring based on Largest Degree First (LDF) heuristic.
    Args:
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
        n_dofs: Number of degrees of freedom (size of the matrix)
    Returns:
        colors: Array of colors assigned to each DOF
    """
    adjacency = get_distance2_adjacency(n_dofs, np.array(row_ptr), np.array(col_idx))
    colors = greedy_coloring_ldf(adjacency)
    return jnp.array(colors)


def smallest_last_distance2_colors(
    row_ptr: Array, col_idx: Array, n_dofs: int
) -> Array:
    """
    Compute distance-2 coloring based on Smallest-Last (SL) ordering.
    Args:
        row_ptr: CSR row pointer array
        col_idx: CSR column indices array
        n_dofs: Number of degrees of freedom (size of the matrix)
    Returns:
        colors: Array of colors assigned to each DOF
    """
    adjacency = get_distance2_adjacency(n_dofs, np.array(row_ptr), np.array(col_idx))
    colors = greedy_coloring_sl(adjacency)
    return jnp.array(colors)


def seeds_from_coloring(colors: Array) -> Array:
    """
    Generate seed vectors from coloring.
    Args:
        colors: Array of colors assigned to each DOF
    Returns:
        seeds: Array of seed vectors for each color. Seeds are stored as int32 arrays for memory efficiency.
    """
    unique = jnp.unique(colors)
    seeds = []
    for c in unique:
        mask = colors == c
        seeds.append(jnp.array(mask, dtype=jnp.int32))
    return jnp.array(seeds)


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
    adjacency = get_distance2_adjacency(n_dofs, np.array(row_ptr), np.array(col_idx))
    colors = greedy_coloring(adjacency)
    seeds = seeds_from_coloring(jnp.array(colors))
    return jnp.array(colors), jnp.stack(seeds, axis=0)
