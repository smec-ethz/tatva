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

    # convert to Adjacency List (Your preferred format)
    # we use the existing csr_to_adjacency logic but on A2
    return csr_to_adjacency_fast(n_dofs, A2.indptr, A2.indices, symmetric=False)


def csr_to_adjacency_fast(n_dofs, row_ptr, col_idx, symmetric=True):
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
    # 1. Initial degrees of all nodes
    degrees = np.array([len(adj) for adj in adjacency], dtype=np.int32)
    max_degree = np.max(degrees)

    # 2. Setup Buckets: list of sets where buckets[d] contains nodes with degree d
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
    adjacency = get_distance2_adjacency_fast(
        n_dofs, np.array(row_ptr), np.array(col_idx)
    )
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
    adjacency = get_distance2_adjacency_fast(
        n_dofs, np.array(row_ptr), np.array(col_idx)
    )
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
    adjacency = get_distance2_adjacency_fast(
        n_dofs, np.array(row_ptr), np.array(col_idx)
    )
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
    adjacency = get_distance2_adjacency_fast(
        n_dofs, np.array(row_ptr), np.array(col_idx)
    )
    colors = greedy_coloring(adjacency)
    seeds = seeds_from_coloring(jnp.array(colors))
    return jnp.array(colors), jnp.stack(seeds, axis=0)


@partial(jax.jit, static_argnames=["F", "n_colors", "color_batch_size"])
def colored_jacobian_batch(
    F: Callable, u: Array, colors: Array, n_colors: int, color_batch_size: int = 1
) -> Array:
    """
    Computes the compressed Jacobian by processing colors in batches. By default,
    processes one color at a time to minimize memory usage.
    Args:
        F: function to differentiate
        u: Point at which to evaluate the Jacobian, shape (N,)
        colors: Array of colors assigned to each DOF
        n_colors: Number of unique colors
        color_batch_size: Number of colors to process in each batch
    Returns:
        J: Compressed Jacobian matrix, shape (N, n_colors)
    """

    def compute_single_jvp(color_id: Array):
        # Discard primal output, keep tangent (Jacobian column)
        seed = jnp.where(colors == color_id, 1.0, 0.0)
        _, jvp_out = jax.jvp(F, (u,), (seed,))
        return jvp_out

    # Use lax.map to iterate over seeds (colors) one-by-one or in small batches
    # This ensures memory is reclaimed between color passes.
    colors_array = jnp.arange(n_colors)
    J_rows = jax.lax.map(compute_single_jvp, colors_array, batch_size=color_batch_size)

    return J_rows.T  # Shape (N_dof, N_colors)


@partial(jax.jit, static_argnames=["F", "n_colors"])
def colored_jacobian(F: Callable, u: Array, colors: Array, n_colors: int) -> Array:
    """
    100% JAX/GPU implementation.
    Reconstructs seeds on-the-fly via masking to avoid OOM and jagged arrays.
    """
    n_dofs = u.shape[0]

    def scan_body(carry, color_id):
        # Generate seed for the current color_id on-the-fly
        # This is done entirely on GPU and is extremely fast
        #seed = (colors == color_id).astype(u.dtype)
        seed = jnp.where(colors == color_id, 1.0, 0.0)#.astype(u.dtype)

        # Compute JVP
        _, jvp_out = jax.jvp(F, (u,), (seed,))

        return None, jvp_out

    # lax.scan iterates from color 0 to n_colors-1
    # This compiles ONCE and runs 90 times on GPU.
    color_ids = jnp.arange(n_colors)
    _, J_columns = jax.lax.scan(scan_body, None, color_ids)

    return J_columns.T  # Final shape (N_dof, n_colors)


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


def jacfwd_with_batch(
    gradient: Callable,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
    color_batch_size: int = 1,
) -> Callable:
    """
    Compute the sparse Jacobian using forward-mode automatic differentiation
    and graph coloring and provided seeds.
    Args:
        gradient: Function whose Jacobian is to be computed
        row_ptr, col_indices: The ORIGINAL sparsity pattern of the matrix
        seeds: List of seed vectors for each color
        colors: The array of colors used for compression
    Returns:
        A function that computes the sparse Jacobian in BCOO format
    """

    n_colors = len(jnp.unique(colors)) + 1


    def _wraped_jacfwd(u: Array) -> jsp.BCOO:
        J_compressed = colored_jacobian_batch(
            gradient, u, colors, n_colors=n_colors, color_batch_size=color_batch_size
        )
        n_dofs = J_compressed.shape[0]

        K_bcoo = recover_stiffness_matrix(
            J_compressed, row_ptr, col_indices, colors, n_dofs
        )
        return K_bcoo

    return _wraped_jacfwd


def jacfwd(
    gradient: Callable,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
) -> Callable:
    """
    Compute the sparse Jacobian using forward-mode automatic differentiation
    and graph coloring. The seeds are reconstructed on-the-fly to save memory.

    Args:
        gradient: Function whose Jacobian is to be computed
        row_ptr, col_indices: The ORIGINAL sparsity pattern of the matrix
        colors: The array of colors used for compression
    Returns:
        A function that computes the sparse Jacobian in BCOO format

    """
    n_colors = len(jnp.unique(colors)) + 1

    def _wraped_jacfwd(u: Array) -> jsp.BCOO:
        J_compressed = colored_jacobian(
            gradient, u=u, colors=colors, n_colors=int(n_colors)
        )
        n_dofs = J_compressed.shape[0]

        K_bcoo = recover_stiffness_matrix(
            J_compressed, row_ptr, col_indices, colors, n_dofs
        )
        return K_bcoo

    return _wraped_jacfwd


def jacfwd_with_streaming_recovery(gradient, row_ptr, col_indices, colors):
    n_colors = int(np.max(colors) + 1)
    n_dofs = colors.shape[0]
    nnz = len(col_indices)
    
    # --- 1. PRE-PROCESSING (CPU/NumPy - Zero VRAM used here) ---
    # Expand row pointers to get row index 'i' for every (i, j)
    diffs = np.diff(row_ptr)
    rows_all = np.repeat(np.arange(n_dofs), diffs)
    
    # Sort ALL non-zero entries by the color of their column
    # This allows us to process the 1D sparse array as a sequence of contiguous blocks
    entry_colors = np.array(colors[col_indices])
    sort_idx = np.argsort(entry_colors)
    
    sorted_rows = rows_all[sort_idx].astype(np.int32)
    sorted_cols = col_indices[sort_idx].astype(np.int32)
    
    # Calculate exactly where each color starts in the sorted 1D array
    counts = np.bincount(entry_colors, minlength=n_colors)
    offsets = np.cumsum(np.insert(counts, 0, 0))[:-1]
    
    # To avoid JAX Tracer errors, we use a fixed block size for the JVP extraction
    # We will process every color using the same 'max_entries_per_color' size
    max_block = int(np.max(counts))
    
    # Pre-calculate a 2D index map for the JVP extraction (CPU side)
    # Shape: (n_colors, max_block)
    padded_rows = np.full((n_colors, max_block), 0, dtype=np.int32)
    for c in range(n_colors):
        c_size = counts[c]
        padded_rows[c, :c_size] = sorted_rows[offsets[c] : offsets[c] + c_size]

    # --- 2. GPU TRANSFER (One-Time) ---
    # We stack the indices on CPU to avoid the GPU 'stack' OOM
    indices_bcoo_jax = jnp.array(np.column_stack([sorted_rows, sorted_cols]))
    padded_rows_jax = jnp.array(padded_rows)
    offsets_jax = jnp.array(offsets)
    counts_jax = jnp.array(counts)
    
    # Free CPU memory immediately
    del sorted_rows, sorted_cols, padded_rows, rows_all, sort_idx

    def _wraped_jacfwd(u: jax.Array) -> jsp.BCOO:
        
        def scan_body(val_carry, loop_vars):
            color_id, row_block, start_idx, num_valid = loop_vars
            
            # A. Compute JVP for this color group
            seed = (colors == color_id).astype(u.dtype)
            _, jvp_out = jax.jvp(gradient, (u,), (seed,))
            
            # B. Extract values from JVP result
            # Using the pre-baked rows for this color
            raw_vals = jvp_out[row_block]
            
            # C. Mask padding to zero to avoid corrupting neighbor color data
            mask = jax.lax.broadcasted_iota(jnp.int32, (max_block,), 0) < num_valid
            clean_vals = jnp.where(mask, raw_vals, 0.0)
            
            # D. IN-PLACE OVERWRITE (The Memory Saver)
            # We update the 1D carry at the specific offset for this color
            # Since max_block is a static Python int, this won't crash the Tracer
            new_val_carry = jax.lax.dynamic_update_slice_in_dim(
                val_carry, clean_vals, start_idx, axis=0
            )
            
            return new_val_carry, None

        # Start with an empty 1D array of size NNZ + slight padding
        # We add 'max_block' at the end to safe-guard the final dynamic_update_slice
        init_values = jnp.zeros(nnz + max_block)
        
        final_values_padded, _ = jax.lax.scan(
            scan_body, 
            init_values, 
            (jnp.arange(n_colors), padded_rows_jax, offsets_jax, counts_jax)
        )
        
        # Trim back to exact NNZ and return BCOO
        return jsp.BCOO((final_values_padded[:nnz], indices_bcoo_jax), shape=(n_dofs, n_dofs))

    return _wraped_jacfwd