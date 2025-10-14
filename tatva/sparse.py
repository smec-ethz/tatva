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


from typing import Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import sparsejac
from jax import Array
from jax.experimental import sparse as jax_sparse

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


def create_sparsity_pattern_master_slave(
    mesh: Mesh,
    n_dofs_per_node: int,
    master_slave_map: Union[Array, Mapping[int, int]],
):
    """Create a sparsity pattern for a system with master–slave DOF mapping, e.g.,
    periodic BCs.

    The returned sparsity corresponds to the reduced system defined on master DOFs only.

    Args:
        mesh: Mesh object.
        n_dofs_per_node: Number of degrees of freedom per node.
        master_slave_map: Either
            - Array of shape (n_full_dofs,) where each entry maps a full DOF index to its
              master DOF index (masters map to themselves). Optionally, an array of shape
              (n_nodes,) mapping nodes to master nodes; this will be expanded to DOF-level
              by preserving the per-node DOF offset. Entries can be set -1 to indicate
              Dirichlet BCs (removal from system).
            - Mapping (dict-like) from slave DOF index to master DOF index. Indices can be
              at DOF-level or node-level; node-level mappings will be expanded to DOFs.

    Returns:
        jax.experimental.sparse.BCOO: Sparsity pattern of the reduced (master-only)
        system.
    """

    n_nodes = int(mesh.coords.shape[0])
    n_full = int(n_nodes * n_dofs_per_node)

    # Build a DOF-level map array `map_arr` of shape (n_full,), where map_arr[i]
    # gives the master DOF index in [0, n_full).
    if isinstance(master_slave_map, Mapping):  # dict-like: slave -> master
        # Start with identity mapping
        map_arr = np.arange(n_full, dtype=np.int64)

        # Heuristics: decide if keys/values are node-level or dof-level
        keys = np.asarray(list(master_slave_map.keys()), dtype=np.int64)
        vals = np.asarray(list(master_slave_map.values()), dtype=np.int64)
        is_node_level = (
            keys.size > 0
            and np.max(keys) < n_nodes
            and np.max(vals) < n_nodes
            and n_full > n_nodes
        )

        if is_node_level:
            for node_s, node_m in master_slave_map.items():
                base_s = int(node_s) * n_dofs_per_node
                master_node = int(node_m)
                if master_node < 0:
                    for k in range(n_dofs_per_node):
                        map_arr[base_s + k] = -1
                    continue
                base_m = master_node * n_dofs_per_node
                for k in range(n_dofs_per_node):
                    map_arr[base_s + k] = base_m + k
        else:
            for dof_s, dof_m in master_slave_map.items():
                dof_s = int(dof_s)
                dof_m = int(dof_m)
                map_arr[dof_s] = dof_m if dof_m >= 0 else -1

        # Ensure transitive closure (resolve chains slave->...->master)
        # Iterate until stable: map_arr = map_arr[map_arr]
        while True:
            valid = map_arr >= 0
            if not np.any(valid):
                break
            new_map = map_arr.copy()
            new_map[valid] = map_arr[map_arr[valid]]
            new_map[~valid] = -1
            if np.array_equal(new_map, map_arr):
                break
            map_arr = new_map

    else:  # Array-like mapping
        arr = np.asarray(master_slave_map, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("master_slave_map array must be 1D.")
        if arr.size == n_nodes:
            # Node-level mapping: expand to DOF-level preserving component offsets
            map_arr = np.full(n_full, -1, dtype=np.int64)
            for node in range(n_nodes):
                base_s = node * n_dofs_per_node
                master_node = int(arr[node])
                if master_node < 0:
                    continue
                base_m = master_node * n_dofs_per_node
                for k in range(n_dofs_per_node):
                    map_arr[base_s + k] = base_m + k
        elif arr.size == n_full:
            map_arr = arr
        else:
            raise ValueError(
                "master_slave_map array must have length n_nodes or n_full_dofs."
            )

        # Normalize to final masters in case of chaining
        # (map may already be idempotent; this is safe either way)
        while True:
            valid = map_arr >= 0
            if not np.any(valid):
                break
            new_map = map_arr.copy()
            new_map[valid] = map_arr[map_arr[valid]]
            new_map[~valid] = -1
            if np.array_equal(new_map, map_arr):
                break
            map_arr = new_map

    # Create full-system sparsity from connectivity, then project via mapping
    full_pattern = create_sparsity_pattern(
        mesh, n_dofs_per_node=n_dofs_per_node, K_shape=(n_full, n_full)
    )

    I_full = np.asarray(full_pattern.indices[:, 0], dtype=np.int64)
    J_full = np.asarray(full_pattern.indices[:, 1], dtype=np.int64)

    # Unique masters and compaction to 0..n_red-1
    masters = np.unique(map_arr)
    # Handle -1 in dofmap (dirichlet BCs)
    masters_reduced = masters[masters >= 0]
    n_red = int(masters_reduced.size)
    # Map master DOF id in [0, n_full) -> compact id in [0, n_red)
    master_to_compact = -np.ones(n_full, dtype=np.int64)
    master_to_compact[masters_reduced] = np.arange(n_red, dtype=np.int64)

    mapped_I = map_arr[I_full]
    mapped_J = map_arr[J_full]
    valid_entries = (mapped_I >= 0) & (mapped_J >= 0)

    mapped_I = mapped_I[valid_entries]
    mapped_J = mapped_J[valid_entries]

    I_red = master_to_compact[mapped_I]
    J_red = master_to_compact[mapped_J]

    # Deduplicate pairs and set data to 1 (pattern only)
    keys = I_red * n_red + J_red
    uniq, _ = np.unique(keys, return_inverse=False), None
    Iu = (uniq // n_red).astype(np.int32)
    Ju = (uniq % n_red).astype(np.int32)
    data = np.ones(Iu.shape[0], dtype=np.int32)

    indices = np.stack([Iu, Ju], axis=1)
    return jax_sparse.BCOO((data, indices), shape=(n_red, n_red))
