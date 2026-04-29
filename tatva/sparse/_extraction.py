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

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from tatva import Mesh

if TYPE_CHECKING:
    from tatva.compound import Compound
    from tatva.lifter import Lifter


def _create_sparse_structure(
    elements: ArrayLike, n_dofs_per_node: int, K_shape: tuple[int, int]
) -> sps.csr_matrix:
    """Creates the sparse structure with maximum performance and guaranteed correctness.

    - Uses SciPy for the fastest possible reduction of duplicate coordinates.
    - Uses NumPy's `unique` on a linearized index to perfectly replicate the
      sorting order of the original JAX code at high speed.

    Args:
        elements: (num_elements, nodes_per_element)
        n_dofs_per_node: Number of degrees of freedom per node
        K_shape: Shape of the global stiffness matrix K
    """

    # Use 64-bit integers throughout the CPU-based setup to prevent any overflow.
    elements_np = np.asarray(elements, dtype=np.int64)
    num_elements, nodes_per_element = elements_np.shape
    num_dofs_per_element = nodes_per_element * n_dofs_per_node

    dofs = elements_np[..., None] * n_dofs_per_node + np.arange(
        n_dofs_per_node, dtype=np.int64
    )
    element_dofs = dofs.reshape(num_elements, -1)

    row_indices = np.repeat(element_dofs, num_dofs_per_element, axis=1).flatten()
    col_indices = np.tile(element_dofs, (1, num_dofs_per_element)).flatten()

    # Use SciPy to efficiently get the unique (but unsorted) coordinate pairs.
    data = np.ones(row_indices.shape[0], dtype=np.int8)
    coo_matrix = sps.coo_matrix((data, (row_indices, col_indices)), shape=K_shape)

    unique_rows = coo_matrix.row
    unique_cols = coo_matrix.col

    # Linearize the indices. This is the key to matching the original sorting logic.
    num_cols = K_shape[1]
    linear_indices = unique_rows.astype(np.int64) * num_cols + unique_cols.astype(
        np.int64
    )

    sorted_linear_indices = np.unique(linear_indices)

    # Delinearize the now-sorted indices.
    sorted_rows = sorted_linear_indices // num_cols
    sorted_cols = sorted_linear_indices % num_cols

    return sps.csr_matrix(
        (np.ones(sorted_rows.shape[0], dtype=np.int8), (sorted_rows, sorted_cols)),
        shape=K_shape,
    )


def create_sparsity_pattern(mesh: Mesh, n_dofs_per_node: int) -> sps.csr_matrix:
    """Create a sparsity pattern using SciPy's COO format for efficient setup on CPU.

    Args:
        mesh: Mesh object
        n_dofs_per_node: Number of degrees of freedom per node
    """
    K_shape = (
        n_dofs_per_node * mesh.coords.shape[0],
        n_dofs_per_node * mesh.coords.shape[0],
    )
    elements = mesh.elements
    return _create_sparse_structure(elements, n_dofs_per_node, K_shape)


def get_bc_indices(
    sparsity_pattern: sps.csr_matrix, fixed_dofs: Array | NDArray
) -> tuple[NDArray, NDArray]:
    """Get the indices of the fixed degrees of freedom.

    Args:
        sparsity_pattern: scipy.sparse.csr_matrix
        fixed_dofs: (num_fixed_dofs,)

    Returns:
        zero_indices: (num_zero_indices,)
        one_indices: (num_one_indices,)
    """

    row_indices, col_indices = sparsity_pattern.nonzero()
    zero_indices = []
    one_indices = []

    for dof in fixed_dofs:
        indexes = np.where(row_indices == dof)[0]
        for idx in indexes:
            zero_indices.append(int(idx))

        idx = np.where((row_indices == dof) & (col_indices == dof))[0][0]
        one_indices.append(int(idx))

    return np.array(zero_indices), np.array(one_indices)


def create_sparsity_pattern_KKT(
    mesh: Mesh, n_dofs_per_node: int, B: ArrayLike
) -> sps.csr_matrix:
    """Create KKT sparsity pattern in SciPy CSR format.

    Block structure:
        [ K   B^T ]
        [ B    C  ]
    where C is identity (matching your current implementation).
    """
    B_np = np.asarray(B)
    nb_cons = B_np.shape[0]

    # K pattern (convert to CSR)
    K = create_sparsity_pattern(mesh, n_dofs_per_node)

    # B pattern from dense B (keep only structure)
    B_pattern = sps.csr_matrix((B_np != 0).astype(np.int8))
    BT_pattern = B_pattern.T

    # TODO: decide if we want the true saddle-point structure with zero C block, or if we
    # want to keep the identity
    C = sps.eye(nb_cons, format="csr", dtype=np.int8)

    KKT = sps.block_array(
        [[K, BT_pattern], [B_pattern, C]],
        format="csr",
    )
    return KKT


def reduce_sparsity_pattern(
    pattern: sps.csr_matrix, free_dofs: ArrayLike
) -> sps.csr_matrix:
    """Reduce a sparse matrix pattern in CSR format to only the free dofs.

    Args:
        pattern (sps.csr_matrix): Sparse matrix pattern in CSR format on the full
            set of dofs.
        free_dofs: Array of free dofs as integer indices.
    """
    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    return pattern[free_dofs][:, free_dofs]


def create_sparsity_pattern_master_slave(
    mesh: Mesh,
    n_dofs_per_node: int,
    master_slave_map: ArrayLike,
) -> sps.csr_matrix:
    """Create a sparsity pattern for a system with master–slave DOF mapping, e.g.,
    periodic BCs.

    The returned sparsity corresponds to the reduced system defined on master DOFs only.

    Args:
        mesh: Mesh object.
        n_dofs_per_node: Number of degrees of freedom per node.
        master_slave_map: Array of shape (n_full_dofs,) where each entry maps a full DOF
            index to its master DOF index (masters map to themselves). Optionally, an
            array of shape (n_nodes,) mapping nodes to master nodes; this will be expanded
            to DOF-level by preserving the per-node DOF offset. Entries can be set -1 to
            indicate Dirichlet BCs (removal from system).

    Returns:
        jax.experimental.sparse.BCOO: Sparsity pattern of the reduced (master-only)
        system.
    """

    n_nodes = int(mesh.coords.shape[0])
    n_full = int(n_nodes * n_dofs_per_node)

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
    full_pattern_csr = _create_sparse_structure(
        mesh.elements, n_dofs_per_node, K_shape=(n_full, n_full)
    )
    full_pattern = full_pattern_csr.tocoo()  # COO for easy access to indices

    I_full = full_pattern.row.astype(np.int64)
    J_full = full_pattern.col.astype(np.int64)

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

    return sps.csr_matrix(
        (data, (Iu, Ju)),
        shape=(n_red, n_red),
    )


def augment_sparsity_with_lifter(
    sparsity: sps.csr_matrix, lifter: Lifter
) -> sps.csr_matrix:
    """Augment the sparsity pattern with constraints from a lifter.

    Args:
        sparsity: Sparsity pattern in SciPy CSR format.
        lifter: Lifter containing constraints.
    """
    return lifter.augment_sparsity(sparsity)


def create_sparsity_pattern_from_compound(
    compound_cls: type[Compound],
    mesh: Mesh,
    block_wise: bool = False,
) -> sps.csr_matrix | list[list[sps.csr_matrix]]:
    """Create a sparsity pattern automatically from a Compound class and its attached
    mesh.

    Fields that provide element DOFs (via field_type.get_element_dofs) are fully coupled
    within elements. All other fields (Local, Shared) are only connected to themselves
    (diagonal entries).

    Args:
        compound_cls: The Compound class defining the state layout.
        mesh: The Mesh object attached to the Compound.
        block_wise: If True, return the pattern as a list of lists of sparse matrices
            corresponding to the compound fields/blocks. Stacked fields are one block.
    """
    main_coupling_list: list[NDArray[np.int32]] = []
    diagonal_dofs_list: list[NDArray[np.int32]] = []

    for name, f in compound_cls.fields:
        field_type_obj = f.field_type.get()

        edofs = field_type_obj.get_element_dofs(f, mesh)
        if edofs is not None:
            main_coupling_list.append(edofs)

        ddofs = field_type_obj.get_diagonal_dofs(f, mesh)
        if ddofs is not None:
            diagonal_dofs_list.append(ddofs)

    K_shape = (compound_cls.size, compound_cls.size)
    all_rows = []
    all_cols = []

    if main_coupling_list:
        cg_element_dofs = np.concatenate(main_coupling_list, axis=1)
        num_elements, num_dofs_per_element = cg_element_dofs.shape

        row_indices = np.repeat(cg_element_dofs, num_dofs_per_element, axis=1).flatten()
        col_indices = np.tile(cg_element_dofs, (1, num_dofs_per_element)).flatten()

        # Filter out -1 (from incomplete Nodal fields)
        valid = (row_indices >= 0) & (col_indices >= 0)
        all_rows.append(row_indices[valid])
        all_cols.append(col_indices[valid])

    if diagonal_dofs_list:
        diag_dofs = np.concatenate(diagonal_dofs_list)
        all_rows.append(diag_dofs)
        all_cols.append(diag_dofs)

    if not all_rows:
        full_matrix = sps.csr_matrix(K_shape, dtype=np.int8)
    else:
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)

        # Fast deduplication using linearized indices
        num_cols = K_shape[1]
        linear_indices = rows.astype(np.int64) * num_cols + cols.astype(np.int64)
        sorted_unique_linear = np.unique(linear_indices)

        sorted_rows = sorted_unique_linear // num_cols
        sorted_cols = sorted_unique_linear % num_cols

        full_matrix = sps.csr_matrix(
            (np.ones(sorted_rows.shape[0], dtype=np.int8), (sorted_rows, sorted_cols)),
            shape=K_shape,
        )

    if not block_wise:
        return full_matrix

    # Identify blocks (unique root slices)
    root_slices = []
    seen = set()
    for _, f in compound_cls.fields:
        s = getattr(f, "_root_slice", getattr(f, "_slice", None))
        if s is not None and s not in seen:
            root_slices.append(s)
            seen.add(s)
    root_slices.sort(key=lambda s: s.start)

    return [[full_matrix[s1, s2] for s2 in root_slices] for s1 in root_slices]
