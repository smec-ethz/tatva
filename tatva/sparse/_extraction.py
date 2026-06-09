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

import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import scipy.sparse as sps
from jax.typing import ArrayLike
from numpy.typing import NDArray

from tatva import Mesh

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from tatva.compound import Compound, CompoundError
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


def pattern_from_mesh(mesh: Mesh, n_dofs_per_node: int) -> sps.csr_matrix:
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


def _adapt_sparsity_with_lifter(
    sparsity: sps.csr_matrix, lifter: Lifter
) -> sps.csr_matrix:
    """Augment the sparsity pattern with constraints from a lifter.

    Args:
        sparsity: Sparsity pattern in SciPy CSR format.
        lifter: Lifter containing constraints.
    """
    return lifter.adapt_sparsity(sparsity)


@overload
def pattern_from_compound(
    compound_cls: type[Compound], block_wise: Literal[False] = False
) -> csr_matrix: ...
@overload
def pattern_from_compound(
    compound_cls: type[Compound], block_wise: Literal[True] = True
) -> list[list[csr_matrix]]: ...
def pattern_from_compound(
    compound_cls: type[Compound],
    block_wise: bool = False,
) -> sps.csr_matrix | list[list[sps.csr_matrix]]:
    """Create a sparsity pattern automatically from a Compound class and its attached
    mesh.

    Nodal fields are fully coupled within elements. All other fields (Local, Shared)
    are only connected to themselves (diagonal entries).

    Args:
        compound_cls: The Compound class defining the state layout.
        block_wise: If True, return the pattern as a list of lists of sparse matrices
            corresponding to the compound fields/blocks. Stacked fields are one block.
    """
    from tatva.compound.field_types import Nodal

    if compound_cls._mesh is None:
        raise CompoundError(
            "Mesh must be set on Compound class to create sparsity pattern."
        )

    n_nodes = compound_cls._mesh.coords.shape[0]
    num_elements = compound_cls._mesh.elements.shape[0]

    main_coupling_list: list[NDArray[np.int32]] = []
    diagonal_dofs_list: list[NDArray[np.int32]] = []

    for name, f in compound_cls.fields:
        field_type_obj = f.field_type.get()
        if isinstance(field_type_obj, Nodal):
            # Map field DOFs to nodes
            indices = np.asarray(f.indices(slice(None)))
            n_items = (
                len(field_type_obj.node_ids)
                if field_type_obj.node_ids is not None
                else n_nodes
            )

            if n_items > 0:
                dofs_per_item = indices.size // n_items

                node_dofs = np.full((n_nodes, dofs_per_item), -1, dtype=np.int32)
                if field_type_obj.node_ids is None:
                    node_dofs[:] = indices.reshape(n_nodes, dofs_per_item)
                else:
                    node_ids = np.asarray(field_type_obj.node_ids, dtype=np.int32)
                    valid_nodes = (node_ids >= 0) & (node_ids < n_nodes)
                    node_dofs[node_ids[valid_nodes]] = indices.reshape(
                        -1, dofs_per_item
                    )[valid_nodes]

                # Map nodes to elements
                f_element_dofs = node_dofs[compound_cls._mesh.elements].reshape(
                    num_elements, -1
                )
                main_coupling_list.append(f_element_dofs)
        else:
            warnings.warn(
                f"Custom space detected for field '{name}'. Only diagonal entries added "
                "to the sparsity pattern. Please provide your own sparsity pattern if "
                "cross-coupling is required.",
                UserWarning,
            )
            diagonal_dofs_list.append(np.asarray(f.indices(slice(None))).flatten())

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
