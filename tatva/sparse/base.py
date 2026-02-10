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
from typing import Any, Callable

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
from jax import Array


@partial(jax.jit, static_argnames=["F", "n_colors", "color_batch_size"])
def colored_jacobian_batch(
    F: Callable, x: Array, colors: Array, n_colors: int, color_batch_size: int = 1
) -> Array:
    """
    Computes the compressed Jacobian by processing colors in batches. By default,
    processes one color at a time to minimize memory usage.
    Args:
        F: function to differentiate
        x: Point at which to evaluate the Jacobian, shape (N,)
        colors: Array of colors assigned to each DOF
        n_colors: Number of unique colors
        color_batch_size: Number of colors to process in each batch
    Returns:
        J: Compressed Jacobian matrix, shape (N, n_colors)
    """

    def compute_single_jvp(color_id: Array):
        # Discard primal output, keep tangent (Jacobian column)
        seed = jnp.where(colors == color_id, 1.0, 0.0)
        _, jvp_out = jax.jvp(F, (x,), (seed,))
        return jvp_out

    # Use lax.map to iterate over seeds (colors) one-by-one or in small batches
    # This ensures memory is reclaimed between color passes.
    colors_array = jnp.arange(n_colors)
    J_rows = jax.lax.map(compute_single_jvp, colors_array, batch_size=color_batch_size)

    return J_rows.T  # Shape (N_dof, N_colors)


@partial(jax.jit, static_argnames=["F", "n_colors"])
def colored_jacobian(F: Callable, x: Array, colors: Array, n_colors: int) -> Array:
    """
    Computes the compressed Jacobian using forward-mode AD and graph coloring without batching.
    Reconstructs seeds on-the-fly via masking to avoid OOM and jagged arrays.
    Args:
        F: function to differentiate
        x: Point at which to evaluate the Jacobian, shape (N,)
        colors: Array of colors assigned to each DOF
        n_colors: Number of unique colors
    Returns:
        J: Compressed Jacobian matrix, shape (N, n_colors)
    """

    def scan_body(carry, color_id):
        # Generate seed for the current color_id on-the-fly
        # seed = (colors == color_id)
        seed = jnp.where(colors == color_id, 1.0, 0.0)

        # Compute JVP
        _, jvp_out = jax.jvp(F, (x,), (seed,))

        return None, jvp_out

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

    indices = jnp.stack([rows, cols], axis=1)
    K_bcoo = jsp.BCOO((values, indices), shape=(n_dofs, n_dofs))

    return K_bcoo


def jacfwd_with_batch(
    gradient: Callable,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
    color_batch_size: int = 1,
    has_aux_args: bool = False,
) -> Callable:
    """
    Compute the sparse Jacobian using forward-mode automatic differentiation and graph coloring and provided seeds.
    Uses jax.lax.map to iterate over colors without materializing the entire compressed Jacobian in memory at once.

    Args:
        gradient: Function whose Jacobian is to be computed
        row_ptr, col_indices: The sparsity pattern of the matrix
        seeds: List of seed vectors for each color
        colors: The array of colors used for compression
        color_batch_size: Number of colors to process in each batch (1 for minimal memory usage, >1 for faster computation but higher memory)
        has_aux_args: Whether the gradient function has auxiliary arguments.
                      If True, the returned function will expect additional arguments after 'u'.
    Returns:
        A function that computes the sparse Jacobian in BCOO format
    """

    n_colors = len(jnp.unique(colors)) + 1

    if not has_aux_args:

        def _wraped_jacfwd(u: Array) -> jsp.BCOO:
            J_compressed = colored_jacobian_batch(
                gradient,
                u,
                colors,
                n_colors=n_colors,
                color_batch_size=color_batch_size,
            )
            n_dofs = J_compressed.shape[0]

            K_bcoo = recover_stiffness_matrix(
                J_compressed, row_ptr, col_indices, colors, n_dofs
            )
            return K_bcoo

        return _wraped_jacfwd

    if has_aux_args:

        def _wraped_jacfwd_with_args(u: Array, *args: Any, **kwargs: Any) -> jsp.BCOO:
            def _gradient_wrt_u(u_in):
                return gradient(u_in, *args, **kwargs)

            J_compressed = colored_jacobian_batch(
                _gradient_wrt_u,
                u,
                colors,
                n_colors=n_colors,
                color_batch_size=color_batch_size,
            )
            n_dofs = J_compressed.shape[0]

            K_bcoo = recover_stiffness_matrix(
                J_compressed, row_ptr, col_indices, colors, n_dofs
            )
            return K_bcoo

        return _wraped_jacfwd_with_args


def jacfwd(
    gradient: Callable,
    row_ptr: Array,
    col_indices: Array,
    colors: Array,
    has_aux_args: bool = False,
) -> Callable:
    """
    Compute the sparse Jacobian using forward-mode automatic differentiation and graph coloring.
    The seeds are reconstructed on-the-fly to save memory. Use jax.lax.scan to iterate over colors
    without materializing the entire compressed Jacobian in memory at once.

    Args:
        gradient: Function whose Jacobian is to be computed
        row_ptr, col_indices: The sparsity pattern of the matrix
        colors: The array of colors used for compression
        has_aux_args: Whether the gradient function has auxiliary arguments.
                     If True, the returned function will expect additional arguments after 'u'.
    Returns:
        A function that computes the sparse Jacobian in BCOO format

    """
    n_colors = len(jnp.unique(colors)) + 1

    if not has_aux_args:

        def _wraped_jacfwd(u: Array) -> jsp.BCOO:
            J_compressed = colored_jacobian(
                gradient, x=u, colors=colors, n_colors=int(n_colors)
            )
            n_dofs = J_compressed.shape[0]

            K_bcoo = recover_stiffness_matrix(
                J_compressed, row_ptr, col_indices, colors, n_dofs
            )
            return K_bcoo

        return _wraped_jacfwd
    if has_aux_args:

        def _wraped_jacfwd_with_args(u: Array, *args: Any, **kwargs: Any) -> jsp.BCOO:
            def _gradient_wrt_u(u_in):
                return gradient(u_in, *args, **kwargs)

            J_compressed = colored_jacobian(
                _gradient_wrt_u, x=u, colors=colors, n_colors=int(n_colors)
            )
            n_dofs = J_compressed.shape[0]

            K_bcoo = recover_stiffness_matrix(
                J_compressed, row_ptr, col_indices, colors, n_dofs
            )
            return K_bcoo

        return _wraped_jacfwd_with_args
