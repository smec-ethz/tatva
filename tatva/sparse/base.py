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

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Concatenate, ParamSpec, Self

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import scipy.sparse as sp
from jax import Array
from jax.tree_util import register_dataclass
from numpy.typing import NDArray
from tatva_coloring import distance2_color_and_seeds

P = ParamSpec("P")


@register_dataclass
@dataclass(frozen=True)
class SparseMatrix:
    """Class to represent the sparsity pattern of a matrix, including the row pointers,
    column indices, and colors for graph coloring.
    """

    data: Array = field(repr=False)
    """Data values of the sparse matrix"""

    indptr: NDArray = field(metadata=dict(static=True), repr=False)
    """Row pointers for the original sparsity pattern (CSR format)"""

    indices: NDArray = field(metadata=dict(static=True), repr=False)
    """Column indices for the original sparsity pattern (CSR format)"""

    shape: tuple[int, int] = field(metadata=dict(static=True))
    """Shape of the sparse matrix"""

    colors: NDArray = field(metadata=dict(static=True), repr=False)
    """Array of colors assigned to each degree of freedom (DOF) for graph coloring"""

    @classmethod
    def from_csr(cls, csr_matrix: sp.csr_matrix, colors: NDArray | None = None) -> Self:
        """Create a SparseMatrix instance from a SciPy CSR matrix and optional colors."""
        indptr = csr_matrix.indptr
        indices = csr_matrix.indices

        if colors is None:
            # If no colors are provided, compute them using distance-2 coloring
            colors, _seeds = distance2_color_and_seeds(
                indptr,  # ty:ignore[invalid-argument-type]
                indices,  # ty:ignore[invalid-argument-type]
                csr_matrix.shape[0],
            )

        return cls(
            data=jnp.asarray(csr_matrix.data),
            indptr=indptr,
            indices=indices,
            shape=csr_matrix.shape,
            colors=colors,
        )

    def jacfwd(
        self, fn: Callable[[Concatenate[Array, P]], Array], *, color_batch_size: int
    ) -> Callable[[Concatenate[Array, P]], SparseMatrix]:
        """Returns a function that computes the Jacobian of `fn` using forward-mode automatic differentiation
        and graph coloring. The returned function takes the same arguments as `fn` and returns a sparse Jacobian
        as a new instance of `Sparsity`.
        """
        nb_colors = len(jnp.unique(self.colors)) + 1
        # precompute the indices needed to recover the full Jacobian from the compressed version
        # rows: row index per non-zero entry (length nnz)
        diffs = jnp.diff(self.indptr)
        rows = jnp.repeat(
            jnp.arange(len(self.indptr) - 1),
            diffs,
            total_repeat_length=len(self.indices),
        )
        # find where the value for (i, j) is hiding in J_compressed
        # The value K_ij is stored at row 'i' and column 'color[j]'
        # col_colors: color of the column for each non-zero entry (length nnz)
        col_colors = self.colors[self.indices]

        def _wrapped_jacfwd(u: Array, *args: P.args, **kwargs: P.kwargs) -> Self:
            # Pass args explicitly to the JIT-compiled function
            J_compressed = colored_jacobian_batch(
                fn, u, self.colors, args, kwargs, nb_colors, color_batch_size
            )
            # TODO: this function is a one-liner, could do here directly:
            # data = J_compressed[rows, col_colors]
            data = recover_matrix_data(J_compressed, rows, col_colors)
            return replace(self, data=data)

        return _wrapped_jacfwd

    def to_csr(self) -> sp.csr_matrix:
        """Convert the sparse matrix to SciPy's CSR format."""
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    def to_bcoo(self) -> jsp.BCOO:
        """Convert the sparse matrix to JAX's BCOO format."""
        # expand row pointers to get row indices for every non-zero
        n_dofs = self.shape[0]
        diffs = jnp.diff(self.indptr)
        rows = jnp.repeat(
            jnp.arange(n_dofs), diffs, total_repeat_length=len(self.indices)
        )
        cols = self.indices
        indices = jnp.stack([rows, cols], axis=1)

        return jsp.BCOO((self.data, indices), shape=(n_dofs, n_dofs))

    def to_dense(self) -> Array:
        """Convert the sparse matrix to a dense array."""
        bcoo = self.to_bcoo()
        return bcoo.todense()


@partial(jax.jit, static_argnames=["fn", "n_colors", "color_batch_size"])
def colored_jacobian_batch(
    fn: Callable[[Concatenate[Array, P]], Array],
    x: Array,
    colors: Array,
    args: P.args,
    kwargs: P.kwargs,
    n_colors: int,
    color_batch_size: int = 1,
) -> Array:
    """
    Computes the compressed Jacobian by processing colors in batches. By default,
    processes one color at a time to minimize memory usage.
    Args:
        fn: function to differentiate
        x: Point at which to evaluate the Jacobian, shape (N,)
        colors: Array of colors assigned to each DOF
        args: Positional arguments to pass to F
        kwargs: Keyword arguments to pass to F
        n_colors: Number of unique colors
        color_batch_size: Number of colors to process in each batch
    Returns:
        J: Compressed Jacobian matrix, shape (N, n_colors)
    """

    def compute_single_jvp(color_id: Array):
        # TODO: Can this be moved outside the loop? Precompute it when creating jacfwd?
        seed = jnp.where(colors == color_id, 1.0, 0.0)

        def fn_partial(u: Array) -> Array:
            return fn(u, *args, **kwargs)

        _, jvp_out = jax.jvp(fn_partial, (x,), (seed,))
        return jvp_out

    # TODO: Can this be moved outside the loop? Precompute it when creating jacfwd?
    colors_array = jnp.arange(n_colors)
    J_rows = jax.lax.map(compute_single_jvp, colors_array, batch_size=color_batch_size)

    return J_rows.T


@jax.jit
def recover_matrix_data(
    J_compressed: Array, coo_rows: Array, col_colors: Array
) -> Array:
    """
    Recover the exact values from the compressed Jacobian.

    Args:
        J_compressed: Output from colored_jacobian, shape (N, n_colors)
        coo_rows: row index per non-zero entry (length nnz)
        col_colors: color of the column for each non-zero entry (length nnz)
    """
    # extract values using fancy indexing
    # values[k] = J_compressed[ rows[k], col_colors[k] ]
    return J_compressed[coo_rows, col_colors]


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
