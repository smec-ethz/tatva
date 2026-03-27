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
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.tree_util import register_dataclass
from numpy.typing import NDArray
from tatva_coloring import distance2_colors

P = ParamSpec("P")


@register_dataclass
@dataclass(frozen=True)
class ColoredMatrix:
    """Class to represent the sparsity pattern of a matrix, including the row pointers,
    column indices, and colors for graph coloring.
    """

    data: Array = field(repr=False)
    """Data values of the sparse matrix"""

    indptr: Array = field(repr=False)
    """Row pointers for the original sparsity pattern (CSR format)"""

    indices: Array = field(repr=False)
    """Column indices for the original sparsity pattern (CSR format)"""

    shape: tuple[int, int] = field(metadata=dict(static=True))
    """Shape of the sparse matrix"""

    colors: Array = field(repr=False)
    """Colors assigned to each degree of freedom (DOF) for graph coloring"""

    @classmethod
    def from_csr(cls, csr_matrix: sp.csr_matrix, colors: NDArray | None = None) -> Self:
        """Create a SparseMatrix instance from a SciPy CSR matrix and optional colors."""
        indptr = csr_matrix.indptr
        indices = csr_matrix.indices

        if colors is None:
            colors = distance2_colors(indptr, indices, csr_matrix.shape[0])

        return cls(
            data=jnp.asarray(csr_matrix.data),
            indptr=jnp.asarray(indptr),
            indices=jnp.asarray(indices),
            shape=csr_matrix.shape,
            colors=jnp.asarray(colors),
        )

    def to_csr(self) -> sp.csr_matrix:
        """Convert the sparse matrix to SciPy's CSR format."""
        return sp.csr_matrix(
            (np.asarray(self.data), np.asarray(self.indices), np.asarray(self.indptr)),
            shape=self.shape,
        )

    def to_bcoo(self) -> jsp.BCOO:
        """Convert the sparse matrix to JAX's BCOO format."""
        # expand row pointers to get row indices for every non-zero
        n_dofs = self.shape[0]
        diffs = jnp.diff(self.indptr)
        rows = jnp.repeat(
            jnp.arange(n_dofs), diffs, total_repeat_length=self.indices.shape[0]
        )
        indices = jnp.stack([rows, self.indices], axis=1)

        return jsp.BCOO((self.data, indices), shape=(n_dofs, n_dofs))

    def to_bcsr(self) -> jsp.BCSR:
        """Convert the sparse matrix to JAX's BCSR format."""
        return jsp.BCSR(
            (self.data, self.indices, self.indptr),
            shape=self.shape,
        )

    def to_dense(self) -> Array:
        """Convert the sparse matrix to a dense array."""
        bcoo = self.to_bcoo()
        return bcoo.todense()


def jacfwd(
    fn: Callable[Concatenate[Array, P], Array],
    colored_matrix: ColoredMatrix,
    *,
    color_batch_size: int | None = None,
) -> Callable[Concatenate[Array, P], ColoredMatrix]:
    """Returns a function that computes the Jacobian of `fn` using forward-mode automatic differentiation
    and graph coloring. The returned function takes the same arguments as `fn` and returns a sparse Jacobian
    as a new instance of `Sparsity`.

    Args:
        fn: Function for which to compute the Jacobian. Must take an Array as its first
            argument and return an Array. Will be differentiated with respect to the first
            argument.
        colored_matrix: An instance of ColoredMatrix representing the sparsity pattern and
            coloring of the Jacobian.
        color_batch_size: Optional batch size for processing colors. If None, processes
            all colors at once. If memory usage is a concern, set to a smaller value to
            process colors in batches.
    """
    nb_colors = int(colored_matrix.colors.max()) + 1
    if color_batch_size is None:
        color_batch_size = nb_colors

    _indptr = colored_matrix.indptr
    _indices = colored_matrix.indices
    _colors = colored_matrix.colors
    # precompute the indices needed to recover the full Jacobian from the compressed version
    # rows: row index per non-zero entry (length nnz)
    diffs = jnp.diff(_indptr)
    rows = jnp.repeat(
        jnp.arange(_indptr.shape[0] - 1),
        diffs,
        total_repeat_length=_indices.shape[0],
    )
    # find where the value for (i, j) is hiding in J_compressed
    # The value K_ij is stored at row 'i' and column 'color[j]'
    # col_colors: color of the column for each non-zero entry (length nnz)
    col_colors = _colors[_indices]

    def _wrapped_jacfwd(u: Array, *args: P.args, **kwargs: P.kwargs) -> ColoredMatrix:
        J_compressed = colored_jacobian_batch(
            fn, u, colored_matrix.colors, nb_colors, color_batch_size, *args, **kwargs
        )
        # TODO: this function is a one-liner, could do here directly:
        # data = J_compressed[rows, col_colors]
        data = recover_matrix_data(J_compressed, rows, col_colors)
        return replace(colored_matrix, data=data)

    return _wrapped_jacfwd


@partial(jax.jit, static_argnames=["fn", "n_colors", "color_batch_size"])
def colored_jacobian_batch(
    fn: Callable[Concatenate[Array, P], Array],
    x: Array,
    colors: Array,
    n_colors: int,
    color_batch_size: int | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Array:
    """
    Computes the compressed Jacobian by processing colors in batches. By default,
    processes one color at a time to minimize memory usage.
    Args:
        fn: function to differentiate
        x: Point at which to evaluate the Jacobian, shape (N,)
        colors: Array of colors assigned to each DOF
        n_colors: Number of unique colors
        color_batch_size: Number of colors to process in each batch
        *args: Positional arguments to pass to F
        **kwargs: Keyword arguments to pass to F
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
