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

from functools import wraps
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from tatva.lifter import Lifter
    from tatva.operator import Operator
    from tatva.sparse import ColoredMatrix

P = ParamSpec("P")


@overload
def virtual_work_to_residual(
    fn: Callable[Concatenate[Array, P], Array],
    /,
    *,
    test_arr: ArrayLike | None = None,
    test_shape: tuple[int, ...] | None = None,
    test_size: int | None = None,
    jit: bool = False,
) -> Callable[P, Array]: ...


@overload
def virtual_work_to_residual(
    *,
    test_arr: ArrayLike | None = None,
    test_shape: tuple[int, ...] | None = None,
    test_size: int | None = None,
    jit: bool = False,
) -> Callable[[Callable[Concatenate[Array, P], Array]], Callable[P, Array]]: ...


def virtual_work_to_residual(
    fn=None,
    /,
    *,
    test_arr: ArrayLike | None = None,
    test_shape: tuple[int, ...] | None = None,
    test_size: int | None = None,
    jit: bool = False,
) -> Callable:
    """Convert a virtual-work function that is linear wrt. the test function
    (first argument) into a residual function.

    The returned callable evaluates the Jacobian of ``fn`` with respect to its first
    argument at ``test_arr``. This utility can be used directly or as a decorator.
    Only one of the arguments ``test_arr``, ``test_shape``, or ``test_size`` should be
    provided.

    Args:
        fn: Function whose first argument is the test array. If ``None``, the function
            returns a decorator.
        test_arr: Test array to use for the Jacobian evaluation.
        test_shape: Shape of the test array to use for the Jacobian evaluation. Will use
            `jnp.zeros(test_shape)` as the test array.
        test_size: Size of the test array to use for the Jacobian evaluation. Will use
            `jnp.zeros(test_size)` as the test array.
        jit: Whether to JIT-compile the generated residual function (Default: False).

    Returns:
        Callable: A residual function with the same remaining signature as ``fn``.
    """
    if test_arr is not None:
        test_arr = jnp.asarray(test_arr)
    elif test_shape is not None:
        test_arr = jnp.zeros(test_shape)
    elif test_size is not None:
        test_arr = jnp.zeros(test_size)
    else:
        raise ValueError(
            "One of 'test_arr', 'test_shape', or 'test_size' must be provided."
        )

    if fn is None:
        return lambda fn: virtual_work_to_residual(
            fn, test_arr=test_arr, test_shape=test_shape, test_size=test_size, jit=jit
        )

    dwork_dtest = jax.jacrev(fn, argnums=0)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Array:
        return dwork_dtest(test_arr, *args, **kwargs)

    if jit:
        wrapper = jax.jit(wrapper)  # pyright: ignore[reportAssignmentType]

    return wrapper


def make_project_function(
    nnodes: int,
    colored_matrix: ColoredMatrix | None = None,
    elements: ArrayLike | None = None,
    lifter: Lifter | None = None,
) -> Callable[[Operator, Array], Array]:
    """Factory to create a projection function for a given mesh (defined by nnodes,
    elements/ColoredMatrix) and optional lifter. The returned function takes an Operator
    and a field to project, and returns the projected field on the nodes.

    Automatically batches the projection if the field has multiple components (e.g.,
    vector or tensor fields) and handles lifting if a lifter is provided.

    Args:
        nnodes: Number of nodes in the mesh.
        colored_matrix: Optional ColoredMatrix defining the sparsity pattern for the
            projection. If not provided, it will be created from the elements.
        elements: Optional array defining the mesh elements, used to create the
            ColoredMatrix if not provided.
        lifter: Optional Lifter object to handle lifting of the solution from the reduced
            system to the full system. If provided, the projection will be performed on
            the reduced system and then lifted back to the full system.
    """
    from tatva.sparse import ColoredMatrix
    from tatva.sparse._extraction import (
        _create_sparse_structure,
        reduce_sparsity_pattern,
    )

    if colored_matrix is None:
        if elements is None:
            raise ValueError("Must provide elements if colored_matrix is not provided")
        # default to scalar projection (multiple RHS for vector/tensor fields)
        dim = 1
        n = nnodes * dim
        sp = _create_sparse_structure(elements, dim, (n, n))
        if lifter is not None:
            sp = reduce_sparsity_pattern(sp, lifter.free_dofs)
        colored_matrix = ColoredMatrix.from_csr(sp)
    else:
        dim = colored_matrix.shape[0] // nnodes
        if lifter is not None and colored_matrix.shape[0] != lifter.size_reduced:
            raise ValueError(
                "Colored matrix size does not match lifter reduced size. "
                f"Expected {lifter.size_reduced}, got {colored_matrix.shape[0]}"
            )

    lhs, rhs = _make_projection(nnodes, dim, colored_matrix, lifter)
    n_solver = colored_matrix.shape[0]

    @jax.jit
    def _wrapped(op: Operator, field: Array) -> Array:
        value_shape = field.shape[2:]
        field_flat = field.reshape(field.shape[:2] + (-1,))

        M = lhs(jnp.zeros(n_solver), op)
        b = rhs(field_flat, op)

        solution_flat = _solve_projection(M, b, lifter)

        if field.ndim > 2:
            return solution_flat.reshape((nnodes,) + value_shape)
        else:
            return solution_flat

    return _wrapped


def _solve_projection(M: ColoredMatrix, b: Array, lifter: Lifter | None) -> Array:
    """Internal helper to solve the projection system and handle lifting."""
    from jax.experimental.sparse.linalg import spsolve

    # spsolve in JAX only supports 1D b. We use lax.map to handle multiple RHS
    # since spsolve does not have a batching rule for vmap.
    if b.ndim > 1:
        # b has shape (n_components, n_solver)
        solution_flat = jax.lax.map(
            lambda _b: spsolve(M.data, M.indices, M.indptr, _b), b
        ).T
    else:
        solution_flat = spsolve(M.data, M.indices, M.indptr, b)

    if lifter is not None:
        if solution_flat.ndim > 1:
            # Handle multiple RHS manually since Lifter.lift_from_zeros only supports 1D
            solution_flat = jax.vmap(lifter.lift_from_zeros, in_axes=1, out_axes=1)(
                solution_flat
            )
        else:
            solution_flat = lifter.lift_from_zeros(solution_flat)

    return solution_flat


def _make_projection(
    nnodes: int,
    dim: int,
    colored_matrix: ColoredMatrix,
    lifter: Lifter | None = None,
) -> tuple[
    Callable[[Array, Operator], ColoredMatrix], Callable[[Array, Operator], Array]
]:
    from tatva.sparse import jacfwd

    n = lifter.size_reduced if lifter is not None else nnodes * max(dim, 1)

    def dot(a: Array, b: Array) -> Array:
        # contracting dot for coupled solve or component-wise dot for scalar solve
        # handles broadcasting if a (test) is scalar and b (field) is tensor
        if a.shape == b.shape:
            return jnp.einsum("...i,...i->...", a, b)
        return a[..., None] * b if a.ndim < b.ndim else a * b

    def reshape_to_solver_dim(arr: Array) -> Array:
        if dim > 0:
            return arr.reshape(-1, dim)
        else:
            return arr

    @virtual_work_to_residual(test_size=n)
    def _lhs(test: Array, trial: Array, op: Operator) -> Array:
        if lifter is not None:
            test = jnp.zeros(lifter.size).at[lifter.free_dofs].set(test)
            trial = lifter.lift_from_zeros(trial)

        test_reshaped = reshape_to_solver_dim(test)
        trial_reshaped = reshape_to_solver_dim(trial)
        v_q = op.eval(test_reshaped)
        u_q = op.eval(trial_reshaped)
        integrand = dot(v_q, u_q)
        return op.integrate(integrand)

    @virtual_work_to_residual(test_size=n)
    def _rhs(test: Array, field: Array, op: Operator) -> Array:
        if lifter is not None:
            test = jnp.zeros(lifter.size).at[lifter.free_dofs].set(test)

        test_reshaped = reshape_to_solver_dim(test)
        test_quad = op.eval(test_reshaped)
        integrand = dot(test_quad, field)
        return op.integrate(integrand)

    lhs = jacfwd(_lhs, colored_matrix, color_batch_size=None)

    return lhs, _rhs
