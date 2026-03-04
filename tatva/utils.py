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


from functools import wraps
from typing import Callable, Concatenate, ParamSpec, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

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
