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
from typing import (
    Any,
    Callable,
    Generic,
    ParamSpec,
    Self,
    TypeVar,
    overload,
)

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from tatva.lifter.common import (
    LifterError,
    RuntimeValueMap,
    _runtime_value_map_is_equal,
)
from tatva.lifter.constraints import Constraint

__all__ = ["Lifter"]

P = ParamSpec("P")
RT = TypeVar("RT")


@register_pytree_node_class
class Lifter:
    """Create a lifter that maps between reduced and full vectors.

    Args:
        size: Total number of dofs in the full vector.
        *constraints: Extra constraints (e.g., periodic maps).

    Examples::

        lifter = Lifter(
            6,
            Fixed(jnp.array([0, 5]), 0.0),
            Periodic(dofs=jnp.array([2]), master_dofs=jnp.array([1])),
        )
        u_reduced = jnp.array([10.0, 20.0, 30.0])
        u_full = lifter.lift_from_zeros(u_reduced)
        # u_full -> [0., 10., 10., 20., 30., 0.]
        u_reduced_back = lifter.reduce(u_full)

    """

    free_dofs: Array
    """Array of free dofs as integer indices (not constrained)."""

    constrained_dofs: Array
    """Array of constrained dofs as integer indices."""

    size: int
    """Total number of dofs in the full vector."""

    size_reduced: int
    """Number of dofs in the reduced vector (free dofs only)."""

    constraints: tuple[Constraint, ...] = ()
    """Tuple of constraints, which are applied in order during lifting. Constraints must
    specify which dofs they apply to, and can optionally specify runtime values that are
    provided to the lifter at runtime. These constraints are bound to the lifter instance,
    which allows them to access the lifter's runtime values when applying the lift."""

    _runtime_values: RuntimeValueMap
    """Mapping of runtime values for dynamic constraints; keys are RuntimeValue keys."""

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        # We want to be able to use lifters as static args in jax transformations, but they contain
        # runtime values that are not hashable. By treating the runtime values as non-static
        # children, we can keep the lifter itself as a static arg while still allowing the
        # runtime values to be passed in and used during lifting.
        children = (self._runtime_values,)
        aux_data = (
            self.size,
            self.constraints,
            self.free_dofs,
            self.constrained_dofs,
            self.size_reduced,
            self._runtime_keys,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> Self:
        # We create a new instance without calling __init__, since we already have all the
        # necessary data and don't want to recompute anything.
        # This is a bit hacky, but it allows us to reconstruct the lifter with the correct
        # runtime values without having to go through the normal initialization process,
        # which would require us to recompute the sizes and runtime pairs.
        size, constraints, free_dofs, constrained_dofs, size_reduced, _runtime_keys = (
            aux_data
        )
        (_runtime_values,) = children

        lifter = cls.__new__(cls)
        lifter.__dict__ = {
            "size": size,
            "constraints": tuple(c._bind(lifter) for c in constraints),
            "free_dofs": free_dofs,
            "constrained_dofs": constrained_dofs,
            "size_reduced": size_reduced,
            "_runtime_keys": _runtime_keys,
        }
        lifter._runtime_values = _runtime_values
        return lifter

    @property
    def at(self) -> RuntimeValueIndexer:
        """Return a ValueIndexer for setting runtime values by key."""
        return RuntimeValueIndexer(self)

    def __init__(
        self,
        size: int,
        /,
        *constraints: Constraint,
    ):
        self.size = size
        self.constraints = tuple(cond._bind(self) for cond in constraints)

        # collect all runtime specs from the constraints and store their keys and default
        # values for easy access during lifting.
        runtime_specs = tuple(
            spec for cond in self.constraints for spec in cond._get_runtime_specs()
        )
        self._runtime_keys = tuple(spec.key for spec in runtime_specs)
        self._runtime_values = {
            spec.key: spec.default for spec in runtime_specs if spec.default is not None
        }

        # compute free/constrained dofs and reduced size based on the constraints; this is
        # only based on the dofs specified in the constraints, not on any runtime values,
        # so we can compute it once at init and not have to worry about it changing at
        # runtime.
        self.free_dofs, self.constrained_dofs, self.size_reduced = self._compute_sizes()

    def __hash__(self):
        return hash((self.size, self.constraints))

    def __eq__(self, other) -> bool:
        """Check equality based on size, constraints, and runtime values. If a lifter is a
        ``static_arg`` in a jax transformation, the runtime values must be equal for
        the lifter to be considered equal.
        """
        return (
            isinstance(other, Lifter)
            and self.size == other.size
            and self.constraints == other.constraints
            and _runtime_value_map_is_equal(self._runtime_values, other._runtime_values)
        )

    def add(self, condition: Constraint) -> Self:
        """Return a new lifter with ``condition`` appended to constraints."""
        return self.__class__(self.size, *self.constraints, condition)

    def lift(
        self,
        u_reduced: Array,
        u_full: Array,
    ) -> Array:
        """Lift reduced displacement vector to full size.

        Args:
            u_reduced: Vector on free dofs (length ``size_reduced``).
            u_full: Base full vector to modify; typically previous solution.

        Returns:
            Full vector with free dofs set to ``u_reduced`` and constraints
            applied (Dirichlet, periodic, etc.).
        """
        u_full = u_full.at[self.free_dofs].set(u_reduced)

        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)

        return u_full

    def lift_from_zeros(
        self,
        u_reduced: Array,
    ) -> Array:
        """Lift reduced vector to a full vector starting from zeros."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return u_full[self.free_dofs]

    def with_values(self, updates: RuntimeValueMap) -> Self:
        """Update the internal runtime values mapping with the given updates."""
        for key in updates:
            if key not in self._runtime_keys:
                raise LifterError(
                    f"There is no runtime value with key={key} in the lifter's constraints"
                )
        return self._replace(_runtime_values=self._runtime_values | updates)

    def _replace(self, **updates) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {**self.__dict__, **updates}
        new.constraints = tuple(cond._bind(new) for cond in new.constraints)
        return new

    def _compute_sizes(self) -> tuple[Array, Array, int]:
        """Compute free/constrained dofs and reduced size."""
        all_dofs = jnp.arange(self.size)

        if not self.constraints:
            # base case: no constraints
            return all_dofs, jnp.array([], dtype=jnp.int32), self.size

        constrained = jnp.concatenate([cond.dofs for cond in self.constraints])
        constrained = jnp.unique(constrained)
        free = jnp.setdiff1d(all_dofs, constrained, assume_unique=True)
        return free, constrained, free.size

    @overload
    def lifted(
        self, *, argnums: tuple[int, ...] | int, reduce_output: bool = False
    ) -> Callable[[Callable[P, RT]], LiftedFunction[P, RT]]: ...
    @overload
    def lifted(
        self,
        fn: Callable[P, RT],
        *,
        argnums: tuple[int, ...] | int,
        reduce_output: bool = False,
    ) -> LiftedFunction[P, RT]: ...
    def lifted(
        self,
        fn: Callable[P, RT] | None = None,
        *,
        argnums: tuple[int, ...] | int,
        reduce_output: bool = False,
    ) -> LiftedFunction | Callable:
        argnums = (argnums,) if isinstance(argnums, int) else argnums

        if fn is None:
            return lambda f: self.lifted(
                f, argnums=argnums, reduce_output=reduce_output
            )

        return LiftedFunction(self, fn, argnums, reduce_output)


@register_pytree_node_class
class LiftedFunction(Generic[P, RT]):
    """A wrapper for a function that has been lifted by a :class:`Lifter`.

    This class is a JAX pytree, meaning it can be passed through JAX transformations like
    :func:`jax.jit`, :func:`jax.vmap`, etc. It also provides access to the original
    function and the lifter used to create it.
    """

    lifter: Lifter
    original_fn: Callable[P, RT]
    argnums: tuple[int, ...]
    reduce_output: bool

    def __init__(
        self,
        lifter: Lifter,
        fn: Callable[P, RT],
        argnums: tuple[int, ...],
        reduce_output: bool = False,
    ):
        self.lifter = lifter
        self.original_fn = fn
        self.argnums = argnums
        self.reduce_output = reduce_output

        # We use wraps to ensure the LiftedFunction instance looks like the original function.
        # This copies attributes like __name__, __doc__, etc.
        wraps(fn)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RT:
        lifted_args = list(args)
        for i, arg in enumerate(args):
            if i not in self.argnums:
                continue

            if not isinstance(arg, Array):
                raise LifterError(
                    f"Argument {i} is not an Array and cannot be lifted by the lifter"
                )
            if not arg.shape == (self.lifter.size_reduced,):
                raise LifterError(
                    f"Argument {i} has shape {arg.shape} but expected "
                    f"{(self.lifter.size_reduced,)} for lifting"
                )

            lifted_args[i] = self.lifter.lift_from_zeros(arg)  # pyright: ignore

        out = self.original_fn(*lifted_args, **kwargs)  # pyright: ignore[reportCallIssue]

        if self.reduce_output:
            if not isinstance(out, Array):
                raise LifterError(
                    "Output is not an Array and cannot be reduced by the lifter"
                )
            return self.lifter.reduce(out)
        return out

    def tree_flatten(self) -> tuple[tuple[Lifter], tuple[Any, ...]]:
        children = (self.lifter,)
        aux_data = (self.original_fn, self.argnums, self.reduce_output)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Lifter]) -> Self:
        (lifter,) = children
        fn, argnums, reduce_output = aux_data
        return cls(lifter, fn, argnums, reduce_output)


class RuntimeValueSetter:
    """Set runtime values on a lifter by position. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter, key: str):
        self.lifter = lifter
        self.key = key

    def set(self, value: ArrayLike) -> Lifter:
        """Set the runtime value for this key and return a new lifter with the updated value."""
        return self.lifter.with_values({self.key: value})


class RuntimeValueIndexer:
    """Set runtime values on a lifter by key. Similar to jnp.array's .at[] setter syntax,
    but for setting values on the lifter's internal runtime value mapping."""

    def __init__(self, lifter: Lifter):
        self.lifter = lifter

    def __getitem__(self, key) -> RuntimeValueSetter:
        return RuntimeValueSetter(self.lifter, key)
