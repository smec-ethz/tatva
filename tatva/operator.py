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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

import jax
import jax.numpy as jnp
from jax import Array

from tatva._interpolation import make_interpolator
from tatva.element import Element, Tri3
from tatva.mesh import Mesh, find_containing_polygons
from tatva.utils import make_project_function

if TYPE_CHECKING:
    from tatva.lifter import Lifter
    from tatva.sparse import ColoredMatrix

P = ParamSpec("P")
RT = TypeVar("RT", bound=jax.Array | tuple, covariant=True)
ElementT = TypeVar("ElementT", bound=Element)
Numeric: TypeAlias = float | int | jnp.number


MappableOverElementsAndQuads: TypeAlias = Callable[Concatenate[jax.Array, P], RT]
"""A Callable that takes a quadrature point (xi) as the first argument, followed by any
number of additional arguments (P.args and P.kwargs), and returns a jax.Array or a tuple.
This is the type of function that can be mapped over elements and quadrature points using
the `Operator.map` method.
"""

MappableOverElements: TypeAlias = Callable[P, RT]
"""A Callable that takes any number of arguments (P.args and P.kwargs) and returns a
jax.Array or a tuple. This is the type of function that can be mapped over elements using
`Operator.map_over_elements` method.
"""

MappedCallable: TypeAlias = Callable[P, RT]
"""A Callable that takes any number of arguments (P.args and P.kwargs) and returns a
jax.Array or a tuple. This is the type of function returned by the `Operator.map` and
`Operator.map_over_elements` methods.
"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Operator(Generic[ElementT]):
    """A class that provides an Operator for finite element method (FEM) assembly.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The element type used for the finite element method.
        batch_size: Optional batch size for mapping operations over elements. If None, it
            defaults to the number of elements in the mesh. If many elements are present,
            setting a smaller batch size can reduce memory usage.
        cache_weights: If True, the integration weights (the product of the determinant of
            the Jacobian and the quadrature weights) are computed once and cached for
            future use. This can speed up repeated integrations at the cost of increased
            memory usage.

    Provides several operators for evaluating and integrating functions over the mesh,
    such as `integrate`, `eval`, and `grad`. These operators can be used to compute
    integrals, evaluate functions at quadrature points, and compute gradients of
    functions at quadrature points.

    Example:
        >>> from tatva import Mesh, Tri3, Operator
        >>> mesh = Mesh.unit_square(10, 10)  # Create a mesh
        >>> element = Tri3()  # Define an element type
        >>> operator = Operator(mesh, element)
        >>> nodal_values = jnp.array(...)  # Nodal values at the mesh nodes
        >>> energy = operator.integrate(energy_density)(nodal_values)
    """

    mesh: Mesh
    element: ElementT = field(metadata=dict(static=True))
    batch_size: int | None = field(metadata=dict(static=True), default=None)
    cache_weights: bool = field(metadata=dict(static=True), default=False)
    _det_J_elements_weights: jax.Array | None = field(
        repr=False, compare=False, default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        # run initialization checks to ensure mesh/element compatibility and basic
        # shape/type validations
        self.__check_init__()
        if self.batch_size is None:
            object.__setattr__(self, "batch_size", self.mesh.elements.shape[0])

        if self.cache_weights:

            def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
                """Calls the function element.get_jacobian and returns the second output."""
                return self.element.get_jacobian(xi, el_nodal_coords)[1]

            det_J_elements = self.map(_get_det_J)(self.mesh.coords)
            object.__setattr__(
                self,
                "_det_J_elements_weights",
                jnp.einsum("eq,q->eq", det_J_elements, self.element.quad_weights),
            )

    def __check_init__(self) -> None:
        """Validates the mesh and element compatibility. Does a series of checks to ensure
        that the mesh and element are useable together.

        Raises:
            ValueError: If the mesh or element are not compatible.
            TypeError: If the mesh element connectivity is not of integer type.
        """
        coords = self.mesh.coords
        elements = self.mesh.elements

        if len(coords.shape) != 2:  # we use shape to allow lowering of operator methods
            # as jax.stages.ArgInfo only contains shape and type information
            raise ValueError(
                "Mesh coordinates must be a 2D array shaped (n_nodes, n_dim)."
            )
        if coords.shape[0] == 0:
            raise ValueError("Mesh must contain at least one node.")

        if (
            len(elements.shape) != 2
        ):  # we use shape to allow lowering of operator methods
            raise ValueError(
                "Mesh elements must be a 2D array shaped (n_elements, n_nodes_per_element)."
            )
        if elements.shape[0] == 0:
            raise ValueError("Mesh must contain at least one element.")
        if not jnp.issubdtype(elements.dtype, jnp.integer):
            raise TypeError("Mesh element connectivity must contain integer indices.")

        if not isinstance(elements, (jax.core.Tracer, jax.stages.ArgInfo)):
            # min/max checks cannot be performed while tracing and lowering stage
            flat_elements = elements.ravel()
            if flat_elements.min() < 0:
                raise ValueError(
                    "Mesh element connectivity contains negative node indices."
                )
            if flat_elements.max() >= coords.shape[0]:
                raise ValueError(
                    "Mesh element connectivity references nodes outside the mesh coordinates array."
                )

    def get_integration_weights(self) -> Array:
        """Returns the integration weights for the quadrature points of the mesh. This is
        the product of the determinant of the Jacobian and the quadrature weights, which
        can be used for integrating functions over the mesh.

        Returns:
            A `jax.Array` with the integration weights at each quadrature point of each
            element (shape: (n_elements, n_quad_points)).
        """
        if self.cache_weights and self._det_J_elements_weights is not None:
            # if cache_weights is True, we have computed the integration weights in
            # __post_init__ and stored them in _det_J_elements_weights
            return self._det_J_elements_weights
        else:

            def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
                """Calls the function element.get_jacobian and returns the second output."""
                return self.element.get_jacobian(xi, el_nodal_coords)[1]

            det_J_elements = self.map(_get_det_J)(self.mesh.coords)
            return jnp.einsum("eq,q->eq", det_J_elements, self.element.quad_weights)

    def _vmap_over_elements_and_quads(
        self, nodal_values: jax.Array, func: MappableOverElementsAndQuads
    ) -> jax.Array:
        """Helper function. Maps a function over the elements and quadrature points of the
        mesh.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            func: The function to map over the elements and quadrature points.

        Returns:
            A jax.Array with the results of the function applied at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values)).
        """

        def _at_each_element(args: tuple[Array, Array]) -> Array:
            el_nodal_values, el_nodal_coords = args
            return jax.vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return jax.lax.map(
            _at_each_element,
            xs=(nodal_values[self.mesh.elements], self.mesh.coords[self.mesh.elements]),
            batch_size=self.batch_size,
        )

    def map(
        self,
        func: MappableOverElementsAndQuads[P, RT],
        *,
        element_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements and quad points of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements and quad points.

        Args:
            func: The function to map over the elements and quadrature points.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
        """

        def _mapped(*values: P.args, **kwargs: P.kwargs) -> RT:
            # values should be arrays!
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(el_values: tuple) -> RT:
                def _at_each_quad(xi: jax.Array) -> RT:
                    return func(xi, *el_values, **kwargs)

                return jax.vmap(_at_each_quad)(self.element.quad_points)

            # Construct the tuple of inputs (xs) by iterating over _values
            # and gathering nodal values to elements where necessary.
            xs = tuple(
                v[self.mesh.elements] if i not in element_quantity else v
                for i, v in enumerate(_values)
            )

            return jax.lax.map(
                _at_each_element,
                xs=xs,
                batch_size=self.batch_size,
            )

        return _mapped

    def map_over_elements(
        self,
        func: MappableOverElements[P, RT],
        *,
        element_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements.

        Args:
            func: The function to map over the elements.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
        """

        def _mapped(*values: P.args, **kwargs: P.kwargs) -> RT:
            # values should be arrays!
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(el_values: tuple) -> RT:
                return func(*el_values, **kwargs)

            # Construct the tuple of inputs (xs) by iterating over _values
            # and gathering nodal values to elements where necessary.
            xs = tuple(
                v[self.mesh.elements] if i not in element_quantity else v
                for i, v in enumerate(_values)
            )
            return jax.lax.map(
                _at_each_element,
                xs=xs,
                batch_size=self.batch_size,
            )

        return _mapped

    def integrate(self, arg: jax.Array | Numeric) -> jax.Array:
        """Integrate a nodal_array, quad_array, or numeric value over the mesh.

        Args:
            arg: An array of nodal values (shape: (n_nodes, n_values)), an array of
                quadrature values (shape: (n_elements, n_quad_points, n_values)), or a
                numeric value (float or int).

        Returns:
            The integral of the nodal values or quadrature values over the mesh.
        """
        res = self.integrate_per_element(arg)
        return jnp.sum(res, axis=(0,))  # Sum over elements and quadrature points

    def integrate_per_element(self, arg: jax.Array | Numeric) -> jax.Array:
        """Integrate a nodal_array, quad_array, or numeric value over the mesh. Returning the
        integral per element.

        Args:
            arg: An array of nodal values (shape: (n_nodes, n_values)), an array of
                quadrature values (shape: (n_elements, n_quad_points, n_values)), or a
                numeric value (float or int).

        Returns:
            A `jax.Array` where each element contains the integral of the values in the
            element (shape: (n_elements, n_values)).
        """
        if isinstance(arg, Numeric):
            res = self._integrate_quad_array(self.eval(jnp.array([arg])))
        elif arg.shape[0] == self.mesh.elements.shape[0]:  # element field
            res = self._integrate_quad_array(arg)
        else:  # nodal field
            field_at_quads = self.eval(arg)
            res = self._integrate_quad_array(field_at_quads)

        return res

    def _integrate_quad_array(self, quad_values: jax.Array) -> jax.Array:
        """Integrates a given array of values at quadrature points over the mesh.

        Args:
            quad_values: The values at the quadrature points
                (shape: (n_elements, n_quad_points, n_values))

        Returns:
            A `jax.Array` where each element contains the integral of the values in the
            element (shape: (n_elements, n_values)).
        """

        # earlier einsum "eq...,eq->e..." although read like a plain reduction, but with n_quad_points > 1
        # XLA sees a batched contraction (batch e, contract q) and emits a `dot`, one tiny
        # GEMM per element. therefore, writing as broadcasted multiplication.
        weights = self.get_integration_weights()  # shape: (n_elements, n_quad_points)

        # broadcast weights to (n_elements, n_quad_points, 1, ..., 1)  match the shape of quad_values
        # for elementwise multiplication
        weights = weights.reshape(
            weights.shape + (1,) * (quad_values.ndim - weights.ndim)
        )
        # we sum over the quadrature points (axis=1) to get the integral per element
        return jnp.sum(quad_values * weights, axis=1)

    def eval(self, nodal_values: jax.Array) -> jax.Array:
        """Evaluates the nodal values at the quadrature points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the values of the nodal values at each quadrature point of
            each element (shape: (n_elements, n_quad_points, n_values)).
        """

        def _eval_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(
                xi, el_nodal_values, el_nodal_coords
            )  # nodal coords are needed for hermite elements, but not for lagrange elements, so we pass them in either way

        return self._vmap_over_elements_and_quads(nodal_values, _eval_quad)

    def grad(self, nodal_values: jax.Array) -> jax.Array:
        """Computes the gradient of the nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the gradient of the nodal values at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values, n_dim)).
        """

        def _gradient_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (gradient) on a quad point."""
            u_grad = self.element.gradient(xi, el_nodal_values, el_nodal_coords)
            return u_grad

        return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)

    def hess(self, nodal_values: jax.Array) -> jax.Array:
        """Computes the Hessian of the nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the Hessian of the nodal values at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values, n_dim, n_dim)).
        """

        def _hessian_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (hessian) on a quad point."""
            u_hess = self.element.hessian(xi, el_nodal_values, el_nodal_coords)
            return u_hess

        return self._vmap_over_elements_and_quads(nodal_values, _hessian_quad)

    def make_interpolate(self, points: jax.Array) -> Callable[[jax.Array], jax.Array]:
        """Build a fixed interpolation stencil on the host.

        Supports Tri3, Quad4, Tetrahedron4, and Hexahedron8 in their full spatial
        dimension. Mesh geometry and ``points`` (shape ``(n_points, n_dim)``)
        must be concrete: call this method outside JAX transformations. Rebuild
        the callable if the mesh geometry or points change.

        The returned callable accepts nodal values shaped ``(n_mesh_nodes, ...)``
        and returns values shaped ``(n_points, ...)``. It supports JIT, vmap,
        and differentiation with respect to nodal values, and captures only
        node indices and shape-function weights.

        Containment uses a reference tolerance of ``max(1e-10, 32 * eps)``
        and a physical tolerance scaled by element size and coordinate magnitude.
        Shared boundaries select the lowest containing element index. Points
        outside the mesh or with failed inverse mappings raise RuntimeError.

        Example:
            >>> interpolate = jax.jit(operator.make_interpolate(points))
            >>> point_values = interpolate(nodal_values)
        """
        return make_interpolator(
            self.element, self.mesh.coords, self.mesh.elements, points
        )

    def interpolate(self, arg: jax.Array, points: jax.Array) -> jax.Array:
        """Interpolate a Tri3 field at live physical points using only JAX.

        Supports dynamic points, geometry, and nodal values under JIT and vmap.
        Values have shape ``(n_mesh_nodes, ...)``, points ``(n_points, 2)``, and
        the result ``(n_points, ...)``. Points outside the mesh return NaN in
        both eager and transformed execution. Derivatives are local to the
        selected triangle; element selection is discrete.

        For fixed points, or other supported element types, construct and reuse
        a host-side stencil with ``make_interpolate(points)`` instead.
        """
        if type(self.element) is not Tri3:
            raise NotImplementedError(
                "Live interpolation supports only Tri3; use make_interpolate "
                "for fixed points with other supported element types."
            )
        if self.mesh.coords.shape[1] != 2:
            raise NotImplementedError(
                "Live interpolation does not support embedded elements."
            )
        if self.mesh.elements.shape[1] != 3:
            raise ValueError("Mesh connectivity width does not match Tri3.")
        arg, points = jnp.asarray(arg), jnp.asarray(points)
        if arg.ndim == 0 or arg.shape[0] != self.mesh.coords.shape[0]:
            raise ValueError(
                f"Nodal values must have shape ({self.mesh.coords.shape[0]}, ...)."
            )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Interpolation points must have shape (n_points, 2).")

        element_indices = find_containing_polygons(
            points, self.mesh.coords[self.mesh.elements]
        )
        valid = element_indices >= 0
        node_indices = self.mesh.elements[jnp.where(valid, element_indices, 0)]
        coords = self.mesh.coords[node_indices]
        jacobian = jnp.swapaxes(coords[:, 1:] - coords[:, :1], -1, -2)
        xi = jnp.linalg.solve(jacobian, (points - coords[:, 0])[..., None])[..., 0]
        weights = jnp.concatenate([1 - jnp.sum(xi, axis=1, keepdims=True), xi], axis=1)
        interpolated = jnp.einsum("pa,pa...->p...", weights, arg[node_indices])
        valid = valid & jnp.all(jnp.isfinite(weights), axis=1)
        mask = valid.reshape((points.shape[0],) + (1,) * (interpolated.ndim - 1))
        return jnp.where(mask, interpolated, jnp.nan)

    def _replace(self, **changes: Any) -> Self:
        """Returns a new instance of the Operator with the specified changes. Same as
        `dataclasses.replace(self, **changes)`. Inspired by NamedTuple's _replace method.

        Args:
            **changes: The attributes to change and their new values.
        """
        return replace(self, **changes)

    def quads(self) -> jax.Array:
        """Returns the quadrature points of the mesh in physical coordinates.

        Same as `op.eval(op.mesh.coords)`.

        Returns:
            An array with the quadrature points of the mesh in physical coordinates
            (shape: (n_elements, n_quad_points, n_dim)).
        """
        return self.eval(self.mesh.coords)

    def project(
        self,
        field: Array,
        colored_matrix: ColoredMatrix | None = None,
        lifter: Lifter | None = None,
    ) -> Array:
        """Projects a given field onto the finite element space defined by the mesh and
        element.

        Uses ``jax.experimental.sparse.linalg.spsolve`` to solve the linear system
        resulting from the projection. If `colored_matrix` is None (the default), a
        compatible colored matrix is assembled from `self.mesh.elements`. When a
        `colored_matrix` is passed explicitly, it must be compatible with the dimensions
        of the projected field and with the chosen fem space.

        Args:
            field: The field to project, defined at the quadrature points
                (shape: (n_elements, n_quad_points, ...)).
            colored_matrix: Optional colored matrix representing the finite element space.
                If omitted, it is constructed from `self.mesh.elements`.
            lifter: Optional lifter used to lift and reduce between the full and reduced
                spaces.
        """
        nnodes = self.mesh.coords.shape[0]

        fn_project = make_project_function(
            nnodes=nnodes,
            colored_matrix=colored_matrix,
            elements=self.mesh.elements,  # ignored if colored_matrix is provided
            lifter=lifter,
        )
        return fn_project(self, field)
