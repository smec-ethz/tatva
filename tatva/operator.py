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
    Generic,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

import jax
import jax.numpy as jnp
from jax import Array
from jax.errors import TracerBoolConversionError
from jax_autovmap import autovmap

from tatva.element import Element
from tatva.mesh import Mesh, find_containing_polygons
from tatva.utils import make_project_function

if TYPE_CHECKING:
    from tatva.lifter import Lifter
    from tatva.sparse import ColoredMatrix

P = ParamSpec("P")
RT = TypeVar("RT", bound=jax.Array | tuple, covariant=True)
ElementT = TypeVar("ElementT", bound=Element)
Numeric: TypeAlias = float | int | jnp.number


class MappableOverElementsAndQuads(Protocol[P, RT]):
    """Internal protocol for functions that are mapped over elements using
    `Operator.map`."""

    @staticmethod
    def __call__(
        xi: jax.Array,
        *el_values: P.args,
        **el_kwargs: P.kwargs,
    ) -> RT: ...


MappableOverElements: TypeAlias = Callable[P, RT]


class MappedCallable(Protocol[P, RT]):
    @staticmethod
    def __call__(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> RT: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Operator(Generic[ElementT]):
    """A class that provides an Operator for finite element method (FEM) assembly.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The element type used for the finite element method.

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

    def __post_init__(self) -> None:
        # run initialization checks to ensure mesh/element compatibility and basic
        # shape/type validations
        self.__check_init__()

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

        if coords.ndim != 2:
            raise ValueError(
                "Mesh coordinates must be a 2D array shaped (n_nodes, n_dim)."
            )
        if coords.shape[0] == 0:
            raise ValueError("Mesh must contain at least one node.")

        if elements.ndim != 2:
            raise ValueError(
                "Mesh elements must be a 2D array shaped (n_elements, n_nodes_per_element)."
            )
        if elements.shape[0] == 0:
            raise ValueError("Mesh must contain at least one element.")
        if not jnp.issubdtype(elements.dtype, jnp.integer):
            raise TypeError("Mesh element connectivity must contain integer indices.")

        flat_elements = elements.ravel()
        try:
            if flat_elements.min() < 0:
                raise ValueError(
                    "Mesh element connectivity contains negative node indices."
                )
            if flat_elements.max() >= coords.shape[0]:
                raise ValueError(
                    "Mesh element connectivity references nodes outside the mesh coordinates array."
                )
        except TracerBoolConversionError:
            pass

    def get_integration_weights(self) -> Array:
        """Returns the integration weights for the quadrature points of the mesh. This is
        the product of the determinant of the Jacobian and the quadrature weights, which
        can be used for integrating functions over the mesh.

        Returns:
            A `jax.Array` with the integration weights at each quadrature point of each
            element (shape: (n_elements, n_quad_points)).
        """
        if self.cache_weights:
            # if cache_weights is True, we have computed the integration weights in
            # __post_init__ and stored them in _det_J_elements_weights
            return self._det_J_elements_weights  # pyright: ignore[reportAttributeAccessIssue]
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

        return jnp.einsum("eq...,eq->e...", quad_values, self.get_integration_weights())

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

    def interpolate(self, arg: jax.Array, points: jax.Array) -> jax.Array:
        """Interpolates nodal values to a set of points in the physical space.

        Args:
            arg: The nodal values to interpolate.
            points: The points to interpolate the function or nodal values to.

        Returns:
            A `jax.Array` with the interpolated values at the given points.
        """

        @jax.jit
        def compute_rhs(point: jax.Array, nodal_coords: jax.Array) -> jax.Array:
            xi0 = self.element.quad_points[0]
            x0, _, _ = self.element.get_local_values(xi0, nodal_coords, nodal_coords)
            return x0 - point

        @jax.jit
        def compute_lhs(nodal_coords: jax.Array) -> jax.Array:
            dfdxi = jax.jacrev(self.element.get_local_values)
            return dfdxi(self.element.quad_points[0], nodal_coords, nodal_coords)[0]

        @autovmap(point=1, nodal_coords=2)
        def _map_physical_to_reference(
            point: jax.Array, nodal_coords: jax.Array
        ) -> jax.Array:
            rhs = compute_rhs(point, nodal_coords)
            lhs = compute_lhs(nodal_coords)
            delta_xi = jnp.linalg.solve(lhs, -rhs)
            return self.element.quad_points[0] + delta_xi

        def map_physical_to_reference(
            points: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            element_indices: Array = find_containing_polygons(
                points, self.mesh.coords[self.mesh.elements]
            )
            valid_indices = element_indices != -1
            safe_element_indices = jnp.where(valid_indices, element_indices, 0)
            valid_elements = self.mesh.elements[safe_element_indices]
            return (
                _map_physical_to_reference(
                    points,
                    self.mesh.coords[valid_elements],
                ),
                valid_elements,
                valid_indices,
            )

        valid_quad_points, valid_elements, valid_indices = map_physical_to_reference(
            points
        )
        interpolated = self._interpolate_direct(arg, valid_quad_points, valid_elements)

        try:
            if bool(jnp.any(~valid_indices)):
                raise RuntimeError(
                    "Some points are outside the mesh, revise the points"
                )
        except TracerBoolConversionError:
            pass

        mask = valid_indices.reshape(
            (valid_indices.shape[0],) + (1,) * (interpolated.ndim - 1)
        )
        return jnp.where(mask, interpolated, jnp.nan)

    def _interpolate_direct(
        self,
        nodal_values: jax.Array,
        valid_quad_points: jax.Array,
        valid_elements: jax.Array,
    ) -> jax.Array:
        """Interpolates the given nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            valid_quad_points: The quadrature points in the reference element
                (shape: (n_valid_points, n_dim))
            valid_elements: The indices of the elements containing the quadrature points
                (shape: (n_valid_points,))

        Returns:
            A `jax.Array` with the values of the nodal values at each quadrature point of
            each element (shape: (n_valid_points, n_values)).
        """

        def _interpolate_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(
                xi, el_nodal_values, el_nodal_coords
            )  # nodal coords are needed for hermite elements, but not for lagrange elements, so we pass them in either way

        return jax.vmap(
            _interpolate_quad,
            in_axes=(0, 0, 0),
        )(
            valid_quad_points,
            nodal_values[valid_elements],
            self.mesh.coords[valid_elements],
        )

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
