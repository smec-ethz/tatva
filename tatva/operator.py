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
from jax.errors import TracerBoolConversionError
from jax_autovmap import autovmap

from tatva.element import Element
from tatva.function_space import FunctionSpace
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
        space: The function space to assemble over, pairing a mesh with an element and
            owning the connectivity between them.
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

    Two connectivity tables are read from the space and are not interchangeable: dof
    values are gathered through `space.dofmap`, and node coordinates through
    `space.geometry_dofmap`. They are the same table for a degree-1 space, and diverge as
    soon as the field carries dofs the geometry does not.

    Example:
        >>> from tatva import FunctionSpace, Mesh, Operator, Tri3
        >>> mesh = Mesh.unit_square(10, 10)  # Create a mesh
        >>> space = FunctionSpace(mesh, Tri3())  # Pair it with an element
        >>> operator = Operator(space)
        >>> dof_values = jnp.array(...)  # Nodal values at the mesh nodes
        >>> energy = operator.integrate(energy_density)(dof_values)
    """

    space: FunctionSpace[ElementT]
    batch_size: int | None = field(metadata=dict(static=True), default=None)
    cache_weights: bool = field(metadata=dict(static=True), default=False)

    def __post_init__(self) -> None:
        # run initialization checks to ensure mesh/element compatibility and basic
        # shape/type validations
        self.__check_init__()
        if self.batch_size is None:
            object.__setattr__(self, "batch_size", self.n_elements)

        if self.cache_weights:

            def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
                """Calls the function element.get_jacobian and returns the second output."""
                return self.element.get_jacobian(xi, el_nodal_coords)[1]

            det_J_elements = self.map(_get_det_J, geometry_quantity=(0,))(
                self.mesh.coords
            )
            object.__setattr__(
                self,
                "_det_J_elements_weights",
                jnp.einsum("eq,q->eq", det_J_elements, self.element.quad_weights),
            )

    # ---------------------------------------------------------------- space accessors

    @property
    def mesh(self) -> Mesh:
        """The mesh underlying the space."""
        return self.space.mesh

    @property
    def element(self) -> ElementT:
        """The element defining the basis. Static under jax transformations."""
        return self.space.element

    @property
    def dofmap(self) -> Array:
        """Global dof index of each element-local dof (shape: (n_elements, n_dofs))."""
        assert self.space.dofmap is not None
        return self.space.dofmap

    @property
    def geometry_dofmap(self) -> Array:
        """Global node index of each element-local geometry node."""
        assert self.space.geometry_dofmap is not None
        return self.space.geometry_dofmap

    @property
    def n_elements(self) -> int:
        """Number of elements in the mesh."""
        return self.geometry_dofmap.shape[0]

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

            det_J_elements = self.map(_get_det_J, geometry_quantity=(0,))(
                self.mesh.coords
            )
            return jnp.einsum("eq,q->eq", det_J_elements, self.element.quad_weights)

    def _vmap_over_elements_and_quads(
        self, dof_values: jax.Array, func: MappableOverElementsAndQuads
    ) -> jax.Array:
        """Helper function. Maps a function over the elements and quadrature points of the
        mesh.

        Args:
            dof_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            func: The function to map over the elements and quadrature points.

        Returns:
            A jax.Array with the results of the function applied at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values)).
        """

        def _at_each_element(args: tuple[Array, Array]) -> Array:
            el_dof_values, el_nodal_coords = args
            return jax.vmap(
                partial(
                    func,
                    el_dof_values=el_dof_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return jax.lax.map(
            _at_each_element,
            xs=(
                dof_values[self.dofmap],
                self.mesh.coords[self.geometry_dofmap],
            ),
            batch_size=self.batch_size,
        )

    def _gather(
        self,
        values: tuple[jax.Array, ...],
        element_quantity: Sequence[int],
        geometry_quantity: Sequence[int],
    ) -> tuple[jax.Array, ...]:
        """Gathers global arrays to element-local ones, one table per argument kind.

        Args:
            values: The global arrays passed to a mapped function.
            element_quantity: Indices of arguments already defined per element, which are
                passed through untouched.
            geometry_quantity: Indices of arguments indexed by mesh node rather than by
                dof, gathered through the geometry table. Node coordinates are the usual
                case.
        """
        overlap = set(element_quantity) & set(geometry_quantity)
        if overlap:
            raise ValueError(
                f"arguments {sorted(overlap)} are marked both element_quantity and "
                f"geometry_quantity; an argument is gathered exactly one way"
            )

        def _gather_one(i: int, v: jax.Array) -> jax.Array:
            if i in element_quantity:
                return v
            if i in geometry_quantity:
                return v[self.geometry_dofmap]
            return v[self.dofmap]

        return tuple(_gather_one(i, v) for i, v in enumerate(values))

    def map(
        self,
        func: MappableOverElementsAndQuads[P, RT],
        *,
        element_quantity: Sequence[int] = (),
        geometry_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements and quad points of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements and quad points.

        Args:
            func: The function to map over the elements and quadrature points.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
            geometry_quantity: Indices of the arguments of `func` that are indexed by mesh
                node instead of by dof, and so are gathered through the geometry table.
                Pass this for node coordinates; the two tables coincide only while every
                dof sits on a vertex.
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
            xs = self._gather(_values, element_quantity, geometry_quantity)

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
        geometry_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements.

        Args:
            func: The function to map over the elements.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
            geometry_quantity: Indices of the arguments of `func` that are indexed by mesh
                node instead of by dof, and so are gathered through the geometry table.
        """

        def _mapped(*values: P.args, **kwargs: P.kwargs) -> RT:
            # values should be arrays!
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(el_values: tuple) -> RT:
                return func(*el_values, **kwargs)

            # Construct the tuple of inputs (xs) by iterating over _values
            # and gathering nodal values to elements where necessary.
            xs = self._gather(_values, element_quantity, geometry_quantity)
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
        elif arg.shape[0] == self.n_elements:  # element field
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

    def eval(self, dof_values: jax.Array) -> jax.Array:
        """Evaluates the nodal values at the quadrature points.

        Args:
            dof_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the values of the nodal values at each quadrature point of
            each element (shape: (n_elements, n_quad_points, n_values)).
        """

        def _eval_quad(
            xi: jax.Array, el_dof_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(
                xi, el_dof_values, el_nodal_coords
            )  # nodal coords are needed for hermite elements, but not for lagrange elements, so we pass them in either way

        return self._vmap_over_elements_and_quads(dof_values, _eval_quad)

    def grad(self, dof_values: jax.Array) -> jax.Array:
        """Computes the gradient of the nodal values at the quad points.

        Args:
            dof_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the gradient of the nodal values at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values, n_dim)).
        """

        def _gradient_quad(
            xi: jax.Array, el_dof_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (gradient) on a quad point."""
            u_grad = self.element.gradient(xi, el_dof_values, el_nodal_coords)
            return u_grad

        return self._vmap_over_elements_and_quads(dof_values, _gradient_quad)

    def make_interpolate(self, points: jax.Array) -> Callable[[jax.Array], jax.Array]:
        """Returns a function that interpolates nodal values to a set of static points.

        Args:
            points: The statically known points to interpolate to.

        Returns:
            A function that takes nodal values (`arg`) and returns the interpolated values
            at the statically known points.
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
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            element_indices: Array = find_containing_polygons(
                points, self.mesh.coords[self.geometry_dofmap]
            )
            valid_indices = element_indices != -1
            safe_element_indices = jnp.where(valid_indices, element_indices, 0)
            # Point location is geometry; the dofs of the same cells are what the field is
            # then read from, so both tables are sliced by the same element indices.
            valid_geometry = self.geometry_dofmap[safe_element_indices]
            valid_dofs = self.dofmap[safe_element_indices]
            return (
                _map_physical_to_reference(
                    points,
                    self.mesh.coords[valid_geometry],
                ),
                valid_geometry,
                valid_dofs,
                valid_indices,
            )

        xi_in_valid_element, valid_geometry, valid_dofs, valid_indices = (
            map_physical_to_reference(points)
        )

        try:
            if bool(jnp.any(~valid_indices)):
                raise RuntimeError(
                    "Some points are outside the mesh, revise the points"
                )
        except TracerBoolConversionError:
            pass

        def interpolate_fn(arg: jax.Array) -> jax.Array:
            interpolated = self._interpolate_direct(
                arg, xi_in_valid_element, valid_geometry, valid_dofs
            )
            mask = valid_indices.reshape(
                (valid_indices.shape[0],) + (1,) * (interpolated.ndim - 1)
            )
            return jnp.where(mask, interpolated, jnp.nan)

        return interpolate_fn

    def interpolate(self, arg: jax.Array, points: jax.Array) -> jax.Array:
        """Interpolates nodal values to a set of points in the physical space.

        Args:
            arg: The nodal values to interpolate.
            points: The points to interpolate the function or nodal values to.

        Returns:
            A `jax.Array` with the interpolated values at the given points.
        """
        return self.make_interpolate(points)(arg)

    def _interpolate_direct(
        self,
        dof_values: jax.Array,
        xi_in_valid_element: jax.Array,
        valid_geometry: jax.Array,
        valid_dofs: jax.Array,
    ) -> jax.Array:
        """Interpolates the given nodal values at the quad points.

        Args:
            dof_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            xi_in_valid_element: The points in the reference element
                (shape: (n_valid_points, n_dim))
            valid_geometry: The geometry nodes of the element containing each point
                (shape: (n_valid_points, n_geometry_nodes)).
            valid_dofs: The dofs of the element containing each point
                (shape: (n_valid_points, n_dofs)).

        Returns:
            A `jax.Array` with the values of the nodal values at each quadrature point of
            each element (shape: (n_valid_points, n_values)).
        """

        def _interpolate_at_xi(
            xi: jax.Array, el_dof_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (interpolator) on arbitrary reference coord."""
            return self.element.interpolate(
                xi, el_dof_values, el_nodal_coords
            )  # nodal coords are needed for hermite elements, but not for lagrange elements, so we pass them in either way

        return jax.vmap(
            _interpolate_at_xi,
            in_axes=(0, 0, 0),
        )(
            xi_in_valid_element,
            dof_values[valid_dofs],
            self.mesh.coords[valid_geometry],
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
        # This pushes the coordinates through the *dof* table, which is the geometry table
        # only while every dof sits on a vertex. Mapping the reference quadrature points
        # through the geometry element is the general form, and is needed as soon as the
        # space carries edge or interior dofs.
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
        compatible colored matrix is assembled from `self.dofmap`. When a
        `colored_matrix` is passed explicitly, it must be compatible with the dimensions
        of the projected field and with the chosen fem space.

        Args:
            field: The field to project, defined at the quadrature points
                (shape: (n_elements, n_quad_points, ...)).
            colored_matrix: Optional colored matrix representing the finite element space.
                If omitted, it is constructed from `self.dofmap`.
            lifter: Optional lifter used to lift and reduce between the full and reduced
                spaces.
        """
        # The sparsity pattern is that of the dof graph, not the mesh node graph — the two
        # coincide only while every dof sits on a vertex.
        fn_project = make_project_function(
            nnodes=self.space.n_scalar_dofs,
            colored_matrix=colored_matrix,
            elements=self.dofmap,  # ignored if colored_matrix is provided
            lifter=lifter,
        )
        return fn_project(self, field)
