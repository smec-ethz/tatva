from __future__ import annotations

from functools import partial
from typing import Any, Callable, Concatenate, ParamSpec, Protocol, TypeAlias, overload

import equinox as eqx
import jax
import jax.numpy as jnp

from femsolver._element import Element
from femsolver.mesh import Mesh, find_containing_polygons
from femsolver.utils import auto_vmap

# TODO: naming of these types

P = ParamSpec("P")


Form: TypeAlias = Callable[Concatenate[jax.Array, jax.Array, P], jax.Array]


class FormCallable(Protocol[P]):
    @staticmethod
    def __call__(
        nodal_values: jax.Array,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> jax.Array: ...


class _VmapOverElementsCallable(Protocol):
    """Internal protocol for functions that are mapped over elements and quadrature points
    using `Operator._vmap_over_elements_and_quads`."""

    @staticmethod
    def __call__(
        xi: jax.Array,
        el_nodal_values: jax.Array,
        el_nodal_coords: jax.Array,
    ) -> jax.Array: ...


class Operator(eqx.Module):
    """A class that provides an Operator for finite element method (FEM) assembly.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The element type used for the finite element method.

    Provides several operators for evaluating and integrating functions over the mesh,
    such as `integrate`, `eval`, and `grad`. These operators can be used to compute
    integrals, evaluate functions at quadrature points, and compute gradients of
    functions at quadrature points.

    Example:
        >>> from femsolver import Mesh, Tri3, Operator
        >>> mesh = Mesh.unit_square(10, 10)  # Create a mesh
        >>> element = Tri3()  # Define an element type
        >>> operator = Operator(mesh, element)
        >>> nodal_values = jnp.array(...)  # Nodal values at the mesh nodes
        >>> energy = operator.integrate(energy_density)(nodal_values)
    """

    mesh: Mesh
    element: Element

    def _vmap_over_elements_and_quads(
        self,
        nodal_values: jax.Array,
        func: _VmapOverElementsCallable,
        additional_values_at_quad: jax.Array = None,
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

        def _at_each_element(
            el_nodal_values: jax.Array,
            el_nodal_coords: jax.Array,
            additional_values_at_quad: jax.Array = None,
        ) -> jax.Array:
            return eqx.filter_vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                    additional_values_at_quad=additional_values_at_quad,
                )
            )(self.element.quad_points)

        return eqx.filter_vmap(
            _at_each_element,
            in_axes=(0, 0, eqx.if_array(0)),
        )(
            nodal_values[self.mesh.elements],
            self.mesh.coords[self.mesh.elements],
            additional_values_at_quad,
        )

    @overload
    def integrate(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def integrate(self, arg: jax.Array) -> jax.Array: ...
    def integrate(self, arg):
        """Integrate a function/nodal_array/quad_array over the mesh.

        If a function is provided, it returns a function that integrates the function given the nodal values.
        If nodal values or quad values are given, it returns the integral.

        Returns:
            A function that integrates the given function over the mesh, or the integral
            (**scalar**) of the nodal values or quadrature values over the mesh.
        """
        if isinstance(arg, Callable):
            return self._integrate_functor(arg, sum=True)

        if arg.shape[0] == self.mesh.elements.shape[0]:  # element field
            res = self._integrate_quad_array(arg)
        else:  # nodal field
            res = self._integrate_nodal_array(arg)

        return jnp.sum(res, axis=(0,))  # Sum over elements and quadrature points

    @overload
    def integrate_per_element(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def integrate_per_element(self, arg: jax.Array) -> jax.Array: ...
    def integrate_per_element(self, arg):
        """Integrate a function/nodal_array/quad_array over the mesh, returning the result per element.

        If a function is provided, it returns a function that integrates the function given the nodal values.
        If nodal values or quad values are given, it returns the integral per element.

        Returns:
            A function that integrates the given function over the mesh, or the integral
            of the nodal values or quadrature values over each element.
        """
        if isinstance(arg, Callable):
            return self._integrate_functor(arg, sum=False)

        if arg.shape[0] == self.mesh.elements.shape[0]:
            return self._integrate_quad_array(arg)
        else:
            return self._integrate_nodal_array(arg)

    def _integrate_functor(
        self, func: Form[P], *, sum: bool = False
    ) -> FormCallable[P]:
        """Decorator to integrate a function over the mesh.

        Returns a function that takes nodal values and additional values at quadrature
        points and returns the integrated value over the mesh.
        """

        @eqx.filter_jit
        def _integrate(
            nodal_values: jax.Array,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> jax.Array:
            """Integrates the given local function over the mesh.

            Args:
                nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
                *args: Additional arguments to pass to the function (optional)
            """

            def _integrate_quads(
                xi: jax.Array,
                el_nodal_values: jax.Array,
                el_nodal_coords: jax.Array,
                additional_values_at_quad: jax.Array = None,
            ) -> jax.Array:
                """Calls the function (integrand) on a quad point. Multiplying by the
                determinant of the Jacobian.
                """
                u, u_grad, detJ = self.element.get_local_values(
                    xi, el_nodal_values, el_nodal_coords
                )
                return func(u, u_grad, *args, **kwargs) * detJ

            res = jnp.einsum(
                "eq...,q...->eq...",
                self._vmap_over_elements_and_quads(nodal_values, _integrate_quads),
                self.element.quad_weights,
            )
            if sum:
                return jnp.sum(
                    res, axis=(0, 1)
                )  # Sum over elements and quadrature points
            else:
                return res

        return _integrate

    def _integrate_quad_array(self, quad_values: jax.Array) -> jax.Array:
        """Integrates a given array of values at quadrature points over the mesh.

        Args:
            quad_values: The values at the quadrature points (shape: (n_elements, n_quad_points, n_values))

        Returns:
            A jax.Array where each element contains the integral of the values in the element
        """
        det_J_elements = self._vmap_over_elements_and_quads(
            jnp.zeros(1),  # Dummy nodal values
            lambda xi, el_nodal_values, el_nodal_coords: self.element.get_jacobian(
                xi, el_nodal_coords
            )[1],
        )
        return jnp.einsum(
            "eq...,eq->e...",
            quad_values,
            jnp.einsum("eq,q->eq", det_J_elements, self.element.quad_weights),
        )

    def _integrate_nodal_array(self, nodal_values: jax.Array) -> jax.Array:
        """Integrates a given array of nodal values over the mesh.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A jax.Array where each element contains the integral of the nodal values in the element
        """
        return self._integrate_quad_array(self.eval(nodal_values))

    @overload
    def eval(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def eval(self, arg: jax.Array, *args: tuple[Any, ...]) -> jax.Array: ...
    def eval(self, arg, *args):
        """Evaluates the function at the quadrature points.

        If a function is provided, it returns a function that interpolates the nodal values
        at the quadrature points. If nodal values are provided, it returns the interpolated
        values at the quadrature points.
        """
        if isinstance(arg, Callable):
            return self._eval_functor(arg, *args)
        else:
            return self._eval_direct(arg)

    def _eval_functor(self, func: Form[P]) -> FormCallable[P]:
        """Decorator to interpolate a local function at the mesh elements quad points.

        Returns a function that takes nodal values and additional values at quadrature
        points and returns the interpolated values at the quadrature points.
        *(shape: (n_elements, n_quad_points, n_values))*
        """

        def _eval(
            nodal_values: jax.Array,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> jax.Array:
            """Interpolates the given function at the mesh nodes.

            Args:
                nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
                *args: Additional arguments to pass to the function (optional)
            """

            def _eval_quad(
                xi: jax.Array,
                el_nodal_values: jax.Array,
                el_nodal_coords: jax.Array,
                additional_values_at_quad: jax.Array = None,
            ) -> jax.Array:
                """Calls the function (interpolator) on a quad point."""
                u, u_grad, _detJ = self.element.get_local_values(
                    xi, el_nodal_values, el_nodal_coords
                )
                return func(u, u_grad, *args, **kwargs)

            return self._vmap_over_elements_and_quads(
                nodal_values, _eval_quad, _additional_values_at_quad
            )

        return _eval

    def _eval_direct(
        self,
        nodal_values: jax.Array,
    ) -> jax.Array:
        """Interpolates the given function at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
        """

        def _eval_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(xi, el_nodal_values)

        return self._vmap_over_elements_and_quads(
            nodal_values, _eval_quad, _additional_values_at_quad
        )

    @overload
    def grad(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def grad(self, arg: jax.Array, *args: tuple[Any, ...]) -> jax.Array: ...
    def grad(self, arg, *args):
        """Evaluates the gradient of the function at the quadrature points.

        If a function is provided, it returns a function that computes the gradient of the
        nodal values at the quadrature points. If nodal values are provided, it returns the
        gradient of the nodal values at the quadrature points.
        """
        if isinstance(arg, Callable):
            return self._grad_functor(arg, *args)
        else:
            return self._grad_direct(arg)

    def _grad_direct(self, nodal_values: jax.Array) -> jax.Array:
        """Computes the gradient of the nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
        """

        def _gradient_quad(
            xi: jax.Array,
            el_nodal_values: jax.Array,
            el_nodal_coords: jax.Array,
            additional_values_at_quad: jax.Array = None,
        ) -> jax.Array:
            """Calls the function (gradient) on a quad point."""
            u_grad = self.element.gradient(xi, el_nodal_values, el_nodal_coords)
            return u_grad

        return self._vmap_over_elements_and_quads(
            nodal_values, _gradient_quad, _additional_values_at_quad
        )

    def _grad_functor(self, func: Form[P]) -> FormCallable[P]:
        """Decorator to compute the gradient of a local function at the mesh elements quad
        points.

        Returns a function that takes nodal values and additional values at nodal
        points and returns the gradient of the evaluated function at the quadrature points.
        *(shape: (n_elements, n_quad_points, n_values))*
        """
        # TODO: Not sure this is useful
        ...

    @overload
    def interpolate(self, arg: Form, points: jax.Array) -> FormCallable: ...
    @overload
    def interpolate(
        self, arg: jax.Array, points: jax.Array, *additional_values_at_quad: jax.Array
    ) -> jax.Array: ...
    def interpolate(self, arg, points, *additional_values_at_quad):
        """Interpolates a function or nodal values to a set of points in the physical space.

        If a function is provided, it returns a function that interpolates the function at the
        given points. If nodal values are provided, it returns the interpolated nodal values
        at the given points.

        Args:
            arg: The function to interpolate or the nodal values to interpolate.
            points: The points to interpolate the function or nodal values to.
            *_additional_values_at_quad: Additional values at the quadrature points (optional)
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

        @auto_vmap(point=1, nodal_coords=2)
        def _map_physical_to_reference(
            point: jax.Array, nodal_coords: jax.Array
        ) -> jax.Array:
            rhs = compute_rhs(point, nodal_coords)
            lhs = compute_lhs(nodal_coords)
            delta_xi = jnp.linalg.solve(lhs, -rhs)
            return self.element.quad_points[0] + delta_xi

        def map_physical_to_reference(points: jax.Array) -> jax.Array:
            element_indices = find_containing_polygons(
                points, self.mesh.coords[self.mesh.elements]
            )
            valid_indices = element_indices != -1
            return (
                _map_physical_to_reference(
                    points[valid_indices],
                    self.mesh.coords[
                        self.mesh.elements[element_indices[valid_indices]]
                    ],
                ),
                self.mesh.elements[element_indices[valid_indices]],
            )

        valid_quad_points, valid_elements = map_physical_to_reference(points)

        if valid_quad_points.shape[0] != points.shape[0]:
            raise RuntimeError("Some points are outside the mesh, revise the points")

        if isinstance(arg, Callable):
            return self._interpolate_functor(arg, valid_quad_points, valid_elements)
        else:
            return self._interpolate_direct(
                arg,
                valid_quad_points,
                valid_elements,
                *additional_values_at_quad,
            )

    def _interpolate_functor(
        self, func: Form, valid_quad_points: jax.Array, valid_elements: jax.Array
    ) -> FormCallable:
        """Decorator to interpolate a local function at the mesh elements quad points.

        Returns a function that takes nodal values and additional values at quadrature
        points and returns the interpolated values at the quadrature points.
        *(shape: (n_elements, n_quad_points, n_values))*
        """

        def _interpolate(
            nodal_values: jax.Array,
            _additional_values_at_quad: jax.Array = None,
        ) -> jax.Array:
            """Interpolates the given function at the mesh nodes.

            Args:
                nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
                *_additional_values_at_quad: Additional values at the quadrature points (optional)
            """

            def _interpolate_quad(
                xi: jax.Array,
                el_nodal_values: jax.Array,
                el_nodal_coords: jax.Array,
                additional_values_at_quad: jax.Array = None,
            ) -> jax.Array:
                """Calls the function (interpolator) on a quad point."""
                u, u_grad, _detJ = self.element.get_local_values(
                    xi, el_nodal_values, el_nodal_coords
                )
                return func(u, u_grad, additional_values_at_quad)

            return eqx.filter_vmap(
                _interpolate_quad,
                in_axes=(0, 0, 0, eqx.if_array(0)),
            )(
                valid_quad_points,
                nodal_values[valid_elements],
                self.mesh.coords[valid_elements],
                _additional_values_at_quad,
            )

        return _interpolate

    def _interpolate_direct(
        self,
        nodal_values: jax.Array,
        valid_quad_points: jax.Array,
        valid_elements: jax.Array,
        _additional_values_at_quad: jax.Array = None,
    ) -> jax.Array:
        """Interpolates the given nodal values at the quad points."""

        def _interpolate_quad(
            xi: jax.Array,
            el_nodal_values: jax.Array,
            el_nodal_coords: jax.Array,
            additional_values_at_quad: jax.Array = None,
        ) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(xi, el_nodal_values)

        return eqx.filter_vmap(
            _interpolate_quad,
            in_axes=(0, 0, 0, eqx.if_array(0)),
        )(
            valid_quad_points,
            nodal_values[valid_elements],
            self.mesh.coords[valid_elements],
            _additional_values_at_quad,
        )


class OperatorUpdatedLagrangian(Operator):
    """A class that provides an Operator for finite element method (FEM) assembly.

    Other than the base `Operator`, this class is specifically designed for total Lagrangian
    formalism. Meaning, integration is done over the deformed configuration of the mesh.
    Nodal values are expected to be the displacements of the nodes, not an arbitrary field.
    Ensure your model is set up accordingly.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The element type used for the finite element method.

    Provides several operators for evaluating and integrating functions over the mesh,
    such as `integrate`, `eval`, and `grad`. These operators can be used to compute
    integrals, evaluate functions at quadrature points, and compute gradients of
    functions at quadrature points.
    """

    def _vmap_over_elements_and_quads(
        self, nodal_values: jax.Array, func: _VmapOverElementsCallable
    ) -> jax.Array:
        """Helper function. Maps a function over the elements and quadrature points of the
        mesh.

        Args:
            nodal_values: The nodal **displacement** at the element's nodes (shape: (n_nodes, n_dim))
            func: The function to map over the elements and quadrature points.

        Returns:
            A jax.Array with the results of the function applied at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values)).
        """

        def _at_each_element(
            el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            return jax.vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return jax.vmap(
            _at_each_element,
            in_axes=(0, 0),
        )(
            nodal_values[self.mesh.elements],
            (self.mesh.coords + nodal_values)[self.mesh.elements],
        )
