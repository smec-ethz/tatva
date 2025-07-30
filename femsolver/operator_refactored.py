from __future__ import annotations

from functools import partial
from typing import Callable, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp

from femsolver._quadrature import Element
from femsolver.mesh import Mesh


class IntegrateFunction(Protocol):
    @staticmethod
    def __call__(
        nodal_values: jax.Array,
        *additional_values_at_quad: jax.Array,
    ) -> jax.Array: ...


Integrand = Callable[
    [jax.Array, jax.Array, *tuple[jax.Array, ...]], jax.Array
]  # values, gradients, *additionals -> result

# # TODO: This will give better type checking (will show the actual expected arguments)
# # but it will report an error if the function arguments are named differently :(
# class Integrand(Protocol):
#     @staticmethod
#     def __call__(
#         values: jax.Array,
#         gradients: jax.Array,
#         *additionals: jax.Array,
#     ) -> jax.Array: ...


class Operator(eqx.Module):
    """A class used to represent a Operator for finite element method (FEM) simulations.

    Attributes:
        element: The element to use for the integration.
        integrand: The integrand to use for the integration.

    Methods:
        interpolate: Interpolates the nodal values at the given points.
        integrate: Integrates the integrand over the cells.
    """

    mesh: Mesh
    element: Element

    def _vmap_over_elements_and_quads(
        self, nodal_values: jax.Array, func: Callable
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
        )(nodal_values[self.mesh.elements], self.mesh.nodes[self.mesh.elements])

    def integrate(self, func: Integrand) -> IntegrateFunction:
        """Decorator to integrate a function over the mesh.

        Returns a function that takes nodal values and additional values at quadrature
        points and returns the integrated value over the mesh.
        """

        def _integrate(
            nodal_values: jax.Array,
            *_additional_values_at_quad: jax.Array,
        ) -> jax.Array:
            """Integrates the given local function over the mesh.

            Args:
                nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
                *_additional_values_at_quad: Additional values at the quadrature points (optional)
            """

            def _integrate_quads(
                xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
            ) -> jax.Array:
                """Calls the function (integrand) on a quad point. Multiplying by the
                determinant of the Jacobian.
                """
                u, u_grad, detJ = self.element.get_local_values(
                    xi, el_nodal_values, el_nodal_coords
                )
                return func(u, u_grad) * detJ

            return jnp.sum(
                jnp.einsum(
                    "eq...,q...->eq...",
                    self._vmap_over_elements_and_quads(nodal_values, _integrate_quads),
                    self.element.quad_weights,
                )
            )

        return _integrate

    def eval_at_quad(self, func: Integrand) -> IntegrateFunction:
        """Decorator to interpolate a local function at the mesh elements quad points.

        Returns a function that takes nodal values and additional values at quadrature
        points and returns the interpolated values at the quadrature points.
        *(shape: (n_elements, n_quad_points, n_values))*
        """

        def _interpolate(
            nodal_values: jax.Array,
            *_additional_values_at_quad: jax.Array,
        ) -> jax.Array:
            """Interpolates the given function at the mesh nodes.

            Args:
                nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
                *_additional_values_at_quad: Additional values at the quadrature points (optional)
            """

            def _interpolate_quad(
                xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
            ) -> jax.Array:
                """Calls the function (interpolator) on a quad point."""
                u, u_grad, _detJ = self.element.get_local_values(
                    xi, el_nodal_values, el_nodal_coords
                )
                return func(u, u_grad, *_additional_values_at_quad)

            return self._vmap_over_elements_and_quads(nodal_values, _interpolate_quad)

        return _interpolate

    def at_quad(
        self,
        nodal_values: jax.Array,
        *_additional_values_at_quad: jax.Array,
    ) -> jax.Array:
        """Interpolates the given function at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            *_additional_values_at_quad: Additional values at the quadrature points (optional)
        """

        def _interpolate_quad(xi: jax.Array, *_) -> jax.Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(xi, nodal_values)

        return self._vmap_over_elements_and_quads(nodal_values, _interpolate_quad)

    def grad(
        self,
        nodal_values: jax.Array,
        *_additional_values_at_quad: jax.Array,
    ) -> jax.Array:
        """Computes the gradient of the nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
            *_additional_values_at_quad: Additional values at the quadrature points (optional)
        """

        def _gradient_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (gradient) on a quad point."""
            u_grad = self.element.gradient(xi, el_nodal_values, el_nodal_coords)
            return u_grad

        return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)
