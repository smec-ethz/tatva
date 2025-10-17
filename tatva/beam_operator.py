# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
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
from functools import partial
from typing import Callable, Generic, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from tatva.mesh import Mesh
from tatva.element import BeamElement


P = ParamSpec("P")
RT = TypeVar("RT", bound=jax.Array | tuple, covariant=True)
BeamElementT = TypeVar("BeamElementT", bound=BeamElement)
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


class BeamOperator(Generic[BeamElementT], eqx.Module):
    """A class that provides an Operator for finite element method (FEM) assembly for beam elements.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The beam element type used for the finite element method.
       
    Provides several operators for evaluating and integrating functions over the mesh,
    such as `integrate`, `eval`, and `grad`. These operators can be used to compute
    integrals, evaluate functions at quadrature points, and compute gradients of
    functions at quadrature points.

    Example:
        >>> from tatva import Mesh, LinearElement, BeamOperator
        >>> element = LinearElement()  # Define an element type
        >>> operator = BeamOperator(mesh, element)
        >>> nodal_values = jnp.array(...)  # Nodal values at the mesh nodes
        >>> energy = operator.integrate(energy_density)(nodal_values)
    """

    mesh: Mesh
    element: BeamElementT
    det_J_elements_weights: Array

    def __init__(self, mesh: Mesh, element: BeamElement):
        self.mesh = mesh
        self.element = element

        def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
            """Calls the function element.get_jacobian and returns the second output."""
            return self.element.get_jacobian(xi, el_nodal_coords)[1]

        det_J_elements = self.map(_get_det_J)(self.mesh.coords)
        self.det_J_elements_weights = jnp.einsum(
            "eq,q->eq", det_J_elements, self.element.quad_weights
        )

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

        def _at_each_element(el_nodal_values: Array, el_nodal_coords: Array) -> Array:
            return eqx.filter_vmap(
                eqx.Partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return eqx.filter_vmap(
            _at_each_element,
            in_axes=(0, 0),
        )(nodal_values[self.mesh.elements], self.mesh.coords[self.mesh.elements])



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

            def _at_each_element(*el_values) -> jax.Array:
                return eqx.filter_vmap(
                    lambda xi: func(xi, *el_values, **kwargs),
                )(self.element.quad_points)

            return eqx.filter_vmap(
                _at_each_element,
                in_axes=(0,) * len(values),
            )(
                *(
                    v[self.mesh.elements] if i not in element_quantity else v
                    for i, v in enumerate(_values)
                )
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


    def _integrate_quad_array(self, quad_values: Array) -> Array:
        """Integrates a given array of values at quadrature points over the mesh.

        Args:
            quad_values: The values at the quadrature points (shape: (n_elements, n_quad_points, n_values))

        Returns:
            A jax.Array where each element contains the integral of the values in the element
        """
        #det_J_elements = self._vmap_over_elements_and_quads(
        #    jnp.zeros(1),  # Dummy nodal values
        #    lambda xi, el_nodal_values, el_nodal_coords: self.element.get_jacobian(
        #        xi, el_nodal_coords
        #    )[1],
        #)
        return jnp.einsum(
            "eq...,eq->e...",
            quad_values,
            self.det_J_elements_weights,
        )


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
            return self.element.interpolate(xi, el_nodal_values, el_nodal_coords)

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