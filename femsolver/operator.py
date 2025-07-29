import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from femsolver.quadrature import Element


class Operator(eqx.Module):
    """
    A class used to represent a Operator for finite element method (FEM) simulations.

    Attributes:
        element: The element to use for the integration.
        integrand: The integrand to use for the integration.

    Methods:
        interpolate: Interpolates the nodal values at the given points.
        integrate: Integrates the integrand over the cells.
    """

    element: eqx.AbstractVar[Element]

    @eqx.filter_vmap(in_axes=(None, 0))
    def interpolate(
        self,
        nodal_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        qp, w = self.element.get_quadrature()

        def _interpolate(xi):
            return self.element.interpolate(xi, nodal_values)

        return jax.vmap(_interpolate)(qp)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def gradient(
        self,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the gradient of the nodal values at the given points.
        TODO: add the option to provide a function instead of nodal values
        """
        qp, w = self.element.get_quadrature()

        def _gradient(xi):
            u_grad = self.element.gradient(xi, nodal_values, nodes)
            return u_grad

        return jax.vmap(_gradient)(qp)

    @eqx.filter_vmap(
        in_axes=(
            None,
            0,
            0,
        )
    )
    def integrate(
        self,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Integrates the energy over the cells.
        """
        xi, w = self.element.get_quadrature()
        return jnp.sum(self.integrand(xi, w, nodal_values, nodes))

    @abc.abstractmethod
    def integrand(self, xi, wi, nodal_values, nodes):
        raise NotImplementedError

