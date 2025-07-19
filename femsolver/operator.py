import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import equinox as eqx
from typing import Callable
from femsolver.jax_utils import auto_vmap, vmap
from femsolver.quadrature import Element


def gather_fields(
    u_flat: jnp.ndarray, connectivity: jnp.ndarray, dof_per_node: int
) -> jnp.ndarray:
    """
    Gathers the fields from the flat array to the full array.
    """
    u_full = u_flat.reshape(-1, dof_per_node)
    return u_full[connectivity]


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

    element: Element
    integrand: Callable

    def __init__(self, element: Element, integrand: Callable):
        self.element = element
        self.integrand = integrand

    @auto_vmap(x=1, nodal_values=None)
    def interpolate(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        N, dNdr = self.element.get_shape_functions(x)
        return N @ nodal_values

    # --- Element-level energy ---
    @vmap(in_axes=(None, 0, 0))
    def integrate(
        self,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Integrates the energy over the cells.
        """
        qp, w = self.element.get_quadrature()

        def integrand(xi, wi):
            N, dNdr = self.element.get_shape_functions(xi)
            J = dNdr @ nodes
            u_quad = self.interpolate(xi, nodal_values)
            value = self.integrand(u_quad)
            return wi * value * jnp.linalg.det(J)

        return jnp.sum(jax.vmap(integrand)(qp, w))


class FemOperator(eqx.Module):
    """
    A class used to represent a FemOperator for finite element method (FEM) simulations.

    Attributes
    ----------
    funcs : dict[str, Callable]
        A dictionary of functions used to compute the quadrature points, shape functions, and energy.

    Methods
    -------
    interpolate(x, nodal_values):
        Interpolates the nodal values at the given points.
    interpolate_cells(x, nodal_values):
        Interpolates the nodal values over the cells.
    gradient(x, nodal_values, nodes):
        Computes the gradient of the nodal values at the given points.
    gradient_over_cells(x, nodal_values, nodes):
        Computes the gradient of the nodal values over the cells.
    integrate(nodal_values, nodes):
        Integrates the energy over the cells.
    """

    funcs: dict[str, Callable]

    def __init__(
        self,
        compute_quads: Callable,
        compute_shape_fn: Callable,
        compute_energy: Callable,
    ):
        self.funcs = {
            "quads": compute_quads,
            "shape_fn": compute_shape_fn,
            "energy": compute_energy,
        }

    @auto_vmap(x=1, nodal_values=None)
    def interpolate(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        N, dNdr = self.funcs["shape_fn"](x)
        return N @ nodal_values

    @auto_vmap(x=None, nodal_values=2)
    def interpolate_cells(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values over the cells.
        """
        return self.interpolate(x, nodal_values)

    @auto_vmap(x=1, nodal_values=None, nodes=None)
    def gradient(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the gradient of the nodal values at the given points.
        """
        N, dNdr = self.funcs["shape_fn"](x)
        J = dNdr @ nodes
        dNdX = jnp.linalg.inv(J) @ dNdr
        return dNdX @ nodal_values

    @auto_vmap(x=None, nodal_values=2, nodes=2)
    def gradient_over_cells(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the gradient of the nodal values over the cells.
        """
        return self.gradient(x, nodal_values, nodes)

    # --- Element-level energy ---
    @vmap(in_axes=(None, 0, 0))
    def integrate(
        self,
        nodal_values: jnp.ndarray,
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Integrates the energy over the cells.
        """
        qp, w = self.funcs["quads"]()

        def integrand(xi, wi):
            N, dNdr = self.funcs["shape_fn"](xi)
            J = dNdr @ nodes
            u_grad = self.gradient(xi, nodal_values, nodes)
            energy = self.funcs["energy"](u_grad)
            return wi * energy * jnp.linalg.det(J)

        return jnp.sum(jax.vmap(integrand)(qp, w))
