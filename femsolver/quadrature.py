import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import equinox as eqx
from femsolver.jax_utils import auto_vmap


class ShapeFunctions(eqx.Module):
    N: jax.Array
    dNdξ: jax.Array
    d2Ndξ2: jax.Array
    wts: jax.Array
    quad_pts: jax.Array

    def __init__(self, nb_quads, nb_nodes_per_element):
        self.N = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.dNdξ = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.d2Ndξ2 = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.wts = jnp.zeros(nb_quads)
        self.quad_pts = jnp.zeros(nb_quads)

    def interpolate(self,x, dofs):
        return self.N(x) @ dofs
    
    def gradient(self, x, dofs):
        return self.dNdξ(x) @ dofs

    def hessian(self, x, dofs): 
        return self.d2Ndξ2(x) @ dofs



@jax.tree_util.register_pytree_node_class
class Basis:
    """
    A class used to represent a Basis for finite element method (FEM) simulations.

    Attributes
    ----------
    N : jnp.ndarray
        Shape functions evaluated at quadrature points, with shape (nb_quads, nb_nodes_per_element).
    dNdξ : jnp.ndarray
        Derivatives of shape functions with respect to the reference coordinates, with shape (nb_quads, nb_nodes_per_element).
    wts : jnp.ndarray
        Quadrature weights, with shape (nb_quads).

    Methods
    -------
    tree_flatten():
        Flattens the Basis object into a tuple of children and auxiliary data for JAX transformations.
    tree_unflatten(aux_data, children):
        Reconstructs the Basis object from flattened children and auxiliary data.
    """

    def __init__(self, nb_quads, nb_nodes_per_element):
        self.N = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.dNdξ = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.d2Ndξ2 = jnp.zeros((nb_quads, nb_nodes_per_element))
        self.wts = jnp.zeros(nb_quads)
        self.quad_pts = jnp.zeros(nb_quads)

    def tree_flatten(self):
        return ((self.N, self.dNdξ, self.wts, self.quad_pts), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        N, dNdξ, ω, quad_pts = children
        instance = cls(N.shape[0], N.shape[1])
        instance.N = N
        instance.dNdξ = dNdξ
        instance.wts = ω
        instance.quad_pts = quad_pts
        return instance


@auto_vmap(x=0)
def interpolate(x, dofs, basis):
    return jnp.einsum("ij,j...->i..", basis.N, dofs, optimize="optimal")


@auto_vmap(x=0)
def gradient(x, dofs, basis):
    return jnp.einsum("ij,j...->i..", basis.dNdξ, dofs, optimize="optimal")


@auto_vmap(x=0)
def hessian(x, dofs, basis):
    return jnp.einsum("ij,j...->i..", basis.d2Ndξ2, dofs, optimize="optimal")
