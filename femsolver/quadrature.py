import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp


# --- Shape functions and quadrature ---
def quad_quad4() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadrature points and weights for a quadrilateral element.

    Returns
    -------
    quad_points : jnp.ndarray
        The quadrature points.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    weights : jnp.ndarray
        The quadrature weights.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    """
    xi = jnp.array([-1.0 / jnp.sqrt(3), 1.0 / jnp.sqrt(3)])
    w = jnp.array([1.0, 1.0])
    quad_points = jnp.stack(jnp.meshgrid(xi, xi), axis=-1).reshape(-1, 2)
    weights = jnp.kron(w, w)
    return quad_points, weights


def shape_fn_quad4(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shape functions and derivatives for a quadrilateral element.

    Parameters
    ----------
    xi : jnp.ndarray
        The local coordinates of the quadrature points.

    Returns
    -------
    N : jnp.ndarray
        The shape functions.
        The shape of the array is (nb_nodes_per_element,).
    dNdr : jnp.ndarray
        The derivatives of the shape functions with respect to the local coordinates. 
        The shape of the array is (nb_axes_in_reference_space, nb_nodes_per_element).
    """
    r, s = xi
    N = 0.25 * jnp.array(
        [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)]
    )
    dNdr = (
        0.25
        * jnp.array(
            [
                [-(1 - s), -(1 - r)],
                [(1 - s), -(1 + r)],
                [(1 + s), (1 + r)],
                [-(1 + s), (1 - r)],
            ]
        ).T
    )
    return N, dNdr


def quad_tri3() -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadrature points and weights for a triangular element.

    Returns
    -------
    quad_points : jnp.ndarray
        The quadrature points.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    weights : jnp.ndarray
        The quadrature weights.
        The shape of the array is (nb_quads, nb_axes_in reference space).
    """
    qp = jnp.array([[1 / 3, 1 / 3]])
    w = jnp.array([0.5])
    return qp, w


def shape_fn_tri3(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shape functions and derivatives for a triangular element.

    Parameters
    ----------
    xi : jnp.ndarray
        The local coordinates of the quadrature points.

    Returns
    -------
    N : jnp.ndarray
        The shape functions.
        The shape of the array is (nb_nodes_per_element,).
    dNdr : jnp.ndarray
        The derivatives of the shape functions with respect to the local coordinates.
        The shape of the array is (nb_axes_in_reference_space, nb_nodes_per_element).
    """
    xi1, xi2 = xi
    xi3 = 1.0 - xi1 - xi2
    N = jnp.array([xi3, xi1, xi2])
    dNdxi = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
    return N, dNdxi


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