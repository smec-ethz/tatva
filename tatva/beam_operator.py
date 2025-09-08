import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from tatva.mesh import Mesh
from tatva.element import BeamElement
import equinox as eqx
from typing import Callable, Protocol, overload, ParamSpec, Concatenate, TypeAlias, Any
import abc


P = ParamSpec("P")
Numeric: TypeAlias = float | int | jnp.number
Form: TypeAlias = Callable[Concatenate[Array, Array, P], Array | float]


class FormCallable(Protocol[P]):
    @staticmethod
    def __call__(
        nodal_values: Array,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Array: ...


class _VmapOverElementsCallable(Protocol):
    """Internal protocol for functions that are mapped over elements and quadrature points."""

    @staticmethod
    def __call__(
        xi: Array,
        el_nodal_values: Array,
        el_nodes_coords: Array,
    ) -> Array | Float: ...


class BeamOperator(eqx.Module):
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
    element: BeamElement

    def _vmap_over_elements_and_quads(
        self, nodal_values: jax.Array, func: _VmapOverElementsCallable
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
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            """Calls the function (gradient) on a quad point."""
            u_grad = self.element.gradient(xi, el_nodal_values, el_nodal_coords)
            return u_grad

        return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)

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
    def integrate(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def integrate(self, arg: Array | Numeric) -> Array: ...
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

        if isinstance(arg, Numeric):
            res = self._integrate_nodal_array(jnp.array([arg]))
        elif arg.shape[0] == self.mesh.elements.shape[0]:  # element field
            res = self._integrate_quad_array(arg)
        else:  # nodal field
            res = self._integrate_nodal_array(arg)

        return jnp.sum(res, axis=(0,))  # Sum over elements and quadrature points

    @overload
    def integrate_per_element(self, arg: Form[P]) -> FormCallable[P]: ...
    @overload
    def integrate_per_element(self, arg: Array) -> Array: ...
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
            ) -> jax.Array:
                """Calls the function (integrand) on a quad point. Multiplying by the
                determinant of the Jacobian.
                """
                u, w, theta, du_dx, dw_dx, d2w_dx2, detJ = (
                    self.element.get_local_values(xi, el_nodal_values, el_nodal_coords)
                )
                return func(u, w, theta, du_dx, dw_dx, d2w_dx2, *args, **kwargs) * detJ

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

    def _integrate_quad_array(self, quad_values: Array) -> Array:
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

    def _integrate_nodal_array(self, nodal_values: Array) -> Array:
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
    def eval(self, arg: Array, *args: tuple[Any, ...]) -> Array: ...
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
                xi: Array, el_nodal_values: Array, el_nodal_coords: Array
            ) -> Array | float:
                """Calls the function (interpolator) on a quad point."""
                u, w, theta, du_dx, dw_dx, d2w_dx2, _detJ = (
                    self.element.get_local_values(xi, el_nodal_values, el_nodal_coords)
                )
                return func(u, w, theta, du_dx, dw_dx, d2w_dx2, *args, **kwargs)

            return self._vmap_over_elements_and_quads(nodal_values, _eval_quad)

        return _eval

    def _eval_direct(
        self,
        nodal_values: Array,
    ) -> Array:
        """Interpolates the given function at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))
        """

        def _eval_quad(
            xi: Array, el_nodal_values: Array, el_nodal_coords: Array
        ) -> Array:
            """Calls the function (interpolator) on a quad point."""
            return self.element.interpolate(xi, el_nodal_values, el_nodal_coords)

        return self._vmap_over_elements_and_quads(nodal_values, _eval_quad)


class Operator(eqx.Module):
    element: eqx.AbstractVar[BeamElement]

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def axial_gradient(
        self, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.element.compute_length(nodes)
        local_dofs = self.element.compute_dofs_local_frame(nodal_values, nodes)

        xi, w = self.element.get_quadrature(nodes)
        return jax.vmap(self.element.dudx, in_axes=(0, None, None))(xi, local_dofs, L)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def transverse_gradient(
        self, nodal_values: jnp.ndarray, nodes: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.element.compute_length(nodes)
        local_dofs = self.element.compute_dofs_local_frame(nodal_values, nodes)

        xi, w = self.element.get_quadrature(nodes)
        return jax.vmap(self.element.dwdx, in_axes=(0, None, None))(xi, local_dofs, L)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def curvature(self, nodal_values: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.element.compute_length(nodes)
        local_dofs = self.element.compute_dofs_local_frame(nodal_values, nodes)

        xi, w = self.element.get_quadrature(nodes)
        return jax.vmap(self.element.d2wdx2, in_axes=(0, None, None))(xi, local_dofs, L)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def rotation(self, nodal_values: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolates the nodal values at the given points.
        """
        L = self.element.compute_length(nodes)
        local_dofs = self.element.compute_dofs_local_frame(nodal_values, nodes)

        xi, w = self.element.get_quadrature(nodes)
        return jax.vmap(self.element.v, in_axes=(0, None, None))(xi, local_dofs, L)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def integrate(self, nodal_values: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
        """
        Integrates the energy over the cells.
        """
        xi, w = self.element.get_quadrature(nodes)
        return jnp.sum(self.integrand(xi, w, nodal_values, nodes))

    @abc.abstractmethod
    def integrand(self, xi, wi, nodal_values, nodes):
        raise NotImplementedError


'''
class CorotationalBeamOperator(eqx.Module):
    NB_DOFS_PER_NODE: int
    funcs: dict[str, Callable]
    E: float
    A: float
    I: float
    qi: jnp.ndarray
    wi: jnp.ndarray
    omega: float

    def __init__(
        self,
        E: float = 0,
        A: float = 0,
        I: float = 0,
        quadrature: Quadrature = None,
        energy: Callable = None,
        omega: float = 0,
    ):
        """
        Initialize the beam operator.

        Args:
            E: The Young's modulus.
            A: The cross-sectional area.
            I: The moment of inertia.
            quads: The quadrature rule.
            energy: The energy function.
            omega: 0 for Euler-Bernoulli beam and 1 for Timoshenko beam.
        """

        self.NB_DOFS_PER_NODE = 3
        self.funcs = {"energy": energy} if energy is not None else {}
        self.E = E
        self.A = A
        self.I = I
        self.qi, self.wi = quadrature() if quadrature is not None else (None, None)
        self.omega = omega

    @jax_utils.vmap(in_axes=(None, 0, 0))
    def __call__(self, dofs: jnp.ndarray, nodes: jnp.ndarray) -> float:
        """
        Compute the energy of the beam.

        Args:
            dofs: The dofs of the beam per cell.
            nodes: The nodes of the beam per cell.

        Returns:
            The energy of the beam.
        """

        local_dofs = self.compute_dofs_local_frame(dofs, nodes)
        L = self.compute_length(nodes)
        detJ = L * 0.5

        x_mapped = L * (self.qi + 1) * 0.5

        def integrand(x, wi):
            # us, tl1, tl2 = local_dofs
            # axial_strain = eb.dudx(x, us, L)
            # bending_strain = eb.d2wdx2(x, tl1, tl2, L, self.omega)

            u_quad, w_quad, axial_strain, dwdx_val, curvature = eb.compute_quad_values(
                x, local_dofs, L, self.omega
            )

            energy = self.funcs["energy"](
                axial_strain, curvature, self.E, self.A, self.I
            )
            return wi * energy * detJ

        return jnp.sum(jax.vmap(integrand)(x_mapped, self.wi))

    @jax_utils.auto_vmap(coords=2)
    def compute_length(self, coords: jnp.ndarray) -> float:
        """
        Compute the length of the beam.

        Args:
            coords: The coordinates of the beam per cell.

        Returns:
            The length of the beam per cell.
        """

        return jnp.linalg.norm(coords[1] - coords[0])

    @eqx.filter_jit
    def compute_dofs_local_frame(
        self, dofs: jnp.ndarray, coords: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the dofs of the beam in the local frame.

        Args:
            dofs: The dofs of the beam per cell.
            coords: The coordinates of the beam per cell.

        Returns:
            The dofs of the beam in the local frame.
        """

        current_coords = coords + dofs[:, :2]
        l0 = self.compute_length(coords)
        ln = self.compute_length(current_coords)

        u_bar = (ln**2 - l0**2) / (ln + l0)
        beta0 = jnp.arctan2(
            (coords[1][1] - coords[0][1]), (coords[1][0] - coords[0][0])
        )

        beta = jnp.arctan2(
            (current_coords[1][1] - current_coords[0][1]),
            (current_coords[1][0] - current_coords[0][0]),
        )

        beta1 = dofs[0][self.NB_DOFS_PER_NODE - 1] + beta0
        beta2 = dofs[1][self.NB_DOFS_PER_NODE - 1] + beta0

        theta1_bar = jnp.arctan2(
            (jnp.cos(beta) * jnp.sin(beta1) - jnp.sin(beta) * jnp.cos(beta1)),
            (jnp.cos(beta) * jnp.cos(beta1) + jnp.sin(beta) * jnp.sin(beta1)),
        )

        theta2_bar = jnp.arctan2(
            (jnp.cos(beta) * jnp.sin(beta2) - jnp.sin(beta) * jnp.cos(beta2)),
            (jnp.cos(beta) * jnp.cos(beta2) + jnp.sin(beta) * jnp.sin(beta2)),
        )

        return jnp.array([u_bar, theta1_bar, theta2_bar])


class BeamOperator(eqx.Module):
    NB_DOFS_PER_NODE: int
    funcs: dict[str, Callable]
    E: float
    A: float
    I: float
    qi: jnp.ndarray
    wi: jnp.ndarray

    def __init__(
        self,
        E: float = 0,
        A: float = 0,
        I: float = 0,
        quadrature: Quadrature = None,
        energy: Callable = None,
    ):
        """
        Initialize the beam operator.

        Args:
            E: The Young's modulus.
            A: The cross-sectional area.
            I: The moment of inertia.
            quadrature: The quadrature rule.
            energy: The energy function.
        """

        self.NB_DOFS_PER_NODE = 3
        self.funcs = {"energy": energy} if energy is not None else {}
        self.E = E
        self.A = A
        self.I = I

        self.qi, self.wi = quadrature() if quadrature is not None else (None, None)

    @jax_utils.vmap(in_axes=(None, 0, 0))
    def __call__(self, dofs: jnp.ndarray, nodes: jnp.ndarray) -> float:
        """
        Compute the energy of the beam.
        """

        local_dofs = self.compute_dofs_local_frame(dofs, nodes)
        L = self.compute_length(nodes)
        detJ = L * 0.5

        def integrand(xi, wi):
            # u1, w1, theta1, u2, w2, theta2 = local_dofs
            # axial_strain = leb.dudx(xi, u1, u2, L)
            # bending_strain = leb.d2wdx2(xi, w1, theta1, w2, theta2, L)

            u_quad, w_quad, axial_strain, dw_dx, curvature = leb.compute_quad_values(
                xi, local_dofs, L
            )

            energy = self.funcs["energy"](
                axial_strain, curvature, self.E, self.A, self.I
            )
            return wi * energy * detJ

        return jnp.sum(jax.vmap(integrand)(self.qi, self.wi))

    @jax_utils.auto_vmap(coords=2)
    def compute_length(self, coords: jnp.ndarray) -> float:
        """
        Compute the length of the beam.
        """
        return jnp.linalg.norm(coords[1] - coords[0])

    @eqx.filter_jit
    def compute_dofs_local_frame(
        self, dofs: jnp.ndarray, coords: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the dofs of the beam in the local frame.
        """
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        L = jnp.sqrt(dx**2 + dy**2)

        cos_theta = dx / L
        sin_theta = dy / L

        # Transformation matrix for each node
        T_node = jnp.array(
            [[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        # Transform DOFs for both nodes
        dofs_local_1 = T_node @ dofs[0]
        dofs_local_2 = T_node @ dofs[1]

        # Return as [u1, w1, theta1, u2, w2, theta2]
        return jnp.array(
            [
                dofs_local_1[0],
                dofs_local_1[1],
                dofs_local_1[2],
                dofs_local_2[0],
                dofs_local_2[1],
                dofs_local_2[2],
            ]
        )
'''
