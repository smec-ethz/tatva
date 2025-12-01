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

import dataclasses
import uuid
from collections.abc import Sequence
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import sparse
from jax_autovmap import autovmap

from tatva.element import Element
from tatva.mesh import Mesh, find_containing_polygons

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


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Operator(Generic[ElementT]):
    mesh: Mesh
    element: ElementT

    # These fields are populated in __init__
    det_J_elements_weights: Array = dataclasses.field(init=False)
    _nse: Optional[int] = dataclasses.field(default=None, init=False)
    _indices_T: Optional[jax.Array] = dataclasses.field(default=None, init=False)
    _id: int = dataclasses.field(init=False)

    # --- Context Configuration ---
    # We provide defaults here so the class structure is valid
    _ctx_mode: str = "global"
    _ctx_active_op_id: Optional[int] = None
    _ctx_el_coords: Optional[Array] = None
    _ctx_source_op: Optional["Operator"] = None

    def __init__(self, mesh: Mesh, element: Element):
        self.mesh = mesh
        self.element = element

        # Assign unique ID
        self._id = uuid.uuid4().int

        self._precompute_sparsity()

        def _get_det_J(xi, x):
            return self.element.get_jacobian(xi, x)[1]

        det_J_elements = self.map(_get_det_J)(self.mesh.coords)
        self.det_J_elements_weights = jnp.einsum(
            "eq,q->eq", det_J_elements, self.element.quad_weights
        )

    def tree_flatten(self):
        # dynamic children  which will be traced by JAX
        # includes arrays, sub-operators, and the mesh/element objects
        children = (
            self.mesh,
            self.element,
            self.det_J_elements_weights,
            self._indices_T,  # JAX Array
            self._ctx_el_coords,  # JAX Array (Tracer during assembly)
            self._ctx_source_op,  # Operator (Recursive PyTree)
        )

        # static metadata that will not be traced by JAX
        # includes integers, strings, and configuration flags
        aux_data = (self._nse, self._id, self._ctx_mode, self._ctx_active_op_id)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Unpack
        mesh, element, weights, indices_t, ctx_coords, ctx_source = children
        nse, uid, ctx_mode, ctx_active_id = aux_data

        # Create a blank instance to bypass __init__
        # bypass __init__ because it contains heavy pre-computation
        obj = object.__new__(cls)

        # 2. Restore State
        obj.mesh = mesh
        obj.element = element
        obj.det_J_elements_weights = weights
        obj._indices_T = indices_t
        obj._ctx_el_coords = ctx_coords
        obj._ctx_source_op = ctx_source

        obj._nse = nse
        obj._id = uid
        obj._ctx_mode = ctx_mode
        obj._ctx_active_op_id = ctx_active_id

        return obj

    def with_context(
        self, mode: str, el_coords: Array = None, active_id: int = None, source_op=None
    ):
        """Returns a shallow copy of this operator with local context baked in."""

        # creates a blank instance (Bypass __init__)
        # this is fast and safe for JAX tracing contexts
        new_op = object.__new__(self.__class__)

        # copy all existing state
        # a shallow copy of __dict__ is sufficient for JAX PyTrees
        new_op.__dict__.update(self.__dict__)

        # update the context fields
        new_op._ctx_mode = mode
        new_op._ctx_el_coords = el_coords
        new_op._ctx_active_op_id = active_id
        new_op._ctx_source_op = source_op

        return new_op

    def _precompute_sparsity(self):
        """Pre-computes the sparsity pattern (NSE and Indices) for BCOO assembly."""

        n_nodes, n_dim = self.mesh.coords.shape
        n_total_dofs = n_nodes * n_dim
        dof_indices = (self.mesh.elements * n_dim)[:, :, None] + jnp.arange(n_dim)
        dof_indices = dof_indices.reshape(self.mesh.elements.shape[0], -1)
        n_dof_el = dof_indices.shape[1]

        rows = jnp.repeat(dof_indices, n_dof_el, axis=1).flatten()
        cols = jnp.tile(dof_indices, (1, n_dof_el)).flatten()
        raw_indices = jnp.stack([rows, cols]).T

        dummy_vals = jnp.ones(raw_indices.shape[0])
        dummy_bcoo = sparse.BCOO(
            (dummy_vals, raw_indices), shape=(n_total_dofs, n_total_dofs)
        )
        structure = dummy_bcoo.sum_duplicates()

        self._nse = structure.nse
        self._indices_T = raw_indices

    def grad(self, nodal_values: jax.Array) -> jax.Array:
        """Computes the gradient of the nodal values at the quad points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the gradient of the nodal values at each quadrature point
            of each element (shape: (n_elements, n_quad_points, n_values, n_dim)).
        """

        if self._ctx_mode == "global":
            # u_vec = u.reshape(-1, self.mesh.coords.shape[1])
            def _gradient_quad(xi, el_nodal_values, el_nodal_coords):
                return self.element.gradient(xi, el_nodal_values, el_nodal_coords)

            return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)

        if self._ctx_mode == "local":
            # We use the baked-in el_coords directly! No global lookup.
            return jax.vmap(
                lambda xi: self.element.gradient(xi, nodal_values, self._ctx_el_coords)
            )(self.element.quad_points)

        # Dormant
        dim = self.mesh.coords.shape[1]
        n_q = self.element.quad_points.shape[0]
        return jnp.zeros((n_q, dim, dim))

    def integrate(self, arg: jax.Array | Numeric) -> jax.Array:
        """Integrates the given argument over the domain defined by this operator.

        Args:
            arg: A scalar, or an array of shape (n_elements, ...), or nodal values of shape (n_nodes, ...).

        Returns:
            The integrated value as a jax.Array.
        """
        if self._ctx_mode == "global":
            res = self.integrate_per_element(arg)
            return jnp.sum(res, axis=(0,))

        if self._ctx_mode == "local":
            det_J = jax.vmap(
                lambda xi: self.element.get_jacobian(xi, self._ctx_el_coords)[1]
            )(self.element.quad_points)
            return jnp.dot(arg * det_J, self.element.quad_weights)

        return jnp.array(0.0)

    def eval(self, u: jax.Array) -> jax.Array:
        """Evaluates the nodal values at the quadrature points.

        Args:
            nodal_values: The nodal values at the element's nodes (shape: (n_nodes, n_values))

        Returns:
            A `jax.Array` with the values of the nodal values at each quadrature point of
            each element (shape: (n_elements, n_quad_points, n_values)).
        """
        if self._ctx_mode == "global":

            def _eval_quad(xi, el_vals, el_coords):
                return self.element.interpolate(xi, el_vals)

            return self._vmap_over_elements_and_quads(u, _eval_quad)

        if self._ctx_mode == "local":
            return jax.vmap(lambda xi: self.element.interpolate(xi, u))(
                self.element.quad_points
            )

        if self._ctx_mode == "project":
            # Fast Path: If meshes match, just evaluate at active points
            if self.mesh is self._ctx_source_op.mesh:
                return jax.vmap(lambda xi: self.element.interpolate(xi, u))(
                    self.element.quad_points
                )

            # Slow Path: Placeholder for spatial search
            return jnp.zeros((self.element.quad_points.shape[0], 1))

        return jnp.zeros((self.element.quad_points.shape[0], 1))

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
        elif arg.shape[0] == self.mesh.elements.shape[0]:
            res = self._integrate_quad_array(arg)
        else:
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
        return jnp.einsum("eq...,eq->e...", quad_values, self.det_J_elements_weights)

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

        # Use simple map if available or lax.map
        def _at_each_element(args: Tuple[Array, Array]) -> Array:
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
            batch_size=10000,
        )

    def map(self, func: Callable, *, element_quantity: Sequence[int] = ()) -> Callable:
        """Maps a function over the elements and quad points of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements and quad points.

        Args:
            func: The function to map over the elements and quadrature points.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
        """

        def _mapped(*values, **kwargs):
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(*el_values) -> jax.Array:
                return jax.vmap(lambda xi: func(xi, *el_values, **kwargs))(
                    self.element.quad_points
                )

            return jax.vmap(_at_each_element, in_axes=(0,) * len(values))(
                *(
                    v[self.mesh.elements] if i not in element_quantity else v
                    for i, v in enumerate(_values)
                )
            )

        return _mapped

    def map_over_elements(
        self, func: Callable, *, element_quantity: Sequence[int] = ()
    ) -> Callable:
        """Maps a function over the elements of the mesh.

        Returns a function that takes values at nodal points (globally) and returns the
        vmapped result over the elements.

        Args:
            func: The function to map over the elements.
            element_quantity: Indices of the arguments of `func` that are quantities
                defined per element. The rest of the arguments are assumed to be defined
                at nodal points.
        """

        def _mapped(*values, **kwargs):
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(*el_values) -> RT:
                return func(*el_values, **kwargs)

            return jax.vmap(_at_each_element, in_axes=(0,) * len(values))(
                *(
                    v[self.mesh.elements] if i not in element_quantity else v
                    for i, v in enumerate(_values)
                )
            )

        return _mapped

    def get_element_assembler(self, energy_wrapper_fn: Callable) -> Callable:
        """Returns a JIT-compiled function that computes the Hessian contribution of THIS
        operator. Used by the API 'assemble' function.
        Args:
            energy_wrapper_fn: A function that computes the energy of a single element.

        Returns:
            A JIT-compiled function that takes the global nodal values and returns the
            global stiffness matrix contribution from this operator as a sparse BCOO matrix.
        """

        def element_kernel(el_u_flat, el_coords, el_idx):
            return energy_wrapper_fn(el_u_flat, el_coords, el_idx)

        k_e_fn = jax.hessian(element_kernel, argnums=0)
        vmapped_ke = jax.vmap(k_e_fn, in_axes=(0, 0, 0))

        @jax.jit
        def assemble_loop(u_flat):
            n_nodes, n_dim = self.mesh.coords.shape
            el_u = u_flat.reshape(n_nodes, -1)[self.mesh.elements].reshape(
                self.mesh.elements.shape[0], -1
            )
            el_x = self.mesh.coords[self.mesh.elements]
            el_indices = jnp.arange(self.mesh.elements.shape[0])

            all_ke = vmapped_ke(el_u, el_x, el_indices)
            vals = all_ke.flatten()

            mat = sparse.BCOO((vals, self._indices_T), shape=(u_flat.size, u_flat.size))
            return mat.sum_duplicates(nse=self._nse)

        return assemble_loop


'''
class Operator(Generic[ElementT], eqx.Module):
    """A class that provides an Operator for finite element method (FEM) assembly.

    Args:
        mesh: The mesh containing the elements and nodes.
        element: The element type used for the finite element method.
    """

    mesh: Mesh
    element: ElementT
    det_J_elements_weights: Array
    _id: int = eqx.field(static=True)

    # Pre-computed sparsity info (NSE) for efficient JIT assembly
    _nse: Optional[int]
    _indices_T: Optional[Array]

    # Context fields (Empty by default, filled during assembly)
    _ctx_mode: str = "global"
    _ctx_el_coords: Optional[Array] = None
    _ctx_active_op_id: Optional[int] = None
    _ctx_source_op: Optional["Operator"] = None  # For projection

    def __init__(self, mesh: Mesh, element: Element):
        self.mesh = mesh
        self.element = element
        # Assign a unique ID to this operator instance
        # We use uuid to ensure uniqueness even if memory addresses are reused
        self._id = uuid.uuid4().int

        # Precompute sparsity for assembly
        self._precompute_sparsity()

        # Precompute weights for GLOBAL integration mode
        def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
            """Calls the function element.get_jacobian and returns the second output."""
            return self.element.get_jacobian(xi, el_nodal_coords)[1]

        # Use map to compute detJ across the mesh
        det_J_elements = self.map(_get_det_J)(self.mesh.coords)
        self.det_J_elements_weights = jnp.einsum(
            "eq,q->eq", det_J_elements, self.element.quad_weights
        )

        # Perform validation checks
        self.__check_init__()

    def _precompute_sparsity(self):
        """Pre-computes the sparsity pattern (NSE and Indices) for BCOO assembly."""
        n_nodes, n_dim = self.mesh.coords.shape
        n_total_dofs = n_nodes * n_dim

        # Generate raw indices for one full assembly
        dof_indices = (self.mesh.elements * n_dim)[:, :, None] + jnp.arange(n_dim)
        dof_indices = dof_indices.reshape(self.mesh.elements.shape[0], -1)
        n_dof_el = dof_indices.shape[1]

        rows = jnp.repeat(dof_indices, n_dof_el, axis=1).flatten()
        cols = jnp.tile(dof_indices, (1, n_dof_el)).flatten()

        # Transpose to shape (NSE_raw, 2) for BCOO
        raw_indices = jnp.stack([rows, cols]).T

        # Compute exact NSE (Number of Specified Elements) after summing duplicates
        # We do this concretely (no JIT) once during init.
        dummy_vals = jnp.ones(raw_indices.shape[0])
        # Use a temporary unjitted call to avoid tracing issues
        dummy_bcoo = sparse.BCOO(
            (dummy_vals, raw_indices), shape=(n_total_dofs, n_total_dofs)
        )
        structure = dummy_bcoo.sum_duplicates()

        # Store these for the assembler to use
        # object.__setattr__(self, "_nse", structure.nse)
        # object.__setattr__(self, "_indices_T", raw_indices)
        self._nse = structure.nse
        self._indices_T = raw_indices

    def __check_init__(self) -> None:
        """Validates the mesh and element compatibility."""
        coords = self.mesh.coords
        elements = self.mesh.elements
        quad_points = self.element.quad_points

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

        if quad_points.ndim != 2 or quad_points.shape[0] == 0:
            raise ValueError(
                "Element must define at least one quadrature point in an (n_q, n_dim) array."
            )

        local_dim = quad_points.shape[1]
        global_dim = coords.shape[1]
        if local_dim > 1 and local_dim != global_dim:
            raise ValueError(
                f"Element {self.element.__class__.__name__} expects {local_dim}D coordinates but mesh provides {global_dim}D nodes."
            )

        n_nodes_per_element = elements.shape[1]
        # Check shape function on first quad point
        shape_fn = np.asarray(self.element.shape_function(self.element.quad_points[0]))
        if shape_fn.shape[0] != n_nodes_per_element:
            raise ValueError(
                f"Mesh connectivity lists {n_nodes_per_element} nodes per element but {self.element.__class__.__name__} expects {shape_fn.shape[0]}."
            )

    def _vmap_over_elements_and_quads(
        self, nodal_values: jax.Array, func: MappableOverElementsAndQuads
    ) -> jax.Array:
        """Helper function for Global Mode. Maps a function over the elements using jax.lax.map for memory efficiency."""

        def _at_each_element(args: Tuple[Array, Array]) -> Array:
            el_nodal_values, el_nodal_coords = args
            return eqx.filter_vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return jax.lax.map(
            _at_each_element,
            xs=(nodal_values[self.mesh.elements], self.mesh.coords[self.mesh.elements]),
            batch_size=20000,
        )

    def map(
        self,
        func: MappableOverElementsAndQuads[P, RT],
        *,
        element_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements and quad points of the mesh (Global Mode)."""

        def _mapped(*values: P.args, **kwargs: P.kwargs) -> RT:
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

    def map_over_elements(
        self,
        func: MappableOverElements[P, RT],
        *,
        element_quantity: Sequence[int] = (),
    ) -> MappedCallable[P, RT]:
        """Maps a function over the elements of the mesh (Global Mode)."""

        def _mapped(*values: P.args, **kwargs: P.kwargs) -> RT:
            _values = cast(tuple[jax.Array, ...], values)

            def _at_each_element(*el_values) -> RT:
                return func(*el_values, **kwargs)

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

    def with_context(
        self,
        mode: str,
        el_coords: Array,
        active_id: int,
        source_op=None,
    ):
        """Returns a copy of this operator with local context baked in."""
        # This uses Equinox to return a new PyTree with updated fields
        temp = eqx.tree_at(lambda o: o._ctx_mode, self, mode)
        temp = eqx.tree_at(lambda o: o._ctx_el_coords, temp, el_coords)
        temp = eqx.tree_at(lambda o: o._ctx_active_op_id, temp, active_id)
        temp = eqx.tree_at(lambda o: o._ctx_source_op, temp, source_op)
        return temp

    def grad(self, nodal_values: jax.Array) -> jax.Array:
        if self._ctx_mode == "global":
            # u_vec = u.reshape(-1, self.mesh.coords.shape[1])
            def _gradient_quad(xi, el_nodal_values, el_nodal_coords):
                return self.element.gradient(xi, el_nodal_values, el_nodal_coords)

            return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)

        if self._ctx_mode == "local":
            # We use the baked-in el_coords directly! No global lookup.
            return jax.vmap(
                lambda xi: self.element.gradient(xi, nodal_values, self._ctx_el_coords)
            )(self.element.quad_points)

        # Dormant
        dim = self.mesh.coords.shape[1]
        n_q = self.element.quad_points.shape[0]
        return jnp.zeros((n_q, dim, dim))

    def integrate(self, arg: jax.Array | Numeric) -> jax.Array:
        if self._ctx_mode == "global":
            res = self.integrate_per_element(arg)
            return jnp.sum(res, axis=(0,))

        if self._ctx_mode == "local":
            det_J = jax.vmap(
                lambda xi: self.element.get_jacobian(xi, self._ctx_el_coords)[1]
            )(self.element.quad_points)
            return jnp.dot(arg * det_J, self.element.quad_weights)

        return 0.0

    def eval(self, u: jax.Array) -> jax.Array:
        if self._ctx_mode == "global":

            def _eval_quad(xi, el_vals, el_coords):
                return self.element.interpolate(xi, el_vals)

            return self._vmap_over_elements_and_quads(u, _eval_quad)

        if self._ctx_mode == "local":
            return jax.vmap(lambda xi: self.element.interpolate(xi, u))(
                self.element.quad_points
            )

        if self._ctx_mode == "project":
            # Projection Logic using stored source operator
            source = self._ctx_source_op
            # ... (Projection logic using source._ctx_el_coords etc) ...
            # For now, fast path placeholder:
            return jax.vmap(lambda xi: self.element.interpolate(xi, u))(
                self.element.quad_points
            )

        return jnp.zeros((self.element.quad_points.shape[0], 1))

    def integrate_per_element(self, arg: jax.Array | Numeric) -> jax.Array:
        """Helper for global integration per element."""
        if isinstance(arg, Numeric):
            res = self._integrate_quad_array(self.eval(jnp.array([arg])))
        elif arg.shape[0] == self.mesh.elements.shape[0]:  # element field
            res = self._integrate_quad_array(arg)
        else:  # nodal field
            field_at_quads = self.eval(arg)
            res = self._integrate_quad_array(field_at_quads)
        return res

    def _integrate_quad_array(self, quad_values: jax.Array) -> jax.Array:
        return jnp.einsum("eq...,eq->e...", quad_values, self.det_J_elements_weights)

    def interpolate(self, arg: jax.Array, points: jax.Array) -> jax.Array:
        """Inverse Mapping: Interpolates nodal values to physical points.
        Used for global interpolation and potentially for the 'Slow Path' projection."""

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

        def map_physical_to_reference(points: jax.Array) -> tuple[jax.Array, jax.Array]:
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

        return self._interpolate_direct(arg, valid_quad_points, valid_elements)

    def _interpolate_direct(
        self,
        nodal_values: jax.Array,
        valid_quad_points: jax.Array,
        valid_elements: jax.Array,
    ) -> jax.Array:
        """Interpolates the given nodal values at the quad points."""

        def _interpolate_quad(
            xi: jax.Array, el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            return self.element.interpolate(xi, el_nodal_values)

        return eqx.filter_vmap(
            _interpolate_quad,
            in_axes=(0, 0, 0),
        )(
            valid_quad_points,
            nodal_values[valid_elements],
            self.mesh.coords[valid_elements],
        )

    def get_element_assembler(self, energy_wrapper_fn: Callable) -> Callable:
        """
        Returns a JIT-compiled function that computes the Hessian contribution of THIS operator.
        Used by the API 'assemble' function.
        """

        # 1. Define the kernel for one element
        # This kernel expects: (el_u_flat, el_coords, el_idx)
        def element_kernel(el_u_flat, el_coords, el_idx):
            return energy_wrapper_fn(el_u_flat, el_coords, el_idx)

        # 2. Differentiate (Hessian of scalar element energy)
        k_e_fn = jax.hessian(element_kernel, argnums=0)

        # 3. Vectorize over the mesh
        vmapped_ke = jax.vmap(k_e_fn, in_axes=(0, 0, 0))

        # 4. The JIT-compiled Assembly Loop
        @jax.jit
        def assemble_loop(u_flat):
            n_nodes, n_dim = self.mesh.coords.shape

            # Gather
            el_u = u_flat.reshape(n_nodes, -1)[self.mesh.elements].reshape(
                self.mesh.elements.shape[0], -1
            )
            el_x = self.mesh.coords[self.mesh.elements]
            el_indices = jnp.arange(self.mesh.elements.shape[0])

            # Compute Ke
            all_ke = vmapped_ke(el_u, el_x, el_indices)
            vals = all_ke.flatten()

            jax.debug.print("vals={vals}", vals=vals)

            # Create Sparse Matrix using PRE-COMPUTED sparsity
            # self._indices_T and self._nse are captured as constants
            mat = sparse.BCOO((vals, self._indices_T), shape=(u_flat.size, u_flat.size))

            jax.debug.print("mat {mat}", mat=mat.todense())

            # Sum duplicates using PRE-COMPUTED NSE
            return mat.sum_duplicates(nse=self._nse)

        return assemble_loop
'''

'''
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import (
    Callable,
    Generic,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    Tuple,
    Optional,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_autovmap import autovmap
from jax.experimental import sparse

from tatva.element import Element
from tatva.mesh import Mesh, find_containing_polygons

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


class Operator(Generic[ElementT], eqx.Module):
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
    element: ElementT
    det_J_elements_weights: Array

    def __init__(self, mesh: Mesh, element: Element):
        self.mesh = mesh
        self.element = element

        def _get_det_J(xi: jax.Array, el_nodal_coords: jax.Array) -> jax.Array:
            """Calls the function element.get_jacobian and returns the second output."""
            return self.element.get_jacobian(xi, el_nodal_coords)[1]

        det_J_elements = self.map(_get_det_J)(self.mesh.coords)
        self.det_J_elements_weights = jnp.einsum(
            "eq,q->eq", det_J_elements, self.element.quad_weights
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
        quad_points = self.element.quad_points

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

        if quad_points.ndim != 2 or quad_points.shape[0] == 0:
            raise ValueError(
                "Element must define at least one quadrature point in an (n_q, n_dim) array."
            )

        local_dim = quad_points.shape[1]
        global_dim = coords.shape[1]
        if local_dim > 1 and local_dim != global_dim:
            raise ValueError(
                f"Element {self.element.__class__.__name__} expects {local_dim}D coordinates but mesh provides {global_dim}D nodes."
            )
        if local_dim == 0:
            raise ValueError("Element must have a positive number of local dimensions.")

        n_nodes_per_element = elements.shape[1]
        shape_fn = np.asarray(self.element.shape_function(self.element.quad_points[0]))
        if shape_fn.ndim != 1:
            raise ValueError(
                "Element shape function must return a 1D array of nodal weights."
            )
        if shape_fn.shape[0] != n_nodes_per_element:
            raise ValueError(
                f"Mesh connectivity lists {n_nodes_per_element} nodes per element but {self.element.__class__.__name__} expects {shape_fn.shape[0]}."
            )

        flat_elements = elements.ravel()
        if flat_elements.min() < 0:
            raise ValueError(
                "Mesh element connectivity contains negative node indices."
            )
        if flat_elements.max() >= coords.shape[0]:
            raise ValueError(
                "Mesh element connectivity references nodes outside the mesh coordinates array."
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

        """def _at_each_element(
            el_nodal_values: jax.Array, el_nodal_coords: jax.Array
        ) -> jax.Array:
            return eqx.filter_vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return eqx.filter_vmap(
            _at_each_element,
            in_axes=(0, 0),
        )(nodal_values[self.mesh.elements], self.mesh.coords[self.mesh.elements])"""

        def _at_each_element(args: Tuple[Array, Array]) -> Array:
            el_nodal_values, el_nodal_coords = args

            return eqx.filter_vmap(
                partial(
                    func,
                    el_nodal_values=el_nodal_values,
                    el_nodal_coords=el_nodal_coords,
                )
            )(self.element.quad_points)

        return jax.lax.map(
            _at_each_element,
            xs=(nodal_values[self.mesh.elements], self.mesh.coords[self.mesh.elements]),
            batch_size=1000,
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

            def _at_each_element(*el_values) -> RT:
                return func(*el_values, **kwargs)

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

    def _integrate_quad_array(self, quad_values: jax.Array) -> jax.Array:
        """Integrates a given array of values at quadrature points over the mesh.

        Args:
            quad_values: The values at the quadrature points
                (shape: (n_elements, n_quad_points, n_values))

        Returns:
            A `jax.Array` where each element contains the integral of the values in the
            element (shape: (n_elements, n_values)).
        """

        return jnp.einsum("eq...,eq->e...", quad_values, self.det_J_elements_weights)

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
            return self.element.interpolate(xi, el_nodal_values)

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

        def map_physical_to_reference(points: jax.Array) -> tuple[jax.Array, jax.Array]:
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

        return self._interpolate_direct(arg, valid_quad_points, valid_elements)

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
            return self.element.interpolate(xi, el_nodal_values)

        return eqx.filter_vmap(
            _interpolate_quad,
            in_axes=(0, 0, 0),
        )(
            valid_quad_points,
            nodal_values[valid_elements],
            self.mesh.coords[valid_elements],
        )
'''
