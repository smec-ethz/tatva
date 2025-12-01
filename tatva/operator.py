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

import threading
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

import equinox as eqx
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


# --- Context Management (Thread Local) ---
_local_state = threading.local()


def get_context():
    if not hasattr(_local_state, "stack"):
        _local_state.stack = []
    return _local_state.stack[-1] if _local_state.stack else None


class AssemblyContext:
    """Holds the state for the current assembly operation."""

    def __init__(self, active_op: "Operator", el_coords: jax.Array, el_idx: jax.Array):
        self.active_op = active_op
        self.el_coords = el_coords
        self.el_idx = el_idx  # Needed for fast-path projection
        self.active_quad_points_phys = None  # Lazy cache


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
    """

    mesh: Mesh
    element: ElementT
    det_J_elements_weights: Array

    # Pre-computed sparsity info (NSE) for efficient JIT assembly
    _nse: Optional[int]
    _indices_T: Optional[Array]

    def __init__(self, mesh: Mesh, element: Element):
        self.mesh = mesh
        self.element = element

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
        dummy_vals = jnp.zeros(raw_indices.shape[0])
        # Use a temporary unjitted call to avoid tracing issues
        dummy_bcoo = sparse.BCOO(
            (dummy_vals, raw_indices), shape=(n_total_dofs, n_total_dofs)
        )
        structure = dummy_bcoo.sum_duplicates()

        # Store these for the assembler to use
        object.__setattr__(self, "_nse", structure.nse)
        object.__setattr__(self, "_indices_T", raw_indices)

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

    def integrate(self, arg: jax.Array | Numeric) -> jax.Array:
        """Integrate a quantity over the mesh. Returns 0 if dormant."""
        ctx = get_context()

        # 1. GLOBAL MODE
        if ctx is None:
            res = self.integrate_per_element(arg)
            return jnp.sum(res, axis=(0,))

        # 2. LOCAL ACTIVE MODE
        if ctx.active_op is self:
            # arg is local quantity at quad points: shape (n_quad,)
            det_J = jax.vmap(
                lambda xi: self.element.get_jacobian(xi, ctx.el_coords)[1]
            )(self.element.quad_points)
            return jnp.dot(arg * det_J, self.element.quad_weights)

        # 3. DORMANT MODE
        return 0.0

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

    def eval(self, nodal_values: jax.Array) -> jax.Array:
        """Evaluates nodal values at quad points. Handles Projection/Multi-mesh."""
        ctx = get_context()

        # 1. GLOBAL MODE
        if ctx is None:

            def _eval_quad(xi, el_nodal_values, el_nodal_coords):
                return self.element.interpolate(xi, el_nodal_values)

            return self._vmap_over_elements_and_quads(nodal_values, _eval_quad)

        # 2. LOCAL ACTIVE MODE
        if ctx.active_op is self:
            # nodal_values is the local element vector here
            return jax.vmap(lambda xi: self.element.interpolate(xi, nodal_values))(
                self.element.quad_points
            )

        # 3. PROJECT MODE (Multiphysics A * B)
        # I am NOT the active operator, but I need to provide values at the ACTIVE operator's points.

        # A. Compute Active Physical Points (Lazy Evaluation)
        # if ctx.active_quad_points_phys is None:
        #    ctx.active_quad_points_phys = jax.vmap(ctx.active_op.element.interpolate, (0, None))(
        #        ctx.active_op.element.quad_points, ctx.el_coords
        #    )
        # target_points = ctx.active_quad_points_phys

        # B. Check for FAST PATH (Identical Meshes)
        if self.mesh is ctx.active_op.mesh:
            # Gather LOCAL u for this operator using the context's element index
            n_nodes = self.mesh.coords.shape[0]
            # nodal_values passed here is Global in project mode
            el_u_flat = nodal_values.reshape(n_nodes, -1)[
                self.mesh.elements[ctx.el_idx]
            ]
            el_u = el_u_flat.reshape(self.element.quad_points.shape[0], -1)

            # Evaluate at the ACTIVE op's reference points (assuming aligned elements)
            return jax.vmap(lambda xi: self.element.interpolate(xi, el_u))(
                ctx.active_op.element.quad_points
            )

        # C. SLOW PATH (Different Meshes) - Placeholder
        target_points = jax.vmap(ctx.active_op.element.interpolate, (0, None))(
            ctx.active_op.element.quad_points, ctx.el_coords
        )
        # (Placeholder for generic point search/interpolation)
        return jnp.zeros((target_points.shape[0], 1))

    def grad(self, nodal_values: jax.Array) -> jax.Array:
        """Computes gradient. Behaves differently based on context."""
        ctx = get_context()

        # 1. GLOBAL MODE
        if ctx is None:

            def _gradient_quad(xi, el_nodal_values, el_nodal_coords):
                return self.element.gradient(xi, el_nodal_values, el_nodal_coords)

            return self._vmap_over_elements_and_quads(nodal_values, _gradient_quad)

        # 2. LOCAL ACTIVE MODE
        if ctx.active_op is self:
            # nodal_values is the LOCAL element vector here
            return jax.vmap(
                lambda xi: self.element.gradient(xi, nodal_values, ctx.el_coords)
            )(self.element.quad_points)

        # 3. DORMANT MODE (Return zeros of correct shape for JAX tracing)
        dim = self.mesh.coords.shape[1]
        n_q = self.element.quad_points.shape[0]
        return jnp.zeros((n_q, dim, dim))

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

            # Create Sparse Matrix using PRE-COMPUTED sparsity
            # self._indices_T and self._nse are captured as constants
            mat = sparse.BCOO((vals, self._indices_T), shape=(u_flat.size, u_flat.size))

            # Sum duplicates using PRE-COMPUTED NSE
            return mat.sum_duplicates(nse=self._nse)

        return assemble_loop


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
