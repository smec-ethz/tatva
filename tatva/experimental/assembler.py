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


from typing import Callable, Dict, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from tatva.operator import Operator


class TotalEnergyCallable(Protocol):
    """
    Type hint for the energy function.
    It expects a displacement array and keyword arguments for each operator.
    Signature: def energy(u, op1=..., op2=...)
    """

    @staticmethod
    def __call__(nodal_values_flat: Array, **operators: Operator) -> Array: ...


def assemble(
    total_energy_fn: TotalEnergyCallable,
    operators: Dict[str, Operator],
    nodal_values_flat: Array,
) -> sparse.BCOO:
    """
    Orchestrates the assembly of the global stiffness matrix.

    Arguments:
        total_energy_fn: User function calculating total energy.
            Must match signature: def func(u, **operators).
        operators: Dictionary of operators involved. Keys must match the argument names
            in total_energy_fn.
            Example: operators={"solid": op1} -> total_energy_fn(u, solid=...)
        u_flat: Global displacement vector.
    """

    if not isinstance(operators, dict):
        raise TypeError(
            "The 'operators' argument must be a dictionary: {'arg_name': operator_instance}"
        )

    sparse_matrices = []

    # We iterate through every operator, treating it as the 'Active' one (calculating stiffness)
    # while treating all others as 'Passive' (providing values via projection).
    for active_op in operators.values():

        def element_context_wrapper(el_u_flat, el_coords, el_idx):
            n_dim = active_op.mesh.coords.shape[1]
            n_nodes_el = el_coords.shape[0]
            el_u = el_u_flat.reshape(n_nodes_el, n_dim)

            # Create Functional Copies of Operators with Context Baked In
            local_ops_kwargs = {}

            # Iterate over the user's dictionary to preserve keys for injection
            for name, op in operators.items():
                if op._id == active_op._id:
                    # Active Operator: Set to Local Mode
                    # This operator will compute gradients/integrals for the current element
                    local_ops_kwargs[name] = op.with_context(
                        "local", el_coords, active_op._id
                    )
                else:
                    # Other Operators: Set to Project Mode
                    # This operator will interpolate its values onto the active operator's points
                    local_ops_kwargs[name] = op.with_context(
                        "project", el_coords, active_op._id, source_op=active_op
                    )

            # CALL: Injection via Keywords
            # We inject the context-aware operators into the user's function
            return total_energy_fn(el_u, **local_ops_kwargs)

        # Generate the efficient JIT-compiled assembler for this specific operator path
        assembler = active_op.get_element_assembler(element_context_wrapper)

        # Compute this operator's contribution to the global stiffness matrix
        K_sub = assembler(nodal_values_flat)
        sparse_matrices.append(K_sub)

    if not sparse_matrices:
        return sparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=(nodal_values_flat.size, nodal_values_flat.size),
        )

    # Sum all contributions
    total_K = sparse_matrices[0]
    for K_sub in sparse_matrices[1:]:
        total_K = total_K + K_sub

    return total_K


"""
def assemble(
    total_energy_fn: Callable, operators: Sequence[Operator], u_flat: jnp.ndarray
) -> sparse.BCOO:
    sparse_matrices = []

    for i, active_op in enumerate(operators):

        def element_context_wrapper(el_u_flat, el_coords, el_idx):
            n_dim = active_op.mesh.coords.shape[1]
            n_nodes_el = el_coords.shape[0]
            el_u = el_u_flat.reshape(n_nodes_el, n_dim)

            # 1. Create Functional Copies of Operators with Context Baked In
            local_ops = []
            for op in operators:
                if op._id == active_op._id:
                    # Active Operator: Set to Local Mode
                    local_ops.append(op.with_context("local", el_coords, active_op._id))
                else:
                    # Other Operators: Set to Project Mode (or Dormant)
                    local_ops.append(
                        op.with_context(
                            "project", el_coords, active_op._id, source_op=active_op
                        )
                    )

            # Call user function with INJECTED operators
            # We unpack the list: total_energy(u, op1, op2, ...)
            return total_energy_fn(el_u, *local_ops)

        assembler = active_op.get_element_assembler(element_context_wrapper)
        K_sub = assembler(u_flat)
        sparse_matrices.append(K_sub)

    if not sparse_matrices:
        return sparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=(u_flat.size, u_flat.size),
        )

    total_K = sparse_matrices[0]
    for K_sub in sparse_matrices[1:]:
        total_K = total_K + K_sub

    return total_K
"""
