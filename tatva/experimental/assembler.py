# Copyright (C) 2025 ETH Zurich
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


from typing import Dict, Protocol

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
    nodal_values_flat: jnp.ndarray,
) -> sparse.BCOO:
    """
    Orchestrates the assembly of the global stiffness matrix using element-wise assembly.

    Args:
        total_energy_fn: User-defined function to compute total energy.
            Must match signature: def func(u, **operators).
        operators: Dictionary of operators involved. Keys must match the argument names
            in total_energy_fn.
            Example: operators={"solid": op1} -> total_energy_fn(u, solid=...)
        nodal_values_flat: Global displacement vector as a flat array.

    Returns:
        Assembled global stiffness matrix in sparse BCOO format.
    """

    if not isinstance(operators, dict):
        raise TypeError("The 'operators' argument must be a dictionary.")

    sparse_matrices = []

    for active_op in operators.values():

        def element_context_wrapper(el_u_flat, el_coords, el_idx):
            n_dim = active_op.mesh.coords.shape[1]
            n_nodes_el = el_coords.shape[0]
            el_u = el_u_flat.reshape(n_nodes_el, n_dim)

            local_ops_kwargs = {}
            for name, op in operators.items():
                if op._id == active_op._id:
                    # Update with el_idx
                    local_ops_kwargs[name] = op.with_context(
                        "local",
                        el_coords=el_coords,
                        el_idx=el_idx,
                        active_id=active_op._id,
                    )
                else:
                    # Update with el_idx
                    local_ops_kwargs[name] = op.with_context(
                        "project",
                        el_coords=el_coords,
                        el_idx=el_idx,
                        active_id=active_op._id,
                        source_op=active_op,
                    )

            return total_energy_fn(el_u, **local_ops_kwargs)

        assembler = active_op.get_element_assembler(element_context_wrapper)
        K_sub = assembler(nodal_values_flat)
        sparse_matrices.append(K_sub)

    if not sparse_matrices:
        return sparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=(nodal_values_flat.size, nodal_values_flat.size),
        )

    total_K = sparse_matrices[0]
    for K_sub in sparse_matrices[1:]:
        total_K = total_K + K_sub

    return total_K
