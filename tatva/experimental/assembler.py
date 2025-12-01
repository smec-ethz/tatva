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


from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from tatva.operator import AssemblyContext, Operator, _local_state


def assemble(
    total_energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    operators: Sequence[Operator],
    u_flat: jnp.ndarray,
) -> sparse.BCOO:
    """
    Orchestrates the assembly of the global stiffness matrix.
    Uses context injection to switch operators between Local/Active and Global/Passive modes.
    """

    # Clean previous state (Safety)
    if hasattr(_local_state, "ctx"):
        _local_state.ctx = None

    sparse_matrices = []

    for active_op in operators:
        # Define the wrapper that injects the context for this element
        def element_context_wrapper(el_u_flat, el_coords, el_idx):
            n_dim = active_op.mesh.coords.shape[1]
            n_nodes_el = el_coords.shape[0]
            el_u = el_u_flat.reshape(n_nodes_el, n_dim)

            # Create context (No side-effects or caching in this object!)
            ctx = AssemblyContext(active_op, el_coords, el_idx)

            # Set Context
            # We use simple attribute setting. JAX allows this on thread-locals.
            _local_state.ctx = ctx

            try:
                # Run Physics
                return total_energy_fn(el_u)
            finally:
                # Clear Context
                _local_state.ctx = None

        # Generate & Run Assembler
        assembler = active_op.get_element_assembler(element_context_wrapper)
        K_sub = assembler(u_flat)
        jax.debug.print("K_sub {K_sub}", K_sub=K_sub.todense())
        sparse_matrices.append(K_sub)

    if not sparse_matrices:
        return sparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=(u_flat.size, u_flat.size),
        )

    # Sum results
    total_K = sparse_matrices[0]
    for K_sub in sparse_matrices[1:]:
        total_K = total_K + K_sub

    return total_K
