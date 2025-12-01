from typing import Callable, Sequence
import jax.numpy as jnp
from jax.experimental import sparse


from tatva.operator import Operator, AssemblyContext, _local_state


def assemble(
    total_energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    operators: Sequence[Operator],
    u_flat: jnp.ndarray
) -> sparse.BCOO:
    """
    Orchestrates the assembly of the global stiffness matrix.
    Uses context injection to switch operators between Local/Active and Global/Passive modes.
    """
    
    # 1. Clean previous state (Safety)
    if hasattr(_local_state, "ctx"):
        _local_state.ctx = None
    
    sparse_matrices = []

    for active_op in operators:
        
        # 2. Define the wrapper that injects the context for this element
        def element_context_wrapper(el_u_flat, el_coords, el_idx):
            n_dim = active_op.mesh.coords.shape[1]
            n_nodes_el = el_coords.shape[0]
            el_u = el_u_flat.reshape(n_nodes_el, n_dim)

            # Create context (No side-effects or caching in this object!)
            ctx = AssemblyContext(active_op, el_coords, el_idx)
            
            # 3. Set Context
            # We use simple attribute setting. JAX allows this on thread-locals.
            _local_state.ctx = ctx
            
            try:
                # 4. Run Physics
                return total_energy_fn(el_u)
            finally:
                # 5. Clear Context
                _local_state.ctx = None

        # 6. Generate & Run Assembler
        assembler = active_op.get_element_assembler(element_context_wrapper)
        K_sub = assembler(u_flat)
        sparse_matrices.append(K_sub)

    if not sparse_matrices:
        return sparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)), 
            shape=(u_flat.size, u_flat.size)
        )

    # 7. Sum results
    total_K = sparse_matrices[0]
    for K_sub in sparse_matrices[1:]:
        total_K = total_K + K_sub
        
    return total_K