import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

from tatva.compound import Compound, field
from tatva.compound.field_types import CG1, DG0, Nodal
from tatva.mesh import Mesh

jax.config.update("jax_enable_x64", True)

def test_cg1_sparsity_pattern():
    # 2x2 nodes, 2 triangles
    coords = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=float)
    elements = jnp.array([[0,1,3], [0,3,2]], dtype=int)
    mesh = Mesh(coords=coords, elements=elements)

    class State(Compound, mesh=mesh):
        u = field((4,), field_type=CG1())

    sparsity = State.get_sparsity()
    
    # Check that it's a CSR matrix
    assert isinstance(sparsity, sps.csr_matrix)
    assert sparsity.shape == (4, 4)
    
    # Connectivity:
    # Element 0: (0,1,3) -> couplings (0,0),(0,1),(0,3),(1,0),(1,1),(1,3),(3,0),(3,1),(3,3)
    # Element 1: (0,3,2) -> couplings (0,0),(0,3),(0,2),(3,0),(3,3),(3,2),(2,0),(2,3),(2,2)
    
    # Expected non-zeros (sorted):
    # Row 0: 0, 1, 2, 3
    # Row 1: 0, 1, 3
    # Row 2: 0, 2, 3
    # Row 3: 0, 1, 2, 3
    
    expected_nonzeros = [
        (0,0), (0,1), (0,2), (0,3),
        (1,0), (1,1), (1,3),
        (2,0), (2,2), (2,3),
        (3,0), (3,1), (3,2), (3,3)
    ]
    
    actual_nonzeros = list(zip(*sparsity.nonzero()))
    assert set(actual_nonzeros) == set(expected_nonzeros)

def test_mixed_cg1_dg0_sparsity():
    coords = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=float)
    elements = jnp.array([[0,1,3], [0,3,2]], dtype=int)
    mesh = Mesh(coords=coords, elements=elements)

    class StokesState(Compound, mesh=mesh):
        u = field((4,), field_type=CG1())
        p = field((2,), field_type=DG0())

    # Total size: 4 (u) + 2 (p) = 6
    assert StokesState.size == 6
    
    sparsity = StokesState.get_sparsity()
    assert sparsity.shape == (6, 6)
    
    # Check coupling between u and p
    # Element 0: nodes (0,1,3), p-index 0 (global index 4)
    # Element 1: nodes (0,3,2), p-index 1 (global index 5)
    
    # Couplings for Element 0:
    # (0,4), (1,4), (3,4) and (4,0), (4,1), (4,3)
    # Couplings for Element 1:
    # (0,5), (3,5), (2,5) and (5,0), (5,3), (5,2)
    
    assert sparsity[0, 4] == 1
    assert sparsity[1, 4] == 1
    assert sparsity[3, 4] == 1
    assert sparsity[4, 0] == 1
    assert sparsity[4, 1] == 1
    assert sparsity[4, 3] == 1
    
    assert sparsity[0, 5] == 1
    assert sparsity[3, 5] == 1
    assert sparsity[2, 5] == 1
    assert sparsity[5, 0] == 1
    assert sparsity[5, 3] == 1
    assert sparsity[5, 2] == 1
    
    # Check that p is diagonal with itself (since DG0 only couples within element, and there's only 1 DOF per element)
    assert sparsity[4, 4] == 1
    assert sparsity[5, 5] == 1
    assert sparsity[4, 5] == 0
    assert sparsity[5, 4] == 0

def test_block_wise_sparsity():
    coords = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=float)
    elements = jnp.array([[0,1,3], [0,3,2]], dtype=int)
    mesh = Mesh(coords=coords, elements=elements)

    class State(Compound, mesh=mesh):
        u = field((4,), field_type=CG1())
        p = field((2,), field_type=DG0())

    blocks = State.get_sparsity(block_wise=True)
    assert len(blocks) == 2
    assert len(blocks[0]) == 2
    
    K_uu = blocks[0][0]
    K_up = blocks[0][1]
    K_pu = blocks[1][0]
    K_pp = blocks[1][1]
    
    assert K_uu.shape == (4, 4)
    assert K_up.shape == (4, 2)
    assert K_pu.shape == (2, 4)
    assert K_pp.shape == (2, 2)
    
    # Check K_up entries
    assert K_up[0, 0] == 1
    assert K_up[1, 0] == 1
    assert K_up[3, 0] == 1
    assert K_up[0, 1] == 1
    assert K_up[3, 1] == 1
    assert K_up[2, 1] == 1
    
    # K_pp should be identity
    assert np.all(K_pp.toarray() == np.eye(2))

def test_get_coupling_block_direct():
    coords = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=float)
    elements = jnp.array([[0,1,3], [0,3,2]], dtype=int)
    mesh = Mesh(coords=coords, elements=elements)

    class State(Compound, mesh=mesh):
        u = field((4,), field_type=CG1())
        p = field((2,), field_type=DG0())
        
    u_field = State.u
    p_field = State.p
    
    K_up = CG1().get_coupling_block(u_field, p_field, mesh)
    assert K_up.shape == (4, 2)
    assert K_up[0, 0] == 1
    assert K_up[1, 0] == 1
    assert K_up[3, 0] == 1
    assert K_up[0, 1] == 1
    assert K_up[3, 1] == 1
    assert K_up[2, 1] == 1

if __name__ == "__main__":
    pytest.main([__file__])
