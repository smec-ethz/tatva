import jax.numpy as jnp
import pytest
from tatva.compound import Compound, field
from tatva.compound.field import FieldSize, FieldType
from tatva.mesh import Mesh

def test_auto_sizing_nodal_fields():
    # Mock mesh
    n_nodes = 10
    n_dim = 2
    coords = jnp.zeros((n_nodes, n_dim))
    mesh = Mesh(coords=coords, elements=None)

    class MyState(Compound, mesh=mesh):
        param1 = field(shape=(5,), field_type=FieldType.LOCAL)
        u = field(shape=(FieldSize.AUTO, 3), field_type=FieldType.NODAL)
        phi = field(shape=(FieldSize.AUTO,), field_type=FieldType.NODAL)
        param2 = field(shape=(2,), field_type=FieldType.LOCAL)

    state = MyState()
    
    # 1. Check original order in cls.fields
    field_names = [name for name, _ in MyState.fields]
    assert field_names == ["param1", "u", "phi", "param2"]
    
    # 2. Check unpacking order matches
    p1, u, phi, p2 = state
    assert p1.shape == (5,)
    assert u.shape == (10, 3)
    assert phi.shape == (10,)
    assert p2.shape == (2,)

    # 3. Check shapes on class
    assert MyState.u.shape == (10, 3)
    assert MyState.phi.shape == (10,)
    assert MyState.param1.shape == (5,)
    assert MyState.param2.shape == (2,)
    
    # 4. Check total size: (10 * 3) + (10 * 1) + 5 + 2 = 30 + 10 + 7 = 47
    assert MyState.size == 47
    assert state.arr.size == 47

    # 5. Check memory layout (as implemented: Nodal block first, then others)
    # u: start=0, end=3. phi: start=3, end=4. K=4.
    # Nodal block size = 10 * 4 = 40.
    # After nodal block: param1 (5), then param2 (2).
    
    # u indices: N*K block starting at 0
    u_indices = MyState.u.indices(slice(None))
    expected_u = jnp.array([i*4 + j for i in range(10) for j in range(0, 3)])
    assert jnp.all(u_indices == expected_u)
    
    # phi indices: N*K block starting at 0, with offset 3
    phi_indices = MyState.phi.indices(slice(None))
    expected_phi = jnp.array([i*4 + 3 for i in range(10)])
    assert jnp.all(phi_indices == expected_phi)
    
    # param1 indices: after nodal block (offset 40)
    param1_indices = MyState.param1.indices(slice(None))
    assert jnp.all(param1_indices == jnp.arange(40, 45))
    
    # param2 indices: after param1 (offset 45)
    param2_indices = MyState.param2.indices(slice(None))
    assert jnp.all(param2_indices == jnp.arange(45, 47))

if __name__ == "__main__":
    pytest.main([__file__])
