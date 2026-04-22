import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tatva.compound import Compound, FieldSize, field, stack_fields
from tatva.mesh import Mesh

jax.config.update("jax_enable_x64", True)


class SimpleState(Compound):
    u = field((2, 3))
    phi = field((2,))


@stack_fields("u", "v", axis=-1)
class StackedState(Compound):
    u = field((2, 2))
    v = field((2, 2))
    w = field((2,))


@pytest.mark.parametrize("state_cls", [SimpleState, StackedState])
def test_compound_size_matches_flat_array(state_cls):
    state = state_cls()
    assert state.arr.shape == (state_cls.size,)
    assert jnp.all(state.arr == 0)


def test_compound_field_access_and_assignment():
    state = SimpleState()
    assert SimpleState.size == 8

    np.testing.assert_array_equal(state.u, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(state.phi, jnp.zeros((2,)))

    u_val = jnp.arange(6.0).reshape(2, 3)
    phi_val = jnp.array([10.0, 20.0])

    state = state.at("u").set(u_val)
    state = state.at("phi").set(phi_val)

    np.testing.assert_array_equal(state.u, u_val)
    np.testing.assert_array_equal(state.phi, phi_val)
    np.testing.assert_array_equal(state.arr[:6], u_val.flatten())
    np.testing.assert_array_equal(state.arr[6:], phi_val.flatten())

    assert len(state) == 2
    shapes = [component.shape for component in state]
    assert shapes == [(2, 3), (2,)]


def test_compound_index_helpers():
    np.testing.assert_array_equal(np.array(SimpleState.u[1]), np.array([3, 4, 5]))
    np.testing.assert_array_equal(np.array(SimpleState.u[:, 1]), np.array([1, 4]))
    np.testing.assert_array_equal(np.array(SimpleState.phi[1]), np.array([7]))


def test_pytree_roundtrip_and_addition():
    arr_a = jnp.arange(SimpleState.size, dtype=jnp.float64)
    arr_b = jnp.arange(SimpleState.size, dtype=jnp.float64) * 2

    state_a = SimpleState(arr_a)
    state_b = SimpleState(arr_b)

    leaves, tree_def = jax.tree_util.tree_flatten(state_a)
    assert len(leaves) == 1
    np.testing.assert_array_equal(leaves[0], arr_a)

    rebuilt = jax.tree_util.tree_unflatten(tree_def, leaves)
    assert isinstance(rebuilt, SimpleState)
    np.testing.assert_array_equal(rebuilt.arr, arr_a)
    np.testing.assert_array_equal(rebuilt.u, state_a.u)
    np.testing.assert_array_equal(rebuilt.phi, state_a.phi)

    summed = state_a + state_b
    np.testing.assert_array_equal(summed.arr, arr_a + arr_b)
    np.testing.assert_array_equal(summed.u, state_a.u + state_b.u)
    np.testing.assert_array_equal(summed.phi, state_a.phi + state_b.phi)


def test_stack_fields_access_and_indices():
    state = StackedState(jnp.arange(StackedState.size, dtype=jnp.float64))

    expected_u = jnp.array([[0.0, 1.0], [4.0, 5.0]])
    expected_v = jnp.array([[2.0, 3.0], [6.0, 7.0]])
    expected_w = jnp.array([8.0, 9.0])

    np.testing.assert_array_equal(state.u, expected_u)
    np.testing.assert_array_equal(state.v, expected_v)
    np.testing.assert_array_equal(state.w, expected_w)

    np.testing.assert_array_equal(np.array(StackedState.u[1]), np.array([4, 5]))
    np.testing.assert_array_equal(np.array(StackedState.v[0]), np.array([2, 3]))


def test_auto_sizing_nodal_fields():
    # Mock mesh
    n_nodes = 10
    n_dim = 2
    coords = jnp.zeros((n_nodes, n_dim))
    mesh = Mesh(coords=coords, elements=None)

    class MyState(Compound, mesh=mesh):
        param1 = field(shape=(5,))
        u = field(shape=(FieldSize.AUTO, 3))
        phi = field(shape=(FieldSize.AUTO,))
        param2 = field(shape=(2,))

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
    expected_u = jnp.array([i * 4 + j for i in range(10) for j in range(3)])
    assert jnp.all(u_indices == expected_u)

    # phi indices: N*K block starting at 0, with offset 3
    phi_indices = MyState.phi.indices(slice(None))
    expected_phi = jnp.array([i * 4 + 3 for i in range(10)])
    assert jnp.all(phi_indices == expected_phi)

    # param1 indices: after nodal block (offset 40)
    param1_indices = MyState.param1.indices(slice(None))
    assert jnp.all(param1_indices == jnp.arange(40, 45))

    # param2 indices: after param1 (offset 45)
    param2_indices = MyState.param2.indices(slice(None))
    assert jnp.all(param2_indices == jnp.arange(45, 47))
