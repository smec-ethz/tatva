from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax_autovmap import autovmap

from tatva.compound import Compound, field
from tatva.lifter import Fixed, Lifter, Periodic
from tatva.mesh import Mesh
from tatva.sparse.tracer.base import _JaxprAnalyzer, make_partition_plan
from tatva.sparse.tracer.partitioning import (
    AllRows,
    ArrayRows,
    AxisProduct,
    ContributionRows,
    RangeRows,
    build_local_program,
    materialize_local_jaxpr,
    pack_runtime_inputs,
    plan_local_jaxpr,
    seed_demand,
    trace_local_program,
)
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  # Diffusion coefficient
    lmbda: float  # Diffusion coefficient

    @classmethod
    def from_youngs_poisson_2d(
        cls, E: float, nu: float, plane_stress: bool = False
    ) -> "Material":
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * nu * mu / (1 - nu)
        else:
            lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
        return cls(mu=mu, lmbda=lmbda)


@autovmap(grad_u=2)
def compute_deformation_gradient(grad_u: Array) -> Array:
    return jnp.eye(2) + grad_u


@autovmap(grad_u=2, mat=None)
def strain_energy_density(grad_u: Array, mat: Material) -> Array:
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)
    return (
        mat.mu / 2 * (jnp.trace(C) - 2)  # 2D case
        - mat.mu * jnp.log(J)
        + (mat.lmbda / 2) * (jnp.log(J)) ** 2
    )


@autovmap(grad_u=2, mat=None)
def get_cauchy_stress(grad_u: Array, mat: Material) -> Array:
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)

    C_inv = jnp.linalg.inv(C)
    S = mat.mu * (jnp.eye(2) - C_inv) + mat.lmbda * jnp.log(J) * C_inv  # 2nd PK
    P = F @ S  # 1st PK

    sigma = (P @ F.T) / J  # Cauchy
    return sigma


@autovmap(grad_u=2, mat=None)
def get_stress(grad_u: Array, mat: Material) -> Array:
    # 2nd Piola-Kirchhoff stress tensor
    F = compute_deformation_gradient(grad_u)
    C = F.T @ F
    J = jnp.linalg.det(F)
    C_inv = jnp.linalg.inv(C)
    S = mat.mu * (jnp.eye(2) - C_inv) + mat.lmbda * jnp.log(J) * C_inv  # 2nd PK
    return S


def von_mises_stress(sig):
    s_xx, s_yy, s_xy = sig[..., 0, 0], sig[..., 1, 1], sig[..., 0, 1]
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)


def _local_result(fn, value, rows, *, root_index=-1):
    closed = jax.make_jaxpr(fn)(value)
    analysis = _JaxprAnalyzer(closed).analyze()
    state = TraceState(analysis.n_dofs, analysis.active_ids, analysis.sub_info)
    state.attach_concrete_values(closed, [np.asarray(value)])
    state.seed_input_dependencies(closed)
    state.run_bound_eqns(analysis.bound_eqns, CouplingAccumulator(analysis.n_dofs))
    root = closed.jaxpr.outvars[root_index]
    plan = plan_local_jaxpr(
        closed.jaxpr,
        state,
        {root: seed_demand(root, np.asarray(rows, dtype=np.int64))},
        [root],
    )
    program = materialize_local_jaxpr(plan, closed.consts)
    (result,) = program.fn(*pack_runtime_inputs(program, [value]))
    return np.asarray(result), plan, program


def test_rowsets_normalize_and_localize_without_dense_full_storage():
    assert isinstance(ContributionRows(ArrayRows(np.array([1, 3]))).rows, ArrayRows)
    np.testing.assert_array_equal(ArrayRows(np.array([2, 5])).localize([5, 2]), [1, 0])
    np.testing.assert_array_equal(RangeRows(4, 7).localize([4, 6]), [0, 2])
    np.testing.assert_array_equal(AllRows(10).localize([0, 9]), [0, 9])


def test_full_nonscalar_local_storage_preserves_tensor_shape():
    result, plan, program = _local_result(
        lambda x: x.reshape(2, 3), jnp.arange(6.0), np.arange(6)
    )
    np.testing.assert_allclose(result, np.arange(6.0).reshape(2, 3))
    assert plan.layouts[plan.requested_outputs[0]].local_aval.shape == (2, 3)
    assert program.input_specs[0].layout.local_aval.shape == (6,)


def test_local_jaxpr_evaluates_requested_intermediate_rows():
    def fn(x):
        squared = x * x
        return squared, jnp.sum(squared)

    result, plan, program = _local_result(fn, jnp.arange(6.0), [1, 4], root_index=0)
    np.testing.assert_allclose(result, [1.0, 16.0])
    assert program.output_specs[0].original_var is plan.requested_outputs[0]
    assert program.input_specs[0].layout.local_size == 2


def test_local_jaxpr_routes_structured_transpose_slice_broadcast_and_reverse():
    result, plan, _ = _local_result(
        lambda x: jnp.transpose(x.reshape(2, 3))[:, 1:],
        jnp.arange(6.0),
        [0, 2],
    )
    np.testing.assert_allclose(np.asarray(result).reshape(-1), [3.0, 5.0])
    assert plan.layouts[plan.requested_outputs[0]].local_aval.shape == (2, 1)

    result, _, _ = _local_result(
        lambda x: jnp.broadcast_to(x[0], (2, 3)),
        jnp.arange(5.0),
        [0, 1, 3, 4],
    )
    np.testing.assert_allclose(result, [[0.0, 0.0], [0.0, 0.0]])

    result, _, program = _local_result(lambda x: x[::-1], jnp.arange(6.0), [0, 1])
    np.testing.assert_allclose(result, [5.0, 4.0])
    assert jax.make_jaxpr(program.fn)(*pack_runtime_inputs(program, [jnp.arange(6.0)]))


def test_local_jaxpr_lifter_literal():
    lifter = Lifter.make(
        100,
    )

    def fn(x):
        x = lifter.lift_from_zeros(x)
        return x[4:] * x[:-4]

    result, _, _ = _local_result(fn, jnp.ones(lifter.size_reduced), [4, 5])
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_local_jaxpr_lifter_periodic_fail():
    n = 4
    mesh = Mesh.unit_square(n, n)

    class Solution(Compound, mesh=mesh):
        u = field((-1, 2))

    bottom = np.where(np.isclose(mesh.coords[:, 1], 0))[0]
    top = np.where(np.isclose(mesh.coords[:, 1], 1))[0]
    corner_0 = np.where((mesh.coords[:, 0] == 0) & (mesh.coords[:, 1] == 0))[0]

    lifter = Lifter.make(
        # lifter = Lifter(
        mesh.coords.shape[0] * 2,
        Fixed(Solution.u[corner_0]),
        # Periodic(Solution.u[right, :], Solution.u[left, :]),
        Periodic(Solution.u[top, :], Solution.u[bottom, :]),
    )

    def fn(x):
        x = lifter.lift_from_zeros(x)
        return x[4:] * x[:-4]

    # fmt: off
    rows = np.array([ 2,  6,  9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34, 37,
        38, 41, 42, 45])
    # fmt: on

    x = jnp.ones(lifter.size_reduced)
    result, _, _ = _local_result(fn, x, rows)
    np.testing.assert_allclose(np.asarray(result).reshape(-1), np.asarray(fn(x)).reshape(-1)[rows])


def test_local_jaxpr_specializes_gather_routes():
    result, _, program = _local_result(
        lambda x: x[jnp.array([4, 1, 3])], jnp.arange(7.0), [0, 2]
    )
    np.testing.assert_allclose(result, [4.0, 3.0])
    # The original global index array is no longer an input or constvar.
    assert len(program.input_specs) == 1


def test_local_jaxpr_emits_structured_elementwise_predicates():
    result, _, _ = _local_result(lambda x: x > 3, jnp.arange(8), [2, 6])
    np.testing.assert_array_equal(result, [False, True])


def test_local_jaxpr_reduces_each_selected_output_row():
    result, plan, _ = _local_result(
        lambda x: jnp.sum(x.reshape(3, 4), axis=1), jnp.arange(12.0), [0, 2]
    )
    np.testing.assert_allclose(result, [6.0, 38.0])
    reduce_eqn = next(eqn for eqn in plan.original_jaxpr.eqns if eqn.primitive.name == "reduce_sum")
    assert plan.layouts[reduce_eqn.invars[0]].local_aval.shape == (2, 4)
    assert isinstance(plan.layouts[reduce_eqn.invars[0]].subset, AxisProduct)

    result, plan, _ = _local_result(
        lambda x: jnp.sum(x.reshape(3, 4), axis=0), jnp.arange(12.0), [1, 3]
    )
    np.testing.assert_allclose(result, [15.0, 21.0])
    reduce_eqn = next(eqn for eqn in plan.original_jaxpr.eqns if eqn.primitive.name == "reduce_sum")
    assert plan.layouts[reduce_eqn.invars[0]].local_aval.shape == (3, 2)


def test_local_jaxpr_emits_selected_dot_general_rows():
    result, _, _ = _local_result(
        lambda x: x.reshape(2, 3) @ jnp.array([2.0, 3.0, 5.0]),
        jnp.arange(6.0),
        [1],
    )
    np.testing.assert_allclose(result, [43.0])


def test_local_jaxpr_executes_selected_fem_dot_blocks():
    reference_gradient = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

    def fn(x):
        element_values = x.reshape(3, 5, 2)
        return jax.lax.dot_general(
            reference_gradient,
            element_values,
            dimension_numbers=(((1,), (0,)), ((), ())),
        )

    # Complete output blocks for non-contiguous elements 1 and 3.
    rows = [2, 3, 6, 7, 12, 13, 16, 17]
    x = jnp.arange(30.0)
    result, plan, program = _local_result(fn, x, rows)
    np.testing.assert_allclose(np.asarray(result).reshape(-1), np.asarray(fn(x)).reshape(-1)[rows])
    assert plan.layouts[plan.requested_outputs[0]].local_aval.shape == (2, 2, 2)
    local_x = pack_runtime_inputs(program, [x])[0]
    assert trace_local_program(program, local_x)
    np.testing.assert_allclose(
        np.asarray(jax.jit(program.fn)(local_x)[0]).reshape(-1),
        np.asarray(fn(x)).reshape(-1)[rows],
    )


def test_local_jaxpr_executes_selected_leading_batch_dot_blocks():
    shared = jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])

    def fn(x):
        return jax.lax.dot_general(
            x.reshape(5, 2, 3),
            shared,
            dimension_numbers=(((2,), (0,)), ((), ())),
        )

    # Complete [2, 2] output blocks for non-contiguous batches 1 and 3.
    rows = [4, 5, 6, 7, 12, 13, 14, 15]
    x = jnp.arange(30.0)
    result, plan, program = _local_result(fn, x, rows)
    np.testing.assert_allclose(np.asarray(result).reshape(-1), np.asarray(fn(x)).reshape(-1)[rows])
    assert plan.layouts[plan.requested_outputs[0]].local_aval.shape == (2, 2, 2)
    local_x = pack_runtime_inputs(program, [x])[0]
    assert trace_local_program(program, local_x)
    np.testing.assert_allclose(
        np.asarray(jax.jit(program.fn)(local_x)[0]).reshape(-1),
        np.asarray(fn(x)).reshape(-1)[rows],
    )


def test_local_jaxpr_specializes_dynamic_slice_starts():
    result, _, program = _local_result(
        lambda x: lax.dynamic_slice(x, (x[0].astype(jnp.int32),), (3,)),
        jnp.array([2.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        [0, 2],
    )
    np.testing.assert_allclose(result, [20.0, 40.0])
    # The dynamic start value is specialized and is not a local input.
    assert program.input_specs[0].layout.rows.to_array().tolist() == [2, 4]


def test_local_jaxpr_rewrites_nested_jit():
    @jax.jit
    def inner(x):
        return jnp.sin(x * 2)

    result, _, _ = _local_result(lambda x: inner(x), jnp.arange(6.0), [1, 4])
    np.testing.assert_allclose(result, np.sin(np.array([2.0, 8.0])))


def test_local_jaxpr_executes_single_iteration_scan_without_carry():
    def fn(x):
        xs = x.reshape(1, 3)
        _, ys = lax.scan(lambda _, value: ((), jnp.sin(value) * 2), (), xs)
        return ys

    result, _, program = _local_result(fn, jnp.arange(3.0), [1, 2])
    np.testing.assert_allclose(np.asarray(result).reshape(-1), 2 * np.sin([1.0, 2.0]))
    local_x = pack_runtime_inputs(program, [jnp.arange(3.0)])[0]
    assert trace_local_program(program, local_x)
    np.testing.assert_allclose(
        np.asarray(jax.jit(program.fn)(local_x)[0]).reshape(-1),
        2 * np.sin([1.0, 2.0]),
    )


def test_local_callable_traces_jits_and_differentiates():
    _, _, program = _local_result(lambda x: jnp.sin(x) * 2, jnp.arange(6.0), [1, 4])
    local_x = pack_runtime_inputs(program, [jnp.arange(6.0)])[0]
    assert trace_local_program(program, local_x)
    np.testing.assert_allclose(
        jax.jit(program.fn)(local_x)[0], [2 * np.sin(1), 2 * np.sin(4)]
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(program.fn(x)[0]))(local_x),
        2 * np.cos(np.array([1.0, 4.0])),
    )


def test_minimal_fem_example():
    n = 20
    mesh = Mesh.unit_square(n, n)

    class Solution(Compound, mesh=mesh):
        u = field((-1, 2))

    bottom = np.where(np.isclose(mesh.coords[:, 1], 0))[0]
    top = np.where(np.isclose(mesh.coords[:, 1], 1))[0]
    right = np.where(mesh.coords[:, 0] == 1)[0]
    left = np.where(mesh.coords[:, 0] == 0)[0]
    corner_0 = np.where((mesh.coords[:, 0] == 0) & (mesh.coords[:, 1] == 0))[0]

    lifter = Lifter.make(
        mesh.coords.shape[0] * 2,
        Fixed(Solution.u[corner_0]),
        Periodic(Solution.u[right, :], Solution.u[left, :]),
        Periodic(Solution.u[top, :], Solution.u[bottom, :]),
    )

    import pymetis

    from tatva import sparse
    from tatva.element.base import Tri3
    from tatva.operator import Operator
    from tatva.sparse._coloring import csr_to_adjacency

    def energy_functional(
        z: Array,  # flat array of reduced dofs
        op: Operator,  # fem operator
        lifter: Lifter,  # lifting operator
        mat: Material,
    ) -> Array:
        z_full = lifter.lift_from_zeros(z)  # lift operation
        (u,) = Solution(z_full)  # reshape flat array into fields
        grad_u = op.grad(u)
        psi = strain_energy_density(grad_u, mat)
        return op.integrate(psi)

    def dummy(
        z: Array,
        lifter: Lifter,
    ):
        z_full = lifter.lift_from_zeros(z)
        # z_full = z
        return jnp.sum(z_full[4:] * z_full[:-4])

    op = Operator(mesh, Tri3())
    mat = Material.from_youngs_poisson_2d(2e3, 0.3)
    trace = sparse.trace_energy(energy_functional)(
        jnp.zeros(lifter.size_reduced), op=op, lifter=lifter, mat=mat
    )

    options = pymetis.Options(
        contig=0,  # require each partition to be connected
        minconn=1,  # reduce number of neighboring partitions
        ncuts=5,  # try several initial partitions
        niter=10,  # more refinement
        seed=1,
    )

    sparsity = trace.pattern
    adj = csr_to_adjacency(sparsity.shape[0], sparsity.indptr, sparsity.indices)

    num_edges = sum(map(len, adj)) // 2
    edgecut, parts = pymetis.part_graph(
        2,
        adjacency=adj,
        recursive=False,  # direct k-way; contig/minconn apply here
        options=options,
    )
    part_map = np.asarray(parts)
    partition_plan = make_partition_plan(trace, part_map)
    rank = 0
    local_program = build_local_program(partition_plan, rank)

    args = pack_runtime_inputs(local_program, trace.concrete_jaxpr.flat_args)
    # this must run
    local_result = local_program.fn(*args)[0]
