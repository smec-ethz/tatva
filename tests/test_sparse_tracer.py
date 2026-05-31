from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, compound, element, lifter, sparse
from tatva.compound import FieldSize
from tatva.sparse import trace_energy_sparsity, trace_virtual_work_sparsity

jax.config.update("jax_enable_x64", True)


def nz_set(m: sps.spmatrix) -> set[tuple[int, int]]:
    """Convert a sparse matrix to a set of ``(row, col)`` nonzero index tuples."""
    r, c = m.nonzero()  # ty:ignore[unresolved-attribute]
    return set(zip(r.tolist(), c.tolist()))


def create_rectangle_box_tetrahedron_mesh(
    lengths: tuple[float, float, float], nb_elems: tuple[int, int, int]
) -> Mesh:
    """Build a structured tetrahedral mesh of a rectangular box (6 tets per hex cell)."""
    x_length, y_length, z_length = lengths
    nx, ny, nz = nb_elems

    x_rng = np.linspace(-x_length / 2, x_length / 2, nx + 1)
    y_rng = np.linspace(-y_length / 2, y_length / 2, ny + 1)
    z_rng = np.linspace(0, z_length, nz + 1)

    Z, Y, X = np.meshgrid(z_rng, y_rng, x_rng, indexing="ij")
    nodes = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    stride_x = 1
    stride_y = nx + 1
    stride_z = (nx + 1) * (ny + 1)

    k_idx, j_idx, i_idx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    n0 = (i_idx * stride_x + j_idx * stride_y + k_idx * stride_z).flatten()

    n1 = n0 + stride_x
    n2 = n0 + stride_y
    n3 = n2 + stride_x
    n4 = n0 + stride_z
    n5 = n4 + stride_x
    n6 = n4 + stride_y
    n7 = n6 + stride_x

    t1 = jnp.stack([n0, n1, n3, n7], axis=-1)
    t2 = jnp.stack([n0, n1, n7, n5], axis=-1)
    t3 = jnp.stack([n0, n5, n7, n4], axis=-1)
    t4 = jnp.stack([n0, n3, n2, n7], axis=-1)
    t5 = jnp.stack([n0, n2, n6, n7], axis=-1)
    t6 = jnp.stack([n0, n6, n4, n7], axis=-1)

    all_tets = jnp.stack([t1, t2, t3, t4, t5, t6], axis=1)
    elements = all_tets.reshape(-1, 4)

    return Mesh(coords=jnp.array(nodes), elements=jnp.array(elements))


@pytest.fixture
def line_operator():
    """Factory fixture: build an Operator over a 1-D Line2 mesh embedded in 2-D.

    ``Line2`` computes its Jacobian from a tangent vector ``[J[0], J[1]]``, so the
    coordinates must live in 2-D; we lay the nodes along the x-axis with ``y = 0``.
    """

    def _make(n_nodes: int, length: float = 1.0) -> Operator:
        x = jnp.linspace(0, length, n_nodes)
        coords = jnp.stack([x, jnp.zeros(n_nodes)], axis=1)
        elements = jnp.stack([jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)], axis=1)
        return Operator(Mesh(coords=coords, elements=elements), element.Line2())

    return _make


# ---------------------------------------------------------------------------
# energy-functional tracing (d²E/du²)
# ---------------------------------------------------------------------------


def test_3d_vector_field_sparsity():
    """3-D Neo-Hookean hyperelasticity with Dirichlet load: the traced Hessian sparsity
    must contain every entry of the tatva-built (reduced) element sparsity pattern."""

    class Material(NamedTuple):
        mu: float
        lmbda: float

    @autovmap(grad_u=2)
    def compute_deformation_gradient(grad_u):
        I = jnp.eye(3)
        F = I + grad_u
        return F

    @autovmap(grad_u=2, mu=0, lmbda=0)
    def neo_hookean_density(grad_u, mu, lmbda):
        F = compute_deformation_gradient(grad_u)
        J = jnp.linalg.det(F)
        C = F.T @ F
        I1 = jnp.trace(C)
        return (mu / 2) * (I1 - 3 - 2 * jnp.log(J)) + (lmbda / 2) * (jnp.log(J)) ** 2

    L, W, H = 10.0, 1.0, 1.0
    nx, ny, nz = 10, 2, 2
    mesh = create_rectangle_box_tetrahedron_mesh((L, H, W), (nx, ny, nz))

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field(shape=(FieldSize.AUTO, 3))

    tet_elem = element.Tetrahedron4()
    op = Operator(mesh, tet_elem)
    mat = Material(mu=500.0, lmbda=1000.0)

    x_min, x_max = jnp.min(mesh.coords[:, 0]), jnp.max(mesh.coords[:, 0])
    fixed_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_min))[0]
    load_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_max))[0]
    applied_u_load = 1.0

    lifter_ = lifter.Lifter(
        Solution.size,
        lifter.Fixed(Solution.u[fixed_nodes, :]),
        lifter.Fixed(Solution.u[load_nodes, 2], values=applied_u_load),
    )

    @jax.jit
    def total_energy(sol: Solution) -> Array:
        (u,) = sol
        grad_u = op.grad(u)
        psi = neo_hookean_density(grad_u, mat.mu, mat.lmbda)
        return op.integrate(psi)

    @jax.jit
    def total_energy_free(u_free: Array, lf: lifter.Lifter) -> Array:
        sol = Solution(lf.lift_from_zeros(u_free))
        return total_energy(sol)

    sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=3)
    reduced_sparsity = sparse.reduce_sparsity_pattern(
        sparsity_pattern, lifter_.free_dofs
    )

    traced_sparsity = trace_energy_sparsity(
        total_energy_free, lifter_.size_reduced, lifter_
    )

    # The tracer must not miss any coupling present in the FEM element pattern.
    assert nz_set(reduced_sparsity) <= nz_set(traced_sparsity)


def test_periodic_constraint_sparsity():
    """Periodic + Dirichlet constraints: the tracer applied to the lifted (reduced)
    energy must reproduce tatva's own augment/reduce sparsity pipeline exactly."""

    n_x, n_y = 4, 4
    mesh = Mesh.unit_square(n_x, n_y)
    op = Operator(mesh, element.Tri3())

    # Define boundaries
    y_min = np.min(mesh.coords[:, 1])
    bot_nodes = np.where(mesh.coords[:, 1] == y_min)[0]

    left_nodes = np.where(mesh.coords[:, 0] == 0.0)[0]
    left_nodes = left_nodes[np.argsort(mesh.coords[left_nodes, 1])]
    right_nodes = np.where(mesh.coords[:, 0] == 1.0)[0]
    right_nodes = right_nodes[np.argsort(mesh.coords[right_nodes, 1])]

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    my_lifter = lifter.Lifter(
        Solution.size,
        lifter.Fixed(Solution.u[bot_nodes, :], 0.0),
        lifter.Periodic(Solution.u[left_nodes, :], Solution.u[right_nodes, :]),
    )

    sparsity_full = Solution.get_sparsity()
    sparsity_augmented = my_lifter.augment_sparsity(sparsity_full)
    pat_reduced_expected = my_lifter.reduce_sparsity(sparsity_augmented)

    mu, lmbda = 1.0, 1.0

    @autovmap(grad_u=2)
    def neo_hookean_energy(grad_u):
        F = jnp.eye(2) + grad_u
        J = jnp.linalg.det(F)
        C = F.T @ F
        I1 = jnp.trace(C)
        return (mu / 2) * (I1 - 2 - 2 * jnp.log(J)) + (lmbda / 2) * (jnp.log(J) ** 2)

    def local_energy_fn(sol: Solution):
        (u,) = sol
        strain = op.integrate(neo_hookean_energy(op.grad(u)))
        return strain

    @jax.jit
    def local_energy_free(u_free: Array, lf: lifter.Lifter) -> Array:
        sol = Solution(lf.lift_from_zeros(u_free))
        return local_energy_fn(sol)

    pat_reduced_actual = trace_energy_sparsity(
        local_energy_free, my_lifter.size_reduced, my_lifter
    )

    assert nz_set(pat_reduced_actual) == nz_set(pat_reduced_expected)


def test_lagrangian_multiplier_sparsity():
    """Mixed displacement / Lagrange-multiplier adhesion potential: the traced sparsity
    must have no false negatives against the dense reference Hessian."""

    Lx, Ly = 6.0, 2.0
    Nx, Ny = 4, 3
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([xv.flatten(), yv.flatten()], axis=1)

    elements = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            n0 = i * Ny + j
            n1 = (i + 1) * Ny + j
            n2 = i * Ny + (j + 1)
            n3 = (i + 1) * Ny + (j + 1)
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    elements = np.array(elements)

    mesh = Mesh(jnp.asarray(coords), jnp.asarray(elements))

    # Identify horizontal line elements on the bottom boundary (y=0)
    line_elements = []
    for i in range(Nx - 1):
        n0 = i * Ny
        n1 = (i + 1) * Ny
        line_elements.append([n0, n1])
    line_elements = np.array(line_elements)

    op = Operator(mesh, element.Tri3())
    op_line = Operator(replace(mesh, elements=line_elements), element.Line2())

    class Material(NamedTuple):
        mu: float
        lmbda: float

    mat = Material(mu=0.5, lmbda=1.0)

    class Solution(compound.Compound):
        u_rigid = compound.field((3,))
        u = compound.field((mesh.coords.shape))
        lm = compound.field((op_line.mesh.elements.shape[0],))

    @autovmap(grad_u=2)
    def compute_deformation_gradient(grad_u: Array) -> Array:
        return jnp.eye(2) + grad_u

    @autovmap(grad_u=2, mat=None)
    def strain_energy_density(grad_u: Array, mat: Material) -> Array:
        F = compute_deformation_gradient(grad_u)
        C = F.T @ F
        J = jnp.linalg.det(F)
        return (
            mat.mu / 2.0 * (jnp.trace(C) - 2.0)
            - mat.mu * jnp.log(J)
            + (mat.lmbda / 2.0) * (jnp.log(J)) ** 2
        )

    def equality_constraint_potential(lm, u, active_set):
        density = jnp.where(
            active_set[..., None],
            lm[..., None] * op_line.eval(u)[..., 1],
            0.5 * lm[..., None] ** 2,
        )
        return op_line.integrate(density)

    @jax.jit
    def total_lagrangian(z, active_set, mat):
        u_rigid, u, lm = Solution(z)
        psi = strain_energy_density(op.grad(u), mat)
        Psi = op.integrate(psi)
        G = equality_constraint_potential(lm, u, active_set)
        return Psi + G

    # Seed all active constraints
    u_rigid_dummy, u_dummy, lm_dummy = Solution()
    active_set = jnp.ones_like(lm_dummy, dtype=bool)

    energy_fn = lambda z: total_lagrangian(z, active_set, mat)

    # Initialize a small non-zero state to ensure valid Jacobian/Hessian evaluation (det(F) > 0)
    z_dummy = np.random.uniform(-0.02, 0.02, size=Solution.size)
    u_rigid_val, u_val, lm_val = Solution(jnp.array(z_dummy))
    # Ensure u has small perturbations on top of zero to avoid zero determinant or inversion errors
    z_dummy = Solution(u_rigid=u_rigid_val, u=u_val, lm=lm_val).flatten()

    h_dense = jax.hessian(energy_fn)(z_dummy)
    ref_sparsity = np.abs(h_dense) > 1e-12

    pat_traced = trace_energy_sparsity(energy_fn, n_dofs=Solution.size)
    traced_sparsity = pat_traced.toarray() > 0

    fn_mask = ref_sparsity & (~traced_sparsity)
    fn_count = np.sum(fn_mask)

    # Building a colored matrix from the traced pattern must succeed (smoke check).
    colored_matrix = sparse.ColoredMatrix.from_csr(pat_traced)
    assert np.max(colored_matrix.colors) + 1 >= 1

    assert fn_count == 0, (
        "Validation failed: False negatives detected in Adhesion Lagrange Multipliers!"
    )


# ---------------------------------------------------------------------------
# virtual-work tracing (tangent stiffness K = dR/du = d²G/dv du)
# ---------------------------------------------------------------------------


def _virtual_work_reference(vw, n_dofs: int, u_dummy: Array) -> sps.csr_matrix:
    """Dense reference tangent-stiffness pattern: K_ij = d/du_j ( dG/dw_i )."""

    def R_fn(u_flat: Array) -> Array:
        return jax.grad(vw, argnums=0)(jnp.zeros(n_dofs), u_flat)

    K_exact = jax.jacobian(R_fn)(u_dummy)
    return sps.csr_matrix(np.abs(np.asarray(K_exact)) > 1e-12, dtype=np.int8)


def _assert_traced_matches(pat, K_pat, lf: lifter.Lifter, exact: bool = True) -> None:
    """Validate the traced sparsity against the reference, both full and lifter-reduced.

    The tracer guarantees no false negatives (traced ⊇ reference). For forms where it is
    also tight (no spurious couplings) we assert exact equality; for conservative cases
    (e.g. purely bilinear saddle-point forms) we only require the superset property.
    """
    free = np.asarray(lf.free_dofs)
    K_reduced = sps.csr_matrix(K_pat.toarray()[np.ix_(free, free)])
    if exact:
        assert nz_set(pat) == nz_set(K_pat)
        assert nz_set(lf.reduce_sparsity(pat)) == nz_set(K_reduced)
    else:
        # no false negatives: every reference coupling must be captured
        assert nz_set(K_pat) <= nz_set(pat)
        assert nz_set(K_reduced) <= nz_set(lf.reduce_sparsity(pat))


def test_advection_diffusion_virtual_work_sparsity(line_operator):
    """Advection-Diffusion (non-symmetric coupling) discretized on a 1-D Line2 mesh."""
    N = 8
    D = 1.0  # Diffusion coeff
    V = 1.5  # Advection velocity
    op = line_operator(N)

    class Field(compound.Compound, mesh=op.mesh):
        u = compound.field((FieldSize.AUTO, 1))

    @jax.jit
    def g_fn(w, u):
        (u_n,) = Field(u)
        (w_n,) = Field(w)
        du = op.grad(u_n)
        dw = op.grad(w_n)
        w_q = op.eval(w_n)
        return op.integrate(D * du * dw + V * du * w_q).sum()

    my_lifter = lifter.Lifter(Field.size, lifter.Fixed(Field.u[jnp.array([0]), :]))

    pat = trace_virtual_work_sparsity(g_fn, Field.size, trial_arg="u", test_arg="w")
    u_dummy = jnp.arange(Field.size, dtype=jnp.float64)
    K_pat = _virtual_work_reference(g_fn, Field.size, u_dummy)

    _assert_traced_matches(pat, K_pat, my_lifter)


def test_nonlinear_coupling_virtual_work_sparsity(line_operator):
    """Nonlinear reaction-diffusion virtual work (nonlinear coupling via w * u**3)."""
    N = 6
    op = line_operator(N)

    class Field(compound.Compound, mesh=op.mesh):
        u = compound.field((FieldSize.AUTO, 1))

    @jax.jit
    def g_fn(w, u):
        (u_n,) = Field(u)
        (w_n,) = Field(w)
        du = op.grad(u_n)
        dw = op.grad(w_n)
        u_q = op.eval(u_n)
        w_q = op.eval(w_n)
        return op.integrate(du * dw + w_q * u_q**3).sum()

    my_lifter = lifter.Lifter(Field.size, lifter.Fixed(Field.u[jnp.array([0]), :]))

    pat = trace_virtual_work_sparsity(g_fn, Field.size, trial_arg="u", test_arg="w")
    # Avoid the trivial derivative of u**3 at zero.
    u_dummy = jnp.arange(N, dtype=jnp.float64) + 1.0
    K_pat = _virtual_work_reference(g_fn, Field.size, u_dummy)

    _assert_traced_matches(pat, K_pat, my_lifter)


def test_mixed_nodal_virtual_work_sparsity(line_operator):
    """Mixed velocity-pressure saddle-point virtual work (block structure via Compound)."""
    N = 5
    op = line_operator(N)

    class Mixed(compound.Compound, mesh=op.mesh):
        v = compound.field((FieldSize.AUTO, 1))
        p = compound.field((FieldSize.AUTO, 1))

    @jax.jit
    def g_fn(test, trial):
        v, p = Mixed(trial)
        w, q = Mixed(test)
        dv = op.grad(v)
        dw = op.grad(w)
        p_q = op.eval(p)
        q_q = op.eval(q)
        elastic = op.integrate(dw * dv).sum()
        coupling1 = op.integrate(dw * p_q).sum()
        coupling2 = op.integrate(q_q * dv).sum()
        return elastic + coupling1 + coupling2

    my_lifter = lifter.Lifter(Mixed.size, lifter.Fixed(Mixed.v[jnp.array([0]), :]))

    pat = trace_virtual_work_sparsity(
        g_fn, Mixed.size, trial_arg="trial", test_arg="test"
    )
    u_dummy = jnp.arange(Mixed.size, dtype=jnp.float64)
    K_pat = _virtual_work_reference(g_fn, Mixed.size, u_dummy)

    # The mixed bilinear form is traced conservatively (extra within-node v-p partner
    # couplings), so we require the no-false-negative guarantee rather than exact equality.
    _assert_traced_matches(pat, K_pat, my_lifter, exact=False)


def test_unsymmetric_sparsity(line_operator):
    """Two-field one-way coupling (a's residual sees b, but b's never sees a), the
    hallmark of a non-associated tangent: the traced pattern must be genuinely
    unsymmetric and contain every coupling of the unsymmetric reference Jacobian."""
    N = 6
    op = line_operator(N)

    class TwoField(compound.Compound, mesh=op.mesh):
        a = compound.field((FieldSize.AUTO, 1))
        b = compound.field((FieldSize.AUTO, 1))

    @jax.jit
    def g_fn(test, trial):
        a, b = TwoField(trial)
        wa, wb = TwoField(test)
        diffusion = op.integrate(op.grad(wa) * op.grad(a)).sum()
        diffusion += op.integrate(op.grad(wb) * op.grad(b)).sum()
        one_way = op.integrate(op.eval(wa) * op.grad(b)).sum()  # a-residual depends on b
        return diffusion + one_way

    my_lifter = lifter.Lifter(TwoField.size, lifter.Fixed(TwoField.a[jnp.array([0]), :]))

    pat = trace_virtual_work_sparsity(
        g_fn, TwoField.size, trial_arg="trial", test_arg="test"
    )
    u_dummy = jnp.arange(TwoField.size, dtype=jnp.float64)
    K_pat = _virtual_work_reference(g_fn, TwoField.size, u_dummy)

    # The one-way coupling must yield a genuinely unsymmetric pattern (otherwise the test
    # would not exercise the non-symmetric tangent path of the virtual-work tracer).
    assert nz_set(K_pat) != nz_set(K_pat.T)
    assert nz_set(pat) != nz_set(pat.T)

    # Two interleaved nodal fields are traced conservatively, so require no false negatives.
    _assert_traced_matches(pat, K_pat, my_lifter, exact=False)
