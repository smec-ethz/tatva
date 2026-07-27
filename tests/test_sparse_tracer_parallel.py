"""Accuracy tests for the element-chunked parallel sparsity tracer.

The tracer restricts every ``Operator`` to a contiguous slice of its own cells on a
*replicated* mesh, traces the energy once per rank, and OR-reduces the patterns. It is
exact rather than conservative because ``H(sum_c E_c) = sum_c H(E_c)``, the pattern of a
sum is the union of the patterns, and every handler in ``tatva.sparse.tracer`` is monotone
in support. So the parallel pattern must equal the serial one entry for entry.

Coverage comes at two levels:

- Most tests drive ``pattern_from_energy`` with :class:`_StubComm`, which exercises the
  real chunking, operator substitution and union code for any rank count *inside a single
  process*. The decomposition is the interesting part and this is where it is tested, so
  it runs under a plain ``pytest`` invocation instead of only under ``mpirun``.
- :func:`test_matches_serial_under_mpi` uses the real ``MPI.COMM_WORLD`` and the real
  collective. It is a consistency check on one rank and gains genuine multi-rank coverage
  when run as::

      mpirun -n 4 python -m pytest tests/test_sparse_tracer_parallel.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps
from jax_autovmap import autovmap

from tatva import Mesh, Operator, compound, element, lifter
from tatva.compound import FieldSize
from tatva.sparse.parallel_tracer import _chunk_operator, pattern_from_energy
from tatva.sparse.tracer import pattern_from_energy as serial_pattern_from_energy

jax.config.update("jax_enable_x64", True)

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# `_allreduce_union` refers to `MPI.SUM` even when driven by a stub communicator, so the
# whole module needs mpi4py importable.
pytestmark = pytest.mark.skipif(not HAS_MPI, reason="mpi4py required")

RANK_COUNTS = [1, 2, 3, 4, 8]


def nz_set(m: sps.spmatrix) -> set[tuple[int, int]]:
    """Convert a sparse matrix to a set of ``(row, col)`` nonzero index tuples."""
    r, c = m.nonzero()  # ty:ignore[unresolved-attribute]
    return set(zip(r.tolist(), c.tolist()))


class _StubComm:
    """Stands in for one rank of a ``size``-rank job, without launching MPI.

    ``allreduce`` returns its argument untouched, so ``pattern_from_energy`` hands back
    this rank's *local* pattern rather than the reduced one. Unioning those across all
    ranks (see :func:`parallel_pattern`) reproduces exactly what the real allreduce
    computes -- which is what lets the decomposition be tested in-process, at rank counts
    that would otherwise need a separate ``mpirun`` per case.
    """

    def __init__(self, size: int, rank: int):
        self.size = size
        self.rank = rank

    def allreduce(self, obj, op=None):
        return obj


def parallel_pattern(energy, *static_args, n_dofs: int, size: int) -> sps.csr_matrix:
    """Union of the per-rank local patterns for a ``size``-rank job."""
    out = sps.csr_matrix((n_dofs, n_dofs), dtype=np.int8)
    for rank in range(size):
        out = out + pattern_from_energy(
            energy, n_dofs, *static_args, comm=_StubComm(size, rank)
        )
    out = out.tocsr()
    if out.nnz:
        out.data[:] = 1
    return out


@autovmap(grad_u=2)
def neo_hookean_2d(grad_u):
    F = jnp.eye(2) + grad_u
    J = jnp.linalg.det(F)
    return 0.5 * (jnp.trace(F.T @ F) - 2 - 2 * jnp.log(J)) + 0.5 * jnp.log(J) ** 2


# ---------------------------------------------------------------------------
# cases, mirroring the serial tracer suite
# ---------------------------------------------------------------------------


def case_unconstrained(n=12):
    """Plain neo-Hookean energy over a Tri3 mesh, no constraints."""
    mesh = Mesh.unit_square(n, n)
    op = Operator(mesh, element.Tri3())

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    def energy(u_flat, op):
        (u,) = Solution(u_flat)
        return op.integrate(neo_hookean_2d(op.grad(u)))

    return energy, (op,), Solution.size


def case_fixed_and_periodic(n=12):
    """Fixed + Periodic constraints, traced through the lifter.

    Mirrors ``test_periodic_constraint_sparsity``: the constraint structure is folded into
    the pattern by tracing the reduced energy. The lift is affine, so couplings are
    remapped rather than created, and the chunk union must still be exact.
    """
    mesh = Mesh.unit_square(n, n)
    op = Operator(mesh, element.Tri3())

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    coords = np.asarray(mesh.coords)
    bot = np.where(coords[:, 1] == coords[:, 1].min())[0]
    left = np.where(coords[:, 0] == 0.0)[0]
    left = left[np.argsort(coords[left, 1])]
    right = np.where(coords[:, 0] == 1.0)[0]
    right = right[np.argsort(coords[right, 1])]

    lf = lifter.Lifter(
        Solution.size,
        lifter.Fixed(Solution.u[bot, :], 0.0),
        lifter.Periodic(Solution.u[left, :], Solution.u[right, :]),
    )

    def energy(u_free, lf, op):
        (u,) = Solution(lf.lift_from_zeros(u_free))
        return op.integrate(neo_hookean_2d(op.grad(u)))

    return energy, (lf, op), lf.size_reduced


def case_two_operators(n=12, n_iface=6):
    """A bulk operator plus a small second operator over a subset of the same cells.

    Covers the two things a single shared cell range would get wrong: each operator must
    be partitioned over *its own* cell count, and an operator with fewer cells than ranks
    must be replicated whole rather than sliced into empty pieces. ``n_iface`` is small
    enough that the higher rank counts in ``RANK_COUNTS`` take the replicate branch.

    A non-Operator argument sits between the two operators so the substitution is also
    checked against positional drift.
    """
    mesh = Mesh.unit_square(n, n)
    op_bulk = Operator(mesh, element.Tri3())
    sub = mesh._replace(elements=mesh.elements[:n_iface])
    op_small = Operator(sub, element.Tri3())

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    def energy(u_flat, scale, op_bulk, op_small):
        (u,) = Solution(u_flat)
        bulk = op_bulk.integrate(neo_hookean_2d(op_bulk.grad(u)))
        extra = op_small.integrate(neo_hookean_2d(op_small.grad(u)))
        return bulk + scale * extra

    return energy, (2.0, op_bulk, op_small), Solution.size


ALL_CASES = {
    "unconstrained": case_unconstrained,
    "fixed_and_periodic": case_fixed_and_periodic,
    "two_operators": case_two_operators,
}


# ---------------------------------------------------------------------------
# cell partitioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", RANK_COUNTS)
def test_chunk_operator_tiles_cells_exactly(size):
    """Every cell must land in exactly one rank's chunk -- no gaps, no overlap.

    A gap would drop couplings, which is the dangerous direction and invisible in the
    final pattern; an overlap only costs redundant work.
    """
    mesh = Mesh.unit_square(8, 8)
    op = Operator(mesh, element.Tri3())
    n_cells = int(op.mesh.elements.shape[0])

    seen = []
    for rank in range(size):
        chunk = _chunk_operator(op, comm=_StubComm(size, rank))
        seen.append(np.asarray(chunk.mesh.elements))

    covered = np.concatenate(seen)
    expected = np.asarray(op.mesh.elements)
    assert covered.shape[0] == n_cells
    assert np.array_equal(covered, expected), "chunks must tile the cells in order"


@pytest.mark.parametrize("size", RANK_COUNTS)
def test_chunk_operator_preserves_dof_space(size):
    """Restriction cuts the cell axis only; coords and DOF numbering are invariant.

    If a chunk renumbered or compacted its DOFs, each rank's dependency-matrix columns
    would mean something different and the union would silently mix index spaces.
    """
    mesh = Mesh.unit_square(8, 8)
    op = Operator(mesh, element.Tri3())
    for rank in range(size):
        chunk = _chunk_operator(op, comm=_StubComm(size, rank))
        assert chunk.mesh.coords.shape == op.mesh.coords.shape
        assert np.array_equal(
            np.asarray(chunk.mesh.coords), np.asarray(op.mesh.coords)
        )
        assert chunk.mesh.elements.shape[1] == op.mesh.elements.shape[1]


def test_chunk_operator_replicates_operator_smaller_than_ranks():
    """An operator with fewer cells than ranks is replicated whole.

    Slicing it would hand ``Operator`` an empty cell list, which ``__check_init__``
    rejects. Replication is exact because the union is idempotent.
    """
    mesh = Mesh.unit_square(2, 1)
    op = Operator(mesh, element.Tri3())
    n_cells = int(op.mesh.elements.shape[0])
    size = n_cells + 3

    for rank in range(size):
        chunk = _chunk_operator(op, comm=_StubComm(size, rank))
        assert int(chunk.mesh.elements.shape[0]) == n_cells


def test_chunk_operator_serial_is_whole_mesh():
    mesh = Mesh.unit_square(6, 6)
    op = Operator(mesh, element.Tri3())
    chunk = _chunk_operator(op, comm=None)
    assert np.array_equal(
        np.asarray(chunk.mesh.elements), np.asarray(op.mesh.elements)
    )


# ---------------------------------------------------------------------------
# accuracy against the serial tracer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(ALL_CASES))
@pytest.mark.parametrize("size", RANK_COUNTS)
def test_matches_serial_tracer(case_name, size):
    """The chunk union must equal the serial pattern exactly, not merely cover it."""
    energy, static_args, n_dofs = ALL_CASES[case_name]()

    ref = serial_pattern_from_energy(energy, n_dofs, *static_args)
    par = parallel_pattern(energy, *static_args, n_dofs=n_dofs, size=size)

    assert nz_set(par) == nz_set(ref)


@pytest.mark.parametrize("case_name", sorted(ALL_CASES))
@pytest.mark.parametrize("size", RANK_COUNTS)
def test_pattern_is_binary(case_name, size):
    """Entries on chunk interfaces are contributed by more than one rank, so the union
    must clamp back to 1. Without the clamp those entries come out as 2, which breaks
    anything reading ``.data`` (``ColoredMatrix.from_csr``) rather than ``nonzero()``."""
    energy, static_args, n_dofs = ALL_CASES[case_name]()
    par = parallel_pattern(energy, *static_args, n_dofs=n_dofs, size=size)
    assert par.nnz > 0
    assert int(par.data.max()) == 1


def test_serial_path_matches_serial_tracer():
    """``comm=None`` delegates straight to the serial tracer."""
    energy, static_args, n_dofs = case_fixed_and_periodic()
    ref = serial_pattern_from_energy(energy, n_dofs, *static_args)
    got = pattern_from_energy(energy, n_dofs, *static_args, comm=None)
    assert nz_set(got) == nz_set(ref)


def test_raises_when_no_operator_in_static_args():
    """Without an Operator there is nothing to chunk, and silently tracing the whole
    energy on every rank would look like it worked while doing size-times the work."""
    mesh = Mesh.unit_square(4, 4)
    op = Operator(mesh, element.Tri3())

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    def energy(u_flat):  # closes over `op` instead of taking it as an argument
        (u,) = Solution(u_flat)
        return op.integrate(neo_hookean_2d(op.grad(u)))

    with pytest.raises(ValueError, match="No Operator"):
        pattern_from_energy(
            energy, n_dofs=Solution.size, comm=_StubComm(2, 0)
        )


# ---------------------------------------------------------------------------
# documented limitation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 4])
def test_global_reduction_energy_loses_couplings(size):
    """A nonlinear function of a *global* reduction is not cell-additive, and chunking
    silently drops its cross-chunk couplings.

    Splitting the cells turns ``(sum_p I_p)**2`` into ``sum_p I_p**2``, so the products
    between different chunks' contributions vanish. This test pins the current behaviour
    rather than endorsing it: such terms must be traced once over the full cell set. If an
    escape hatch is added, this test should start failing and be replaced by one asserting
    the pattern is complete.
    """
    mesh = Mesh.unit_square(8, 8)
    op = Operator(mesh, element.Tri3())

    class Solution(compound.Compound, mesh=mesh):
        u = compound.field((FieldSize.AUTO, 2))

    def energy(u_flat, op):
        (u,) = Solution(u_flat)
        total = jnp.sum(op.integrate(op.grad(u)[..., 0, 0]))
        return total**2

    n_dofs = Solution.size
    ref = serial_pattern_from_energy(energy, n_dofs, op)
    par = parallel_pattern(energy, op, n_dofs=n_dofs, size=size)

    missing = nz_set(ref) - nz_set(par)
    assert missing, "expected the global-reduction couplings to be lost when chunked"
    # Never the other direction: chunking must not invent couplings.
    assert not (nz_set(par) - nz_set(ref))


# ---------------------------------------------------------------------------
# real MPI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(ALL_CASES))
def test_matches_serial_under_mpi(case_name):
    """Same assertion as :func:`test_matches_serial_tracer`, but through the real
    collective. Trivial on one rank; run under ``mpirun -n 4`` for real coverage."""
    comm = MPI.COMM_WORLD
    energy, static_args, n_dofs = ALL_CASES[case_name]()

    ref = serial_pattern_from_energy(energy, n_dofs, *static_args)
    par = pattern_from_energy(energy, n_dofs, *static_args, comm=comm)

    # Every rank must end up with the same, complete pattern.
    assert nz_set(par) == nz_set(ref)
    assert int(par.data.max()) == 1

    n_disagree = comm.allreduce(0 if nz_set(par) == nz_set(ref) else 1, op=MPI.SUM)
    assert n_disagree == 0
