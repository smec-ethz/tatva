from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps

if TYPE_CHECKING:
    # `mpi4py` is an optional dependency (the `mpi` extra), and this module is imported
    # by `tatva.sparse.__init__`, so importing it at runtime would make MPI mandatory for
    # the whole sparse package. `from __future__ import annotations` keeps the `MPI.Comm`
    # annotation below a string, so it is never evaluated; the only runtime import lives
    # inside `_allreduce_union`, which is reached only when a communicator was supplied.
    from mpi4py import MPI

from tatva.operator import Operator
from tatva.sparse.tracer import pattern_from_energy as _serial_pattern_from_energy


def element_bounds(n_elements: int, n_chunks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_elements, n_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _chunk_operator(op, comm=None):
    """An Operator over this rank's contiguous slice of ``op.mesh.elements``, with coords
    and DOF numbering intact.

    Each rank gets exactly **one contiguous** chunk, so the chunk count is always
    ``comm.size`` and there is nothing to configure. Contiguity is deliberate: it keeps
    the DOFs touched by different ranks nearly disjoint, so the coordinate exchange in
    the final union carries about ``nnz_global`` entries rather than ``size *
    nnz_global``. Bounds come from *this* operator's own cell count, so operators with
    different cell counts are each partitioned over their own range.

    An operator with fewer cells than ranks is replicated whole onto every rank instead
    of being split. That is safe rather than merely tolerable -- the union is idempotent,
    so tracing a term on every rank gives exactly the same pattern as tracing it once --
    and it avoids handing ``Operator`` an empty cell list, which ``__check_init__``
    rejects.
    """

    n_elements = int(op.mesh.elements.shape[0])
    size = comm.size if comm is not None else 1
    rank = comm.rank if comm is not None else 0

    if n_elements < size:
        # Too small to split: replicate whole.
        lo, hi = 0, n_elements
    else:
        lo, hi = element_bounds(n_elements, size)[rank]

    subelements = op.mesh.elements[lo:hi]
    replaced_mesh = op.mesh._replace(elements=subelements)

    return op.__class__(
        replaced_mesh,
        op.element,
        batch_size=min(op.batch_size, hi - lo) if op.batch_size else None,
        cache_weights=op.cache_weights,
    )


def _get_operators_from_args(static_args):
    """Extract Operator instances from static_args."""
    return [arg for arg in static_args if isinstance(arg, Operator)]


def _allreduce_union(pattern: sps.csr_matrix, comm, n_dofs: int) -> sps.csr_matrix:
    """OR-reduce a replicated pattern across ranks; every rank gets the full result.

    ``MPI.SUM`` cannot be applied to CSR matrices through the *buffer* interface: that
    reduces fixed-layout buffers elementwise, and two ranks' matrices have different
    ``nnz``, ``indices`` and ``indptr``, so there is no positional correspondence to sum.
    The only common layout is dense, which costs ``n_dofs**2``.

    So the entries are moved rather than reduced. Each ``(row, col)`` is packed into one
    ``int64`` key, the keys are exchanged with a single ``Allgatherv``, and the union falls
    out of rebuilding a CSR from the concatenation. Duplicates -- entries on chunk
    interfaces, contributed by more than one rank -- are collapsed by ``sum_duplicates``
    during the COO to CSR conversion, which has to happen anyway, so no separate
    deduplication pass is needed. The data is then clamped back to a binary pattern.

    mpi4py's *object* allreduce (``comm.allreduce(pattern, op=MPI.SUM)``) also works, by
    pickling the matrices and applying Python ``+``, and is roughly 3x faster at moderate
    size. It is not used here because pickled messages hit a ~2 GB ceiling in some MPI
    builds, which at ~5 bytes of CSR per entry puts a hard wall around 25M DOFs for a
    typical nnz/DOF ratio. The buffer interface has no byte ceiling -- MPI counts are
    32-bit *element* counts, so one ``int64`` key per entry allows ~2**31 entries per
    message, which at the same ratio is past 100M DOFs and well beyond the point where
    holding the result at all becomes the binding constraint.

    """
    from mpi4py import MPI

    if n_dofs > int(np.sqrt(np.iinfo(np.int64).max)):
        raise ValueError(
            f"n_dofs={n_dofs} is too large to pack (row, col) pairs into int64 keys."
        )

    row, col = pattern.nonzero()
    keys = row.astype(np.int64) * n_dofs + col.astype(np.int64)

    counts = np.array(comm.allgather(keys.size), dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return sps.csr_matrix((n_dofs, n_dofs), dtype=np.int8)

    displ = np.concatenate(([0], np.cumsum(counts)[:-1]))
    gathered = np.empty(total, dtype=np.int64)
    comm.Allgatherv(
        [keys, MPI.INT64_T],
        [gathered, (counts.astype(np.int32), displ.astype(np.int32)), MPI.INT64_T],
    )
    del keys

    rows, cols = np.divmod(gathered, n_dofs)
    del gathered
    out = sps.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_dofs, n_dofs)
    )
    out.sum_duplicates()
    if out.nnz:
        out.data[:] = 1
    return out


def pattern_from_energy(
    energy: Callable, n_dofs: int, *static_args, comm: MPI.Comm | None = None
) -> sps.csr_matrix:
    """Trace the Hessian pattern of an energy function, optionally across MPI ranks.

    With ``comm=None`` this is exactly the serial tracer. With a communicator, every rank
    holds the full mesh and global DOF numbering, each ``Operator`` in ``static_args`` is
    restricted to that rank's contiguous slice of its own cells, and the per-rank patterns
    are OR-reduced -- so this distributes the *tracing work*, not the mesh.

    The signature is deliberately a superset of the serial
    ``tatva.sparse.tracer.pattern_from_energy``: ``n_dofs`` stays positional so this can
    stand in for it directly, with ``comm`` as the only addition.

    Only energies that are a sum over cells of a cell-local density decompose this way. A
    nonlinear function of a *global* reduction -- ``jnp.sum(op.integrate(g)) ** 2``, a
    global volume constraint, a homogenization term -- does not: chunking turns
    ``(sum_p I_p)**2`` into ``sum_p I_p**2`` and the cross-chunk couplings vanish
    silently. Trace those serially.

    Args:
        energy: ``energy(u, *static_args)`` returning a scalar. ``u`` must be the first
            argument -- the tracer seeds ``jaxpr.invars[0]`` as the DOF vector.
        n_dofs: global DOF count -- identical on every rank, and never derived from a
            chunk.
        static_args: extra arguments, treated as constants. Operators found here at the
            top level are the ones that get chunked, so the energy must take them as
            arguments rather than closing over them.
        comm: an ``mpi4py`` communicator. Every rank returns the same complete pattern.

    Returns:
        A CSR matrix of shape ``(n_dofs, n_dofs)`` with binary entries.
    """

    if comm is None:
        # No MPI communicator provided; trace the energy function directly.
        return _serial_pattern_from_energy(energy, n_dofs, *static_args)

    # Extract Operator instances from static_args
    ops = _get_operators_from_args(static_args)
    if not ops:
        raise ValueError(
            "No Operator instances found in static_args. Cannot chunk the energy function for parallel tracing."
        )

    # Restrict every Operator to this rank's cells. Each is partitioned over its own cell
    # count, so operators over different meshes (bulk + interface) need not agree in size.
    chunked_ops = [_chunk_operator(op, comm=comm) for op in ops]

    # Replace the original Operators in static_args with their chunked versions.
    chunked_iter = iter(chunked_ops)
    chunked_static_args = [
        next(chunked_iter) if isinstance(arg, Operator) else arg for arg in static_args
    ]

    # Trace this rank's chunk of the energy function
    local_pattern = _serial_pattern_from_energy(energy, n_dofs, *chunked_static_args)

    return _allreduce_union(local_pattern, comm, n_dofs)
