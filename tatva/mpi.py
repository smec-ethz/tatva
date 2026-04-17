from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
from jax import Array
from mpi4py import MPI
from scipy.sparse import csr_matrix

from tatva import Mesh, sparse
from tatva.lifter import Lifter
from tatva.sparse import ColoredMatrix


def _dof_range(n_dofs: int, n_ranks: int, r: int) -> tuple[int, int]:
    """Block DOF distribution matching PETSc's default.

    The first (n_dofs % n_ranks) ranks each get one extra DOF so that
    all rows are covered with minimal imbalance.

    Args:
        n_dofs:   total number of DOFs
        n_ranks:  total number of MPI ranks
        r:        rank for which to compute the range

    Returns:
        (rstart, rend): global DOF index range [rstart, rend) owned by rank r
    """
    base = n_dofs // n_ranks
    extra = n_dofs % n_ranks
    rstart = r * base + min(r, extra)
    rend = rstart + base + (1 if r < extra else 0)
    return rstart, rend


@dataclass
class PartitionInfo:
    """Data structure to hold partitioning information for the local mesh."""

    active_nodes: np.ndarray
    owned_nodes_mask: np.ndarray
    nodes_global_to_local: np.ndarray


@dataclass
class _LocalProblemContext:
    """Extra context set only when constructing via from_local_problem.

    Required by zero_ghost_values and gather; not needed for the core
    routing tables.
    """

    active_dofs: np.ndarray  # global reduced DOF indices of active nodes
    free_dofs_local: np.ndarray  # indices into active space (lifter.free_dofs)
    partition_info: PartitionInfo
    n_dofs_per_node: int
    n_nodes_global: int


class NeighborExchangePlan:
    """Precomputed MPI communication plan for neighbor-exchange parallel FEM assembly.

    Encapsulates all routing tables for three operations:
      - Ghost gather:      distribute x_owned → u_active (all active DOFs)
      - Gradient assembly: u_active contributions → owned gradient rows
      - Hessian assembly:  u_active contributions → owned sparse matrix entries
    """

    def __init__(
        self,
        global_mesh: Mesh,
        partition_info: PartitionInfo,
        n_dofs_per_node: int,
        local_colored_matrix: ColoredMatrix,
        lifter: Lifter,
        comm: MPI.Comm,
    ):
        """Build the neighbor-exchange plan from local mesh artefacts.

        Args:
            global_mesh:        full global mesh (replicated on all ranks)
            partition_info:  partition info for this rank's local mesh
            n_dofs_per_node: DOFs per mesh node
            local_colored_matrix:   ColoredMatrix built from the local mesh
            lifter:          Lifter for this rank's local problem
            comm:            MPI communicator
        """
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._neighbor_nnz_data = []
        self._neighbor_dof_data = []

        # Compute global DOF indices of this rank's active DOFs (full space, not reduced by lifter).
        active_dofs = np.sort(
            np.concatenate(
                [
                    partition_info.active_nodes * n_dofs_per_node + d
                    for d in range(n_dofs_per_node)
                ]
            )
        )

        # Identify the global DOF indices of all fixed DOFs on this rank
        local_fixed_global = active_dofs[np.asarray(lifter.constrained_dofs)]
        # gather fixed DOF indices from all ranks to identify the global set of free DOFs
        all_fixed = comm.allgather(local_fixed_global)
        global_fixed_indices = np.unique(np.concatenate(all_fixed))
        # compute the global free dofs
        n_dofs_total = global_mesh.coords.shape[0] * n_dofs_per_node
        free_dofs_global = np.setdiff1d(
            np.arange(n_dofs_total), global_fixed_indices, assume_unique=True
        )

        global_sparsity = sparse.create_sparsity_pattern(global_mesh, n_dofs_per_node)
        global_free_sparsity = sparse.reduce_sparsity_pattern(
            global_sparsity, free_dofs_global
        )

        # compute active and free global DOF indices corresponding to this rank's local active DOFs
        active_free_global_full = active_dofs[np.array(lifter.free_dofs)]
        # reduced global Dofs
        active_free_global_reduced = np.searchsorted(
            np.asarray(free_dofs_global), active_free_global_full
        ).astype(np.int64)

        self._precompute_routing_tables(
            local_colored_matrix, active_free_global_reduced, global_free_sparsity
        )

        # Cache owned-row CSR (used for solver preallocation and assembly)
        indptr_g = np.asarray(global_free_sparsity.indptr)
        indices_g = np.asarray(global_free_sparsity.indices)
        self._owned_ptr = (
            indptr_g[self._rstart : self._rend + 1] - indptr_g[self._rstart]
        ).astype(np.int32)
        self._owned_indices = indices_g[
            indptr_g[self._rstart] : indptr_g[self._rend]
        ].astype(np.int32)

        self._local_context = _LocalProblemContext(
            active_dofs=active_free_global_reduced,
            free_dofs_local=np.array(lifter.free_dofs),
            partition_info=partition_info,
            n_dofs_per_node=n_dofs_per_node,
            n_nodes_global=global_mesh.coords.shape[0],
        )

    @property
    def global_size(self) -> int:
        """Total number of free DOFs in the global linear system."""
        return self._n_free

    @property
    def rstart(self) -> int:
        """First owned DOF index."""
        return self._rstart

    @property
    def rend(self) -> int:
        """One-past-last owned DOF index."""
        return self._rend

    @property
    def local_size(self) -> int:
        """Number of DOFs owned by this rank under the block distribution.

        Computed by the same block-range formula used to build the routing
        tables. Pass this as the explicit local size when creating distributed
        solver objects so their row distribution matches this plan — do not let
        the solver pick its own default.

        Example (PETSc):
            J = PETSc.Mat().createAIJ(
                [(plan.local_size, n_free), (plan.local_size, n_free)],
                comm=comm
            )
        """
        rstart, rend = self._dof_range(self._n_free, self._size, self._rank)
        return rend - rstart

    @property
    def owned_nnz(self) -> int:
        """Number of sparse matrix entries owned by this rank."""
        return self._owned_nnz

    @property
    def owned_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """(indptr, indices) for owned rows — pass directly to PETSc setPreallocationCSR."""
        return self._owned_ptr, self._owned_indices

    # ------------------------------------------------------------------
    # Assembled-quantity helper
    # ------------------------------------------------------------------

    def zero_ghost_values(self, nodal_array: Array) -> Array:
        """Zero out ghost-DOF entries of any quantity assembled via scatter-add.

        Operates in the full active DOF space (length n_active).  Ghost free DOFs
        — nodes shared with a neighbouring rank — are set to zero so that the
        scatter-add in make_scatter_rev_add counts each contribution exactly once.  Fixed
        DOFs are left unchanged (they are already zero in typical load vectors).

        Use this for quantities built by summing contributions across ranks:
        external forces, body-force vectors, reaction forces, source terms.

        Do NOT use for state variables (displacement, temperature, ...) that flow
        in the opposite direction: those are filled from owning ranks by
        gather_ghost_values, and zeroing them would corrupt local element computations at
        partition boundaries.

        Requires construction via from_local_problem (which stores the local
        free-DOF index map needed to identify ghost entries in the full space).

        Args:
            nodal_array: array of length n_active (full local active DOF space)

        Returns:
            same shape, ghost free-DOF entries set to zero
        """
        if self._local_context is None:
            raise ValueError(
                "zero_ghost_values requires construction via from_local_problem."
            )
        ctx = self._local_context
        owned_mask = np.asarray(
            (ctx.active_dofs >= self._rstart) & (ctx.active_dofs < self._rend)
        )
        ghost_free_dofs = jnp.array(ctx.free_dofs_local[~owned_mask])
        return jnp.asarray(nodal_array).at[ghost_free_dofs].set(0.0)

    # ------------------------------------------------------------------
    # JIT'd function factories
    # ------------------------------------------------------------------

    def make_scatter_fwd_set(self, n_free_local: int) -> Callable[[Array], Array]:
        """Return a JIT'd function: x_owned → u_free_local.

        Gathers ghost DOF values from owning ranks so every rank has the full
        local free-DOF vector needed for local FEM computation.

        This is the transpose of make_scatter_rev_add's assembly step:
          assembly:     ghost contributors SEND  → owners RECEIVE (scatter-add)
          ghost gather: owners             SEND  → ghosts RECEIVE (scatter-set)

        Args:
            n_free_local: length of the output vector; pass lifter.size_reduced.
        """

        self_send = jnp.array(self._send_dof, dtype=jnp.int64)
        self_recv = jnp.array(self._recv_dof, dtype=jnp.int64)
        has_self = len(self._send_dof) > 0
        # Roles swap vs assembly: send ← recv_local_idx, recv ← local_send_idx
        nbr_send = [
            jnp.array(d["recv_local_idx"], dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        nbr_recv = [
            jnp.array(d["local_send_idx"], dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        neighbor_dof_data = self._neighbor_dof_data
        _comm = self._comm

        @jax.jit
        def fn(x_owned: Array) -> Array:
            u_free_local = jnp.zeros(n_free_local)
            if has_self:
                u_free_local = u_free_local.at[self_send].set(x_owned[self_recv])
            for i, info in enumerate(neighbor_dof_data):
                nbr = np.int32(info["rank"])
                send_buf = x_owned[nbr_send[i]]
                recv_tmpl = jnp.zeros(info["send_size"], dtype=x_owned.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=_comm
                )
                if info["send_size"] > 0:
                    u_free_local = u_free_local.at[nbr_recv[i]].set(recv_vals)
            return u_free_local

        return fn

    def make_scatter_rev_add(
        self, local_fn: Callable, is_hessian: bool = False
    ) -> Callable:
        """Return a JIT'd function: u_active → owned_data.

        Computes the local function (gradient or hessian), then gathers contributions
        from ghost rows into owned rows via point-to-point exchange.

        Args:
            local_fn: function returning either a vector (for gradient) or ColoredMatrix (for Hessian)
            is_hessian: whether the local_fn returns a Hessian (ColoredMatrix) or gradient (vector).
            This determines which routing tables to use for assembly.
        """

        if is_hessian:
            self_send = jnp.array(self._local_send_idx, dtype=jnp.int64)
            self_recv = jnp.array(self._recv_local_idx, dtype=jnp.int64)
            has_self = len(self._local_send_idx) > 0

            output_size = self.owned_nnz
            nbr_send = [
                jnp.array(d["local_send_idx"], dtype=jnp.int64)
                for d in self._neighbor_nnz_data
            ]
            nbr_recv = [
                jnp.array(d["recv_local_idx"], dtype=jnp.int64)
                for d in self._neighbor_nnz_data
            ]
            neighbor_routing_data = self._neighbor_nnz_data

        else:
            self_send = jnp.array(self._send_dof, dtype=jnp.int64)
            self_recv = jnp.array(self._recv_dof, dtype=jnp.int64)
            has_self = len(self._send_dof) > 0

            output_size = self.local_size
            nbr_send = [
                jnp.array(d["local_send_idx"], dtype=jnp.int64)
                for d in self._neighbor_dof_data
            ]
            nbr_recv = [
                jnp.array(d["recv_local_idx"], dtype=jnp.int64)
                for d in self._neighbor_dof_data
            ]

            neighbor_routing_data = self._neighbor_dof_data
        _comm = self._comm

        @jax.jit
        def fn(*args, **kwargs):
            result = local_fn(*args, **kwargs)
            owned_data = jnp.zeros(output_size)

            if isinstance(result, ColoredMatrix):
                data = result.data
            else:
                data = result
            if has_self:
                owned_data = owned_data.at[self_recv].add(data[self_send])
            for i, info in enumerate(neighbor_routing_data):
                nbr = np.int32(info["rank"])
                send_buf = data[nbr_send[i]]
                recv_tmpl = jnp.zeros(info["recv_size"], dtype=data.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=_comm
                )
                if info["recv_size"] > 0:
                    owned_data = owned_data.at[nbr_recv[i]].add(recv_vals)
            if isinstance(result, ColoredMatrix):
                return replace(result, data=owned_data)
            else:
                return owned_data

        return fn

    def gather(self, u_active: Array, rank: int = 0) -> np.ndarray | None:
        """Gather the full solution on rank 0 by node ownership.

        Each rank contributes the values at its owned nodes (determined by
        element-partition ownership stored in partition_info).  Fixed-DOF
        values reflect whatever the lifter set before calling this — typically
        lift_from_zeros, so constrained DOFs are zero in the output.

        Requires construction via from_local_problem.

        Args:
            u_active: array of length n_active (full local active DOF space,
                      e.g. output of lifter.lift_from_zeros)
            rank: the rank on which to gather the full solution (default 0)

        Returns:
            Full global DOF array (length n_nodes_global * n_dofs_per_node)
            on rank 0, None elsewhere.
        """
        if self._local_context is None:
            raise ValueError("gather requires construction via from_local_problem.")
        ctx = self._local_context

        n_dof = ctx.n_dofs_per_node
        owned = ctx.partition_info.owned_nodes_mask
        owned_vals = np.asarray(u_active).reshape(-1, n_dof)[owned]
        owned_global_nodes = ctx.partition_info.active_nodes[owned]

        all_vals = self._comm.gather(owned_vals, root=rank)
        all_nodes = self._comm.gather(owned_global_nodes, root=rank)

        if self._rank == rank:
            out = np.empty(ctx.n_nodes_global * n_dof)
            for vals, nodes in zip(all_vals, all_nodes):
                out[(nodes[:, None] * n_dof + np.arange(n_dof)).ravel()] = vals.ravel()
            return out
        return None

    def _precompute_routing_tables(
        self,
        local_colored_matrix: ColoredMatrix,
        active_dofs_np: np.ndarray,
        global_sparsity,
    ):
        """Precompute routing tables for ghost gather and assembly.

        Args:
            local_colored_matrix:   ColoredMatrix for this rank's active DOFs
            active_dofs_np:  np.ndarray (n_active,) of global DOF indices (full or reduced space)
            global_sparsity: global CSR sparsity pattern in the same DOF space

        Returns:

        """
        local_to_global_nnz, global_row_dofs = self._build_local_to_global_nnz(
            local_colored_matrix, active_dofs_np, global_sparsity
        )

        self._n_free = int(np.asarray(global_sparsity.indptr).shape[0]) - 1
        # compute the global DOF index ranges owned by each rank under the block distribution.
        self._rstart, self._rend = self._dof_range(self._n_free, self._size, self._rank)
        # gather all ranks' owned DOF index ranges so we can determine which global rows/DOFs we need from each neighbor.
        all_ranges = self._comm.allgather((self._rstart, self._rend))

        # --- NNZ routing (hessian) ---
        # For each neighbor, identify which local nonzeros contributes to rows owned by that neighbor → send_to,
        send_to = [
            np.where((global_row_dofs >= rs) & (global_row_dofs < re))[0].astype(
                np.int64
            )
            for rs, re in all_ranges
        ]
        send_counts = np.array([len(s) for s in send_to], dtype=np.int32)
        recv_counts = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(
            send_counts, recv_counts
        )  # exchange counts so we know how many entries to expect from each neighbor

        owned_offset = int(np.asarray(global_sparsity.indptr)[self._rstart])
        self._owned_nnz = (
            int(np.asarray(global_sparsity.indptr)[self._rend]) - owned_offset
        )  # number of NNZ entries owned by this rank

        self._local_send_idx = send_to[
            self._rank
        ]  # local nonzero indices that contribute to owned rows (to send to self)
        self._recv_local_idx = (
            local_to_global_nnz[self._local_send_idx] - owned_offset
        ).astype(
            np.int64
        )  # local indices of the received contributions from neighbors that we will add into owned rows

        # neighbor_ranks are the ranks we actually need to exchange with (those with nonzero send or receive counts)
        neighbor_ranks = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts[d] > 0 or recv_counts[d] > 0)
        )
        # For each neighbor, exchange the global nonzero indices corresponding to the local send indices → recv_buf,
        # so we know which local entries in the neighbor's output correspond to our owned rows and can add them correctly
        # in the JIT'd function.
        for nbr in neighbor_ranks:
            send_global_nnz = local_to_global_nnz[
                send_to[nbr]
            ]  # global nonzero indices corresponding to the local send indices for this neighbor
            recv_buf = np.empty(
                int(recv_counts[nbr]), dtype=np.int64
            )  # recieve buffer for the global nonzero indices from the neighbor
            self._comm.Sendrecv(
                sendbuf=send_global_nnz, dest=nbr, recvbuf=recv_buf, source=nbr
            )  # exchange the global nonzero indices corresponding to the local send indices for this neighbor
            self._neighbor_nnz_data.append(
                {
                    "rank": nbr,
                    "local_send_idx": send_to[nbr],
                    "recv_local_idx": (recv_buf - owned_offset).astype(np.int64),
                    "send_size": int(send_counts[nbr]),
                    "recv_size": int(recv_counts[nbr]),
                }
            )

        # --- DOF routing (gradient / ghost gather) ---
        # For each neighbor, identify which local DOFs contribute to rows owned by that neighbor
        send_to_dof = [
            np.where((active_dofs_np >= rs) & (active_dofs_np < re))[0].astype(np.int64)
            for rs, re in all_ranges
        ]
        send_counts_dof = np.array([len(s) for s in send_to_dof], dtype=np.int32)
        recv_counts_dof = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(send_counts_dof, recv_counts_dof)

        self._send_dof = send_to_dof[
            self._rank
        ]  # local DOF indices that contribute to owned rows (to send to self in gradient assembly, or receive from self in ghost gather)
        self._recv_dof = (active_dofs_np[self._send_dof] - self._rstart).astype(
            np.int64
        )  # local indices of the received contributions from self that we will add into owned rows (gradient) or fill from self (ghost gather)

        neighbor_ranks_dof = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts_dof[d] > 0 or recv_counts_dof[d] > 0)
        )
        # similar to the NNZ routing, exchange the global DOF indices corresponding to the local send DOF indices
        # for each neighbor so we can route contributions correctly in the JIT'd function.
        for nbr in neighbor_ranks_dof:
            send_global_dofs = active_dofs_np[send_to_dof[nbr]]
            recv_buf_dof = np.empty(int(recv_counts_dof[nbr]), dtype=np.int64)
            self._comm.Sendrecv(
                sendbuf=send_global_dofs, dest=nbr, recvbuf=recv_buf_dof, source=nbr
            )
            self._neighbor_dof_data.append(
                {
                    "rank": nbr,
                    "local_send_idx": send_to_dof[nbr],
                    "recv_local_idx": (recv_buf_dof - self._rstart).astype(np.int64),
                    "send_size": int(send_counts_dof[nbr]),
                    "recv_size": int(recv_counts_dof[nbr]),
                }
            )

    @staticmethod
    def _dof_range(n_dofs: int, n_ranks: int, r: int) -> tuple[int, int]:
        return _dof_range(n_dofs, n_ranks, r)

    @staticmethod
    def _build_local_to_global_nnz(
        local_colored_matrix: ColoredMatrix,
        active_dofs: np.ndarray,
        global_sparsity: csr_matrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a mapping from local active DOF pairs → global sparse matrix entries.
        For each local nonzero (i,j) in the ColoredMatrix, we want to know which global sparse
        matrix entry it corresponds to in the global CSR pattern. This allows us to route local
        Hessian contributions to the correct owned rows during assembly.

        Args:
            local_colored_matrix: ColoredMatrix for this rank's active DOFs
            active_dofs: np.ndarray of global DOF indices corresponding to local active DOFs
            global_sparsity: global CSR sparsity pattern in the same DOF space

        Returns:
            local_to_global_nnz: np.ndarray mapping local nonzero indices to global CSR indices
            global_row_dofs: np.ndarray of global DOF indices for each local row (length = n_active)
        """
        indptr_local = np.asarray(local_colored_matrix.indptr)
        indices_local = np.asarray(local_colored_matrix.indices)
        row_lengths = np.diff(indptr_local)
        local_coo_rows = np.repeat(np.arange(len(row_lengths)), row_lengths)
        global_row_dofs = active_dofs[local_coo_rows]
        global_col_dofs = active_dofs[indices_local]

        indptr_g = np.asarray(global_sparsity.indptr)
        indices_g = np.asarray(global_sparsity.indices)
        n_local_nnz = len(local_coo_rows)

        order = np.argsort(global_row_dofs, stable=True)
        s_rows = global_row_dofs[order]
        s_cols = global_col_dofs[order]
        unique_rows, g_starts = np.unique(s_rows, return_index=True)
        g_ends = np.append(g_starts[1:], n_local_nnz)

        result_sorted = np.empty(n_local_nnz, dtype=np.int64)
        for r, gs, ge in zip(unique_rows, g_starts, g_ends):
            row_start = int(indptr_g[r])
            row_cols = indices_g[row_start : int(indptr_g[r + 1])]
            pos = np.searchsorted(row_cols, s_cols[gs:ge])
            bad = (pos >= len(row_cols)) | (
                row_cols[np.minimum(pos, len(row_cols) - 1)] != s_cols[gs:ge]
            )
            if np.any(bad):
                raise ValueError(
                    f"Local NNZ entries (global row={r}, cols={s_cols[gs:ge][bad]}) "
                    "are not present in the global sparsity pattern. Ensure "
                    "local_colored_matrix and global_sparsity are built from the same "
                    "mesh with the same n_dofs_per_node."
                )
            result_sorted[gs:ge] = row_start + pos

        local_to_global_nnz = np.empty(n_local_nnz, dtype=np.int64)
        local_to_global_nnz[order] = result_sorted
        return local_to_global_nnz, global_row_dofs


class AllreducePlan:
    """MPI communication plan for allreduce-based parallel FEM assembly.

    Every rank holds the full replicated solution vector. Each rank computes
    local gradient/hessian contributions over its element subset; an allreduce
    sums them into the global result on every rank. Each rank then assembles
    only its owned rows into the distributed solver.

    Use when DOF coupling is non-nearest-neighbor (cohesive interfaces, contact,
    periodic BCs) or as a simpler starting point before switching to
    NeighborExchangePlan for better weak scaling.

    Args:
        global_colored_matrix: ColoredMatrix for the global (full) free-DOF sparsity
        lifter:                Lifter for the global problem
        comm:                  MPI communicator
    """

    def __init__(
        self,
        global_colored_matrix: ColoredMatrix,
        lifter: Lifter,
        comm: MPI.Comm,
    ):
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

        self._n_free = lifter.size_reduced
        self._rstart, self._rend = _dof_range(self._n_free, self._size, self._rank)

        indptr = np.asarray(global_colored_matrix.indptr)
        indices = np.asarray(global_colored_matrix.indices)
        nnz_start = int(indptr[self._rstart])
        nnz_end = int(indptr[self._rend])
        self._owned_nnz = nnz_end - nnz_start
        self._owned_nnz_start = nnz_start
        self._owned_ptr = (indptr[self._rstart : self._rend + 1] - nnz_start).astype(
            np.int32
        )
        self._owned_indices = indices[nnz_start:nnz_end].astype(np.int32)

    @property
    def global_size(self) -> int:
        """Total number of free DOFs in the global linear system."""
        return self._n_free

    @property
    def rstart(self) -> int:
        """First owned DOF index."""
        return self._rstart

    @property
    def rend(self) -> int:
        """One-past-last owned DOF index."""
        return self._rend

    @property
    def local_size(self) -> int:
        """Number of DOFs owned by this rank under the block distribution.

        Pass this as the explicit local size when creating distributed solver
        objects so their row distribution matches this plan.
        """
        rstart, rend = _dof_range(self._n_free, self._size, self._rank)
        return rend - rstart

    @property
    def owned_nnz(self) -> int:
        """Number of sparse matrix entries owned by this rank."""
        return self._owned_nnz

    @property
    def owned_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """(indptr, indices) for owned rows — pass directly to solver preallocation."""
        return self._owned_ptr, self._owned_indices

    @property
    def owned_free_mask(self) -> np.ndarray:
        """Boolean mask of length n_free_global; True for DOFs owned by this rank.

        Use this when applying nodal point loads: set the load only at owned DOFs
        so that after the allreduce sum each DOF receives exactly one rank's
        contribution — regardless of how many ranks can see the boundary nodes.

        Example:
            top_free_reduced = np.searchsorted(free_dofs_np, top_dofs)
            owned_top = top_free_reduced[plan.owned_free_mask[top_free_reduced]]
            f_ext_local = jnp.zeros(lifter.size_reduced).at[owned_top].set(load)
        """
        mask = np.zeros(self._n_free, dtype=bool)
        mask[self._rstart : self._rend] = True
        return mask

    def make_allgather(self) -> Callable[[Array], Array]:
        """Return a function: x_owned → u_free_global via Allgatherv.

        Gathers the owned DOF slice from every rank into the full replicated
        solution vector needed for local FEM computation on each rank.

        NOT JIT-compatible: uses MPI.Allgatherv directly. Call this function
        in the PETSc callback before passing its result into any JIT-compiled code.
        """
        all_ranges = self._comm.allgather((self._rstart, self._rend))
        counts = [re - rs for rs, re in all_ranges]
        n_free = self._n_free
        _comm = self._comm

        def fn(x_owned: Array) -> Array:
            if isinstance(x_owned, jax.core.Tracer):
                raise TypeError(
                    "make_allgather returns a function that uses MPI.Allgatherv "
                    "directly and cannot be called inside jax.jit. Call it in the "
                    "PETSc callback before any JIT-compiled code."
                )
            u_global = np.empty(n_free, dtype=np.float64)
            _comm.Allgatherv(np.ascontiguousarray(x_owned), [u_global, counts])
            return jnp.asarray(u_global)

        return fn

    def make_allreduce(self, local_fn: Callable, is_hessian: bool = False) -> Callable:
        """Return a JIT'd function that allreduces local_fn output → owned rows.

        Computes the local function (gradient or hessian) over the full replicated
        DOF vector, allreduces across all ranks, then returns only the owned slice.

        Args:
            local_fn:   function returning a vector (gradient) or ColoredMatrix (hessian)
            is_hessian: whether local_fn returns a ColoredMatrix
        """
        _comm = self._comm
        rstart = self._rstart
        rend = self._rend
        nnz_start = self._owned_nnz_start
        nnz_end = self._owned_nnz_start + self._owned_nnz

        @jax.jit
        def fn(*args, **kwargs):
            result = local_fn(*args, **kwargs)
            if isinstance(result, ColoredMatrix):
                data_reduced = mpi4jax.allreduce(result.data, op=MPI.SUM, comm=_comm)
                return replace(result, data=data_reduced[nnz_start:nnz_end])
            else:
                data_reduced = mpi4jax.allreduce(result, op=MPI.SUM, comm=_comm)
                return data_reduced[rstart:rend]

        return fn
