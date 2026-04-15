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


@dataclass
class PartitionInfo:
    """Data structure to hold partitioning information for the local mesh."""

    active_nodes: np.ndarray
    owned_nodes_mask: np.ndarray
    nodes_global_to_local: np.ndarray


class NeighborExchangePlan:
    """Precomputed MPI communication plan for neighbor-exchange parallel FEM assembly.

    Encapsulates all routing tables for three operations:
      - Ghost gather:      distribute x_owned (PETSc vec) → u_active (all active DOFs)
      - Gradient assembly: u_active contributions → owned gradient rows
      - Hessian assembly:  u_active contributions → owned sparse matrix entries

    The DOF space (full or reduced/free) is determined by what is passed in:
    active_dofs_np and global_sparsity must be in the same space.

    Args:
        local_colored:   ColoredMatrix for this rank's active DOFs
        active_dofs_np:  np.ndarray (n_active,) of global DOF indices (full or reduced space)
        global_sparsity: global CSR sparsity pattern in the same DOF space
        comm:            MPI communicator
    """

    def __init__(
        self,
        local_colored: ColoredMatrix,
        active_dofs: np.ndarray,
        global_sparsity: csr_matrix,
        comm: MPI.Comm,
        free_dofs_global: np.ndarray | None = None,
    ):
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._global_sparsity = global_sparsity
        self._free_dofs_global = free_dofs_global

        self._active_dofs = active_dofs
        self._free_dofs_local: np.ndarray | None = None
        self._partition_info: PartitionInfo | None = None
        self._n_dofs_per_node: int | None = None
        self._n_nodes_global: int | None = None
        self._neighbor_data = []
        self._neighbor_dof_data = []

        self._precompute_routing_tables(local_colored, active_dofs, global_sparsity)

        # Cache owned-row CSR (used for PETSc preallocation and setValuesCSR)
        indptr_g = np.asarray(global_sparsity.indptr)
        indices_g = np.asarray(global_sparsity.indices)
        self._owned_ptr = (
            indptr_g[self._rstart : self._rend + 1] - indptr_g[self._rstart]
        ).astype(np.int32)
        self._owned_indices = indices_g[
            indptr_g[self._rstart] : indptr_g[self._rend]
        ].astype(np.int32)

    @classmethod
    def from_local_problem(
        cls,
        raw_mesh: Mesh,
        partition_info: PartitionInfo,
        n_dofs_per_node: int,
        local_colored: ColoredMatrix,
        lifter: Lifter,
        comm: MPI.Comm,
        free_dofs_global: np.ndarray | None = None,
    ):
        """Convenience constructor — derives global sparsity and active-DOF mapping internally.

        The caller only needs to supply the local mesh artefacts they already have.
        The global CSR construction and the searchsorted mapping from active local
        free-DOFs → global free-DOF indices are handled here.

        Args:
            raw_mesh:          full global mesh (replicated on all ranks)
            n_dofs_per_node:   DOFs per mesh node
            local_colored:     ColoredMatrix built from the local mesh
            active_dofs:       global full-DOF indices of this rank's active nodes
            lifter:            Lifter for this rank's local problem
            comm:              MPI communicator
            free_dofs_global:  global free-DOF indices (same on all ranks). If None,
                               it is derived by synchronizing lifter.constrained_dofs
                               across all ranks.
        """

        active_dofs = np.sort(
            np.concatenate(
                [
                    partition_info.active_nodes * n_dofs_per_node + d
                    for d in range(n_dofs_per_node)
                ]
            )
        )

        if free_dofs_global is None:
            # Derive global free DOFs by gathering all fixed DOFs from all ranks.
            # Ranks only know about fixed DOFs that are "active" on them.
            local_fixed_global = active_dofs[np.asarray(lifter.constrained_dofs)]
            all_fixed = comm.allgather(local_fixed_global)
            global_fixed_indices = np.unique(np.concatenate(all_fixed))

            n_dofs_total = raw_mesh.coords.shape[0] * n_dofs_per_node
            free_dofs_global = np.setdiff1d(
                np.arange(n_dofs_total), global_fixed_indices, assume_unique=True
            )

        global_sparsity = sparse.create_sparsity_pattern(raw_mesh, n_dofs_per_node)
        global_free_sparsity = sparse.reduce_sparsity_pattern(
            global_sparsity, free_dofs_global
        )
        active_free_global_full = active_dofs[np.array(lifter.free_dofs)]
        active_free_global_reduced = np.searchsorted(
            np.asarray(free_dofs_global), active_free_global_full
        ).astype(np.int64)
        instance = cls(
            local_colored,
            active_free_global_reduced,
            global_free_sparsity,
            comm,
            free_dofs_global=free_dofs_global,
        )
        # Store local free-DOF indices (into n_active space) so zero_ghost_values can
        # zero ghost entries in the full active DOF vector.
        instance._free_dofs_local = np.array(lifter.free_dofs)
        instance._partition_info = partition_info
        instance._n_dofs_per_node = n_dofs_per_node
        instance._n_nodes_global = raw_mesh.coords.shape[0]
        return instance

    @property
    def free_dofs_global(self) -> np.ndarray:
        """Global free-DOF indices."""
        if self._free_dofs_global is None:
            raise ValueError(
                "free_dofs_global was not provided or derived during initialization."
            )
        return self._free_dofs_global

    @property
    def rstart(self) -> int:
        """First owned DOF index."""
        return self._rstart

    @property
    def rend(self) -> int:
        """One-past-last owned DOF index."""
        return self._rend

    @property
    def owned_dofs(self) -> int:
        """Number of DOFs owned by this rank."""
        return self.rend - self.rstart

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
        owned_mask = np.asarray(
            (self._active_dofs >= self._rstart) & (self._active_dofs < self._rend)
        )
        ghost_free_dofs = jnp.array(self._free_dofs_local[~owned_mask])
        return jnp.asarray(nodal_array).at[ghost_free_dofs].set(0.0)

    # ------------------------------------------------------------------
    # JIT'd function factories
    # ------------------------------------------------------------------

    def make_scatter_fwd_set(self, n_active: int) -> Callable[[Array], Array]:
        """Return a JIT'd function: x_owned → u_active.

        Gathers ghost DOF values from owning ranks so every rank has the full
        active-DOF vector needed for local FEM computation.

        This is the transpose of make_grad_fn's assembly step:
          assembly:     ghost contributors SEND  → owners RECEIVE (scatter-add)
          ghost gather: owners             SEND  → ghosts RECEIVE (scatter-set)

        Args:
            n_active: length of the output u_active vector
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
            u_active = jnp.zeros(n_active)
            if has_self:
                u_active = u_active.at[self_send].set(x_owned[self_recv])
            for i, info in enumerate(neighbor_dof_data):
                nbr = np.int32(info["rank"])
                send_buf = x_owned[nbr_send[i]]
                recv_tmpl = jnp.zeros(info["send_size"], dtype=x_owned.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=_comm
                )
                if info["send_size"] > 0:
                    u_active = u_active.at[nbr_recv[i]].set(recv_vals)
            return u_active

        return fn

    def make_scatter_rev_add(self, local_fn: Callable, is_hessian: bool = False) -> Callable:
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

            owned_dofs = (
                self.owned_nnz
            )  # for Hessian; for gradient, this will be owned_dofs
            nbr_send = [
                jnp.array(d["local_send_idx"], dtype=jnp.int64)
                for d in self._neighbor_data
            ]
            nbr_recv = [
                jnp.array(d["recv_local_idx"], dtype=jnp.int64)
                for d in self._neighbor_data
            ]
            neighbor_routing_data = self._neighbor_data

        else:
            self_send = jnp.array(self._send_dof, dtype=jnp.int64)
            self_recv = jnp.array(self._recv_dof, dtype=jnp.int64)
            has_self = len(self._send_dof) > 0

            owned_dofs = (
                self.owned_dofs
            )  # for gradient; for Hessian, this will be owned_nnz
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
            owned_data = jnp.zeros(owned_dofs)

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
        if self._partition_info is None or self._n_dofs_per_node is None:
            raise ValueError("gather requires construction via from_local_problem.")

        n_dof = self._n_dofs_per_node
        owned = self._partition_info.owned_nodes_mask
        owned_vals = np.asarray(u_active).reshape(-1, n_dof)[owned]
        owned_global_nodes = self._partition_info.active_nodes[owned]

        all_vals = self._comm.gather(owned_vals, root=rank)
        all_nodes = self._comm.gather(owned_global_nodes, root=rank)

        if self._rank == rank:
            out = np.empty(self._n_nodes_global * n_dof)
            for vals, nodes in zip(all_vals, all_nodes):
                out[(nodes[:, None] * n_dof + np.arange(n_dof)).ravel()] = vals.ravel()
            return out
        return None

    def _precompute_routing_tables(
        self, local_colored: ColoredMatrix, active_dofs_np: np.ndarray, global_sparsity
    ):
        """Precompute routing tables for ghost gather and assembly.

        Args:
            local_colored:   ColoredMatrix for this rank's active DOFs
            active_dofs_np:  np.ndarray (n_active,) of global DOF indices (full or reduced space)
            global_sparsity: global CSR sparsity pattern in the same DOF space

        Returns:

        """
        local_to_global_nnz, global_row_dofs = self._build_local_to_global_nnz(
            local_colored, active_dofs_np, global_sparsity
        )

        n_dofs = int(np.asarray(global_sparsity.indptr).shape[0]) - 1
        self._rstart, self._rend = self._dof_range(n_dofs, self._size, self._rank)
        all_ranges = self._comm.allgather((self._rstart, self._rend))
        # self._owned_dofs = self._rend - self._rstart

        # --- NNZ routing (hessian) ---
        send_to = [
            np.where((global_row_dofs >= rs) & (global_row_dofs < re))[0].astype(
                np.int64
            )
            for rs, re in all_ranges
        ]
        send_counts = np.array([len(s) for s in send_to], dtype=np.int32)
        recv_counts = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(send_counts, recv_counts)

        owned_offset = int(np.asarray(global_sparsity.indptr)[self._rstart])
        self._owned_nnz = (
            int(np.asarray(global_sparsity.indptr)[self._rend]) - owned_offset
        )

        self._local_send_idx = send_to[self._rank]
        self._recv_local_idx = (
            local_to_global_nnz[self._local_send_idx] - owned_offset
        ).astype(np.int64)

        neighbor_ranks = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts[d] > 0 or recv_counts[d] > 0)
        )
        for nbr in neighbor_ranks:
            send_global_nnz = local_to_global_nnz[send_to[nbr]]
            recv_buf = np.empty(int(recv_counts[nbr]), dtype=np.int64)
            self._comm.Sendrecv(
                sendbuf=send_global_nnz, dest=nbr, recvbuf=recv_buf, source=nbr
            )
            self._neighbor_data.append(
                {
                    "rank": nbr,
                    "local_send_idx": send_to[nbr],
                    "recv_local_idx": (recv_buf - owned_offset).astype(np.int64),
                    "send_size": int(send_counts[nbr]),
                    "recv_size": int(recv_counts[nbr]),
                }
            )

        # --- DOF routing (gradient / ghost gather) ---
        send_to_dof = [
            np.where((active_dofs_np >= rs) & (active_dofs_np < re))[0].astype(np.int64)
            for rs, re in all_ranges
        ]
        send_counts_dof = np.array([len(s) for s in send_to_dof], dtype=np.int32)
        recv_counts_dof = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(send_counts_dof, recv_counts_dof)

        self._send_dof = send_to_dof[self._rank]
        self._recv_dof = (active_dofs_np[self._send_dof] - self._rstart).astype(
            np.int64
        )

        neighbor_ranks_dof = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts_dof[d] > 0 or recv_counts_dof[d] > 0)
        )
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
        """Compute the global DOF range [rstart, rend) owned by rank r.
        Matches PETSc's distribution: the first (n_dofs % n_ranks) ranks get one extra DOF.

        Args:
            n_dofs: total number of DOFs in the global problem
            n_ranks: total number of MPI ranks
            r: the rank for which to compute the range

        Returns:
            (rstart, rend): the global DOF index range owned by rank r
        """
        base = n_dofs // n_ranks
        extra = n_dofs % n_ranks
        rstart = r * base + min(r, extra)
        rend = rstart + base + (1 if r < extra else 0)
        return rstart, rend

    @staticmethod
    def _build_local_to_global_nnz(
        local_colored: ColoredMatrix,
        active_dofs: np.ndarray,
        global_sparsity: csr_matrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a mapping from local active DOF pairs → global sparse matrix entries.
        For each local nonzero (i,j) in the ColoredMatrix, we want to know which global sparse
        matrix entry it corresponds to in the global CSR pattern. This allows us to route local
        Hessian contributions to the correct owned rows during assembly.

        Args:
            local_colored: ColoredMatrix for this rank's active DOFs
            active_dofs: np.ndarray of global DOF indices corresponding to local active DOFs
            global_sparsity: global CSR sparsity pattern in the same DOF space

        Returns:
            local_to_global_nnz: np.ndarray mapping local nonzero indices to global CSR indices
            global_row_dofs: np.ndarray of global DOF indices for each local row (length = n_active)
        """
        indptr_local = np.asarray(local_colored.indptr)
        indices_local = np.asarray(local_colored.indices)
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
            result_sorted[gs:ge] = row_start + np.searchsorted(row_cols, s_cols[gs:ge])

        local_to_global_nnz = np.empty(n_local_nnz, dtype=np.int64)
        local_to_global_nnz[order] = result_sorted
        return local_to_global_nnz, global_row_dofs
