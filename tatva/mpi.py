from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
from jax import Array
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from tatva import sparse
from tatva.compound import Compound, FieldType
from tatva.lifter import Lifter
from tatva.sparse import ColoredMatrix

log = logging.getLogger(__name__)


class PartitionInfo(NamedTuple):
    """Information about the partitioning of the mesh across MPI ranks."""

    nodes_local_to_global: NDArray[np.int32]
    """Local to global node mapping array, where node_l2g[i] gives the global index of the
    local node i."""

    n_owned_nodes: int
    """Number of nodes owned by the local process. Since the local mesh sorts owned nodes
    before ghosts, nodes 0:n_owned_nodes are owned."""


class _LocalLayout(NamedTuple):
    """Local DOF layout information for a single rank."""

    local_to_global: NDArray[np.int32]
    offset: int
    n_owned: int
    n_total: int
    n_global: int
    owned_mask: NDArray[np.bool_]


class _NeighborDofRoute(NamedTuple):
    """Routing information for communicating DOF values to/from a neighboring rank."""

    rank: int
    local_send_idx: NDArray[np.int32]
    recv_local_idx: NDArray[np.int32]
    send_size: int
    recv_size: int


def _dof_map_from_node_map(
    node_map: NDArray[np.int32], n_dofs_per_node: int
) -> NDArray[np.int32]:
    """Convert a node map to a degree of freedom (DOF) map."""
    dof_map = node_map[:, None] * n_dofs_per_node + np.arange(n_dofs_per_node)
    return dof_map.flatten().astype(np.int32)


def _create_dof_layout(
    natural_dof_map: NDArray[np.int32],
    owned_mask: NDArray[np.bool_],
    n_natural_global: int,
    comm: MPI.Comm,
) -> _LocalLayout:
    """Create a global DOF layout for all DOFs across all processes."""
    _dtype = np.int32
    n_owned = int(np.sum(owned_mask))
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    n_per_rank = comm.allgather(n_owned)
    offset = (
        np.cumsum([0] + n_per_rank[: comm.rank], dtype=_dtype)[-1]
        if comm.rank > 0
        else 0
    )

    # Assign global indices to owned DOFs
    l2g = np.full(natural_dof_map.size, -1, dtype=_dtype)
    owned_indices = np.where(owned_mask)[0]
    l2g[owned_indices] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using the original global IDs (natural_dof_map)
    local_directory = np.full(n_natural_global, -1, dtype=_dtype)
    local_directory[natural_dof_map[owned_indices]] = l2g[owned_indices]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    ghost_indices = np.where(~owned_mask)[0]
    if ghost_indices.size > 0:
        l2g[ghost_indices] = global_directory[natural_dof_map[ghost_indices]]

    log.debug(
        f"Rank {comm.rank}: DOF layout - n_owned={n_owned}, "
        "n_total={natural_dof_map.size}, n_global={n_global}, offset={offset}"
    )

    return _LocalLayout(
        local_to_global=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=natural_dof_map.size,
        n_global=n_global,
        owned_mask=owned_mask,
    )


def _layout_from_compound(
    compound_cls: type[Compound], partition_info: PartitionInfo, comm: MPI.Comm
) -> _LocalLayout:
    """Create a DOF layout from the compound class and partition info."""
    _rank = comm.Get_rank()
    natural_dof_map = np.full(compound_cls.size, -1, dtype=np.int32)
    owned_mask = np.zeros(compound_cls.size, dtype=bool)

    # figure out the total number of nodes globally by finding the max global node
    # index across all ranks, then add 1 since node indices are zero-based
    local_max_node = (
        np.max(partition_info.nodes_local_to_global)
        if partition_info.nodes_local_to_global.size > 0
        else -1
    )
    n_nodes_global = comm.allreduce(local_max_node, op=MPI.MAX) + 1

    # loop over fields in the compound, assigning global DOF indices to owned DOFs and
    # building the natural DOF map for ghost resolution
    current_natural_offset: int = 0
    processed_slices: set[slice] = set()
    for _, field in compound_cls.fields:
        # if stacked fields: they have a _root_slice that identifies the original
        # slice of the base field; use that for DOF mapping
        # the cache processed_slices ensures, further fields stacked on the same base
        # field are skipped
        f_slice = getattr(field, "_root_slice", field._slice)
        f_slice = cast(slice, f_slice)
        if f_slice in processed_slices:
            continue
        processed_slices.add(f_slice)

        f_size_local = f_slice.stop - f_slice.start

        if field.field_type == FieldType.LOCAL:
            f_size_global = comm.allreduce(f_size_local, op=MPI.SUM)
            rank_offset = comm.exscan(f_size_local, op=MPI.SUM)  # ty:ignore[unresolved-attribute]
            if _rank == 0:
                rank_offset = 0

            natural_dof_map[f_slice] = (
                current_natural_offset + rank_offset + np.arange(f_size_local)
            )
            owned_mask[f_slice] = True
            current_natural_offset += f_size_global

        elif field.field_type == FieldType.NODAL:
            root_shape = getattr(field, "_root_shape", field.shape)
            n_dofs_per_node = int(np.prod(root_shape[1:]))

            if field.nodal_local_to_global is not None:
                # Note: incomplete nodal fields could be lagrange multipliers attached
                # to a subset of nodes
                # incomplete subset of nodes, possibly different across ranks. Use the
                # provided local_to_global mapping to identify the global nodes in
                # this subset, then build the DOF map from that.
                subset_global_nodes = np.asarray(field.nodal_local_to_global)
                field_dof_map = _dof_map_from_node_map(
                    subset_global_nodes, n_dofs_per_node
                )
                natural_dof_map[f_slice] = current_natural_offset + field_dof_map

                owned_global_nodes = partition_info.nodes_local_to_global[
                    : partition_info.n_owned_nodes
                ]
                is_owned_node = np.isin(subset_global_nodes, owned_global_nodes)
                field_owned_mask = np.repeat(is_owned_node, n_dofs_per_node)
                owned_mask[f_slice.start : f_slice.stop] = field_owned_mask

                # For incomplete fields, global size is more complex.
                # We need to know the total unique nodes in this subset across all ranks.
                all_subset = comm.allgather(subset_global_nodes)
                global_subset = np.unique(np.concatenate(all_subset))
                f_size_global = len(global_subset) * n_dofs_per_node

                # Re-map natural indices to be contiguous within this field's global space
                # to avoid massive gaps in the directory array.
                subset_node_indices = np.searchsorted(
                    global_subset, subset_global_nodes
                )
                natural_dof_map[f_slice] = (
                    current_natural_offset
                    + np.repeat(subset_node_indices, n_dofs_per_node) * n_dofs_per_node
                    + np.tile(np.arange(n_dofs_per_node), len(subset_global_nodes))
                )
            else:
                # Full nodal field
                f_size_global = n_nodes_global * n_dofs_per_node
                field_dof_map = _dof_map_from_node_map(
                    partition_info.nodes_local_to_global, n_dofs_per_node
                )
                natural_dof_map[f_slice] = current_natural_offset + field_dof_map

                n_owned_in_field = partition_info.n_owned_nodes * n_dofs_per_node
                owned_mask[f_slice.start : f_slice.start + n_owned_in_field] = True

            current_natural_offset += f_size_global

        elif field.field_type == FieldType.SHARED:
            natural_dof_map[f_slice] = current_natural_offset + np.arange(f_size_local)
            if _rank == 0:
                owned_mask[f_slice] = True
            current_natural_offset += f_size_local

    # from the natural DOF map and owned mask, build the global DOF layout and routing
    # tables for ghost exchange
    layout = _create_dof_layout(
        natural_dof_map, owned_mask, current_natural_offset, comm
    )
    return layout


class ExchangePlan:
    """An MPI communication plan for parallel FEM assembly."""

    def __init__(
        self,
        compound_cls: type[Compound],
        partition_info: PartitionInfo,
        comm: MPI.Comm,
    ):
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._partition_info = partition_info

        self.layout = _layout_from_compound(compound_cls, partition_info, comm)

        self._rstart = self.layout.offset
        self._rend = self.layout.offset + self.layout.n_owned
        self._global_size = self.layout.n_global

        self._precompute_routing_tables()

    def _precompute_routing_tables(self):
        """Precompute the static routing tables required for JIT-compiled comm.

        This method bridges the gap between the unstructured local memory (indices into
        the Compound state) and the structured global memory (contiguous blocks owned by
        each rank). It establishes the exact memory offsets for point-to-point exchanges
        used in forward ghost gathers and reverse residual assembly.

        Follows a 3-step handshake:
        - discovery: gathers global [rstart, rend) ranges from all ranks to identify which
          local ids map to which nbr's block
        - negotiation: exchanges send/receive counts via comm.Alltoall so each rank knows
          the exact buffer sizes required for its nbrs.
        - resolution: exchanges actual global DOF ids. This allows the owner rank o
          translate ghost rank's request into a local offset within its owned block

        Results in a set of integer-indexed routing instructions that allow the JIT'd
        `make_scatter` functions to perform MPI comm without knowledge of the underlying
        mesh or fields.
        """
        all_ranges = self._comm.allgather((self._rstart, self._rend))
        self._neighbor_dof_data: list[_NeighborDofRoute] = []

        # For each rank, identify which local DOFs we need to send to it
        dofs_to_send: list[NDArray[np.int32]] = []
        for nbr in range(self._size):
            if nbr == self._rank:
                dofs_to_send.append(np.array([], dtype=np.int32))
                continue
            rs, re = all_ranges[nbr]
            indices = np.where(
                (self.layout.local_to_global >= rs) & (self.layout.local_to_global < re)
            )[0]
            dofs_to_send.append(indices.astype(np.int32))

        send_counts = np.array([len(s) for s in dofs_to_send], dtype=np.int32)
        recv_counts = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(send_counts, recv_counts)

        neighbor_ranks: list[int] = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts[d] > 0 or recv_counts[d] > 0)
        )

        for nbr in neighbor_ranks:
            send_global_dofs = self.layout.local_to_global[dofs_to_send[nbr]]
            recv_buf_dof = np.empty(int(recv_counts[nbr]), dtype=np.int32)
            self._comm.Sendrecv(
                sendbuf=send_global_dofs, dest=nbr, recvbuf=recv_buf_dof, source=nbr
            )
            # recv_local_idx is the offset into our owned block [rstart, rend)
            self._neighbor_dof_data.append(
                _NeighborDofRoute(
                    rank=nbr,
                    local_send_idx=dofs_to_send[nbr],
                    recv_local_idx=(recv_buf_dof - self._rstart).astype(np.int32),
                    send_size=int(send_counts[nbr]),
                    recv_size=int(recv_counts[nbr]),
                )
            )

        # Self routing (owned DOFs)
        self._send_dof = np.where(self.layout.owned_mask)[0].astype(np.int32)
        self._recv_dof = (
            self.layout.local_to_global[self._send_dof] - self._rstart
        ).astype(np.int32)

    @property
    def global_size(self) -> int:
        """Total number of free DOFs in the global linear system."""
        return self._global_size

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
        """Number of DOFs owned by this rank."""
        return self._rend - self._rstart

    def zero_ghost_values(self, nodal_array: Array) -> Array:
        """Zero out ghost-DOF entries of any quantity assembled via scatter-add.

        Args:
            nodal_array: array of length compound_cls.size
        """
        ghost_indices = jnp.where(~self.layout.owned_mask)[0]
        return jnp.asarray(nodal_array).at[ghost_indices].set(0.0)

    def make_scatter_fwd_set(self) -> Callable[[Array], Array]:
        """Return a JIT'd function: x_owned → u_local.

        Gathers ghost DOF values from owning ranks.
        """
        self_send = jnp.array(self._send_dof, dtype=jnp.int64)
        self_recv = jnp.array(self._recv_dof, dtype=jnp.int64)
        has_self = len(self._send_dof) > 0

        # Roles swap vs assembly: send ← recv_local_idx, recv ← local_send_idx
        nbr_send = [
            jnp.array(d.recv_local_idx, dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        nbr_recv = [
            jnp.array(d.local_send_idx, dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        neighbor_dof_data = self._neighbor_dof_data
        _comm = self._comm
        local_size = len(self.layout.local_to_global)

        @jax.jit
        def fn(x_owned: Array) -> Array:
            u_local = jnp.zeros(local_size, dtype=x_owned.dtype)
            if has_self:
                u_local = u_local.at[self_send].set(x_owned[self_recv])
            for i, info in enumerate(neighbor_dof_data):
                nbr = np.int32(info.rank)
                send_buf = x_owned[nbr_send[i]]
                recv_tmpl = jnp.zeros(info.send_size, dtype=x_owned.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=_comm
                )
                if info.send_size > 0:
                    u_local = u_local.at[nbr_recv[i]].set(recv_vals)
            return u_local

        return fn

    def make_scatter_rev_add(
        self, local_fn: Callable, is_hessian: bool = False
    ) -> Callable:
        """Return a JIT'd function: u_local → owned_data.

        Computes the local function, then gathers contributions from ghosts.
        """
        if is_hessian:
            raise NotImplementedError(
                "NNZ routing for Hessian not yet implemented in ExchangePlan."
            )

        self_send = jnp.array(self._send_dof, dtype=jnp.int64)
        self_recv = jnp.array(self._recv_dof, dtype=jnp.int64)
        has_self = len(self._send_dof) > 0

        output_size = self.local_size
        nbr_send = [
            jnp.array(d.local_send_idx, dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        nbr_recv = [
            jnp.array(d.recv_local_idx, dtype=jnp.int64)
            for d in self._neighbor_dof_data
        ]
        neighbor_routing_data = self._neighbor_dof_data
        _comm = self._comm

        @jax.jit
        def fn(*args, **kwargs):
            data = local_fn(*args, **kwargs)
            owned_data = jnp.zeros(output_size, dtype=data.dtype)

            if has_self:
                owned_data = owned_data.at[self_recv].add(data[self_send])
            for i, info in enumerate(neighbor_routing_data):
                nbr = np.int32(info.rank)
                send_buf = data[nbr_send[i]]
                recv_tmpl = jnp.zeros(info.recv_size, dtype=data.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=_comm
                )
                if info.recv_size > 0:
                    owned_data = owned_data.at[nbr_recv[i]].add(recv_vals)
            return owned_data

        return fn


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
        global_sparsity,
        partition_info: PartitionInfo,
        n_dofs_per_node: int,
        local_colored_matrix: ColoredMatrix,
        lifter: Lifter,
        comm: MPI.Comm,
    ):
        """Build the neighbor-exchange plan from local mesh artefacts.

        Args:
            global_sparsity:     global sparsity pattern (CSR) covering the full DOF
                                 space before BC reduction. For pure mesh problems use
                                 ``sparse.create_sparsity_pattern(mesh, n_dofs_per_node)``;
                                 for problems with non-mesh coupling (interfaces, contact)
                                 add the extra entries before passing.
            partition_info:      partition info for this rank's local mesh
            n_dofs_per_node:     DOFs per mesh node
            local_colored_matrix: ColoredMatrix built from the local mesh
            lifter:              Lifter for this rank's local problem
            comm:                MPI communicator
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
        n_dofs_total = global_sparsity.shape[0]
        free_dofs_global = np.setdiff1d(
            np.arange(n_dofs_total), global_fixed_indices, assume_unique=True
        )

        # compute the reduced global sparsity pattern
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
            n_nodes_global=n_dofs_total // n_dofs_per_node,
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
