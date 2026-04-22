# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, NamedTuple, ParamSpec, overload

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
from jax import Array
from mpi4py import MPI
from numpy.typing import NDArray

from tatva.lifter import Lifter
from tatva.sparse import ColoredMatrix

log = logging.getLogger(__name__)

P = ParamSpec("P")


class _NeighborDofRoute(NamedTuple):
    """Routing information for communicating DOF values to/from a neighboring rank."""

    rank: int
    local_send_idx: NDArray[np.int32]
    recv_local_idx: NDArray[np.int32]
    send_size: int
    recv_size: int


class _NeighborNnzRoute(NamedTuple):
    """Routing information for communicating non-zero Hessian values."""

    rank: int
    local_send_idx: NDArray[np.int32]
    recv_local_idx: NDArray[np.int32]
    send_size: int
    recv_size: int


class _HessianLayout(NamedTuple):
    """Layout and routing information for Hessian assembly on a single rank."""

    owned_nnz: int
    owned_ptr: NDArray[np.int32]
    owned_indices: NDArray[np.int32]
    local_send_idx: NDArray[np.int32]
    recv_local_idx: NDArray[np.int32]
    neighbor_data: list[_NeighborNnzRoute]


class _LocalLayout(NamedTuple):
    """Local DOF layout information for a single rank."""

    local_to_global: NDArray[np.int32]
    offset: int
    n_owned: int
    n_total: int
    n_global: int
    owned_mask: NDArray[np.bool_]
    natural_l2g: NDArray[np.int32]


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
        f"n_total={natural_dof_map.size}, n_global={n_global}, offset={offset}"
    )

    return _LocalLayout(
        local_to_global=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=natural_dof_map.size,
        n_global=n_global,
        owned_mask=owned_mask,
        natural_l2g=natural_dof_map,
    )


class ExchangePlan:
    """An MPI communication plan for parallel FEM assembly."""

    def __init__(
        self,
        layout: _LocalLayout,
        comm: MPI.Comm,
        local_colored_matrix: ColoredMatrix | None = None,
    ):
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

        self.layout = layout

        self._rstart = layout.offset
        self._rend = layout.offset + layout.n_owned
        self._global_size = layout.n_global

        self._precompute_routing_tables()

        self.hessian_layout: _HessianLayout | None = None
        if local_colored_matrix is not None:
            self.hessian_layout = self._precompute_nnz_routing_tables(
                local_colored_matrix
            )

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

        # Convert layout to numpy for mpi4py calls
        l2g = np.asarray(self.layout.local_to_global)

        # For each rank, identify which local DOFs we need to send to it
        dofs_to_send: list[NDArray[np.int32]] = []
        for nbr in range(self._size):
            if nbr == self._rank:
                dofs_to_send.append(np.array([], dtype=np.int32))
                continue
            rs, re = all_ranges[nbr]
            indices = np.where((l2g >= rs) & (l2g < re))[0]
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
            send_global_dofs = np.ascontiguousarray(l2g[dofs_to_send[nbr]])
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
        owned_mask = np.asarray(self.layout.owned_mask)
        self._send_dof = np.where(owned_mask)[0].astype(np.int32)
        self._recv_dof = (l2g[self._send_dof] - self._rstart).astype(np.int32)

    def _precompute_nnz_routing_tables(
        self, local_colored: ColoredMatrix
    ) -> _HessianLayout:
        """Precompute routing tables for Hessian assembly without global sparsity."""
        indptr = np.asarray(local_colored.indptr)
        indices = np.asarray(local_colored.indices)

        # Convert layout to numpy for mpi4py calls
        l2g = np.asarray(self.layout.local_to_global)

        # 1. Local COO -> Global COO
        l_row = np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))
        l_col = indices

        g_row = l2g[l_row]
        g_col = l2g[l_col]

        # 2. Filter valid nonzeros and identify owners
        # (Negative indices represent constrained DOFs and are ignored)
        valid = (g_row >= 0) & (g_col >= 0)
        l_nnz_idx = np.where(valid)[0]
        g_row_valid = g_row[valid]
        g_col_valid = g_col[valid]

        all_ranges = self._comm.allgather((self._rstart, self._rend))
        send_to_nnz: list[NDArray[np.int32]] = []
        for nbr in range(self._size):
            rs, re = all_ranges[nbr]
            in_range = (g_row_valid >= rs) & (g_row_valid < re)
            send_to_nnz.append(l_nnz_idx[in_range].astype(np.int32))

        send_counts = np.array([len(s) for s in send_to_nnz], dtype=np.int32)
        recv_counts = np.empty(self._size, dtype=np.int32)
        self._comm.Alltoall(send_counts, recv_counts)

        neighbor_ranks = sorted(
            d
            for d in range(self._size)
            if d != self._rank and (send_counts[d] > 0 or recv_counts[d] > 0)
        )

        # 3. Exchange Coordinates to build global structural directory
        my_mask = (g_row_valid >= self._rstart) & (g_row_valid < self._rend)
        all_received_rows = [g_row_valid[my_mask]]
        all_received_cols = [g_col_valid[my_mask]]

        for nbr in neighbor_ranks:
            rs, re = all_ranges[nbr]
            # Which nonzeros do we send to nbr?
            nbr_mask = (g_row_valid >= rs) & (g_row_valid < re)
            s_row, s_col = g_row_valid[nbr_mask], g_col_valid[nbr_mask]

            r_row = np.empty(int(recv_counts[nbr]), dtype=np.int32)
            r_col = np.empty(int(recv_counts[nbr]), dtype=np.int32)
            self._comm.Sendrecv(sendbuf=s_row, dest=nbr, recvbuf=r_row, source=nbr)
            self._comm.Sendrecv(sendbuf=s_col, dest=nbr, recvbuf=r_col, source=nbr)

            all_received_rows.append(r_row)
            all_received_cols.append(r_col)

        # 4. Build Owned CSR and reverse mapping
        stacked_rows = np.concatenate(all_received_rows)
        stacked_cols = np.concatenate(all_received_cols)

        # Map to unique structural nonzeros
        pairs = np.column_stack((stacked_rows, stacked_cols))
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

        owned_nnz = unique_pairs.shape[0]
        local_u_rows = unique_pairs[:, 0] - self._rstart
        counts = np.bincount(local_u_rows, minlength=self.local_size)
        owned_ptr = np.cumsum(np.insert(counts, 0, 0)).astype(np.int32)
        owned_indices = unique_pairs[:, 1].astype(np.int32)

        # 5. Pack routing data
        local_send_idx_nnz = send_to_nnz[self._rank]
        recv_local_idx_nnz = inverse[: len(all_received_rows[0])].astype(np.int32)

        neighbor_nnz_data: list[_NeighborNnzRoute] = []
        curr = len(all_received_rows[0])
        for nbr in neighbor_ranks:
            size = int(recv_counts[nbr])
            neighbor_nnz_data.append(
                _NeighborNnzRoute(
                    rank=nbr,
                    local_send_idx=send_to_nnz[nbr],
                    recv_local_idx=inverse[curr : curr + size].astype(np.int32),
                    send_size=int(send_counts[nbr]),
                    recv_size=int(recv_counts[nbr]),
                )
            )
            curr += size

        return _HessianLayout(
            owned_nnz=owned_nnz,
            owned_ptr=owned_ptr,
            owned_indices=owned_indices,
            local_send_idx=local_send_idx_nnz,
            recv_local_idx=recv_local_idx_nnz,
            neighbor_data=neighbor_nnz_data,
        )

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

    @property
    def owned_nnz(self) -> int:
        """Number of sparse matrix entries owned by this rank."""
        if self.hessian_layout is None:
            raise ValueError("Hessian layout not initialized.")
        return self.hessian_layout.owned_nnz

    @property
    def owned_csr(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """(indptr, indices) for owned rows."""
        if self.hessian_layout is None:
            raise ValueError("Hessian layout not initialized.")
        return self.hessian_layout.owned_ptr, self.hessian_layout.owned_indices

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

    @overload
    def make_scatter_rev_add(
        self, local_fn: Callable[P, Array], is_hessian: bool = False
    ) -> Callable[P, Array]: ...

    @overload
    def make_scatter_rev_add(
        self, local_fn: Callable[P, ColoredMatrix], is_hessian: bool = True
    ) -> Callable[P, ColoredMatrix]: ...

    def make_scatter_rev_add(self, local_fn, is_hessian=False):
        """Return a JIT'd function: u_local → owned_data.

        Computes the local function, then gathers contributions from ghosts. If
        `is_hessian=True`, the local function must return a ColoredMatrix.
        """
        if is_hessian:
            return self._make_scatter_rev_add_hessian(local_fn)
        else:
            return self._make_scatter_rev_add_gradient(local_fn)

    def _make_scatter_rev_add_hessian(
        self, local_fn: Callable[P, ColoredMatrix]
    ) -> Callable[P, ColoredMatrix]:
        """Return a JIT'd function: local_fn output → owned_data for Hessian case."""
        if self.hessian_layout is None:
            raise ValueError("Hessian layout not initialized.")

        self_send = jnp.array(self.hessian_layout.local_send_idx, dtype=jnp.int64)
        self_recv = jnp.array(self.hessian_layout.recv_local_idx, dtype=jnp.int64)
        has_self = len(self.hessian_layout.local_send_idx) > 0

        output_size = self.hessian_layout.owned_nnz
        nbr_send = [
            jnp.array(d.local_send_idx, dtype=jnp.int64)
            for d in self.hessian_layout.neighbor_data
        ]
        nbr_recv = [
            jnp.array(d.recv_local_idx, dtype=jnp.int64)
            for d in self.hessian_layout.neighbor_data
        ]
        neighbor_routing_data = self.hessian_layout.neighbor_data

        @jax.jit
        def fn(*args, **kwargs):
            result = local_fn(*args, **kwargs)
            data = result.data
            owned_data = jnp.zeros(output_size, dtype=data.dtype)

            if has_self:
                owned_data = owned_data.at[self_recv].add(data[self_send])
            for i, info in enumerate(neighbor_routing_data):
                nbr = np.int32(info.rank)
                send_buf = data[nbr_send[i]]
                recv_tmpl = jnp.zeros(info.recv_size, dtype=data.dtype)
                recv_vals = mpi4jax.sendrecv(
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=self._comm
                )
                if info.recv_size > 0:
                    owned_data = owned_data.at[nbr_recv[i]].add(recv_vals)

            # in the Hessian case, it is assumed local_fn returns a ColoredMatrix
            # which is a dataclass. That's why it is safe to use replace here.
            return replace(result, data=owned_data)

        return fn

    def _make_scatter_rev_add_gradient(
        self, local_fn: Callable[P, Array]
    ) -> Callable[P, Array]:
        """Return a JIT'd function: local_fn output → owned_data for gradient case."""
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
                    send_buf, recv_tmpl, source=nbr, dest=nbr, comm=self._comm
                )
                if info.recv_size > 0:
                    owned_data = owned_data.at[nbr_recv[i]].add(recv_vals)
            return owned_data

        return fn


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


def _dof_range(n: int, size: int, rank: int) -> tuple[int, int]:
    """Calculate the [start, end) range of DOFs owned by a rank."""
    base = n // size
    rem = n % size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rank * base + rem
        end = start + base
    return start, end
