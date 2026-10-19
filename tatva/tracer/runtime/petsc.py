# ty: ignore[invalid-argument-type]
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from numpy.typing import ArrayLike, NDArray

from tatva import sparse
from tatva.tracer.local.dof_plan import LocalDofPlan

try:
    from petsc4py import PETSc
except ImportError as e:
    raise ImportError(
        "petsc4py is required for the PETSc runtime backend. "
        "Please install it via `pip install petsc4py` or your package manager."
    ) from e

if TYPE_CHECKING:
    from mpi4py import MPI


class PetscSNESProblem:
    """PETSc SNES adapter for a rank-local Tatva energy functional.

    The energy argument is the Tatva storage vector ordered as

        [owned DOFs | ghost DOFs]

    and returns this rank's contribution to the global energy.
    """

    def __init__(
        self,
        energy: Callable,
        *,
        dof_plan: LocalDofPlan,
        local_sparsity: sps.csr_matrix,
        comm: PETSc.Comm | MPI.Intracomm = PETSc.COMM_WORLD,
        jit: bool = True,
    ):
        self.energy = energy

        self.dofs = dof_plan
        self.comm = comm

        rank = comm.Get_rank()

        if self.dofs.rank != rank:
            raise ValueError(
                f"DOF plan is for rank {self.dofs.rank}, "
                f"but communicator rank is {rank}"
            )

        self.n_global = self.dofs.global_size
        self._ghost_petsc = self._build_petsc_ghost_indices()

        # Differentiation
        gradient = jax.grad(energy, argnums=0)

        n = self.dofs.storage.local_size

        if local_sparsity.shape != (n, n):
            raise ValueError(
                f"local Hessian pattern has shape {local_sparsity.shape}; expected {(n, n)}"
            )

        self.local_sparsity = local_sparsity

        colored_mat = sparse.ColoredMatrix.from_csr(local_sparsity)
        hessian = sparse.jacfwd(gradient, colored_mat)

        if jit:
            gradient = jax.jit(gradient)
            hessian = jax.jit(hessian)

        self.gradient = gradient
        self.hessian = hessian

        # COO representation of exactly the same local local_sparsity.
        #
        # These indices are storage-local indices, not Tatva global
        # and not PETSc global indices.
        self._coo_rows = np.repeat(
            np.arange(n, dtype=PETSc.IntType),
            np.diff(local_sparsity.indptr),
        )
        self._coo_cols = np.asarray(local_sparsity.indices, dtype=PETSc.IntType).copy()

    def _build_petsc_ghost_indices(
        self,
    ) -> NDArray:
        rank = self.comm.Get_rank()

        owned = np.asarray(self.dofs.owned_global, dtype=PETSc.IntType)
        n_owned = owned.size

        # PETSc's global numbering groups the owned entries
        # of each rank contiguously.
        counts = self.comm.allgather(n_owned)  # ty: ignore[unresolved-attribute]

        offset = sum(counts[:rank])
        petsc_owned = np.arange(offset, offset + n_owned, dtype=PETSc.IntType)

        # Application numbering (Tatva) -> PETSc numbering.
        ao = PETSc.AO().createBasic(owned, petsc_owned, comm=self.comm)

        ghosts = np.asarray(self.dofs.ghost_global, dtype=PETSc.IntType).copy()

        ao.app2petsc(ghosts)
        ao.destroy()

        return ghosts

    def create_vec(self) -> PETSc.Vec:
        return PETSc.Vec().createGhost(
            ghosts=self._ghost_petsc,
            size=(self.dofs.storage.n_owned, self.n_global),
            comm=self.comm,
        )

    def create_matrix(self, lgmap: PETSc.LGMap) -> PETSc.Mat:
        n_owned = self.dofs.storage.n_owned

        J = PETSc.Mat().create(comm=self.comm)
        J.setSizes(
            (
                (n_owned, self.n_global),
                (n_owned, self.n_global),
            )
        )
        J.setType(PETSc.Mat.Type.AIJ)

        # Matrix local indices now mean exactly the same thing
        # as Vec local indices / Tatva storage indices.
        J.setLGMap(lgmap, lgmap)

        # PETSc is allowed to modify the index arrays, hence copy().
        J.setPreallocationCOOLocal(
            self._coo_rows.copy(),
            self._coo_cols.copy(),
        )

        return J

    def create_containers(self) -> tuple[PETSc.Vec, PETSc.Vec, PETSc.Mat]:
        x = self.create_vec()
        f = self.create_vec()

        lgmap = x.getLGMap()
        if lgmap is None:
            raise RuntimeError("PETSc ghost vector has no local-to-global mapping")

        J = self.create_matrix(lgmap)
        return x, f, J

    def create_snes(self) -> tuple[PETSc.SNES, PETSc.Vec]:
        x, f, J = self.create_containers()

        snes = PETSc.SNES().create(comm=self.comm)

        snes.setFunction(self.residual, f)
        snes.setJacobian(self.jacobian, J, J)
        snes.setFromOptions()

        return snes, x

    def set_initial_owned(
        self,
        x: PETSc.Vec,
        values: ArrayLike,
    ) -> None:
        values = np.asarray(values, dtype=PETSc.ScalarType)

        expected = (self.dofs.storage.n_owned,)
        if values.shape != expected:
            raise ValueError(
                f"expected owned vector shape {expected}, got {values.shape}"
            )

        with x.localForm() as xl:
            xl.array[: self.dofs.storage.n_owned] = values

        self.forward(x)

    def set_initial_global(
        self,
        x: PETSc.Vec,
        values: ArrayLike,
    ) -> None:
        values = np.asarray(values)

        if values.shape != (self.n_global,):
            raise ValueError("global initial vector has incorrect shape")

        self.set_initial_owned(x, values[self.dofs.owned_global])

    def set_args(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self) -> tuple:
        return getattr(self, "_args", ())

    @property
    def kwargs(self) -> dict:
        return getattr(self, "_kwargs", {})

    def forward(
        self,
        x: PETSc.Vec,
    ) -> None:
        x.ghostUpdate(
            PETSc.InsertMode.INSERT_VALUES,
            PETSc.ScatterMode.FORWARD,
        )

    def residual(
        self,
        snes: PETSc.SNES,
        x: PETSc.Vec,
        f: PETSc.Vec,
    ) -> None:
        # Owners -> ghosts.
        self.forward(x)

        with x.localForm() as xl:
            z = jnp.asarray(xl.array_r)
            gradient = np.asarray(
                self.gradient(z, *self.args, **self.kwargs), dtype=PETSc.ScalarType
            )

        expected = (self.dofs.storage.local_size,)
        if gradient.shape != expected:
            raise RuntimeError(
                f"local gradient has shape {gradient.shape}; expected {expected}"
            )

        with f.localForm() as fl:
            fl.array[:] = gradient

        # Sum ghost contributions back onto owners.
        f.ghostUpdate(
            PETSc.InsertMode.ADD_VALUES,
            PETSc.ScatterMode.REVERSE,
        )

    def jacobian(
        self,
        snes: PETSc.SNES,
        x: PETSc.Vec,
        J: PETSc.Mat,
        P: PETSc.Mat,
    ) -> None:
        self.forward(x)

        with x.localForm() as xl:
            z = jnp.asarray(xl.array_r)
            local_hessian = self.hessian(z, *self.args, **self.kwargs)

        J.zeroEntries()
        J.setValuesCOO(
            np.asarray(local_hessian.data, dtype=PETSc.ScalarType),
            addv=PETSc.InsertMode.ADD_VALUES,
        )

    def eval(self, x: PETSc.Vec) -> float:
        self.forward(x)

        with x.localForm() as xl:
            z = jnp.asarray(xl.array_r)
            energy = self.energy(z, *self.args, **self.kwargs)

        return energy

    def solve(self, initial_owned: ArrayLike) -> tuple[PETSc.Vec, PETSc.SNES]:
        snes, x = self.create_snes()
        self.set_initial_owned(x, initial_owned)
        snes.solve(None, x)

        return x, snes

    def owned_solution(self, x: PETSc.Vec) -> tuple[NDArray[np.int64], NDArray]:
        n_owned = self.dofs.storage.n_owned

        with x.localForm() as xl:
            values = np.asarray(xl.array_r[:n_owned]).copy()

        return (self.dofs.owned_global.copy(), values)
