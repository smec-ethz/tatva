from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI
from numpy.typing import NDArray

from tatva.compound import (
    Compound,
    CompoundStackError,
    FieldStackedView,
    _apply_stacked_fields,
    _CompoundMeta,
    field,
)
from tatva.mesh import Mesh
from tatva.sparse import create_sparsity_pattern, reduce_sparsity_pattern

if TYPE_CHECKING:
    from petsc4py import PETSc

__all__ = ["DofLayout", "MeshMapping", "StackedCompound", "field"]

log = logging.getLogger(__name__)


class Space(Enum):
    FULL = "full"
    REDUCED = "reduced"


SpaceLiteral = Space | Literal["full", "reduced"]


class MeshMapping(NamedTuple):
    node_l2g: NDArray[np.int32]
    """Local to global node mapping array, where node_l2g[i] gives the global index of the
    local node i."""

    node_ownership: NDArray[np.bool]
    """Boolean array indicating which local nodes are owned by the local process (True)
    and which are ghosts (False)."""


class DofLayout(NamedTuple):
    l2g: NDArray
    """Local to global DOF mapping array, where l2g[i] gives the global index of the local
    DOF i."""

    offset: int
    """Starting global index of the owned DOFs for the local process. They are contiguous
    in the global numbering."""

    n_owned: int
    """Number of local DOFs owned by the local process."""

    n_total: int
    """Number of local DOFs (including ghosts) for the local process."""

    n_global: int
    """Total number of global DOFs across all processes."""

    ownership: NDArray
    """Boolean array indicating which local DOFs are owned by the local process (True) and
    which are ghosts (False)."""


class _StackedCompoundMeta(_CompoundMeta):
    """Metaclass for StackedCompound that automatically stacks all its fields along the last axis."""

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs
    ) -> type:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Skip stacking for the base class itself
        if name == "StackedCompound":
            return cls

        # Filter out fields that might have been added by decorators or other means,
        # ensuring we stack all newly defined fields in the class.
        field_names = [n for n, f in cls.fields if not isinstance(f, FieldStackedView)]
        if field_names:
            try:
                _apply_stacked_fields(cls, tuple(field_names), stack_axis=-1)
            except CompoundStackError as e:
                # Provide a more descriptive error message in the context of StackedCompound
                raise TypeError(
                    f"StackedCompound '{name}' fields are incompatible for stacking: {e}"
                ) from e
        return cls


class StackedCompound(Compound, metaclass=_StackedCompoundMeta):
    """A derivative of Compound where all fields are automatically stacked and interleaved.

    This class enforces that all fields have the same first dimension (typically the
    number of nodes) and stacks them along the last axis. This ensures that all degrees
    of freedom for a single node are topologically close in the flat array.

    Example:
        class MyState(StackedCompound):
            u = field((100, 3))
            p = field((100, 1))

        state = MyState()
        # state.arr.size is 400 (100 * (3 + 1))
        # Indices for node i are [4*i, 4*i+1, 4*i+2, 4*i+3]
    """

    _mesh: Mesh
    _layout: dict[Space, DofLayout | None]
    _layout_reduced: DofLayout | None
    _has_layout: bool = False
    _sparsity_pattern: dict[Space, sps.csr_matrix | None]

    @classmethod
    def attach_mesh(
        cls, mesh: Mesh, mesh_l2g: MeshMapping, comm: MPI.Comm = MPI.COMM_WORLD
    ) -> None:
        """Attach a mesh to the StackedCompound, creating the global DOF layout and
        sparsity pattern.

        Args:
            mesh: The local mesh for the current process.
            mesh_l2g: A MeshMapping containing the node-level local-to-global mapping and ownership
            comm: The MPI communicator
        """
        cls._mesh = mesh
        cls._comm = comm

        _, master_field = cls.fields[0]
        n_dofs_per_node = cls.size // master_field.shape[0]
        cls._n_dofs_per_node = n_dofs_per_node

        log.debug(
            f"Attaching (local) mesh with {mesh.coords.shape[0]} nodes and {n_dofs_per_node} DOFs per node."
        )

        dof_map = _dof_map_from_node_map(mesh_l2g.node_l2g, n_dofs_per_node)
        dof_ownership = np.repeat(mesh_l2g.node_ownership, n_dofs_per_node)

        # store the original dof_map for reordering to global original mesh
        cls._dof_map = dof_map

        # compute the contiguous dof layout
        cls._layout: dict[Space, DofLayout | None] = {
            Space.FULL: None,
            Space.REDUCED: None,
        }
        cls._layout[Space.FULL] = _create_dof_layout(dof_map, dof_ownership, comm)

        if cls.lifter is not None:
            cls._layout[Space.REDUCED] = _reduce_dof_layout(
                cls._layout[Space.FULL], np.asarray(cls.lifter.free_dofs), comm
            )

    @classmethod
    def get_layout(cls, space: SpaceLiteral = Space.FULL) -> DofLayout:
        """Get the global DOF layout for the StackedCompound.

        Args:
            space: If 'reduced', return the layout for the reduced free DOF space.
                Otherwise, return the layout for all DOFs.
        """
        space = Space(space)
        if not hasattr(cls, "_layout"):
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting layout."
            )
        if space == Space.REDUCED and cls._layout[Space.REDUCED] is None:
            raise RuntimeError(
                "Lifter must be attached to StackedCompound before getting reduced layout."
            )
        return cast(DofLayout, cls._layout[space])

    @classmethod
    def get_sparsity_pattern(cls, space: SpaceLiteral = Space.FULL) -> sps.csr_matrix:
        """Get the local sparsity pattern for the Jacobian based on the attached mesh and DOF layout."""
        space = Space(space)

        if not hasattr(cls, "_mesh"):
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting sparsity pattern."
            )

        if not hasattr(cls, "_sparsity_pattern"):
            cls._sparsity_pattern = {Space.FULL: None, Space.REDUCED: None}

        if cls._sparsity_pattern[space] is None:
            layout = cls._layout[space]
            if layout is None:
                raise RuntimeError(
                    f"Lifter must be attached to StackedCompound before getting {space} sparsity pattern."
                )
            cls._sparsity_pattern[Space.FULL] = create_sparsity_pattern(
                cls._mesh, cls._n_dofs_per_node
            )
            if space == Space.REDUCED:
                assert cls.lifter is not None, (
                    "Lifter must be attached for reduced sparsity pattern."
                )
                cls._sparsity_pattern[Space.REDUCED] = reduce_sparsity_pattern(
                    cls._sparsity_pattern[Space.FULL], np.asarray(cls.lifter.free_dofs)
                )

        return cls._sparsity_pattern[space]  # ty:ignore[invalid-return-type]

    @classmethod
    def _get_petsc_lg_map(cls, space: Space = Space.FULL) -> PETSc.LGMap:
        if not hasattr(cls, "_petsc_lg_map"):
            cls._petsc_lg_map = {Space.FULL: None, Space.REDUCED: None}

        if not hasattr(cls, "_layout") and not hasattr(cls, "_comm"):
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting PETSc LGMap."
            )
        from petsc4py import PETSc

        if cls._petsc_lg_map[space] is None:
            layout = cls._layout[space]
            cls._petsc_lg_map[space] = PETSc.LGMap().create(layout.l2g, comm=cls._comm)  # ty:ignore[invalid-argument-type, unresolved-attribute]

        return cls._petsc_lg_map[space]  # ty:ignore[invalid-return-type]

    @classmethod
    def petsc_get_jacobian(cls, space: SpaceLiteral = Space.FULL) -> PETSc.Mat:
        """Get a PETSc matrix for the Jacobian based on the DOF layout.

        Args:
            reduced_space: If True, return a matrix for the reduced free DOF space.
                Otherwise, return for all DOFs.
        """
        space = Space(space)
        if not hasattr(cls, "_sparsity_pattern"):
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting Jacobian."
            )

        if space == Space.REDUCED and cls._layout[Space.REDUCED] is None:
            raise RuntimeError(
                "Lifter must be attached to StackedCompound before getting reduced Jacobian."
            )
        from petsc4py import PETSc

        layout = cast(DofLayout, cls._layout[space])

        J = PETSc.Mat().create(comm=cls._comm)  # ty:ignore[invalid-argument-type]
        J = cast(PETSc.Mat, J)
        J.setSizes(((layout.n_owned, PETSc.DECIDE), (layout.n_owned, PETSc.DECIDE)))
        J.setType(PETSc.Mat.Type.MPIAIJ)

        # Pre-calculate global COO indices for the Jacobian
        layout_reduced = cls.get_layout(space)
        local_sparsity = cls.get_sparsity_pattern(space)
        local_rows = np.repeat(
            np.arange(local_sparsity.shape[0], dtype=np.int32),
            np.diff(local_sparsity.indptr),
        )
        local_cols = local_sparsity.indices
        global_rows = layout_reduced.l2g[local_rows]
        global_cols = layout_reduced.l2g[local_cols]
        J.setPreallocationCOO(global_rows, global_cols)

        J.setLGMap(cls._get_petsc_lg_map(space))
        J.setUp()
        return J

    @classmethod
    def petsc_get_vector(cls, space: SpaceLiteral = Space.FULL) -> PETSc.Vec:
        """Get a PETSc vector for the solution based on the DOF layout.

        Args:
            reduced_space: If True, return a vector for the reduced free DOF space.
                Otherwise, return for all DOFs.
        """
        space = Space(space)
        if not hasattr(cls, "_layout"):
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting vector."
            )
        from petsc4py import PETSc

        layout = cast(DofLayout, cls._layout[space])
        ghost_global_ids = layout.l2g[~layout.ownership]

        # Create the vector: size=(local_owned_size, global_total_size)
        v = PETSc.Vec().createGhost(
            ghost_global_ids.astype(np.int32),
            size=(layout.n_owned, layout.n_global),
            comm=cls._comm,  # ty:ignore[invalid-argument-type]
        )
        return v


def _dof_map_from_node_map(
    node_map: NDArray[np.int32], n_dofs_per_node: int
) -> NDArray[np.int32]:
    """Convert a node map to a degree of freedom (DOF) map.

    Args:
        node_map: An array of node indices.
        n_dofs_per_node: The number of degrees of freedom associated with each node.

    Returns:
        An array of DOF indices corresponding to the input node indices.
    """
    dof_map = node_map[:, None] * n_dofs_per_node + np.arange(n_dofs_per_node)
    return dof_map.flatten().astype(np.int32)


def _create_dof_layout(
    dof_map: NDArray, dof_ownership: NDArray, comm: MPI.Comm
) -> DofLayout:
    """Create a global DOF layout for all DOFs across all processes.

    Args:
        dof_map: An array of original global DOF indices for the local process.
        dof_ownership: A mask indicating if DOFs are owned by the local process.
        comm: The MPI communicator.

    Returns:
        A DofLayout containing the global DOF indices for all DOFs.
    """
    _dtype = np.int32
    n_owned = np.sum(dof_ownership)
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    n_per_rank = comm.allgather(n_owned)
    offset = np.cumsum([0] + n_per_rank[:-1])[comm.rank]

    # Assign global indices to owned DOFs
    l2g = np.full(dof_map.size, -1, dtype=_dtype)
    l2g[dof_ownership] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using the original global IDs (dof_map)
    local_directory = np.full(n_global, -1, dtype=_dtype)
    local_directory[dof_map[dof_ownership]] = l2g[dof_ownership]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    ghost_mask = ~dof_ownership
    l2g[ghost_mask] = global_directory[dof_map[ghost_mask]]

    log.debug(
        f"Rank {comm.rank}: DOF layout - n_owned={n_owned}, n_total={dof_map.size}, n_global={n_global}, offset={offset}"
    )

    return DofLayout(
        l2g=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=dof_map.size,
        n_global=n_global,
        ownership=dof_ownership,
    )


def _reduce_dof_layout(
    all_layout: DofLayout, free_dofs: NDArray, comm: MPI.Comm
) -> DofLayout:
    """Reduce an all-DOF layout to only include free DOFs.

    Args:
        all_layout: The DofLayout for all DOFs.
        free_dofs: Local indices (relative to the original dof_map) that are free.
        comm: The MPI communicator.

    Returns:
        A DofLayout containing the global DOF indices for the free DOFs.
    """
    _dtype = np.int32
    # Create mask for free DOFs
    mask_free = np.zeros(all_layout.n_total, dtype=bool)
    mask_free[free_dofs] = True

    # Mask for free DOFs owned by this rank
    mask_free_owned = mask_free & all_layout.ownership
    n_owned = np.sum(mask_free_owned)
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    # New contiguous global indexing for free DOFs
    n_per_rank = comm.allgather(n_owned)
    offset = np.cumsum([0] + n_per_rank[:-1])[comm.rank]

    l2g = np.full(all_layout.n_total, -1, dtype=_dtype)
    l2g[mask_free_owned] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using all_layout.l2g as the unique global identifier
    local_directory = np.full(all_layout.n_global, -1, dtype=_dtype)
    local_directory[all_layout.l2g[mask_free_owned]] = l2g[mask_free_owned]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    ghost_mask_free = mask_free & ~all_layout.ownership
    l2g[ghost_mask_free] = global_directory[all_layout.l2g[ghost_mask_free]]

    log.debug(
        f"Rank {comm.rank}: Reduced DOF layout - n_owned={n_owned}, n_total={np.sum(mask_free)}, n_global={n_global}, offset={offset}"
    )

    return DofLayout(
        l2g=l2g[mask_free],
        offset=offset,
        n_owned=n_owned,
        n_total=np.sum(mask_free),
        n_global=n_global,
        ownership=mask_free_owned[mask_free],
    )
