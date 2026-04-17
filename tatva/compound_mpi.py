from __future__ import annotations

import logging
from dataclasses import dataclass
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
    field,
)
from tatva.mesh import Mesh
from tatva.sparse import create_sparsity_pattern, reduce_sparsity_pattern

if TYPE_CHECKING:
    from petsc4py import PETSc

__all__ = [
    "PartitionInfo",
    "StackedCompound",
    "_LocalLayout",
    "create_petsc_jacobian",
    "create_petsc_vector",
    "field",
    "get_petsc_lg_map",
]

log = logging.getLogger(__name__)


class Space(Enum):
    FULL = "full"
    REDUCED = "reduced"


SpaceLiteral = Space | Literal["full", "reduced"]


class PartitionInfo(NamedTuple):
    nodes_local_to_global: NDArray[np.int32]
    """Local to global node mapping array, where node_l2g[i] gives the global index of the
    local node i."""

    n_owned_nodes: int
    """Number of nodes owned by the local process. Since the local mesh sorts owned nodes
    before ghosts, nodes 0:n_owned_nodes are owned."""


class _LocalLayout(NamedTuple):
    local_to_global: NDArray[np.int32]
    """Local to global DOF mapping array, where l2g[i] gives the global index of the local
    DOF i."""

    offset: int
    """Starting global index of the owned DOFs for the local process. They are contiguous
    in the global numbering."""

    n_owned: int
    """Number of local DOFs owned by the local process. Owned DOFs are implicitly the
    first n_owned entries in the local DOF array."""

    n_total: int
    """Number of local DOFs (including ghosts) for the local process."""

    n_global: int
    """Total number of global DOFs across all processes."""


@dataclass
class _LocalProblemContext:
    """Encapsulates MPI-related state for a StackedCompound subclass."""

    mesh: Mesh
    comm: MPI.Comm
    dof_map: NDArray[np.int32]
    layouts: dict[Space, _LocalLayout | None]


class StackedCompound(Compound):
    """A derivative of Compound where all fields are automatically stacked and interleaved.

    This class enforces that all fields have the same first dimension (typically the
    number of nodes) and stacks them along the last axis. This ensures that all degrees
    of freedom for a single node are topologically close in the flat array.
    """

    mesh: Mesh
    _mpi_context: _LocalProblemContext | None = None
    _sparsity_patterns: dict[Space, sps.csr_matrix | None]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip stacking for the base class itself
        if cls.__name__ == "StackedCompound":
            return

        # Filter out fields that might have been added by decorators or other means,
        # ensuring we stack all newly defined fields in the class.
        field_names = [n for n, f in cls.fields if not isinstance(f, FieldStackedView)]
        if field_names:
            try:
                _apply_stacked_fields(cls, tuple(field_names), stack_axis=-1)
            except CompoundStackError as e:
                # Provide a more descriptive error message in the context of StackedCompound
                raise TypeError(
                    f"StackedCompound '{cls.__name__}' fields are incompatible for stacking: {e}"
                ) from e

        # Initialize sparsity patterns for both full and reduced spaces to None; they will be
        # lazily created when attach_mesh is called.
        cls._sparsity_patterns = {Space.FULL: None, Space.REDUCED: None}

        # check if user has assigned a mesh to the class
        if not hasattr(cls, "mesh") or not isinstance(cls.mesh, Mesh):
            raise TypeError(
                f"StackedCompound subclass '{cls.__name__}' must have a 'mesh' attribute of type Mesh."
            )

    @classmethod
    def n_dofs_per_node(cls) -> int:
        """Return the number of degrees of freedom per node, inferred from the shape of the first field."""
        if not cls.fields:
            raise RuntimeError("No fields defined in StackedCompound.")
        _, master_field = cls.fields[0]
        return cls.size // master_field.shape[0]

    @classmethod
    def attach_mesh(
        cls, mesh: Mesh, mesh_l2g: PartitionInfo, comm: MPI.Comm = MPI.COMM_WORLD
    ) -> None:
        """Attach a mesh to the StackedCompound, creating the global DOF layout and
        sparsity pattern.

        Args:
            mesh: The local mesh for the current process.
            mesh_l2g: A PartitionInfo containing the node-level local-to-global mapping and ownership
            comm: The MPI communicator
        """
        n_dofs_per_node = cls.n_dofs_per_node()

        log.debug(
            f"Attaching (local) mesh with {mesh.coords.shape[0]} nodes and {n_dofs_per_node} DOFs per node."
        )

        dof_map = _dof_map_from_node_map(
            mesh_l2g.nodes_local_to_global, n_dofs_per_node
        )
        n_owned_dofs = mesh_l2g.n_owned_nodes * n_dofs_per_node

        layouts: dict[Space, _LocalLayout | None] = {
            Space.FULL: _create_dof_layout(dof_map, n_owned_dofs, comm),
            Space.REDUCED: None,
        }

        if cls.lifter is not None:
            layouts[Space.REDUCED] = _reduce_dof_layout(
                layouts[Space.FULL],  # ty:ignore[invalid-argument-type]
                np.asarray(cls.lifter.free_dofs),
                comm,
            )

        cls._mpi_context = _LocalProblemContext(
            mesh=mesh,
            comm=comm,
            dof_map=dof_map,
            layouts=layouts,
        )

    @classmethod
    def get_layout(cls, space: SpaceLiteral = Space.FULL) -> _LocalLayout:
        """Get the global DOF layout for the StackedCompound.

        Args:
            space: If 'reduced', return the layout for the reduced free DOF space.
                Otherwise, return the layout for all DOFs.
        """
        if cls._mpi_context is None:
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting layout."
            )
        space = Space(space)
        layout = cls._mpi_context.layouts[space]
        if layout is None:
            raise RuntimeError(
                f"Lifter must be attached to StackedCompound before getting {space.value} layout."
            )
        return layout

    @classmethod
    def get_sparsity_pattern(cls, space: SpaceLiteral = Space.FULL) -> sps.csr_matrix:
        """Get the local sparsity pattern for the Jacobian based on the attached mesh and DOF layout."""
        if cls._mpi_context is None:
            raise RuntimeError(
                "Mesh must be attached to StackedCompound before getting sparsity pattern."
            )
        space = Space(space)

        if cls._sparsity_patterns[space] is None:
            if cls._sparsity_patterns[Space.FULL] is None:
                cls._sparsity_patterns[Space.FULL] = create_sparsity_pattern(
                    cls._mpi_context.mesh, cls.n_dofs_per_node()
                )

            if space == Space.REDUCED:
                if cls.lifter is None:
                    raise RuntimeError(
                        "Lifter must be attached for reduced sparsity pattern."
                    )
                cls._sparsity_patterns[Space.REDUCED] = reduce_sparsity_pattern(
                    cls._sparsity_patterns[Space.FULL],  # ty:ignore[invalid-argument-type]
                    np.asarray(cls.lifter.free_dofs),
                )

        return cast(sps.csr_matrix, cls._sparsity_patterns[space])


def get_petsc_lg_map(layout: _LocalLayout, comm: MPI.Comm) -> PETSc.LGMap:
    """Create a PETSc local-to-global mapping from a DOF layout."""
    from petsc4py import PETSc

    return PETSc.LGMap().create(layout.local_to_global, comm=comm)  # ty:ignore[invalid-argument-type]


def create_petsc_jacobian(
    layout: _LocalLayout,
    sparsity: sps.csr_matrix,
    comm: MPI.Comm,
    lg_map: PETSc.LGMap | None = None,
) -> PETSc.Mat:
    """Create a PETSc matrix for the Jacobian based on the DOF layout and sparsity pattern."""
    from petsc4py import PETSc

    J = PETSc.Mat().create(comm=comm)  # ty:ignore[invalid-argument-type]
    J = cast(PETSc.Mat, J)
    J.setSizes(((layout.n_owned, PETSc.DECIDE), (layout.n_owned, PETSc.DECIDE)))
    J.setType(PETSc.Mat.Type.MPIAIJ)

    # Pre-calculate global COO indices for the Jacobian
    local_rows = np.repeat(
        np.arange(sparsity.shape[0], dtype=np.int32),
        np.diff(sparsity.indptr),
    )
    local_cols = sparsity.indices
    global_rows = layout.local_to_global[local_rows]
    global_cols = layout.local_to_global[local_cols]
    J.setPreallocationCOO(global_rows, global_cols)

    if lg_map is not None:
        J.setLGMap(lg_map)

    J.setUp()
    return J


def create_petsc_vector(layout: _LocalLayout, comm: MPI.Comm) -> PETSc.Vec:
    """Create a PETSc vector with ghost slots based on the DOF layout."""
    from petsc4py import PETSc

    ghost_global_ids = layout.local_to_global[layout.n_owned :]

    # Create the vector: size=(local_owned_size, global_total_size)
    v = PETSc.Vec().createGhost(
        ghost_global_ids.astype(np.int32),  # ty:ignore[invalid-argument-type]
        size=(layout.n_owned, layout.n_global),
        comm=comm,  # ty:ignore[invalid-argument-type]
    )
    return v


def _dof_map_from_node_map(
    node_map: NDArray[np.int32], n_dofs_per_node: int
) -> NDArray[np.int32]:
    """Convert a node map to a degree of freedom (DOF) map."""
    dof_map = node_map[:, None] * n_dofs_per_node + np.arange(n_dofs_per_node)
    return dof_map.flatten().astype(np.int32)


def _create_dof_layout(
    dof_map: NDArray[np.int32], n_owned: int, comm: MPI.Comm
) -> _LocalLayout:
    """Create a global DOF layout for all DOFs across all processes."""
    _dtype = np.int32
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    n_per_rank = comm.allgather(n_owned)
    offset = np.cumsum([0] + n_per_rank[:-1])[comm.rank]

    # Assign global indices to owned DOFs
    l2g = np.full(dof_map.size, -1, dtype=_dtype)
    l2g[:n_owned] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using the original global IDs (dof_map)
    local_directory = np.full(n_global, -1, dtype=_dtype)
    local_directory[dof_map[:n_owned]] = l2g[:n_owned]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    l2g[n_owned:] = global_directory[dof_map[n_owned:]]

    log.debug(
        f"Rank {comm.rank}: DOF layout - n_owned={n_owned}, n_total={dof_map.size}, n_global={n_global}, offset={offset}"
    )

    return _LocalLayout(
        local_to_global=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=dof_map.size,
        n_global=n_global,
    )


def _reduce_dof_layout(
    all_layout: _LocalLayout, free_dofs: NDArray[np.int32], comm: MPI.Comm
) -> _LocalLayout:
    """Reduce an all-DOF layout to only include free DOFs."""
    _dtype = np.int32
    # Mask for free DOFs owned by this rank
    mask_free_owned = free_dofs < all_layout.n_owned
    n_owned = int(np.sum(mask_free_owned))
    n_global = comm.allreduce(n_owned, op=MPI.SUM)

    # New contiguous global indexing for free DOFs
    n_per_rank = comm.allgather(n_owned)
    offset = np.cumsum([0] + n_per_rank[:-1])[comm.rank]

    l2g = np.full(free_dofs.size, -1, dtype=_dtype)
    l2g[:n_owned] = offset + np.arange(n_owned, dtype=_dtype)

    # Resolve ghosts using all_layout.l2g as the unique global identifier
    local_directory = np.full(all_layout.n_global, -1, dtype=_dtype)
    # The global ID of the owned free DOFs in the ALL-DOF layout
    local_directory[all_layout.local_to_global[free_dofs[:n_owned]]] = l2g[:n_owned]

    global_directory = np.empty_like(local_directory)
    comm.Allreduce(local_directory, global_directory, op=MPI.MAX)

    l2g[n_owned:] = global_directory[all_layout.local_to_global[free_dofs[n_owned:]]]

    log.debug(
        f"Rank {comm.rank}: Reduced DOF layout - n_owned={n_owned}, n_total={free_dofs.size}, n_global={n_global}, offset={offset}"
    )

    return _LocalLayout(
        local_to_global=l2g,
        offset=offset,
        n_owned=n_owned,
        n_total=free_dofs.size,
        n_global=n_global,
    )
