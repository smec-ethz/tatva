from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, ParamSpec, cast

import jax
import numpy as np
import scipy.sparse as sps
from jax import Array
from mpi4py import MPI
from numpy.typing import NDArray

from tatva import sparse
from tatva.compound import (
    Compound,
    CompoundStackError,
    Field,
    FieldStackedView,
    _apply_stacked_fields,
)
from tatva.compound import field as field
from tatva.lifter import Constraint, Lifter
from tatva.mesh import Mesh, MeshLocal
from tatva.sparse import ColoredMatrix, create_sparsity_pattern, reduce_sparsity_pattern

if TYPE_CHECKING:
    from petsc4py import PETSc

log = logging.getLogger(__name__)


class Space(Enum):
    FULL = "full"
    REDUCED = "reduced"


SpaceLiteral = Space | Literal["full", "reduced"]
P = ParamSpec("P")


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

    comm: MPI.Comm
    dof_map: NDArray[np.int32]
    layouts: dict[Space, _LocalLayout | None]


class FieldSize(IntEnum):
    AUTO = -1


class StackedCompound(Compound):
    """A derivative of Compound where all fields are automatically stacked and interleaved.

    This class enforces that all fields have the same first dimension (typically the
    number of nodes) and stacks them along the last axis. This ensures that all degrees
    of freedom for a single node are topologically close in the flat array.
    """

    MESH: Mesh
    _sparsity_patterns: dict[Space, sps.csr_matrix | None]
    _comm: MPI.Comm = MPI.COMM_WORLD
    _mpi_context: _LocalProblemContext  # remains unset if MESH is not MeshLocal
    size_reduced: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # check if user has assigned a mesh to the class
        if not hasattr(cls, "MESH") or not isinstance(cls.MESH, Mesh):
            raise TypeError(
                f"StackedCompound subclass '{cls.__name__}' must have a 'mesh' attribute of type Mesh."
            )

        # replace FieldSize.AUTO with the number of nodes in the mesh for all fields
        for attr_value in cls.__dict__.values():
            if isinstance(attr_value, Field) and attr_value.shape[0] == FieldSize.AUTO:
                attr_value.shape = (cls.MESH.coords.shape[0],) + attr_value.shape[1:]

        super().__init_subclass__(**kwargs)
        cls.size_reduced = cls.size  # Initialize size_reduced to the full size; it will be updated when constraints are added.

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

        if isinstance(cls.MESH, MeshLocal):
            cls._initialize_mpi_mesh()

    @classmethod
    def _initialize_mpi_mesh(cls) -> None:
        """Internal method to initialize the MPI mesh and related data structures."""
        mesh = cls.MESH
        assert isinstance(mesh, MeshLocal), (
            "MESH must be of type MeshLocal for MPI initialization."
        )
        n_nodes = mesh.coords.shape[0]
        n_dofs_per_node = cls.size // n_nodes
        log.debug(
            f"Initializing MPI map for StackedCompound '{cls.__name__}' with {n_nodes} nodes and {n_dofs_per_node} DOFs per node."
        )

        dof_map = _dof_map_from_node_map(
            mesh.partition_info.nodes_local_to_global, n_dofs_per_node
        )
        n_owned_dofs = mesh.partition_info.n_owned_nodes * n_dofs_per_node

        layouts: dict[Space, _LocalLayout | None] = {
            Space.FULL: _create_dof_layout(dof_map, n_owned_dofs, cls._comm),
            Space.REDUCED: None,
        }

        cls._mpi_context = _LocalProblemContext(
            comm=cls._comm, dof_map=dof_map, layouts=layouts
        )

    @classmethod
    def add_constraints(cls, *args: Constraint) -> None:
        """Add constraints to the internal lifter and invalidate the reduced layout and
        sparsity pattern."""
        existing_constraints: tuple[Constraint, ...] = ()
        if isinstance(cls.lifter, Lifter):
            existing_constraints = cls.lifter.constraints

        cls.lifter = Lifter(cls.size, *chain(existing_constraints, args))
        cls.size_reduced = cls.lifter.size_reduced

        # if this is a parallel compound with an attached mesh, we can immediately create the reduced layout
        if hasattr(cls, "_mpi_context"):
            layouts = cls._mpi_context.layouts
            layouts[Space.REDUCED] = _reduce_dof_layout(
                layouts[Space.FULL],  # ty:ignore[invalid-argument-type]
                np.asarray(cls.lifter.free_dofs),
                cls._comm,
            )

    @classmethod
    def attach_lifter(cls, lifter: Lifter) -> None:
        cls.lifter = lifter
        cls.size_reduced = lifter.size_reduced

        if hasattr(cls, "_mpi_context"):
            layouts = cls._mpi_context.layouts
            layouts[Space.REDUCED] = _reduce_dof_layout(
                layouts[Space.FULL],  # ty:ignore[invalid-argument-type]
                np.asarray(lifter.free_dofs),
                cls._comm,
            )

    @classmethod
    def n_dofs_per_node(cls) -> int:
        """Return the number of degrees of freedom per node, inferred from the shape of the first field."""
        return cls.size // cls.MESH.coords.shape[0]

    @classmethod
    def get_layout(cls, space: SpaceLiteral = Space.FULL) -> _LocalLayout:
        """Get the global DOF layout for the StackedCompound.

        Args:
            space: If 'reduced', return the layout for the reduced free DOF space.
                Otherwise, return the layout for all DOFs.
        """
        if not hasattr(cls, "_mpi_context"):
            raise RuntimeError(
                "MPI context not initialized. Ensure that the MESH is a MeshLocal and that the class is properly initialized."
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
        space = Space(space)

        if cls._sparsity_patterns[space] is None:
            if cls._sparsity_patterns[Space.FULL] is None:
                cls._sparsity_patterns[Space.FULL] = create_sparsity_pattern(
                    cls.MESH, cls.n_dofs_per_node()
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

    @classmethod
    def autodiff(
        cls, fn: Callable[P, Array]
    ) -> tuple[Callable[P, Array], Callable[P, ColoredMatrix]]:
        """Return a tuple of (function, jacobian) where the Jacobian is represented as a ColoredMatrix."""
        if cls.lifter is None:
            space = Space.FULL
        else:
            space = Space.REDUCED

        sparsity = cls.get_sparsity_pattern(space)
        colored_matrix = sparse.ColoredMatrix.from_csr(sparsity)
        n_colors = int(colored_matrix.colors.max() + 1)

        grad_fn = jax.jacrev(fn)
        jac_fn = sparse.jacfwd(grad_fn, colored_matrix, color_batch_size=n_colors)

        return jax.jit(grad_fn), jax.jit(jac_fn)


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


def gather_array_ordered(
    arr_local: NDArray | Array,
    all_layout: _LocalLayout,
    dof_map: NDArray[np.int32],
    comm: MPI.Comm,
) -> NDArray | None:
    """Gathers the distributed active solution into a full global array on Rank 0."""
    # get the owned DOF indices for this rank (these are the first n_owned entries in the local DOF array)
    dofs_owned = dof_map[: all_layout.n_owned]

    # Gather to Rank 0
    all_arr: list = comm.gather(arr_local[: all_layout.n_owned], root=0)  # ty:ignore[invalid-assignment]
    all_ids: list = comm.gather(dofs_owned, root=0)  # ty:ignore[invalid-assignment]

    if comm.rank == 0:
        u_full = np.empty(all_layout.n_global, dtype=arr_local.dtype)
        u_full[np.concatenate(all_ids)] = np.concatenate(all_arr)
        return u_full
    return None
