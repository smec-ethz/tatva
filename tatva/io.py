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

"""Mesh I/O utilities for tatva."""

from contextlib import contextmanager
from typing import NamedTuple, overload

import numpy as np
from numpy.typing import NDArray

from tatva.mesh import Mesh, PartitionInfo, extract_local_mesh

try:
    import gmsh

    model = gmsh.model
except ImportError as e:
    raise ImportError(
        "The 'gmsh' Python API is required to use tatva's mesh I/O utilities. "
        "Please install gmsh and ensure that the Python API is available."
    ) from e

__all__ = [
    "extract_mesh",
    "extract_mesh_from_file",
]


@overload
def extract_mesh_from_file(file_path: str, rank: int) -> tuple[Mesh, PartitionInfo]: ...
@overload
def extract_mesh_from_file(file_path: str, rank: None = None) -> tuple[Mesh, None]: ...
def extract_mesh_from_file(
    file_path: str, rank: int | None = None
) -> tuple[Mesh, PartitionInfo | None]:
    """Read a mesh from a file. (Format: gmsh 4.1)

    Args:
        file_path: The path to the mesh file.

    Returns:
        A tuple of (mesh, partition_info) where: mesh is a Mesh object containing the
        coordinates and connectivity; partition_info contains metadata about node ownership
        and global indexing if the mesh is partitioned, or None if the mesh is not
        partitioned.
    """
    with _gmsh_context(mute=True):
        gmsh.open(file_path)
        return extract_mesh(None, rank)


@overload
def extract_mesh(model_name: str | None, rank: int) -> tuple[Mesh, PartitionInfo]: ...
@overload
def extract_mesh(model_name: str | None, rank: None = None) -> tuple[Mesh, None]: ...
def extract_mesh(
    model_name: str | None = None, rank: int | None = None
) -> tuple[Mesh, PartitionInfo | None]:
    """Extract the mesh from the currently loaded gmsh model.

    Args:
        model_name: The name of the gmsh model to extract the mesh from. If None, the current
            model will be used.
        rank: The rank of the current process (if the mesh is partitioned). If None, the
            mesh is assumed to be unpartitioned.

    Returns:
        A tuple of (mesh, partition_info) where: mesh is a Mesh object containing the
        coordinates and connectivity; partition_info contains metadata about node ownership
        and global indexing if the mesh is partitioned, or None if the mesh is not
        partitioned.
    """
    if model_name is not None:
        gmsh.model.setCurrent(model_name)

    coords = extract_coordinates()
    topology, physical_groups = extract_topology()
    all_elements = extract_global_elements(topology)
    element_partition = extract_element_partition(topology)
    num_partitions = gmsh.model.getNumberOfPartitions()

    if (num_partitions == 0) or (rank is None):
        mesh = Mesh(coords, all_elements)  # ty:ignore[invalid-argument-type]
        return mesh, None
    else:
        mesh_global = Mesh(coords, all_elements)  # ty:ignore[invalid-argument-type]
        mesh, partition_info = extract_local_mesh(mesh_global, element_partition, rank)
        return mesh, partition_info


@contextmanager
def _gmsh_context(mute: bool = True):
    gmsh.initialize()
    if mute:
        gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield
    finally:
        gmsh.finalize()


class PhysicalGroup(NamedTuple):
    dim: int
    tag: int
    entities: list[int]  # list of entity tags that belong to this physical group


class TopologyData(NamedTuple):
    element_ids: NDArray[np.int_]
    connectivity: NDArray[np.int_]
    cell_type: NDArray[np.int_]
    marker: NDArray[np.int_]
    dim: int
    n_nodes_per_element: int


RankwiseTopologyData = dict[int, dict[int, TopologyData]]
PhysicalGroupsData = dict[str, PhysicalGroup]


def extract_coordinates() -> NDArray[np.float64]:
    """Extract the mesh coordinates from a gmsh model."""
    dim = model.getDimension()
    indices, coords, _ = model.mesh.getNodes()

    # coords is a flat array of length num_nodes * dim, we need to reshape it to
    # (num_nodes, dim)
    coords = coords.reshape(-1, dim)

    # gmsh indices start at 1, so we need to shift them to start at 0
    indices -= 1

    sorted_indices = np.argsort(indices)
    return coords[sorted_indices]


def extract_topology() -> tuple[RankwiseTopologyData, PhysicalGroupsData]:
    """Extract the topology of the mesh from the gmsh model."""
    topologies: RankwiseTopologyData = {}
    physical_groups: PhysicalGroupsData = {}

    max_dim = model.getDimension()
    is_partitioned = model.getNumberOfPartitions() > 0

    # 1. Populate physical groups metadata
    for dim, tag in model.getPhysicalGroups():
        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        name = model.getPhysicalName(dim, tag)
        physical_groups[name] = PhysicalGroup(dim, tag, entities)

    # 2. Determine which entities of max_dim we should extract elements from
    max_dim_physical_entities = []
    for dim, tag in model.getPhysicalGroups():
        if dim == max_dim:
            max_dim_physical_entities.extend(
                model.getEntitiesForPhysicalGroup(dim, tag)
            )

    if len(max_dim_physical_entities) > 0:
        entities_to_extract = list(dict.fromkeys(max_dim_physical_entities))
    else:
        entities_to_extract = [tag for dim, tag in model.getEntities(max_dim)]

    # Map entity tag of max_dim to its physical group tag
    entity_to_tag = {}
    for dim, tag in model.getPhysicalGroups():
        if dim == max_dim:
            for entity in model.getEntitiesForPhysicalGroup(dim, tag):
                entity_to_tag[entity] = tag

    # 3. Extract elements from these entities
    for entity in entities_to_extract:
        if is_partitioned:
            parts = model.getPartitions(max_dim, entity)
            if len(parts) == 0:
                continue
            part = int(parts[0]) - 1
        else:
            part = 0

        tag = entity_to_tag.get(entity, 0)
        el_types, el_tags, el_node_tags = model.mesh.getElements(max_dim, entity)

        for el_type, el_tag, conn in zip(el_types, el_tags, el_node_tags, strict=True):
            _, dim, _, n_nodes_per_element, _, _ = model.mesh.getElementProperties(
                el_type
            )
            conn = np.asarray(conn).reshape(-1, n_nodes_per_element)
            conn -= 1  # Shift from 1-based to 0-based indexing

            marker = np.full_like(el_tag, tag)

            # Store / Concatenate
            if el_type in topologies.setdefault(part, {}):
                existing = topologies[part][el_type]
                topologies[part][el_type] = TopologyData(
                    element_ids=np.concatenate(
                        [existing.element_ids, np.asarray(el_tag, dtype=np.int32) - 1]
                    ),
                    connectivity=np.concatenate([existing.connectivity, conn], axis=0),
                    cell_type=el_type,
                    marker=np.concatenate([existing.marker, marker]),
                    dim=dim,
                    n_nodes_per_element=n_nodes_per_element,
                )
            else:
                topologies[part][el_type] = TopologyData(
                    element_ids=np.asarray(el_tag, dtype=np.int32) - 1,
                    connectivity=conn,
                    cell_type=el_type,
                    marker=marker,
                    dim=dim,
                    n_nodes_per_element=n_nodes_per_element,
                )

    return topologies, physical_groups


def extract_element_partition(topology: RankwiseTopologyData) -> NDArray[np.int8]:
    """Extract an array mapping each element to its partition ID."""
    num_elements = sum(
        data.element_ids.size
        for rank_data in topology.values()
        for data in rank_data.values()
    )
    element_partition = np.full(num_elements, -1, dtype=np.int8)

    current_index = 0
    for part_id, rank_data in topology.items():
        for data in rank_data.values():
            n_elements = data.element_ids.size
            element_partition[current_index : current_index + n_elements] = part_id
            current_index += n_elements

    return element_partition


def extract_global_elements(topology: RankwiseTopologyData) -> NDArray[np.int32]:
    """Extract a global connectivity array for all elements in the mesh."""
    num_elements = sum(
        data.element_ids.size
        for rank_data in topology.values()
        for data in rank_data.values()
    )
    max_nodes_per_element = max(
        data.n_nodes_per_element
        for rank_data in topology.values()
        for data in rank_data.values()
    )
    global_elements = np.full((num_elements, max_nodes_per_element), -1, dtype=np.int32)

    current_index = 0
    for rank_data in topology.values():
        for data in rank_data.values():
            n_elements = data.element_ids.size
            global_elements[
                current_index : current_index + n_elements, : data.n_nodes_per_element
            ] = data.connectivity
            current_index += n_elements

    return global_elements
