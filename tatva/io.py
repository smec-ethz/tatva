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
from typing import NamedTuple

import gmsh
import numpy as np
from numpy.typing import NDArray

from tatva.mesh import Mesh, PartitionInfo, extract_local_mesh

model = gmsh.model


def read_mesh_from_file(file_path: str, rank: int) -> tuple[Mesh, PartitionInfo | None]:
    """Read a mesh from a file. (Format: gmsh 4.1)

    Args:
        file_path: The path to the mesh file.

    Returns:
        A tuple of (mesh, partition_info) where: mesh is a Mesh object containing the
        coordinates and connectivity; partition_info contains metadata about node ownership
        and global indexing if the mesh is partitioned, or None if the mesh is not
        partitioned.
    """
    with gmsh_context(mute=True):
        gmsh.open(file_path)
        coords = extract_coordinates()
        topology, physical_groups = extract_topology()
        all_elements = extract_global_elements(topology)
        element_partition = extract_element_partition(topology)

    mesh_global = Mesh(coords, all_elements)
    mesh, partition_info = extract_local_mesh(mesh_global, element_partition, rank)
    return mesh, partition_info


@contextmanager
def gmsh_context(mute: bool = True):
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


def extract_coordinates():
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

    for dim, tag in model.getPhysicalGroups():
        entities = model.getEntitiesForPhysicalGroup(dim, tag)

        name = model.getPhysicalName(dim, tag)
        physical_groups[name] = PhysicalGroup(dim, tag, entities)

        for entity in entities:
            parts = model.getPartitions(dim, entity)
            if len(parts) == 0:
                # Entity is not partitioned, proceed with normal extraction
                continue

            part = int(parts[0]) - 1
            (el_types, el_tags, el_node_tags) = model.mesh.getElements(dim, entity)

            for el_type, el_tag, conn in zip(
                el_types, el_tags, el_node_tags, strict=True
            ):
                # 3rd index is n_nodes per element
                _, dim, _, n_nodes_per_element, _, _ = model.mesh.getElementProperties(
                    el_type
                )
                conn = np.asarray(conn).reshape(-1, n_nodes_per_element)
                conn -= 1  # Shift from 1-based to 0-based indexing

                marker = np.full_like(el_tag, tag)

                topologies.setdefault(part, {})[int(el_type)] = TopologyData(
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

    for part_id, rank_data in topology.items():
        for data in rank_data.values():
            element_partition[data.element_ids] = part_id

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
