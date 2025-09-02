# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
#
# This file is part of femsolver.
#
# femsolver is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# femsolver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with femsolver.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pyvista as pv


def get_pyvista_grid(mesh, cell_type="quad"):
    pv_points = np.hstack((mesh.coords, np.zeros(shape=(mesh.coords.shape[0], 1))))

    cell_type_dict = {
        "quad": 4,
        "triangle": 3,
    }

    pv_cells = np.hstack(
        (
            np.full(
                fill_value=cell_type_dict[cell_type], shape=(mesh.elements.shape[0], 1)
            ),
            mesh.elements,
        )
    )

    pv_cell_type_dict = {
        "quad": pv.CellType.QUAD,
        "triangle": pv.CellType.TRIANGLE,
    }
    cell_types = np.full(
        fill_value=pv_cell_type_dict[cell_type], shape=(mesh.elements.shape[0],)
    )

    grid = pv.UnstructuredGrid(pv_cells, cell_types, pv_points)

    return grid
