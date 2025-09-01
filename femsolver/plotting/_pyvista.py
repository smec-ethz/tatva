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
