import os

# Path to custom style
STYLE_PATH = os.path.join(os.path.dirname(__file__), "latex_sans_serif.mplstyle")


import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import pyvista as pv
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from jax import Array
from femsolver import Mesh
from typing import Optional


def plot_element_values(
    mesh: Mesh,
    values: Array,
    u: Optional[Array] = None,
    ax: Optional[mpl.axes.Axes] = None,
    scale: float = 1.0,
    label: Optional[str] = None,
    cmap=cmc.managua_r,
):
    """
    Plot the element values of a field on a mesh.

    Args:
        u : The displacement field.
        mesh : The mesh.
        values : The element values to plot.
        ax : The axes to plot on.
        scale : The scale of the displacement field.
        label : The label of the colorbar.
        cmap : The colormap to use.
    """

    if u is not None:
        displaced = mesh.coords + scale * u
    else:
        displaced = mesh.coords
    tri_elements = mesh.elements
    vertices = displaced[tri_elements]

    plt.style.use(STYLE_PATH)
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()

    poly = mpl.collections.PolyCollection(vertices, cmap=cmap)
    poly.set_array(values.flatten())
    poly.set_edgecolor("face")
    poly.set_lw(0.3)
    ax.add_collection(poly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)

    fig = ax.get_figure()
    fig.colorbar(poly, cax=cax, label=label, orientation="horizontal", location="top")
    if ax is None:
        plt.show()


def plot_nodal_values(
    mesh: Mesh,
    nodal_values: Array,
    u: Optional[Array] = None,
    ax: Optional[mpl.axes.Axes] = None,
    scale: float = 1.0,
    label: Optional[str] = None,
    cmap=cmc.managua_r,
    edgecolors: Optional[str] = "none",
    shading: Optional[str] = "gouraud",
):
    """
    Plot the nodal values of a field on a mesh.

    Args:
        u : The displacement field.
        mesh : The mesh.
        nodal_values : The nodal values to plot.
        ax : The axes to plot on.
        scale : The scale of the displacement field.
        label : The label of the colorbar.
        cmap : The colormap to use.
    """

    if u is not None:
        displaced = mesh.coords + scale * u
    else:
        displaced = mesh.coords
    tri_elements = mesh.elements

    plt.style.use(STYLE_PATH)
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()

    cb = ax.tripcolor(
        displaced[:, 0],
        displaced[:, 1],
        tri_elements,
        nodal_values,
        shading=shading,
        cmap=cmap,
        edgecolors=edgecolors,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)

    fig = ax.get_figure()
    fig.colorbar(cb, cax=cax, label=label, orientation="horizontal", location="top")

    if ax is None:
        plt.show()



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