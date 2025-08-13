import os

# Path to custom style
STYLE_PATH = os.path.join(os.path.dirname(__file__), "latex_sans_serif.mplstyle")


import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

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
        shading="flat",
        cmap=cmap,
        edgecolors="black",
    )

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="15%", pad=0.05)

    # fig = ax.get_figure()
    # fig.colorbar(cb, cax=cax, label=label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)

    fig = ax.get_figure()
    fig.colorbar(cb, cax=cax, label=label, orientation="horizontal", location="top")

    if ax is None:
        plt.show()
