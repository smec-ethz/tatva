import os

# Path to custom style
STYLE_PATH = os.path.join(
    os.path.dirname(__file__), "latex_sans_serif.mplstyle"
)


import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_element_values(
    u, mesh, values, ax=None, scale=1.0, label=None, cmap=cmc.managua_r
):
    displaced = mesh.coords + scale * u
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
    u, mesh, nodal_values, ax=None, scale=1.0, label=None, cmap=cmc.managua_r
):
    displaced = mesh.coords + scale * u
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

    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="15%", pad=0.05)

    #fig = ax.get_figure()
    #fig.colorbar(cb, cax=cax, label=label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)

    fig = ax.get_figure()
    fig.colorbar(cb, cax=cax, label=label, orientation="horizontal", location="top")

    if ax is None:
        plt.show()
