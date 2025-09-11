# Copyright (C) 2025 ETH Zurich (Mohit Pundir)
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


import os

# Path to custom style
STYLE_PATH = os.path.join(os.path.dirname(__file__), "latex_sans_serif.mplstyle")


from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import Array
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tatva import Mesh

try:
    from ._pyvista import get_pyvista_grid
except ImportError:
    get_pyvista_grid = None


def plot_element_values(
    mesh: Mesh,
    values: Array,
    u: Optional[Array] = None,
    ax: Optional[mpl.axes.Axes] = None,
    scale: float = 1.0,
    label: Optional[str] = None,
    cmap="managua_r",
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
    cmap="managua_r",
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

