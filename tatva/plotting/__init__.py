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

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from jax import Array

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.tri import Triangulation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tatva.plotting requires the optional dependency 'matplotlib'. "
        "Install it with `pip install tatva[plotting]` or `pip install matplotlib`."
    ) from exc

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = object


try:
    from ._pyvista import get_pyvista_grid as get_pyvista_grid
except ImportError:
    get_pyvista_grid = None


def plot_element_values(
    coords: Array,
    elements: Array,
    values: Array,
    *,
    u: Optional[Array] = None,
    ax: Optional[Axes] = None,
    scale: float = 1.0,
    label: Optional[str] = None,
    cmap="managua_r",
    **kwargs,
):
    """
    Plot the element values of a field on a mesh.

    Args:
        coords: The coordinates of the mesh nodes.
        elements: The connectivity of the mesh elements.
        values: The element values to plot.
        u: The displacement field.
        ax: The axes to plot on.
        scale: The scale of the displacement field.
        label: The label of the colorbar.
        cmap: The colormap to use.
        **kwargs : Additional keyword arguments to pass to PolyCollection.
    """

    if u is not None:
        coords = coords + scale * u

    vertices = coords[elements]

    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

    poly = PolyCollection(vertices, cmap=cmap, linewidths=0.3, **kwargs)  # pyright: ignore[reportArgumentType]
    poly.set_array(values.flatten())
    poly.set_edgecolor("face")
    ax.add_collection(poly)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)
    plt.colorbar(poly, cax=cax, label=label, orientation="horizontal", location="top")


def plot_nodal_values(
    coords: Array,
    elements: Array,
    values: Array,
    *,
    u: Optional[Array] = None,
    ax: Optional[Axes] = None,
    scale: float = 1.0,
    label: Optional[str] = None,
    cmap="managua_r",
    edgecolors: str = "none",
    shading: Literal["flat", "gouraud"] = "gouraud",
    **kwargs,
):
    """
    Plot the nodal values of a field on a mesh.

    Args:
        coords: The coordinates of the mesh nodes.
        elements: The connectivity of the mesh elements.
        values: The element values to plot.
        u: The displacement field.
        ax: The axes to plot on.
        scale: The scale of the displacement field.
        label: The label of the colorbar.
        cmap: The colormap to use.
        edgecolors: The edge colors of the triangles.
        shading: The shading of the triangles.
        **kwargs: Additional keyword arguments to pass to tripcolor.
    """

    coords_np = np.asarray(coords)
    if u is not None:
        coords_np = coords_np + scale * np.asarray(u)

    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

    elements_np = np.asarray(elements)

    if elements_np.shape[1] == 3:
        triangles = elements_np
    elif elements_np.shape[1] == 4:
        # Split each quad into two triangles along the (0, 2) diagonal.
        first_half = elements_np[:, [0, 1, 2]]
        second_half = elements_np[:, [0, 2, 3]]
        triangles = np.concatenate((first_half, second_half), axis=0)
    else:
        raise ValueError(
            "Only triangular or quadrilateral elements are supported for nodal values."
        )

    triangles = np.asarray(triangles, dtype=int)
    tri = Triangulation(coords_np[:, 0], coords_np[:, 1], triangles)

    cb = ax.tripcolor(
        tri,
        np.asarray(values),
        shading=shading,
        cmap=cmap,
        edgecolors=edgecolors,
        **kwargs,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.2)
    plt.colorbar(cb, cax=cax, label=label, orientation="horizontal", location="top")
