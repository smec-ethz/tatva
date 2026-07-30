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

from __future__ import annotations

import numpy as np

__all__ = ["vertex_permutation"]


TATVA_VERTEX_ORDER: dict[str, np.ndarray] = {
    "interval": np.array([[0.0], [1.0]]),
    "triangle": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    "quadrilateral": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    "tetrahedron": np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    "hexahedron": np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    ),
}

# Reference vertices are exact binary fractions in both conventions, so the match should be
# exact; the tolerance only absorbs a table entry written in decimal form.
_MATCH_TOL = 1e-12


def vertex_permutation(cell: str) -> np.ndarray:
    """Returns the permutation taking tatva's vertex order to basix's.

    `perm[i]` is the column of `Mesh.elements` holding basix's local vertex `i`, so
    `mesh.elements[:, perm]` lists each cell's vertices the way a basix element expects.
    Global node numbering is untouched — only the order within each row changes.

    Args:
        cell: basix cell name, e.g. `"quadrilateral"`.

    Returns:
        An integer array of length `n_vertices`. It is the identity for simplices, whose
        conventions already agree.

    Raises:
        KeyError: If `cell` has no recorded tatva vertex order.
        RuntimeError: If the two conventions do not describe the same vertex set, which
            means one of them changed and this module is out of date.
    """
    import basix

    try:
        tatva_order = TATVA_VERTEX_ORDER[cell]
    except KeyError:
        raise KeyError(
            f"no tatva vertex order recorded for cell {cell!r}; known cells are "
            f"{sorted(TATVA_VERTEX_ORDER)}"
        ) from None

    basix_order = np.asarray(basix.geometry(basix.CellType[cell]), dtype=float)
    if basix_order.shape != tatva_order.shape:
        raise RuntimeError(
            f"cell {cell!r} has {basix_order.shape[0]} vertices in basix but "
            f"{tatva_order.shape[0]} in TATVA_VERTEX_ORDER"
        )

    # check that the vertex sets correspond one-to-one
    distance = np.abs(basix_order[:, None, :] - tatva_order[None, :, :]).sum(axis=2)
    perm = np.argmin(distance, axis=1)  # each row with 0 entry gives the column index

    # A permutation is the only acceptable answer. Anything else means the vertex sets
    # genuinely differ — e.g. the reference cells are not the same domain — and no
    # reordering can bridge them, so say so instead of returning a plausible-looking map.
    if sorted(perm.tolist()) != list(range(len(perm))):
        raise RuntimeError(
            f"tatva and basix vertices for cell {cell!r} do not correspond one-to-one; "
            f"nearest-match gave {perm.tolist()}, which is not a permutation"
        )
    gap = distance[np.arange(len(perm)), perm].max()
    if gap > _MATCH_TOL:
        raise RuntimeError(
            f"tatva and basix reference vertices for cell {cell!r} do not coincide "
            f"(largest gap {gap:.3e})"
        )

    return perm.astype(np.int64)
