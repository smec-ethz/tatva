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

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import basix
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from tatva.element.base import Element, MapType, Sobolev

__all__ = ["BasixElement", "Quadrature", "from_basix"]

# Oversampling and seed for the polyset -> monomial least-squares fit. Fixed so that a
# given (family, cell, degree, variant) always produces bit-identical coefficients.
_FIT_OVERSAMPLING = 4
_FIT_SEED = 0

# The extracted basis is checked against basix.tabulate before being accepted.
_EXTRACTION_TOL = 1e-9

_SIMPLEX_CELLS = frozenset({"interval", "triangle", "tetrahedron"})
_TENSOR_CELLS = frozenset({"quadrilateral", "hexahedron"})


@dataclass(frozen=True, eq=False)
class Quadrature:
    """A quadrature rule, optionally tagged with the cell it was built for.

    Compares and hashes by exact array contents, so two rules built by the same call are
    interchangeable keys. `frozen=True` is load-bearing: `__post_init__` caches the digest
    once, and a rule mutated afterwards would keep hashing to its old value.

    Attributes:
        quad_points: Quadrature points in local coordinates (shape: (n_q, tdim)).
        quad_weights: Quadrature weights (shape: (n_q,)).
        cell: The reference cell the rule integrates over, when known. `rule` fills this
            in; a rule built by hand leaves it `None`, which means "the caller asserts
            this rule is right" and skips the cell check in `from_basix`.
    """

    quad_points: Array
    quad_weights: Array
    cell: basix.CellType | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quad_points", jnp.asarray(self.quad_points))
        object.__setattr__(self, "quad_weights", jnp.asarray(self.quad_weights))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, Quadrature)
        return (
            np.array_equal(self.quad_points, other.quad_points)
            and np.array_equal(self.quad_weights, other.quad_weights)
            and self.cell == other.cell
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.cell,
                np.array(self.quad_points).tobytes(),
                np.array(self.quad_weights).tobytes(),
            )
        )

    @classmethod
    def rule(
        cls,
        cell: basix.CellType | str,
        quadrature_degree: int,
        quadrature_type: basix.QuadratureType | str = basix.QuadratureType.default,
    ) -> Quadrature:
        if isinstance(quadrature_type, str):
            quadrature_type = basix.quadrature.string_to_type(quadrature_type)

        if isinstance(cell, str):
            cell = basix.CellType[cell]

        quad_points, quad_weights = basix.make_quadrature(
            cell,
            degree=quadrature_degree,
            rule=quadrature_type,
        )
        return cls(jnp.asarray(quad_points), jnp.asarray(quad_weights), cell)


def from_basix(
    family: str,
    cell: str,
    degree: int,
    quadrature_rule: Quadrature,
    *,
    lagrange_variant: basix.LagrangeVariant = basix.LagrangeVariant.unset,
    dpc_variant: basix.DPCVariant = basix.DPCVariant.unset,
    discontinuous: bool = False,
    dof_ordering: list[int] | None = None,
    dtype: npt.DTypeLike | None = np.float64,
    geometry_degree: int | None = None,
) -> BasixElement:
    """Builds a tatva `Element` from a basix element definition.

    basix is imported here and dropped before returning; the result holds only frozen
    arrays and is usable anywhere a hand-written element is.

    Args:
        family: basix family name, e.g. `"P"`, `"RT"`, `"N1E"`, `"N2E"`, `"BDM"`.
        cell: basix cell name, e.g. `"triangle"`, `"tetrahedron"`, `"quadrilateral"`.
        degree: Element degree.
        variant: Lagrange variant name, e.g. `"gll_isaac"`. basix requires one for
            Lagrange elements of degree > 2.
        discontinuous: Request the discontinuous (L2) variant of the family.
        quadrature: An explicit `(points, weights)` rule. Pass the same rule to every
            space appearing in one integral.
        quadrature_degree: Degree of the rule to generate when `quadrature` is not given.
            Defaults to `degree`; pass a higher value for integrands that are not
            polynomials of the element degree.
        geometry_degree: Degree of the scalar Lagrange element describing the cell
            geometry, which is what `get_jacobian` tabulates. Defaults to `degree` for
            identity-mapped scalar families (isoparametric, matching the hand-written
            elements) and to 1 for Piola-mapped families, whose own degree says nothing
            about the geometry. Pass 1 explicitly for a straight-sided mesh carrying a
            higher-degree field.
        dof_ordering: Optional permutation reordering basix's dofs into a tatva
            convention, e.g. `[0, 1, 3, 2]` to turn basix's lexicographic quadrilateral
            vertices into tatva's counter-clockwise `Quad4` order. Only needed when
            re-deriving an element tatva already has; new families have no prior
            convention and should keep basix's ordering.

    Returns:
        A `BasixElement` reproducing the basix basis to machine precision.

    Raises:
        ImportError: If basix is not installed.
        RuntimeError: If the extracted basis does not reproduce `basix.tabulate`.
    """

    cell_type = basix.CellType[cell]

    if quadrature_rule.cell is not None and quadrature_rule.cell != cell_type:
        raise ValueError(
            f"quadrature rule was built for {quadrature_rule.cell.name!r} but the "
            f"element is on {cell!r}; build the rule with "
            f"Quadrature.rule({cell!r}, ...)"
        )

    element = basix.create_element(
        basix.ElementFamily[family],
        cell_type,
        degree,
        lagrange_variant,
        dpc_variant,
        discontinuous,
        dof_ordering,
        dtype,
    )

    tdim = len(basix.geometry(cell_type)[0])
    n_dofs, value_size = element.dim, element.value_size
    superdegree = element.embedded_superdegree

    # basix reads dof_ordering as "new position of old dof i", and applies it to points,
    # tabulate and entity_dofs — but not to coefficient_matrix or base_transformations,
    # which stay in its canonical order. `reorder` brings those two into line with the
    # rest; everything else is already permuted and must be left alone.
    reorder = _validate_dof_order(dof_ordering, n_dofs)
    if reorder is not None:
        reorder = np.argsort(reorder)

    exps = _monomial_exponents(cell, tdim, superdegree)
    points = _fit_points(cell, tdim, _FIT_OVERSAMPLING * len(exps))

    polyset = basix.tabulate_polynomials(
        basix.PolynomialType.legendre, cell_type, superdegree, points
    )
    if polyset.shape[0] != len(exps):
        raise RuntimeError(
            f"polyset size {polyset.shape[0]} does not match the {len(exps)} monomials "
            f"assumed for cell {cell!r} at degree {superdegree}"
        )

    monomials = np.prod(points[None, :, :] ** exps[:, None, :], axis=2)
    change_of_basis, *_ = np.linalg.lstsq(monomials.T, polyset.T, rcond=None)

    coeff = np.einsum(
        "dvq,qm->dvm",
        element.coefficient_matrix.reshape(n_dofs, value_size, polyset.shape[0]),
        change_of_basis.T,
    )
    if reorder is not None:
        coeff = coeff[reorder]

    # Reject a bad fit rather than let a mis-conditioned change of basis through.
    # `reference` is already in the requested dof order, so this also checks `reorder`.
    reference = element.tabulate(0, points)[0]
    reconstructed = np.einsum("dvm,mp->pdv", coeff, monomials)
    error = np.abs(reconstructed - reference).max()
    if error > _EXTRACTION_TOL * max(1.0, float(np.abs(reference).max())):
        raise RuntimeError(
            f"extracted basis for {family}{degree} on {cell} disagrees with "
            f"basix.tabulate by {error:.3e}; the monomial change of basis is "
            f"ill-conditioned at this degree"
        )

    entity_dofs = element.entity_dofs
    transformations = (
        None
        if element.dof_transformations_are_identity
        else np.asarray(element.base_transformations())
    )
    if transformations is not None and reorder is not None:
        transformations = transformations[:, reorder, :][:, :, reorder]
    nodes = np.asarray(element.points)
    reference_points = (
        nodes
        if element.map_type == basix.MapType.identity and nodes.shape[0] == n_dofs
        else None
    )

    map_type = MapType(element.map_type.name)
    value_shape = () if value_size == 1 else (value_size,)
    is_isoparametric = map_type is MapType.IDENTITY and value_shape == ()

    # create a gemoetry element, because basix element doesnt have information
    # about jacobian and detJ
    if geometry_degree is None:
        geometry_degree = degree if is_isoparametric else 1

    # An isoparametric element is its own geometry, so we skip the build.
    if is_isoparametric and geometry_degree == degree:
        geometry = None
    else:
        geometry = from_basix(
            "P",
            cell,
            geometry_degree,
            dtype=dtype,
            quadrature_rule=quadrature_rule,
        )

    return BasixElement(
        coeff=coeff,
        exps=exps,
        value_shape=value_shape,
        map_type=map_type,
        sobolev=Sobolev(element.sobolev_space.name),
        entity_dofs=tuple(tuple(tuple(ent) for ent in dim) for dim in entity_dofs),
        base_transformations=transformations,
        reference_points=reference_points,
        family=family,
        cell=cell,
        degree=degree,
        variant=lagrange_variant,
        discontinuous=discontinuous,
        quad_points=quadrature_rule.quad_points,
        quad_weights=quadrature_rule.quad_weights,
        geometry=geometry,
    )


class BasixElement(Element):
    """An `Element` whose basis is a constant coefficient matrix over monomials.

    Instances are produced by `from_basix` and are immutable. Unlike the hand-written
    elements, one class backs many distinct bases, so `__eq__`/`__hash__` are keyed on the
    coefficients rather than on `type(self)` — without that, two different elements sharing
    a quadrature rule collide as jax static arguments and one silently reuses the other's
    trace.
    """

    def __init__(
        self,
        *,
        coeff: np.ndarray,
        exps: np.ndarray,
        value_shape: tuple[int, ...],
        map_type: MapType,
        sobolev: Sobolev,
        entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
        base_transformations: np.ndarray | None,
        reference_points: np.ndarray | None,
        family: str,
        cell: str,
        degree: int,
        variant: basix.LagrangeVariant | None,
        discontinuous: bool,
        quad_points: Array,
        quad_weights: Array,
        geometry: BasixElement | None = None,
    ):
        """Prefer `from_basix`; this constructor takes already-extracted tables.

        Args:
            coeff: Basis coefficients over the monomials (shape: (n_dofs, value_size,
                n_monomials)).
            exps: Monomial exponents (shape: (n_monomials, tdim)).
            value_shape: `()` for scalar-valued families, `(tdim,)` for H(div)/H(curl).
            map_type: Push-forward to apply to reference values.
            sobolev: Conforming space, used to gate `divergence` and `curl`.
            entity_dofs: Dofs attached to each mesh entity, by dimension then entity index.
            base_transformations: Per-entity orientation corrections, or `None` when the
                element needs none. Exposed for the dofmap to apply; not applied here.
            reference_points: Nodal positions, when the dofs are point evaluations.
            family: basix family name, e.g. `"P"` or `"RT"`.
            cell: basix cell name, e.g. `"triangle"`.
            degree: Element degree.
            variant: Lagrange variant name, when one was given.
            discontinuous: Whether the discontinuous variant was requested.
            quad_points: Quadrature points (shape: (n_q, tdim)).
            quad_weights: Quadrature weights (shape: (n_q,)).
            geometry: Scalar Lagrange element describing the cell geometry, used by
                `get_jacobian`. `None` means isoparametric — this element is its own
                geometry, which is only valid for a scalar identity-mapped basis.
        """
        self._coeff_bytes = np.ascontiguousarray(coeff, dtype=np.float64).tobytes()
        self.coeff = jnp.asarray(coeff)
        self.exps = jnp.asarray(exps)

        self.value_shape = value_shape
        self.entity_dofs = entity_dofs
        self.base_transformations = (
            None if base_transformations is None else jnp.asarray(base_transformations)
        )

        self.family = family
        self.cell = cell
        self.degree = degree
        self.variant = variant
        self.discontinuous = discontinuous

        self._reference_points = (
            None if reference_points is None else jnp.asarray(reference_points)
        )

        if geometry is None and (map_type is not MapType.IDENTITY or value_shape != ()):
            raise ValueError(
                f"{family}{degree} on {cell} maps as {map_type.value} with value shape "
                f"{value_shape}, so it cannot be its own geometry; pass a scalar Lagrange "
                f"element as geometry="
            )
        self.geometry = geometry

        super().__init__(quad_points, quad_weights, sobolev=sobolev, map_type=map_type)

    # ------------------------------------------------------------------ properties

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom on the reference cell."""
        return self.coeff.shape[0]

    @property
    def value_size(self) -> int:
        """Number of components each basis function returns."""
        return self.coeff.shape[1]

    @property
    def tdim(self) -> int:
        """Topological dimension of the reference cell."""
        return self.exps.shape[1]

    def __repr__(self) -> str:
        return (
            f"BasixElement(family={self.family!r}, cell={self.cell!r}, "
            f"degree={self.degree}, n_dofs={self.n_dofs}, "
            f"map={self.map_type.value}, sobolev={self.sobolev.value})"
        )

    # ------------------------------------------------------------------ identity

    @property
    def _key(self) -> tuple:
        return (
            self.family,
            self.cell,
            self.degree,
            self.variant,
            self.discontinuous,
            self._coeff_bytes,
            None if self.geometry is None else self.geometry._key,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, BasixElement)
        return (
            self._key == other._key
            and np.array_equal(self.quad_points, other.quad_points)
            and np.array_equal(self.quad_weights, other.quad_weights)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._key,
                np.array(self.quad_points).tobytes(),
                np.array(self.quad_weights).tobytes(),
            )
        )

    # ------------------------------------------------------------------ basis

    def _reference_nodes(self) -> Array:
        if self._reference_points is None:
            raise NotImplementedError(
                f"{self!r} has dofs that are not point evaluations, so it has no "
                f"reference nodes. Use entity_dofs to build the dofmap instead."
            )
        return self._reference_points

    def _default_quadrature(self) -> tuple[Array, Array]:
        raise RuntimeError(
            "BasixElement is always constructed with an explicit quadrature rule; "
            "use from_basix(..., quadrature=...) or quadrature_degree=..."
        )

    def shape_function(self, xi: Array) -> Array:
        """Returns the reference basis at `xi` (shape: (n_dofs, *value_shape)).

        Args:
            xi: Local coordinates (shape: (tdim,)).
        """
        monomials = jnp.prod(xi[None, :] ** self.exps, axis=1)
        phi = jnp.einsum("dvm,m->dv", self.coeff, monomials)
        return phi.reshape(self.n_dofs, *self.value_shape)

    def shape_function_derivative(self, xi: Array) -> Array:
        """Returns the reference derivative (shape: (tdim, n_dofs, *value_shape)).

        For scalar families this is `(tdim, n_dofs)`, matching the hand-written elements.

        Args:
            xi: Local coordinates (shape: (tdim,)).
        """
        d_phi = jax.jacfwd(self.shape_function)(xi)
        return jnp.moveaxis(d_phi, -1, 0)

    # ------------------------------------------------------------------ mapping
    def get_jacobian(self, xi: Array, nodal_coords: Array) -> tuple[Array, Array]:
        """Returns the geometry Jacobian and its determinant at `xi`.

        The Jacobian belongs to the geometry, not to the basis: a basix element only knows
        the reference cell, so it is `self.geometry` — a scalar Lagrange element on the same
        cell — that is tabulated against `nodal_coords`. For a scalar identity-mapped basis
        `self.geometry` may be `None`, meaning isoparametric.

        For a cell embedded in a higher-dimensional space (`gdim > tdim`) `J` is not square
        and the returned scalar is the pseudo-determinant `sqrt(det(J J^T))`, i.e. the
        surface or curve measure.

        Args:
            xi: Local coordinates (shape: (tdim,)).
            nodal_coords: Coordinates of the geometry nodes of the cell (shape:
                (n_geometry_nodes, gdim)).

        Returns:
            `J` (shape: (tdim, gdim)) with `J[k, i] = dx_i / dxi_k`, and its determinant.
        """
        geometry = self if self.geometry is None else self.geometry
        d_phi = geometry.shape_function_derivative(xi)  # (tdim, n_geometry_nodes)
        jacobian = d_phi @ nodal_coords  # (tdim, gdim)
        tdim, gdim = jacobian.shape
        if tdim == gdim:
            return jacobian, jnp.linalg.det(jacobian)
        return jacobian, jnp.sqrt(jnp.linalg.det(jacobian @ jacobian.T))

    def push_forward(self, ref_values: Array, jacobian: Array, det_j: Array) -> Array:
        """Maps reference values to the physical cell.

        Row-vector convention: the trailing axis of `ref_values` is the value component.

        `jacobian` follows tatva's convention, `J[k, i] = dx_i / dxi_k` — the transpose of
        the usual `F = dx/dxi`, and what `Element.get_jacobian` returns. In terms of `F`
        the maps are the standard `F^-T phi` (covariant) and `F phi / det F`
        (contravariant); written against `J` and row vectors they become the expressions
        below.

        Args:
            ref_values: Values on the reference cell (shape: (..., value_size)).
            jacobian: Geometry Jacobian of the cell (shape: (tdim, gdim)).
            det_j: Determinant of the geometry Jacobian.

        Returns:
            The values pushed forward to the physical cell.
        """

        match self.map_type:
            case MapType.IDENTITY:
                return ref_values
            case MapType.COVARIANT_PIOLA:
                return ref_values @ jnp.linalg.inv(jacobian).T
            case MapType.CONTRAVARIANT_PIOLA:
                return (ref_values @ jacobian) / det_j
            case _:
                raise NotImplementedError(
                    f"push-forward for {self.map_type.value} is not implemented"
                )

    def divergence(self, xi: Array, dof_values: Array, nodal_coords: Array) -> Array:
        """Returns the physical divergence of the interpolated field at `xi`.

        Uses the contravariant Piola identity `div_x (P phi) = div_xi(phi) / det J`, which
        needs neither the inverse Jacobian nor the physical gradient.

        Args:
            xi: Local coordinates (shape: (tdim,)).
            dof_values: Dof values on this cell (shape: (n_dofs,)).
            nodal_coords: Nodal coordinates of the cell (shape: (n_dofs, tdim)).
        """
        if self.sobolev is not Sobolev.HDIV:
            raise ValueError(
                f"divergence is not defined on {self.sobolev.value}; "
                f"{self!r} supports gradient"
            )
        _, det_j = self.get_jacobian(xi, nodal_coords)
        d_phi = self.shape_function_derivative(xi)
        div_ref = jnp.einsum("dnd->n", d_phi)
        return jnp.dot(dof_values, div_ref) / det_j

    def curl(self, xi: Array, dof_values: Array, nodal_coords: Array) -> Array:
        """Returns the physical curl of the interpolated field at `xi`.

        Uses the covariant Piola identity `curl_x (P phi) = F curl_xi(phi) / det F`, with
        `F = J.T` in tatva's Jacobian convention (see `push_forward`).

        Args:
            xi: Local coordinates (shape: (tdim,)).
            dof_values: Dof values on this cell (shape: (n_dofs,)).
            nodal_coords: Nodal coordinates of the cell (shape: (n_dofs, tdim)).
        """
        if self.sobolev is not Sobolev.HCURL:
            raise ValueError(
                f"curl is not defined on {self.sobolev.value}; "
                f"{self!r} supports gradient"
            )
        jacobian, det_j = self.get_jacobian(xi, nodal_coords)
        d_phi = self.shape_function_derivative(xi)
        if self.tdim == 3:
            curl_ref = jnp.stack(
                [
                    d_phi[1, :, 2] - d_phi[2, :, 1],
                    d_phi[2, :, 0] - d_phi[0, :, 2],
                    d_phi[0, :, 1] - d_phi[1, :, 0],
                ],
                axis=-1,
            )
            return (jnp.einsum("n,nv->v", dof_values, curl_ref) @ jacobian) / det_j
        if self.tdim == 2:
            # The 2D curl is the scalar d_x phi_y - d_y phi_x; no Jacobian rotation.
            curl_ref = d_phi[0, :, 1] - d_phi[1, :, 0]
            return jnp.dot(dof_values, curl_ref) / det_j
        raise NotImplementedError(f"curl is not defined for tdim={self.tdim}")

    # ------------------------------------------------- identity-mapped conveniences

    def _require_identity(self, what: str) -> None:
        if self.map_type is not MapType.IDENTITY:
            raise ValueError(
                f"{what} assumes the identity map; {self!r} needs "
                f"{self.map_type.value}. Use push_forward, divergence or curl."
            )
        if self.value_shape != ():
            raise ValueError(f"{what} assumes a scalar-valued basis; {self!r} is not")

    def interpolate(self, xi: Array, dof_values: Array, nodal_coords: Array) -> Array:
        self._require_identity("interpolate")
        return super().interpolate(xi, dof_values, nodal_coords)

    def gradient(self, xi: Array, dof_values: Array, nodal_coords: Array) -> Array:
        self._require_identity("gradient")
        return super().gradient(xi, dof_values, nodal_coords)

    def get_local_values(
        self, xi: Array, dof_values: Array, nodal_coords: Array
    ) -> tuple[Array, Array, Array]:
        self._require_identity("get_local_values")
        return super().get_local_values(xi, dof_values, nodal_coords)


# --------------------------------------------------------------------------- factory


def _monomial_exponents(cell: str, tdim: int, degree: int) -> np.ndarray:
    """Exponents spanning the same polynomial space as basix's polyset for `cell`."""
    if cell in _SIMPLEX_CELLS:
        exps = [
            e
            for e in itertools.product(range(degree + 1), repeat=tdim)
            if sum(e) <= degree
        ]
    elif cell in _TENSOR_CELLS:
        exps = list(itertools.product(range(degree + 1), repeat=tdim))
    else:
        raise NotImplementedError(
            f"cell {cell!r} has a polyset that is neither total-degree nor "
            f"tensor-product; extraction is not implemented for it"
        )
    return np.array(sorted(exps, key=lambda e: (sum(e), e)), dtype=np.int32)


def _fit_points(cell: str, tdim: int, n_points: int) -> np.ndarray:
    """Deterministic sample points strictly inside the reference cell."""
    rng = np.random.default_rng(_FIT_SEED)
    if cell not in _SIMPLEX_CELLS:
        return rng.random((n_points, tdim))
    kept: list[np.ndarray] = []
    while len(kept) < n_points:
        batch = rng.random((4 * n_points, tdim))
        kept.extend(batch[batch.sum(axis=1) < 1.0])
    return np.array(kept[:n_points])


def _validate_dof_order(
    dof_order: Sequence[int] | None, n_dofs: int
) -> np.ndarray | None:
    if dof_order is None:
        return None
    perm = np.asarray(dof_order, dtype=np.int64)
    if perm.shape != (n_dofs,) or sorted(perm.tolist()) != list(range(n_dofs)):
        raise ValueError(
            f"dof_order must be a permutation of range({n_dofs}), got {dof_order!r}"
        )
    return perm
