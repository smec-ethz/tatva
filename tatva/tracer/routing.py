from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of

if TYPE_CHECKING:
    from tatva.tracer.rules import ConcreteEnv


class Route:
    pass


@dataclass(frozen=True)
class GatherRoute(Route):
    source_rows: NDArray[np.int64]  # (n_output,)


@dataclass(frozen=True)
class ScatterRoute(Route):
    # one entry per flattened update element
    target_rows: NDArray[np.int64]
    # optional: only needed if the orig index-producing graph remains live/runtime localized
    index_rows: NDArray[np.int64] | None = None


def resolve_routes(
    eqns: tuple[JaxprEqn, ...],
    concrete: ConcreteEnv,
) -> dict[JaxprEqn, Route]:
    # Import lazily: rules import Route for their type signatures, while route
    # resolution needs the populated semantic registry only at execution time.
    from tatva.tracer.rules import SEMANTICS

    routes: dict[JaxprEqn, Route] = {}

    for eqn in eqns:
        rule = SEMANTICS.get(eqn.primitive)
        if rule is None:
            raise ValueError(f"Primitive {eqn.primitive} has no registered rule.")

        route = rule.route(eqn, concrete)

        if route is not None:
            routes[eqn] = route

    return routes


def resolve_gather_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> GatherRoute | None:
    indices = concrete.get(eqn.invars[1])
    if indices is None:
        return None

    return _compute_gather_route(eqn, np.asarray(indices))


def resolve_scatter_route(
    eqn: JaxprEqn,
    concrete: ConcreteEnv,
) -> ScatterRoute | None:
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None

    # operand, indices, updates = eqn.invars[:3]
    indices = concrete.get(eqn.invars[1])
    if indices is None:
        return None

    return _compute_scatter_route(eqn, np.asarray(indices))


def _compute_gather_route(
    eqn: JaxprEqn, indices: NDArray[np.int64]
) -> GatherRoute | None:
    if len(eqn.invars) < 2 or not eqn.outvars:
        return None

    operand_shape = tuple(_shape_of(eqn.invars[0]))

    row_ids = np.arange(np.prod(operand_shape), dtype=np.int64).reshape(operand_shape)
    params = dict(eqn.params)

    try:
        gathered = eqn.primitive.bind(
            jnp.asarray(row_ids), jnp.asarray(indices), **params
        )
    except (TypeError, ValueError, RuntimeError):
        return None

    return GatherRoute(
        source_rows=np.asarray(gathered, dtype=np.int64).ravel(),
    )


def _compute_scatter_route(eqn: JaxprEqn, indices: NDArray) -> ScatterRoute | None:
    # NOTE: mostly unchecked
    if len(eqn.invars) < 3 or not eqn.outvars:
        return None

    operand_shape = tuple(_shape_of(eqn.invars[0]))
    indices_shape = tuple(_shape_of(eqn.invars[1]))
    updates_shape = tuple(_shape_of(eqn.invars[2]))

    indices = np.asarray(indices)

    if indices.ndim < 1 or tuple(indices.shape) != indices_shape:
        return None

    try:
        dnums = eqn.params["dimension_numbers"]

        window_dims = tuple(dnums.update_window_dims)
        inserted_dims = tuple(dnums.inserted_window_dims)
        scatter_dims = tuple(dnums.scatter_dims_to_operand_dims)
        operand_batch_dims = tuple(dnums.operand_batching_dims)
        indices_batch_dims = tuple(dnums.scatter_indices_batching_dims)
    except (KeyError, TypeError, AttributeError):
        return None

    index_vector_size = indices_shape[-1]

    if index_vector_size != len(scatter_dims):
        return None

    if len(operand_batch_dims) != len(indices_batch_dims):
        return None

    n_updates = int(np.prod(updates_shape))
    update_rows = np.arange(n_updates, dtype=np.int64)

    try:
        update_coords = np.stack(
            np.unravel_index(update_rows, updates_shape),
            axis=1,
        )

        # Map update dimensions that are NOT window dimensions
        # onto indices.shape[:-1].
        batch_update_dims = tuple(
            d for d in range(len(updates_shape)) if d not in window_dims
        )

        if len(batch_update_dims) != len(indices_shape) - 1:
            return None

        index_batch_coords = update_coords[:, batch_update_dims]

        if index_vector_size:
            key = tuple(
                index_batch_coords[:, i] for i in range(index_batch_coords.shape[1])
            )

            index_vectors = np.asarray(indices[key], dtype=np.int64)
            index_vectors = index_vectors.reshape(
                n_updates,
                index_vector_size,
            )
        else:
            index_vectors = np.empty(
                (n_updates, 0),
                dtype=np.int64,
            )

        # Construct operand coordinate for each update scalar.
        target_coords = np.zeros(
            (n_updates, len(operand_shape)),
            dtype=np.int64,
        )

        # Explicit scatter indices.
        for component, operand_axis in enumerate(scatter_dims):
            target_coords[:, operand_axis] = index_vectors[:, component]

        # Batched dimensions.
        for operand_axis, indices_axis in zip(
            operand_batch_dims,
            indices_batch_dims,
        ):
            target_coords[:, operand_axis] = index_batch_coords[:, indices_axis]

        # Window dimensions.
        window_operand_dims = tuple(
            d
            for d in range(len(operand_shape))
            if d not in inserted_dims and d not in operand_batch_dims
        )

        if len(window_operand_dims) != len(window_dims):
            return None

        for update_axis, operand_axis in zip(
            window_dims,
            window_operand_dims,
        ):
            target_coords[:, operand_axis] += update_coords[:, update_axis]

        # Dropped / out-of-bounds updates become -1.
        valid = np.ones(n_updates, dtype=bool)

        for axis, size in enumerate(operand_shape):
            valid &= target_coords[:, axis] >= 0
            valid &= target_coords[:, axis] < size

        target_rows = np.full(n_updates, -1, dtype=np.int64)

        if np.any(valid):
            target_rows[valid] = np.ravel_multi_index(
                tuple(target_coords[valid].T),
                operand_shape,
            )

        return ScatterRoute(
            target_rows=target_rows,
        )

    except (ValueError, IndexError, TypeError):
        return None
