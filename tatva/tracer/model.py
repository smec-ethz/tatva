from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.extend.core import JaxprEqn, Var
from numpy.typing import NDArray

type Shape = tuple[int, ...]

type ConcreteValue = NDArray[Any] | np.generic | bool | int | float | complex
type ConcreteEnv = Mapping[Var, ConcreteValue]


@dataclass(frozen=True, slots=True)
class GatherRoute:
    """Resolved gather geometry in global scalar rows.

    `source_rows[o]` is the operand row read for output row `o`.
    `index_rows[o, c]` is the flattened gather-index row supplying index
    component `c` for output row `o`.
    """

    source_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class ScatterRoute:
    """For each flattened update row, the flattened operand row it targets.

    A value of -1 represents an out-of-bounds/dropped update.
    """

    target_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class SelectNRoute:
    # for each output row, which case operand is selected
    # 0 means eqn.invars[1], 1 means eqn.invars[2], etc.
    case_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicSliceRoute:
    source_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DynamicUpdateSliceRoute:
    target_rows: NDArray[np.int64]


type Route = (
    GatherRoute
    | ScatterRoute
    | SelectNRoute
    | DynamicSliceRoute
    | DynamicUpdateSliceRoute
)
type RouteEnv = Mapping[JaxprEqn, Route]
