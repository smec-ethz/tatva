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
    """For each flattened output row, the flattened source row it reads."""

    source_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ScatterRoute:
    """For each flattened update row, the flattened operand row it targets.

    A value of -1 represents an out-of-bounds/dropped update.
    """

    target_rows: NDArray[np.int64]
    index_rows: NDArray[np.int64] | None = None


type Route = GatherRoute | ScatterRoute
type RouteEnv = Mapping[JaxprEqn, Route]
