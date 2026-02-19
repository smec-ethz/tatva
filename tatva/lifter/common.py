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

from typing import (
    Any,
    Generator,
    Generic,
    Hashable,
    Mapping,
    Sequence,
    Set,
    TypeAlias,
    TypeVar,
)
from uuid import uuid4

import numpy as np
from jax.typing import ArrayLike

__all__ = ["RuntimeValue", "LifterError"]

T = TypeVar("T", bound=ArrayLike)

RuntimeValueMap: TypeAlias = dict[Hashable, Any]


class LifterError(ValueError):
    """Error raised when there is a problem with the lifter, e.g., missing runtime values."""


class RuntimeValue(Generic[T]):
    """Descriptor for a constraint value that must be provided at runtime.

    Args:
        key: A hashable key to identify this runtime value; used for setting values on the
            lifter and for error messages when the value is missing at runtime.
        default: An optional default value to use. If not provided, the lifter will raise
            an error if this value is not set at runtime.
    """

    def __init__(self, key: Hashable | None = None, default: T | None = None):
        self.key = key or uuid4()
        self.default = default

    def get_value(self, runtime_values: RuntimeValueMap) -> T:
        """Get the value of this attribute from the given lifter instance."""
        if self.key not in runtime_values:
            raise LifterError(f"Runtime value for (key={self.key}) not set on lifter")
        return runtime_values[self.key]


def _iter_runtime_values(
    obj, seen: set[int] | None = None
) -> Generator[RuntimeValue[Any], None, None]:
    """Recursively iterate over all RuntimeValue attributes in obj, yielding them one by
    one.

    This is used to collect all runtime specs from a constraint instance, which may be
    nested in dicts, lists, or other objects. We keep track of seen object ids to avoid
    infinite recursion in case of cycles in the object graph.
    """
    if seen is None:
        seen = set()

    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    if isinstance(obj, RuntimeValue):
        yield obj
        return

    # dict / mapping
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_runtime_values(v, seen)
        return

    # list / tuple / set (but not str/bytes)
    if isinstance(obj, (Sequence, Set)) and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        for v in obj:
            yield from _iter_runtime_values(v, seen)
        return

    # normal Python objects
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            yield from _iter_runtime_values(v, seen)
        return

    # optional: __slots__
    if hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            if hasattr(obj, name):
                yield from _iter_runtime_values(getattr(obj, name), seen)


def _runtime_value_map_is_equal(left: RuntimeValueMap, right: RuntimeValueMap) -> bool:
    """Check if two RuntimeValueMaps are equal, treating array values as equal if they
    have the same shape and contents.
    """
    if left.keys() != right.keys():
        return False

    for key in left:
        _left = left[key]
        _right = right[key]

        if _left is _right:
            continue

        if np.array_equal(np.asarray(_left), np.asarray(_right)):
            continue

        return False

    return True
