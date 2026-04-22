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

from dataclasses import dataclass
from enum import IntEnum
from typing import Self

from jax import Array
from numpy.typing import NDArray


class _FieldType:
    def get(self) -> Self:
        return self


@dataclass
class Nodal(_FieldType):
    node_ids: Array | NDArray | None = None
    stack: bool = True


@dataclass
class Local(_FieldType):
    pass


@dataclass
class Shared(_FieldType):
    pass


class FieldType(IntEnum):
    LOCAL = 0
    NODAL = 1
    SHARED = 2

    def get(self) -> _FieldType:
        if self not in _enum_to_class:
            raise ValueError(f"Cannot get corresponding class for {self}")
        _cls = _enum_to_class[self]
        return _cls()


_enum_to_class: dict[FieldType, type[_FieldType]] = {
    FieldType.LOCAL: Local,
    FieldType.SHARED: Shared,
    FieldType.NODAL: Nodal,
}
