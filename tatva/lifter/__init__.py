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

"""Lifter module for mapping between reduced and full vectors."""

from tatva.lifter.base import Lifter as Lifter
from tatva.lifter.common import LifterError as LifterError
from tatva.lifter.common import RuntimeValue as RuntimeValue
from tatva.lifter.constraints import Constraint as Constraint
from tatva.lifter.constraints import Fixed as Fixed
from tatva.lifter.constraints import Periodic as Periodic


def __getattr__(name: str):
    from warnings import warn

    if name == "DirichletBC":
        warn(
            "`DirichletBC` is deprecated; use `Fixed` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Fixed
    if name == "PeriodicMap":
        warn(
            "`PeriodicMap` is deprecated; use `Periodic` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Periodic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
