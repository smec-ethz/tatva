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

import warnings
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from tatva.sparse.tracer.handlers import PrimitiveHandler

RegistryHandle: TypeAlias = tuple[Any, ...] | str


class TracerRegistry:
    """Registry for JAX primitive dependency propagation handlers."""

    def __init__(self):
        self._handlers: dict[str, PrimitiveHandler] = {}

    def _register(self, primitive_name: str, handler: PrimitiveHandler):
        """Register a handler for a specific JAX primitive."""
        self._handlers[primitive_name] = handler

    def register(self, *args: RegistryHandle):
        """Register handlers with a decorator for a class that implements the PrimitiveHandler interface."""

        def decorator(cls):
            for primitive in args:
                if isinstance(primitive, str):
                    self._register(primitive, cls())
                elif isinstance(primitive, tuple):
                    primitive_name, cls_args = primitive[0], primitive[1:]
                    self._register(primitive_name, cls(*cls_args))

            return cls

        return decorator

    def get(
        self, primitive_name: str, default: PrimitiveHandler | None = None
    ) -> PrimitiveHandler:
        """Get the registered handler, or return the fallback."""
        handler = self._handlers.get(primitive_name)
        if handler is None:
            from tatva.sparse.tracer.handlers import OpaqueBlackBoxHandler

            warnings.warn(
                f"No handler registered for primitive '{primitive_name}'. Using fallback handler."
            )
            return default or OpaqueBlackBoxHandler(record_couplings=False)
        return handler


# Global registry instance
TR = TracerRegistry()
