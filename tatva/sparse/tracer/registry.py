from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tatva.sparse.tracer.handlers import PrimitiveHandler


@dataclass(slots=True)
class PrimitiveRegistry:
    """Registry of fully composed primitive semantics."""

    _handlers: dict[str, PrimitiveHandler] = field(default_factory=dict)
    _default: PrimitiveHandler | None = None

    def add(
        self,
        name: str,
        handler: PrimitiveHandler,
        *,
        replace: bool = False,
    ) -> None:
        if not replace and name in self._handlers:
            raise ValueError(f"Primitive {name!r} is already registered")
        self._handlers[name] = handler

    def set_default(self, handler: PrimitiveHandler) -> None:
        self._default = handler

    def get(self, name: str) -> PrimitiveHandler:
        handler = self._handlers.get(name)
        if handler is not None:
            return handler
        if self._default is not None:
            return self._default
        raise KeyError(f"No primitive handler registered for {name!r}")

    def registered(self) -> dict[str, PrimitiveHandler]:
        return dict(self._handlers)


TR = PrimitiveRegistry()
