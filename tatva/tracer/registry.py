from jax.extend.core import Primitive

from tatva.tracer.rules.registration import register_builtin_rules
from tatva.tracer.semantics import (
    NestedOperationSemantics,
    OperationSemantics,
    RegisteredOperationSemantics,
)


class PrimitiveRegistry:
    def __init__(self):
        self._rules: dict[Primitive, RegisteredOperationSemantics] = {}

    def register(
        self, primitive: Primitive, rule: RegisteredOperationSemantics
    ) -> None:
        if primitive in self._rules:
            raise ValueError(f"Primitive {primitive.name} is already registered.")

        self._rules[primitive] = rule

    def get(self, primitive: Primitive) -> RegisteredOperationSemantics:
        try:
            return self._rules[primitive]
        except KeyError:
            raise NotImplementedError(
                f"No semantics registered for {primitive.name}"
            ) from None

    def get_ordinary(self, primitive: Primitive) -> OperationSemantics:
        rule = self.get(primitive)
        if isinstance(rule, NestedOperationSemantics):
            raise TypeError(
                f"Primitive {primitive.name} is a nested operation; "
                "ordinary primitive semantics were requested"
            )

        return rule


def get_primitive_registry() -> PrimitiveRegistry:
    """Return the global primitive registry."""
    reg = PrimitiveRegistry()
    register_builtin_rules(reg)
    return reg


SEMANTICS = get_primitive_registry()
