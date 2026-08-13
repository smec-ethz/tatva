from jax.extend.core import Primitive

from tatva.tracer.rules.registration import register_builtin_rules
from tatva.tracer.semantics import OperationSemantics


class PrimitiveRegistry:
    def __init__(self):
        self._rules: dict[Primitive, OperationSemantics] = {}

    def register(self, primitive: Primitive, rule: OperationSemantics) -> None:
        if primitive in self._rules:
            raise ValueError(f"Primitive {primitive.name} is already registered.")

        self._rules[primitive] = rule

    def get(self, primitive: Primitive) -> OperationSemantics:
        try:
            return self._rules[primitive]
        except KeyError:
            raise NotImplementedError(
                f"No semantics registered for {primitive.name}"
            ) from None


def get_primitive_registry() -> PrimitiveRegistry:
    """Return the global primitive registry."""
    reg = PrimitiveRegistry()
    register_builtin_rules(reg)
    return reg


SEMANTICS = get_primitive_registry()
