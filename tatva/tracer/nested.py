from dataclasses import dataclass

from jax.extend.core import ClosedJaxpr, Jaxpr


@dataclass(frozen=True)
class NestedJaxpr:
    jaxpr: Jaxpr
    consts: tuple[object, ...]


def normalize_nested_jaxpr(
    value: Jaxpr | ClosedJaxpr,
) -> NestedJaxpr:
    if isinstance(value, ClosedJaxpr):
        return NestedJaxpr(
            jaxpr=value.jaxpr,
            consts=tuple(value.consts),
        )

    if isinstance(value, Jaxpr):
        return NestedJaxpr(
            jaxpr=value,
            consts=(),
        )

    raise TypeError(f"expected Jaxpr or ClosedJaxpr, got {type(value)!r}")
