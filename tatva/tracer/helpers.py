from jax.core import Atom

from tatva.tracer.model import Shape


def _shape_of(var: Atom) -> Shape:
    try:
        return var.aval.shape  # ty: ignore[unresolved-attribute]
    except AttributeError:
        return ()
