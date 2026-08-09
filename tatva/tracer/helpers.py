from jax.core import Atom


def _shape_of(var: Atom) -> tuple[int, ...]:
    try:
        return var.aval.shape  # ty: ignore[unresolved-attribute]
    except AttributeError:
        return ()
