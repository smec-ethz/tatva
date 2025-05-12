import jax
import functools

def vmap(use_vmap=True, *vmap_args, **vmap_kwargs):
    """
    Decorator to optionally apply jax.vmap to a function.

    Arguments:
        use_vmap (bool): Whether to apply vmap or not.
        *vmap_args, **vmap_kwargs: Passed to jax.vmap.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if use_vmap:
                vmapped_func = jax.jit(jax.vmap(func, *vmap_args, **vmap_kwargs))
                return vmapped_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapped

    return decorator

