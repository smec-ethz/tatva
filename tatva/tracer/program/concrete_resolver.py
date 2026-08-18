from collections.abc import Callable
from typing import Any

import numpy as np
from jax import lax
from jax.extend.core import JaxprEqn, Primitive, Var


class DynamicRoutingError(RuntimeError):
    """Planning-time routing depends on a value unavailable without the DOFs."""


type ConcreteEvalRule = Callable[[tuple[Any, ...], dict[str, Any]], tuple[Any, ...]]

type ConcreteValue = Any
type ConcreteEnv = dict[Var, ConcreteValue]


def evaluate_concrete_eqn(
    eqn: JaxprEqn,
    inputs: tuple[ConcreteValue, ...],
) -> tuple[ConcreteValue, ...]:
    # these are numpy fast paths for some primitives, but not all. For the rest we fall
    # back to the primitive's bind method.
    evaluator = _CONCRETE_EVALS.get(eqn.primitive)
    if evaluator is not None:
        outputs = evaluator(inputs, eqn.params)
    else:
        print(
            f"Warning: no concrete evaluator for {eqn.primitive.name}, falling back to bind"
        )
        result = eqn.primitive.bind(*inputs, **eqn.params)
        outputs = tuple(result) if eqn.primitive.multiple_results else (result,)

    if len(outputs) != len(eqn.outvars):
        raise RuntimeError(
            f"{eqn.primitive.name} produced {len(outputs)} concrete outputs "
            f"for {len(eqn.outvars)} Jaxpr outputs"
        )

    # Keep planning data host-side.
    normalized = tuple(
        np.asarray(value) if hasattr(value, "shape") else value for value in outputs
    )

    return normalized


# --------------------------------------
# Concrete primitive evaluation rules
# --------------------------------------

_CONCRETE_EVALS: dict[Primitive, ConcreteEvalRule] = {}


def _simple_np(fn):
    def evaluate(inputs, _params):
        return (fn(*inputs),)

    return evaluate


def register(
    *primitives: Primitive,
) -> Callable[[ConcreteEvalRule], ConcreteEvalRule]:
    def decorator(
        rule: ConcreteEvalRule,
    ) -> ConcreteEvalRule:
        for primitive in primitives:
            _CONCRETE_EVALS[primitive] = rule
        return rule

    return decorator


register(lax.neg_p)(_simple_np(np.negative))
register(lax.abs_p)(_simple_np(np.abs))
register(lax.exp_p)(_simple_np(np.exp))
register(lax.add_p)(_simple_np(np.add))
register(lax.sub_p)(_simple_np(np.subtract))
register(lax.mul_p)(_simple_np(np.multiply))
register(lax.div_p)(_simple_np(np.divide))
register(lax.lt_p)(_simple_np(np.less))
register(lax.le_p)(_simple_np(np.less_equal))
register(lax.gt_p)(_simple_np(np.greater))
register(lax.ge_p)(_simple_np(np.greater_equal))
register(lax.eq_p)(_simple_np(np.equal))


@register(lax.copy_p, lax.stop_gradient_p)
def eval_identity(inputs, _params):
    return inputs


@register(lax.reshape_p)
def eval_reshape(inputs, params):
    (x,) = inputs
    return (np.reshape(np.asarray(x), params["new_sizes"]),)


@register(lax.transpose_p)
def eval_transpose(inputs, params):
    (x,) = inputs
    return (np.transpose(np.asarray(x), params["permutation"]),)


@register(lax.broadcast_in_dim_p)
def eval_broadcast_in_dim(inputs, params):
    (x,) = inputs

    shape = tuple(params["shape"])
    dims = tuple(params["broadcast_dimensions"])
    x = np.asarray(x)
    expanded = [1] * len(shape)

    for input_axis, output_axis in enumerate(dims):
        expanded[output_axis] = x.shape[input_axis]

    return (np.broadcast_to(x.reshape(expanded), shape),)


@register(lax.convert_element_type_p)
def _eval_convert_dtype(inputs, params):
    (x,) = inputs
    return (x.astype(params["new_dtype"]),)


@register(lax.iota_p)
def _eval_iota(_inputs, params):
    dim = params.get("dimension")
    shp = params.get("shape")
    newshp = [1] * len(shp)
    newshp[dim] = shp[dim]
    return (np.broadcast_to(np.arange(shp[dim]).reshape(newshp), shp),)


try:
    from jax._src.lax.control_flow import platform_index_p

    @register(platform_index_p)
    def _eval_platform_index(_inputs, params):
        platforms = params.get("platforms", ("cpu",))
        import jax

        backend = jax.default_backend()
        idx = platforms.index(backend) if backend in platforms else 0
        return (np.int32(idx),)
except ImportError:
    pass


@register(lax.select_n_p)
def eval_select_n(inputs, _params):
    selector = np.asarray(inputs[0])
    cases = tuple(np.asarray(x) for x in inputs[1:])
    if not cases:
        raise ValueError("select_n requires at least one case")

    result = cases[0]
    for index, case in enumerate(cases[1:], start=1):
        result = np.where(selector == index, case, result)

    return (result,)


@register(lax.clamp_p)
def eval_clamp(inputs, _params):
    min_val, x_val, max_val = inputs
    return (np.clip(np.asarray(x_val), np.asarray(min_val), np.asarray(max_val)),)
