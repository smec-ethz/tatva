"""Extraction and validation of JAX 0.11 custom-JVP equation parameters."""

from __future__ import annotations

from dataclasses import dataclass

from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn


@dataclass(frozen=True, slots=True)
class CustomJvpParameters:
    primal_jaxpr: Jaxpr
    jvp_jaxpr: Jaxpr
    jvp_consts: tuple[object, ...]
    num_consts: int
    output_zeros: tuple[bool, ...]


def extract_custom_jvp_parameters(eqn: JaxprEqn) -> CustomJvpParameters:
    """Materialize the supported all-tangents-present JVP program."""
    params = eqn.params
    primal_param = params.get("call_jaxpr")
    primal = (
        primal_param.jaxpr if isinstance(primal_param, ClosedJaxpr) else primal_param
    )
    jvp_fun = params.get("jvp_jaxpr_fun")
    num_consts = params.get("num_consts")
    symbolic_zeros = params.get("symbolic_zeros")

    if not isinstance(primal, Jaxpr):
        raise NotImplementedError("custom_jvp is missing a Jaxpr-valued call_jaxpr")
    if jvp_fun is None or not hasattr(jvp_fun, "call_wrapped"):
        raise NotImplementedError("custom_jvp is missing jvp_jaxpr_fun metadata")
    if not isinstance(num_consts, int) or not 0 <= num_consts <= len(eqn.invars):
        raise NotImplementedError("custom_jvp has an invalid num_consts parameter")
    if symbolic_zeros is not False:
        raise NotImplementedError(
            "localized custom_jvp currently requires symbolic_zeros=False"
        )
    if primal.effects:
        raise NotImplementedError(
            "effectful custom_jvp primal callbacks are unsupported"
        )

    dynamic_arity = len(eqn.invars) - num_consts
    try:
        jvp, jvp_consts, output_zeros = jvp_fun.call_wrapped(
            *(False for _ in range(dynamic_arity))
        )
    except Exception as exc:
        raise NotImplementedError(
            "could not instantiate the custom_jvp callback program"
        ) from exc

    if not isinstance(jvp, Jaxpr):
        raise NotImplementedError("custom_jvp callback did not produce a Jaxpr")

    jvp_consts = tuple(jvp_consts)
    # JAX 0.11 trace_to_jaxpr_dynamic returns callback constants
    # separately. Until their values are attached, their variables are
    # exposed as leading jvp.invars rather than jvp.constvars.
    if jvp_consts:
        if jvp.consts:
            if len(jvp.consts) != len(jvp_consts):
                raise NotImplementedError(
                    "custom_jvp JVP callback returned inconsistent constants"
                )
        else:
            if len(jvp_consts) > len(jvp.invars):
                raise NotImplementedError(
                    "custom_jvp JVP callback returned more constants than leading Jaxpr inputs"
                )

            jvp = jvp.with_consts(jvp_consts)

    if jvp.effects:
        raise NotImplementedError("effectful custom_jvp JVP callbacks are unsupported")

    # JAX keeps JVP captures in jvp.constvars and returns their values
    # separately as jvp_consts. They are not leading explicit JVP inputs.
    expected_inputs = 2 * dynamic_arity
    if len(jvp.invars) != expected_inputs:
        raise NotImplementedError(
            "custom_jvp JVP input ABI is unsupported: expected "
            f"{expected_inputs} inputs, got {len(jvp.invars)}"
        )
    if len(jvp.constvars) != len(jvp_consts):
        raise NotImplementedError(
            "custom_jvp JVP capture ABI is unsupported: expected "
            f"{len(jvp.constvars)} constants, got {len(jvp_consts)}"
        )

    zeros = tuple(bool(value) for value in output_zeros)
    if len(zeros) != len(eqn.outvars):
        raise NotImplementedError("custom_jvp JVP output-zero metadata has wrong arity")

    expected_outputs = len(eqn.outvars) + sum(not value for value in zeros)
    if len(jvp.outvars) != expected_outputs:
        raise NotImplementedError(
            "custom_jvp JVP output ABI is unsupported: expected "
            f"{expected_outputs} outputs, got {len(jvp.outvars)}"
        )

    return CustomJvpParameters(primal, jvp, tuple(jvp_consts), num_consts, zeros)
