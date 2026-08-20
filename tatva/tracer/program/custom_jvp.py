"""Extraction and validation of JAX 0.11 custom-JVP equation parameters."""

from __future__ import annotations

from dataclasses import dataclass

from jax.extend.core import Jaxpr, JaxprEqn


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
    primal = params.get("call_jaxpr")
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

    if jvp.effects:
        raise NotImplementedError("effectful custom_jvp JVP callbacks are unsupported")

    # jax's staged custom-jvp ABI deliberately excludes the lifted primal constants from
    # the jvp callback. The callback receives its own captures as leading inputs,
    # followed by dynamic primals and their nonzero tangents. Normalize those leading
    # inputs into constvars for Tatva's nested-program representation.
    num_jvp_consts = len(jvp_consts)
    expected_inputs = num_jvp_consts + 2 * dynamic_arity
    if len(jvp.invars) != expected_inputs:
        raise NotImplementedError(
            "custom_jvp JVP input ABI is unsupported: expected "
            f"{expected_inputs} inputs, got {len(jvp.invars)}"
        )
    jvp = jvp.replace(
        constvars=(*jvp.constvars, *jvp.invars[:num_jvp_consts]),
        invars=jvp.invars[num_jvp_consts:],
        consts=jvp_consts,
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
