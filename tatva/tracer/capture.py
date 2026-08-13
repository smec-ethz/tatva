from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
from jax.extend.core import ClosedJaxpr, Jaxpr


@dataclass(frozen=True)
class CapturedJaxpr[**P, R]:
    fn: Callable[P, R]
    closed_jaxpr: ClosedJaxpr
    flat_args: tuple[Any, ...]
    call_abi: CallABI

    @classmethod
    def from_fn(
        cls, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> CapturedJaxpr[P, R]:
        call_abi, flat_args = CallABI.from_call(fn, args, kwargs)

        # Crucial: we create the JAXPR from the exact canonical flat ABI. then
        # jaxpr.invars[i] <-> flat_args[i] is guaranteed
        def flat_fn(*flat):
            bound = call_abi.unflatten(tuple(flat))
            return fn(*bound.args, **bound.kwargs)

        closed_jaxpr = jax.make_jaxpr(flat_fn)(*flat_args)

        return cls(fn, closed_jaxpr, flat_args, call_abi)

    @property
    def jaxpr(self) -> Jaxpr:
        return self.closed_jaxpr.jaxpr

    @property
    def consts(self) -> list[Any]:
        return self.closed_jaxpr.consts

    @property
    def constvars(self) -> list[Any]:
        return self.closed_jaxpr.constvars

    @property
    def invars(self) -> list[Any]:
        return self.closed_jaxpr.invars

    @property
    def outvars(self) -> list[Any]:
        return self.closed_jaxpr.outvars


def make_captured_jaxpr[**P, R](
    fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> CapturedJaxpr[P, R]:
    """Capture a function as a JAXPR with its call ABI."""
    return CapturedJaxpr.from_fn(fn, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class InputOrigin:
    flat_index: int
    parameter_index: int
    parameter_name: str
    key_path: tuple[Any, ...]

    @property
    def display_path(self) -> str:
        suffix = jax.tree_util.keystr(self.key_path)
        return f"{self.parameter_name}{suffix}"


@dataclass(frozen=True, slots=True)
class CallABI:
    """Canonical mapping between a Python function call and flat JAXPR inputs.

    The canonical tree is one entry per declared function parameter,
    in signature order. Therefore positional-vs-keyword spelling does not
    affect the flat ABI.
    """

    signature: inspect.Signature
    parameter_names: tuple[str, ...]
    treedef: jax.tree_util.PyTreeDef
    input_origins: tuple[InputOrigin, ...]

    @classmethod
    def from_call(
        cls,
        fn: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[CallABI, tuple[Any, ...]]:
        signature = inspect.signature(fn)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        parameter_names = tuple(signature.parameters)
        canonical = tuple(bound.arguments[name] for name in parameter_names)
        # flat, treedef = jax.tree_util.tree_flatten(canonical)
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(canonical)
        flat: list[Any] = []
        origins: list[InputOrigin] = []

        for flat_index, (path, value) in enumerate(path_leaves):
            if not path or not isinstance(path[0], jax.tree_util.SequenceKey):
                raise TypeError(
                    "canonical call PyTree did not start with a parameter tuple index"
                )

            parameter_index = int(path[0].idx)
            if parameter_index < 0 or parameter_index >= len(parameter_names):
                raise RuntimeError("invalid parameter index in canonical PyTree path")

            flat.append(value)
            origins.append(
                InputOrigin(
                    flat_index,
                    parameter_index,
                    parameter_names[parameter_index],
                    path[1:],
                )
            )

        return (
            cls(
                signature=signature,
                parameter_names=parameter_names,
                treedef=treedef,
                input_origins=tuple(origins),
            ),
            tuple(flat),
        )

    def bind(
        self,
        *args,
        **kwargs,
    ) -> inspect.BoundArguments:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def flatten_call(
        self,
        *args,
        **kwargs,
    ) -> tuple[Any, ...]:
        bound = self.bind(*args, **kwargs)
        canonical = tuple(bound.arguments[name] for name in self.parameter_names)
        flat, treedef = jax.tree_util.tree_flatten(canonical)

        if treedef != self.treedef:
            raise ValueError(
                "call argument pytree differs from the "
                "structure used when the functional was captured"
            )

        return tuple(flat)

    def unflatten(
        self,
        flat: tuple[Any, ...],
    ) -> inspect.BoundArguments:
        canonical = jax.tree_util.tree_unflatten(self.treedef, flat)
        arguments = dict(zip(self.parameter_names, canonical, strict=True))

        return inspect.BoundArguments(self.signature, arguments)
