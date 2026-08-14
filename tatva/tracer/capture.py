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
class CallABI:
    """Canonical mapping between a Python call and flat JAXPR inputs."""

    signature: inspect.Signature
    treedef: jax.tree_util.PyTreeDef

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.signature.parameters)

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
        flat, treedef = jax.tree_util.tree_flatten(canonical)

        return (
            cls(signature=signature, treedef=treedef),
            tuple(flat),
        )

    def bind(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> inspect.BoundArguments:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def flatten_bound(
        self,
        bound: inspect.BoundArguments,
    ) -> tuple[Any, ...]:
        canonical = tuple(bound.arguments[name] for name in self.parameter_names)

        try:
            return tuple(self.treedef.flatten_up_to(canonical))
        except ValueError as exc:
            raise ValueError(
                "call argument PyTree differs from the "
                "structure used when the functional was captured"
            ) from exc

    def unflatten(
        self,
        flat: tuple[Any, ...],
    ) -> inspect.BoundArguments:
        canonical = self.treedef.unflatten(flat)
        arguments = dict(zip(self.parameter_names, canonical, strict=True))

        return inspect.BoundArguments(self.signature, arguments)

    def parameter_trees(
        self,
    ) -> tuple[
        tuple[
            str,
            jax.tree_util.PyTreeDef,
            slice,
        ],
        ...,
    ]:
        """Top-level parameter trees and their flat ABI ranges."""

        trees = self.treedef.children()
        if len(trees) != len(self.parameter_names):
            raise RuntimeError(
                "canonical call tree does not match the Python function signature"
            )

        offset = 0
        result = []

        for name, tree in zip(self.parameter_names, trees, strict=True):
            stop = offset + tree.num_leaves
            result.append((name, tree, slice(offset, stop)))
            offset = stop

        return tuple(result)
