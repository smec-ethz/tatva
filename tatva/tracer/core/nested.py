from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, cast

from jax.extend.core import ClosedJaxpr, Jaxpr


class NestedKind(Enum):
    CALL = auto()
    MAP = auto()
    SCAN = auto()


class CallKind(Enum):
    JIT = auto()
    REMAT = auto()


class TraversalOrder(Enum):
    EXECUTION = auto()
    REVERSE_EXECUTION = auto()
    LOGICAL = auto()


@dataclass(frozen=True)
class CallSpec:
    call_kind: CallKind

    @property
    def kind(self) -> NestedKind:
        return NestedKind.CALL


class RepeatedSpec:
    length: int
    reverse: bool

    def execution_indices(self) -> tuple[int, ...]:
        indices = range(self.length - 1, -1, -1) if self.reverse else range(self.length)
        return tuple(indices)


@dataclass(frozen=True)
class MapSpec(RepeatedSpec):
    num_consts: int
    length: int
    reverse: bool

    @property
    def kind(self) -> NestedKind:
        return NestedKind.MAP


@dataclass(frozen=True)
class ScanSpec(RepeatedSpec):
    num_consts: int
    num_carry: int
    length: int
    reverse: bool

    @property
    def kind(self) -> NestedKind:
        return NestedKind.SCAN


type NestedSpec = CallSpec | MapSpec | ScanSpec


class NestedSpecHandler[R](Protocol):
    def call(self, spec: CallSpec) -> R: ...
    def map(self, spec: MapSpec) -> R: ...
    def scan(self, spec: ScanSpec) -> R: ...


def dispatch_nested_spec[R](spec: NestedSpec, handler: NestedSpecHandler[R]) -> R:
    if isinstance(spec, CallSpec):
        return handler.call(spec)
    if isinstance(spec, MapSpec):
        return handler.map(spec)
    if isinstance(spec, ScanSpec):
        return handler.scan(spec)
    raise AssertionError(f"unsupported nested spec {spec!r}")


@dataclass(frozen=True, slots=True)
class FrameStep:
    """One step from a parent frame into a nested invocation."""

    eqn_index: int
    kind: NestedKind
    iteration: int | None = None


type FramePath = tuple[FrameStep, ...]


@dataclass(frozen=True)
class IndexedChild[T]:
    """A repeated child stored with its logical map/scan index."""

    index: int
    body: T


@dataclass(frozen=True)
class NestedChild[T]:
    """A child together with its canonical path and logical identity."""

    payload: T
    frame_step: FrameStep
    logical_index: int | None


class NestedInvocation[T](Protocol):
    eqn_index: int

    @property
    def kind(self) -> NestedKind: ...

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]: ...

    def child_at(self, step: FrameStep) -> T: ...

    def map_children[U](
        self, fn: Callable[[NestedChild[T]], U]
    ) -> NestedInvocation[U]: ...


@dataclass(frozen=True)
class CallInvocation[T]:
    eqn_index: int
    body: T

    @property
    def kind(self) -> NestedKind:
        return NestedKind.CALL

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]:
        del order
        step = FrameStep(eqn_index=self.eqn_index, kind=NestedKind.CALL)
        return (NestedChild(self.body, step, None),)

    def child_at(self, step: FrameStep) -> T:
        _validate_step(self.eqn_index, self.kind, step)
        if step.iteration is not None:
            raise ValueError("call frame step must not specify an iteration")
        return self.body

    def map_children[U](self, fn: Callable[[NestedChild[T]], U]) -> CallInvocation[U]:
        return CallInvocation(self.eqn_index, fn(self.children()[0]))


@dataclass(frozen=True)
class RepeatedInvocation[T]:
    """Map or scan children, stored once in actual execution order."""

    eqn_index: int
    kind: NestedKind
    iterations: tuple[IndexedChild[T], ...]

    def __post_init__(self) -> None:
        if self.kind not in (NestedKind.MAP, NestedKind.SCAN):
            raise ValueError("repeated invocation kind must be MAP or SCAN")
        indices = tuple(item.index for item in self.iterations)
        if len(indices) != len(set(indices)):
            raise ValueError("repeated invocation has duplicate logical indices")

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]:
        items = self.iterations
        if order is TraversalOrder.REVERSE_EXECUTION:
            items = tuple(reversed(items))
        elif order is TraversalOrder.LOGICAL:
            items = tuple(sorted(items, key=lambda item: item.index))

        return tuple(
            NestedChild(
                payload=item.body,
                frame_step=FrameStep(
                    eqn_index=self.eqn_index,
                    kind=self.kind,
                    iteration=item.index,
                ),
                logical_index=item.index,
            )
            for item in items
        )

    def child_at(self, step: FrameStep) -> T:
        _validate_step(self.eqn_index, self.kind, step)
        if step.iteration is None:
            raise ValueError(
                f"{self.kind.name.lower()} frame step requires an iteration"
            )
        return self.child_at_index(step.iteration)

    def frame_step(self, index: int) -> FrameStep:
        return FrameStep(self.eqn_index, self.kind, iteration=index)

    def child_at_index(self, index: int) -> T:
        child = next(
            (item.body for item in self.iterations if item.index == index),
            None,
        )
        if child is None:
            raise KeyError(f"{self.kind.name.lower()} has no iteration {index}")
        return child

    def map_children[U](
        self, fn: Callable[[NestedChild[T]], U]
    ) -> RepeatedInvocation[U]:
        children = self.children()
        return self.with_children(
            tuple(
                IndexedChild(cast(int, child.logical_index), fn(child))
                for child in children
            )
        )

    def with_children[U](
        self, iterations: tuple[IndexedChild[U], ...]
    ) -> RepeatedInvocation[U]:
        return RepeatedInvocation(self.eqn_index, self.kind, iterations)

    @staticmethod
    def from_spec[U](
        eqn_index: int,
        spec: MapSpec | ScanSpec,
        iterations: tuple[IndexedChild[U], ...],
    ) -> RepeatedInvocation[U]:
        return RepeatedInvocation(eqn_index, spec.kind, iterations)


type AnyNestedInvocation[T] = CallInvocation[T] | RepeatedInvocation[T]


@dataclass(frozen=True)
class CallContext[T]:
    spec: CallSpec
    invocation: CallInvocation[T]


@dataclass(frozen=True)
class MapContext[T]:
    spec: MapSpec
    invocation: RepeatedInvocation[T]


@dataclass(frozen=True)
class ScanContext[T]:
    spec: ScanSpec
    invocation: RepeatedInvocation[T]


type NestedContext[T] = CallContext[T] | MapContext[T] | ScanContext[T]


def _validate_step(eqn_index: int, kind: NestedKind, step: FrameStep) -> None:
    if step.eqn_index != eqn_index or step.kind is not kind:
        raise ValueError(
            f"frame step expects {step.kind.name.lower()} at equation "
            f"{step.eqn_index}, got {kind.name.lower()} at equation {eqn_index}"
        )


class NestedHandler[T, R](Protocol):
    def call(self, context: CallContext[T]) -> R: ...
    def map(self, context: MapContext[T]) -> R: ...
    def scan(self, context: ScanContext[T]) -> R: ...


def dispatch_nested[T, R](
    spec: NestedSpec,
    node: AnyNestedInvocation[T],
    handler: NestedHandler[T, R],
) -> R:
    """Validate a spec/invocation pair once, then dispatch a typed context."""
    if isinstance(spec, CallSpec) and isinstance(node, CallInvocation):
        return handler.call(CallContext(spec, cast(CallInvocation[T], node)))
    if (
        isinstance(spec, MapSpec)
        and isinstance(node, RepeatedInvocation)
        and node.kind is NestedKind.MAP
    ):
        return handler.map(MapContext(spec, cast(RepeatedInvocation[T], node)))
    if (
        isinstance(spec, ScanSpec)
        and isinstance(node, RepeatedInvocation)
        and node.kind is NestedKind.SCAN
    ):
        return handler.scan(ScanContext(spec, cast(RepeatedInvocation[T], node)))
    raise TypeError(
        f"nested spec/invocation mismatch: {spec.kind.name.lower()} spec with "
        f"{node.kind.name.lower()} invocation"
    )


def collect_logical_output[T](
    entries: Iterable[tuple[int, Sequence[T | None]]],
    *,
    output_index: int,
    length: int,
    label: str,
) -> tuple[T, ...]:
    """Collect one repeated output in logical-index order and validate coverage."""
    values: list[T | None] = [None] * length
    for logical_index, outputs in entries:
        if logical_index < 0 or logical_index >= length:
            raise IndexError(f"{label} logical index {logical_index} is out of range")
        if output_index >= len(outputs):
            raise RuntimeError(f"{label} body has no output {output_index}")
        values[logical_index] = outputs[output_index]

    if any(value is None for value in values):
        raise RuntimeError(
            f"{label} output {output_index} is missing one or more iterations"
        )
    return tuple(cast(T, value) for value in values)


@dataclass(frozen=True)
class NestedJaxpr:
    jaxpr: Jaxpr
    consts: tuple[object, ...]


def normalize_nested_jaxpr(value: Jaxpr | ClosedJaxpr) -> NestedJaxpr:
    if isinstance(value, ClosedJaxpr):
        return NestedJaxpr(value.jaxpr, tuple(value.consts))
    if isinstance(value, Jaxpr):
        return NestedJaxpr(value, ())
    raise TypeError(f"expected Jaxpr or ClosedJaxpr, got {type(value)!r}")
