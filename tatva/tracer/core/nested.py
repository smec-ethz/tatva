from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, cast

from jax.extend.core import ClosedJaxpr, Jaxpr


class NestedKind(Enum):
    CALL = auto()
    CUSTOM_JVP = auto()
    MAP = auto()
    SCAN = auto()
    COND = auto()
    LINEAR_SOLVE = auto()


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

    # child invar i receives outer equation invar input_indices[i]
    # None means the ordinary identity boundary:
    #   child.invars == outer.eqn.invars
    input_indices: tuple[int, ...] | None = None

    @property
    def kind(self) -> NestedKind:
        return NestedKind.CALL

    def resolved_input_indices(self, outer_arity: int) -> tuple[int, ...]:
        indices = (
            tuple(range(outer_arity))
            if self.input_indices is None
            else self.input_indices
        )
        for index in indices:
            if index < 0 or index >= outer_arity:
                raise IndexError(
                    f"call boundary input index {index} outside outer arity {outer_arity}"
                )
        return indices

    def select_inputs[T](self, inputs: Sequence[T]) -> tuple[T, ...]:
        return tuple(inputs[i] for i in self.resolved_input_indices(len(inputs)))

    def outer_input_index(self, child_index: int, *, outer_arity: int) -> int:
        indices = self.resolved_input_indices(outer_arity)
        if child_index < 0 or child_index >= len(indices):
            raise IndexError(
                f"call child input {child_index} outside child arity {len(indices)}"
            )
        return indices[child_index]


@dataclass(frozen=True)
class CondSpec:
    num_branches: int

    @property
    def kind(self) -> NestedKind:
        return NestedKind.COND

    def select_inputs[T](self, inputs: Sequence[T]) -> tuple[T, ...]:
        """Extract operand inputs (skipping index 0, the branch selector)."""
        return tuple(inputs[1:])

    def outer_input_index(self, child_index: int, *, outer_arity: int) -> int:
        """Map child operand index to outer equation index (offset by +1)."""
        outer_index = 1 + child_index
        if outer_index >= outer_arity:
            raise IndexError(
                f"cond child index {child_index} outside outer arity {outer_arity}"
            )
        return outer_index


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


@dataclass(frozen=True)
class CallbackBinding:
    """A callback argument: either an outer operand or its runtime argument."""

    outer_input_index: int | None = None

    @property
    def runtime(self) -> bool:
        return self.outer_input_index is None


@dataclass(frozen=True)
class LinearSolveCallbackSpec:
    name: str
    inputs: tuple[CallbackBinding, ...]


@dataclass(frozen=True)
class LinearSolveSpec:
    matvec: LinearSolveCallbackSpec
    solve: LinearSolveCallbackSpec
    transpose_solve: LinearSolveCallbackSpec
    rhs_indices: tuple[int, ...]
    has_aux: bool

    @property
    def kind(self) -> NestedKind:
        return NestedKind.LINEAR_SOLVE

    def callbacks(self) -> tuple[LinearSolveCallbackSpec, ...]:
        return (self.matvec, self.solve, self.transpose_solve)


@dataclass(frozen=True)
class CustomJvpBinding:
    """One JVP-program input sourced from an outer primal or its tangent."""

    outer_input_index: int
    tangent: bool = False


@dataclass(frozen=True)
class CustomJvpSpec:
    """Runtime mapping for the staged JVP callback.

    jvp_bindings covers only explicit JVP jaxpr inputs: dynamic primals followed by their
    tangents. lifted primal constants stay on the outer custom-jvp equation, while
    jvp-only captures are child jaxpr constvars.
    """

    jvp_bindings: tuple[CustomJvpBinding, ...]
    output_zeros: tuple[bool, ...]

    @property
    def kind(self) -> NestedKind:
        return NestedKind.CUSTOM_JVP


type NestedSpec = (
    CallSpec | CustomJvpSpec | MapSpec | ScanSpec | CondSpec | LinearSolveSpec
)


class NestedSpecHandler[R](Protocol):
    def call(self, spec: CallSpec) -> R: ...
    def custom_jvp(self, spec: CustomJvpSpec) -> R: ...
    def map(self, spec: MapSpec) -> R: ...
    def scan(self, spec: ScanSpec) -> R: ...
    def cond(self, spec: CondSpec) -> R: ...
    def linear_solve(self, spec: LinearSolveSpec) -> R: ...


def dispatch_nested_spec[R](spec: NestedSpec, handler: NestedSpecHandler[R]) -> R:
    if isinstance(spec, CallSpec):
        return handler.call(spec)
    if isinstance(spec, CustomJvpSpec):
        return handler.custom_jvp(spec)
    if isinstance(spec, MapSpec):
        return handler.map(spec)
    if isinstance(spec, ScanSpec):
        return handler.scan(spec)
    if isinstance(spec, CondSpec):
        return handler.cond(spec)
    if isinstance(spec, LinearSolveSpec):
        return handler.linear_solve(spec)
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
class CustomJvpInvocation[T]:
    eqn_index: int
    primal: T
    jvp: T

    @property
    def kind(self) -> NestedKind:
        return NestedKind.CUSTOM_JVP

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]:
        del order
        return (
            NestedChild(
                self.primal,
                FrameStep(self.eqn_index, NestedKind.CUSTOM_JVP, 0),
                0,
            ),
            NestedChild(
                self.jvp,
                FrameStep(self.eqn_index, NestedKind.CUSTOM_JVP, 1),
                1,
            ),
        )

    def child_at(self, step: FrameStep) -> T:
        _validate_step(self.eqn_index, self.kind, step)
        if step.iteration == 0:
            return self.primal
        if step.iteration == 1:
            return self.jvp
        raise KeyError("custom_jvp callback index must be 0 or 1")

    def map_children[U](
        self, fn: Callable[[NestedChild[T]], U]
    ) -> CustomJvpInvocation[U]:
        primal, jvp = self.children()
        return CustomJvpInvocation(self.eqn_index, fn(primal), fn(jvp))


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


@dataclass(frozen=True)
class CondInvocation[T]:
    eqn_index: int
    branch_index: int
    body: T

    @property
    def kind(self) -> NestedKind:
        return NestedKind.COND

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]:
        del order
        step = FrameStep(
            eqn_index=self.eqn_index,
            kind=NestedKind.COND,
            iteration=self.branch_index,
        )
        return (NestedChild(self.body, step, self.branch_index),)

    def child_at(self, step: FrameStep) -> T:
        _validate_step(self.eqn_index, self.kind, step)
        if step.iteration != self.branch_index:
            raise KeyError(
                f"cond has active branch {self.branch_index}, requested {step.iteration}"
            )
        return self.body

    def map_children[U](self, fn: Callable[[NestedChild[T]], U]) -> CondInvocation[U]:
        return CondInvocation(self.eqn_index, self.branch_index, fn(self.children()[0]))


@dataclass(frozen=True)
class LinearSolveInvocation[T]:
    eqn_index: int
    matvec: T
    solve: T
    transpose_solve: T

    @property
    def kind(self) -> NestedKind:
        return NestedKind.LINEAR_SOLVE

    def children(
        self, order: TraversalOrder = TraversalOrder.EXECUTION
    ) -> tuple[NestedChild[T], ...]:
        del order
        return tuple(
            NestedChild(value, FrameStep(self.eqn_index, NestedKind.LINEAR_SOLVE, i), i)
            for i, value in enumerate((self.matvec, self.solve, self.transpose_solve))
        )

    def child_at(self, step: FrameStep) -> T:
        _validate_step(self.eqn_index, self.kind, step)
        if step.iteration not in (0, 1, 2):
            raise KeyError("linear solve callback index must be 0, 1, or 2")
        return (self.matvec, self.solve, self.transpose_solve)[step.iteration]

    def map_children[U](
        self, fn: Callable[[NestedChild[T]], U]
    ) -> LinearSolveInvocation[U]:
        children = self.children()
        return LinearSolveInvocation(self.eqn_index, *(fn(c) for c in children))


type AnyNestedInvocation[T] = (
    CallInvocation[T]
    | CustomJvpInvocation[T]
    | RepeatedInvocation[T]
    | CondInvocation[T]
    | LinearSolveInvocation[T]
)


@dataclass(frozen=True)
class CallContext[T]:
    spec: CallSpec
    invocation: CallInvocation[T]


@dataclass(frozen=True)
class CustomJvpContext[T]:
    spec: CustomJvpSpec
    invocation: CustomJvpInvocation[T]


@dataclass(frozen=True)
class MapContext[T]:
    spec: MapSpec
    invocation: RepeatedInvocation[T]


@dataclass(frozen=True)
class ScanContext[T]:
    spec: ScanSpec
    invocation: RepeatedInvocation[T]


@dataclass(frozen=True)
class CondContext[T]:
    spec: CondSpec
    invocation: CondInvocation[T]


@dataclass(frozen=True)
class LinearSolveContext[T]:
    spec: LinearSolveSpec
    invocation: LinearSolveInvocation[T]


type NestedContext[T] = (
    CallContext[T]
    | CustomJvpContext[T]
    | MapContext[T]
    | ScanContext[T]
    | CondContext[T]
    | LinearSolveContext[T]
)


def _validate_step(eqn_index: int, kind: NestedKind, step: FrameStep) -> None:
    if step.eqn_index != eqn_index or step.kind is not kind:
        raise ValueError(
            f"frame step expects {step.kind.name.lower()} at equation "
            f"{step.eqn_index}, got {kind.name.lower()} at equation {eqn_index}"
        )


class NestedHandler[T, R](Protocol):
    def call(self, context: CallContext[T]) -> R: ...
    def custom_jvp(self, context: CustomJvpContext[T]) -> R: ...
    def map(self, context: MapContext[T]) -> R: ...
    def scan(self, context: ScanContext[T]) -> R: ...
    def cond(self, context: CondContext[T]) -> R: ...
    def linear_solve(self, context: LinearSolveContext[T]) -> R: ...


def dispatch_nested[T, R](
    spec: NestedSpec,
    node: AnyNestedInvocation[T],
    handler: NestedHandler[T, R],
) -> R:
    """Validate a spec/invocation pair once, then dispatch a typed context."""
    if isinstance(spec, CallSpec) and isinstance(node, CallInvocation):
        return handler.call(CallContext(spec, cast(CallInvocation[T], node)))
    if isinstance(spec, CustomJvpSpec) and isinstance(node, CustomJvpInvocation):
        return handler.custom_jvp(
            CustomJvpContext(spec, cast(CustomJvpInvocation[T], node))
        )
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
    if isinstance(spec, CondSpec) and isinstance(node, CondInvocation):
        return handler.cond(CondContext(spec, cast(CondInvocation[T], node)))
    if isinstance(spec, LinearSolveSpec) and isinstance(node, LinearSolveInvocation):
        return handler.linear_solve(
            LinearSolveContext(spec, cast(LinearSolveInvocation[T], node))
        )
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


def normalize_nested_jaxpr(value: object) -> NestedJaxpr:
    if isinstance(value, ClosedJaxpr):
        return NestedJaxpr(value.jaxpr, tuple(value.consts))
    if isinstance(value, Jaxpr):
        return NestedJaxpr(value, ())

    jaxpr = getattr(value, "jaxpr", None)
    if isinstance(jaxpr, Jaxpr):
        return NestedJaxpr(jaxpr, tuple(getattr(value, "consts", ())))

    raise TypeError(
        f"expected Jaxpr, ClosedJaxpr, or Jaxpr wrapper; got {type(value)!r}"
    )
