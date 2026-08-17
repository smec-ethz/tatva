import pytest

from tatva.tracer.core.nested import (
    CallInvocation,
    CondContext,
    CondInvocation,
    CondSpec,
    FrameStep,
    IndexedChild,
    MapContext,
    MapSpec,
    NestedKind,
    RepeatedInvocation,
    TraversalOrder,
    collect_logical_output,
    dispatch_nested,
)


def test_cond_invocation_owns_its_frame_step_and_child_lookup():
    invocation = CondInvocation(eqn_index=5, branch_index=1, body="branch_1")
    child = invocation.children()[0]

    assert invocation.kind is NestedKind.COND
    assert child.frame_step == FrameStep(5, NestedKind.COND, iteration=1)
    assert child.logical_index == 1
    assert invocation.child_at(child.frame_step) == "branch_1"

    mapped = invocation.map_children(lambda c: c.payload.upper())
    assert mapped.body == "BRANCH_1"
    assert mapped.branch_index == 1


def test_call_invocation_owns_its_frame_step_and_child_lookup():
    invocation = CallInvocation(eqn_index=4, body="body")
    child = invocation.children()[0]

    assert invocation.kind is NestedKind.CALL
    assert child.frame_step == FrameStep(4, NestedKind.CALL)
    assert child.logical_index is None
    assert invocation.child_at(child.frame_step) == "body"


def test_repeated_invocation_centralizes_execution_and_logical_order():
    invocation = RepeatedInvocation(
        eqn_index=7,
        kind=NestedKind.SCAN,
        iterations=(
            IndexedChild(index=2, body="two"),
            IndexedChild(index=1, body="one"),
            IndexedChild(index=0, body="zero"),
        ),
    )

    assert [child.payload for child in invocation.children()] == ["two", "one", "zero"]
    assert [
        child.payload for child in invocation.children(TraversalOrder.REVERSE_EXECUTION)
    ] == ["zero", "one", "two"]
    assert [child.payload for child in invocation.children(TraversalOrder.LOGICAL)] == [
        "zero",
        "one",
        "two",
    ]


def test_repeated_invocation_resolves_logical_frame_and_maps_topology():
    invocation = RepeatedInvocation(
        eqn_index=3,
        kind=NestedKind.MAP,
        iterations=(
            IndexedChild(index=1, body="b"),
            IndexedChild(index=0, body="a"),
        ),
    )

    step = FrameStep(3, NestedKind.MAP, iteration=0)
    assert invocation.child_at(step) == "a"

    mapped = invocation.map_children(lambda child: child.payload.upper())
    assert [item.body for item in mapped.iterations] == ["B", "A"]


def test_invocation_rejects_invalid_kind_paths_and_duplicate_indices():
    invocation = RepeatedInvocation(
        eqn_index=3,
        kind=NestedKind.MAP,
        iterations=(IndexedChild(index=0, body="a"),),
    )

    with pytest.raises(ValueError, match="expects scan"):
        invocation.child_at(FrameStep(3, NestedKind.SCAN, iteration=0))

    with pytest.raises(ValueError, match="duplicate logical indices"):
        RepeatedInvocation(
            eqn_index=3,
            kind=NestedKind.MAP,
            iterations=(
                IndexedChild(index=0, body="a"),
                IndexedChild(index=0, body="b"),
            ),
        )


def test_repeated_spec_owns_execution_order():
    forward = MapSpec(num_consts=0, length=3, reverse=False)
    reverse = MapSpec(num_consts=0, length=3, reverse=True)

    assert forward.execution_indices() == (0, 1, 2)
    assert reverse.execution_indices() == (2, 1, 0)


def test_dispatch_validates_spec_and_invocation_before_handler():
    invocation = RepeatedInvocation.from_spec(
        2,
        MapSpec(num_consts=0, length=1, reverse=False),
        (IndexedChild(0, "body"),),
    )

    class Handler:
        def call(self, context):
            raise AssertionError

        def map(self, context: MapContext[str]):
            return context.spec.length, context.invocation.child_at_index(0)

        def scan(self, context):
            raise AssertionError

    assert dispatch_nested(
        MapSpec(num_consts=0, length=1, reverse=False), invocation, Handler()
    ) == (1, "body")


def test_collect_logical_output_validates_and_reorders_entries():
    entries = ((1, ("b",)), (0, ("a",)))
    assert collect_logical_output(entries, output_index=0, length=2, label="map") == (
        "a",
        "b",
    )

    with pytest.raises(RuntimeError, match="missing one or more iterations"):
        collect_logical_output(((0, ("a",)),), output_index=0, length=2, label="map")
