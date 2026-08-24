from __future__ import annotations

import pytest

from flowstate.state_catalog import (
    CheckpointCandidate,
    is_compatible,
    is_lineage_prefix,
)
from flowstate.workflow import PendingContinuation


def make_continuation(
    *,
    workflow_id: str = "workflow-a",
    lineage_path: tuple[str, ...] = ("P",),
    anchor_pos: int = 32_768,
    resident_fa_frontier: int = 32_768,
) -> PendingContinuation:
    return PendingContinuation(
        continuation_id="continuation-1",
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        anchor_pos=anchor_pos,
        resident_fa_frontier=resident_fa_frontier,
    )


def make_checkpoint(
    *,
    workflow_id: str = "workflow-a",
    lineage_path: tuple[str, ...] = ("P",),
    token_pos: int = 16_000,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id="checkpoint-1",
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        token_pos=token_pos,
        memory_bytes=51_511_296,
    )


def test_planning_target_uses_anchor_when_frontier_matches() -> None:
    continuation = make_continuation(
        anchor_pos=32_768,
        resident_fa_frontier=32_768,
    )

    assert continuation.planning_target == 32_768


def test_planning_target_is_limited_by_resident_fa_frontier() -> None:
    continuation = make_continuation(
        anchor_pos=32_768,
        resident_fa_frontier=16_384,
    )

    assert continuation.planning_target == 16_384


def test_matching_workflow_and_lineage_path_before_target_is_compatible() -> None:
    checkpoint = make_checkpoint(token_pos=16_000)
    continuation = make_continuation()

    assert is_compatible(checkpoint, continuation) is True


def test_different_workflow_is_incompatible_at_same_token_position() -> None:
    checkpoint = make_checkpoint(
        workflow_id="workflow-b",
        token_pos=32_768,
    )
    continuation = make_continuation()

    assert is_compatible(checkpoint, continuation) is False


def test_sibling_lineage_is_incompatible() -> None:
    checkpoint = make_checkpoint(lineage_path=("P", "A"))
    continuation = make_continuation(lineage_path=("P", "B"))

    assert is_compatible(checkpoint, continuation) is False


def test_parent_checkpoint_is_compatible_with_child_b_at_parent_anchor() -> None:
    checkpoint = make_checkpoint(
        lineage_path=("P",),
        token_pos=32_768,
    )
    continuation = make_continuation(
        lineage_path=("P",),
        anchor_pos=32_768,
        resident_fa_frontier=32_768,
    )

    assert is_compatible(checkpoint, continuation) is True


def test_child_a_checkpoint_is_incompatible_with_child_b_at_parent_anchor() -> None:
    checkpoint = make_checkpoint(
        lineage_path=("P", "A"),
        token_pos=32_832,
    )
    continuation = make_continuation(
        lineage_path=("P",),
        anchor_pos=32_768,
        resident_fa_frontier=32_768,
    )

    assert is_lineage_prefix(
        checkpoint.lineage_path,
        continuation.lineage_path,
    ) is False
    assert is_compatible(checkpoint, continuation) is False


def test_parent_checkpoint_is_compatible_with_descendant_lineage() -> None:
    checkpoint = make_checkpoint(
        lineage_path=("P",),
        token_pos=32_768,
    )
    continuation = make_continuation(
        lineage_path=("P", "B"),
        anchor_pos=32_832,
        resident_fa_frontier=32_832,
    )

    assert is_lineage_prefix(
        checkpoint.lineage_path,
        continuation.lineage_path,
    ) is True
    assert is_compatible(checkpoint, continuation) is True


def test_sibling_checkpoint_is_not_a_lineage_prefix() -> None:
    checkpoint = make_checkpoint(
        lineage_path=("P", "A"),
        token_pos=16_000,
    )
    continuation = make_continuation(lineage_path=("P", "B"))

    assert is_lineage_prefix(
        checkpoint.lineage_path,
        continuation.lineage_path,
    ) is False
    assert is_compatible(checkpoint, continuation) is False


def test_lineage_prefix_compares_tuple_elements() -> None:
    assert is_lineage_prefix(("P", "A"), ("P", "AB")) is False


def test_checkpoint_after_planning_target_is_incompatible() -> None:
    checkpoint = make_checkpoint(token_pos=16_385)
    continuation = make_continuation(resident_fa_frontier=16_384)

    assert is_compatible(checkpoint, continuation) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("anchor_pos", -1),
        ("resident_fa_frontier", -1),
    ],
)
def test_pending_continuation_rejects_negative_positions(
    field: str,
    value: int,
) -> None:
    values = {
        "anchor_pos": 0,
        "resident_fa_frontier": 0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        make_continuation(**values)


@pytest.mark.parametrize(
    ("token_pos", "memory_bytes"),
    [
        (-1, 1),
        (0, 0),
        (0, -1),
    ],
)
def test_checkpoint_candidate_rejects_invalid_size_or_position(
    token_pos: int,
    memory_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        CheckpointCandidate(
            checkpoint_id="checkpoint-invalid",
            workflow_id="workflow-a",
            lineage_path=("P",),
            token_pos=token_pos,
            memory_bytes=memory_bytes,
        )
