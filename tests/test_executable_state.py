from __future__ import annotations

import pytest

import flowstate.executable_state as executable_state_module
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


MEMORY_BYTES = 51_511_296


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
    checkpoint_id: str,
    token_pos: int,
    *,
    workflow_id: str = "workflow-a",
    lineage_path: tuple[str, ...] = ("P",),
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        token_pos=token_pos,
        memory_bytes=MEMORY_BYTES,
    )


def test_empty_selection_has_root_frontier_and_full_gap() -> None:
    continuation = make_continuation()

    assert executable_frontier(continuation, []) == 0
    assert recovery_gap(continuation, []) == 32_768


def test_single_compatible_checkpoint_sets_frontier() -> None:
    continuation = make_continuation()
    selected = [make_checkpoint("checkpoint-16384", 16_384)]

    assert executable_frontier(continuation, selected) == 16_384
    assert recovery_gap(continuation, selected) == 16_384


def test_deepest_compatible_checkpoint_is_selected() -> None:
    continuation = make_continuation()
    selected = [
        make_checkpoint("checkpoint-8192", 8_192),
        make_checkpoint("checkpoint-16384", 16_384),
        make_checkpoint("checkpoint-32768", 32_768),
    ]

    assert executable_frontier(continuation, selected) == 32_768
    assert recovery_gap(continuation, selected) == 0


def test_deeper_checkpoint_from_another_workflow_is_ignored() -> None:
    continuation = make_continuation()
    selected = [
        make_checkpoint("compatible", 16_384),
        make_checkpoint(
            "other-workflow",
            32_768,
            workflow_id="workflow-b",
        ),
    ]

    assert executable_frontier(continuation, selected) == 16_384
    assert recovery_gap(continuation, selected) == 16_384


def test_parent_checkpoint_wins_for_sibling_continuation() -> None:
    continuation = make_continuation(lineage_path=("P", "B"))
    parent = make_checkpoint(
        "parent",
        32_768,
        lineage_path=("P",),
    )
    child_a = make_checkpoint(
        "child-a",
        32_832,
        lineage_path=("P", "A"),
    )

    assert is_compatible(parent, continuation) is True
    assert is_compatible(child_a, continuation) is False
    assert executable_frontier(continuation, [parent, child_a]) == 32_768
    assert recovery_gap(continuation, [parent, child_a]) == 0


def test_sibling_checkpoint_alone_cannot_advance_frontier() -> None:
    continuation = make_continuation(lineage_path=("P", "B"))
    child_a = make_checkpoint(
        "child-a",
        32_832,
        lineage_path=("P", "A"),
    )

    assert executable_frontier(continuation, [child_a]) == 0
    assert recovery_gap(continuation, [child_a]) == 32_768


def test_checkpoint_beyond_planning_target_is_ignored() -> None:
    continuation = make_continuation(lineage_path=("P", "B"))
    checkpoint = make_checkpoint(
        "too-deep",
        32_769,
        lineage_path=("P",),
    )

    assert executable_frontier(continuation, [checkpoint]) == 0
    assert recovery_gap(continuation, [checkpoint]) == 32_768


def test_wp3b_planning_time_frontiers() -> None:
    continuation = make_continuation(lineage_path=("P", "B"))
    parent = make_checkpoint(
        "parent",
        32_768,
        lineage_path=("P",),
    )
    child_a = make_checkpoint(
        "child-a",
        32_832,
        lineage_path=("P", "A"),
    )

    assert executable_frontier(continuation, [child_a]) == 0
    assert recovery_gap(continuation, [child_a]) == 32_768
    assert executable_frontier(continuation, [parent]) == 32_768
    assert recovery_gap(continuation, [parent]) == 0


def test_negative_recovery_gap_raises_clear_error(monkeypatch) -> None:
    continuation = make_continuation()
    monkeypatch.setattr(
        executable_state_module,
        "executable_frontier",
        lambda continuation, selected: 32_769,
    )

    with pytest.raises(ValueError, match="恢复间隔不能为负数"):
        recovery_gap(continuation, [])

