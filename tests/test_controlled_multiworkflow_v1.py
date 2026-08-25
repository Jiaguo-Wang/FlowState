from __future__ import annotations

from evaluation.controlled_multiworkflow_v1.scenario import (
    BUDGET_CHECKPOINTS,
    CHECKPOINT_SIZE_BYTES,
    build_scenario,
)
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel


def test_controlled_scenario_structure_and_metadata() -> None:
    scenario = build_scenario()
    workflow_metadata = {
        workflow.workflow_id: (
            workflow.root_lineage,
            workflow.anchor_pos,
            workflow.pending_fanout,
            workflow.pending_branches,
        )
        for workflow in scenario.metadata.workflows
    }

    assert workflow_metadata == {
        "W1": ("ROOT1", 32_768, 2, ("A", "B")),
        "W2": ("ROOT2", 16_384, 1, ("B",)),
        "W3": ("ROOT3", 8_192, 1, ("B",)),
        "W4": ("ROOT4", 4_096, 3, ("A", "B", "C")),
    }
    assert len(scenario.continuations) == 7
    assert len(scenario.candidates) == 5
    assert scenario.metadata.checkpoint_size_bytes == CHECKPOINT_SIZE_BYTES
    assert scenario.metadata.budget_checkpoints == BUDGET_CHECKPOINTS
    assert scenario.budget_bytes == 3 * CHECKPOINT_SIZE_BYTES

    continuations = {
        continuation.continuation_id: (
            continuation.workflow_id,
            continuation.lineage_path,
            continuation.anchor_pos,
            continuation.resident_fa_frontier,
        )
        for continuation in scenario.continuations
    }
    assert continuations == {
        "W1-A": ("W1", ("ROOT1", "A"), 32_768, 32_768),
        "W1-B": ("W1", ("ROOT1", "B"), 32_768, 32_768),
        "W2-B": ("W2", ("ROOT2", "B"), 16_384, 16_384),
        "W3-B": ("W3", ("ROOT3", "B"), 8_192, 8_192),
        "W4-A": ("W4", ("ROOT4", "A"), 4_096, 4_096),
        "W4-B": ("W4", ("ROOT4", "B"), 4_096, 4_096),
        "W4-C": ("W4", ("ROOT4", "C"), 4_096, 4_096),
    }

    candidates = {
        candidate.checkpoint_id: (
            candidate.workflow_id,
            candidate.lineage_path,
            candidate.token_pos,
        )
        for candidate in scenario.candidates
    }
    assert candidates == {
        "W1_PARENT": ("W1", ("ROOT1",), 32_768),
        "W1_SHALLOW": ("W1", ("ROOT1",), 16_384),
        "W2_PARENT": ("W2", ("ROOT2",), 16_384),
        "W3_PARENT": ("W3", ("ROOT3",), 8_192),
        "W4_PARENT": ("W4", ("ROOT4",), 4_096),
    }
    assert all(
        candidate.memory_bytes == CHECKPOINT_SIZE_BYTES
        and candidate.recurrent_resident
        and candidate.fa_resident
        for candidate in scenario.candidates
    )


def test_existing_optimizer_matches_controlled_scenario_analysis() -> None:
    scenario = build_scenario()
    optimizer = GlobalOptimizer(RecoveryCostModel())

    result = optimizer.select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )
    selected_ids = tuple(
        candidate.checkpoint_id for candidate in result.selected
    )
    w3 = next(
        continuation
        for continuation in scenario.continuations
        if continuation.continuation_id == "W3-B"
    )

    assert selected_ids == (
        "W1_PARENT",
        "W2_PARENT",
        "W4_PARENT",
    )
    assert recovery_gap(w3, result.selected) == 8_192
