from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evaluation.scalable_multiworkflow_v2.offline_analysis import (
    POLICY_NAMES,
    STEP8A_POLICY_NAMES,
    load_offline_artifact,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    ANCHOR_DEPTHS,
    BUDGETS_BY_WORKFLOW_COUNT,
    FANOUTS_BY_WORKFLOW_COUNT,
    build_scenario,
)


_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "scalable_multiworkflow_v2"
    / "offline_summary.json"
)


@pytest.mark.parametrize(
    ("workflow_count", "pending_count", "candidate_count"),
    ((8, 20, 12), (16, 60, 20)),
)
def test_workload_sizes(
    workflow_count: int,
    pending_count: int,
    candidate_count: int,
) -> None:
    scenario = build_scenario(workflow_count)

    assert len(scenario.metadata.workflows) == workflow_count
    assert len(scenario.continuations) == pending_count
    assert len(scenario.candidates) == candidate_count


@pytest.mark.parametrize("workflow_count", (8, 16))
def test_factorial_combinations_are_complete(workflow_count: int) -> None:
    scenario = build_scenario(workflow_count)
    observed = Counter(
        (workflow.anchor_pos, workflow.pending_fanout)
        for workflow in scenario.metadata.workflows
    )
    expected = {
        (anchor_pos, fanout): 1
        for anchor_pos in ANCHOR_DEPTHS
        for fanout in FANOUTS_BY_WORKFLOW_COUNT[workflow_count]
    }

    assert observed == expected
    assert all(
        {workflow.anchor_pos for workflow in scenario.metadata.workflows
         if workflow.pending_fanout == fanout}
        == set(ANCHOR_DEPTHS)
        for fanout in FANOUTS_BY_WORKFLOW_COUNT[workflow_count]
    )


@pytest.mark.parametrize("workflow_count", (8, 16))
def test_shallow_checkpoint_rule(workflow_count: int) -> None:
    scenario = build_scenario(workflow_count)
    candidates_by_workflow: dict[str, list] = {}
    for candidate in scenario.candidates:
        candidates_by_workflow.setdefault(
            candidate.workflow_id,
            [],
        ).append(candidate)

    for anchor_pos in ANCHOR_DEPTHS:
        group = tuple(
            workflow
            for workflow in scenario.metadata.workflows
            if workflow.anchor_pos == anchor_pos
        )
        first = group[0]
        assert first.pending_fanout == min(
            FANOUTS_BY_WORKFLOW_COUNT[workflow_count]
        )
        shallow = tuple(
            candidate
            for candidate in candidates_by_workflow[first.workflow_id]
            if candidate.checkpoint_id.endswith("_SHALLOW")
        )
        assert len(shallow) == 1
        assert shallow[0].token_pos == anchor_pos // 2
        assert all(
            not any(
                candidate.checkpoint_id.endswith("_SHALLOW")
                for candidate in candidates_by_workflow[workflow.workflow_id]
            )
            for workflow in group[1:]
        )


@pytest.mark.parametrize("workflow_count", (8, 16))
def test_budget_options_and_planning_targets(workflow_count: int) -> None:
    for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]:
        scenario = build_scenario(workflow_count, budget)
        assert scenario.metadata.budget_checkpoints == budget
        assert scenario.budget_bytes == (
            budget * scenario.metadata.checkpoint_size_bytes
        )
        workflows_by_id = {
            workflow.workflow_id: workflow
            for workflow in scenario.metadata.workflows
        }
        assert all(
            continuation.planning_target
            == workflows_by_id[continuation.workflow_id].anchor_pos
            for continuation in scenario.continuations
        )
        assert all(
            continuation.lineage_path[0] == "P"
            for continuation in scenario.continuations
        )


def test_offline_artifact_contains_complete_policy_matrix() -> None:
    result = load_offline_artifact(_ARTIFACT_PATH)
    artifact_policy_names = {
        row.policy_name for row in result.rows
    }
    assert artifact_policy_names in (
        set(STEP8A_POLICY_NAMES),
        set(POLICY_NAMES),
    )

    policy_count = len(artifact_policy_names)
    assert len(result.rows) == 8 * policy_count
    assert Counter(
        (row.workflow_count, row.budget_checkpoints)
        for row in result.rows
    ) == {
        (workflow_count, budget): policy_count
        for workflow_count in (8, 16)
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]
    }


def test_all_policies_obey_budget_uniqueness_and_residency() -> None:
    result = load_offline_artifact(_ARTIFACT_PATH)

    for row in result.rows:
        scenario = build_scenario(
            row.workflow_count,
            row.budget_checkpoints,
        )
        resident_ids = {
            candidate.checkpoint_id
            for candidate in scenario.candidates
            if candidate.recurrent_resident
        }
        assert len(row.selected_checkpoint_ids) <= row.budget_checkpoints
        assert len(set(row.selected_checkpoint_ids)) == len(
            row.selected_checkpoint_ids
        )
        assert set(row.selected_checkpoint_ids) <= resident_ids


def test_oracle_is_no_worse_and_flowstate_is_monotonic() -> None:
    result = load_offline_artifact(_ARTIFACT_PATH)
    artifact_policy_names = {
        row.policy_name for row in result.rows
    }
    rows = {
        (
            row.workflow_count,
            row.budget_checkpoints,
            row.policy_name,
        ): row
        for row in result.rows
    }

    for workflow_count in (8, 16):
        budgets = BUDGETS_BY_WORKFLOW_COUNT[workflow_count]
        flowstate_costs = tuple(
            rows[(workflow_count, budget, "FlowState")]
            .estimated_recovery_cost_ms
            for budget in budgets
        )
        assert all(
            current >= following
            for current, following in zip(
                flowstate_costs,
                flowstate_costs[1:],
            )
        )
        for budget in budgets:
            oracle_cost = rows[
                (workflow_count, budget, "Oracle")
            ].estimated_recovery_cost_ms
            assert all(
                oracle_cost
                <= rows[
                    (workflow_count, budget, policy_name)
                ].estimated_recovery_cost_ms + 1e-9
                for policy_name in artifact_policy_names
            )


def test_full_budget_reaches_complete_coverage() -> None:
    result = load_offline_artifact(_ARTIFACT_PATH)
    rows = {
        (
            row.workflow_count,
            row.budget_checkpoints,
            row.policy_name,
        ): row
        for row in result.rows
    }

    for workflow_count in (8, 16):
        for policy_name in ("FlowState", "Oracle"):
            row = rows[(workflow_count, workflow_count, policy_name)]
            assert row.total_recovery_gap == 0
            assert row.estimated_recovery_cost_ms == 0.0
            assert row.planning_executable_prefix_ratio == 1.0


def test_workflow_and_recovery_only_differentiate() -> None:
    result = load_offline_artifact(_ARTIFACT_PATH)
    rows = {
        (
            row.workflow_count,
            row.budget_checkpoints,
            row.policy_name,
        ): row
        for row in result.rows
    }

    assert any(
        rows[(workflow_count, budget, "Workflow-Only")]
        .selected_checkpoint_ids
        != rows[(workflow_count, budget, "Recovery-Only")]
        .selected_checkpoint_ids
        for workflow_count in (8, 16)
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]
    )
