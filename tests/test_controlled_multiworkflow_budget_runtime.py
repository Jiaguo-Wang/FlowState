from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from evaluation.controlled_multiworkflow_v1.budget_runtime import (
    build_budget_scenario,
    build_combined_budget_rows,
    build_runtime_budget_plan,
    load_case_records,
    load_runtime_summary,
    validate_case_agreement,
    validate_flowstate_monotonicity,
    validate_runtime_summary,
    write_combined_budget_csv,
)
from evaluation.controlled_multiworkflow_v1.budget_sweep import (
    build_budget_sweep,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    build_scenario,
)
from evaluation.controlled_multiworkflow_v1.snapshot_cases import (
    POLICY_NAMES,
)


_K3_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "controlled_multiworkflow_v1"
    / "artifacts"
    / "snapshot_runtime_20260825_022400_327223"
)


def test_budget_scenario_only_changes_shared_budget() -> None:
    base = build_scenario()

    for budget_checkpoints in (1, 4):
        scenario = build_budget_scenario(budget_checkpoints, base)
        assert scenario.continuations == base.continuations
        assert scenario.candidates == base.candidates
        assert scenario.metadata.workflows == base.metadata.workflows
        assert (
            scenario.metadata.checkpoint_recency
            == base.metadata.checkpoint_recency
        )
        assert scenario.budget_bytes == (
            budget_checkpoints * CHECKPOINT_SIZE_BYTES
        )
        assert scenario.metadata.budget_checkpoints == budget_checkpoints

    assert base.metadata.budget_checkpoints == 3


def test_budget_runtime_plan_reuses_offline_policy_results() -> None:
    sweep_rows = {
        (row.budget_checkpoints, row.policy_name): row
        for row in build_budget_sweep().rows
    }

    for budget_checkpoints in (1, 4):
        plan = build_runtime_budget_plan(budget_checkpoints)
        assert len(plan.cases) == 28
        assert {
            summary.policy_name: summary.selected_checkpoint_ids
            for summary in plan.planning_summaries
        } == {
            policy_name: sweep_rows[
                (budget_checkpoints, policy_name)
            ].selected_checkpoint_ids
            for policy_name in POLICY_NAMES
        }


def test_existing_k3_artifact_is_complete_and_agrees() -> None:
    summary = load_runtime_summary(_K3_ARTIFACT)
    records = load_case_records(_K3_ARTIFACT)

    validate_runtime_summary(summary)
    validate_case_agreement(records)
    assert summary["status"] == "PASS"
    assert len(records) == 28


def test_case_agreement_rejects_more_than_one_token_difference() -> None:
    records = [
        {
            "case_id": f"case-{index}",
            "status": "PASS",
            "planning_gap": 100,
            "runtime_gap": 100,
        }
        for index in range(28)
    ]
    records[7]["runtime_gap"] = 98

    with pytest.raises(ValueError, match="规划与运行时恢复间隔不一致"):
        validate_case_agreement(records)


def _summary_for_budget(budget_checkpoints: int) -> dict:
    """用离线计算结果构造聚合函数所需的最小通过汇总。"""
    rows = {
        row.policy_name: row
        for row in build_budget_sweep(
            budget_checkpoints=(budget_checkpoints,)
        ).rows
    }
    return {
        "status": "PASS",
        "cases_passed": 28,
        "cases_failed": 0,
        "planning_runtime_agreement": True,
        "artifact_directory": f"artifact-k{budget_checkpoints}",
        "safety": {
            "fa_preserved": True,
            "allocator_invariant": True,
            "tree_path_invariant": True,
            "sanity_check": True,
            "cascade_called": False,
        },
        "runtime_summary": {
            policy_name: {
                "n_cases": 7,
                "runtime_total_gap": row.total_recovery_gap,
                "executable_prefix_ratio": (
                    row.planning_executable_prefix_ratio
                ),
                "estimated_recovery_cost_ms": (
                    row.estimated_recovery_cost_ms
                ),
                "mean_gap_per_request": (
                    row.mean_recovery_gap_per_request
                ),
                "mean_request_e2e_ms": None,
            }
            for policy_name, row in rows.items()
        },
    }


def test_combined_rows_cover_three_budgets_and_four_policies() -> None:
    summaries = {
        budget: _summary_for_budget(budget)
        for budget in (1, 3, 4)
    }

    rows = build_combined_budget_rows(summaries)

    assert len(rows) == 12
    assert tuple(
        (row["K"], row["policy"]) for row in rows
    ) == tuple(
        (budget, policy_name)
        for budget in (1, 3, 4)
        for policy_name in POLICY_NAMES
    )
    validate_flowstate_monotonicity(rows)


def test_flowstate_runtime_cost_increase_is_rejected() -> None:
    summaries = {
        budget: _summary_for_budget(budget)
        for budget in (1, 3, 4)
    }
    rows = list(build_combined_budget_rows(summaries))
    changed = []
    for row in rows:
        item = dict(row)
        if item["K"] == 4 and item["policy"] == "FlowState":
            item["estimated_recovery_cost_ms"] = 10_000.0
        changed.append(item)

    with pytest.raises(ValueError, match="恢复成本非单调"):
        validate_flowstate_monotonicity(changed)


def test_combined_csv_contains_required_runtime_fields(
    tmp_path: Path,
) -> None:
    summaries = {
        budget: _summary_for_budget(budget)
        for budget in (1, 3, 4)
    }
    rows = build_combined_budget_rows(summaries)

    path = write_combined_budget_csv(
        rows,
        artifact_root=tmp_path,
        timestamp="fixed",
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 12
    assert set(saved[0]) >= {
        "K",
        "policy",
        "runtime_total_gap",
        "runtime_EPR",
        "estimated_recovery_cost_ms",
        "mean_gap_per_request",
    }


def test_runtime_summary_rejects_failed_safety() -> None:
    summary = deepcopy(_summary_for_budget(1))
    summary["safety"]["allocator_invariant"] = False

    with pytest.raises(ValueError, match="安全条件失败"):
        validate_runtime_summary(summary)
