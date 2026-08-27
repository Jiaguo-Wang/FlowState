from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.controlled_multiworkflow_v1.policies import select_oracle
from evaluation.formal_model_regression import (
    BASELINE_SELECTION_INDEPENDENT_POLICIES,
    build_controlled_regression,
    build_h100_selection_audit,
    build_trace_model_compatibility,
    exact_oracle_selection,
)
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel


ROOT = Path(__file__).resolve().parents[1]


class RecordingRecoveryModel:
    """记录 optimizer 是否显式传入固定规划目标。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def estimate(self, gap_tokens: int, target_tokens: int) -> float:
        self.calls.append((gap_tokens, target_tokens))
        return float(gap_tokens)


def test_optimizer_passes_each_continuation_target() -> None:
    scenario = build_scenario()
    model = RecordingRecoveryModel()
    GlobalOptimizer(model).select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )
    expected_targets = {
        continuation.planning_target for continuation in scenario.continuations
    }
    assert model.calls
    assert {target for _, target in model.calls} == expected_targets
    assert all(0 <= gap <= target for gap, target in model.calls)


def test_exact_oracle_uses_position_aware_cost_and_respects_budget() -> None:
    scenario = build_scenario()
    model = RecordingRecoveryModel()
    selected = exact_oracle_selection(
        scenario.continuations,
        scenario.candidates,
        3,
        model,
    )
    assert len(selected) <= 3
    assert len(set(selected)) == len(selected)
    assert model.calls
    assert all(0 <= gap <= target for gap, target in model.calls)


def test_dynamic_oracle_matches_independent_brute_force() -> None:
    scenario = build_scenario()
    model = RecoveryCostModel()
    dynamic = exact_oracle_selection(
        scenario.continuations,
        scenario.candidates,
        3,
        model,
    )
    brute_force = select_oracle(
        scenario.continuations,
        scenario.candidates,
        3 * scenario.metadata.checkpoint_size_bytes,
        model,
    )
    assert dynamic == brute_force


def test_formal_model_preserves_diminishing_returns_structure() -> None:
    scenario = build_scenario()
    model = RecoveryCostModel()
    by_id = {
        candidate.checkpoint_id: candidate for candidate in scenario.candidates
    }
    continuation = next(
        item for item in scenario.continuations if item.continuation_id == "W1-A"
    )
    shallow = by_id["W1_SHALLOW"]
    parent = by_id["W1_PARENT"]

    def cost(selected) -> float:
        return model.estimate(
            recovery_gap(continuation, selected),
            continuation.planning_target,
        )

    shallow_gain_from_empty = cost(()) - cost((shallow,))
    shallow_gain_after_parent = cost((parent,)) - cost((parent, shallow))
    parent_gain_from_empty = cost(()) - cost((parent,))
    parent_gain_after_shallow = cost((shallow,)) - cost((shallow, parent))
    assert shallow_gain_from_empty > 0.0
    assert shallow_gain_after_parent == pytest.approx(0.0)
    assert parent_gain_after_shallow <= parent_gain_from_empty


@pytest.fixture(scope="module")
def regression_result():
    return build_controlled_regression()


def test_controlled_regression_and_oracle_are_complete(regression_result) -> None:
    diffs, oracle_rows = regression_result
    assert len(diffs) == 17 * 8
    assert len(oracle_rows) == 17
    flowstate = tuple(row for row in diffs if row.policy_name == "FlowState")
    assert len(flowstate) == 17
    assert sum(row.selection_changed for row in flowstate) == 3
    assert all(row.absolute_regret_ms >= 0.0 for row in oracle_rows)
    assert all(row.absolute_regret_ms == pytest.approx(0.0) for row in oracle_rows)


def test_frozen_baseline_selection_does_not_change(regression_result) -> None:
    diffs, _ = regression_result
    assert not tuple(
        row
        for row in diffs
        if row.policy_name in BASELINE_SELECTION_INDEPENDENT_POLICIES
        and row.selection_changed
    )


def test_h100_reusability_gate_identifies_only_changed_points() -> None:
    audit = build_h100_selection_audit()
    by_point = {row["point"]: row for row in audit["points"]}
    assert by_point["Scalable N16 K4"]["classification"] == "CHANGED"
    assert by_point["Scalable N16 K12"]["classification"] == "IDENTICAL"
    assert by_point["SOTA-signal K4"]["classification"] == "IDENTICAL"
    assert by_point["SOTA-signal K8"]["classification"] == "CHANGED"
    assert audit["gpu_rerun_points"] == [
        "Scalable N16 K4",
        "SOTA-signal K8",
    ]


def test_tracelab_compatibility_dry_run_executes_no_policy(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("TraceLab 兼容性检查不得执行策略")

    monkeypatch.setattr(GlobalOptimizer, "select", forbidden)
    result = build_trace_model_compatibility(sample_snapshot_count=4)
    assert result["status"] == "PASS"
    assert result["policy_comparison_executed"] is False
    assert result["continuations_checked"] > 0


def test_written_trace_artifact_records_no_policy_comparison() -> None:
    path = (
        ROOT
        / "evaluation"
        / "formal_model_regression"
        / "trace_model_compatibility.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["policy_comparison_executed"] is False
