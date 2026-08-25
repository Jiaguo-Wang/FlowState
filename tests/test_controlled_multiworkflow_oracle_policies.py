from __future__ import annotations

from dataclasses import replace

import pytest

import evaluation.controlled_multiworkflow_v1.policies as policies_module
from evaluation.controlled_multiworkflow_v1.budget_sweep import (
    BUDGET_SWEEP_POLICY_NAMES,
    build_budget_sweep,
)
from evaluation.controlled_multiworkflow_v1.policies import (
    select_oracle,
    select_workflow_only,
)
from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


def test_workflow_only_does_not_access_recovery_cost_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = build_scenario()

    class ForbiddenRecoveryCostModel:
        """在策略意外构造恢复模型时立即使测试失败。"""

        def __init__(self) -> None:
            raise AssertionError("Workflow-Only 不得访问恢复成本模型")

    monkeypatch.setattr(
        policies_module,
        "RecoveryCostModel",
        ForbiddenRecoveryCostModel,
    )

    selected = select_workflow_only(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )

    assert selected == ("W4_PARENT", "W1_PARENT", "W2_PARENT")


def test_workflow_only_does_not_score_token_depth() -> None:
    continuation = PendingContinuation(
        continuation_id="pending",
        workflow_id="W",
        lineage_path=("ROOT", "B"),
        anchor_pos=100,
        resident_fa_frontier=100,
    )
    candidates = (
        CheckpointCandidate(
            checkpoint_id="A",
            workflow_id="W",
            lineage_path=("ROOT",),
            token_pos=1,
            memory_bytes=10,
        ),
        CheckpointCandidate(
            checkpoint_id="B",
            workflow_id="W",
            lineage_path=("ROOT",),
            token_pos=99,
            memory_bytes=10,
        ),
    )

    shallow_a = select_workflow_only(
        (continuation,),
        candidates,
        budget_bytes=10,
    )
    deep_a = select_workflow_only(
        (continuation,),
        (
            replace(candidates[0], token_pos=99),
            replace(candidates[1], token_pos=1),
        ),
        budget_bytes=10,
    )

    assert shallow_a == ("A",)
    assert deep_a == ("A",)


def test_oracle_obeys_budget_and_ignores_nonresident_candidate() -> None:
    scenario = build_scenario()
    nonresident = replace(
        scenario.candidates[0],
        checkpoint_id="A_NONRESIDENT",
        recurrent_resident=False,
    )
    selected = select_oracle(
        scenario.continuations,
        scenario.candidates + (nonresident,),
        scenario.budget_bytes,
        RecoveryCostModel(),
    )

    assert len(selected) <= 3
    assert len(set(selected)) == len(selected)
    assert "A_NONRESIDENT" not in selected


def test_oracle_objective_is_no_worse_than_every_policy() -> None:
    rows = build_budget_sweep().rows

    for budget_checkpoints in range(1, 6):
        budget_rows = tuple(
            row
            for row in rows
            if row.budget_checkpoints == budget_checkpoints
        )
        oracle = next(
            row for row in budget_rows if row.policy_name == "Oracle"
        )
        assert all(
            oracle.estimated_recovery_cost_ms
            <= row.estimated_recovery_cost_ms + 1e-9
            for row in budget_rows
        )


def test_oracle_is_deterministic_under_candidate_reordering() -> None:
    scenario = build_scenario()
    model = RecoveryCostModel()
    orders = (
        scenario.candidates,
        tuple(reversed(scenario.candidates)),
        (
            scenario.candidates[3],
            scenario.candidates[0],
            scenario.candidates[4],
            scenario.candidates[1],
            scenario.candidates[2],
        ),
    )

    selections = tuple(
        select_oracle(
            scenario.continuations,
            order,
            scenario.budget_bytes,
            model,
        )
        for order in orders
    )

    assert selections[0] == selections[1] == selections[2]


def test_oracle_reuses_core_recovery_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = build_scenario()
    original_recovery_gap = policies_module.recovery_gap
    calls = 0

    def tracked_recovery_gap(continuation, selected):
        """记录 Oracle 对核心恢复间隔函数的直接调用。"""
        nonlocal calls
        calls += 1
        return original_recovery_gap(continuation, selected)

    monkeypatch.setattr(
        policies_module,
        "recovery_gap",
        tracked_recovery_gap,
    )

    select_oracle(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
        RecoveryCostModel(),
    )

    assert calls > 0


def test_sweep_contains_eight_policies_and_oracle_differences() -> None:
    result = build_budget_sweep()

    assert len(result.rows) == 40
    assert {
        row.policy_name for row in result.rows
    } == set(BUDGET_SWEEP_POLICY_NAMES)
    assert len(result.oracle_comparisons) == 5
    assert all(
        comparison.oracle_cost_difference >= 0.0
        for comparison in result.oracle_comparisons
    )
