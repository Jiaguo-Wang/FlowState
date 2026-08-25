from __future__ import annotations

import pytest

from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.controlled_multiworkflow_v1.snapshot_cases import (
    build_planning_summaries,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    PolicyOptimizerAdapter,
    aggregate_case_records,
    validate_clean_cache,
    validate_snapshot_runtime_observation,
)
from flowstate.recovery_model import RecoveryCostModel


def test_policy_optimizer_adapter_matches_frozen_baselines() -> None:
    scenario = build_scenario()
    model = RecoveryCostModel()
    expected = {
        summary.policy_name: summary.selected_checkpoint_ids
        for summary in build_planning_summaries(scenario, model)
    }

    for policy_name in ("Global-LRU", "Equal-Share", "Recovery-Only"):
        result = PolicyOptimizerAdapter(
            policy_name,
            scenario,
            model,
        ).select(
            scenario.continuations,
            tuple(reversed(scenario.candidates)),
            scenario.budget_bytes,
        )
        assert tuple(
            candidate.checkpoint_id for candidate in result.selected
        ) == expected[policy_name]
        assert result.used_bytes == scenario.budget_bytes
        assert result.total_benefit_ms >= 0.0


def test_policy_optimizer_adapter_rejects_non_baseline_name() -> None:
    scenario = build_scenario()
    with pytest.raises(ValueError, match="不支持的基线策略"):
        PolicyOptimizerAdapter(
            "FlowState",
            scenario,
            RecoveryCostModel(),
        )


def test_clean_cache_validation() -> None:
    census = {
        "tree": {
            "node_count": 1,
            "mamba_node_count": 0,
            "full_rows": [[0, None, None]],
        },
        "accounting": {
            "mamba_evictable": 0,
            "mamba_protected": 0,
            "full_evictable": 0,
            "full_protected": 0,
        },
    }

    assert all(validate_clean_cache(census).values())
    census["tree"]["mamba_node_count"] = 1
    with pytest.raises(RuntimeError, match="cache 未清空"):
        validate_clean_cache(census)


@pytest.mark.parametrize(
    ("planning_gap", "runtime_gap"),
    ((0, 0), (0, 1), (100, 99), (100, 100), (100, 101)),
)
def test_snapshot_runtime_gap_accepts_one_token_boundary(
    planning_gap: int,
    runtime_gap: int,
) -> None:
    anchor = 100
    if planning_gap == 0:
        physical_hit = anchor + runtime_gap
    elif runtime_gap <= planning_gap:
        physical_hit = anchor
    else:
        physical_hit = anchor + 1
    executable_prefix = physical_hit - runtime_gap

    observation = validate_snapshot_runtime_observation(
        anchor_pos=anchor,
        planning_gap=planning_gap,
        metrics={
            "physical_fa_hit": physical_hit,
            "executable_prefix": executable_prefix,
            "replay_gap": runtime_gap,
        },
    )

    assert observation["recovery_gap"] == runtime_gap


def test_snapshot_runtime_gap_rejects_larger_difference() -> None:
    with pytest.raises(RuntimeError):
        validate_snapshot_runtime_observation(
            anchor_pos=100,
            planning_gap=100,
            metrics={
                "physical_fa_hit": 100,
                "executable_prefix": 2,
                "replay_gap": 98,
            },
        )


def test_policy_aggregate_contains_required_metrics() -> None:
    scenario = build_scenario()
    planning = build_planning_summaries(scenario)
    records = [
        {
            "status": "PASS",
            "policy": "FlowState",
            "continuation_id": "W3",
            "workflow_id": "W3",
            "physical_hit": 8_192,
            "executable_prefix": 0,
            "runtime_gap": 8_192,
            "estimated_runtime_recovery_cost_ms": 10.0,
            "request_e2e_ms": 20.0,
        }
    ]

    result = aggregate_case_records(records, planning)["FlowState"]

    assert result["n_cases"] == 1
    assert result["runtime_total_gap"] == 8_192
    assert result["mean_gap_per_request"] == 8_192
    assert result["executable_prefix_ratio"] == 0.0
    assert result["estimated_recovery_cost_ms"] == 10.0
    assert result["mean_request_e2e_ms"] == 20.0
    assert result["per_workflow_runtime_gap"] == {"W3": 8_192}
