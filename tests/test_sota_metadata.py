from __future__ import annotations

from dataclasses import replace

import pytest

import evaluation.sota_metadata as metadata_module
import evaluation.scalable_multiworkflow_v2.offline_analysis as analysis_module
from evaluation.controlled_multiworkflow_v1.budget_sweep import (
    BUDGET_SWEEP_POLICY_NAMES,
    build_budget_sweep,
)
from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.scenario import (
    build_scenario as build_controlled_scenario,
)
from evaluation.scalable_multiworkflow_v2.offline_analysis import (
    POLICY_NAMES,
    STEP8A_REGRESSION_DIGEST,
    analyze_workload,
    build_step8a_regression_digest,
    run_offline_analysis,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    BUDGETS_BY_WORKFLOW_COUNT,
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_controlled_sota_metadata,
    build_kvflow_steps,
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import KVFlowStylePolicy, MarconiStylePolicy
from flowstate.state_catalog import CheckpointCandidate


@pytest.mark.parametrize("workflow_count", (4, 8, 16))
def test_all_controlled_continuations_have_immediate_kvflow_steps(
    workflow_count: int,
) -> None:
    scenario = (
        build_controlled_scenario()
        if workflow_count == 4
        else build_scalable_scenario(workflow_count)
    )

    steps = build_kvflow_steps(scenario.continuations)

    assert set(steps) == {
        continuation.continuation_id
        for continuation in scenario.continuations
    }
    assert set(steps.values()) == {1}


def test_metadata_builders_are_deterministic_under_input_reordering() -> None:
    scenario = build_scalable_scenario(8)

    assert build_kvflow_steps(
        tuple(reversed(scenario.continuations))
    ) == build_kvflow_steps(scenario.continuations)
    assert build_marconi_recency(
        tuple(reversed(scenario.candidates)),
        tuple(reversed(scenario.metadata.checkpoint_recency)),
    ) == build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    assert build_marconi_flop_saved(
        tuple(reversed(scenario.candidates))
    ) == build_marconi_flop_saved(scenario.candidates)


def test_combined_metadata_snapshot_is_built_before_policy_comparison() -> None:
    scenario = build_controlled_scenario()

    metadata = build_controlled_sota_metadata(
        scenario.continuations,
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )

    assert set(metadata.kvflow_steps.values()) == {1}
    assert metadata.marconi_alpha == 1.0
    with pytest.raises(TypeError):
        metadata.kvflow_steps["NEW"] = 2


@pytest.mark.parametrize("workflow_count", (4, 8, 16))
def test_marconi_alpha_zero_exactly_matches_existing_global_lru(
    workflow_count: int,
) -> None:
    if workflow_count == 4:
        base_scenario = build_controlled_scenario()
        budgets = range(1, 6)
    else:
        base_scenario = build_scalable_scenario(workflow_count)
        budgets = BUDGETS_BY_WORKFLOW_COUNT[workflow_count]

    recency = build_marconi_recency(
        base_scenario.candidates,
        base_scenario.metadata.checkpoint_recency,
    )
    flop_saved = build_marconi_flop_saved(base_scenario.candidates)
    for budget_k in budgets:
        budget_bytes = budget_k * base_scenario.metadata.checkpoint_size_bytes
        lru_selection = select_global_lru(
            base_scenario.candidates,
            base_scenario.metadata.checkpoint_recency,
            budget_bytes,
        )
        marconi_selection = MarconiStylePolicy().select(
            base_scenario.candidates,
            budget_k,
            recency,
            flop_saved,
            alpha=0.0,
        ).selected_checkpoint_ids

        assert marconi_selection == lru_selection


@pytest.mark.parametrize("workflow_count", (4, 8, 16))
def test_marconi_recency_is_existing_lru_rank(
    workflow_count: int,
) -> None:
    scenario = (
        build_controlled_scenario()
        if workflow_count == 4
        else build_scalable_scenario(workflow_count)
    )
    expected = {
        item.checkpoint_id: float(item.last_access_order)
        for item in scenario.metadata.checkpoint_recency
    }

    assert build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    ) == expected


def test_controlled_incremental_flop_proxy_is_parent_relative() -> None:
    scenario = build_controlled_scenario()

    flop_saved = build_marconi_flop_saved(scenario.candidates)

    assert flop_saved == {
        "W1_PARENT": 16_384.0,
        "W1_SHALLOW": 16_384.0,
        "W2_PARENT": 16_384.0,
        "W3_PARENT": 8_192.0,
        "W4_PARENT": 4_096.0,
    }


@pytest.mark.parametrize("workflow_count", (8, 16))
def test_scalable_shallow_and_main_use_incremental_span(
    workflow_count: int,
) -> None:
    scenario = build_scalable_scenario(workflow_count)
    flop_saved = build_marconi_flop_saved(scenario.candidates)
    candidates_by_workflow: dict[str, list[CheckpointCandidate]] = {}
    for candidate in scenario.candidates:
        candidates_by_workflow.setdefault(
            candidate.workflow_id,
            [],
        ).append(candidate)

    for workflow_candidates in candidates_by_workflow.values():
        main = next(
            candidate
            for candidate in workflow_candidates
            if candidate.checkpoint_id.endswith("_MAIN")
        )
        shallow = next(
            (
                candidate
                for candidate in workflow_candidates
                if candidate.checkpoint_id.endswith("_SHALLOW")
            ),
            None,
        )
        if shallow is None:
            assert flop_saved[main.checkpoint_id] == float(main.token_pos)
        else:
            assert flop_saved[shallow.checkpoint_id] == float(
                shallow.token_pos
            )
            assert flop_saved[main.checkpoint_id] == float(
                main.token_pos - shallow.token_pos
            )


def test_flop_proxy_uses_lineage_ancestry_and_rejects_zero_span() -> None:
    candidates = (
        CheckpointCandidate("P", "W", ("P",), 10, 100),
        CheckpointCandidate("A", "W", ("P", "A"), 20, 100),
        CheckpointCandidate("B", "W", ("P", "B"), 30, 100),
    )

    assert build_marconi_flop_saved(candidates) == {
        "A": 10.0,
        "B": 20.0,
        "P": 10.0,
    }

    zero = CheckpointCandidate("ZERO", "W", ("P",), 0, 100)
    with pytest.raises(ValueError, match="ZERO"):
        build_marconi_flop_saved((zero,))


def test_metadata_does_not_access_flowstate_or_oracle_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = build_controlled_scenario()

    def forbidden(*args, **kwargs):
        """在 metadata builder 访问禁止信息时立即使测试失败。"""
        raise AssertionError("metadata 不得读取 allocation 或 recovery 结果")

    for name in (
        "RecoveryCostModel",
        "recovery_gap",
        "executable_frontier",
        "GlobalOptimizer",
        "select_oracle",
    ):
        monkeypatch.setattr(metadata_module, name, forbidden, raising=False)

    steps = build_kvflow_steps(scenario.continuations)
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    flop_saved = build_marconi_flop_saved(scenario.candidates)

    assert set(steps.values()) == {1}
    assert recency
    assert flop_saved


def test_sota_selections_are_independent_of_recovery_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = build_scalable_scenario(8, 4)
    steps = build_kvflow_steps(scenario.continuations)
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    flop_saved = build_marconi_flop_saved(scenario.candidates)
    kvflow_before = KVFlowStylePolicy().select(
        scenario.continuations,
        scenario.candidates,
        4,
        steps,
        recency,
    ).selected_checkpoint_ids
    marconi_before = MarconiStylePolicy().select(
        scenario.candidates,
        4,
        recency,
        flop_saved,
        CONTROLLED_MARCONI_ALPHA,
    ).selected_checkpoint_ids

    def changed_estimate(self, replay_tokens):
        """模拟完全不同的 Phi，但不参与两个 SOTA-style 选择。"""
        return float(replay_tokens * replay_tokens)

    monkeypatch.setattr(
        "flowstate.recovery_model.RecoveryCostModel.estimate",
        changed_estimate,
    )

    assert KVFlowStylePolicy().select(
        scenario.continuations,
        scenario.candidates,
        4,
        steps,
        recency,
    ).selected_checkpoint_ids == kvflow_before
    assert MarconiStylePolicy().select(
        scenario.candidates,
        4,
        recency,
        flop_saved,
        CONTROLLED_MARCONI_ALPHA,
    ).selected_checkpoint_ids == marconi_before


def test_controlled_v1_contains_eight_policies() -> None:
    result = build_budget_sweep()

    assert len(result.rows) == 40
    assert set(BUDGET_SWEEP_POLICY_NAMES) == {
        "FlowState",
        "Global-LRU",
        "Equal-Share",
        "Recovery-Only",
        "Workflow-Only",
        "KVFlow-style",
        "Marconi-style",
        "Oracle",
    }


def test_equal_priority_kvflow_matches_global_lru_in_controlled_workloads() -> None:
    controlled_rows = {
        (row.budget_checkpoints, row.policy_name): row
        for row in build_budget_sweep().rows
    }
    for budget in range(1, 6):
        assert controlled_rows[
            (budget, "KVFlow-style")
        ].selected_checkpoint_ids == controlled_rows[
            (budget, "Global-LRU")
        ].selected_checkpoint_ids

    scalable_rows = {
        (row.workflow_count, row.budget_checkpoints, row.policy_name): row
        for row in run_offline_analysis().rows
    }
    for workflow_count in (8, 16):
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]:
            assert scalable_rows[
                (workflow_count, budget, "KVFlow-style")
            ].selected_checkpoint_ids == scalable_rows[
                (workflow_count, budget, "Global-LRU")
            ].selected_checkpoint_ids


def test_scalable_analysis_contains_eight_policies_and_preserves_step8a() -> None:
    result = run_offline_analysis()

    assert len(result.rows) == 64
    assert {row.policy_name for row in result.rows} == set(POLICY_NAMES)
    assert build_step8a_regression_digest(result.rows) == (
        STEP8A_REGRESSION_DIGEST
    )


def test_scalable_analysis_reuses_frozen_oracle_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        """在离线接入意外重跑 Oracle 时立即使测试失败。"""
        raise AssertionError("固定 Oracle 不得重新搜索")

    monkeypatch.setattr(analysis_module, "select_oracle", forbidden)

    rows = analyze_workload(16, budget_options=(4,))

    assert len(rows) == 8
    assert any(row.policy_name == "Oracle" for row in rows)


def test_oracle_is_no_worse_than_new_sota_baselines() -> None:
    result = run_offline_analysis()
    rows = {
        (row.workflow_count, row.budget_checkpoints, row.policy_name): row
        for row in result.rows
    }

    for workflow_count in (8, 16):
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]:
            oracle_cost = rows[
                (workflow_count, budget, "Oracle")
            ].estimated_recovery_cost_ms
            for policy_name in ("KVFlow-style", "Marconi-style"):
                assert oracle_cost <= rows[
                    (workflow_count, budget, policy_name)
                ].estimated_recovery_cost_ms + 1e-9
