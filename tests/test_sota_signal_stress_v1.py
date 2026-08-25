from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from evaluation.sota_metadata import (
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import KVFlowStylePolicy
from evaluation.sota_signal_stress_v1.offline_analysis import (
    POLICY_NAMES,
    SignalAnalysisResult,
    load_offline_artifact,
    run_offline_analysis,
    validate_factorial_scenario,
    write_offline_artifacts,
)
from evaluation.sota_signal_stress_v1.scenario import (
    ANCHOR_DEPTHS,
    BUDGET_CHECKPOINTS,
    FANOUTS,
    RECENCY_CLASSES,
    STEPS_TO_EXECUTION,
    SignalScenario,
    build_scenario,
)
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import is_compatible


@pytest.fixture(scope="module")
def analysis_result() -> SignalAnalysisResult:
    """只执行一次新场景的精确离线分析。"""
    return run_offline_analysis()


def test_scenario_sizes_and_single_main_candidate_rule() -> None:
    scenario = build_scenario()

    assert len(scenario.metadata.workflows) == 16
    assert len(scenario.candidates) == 16
    assert len(scenario.continuations) == 40
    assert all(
        candidate.checkpoint_id.endswith("_MAIN")
        for candidate in scenario.candidates
    )
    assert all(
        candidate.token_pos
        == next(
            workflow.anchor_depth
            for workflow in scenario.metadata.workflows
            if workflow.workflow_id == candidate.workflow_id
        )
        for candidate in scenario.candidates
    )


def test_factorial_is_complete_unique_and_independent() -> None:
    scenario = build_scenario()
    validate_factorial_scenario(scenario)
    factor_tuples = tuple(
        workflow.factor_tuple for workflow in scenario.metadata.workflows
    )

    assert len(set(factor_tuples)) == 16
    assert set(factor_tuples) == set(
        product(
            ANCHOR_DEPTHS,
            FANOUTS,
            STEPS_TO_EXECUTION,
            RECENCY_CLASSES,
        )
    )
    for factor_index, levels in enumerate(
        (
            ANCHOR_DEPTHS,
            FANOUTS,
            STEPS_TO_EXECUTION,
            RECENCY_CLASSES,
        )
    ):
        for level in levels:
            remaining = {
                factors[:factor_index] + factors[factor_index + 1 :]
                for factors in factor_tuples
                if factors[factor_index] == level
            }
            other_levels = (
                ANCHOR_DEPTHS,
                FANOUTS,
                STEPS_TO_EXECUTION,
                RECENCY_CLASSES,
            )
            expected_remaining = set(
                product(
                    *(
                        values
                        for index, values in enumerate(other_levels)
                        if index != factor_index
                    )
                )
            )
            assert remaining == expected_remaining


def test_continuations_inherit_steps_and_only_match_own_workflow() -> None:
    scenario = build_scenario()
    workflows_by_id = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }

    for continuation in scenario.continuations:
        workflow = workflows_by_id[continuation.workflow_id]
        assert continuation.planning_target == workflow.anchor_depth
        assert scenario.metadata.steps_to_execution_by_continuation[
            continuation.continuation_id
        ] == workflow.steps_to_execution

    for candidate in scenario.candidates:
        compatible = tuple(
            continuation
            for continuation in scenario.continuations
            if is_compatible(candidate, continuation)
        )
        assert len(compatible) == workflows_by_id[
            candidate.workflow_id
        ].fanout
        assert {
            continuation.workflow_id for continuation in compatible
        } == {candidate.workflow_id}


def test_recency_history_is_mechanical_and_class_separated() -> None:
    scenario = build_scenario()
    workflows_by_id = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }
    recency_by_id = {
        item.checkpoint_id: item.last_access_order
        for item in scenario.metadata.checkpoint_recency
    }
    old_ids = tuple(
        sorted(
            candidate.checkpoint_id
            for candidate in scenario.candidates
            if workflows_by_id[candidate.workflow_id].recency_class == "old"
        )
    )
    recent_ids = tuple(
        sorted(
            candidate.checkpoint_id
            for candidate in scenario.candidates
            if workflows_by_id[candidate.workflow_id].recency_class
            == "recent"
        )
    )

    assert tuple(recency_by_id[checkpoint_id] for checkpoint_id in old_ids) == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )
    assert tuple(
        recency_by_id[checkpoint_id] for checkpoint_id in recent_ids
    ) == (9, 10, 11, 12, 13, 14, 15, 16)


def test_marconi_flop_proxy_equals_anchor_without_parent_candidate() -> None:
    scenario = build_scenario()
    flop_saved = build_marconi_flop_saved(scenario.candidates)

    assert {
        flop_saved[candidate.checkpoint_id]
        for candidate in scenario.candidates
        if candidate.token_pos == 8_192
    } == {8_192.0}
    assert {
        flop_saved[candidate.checkpoint_id]
        for candidate in scenario.candidates
        if candidate.token_pos == 32_768
    } == {32_768.0}
    assert all(
        candidate.memory_bytes == scenario.metadata.checkpoint_size_bytes
        for candidate in scenario.candidates
    )
    assert scenario.metadata.marconi_alpha == 1.0


def test_kvflow_smaller_steps_wins_even_when_older() -> None:
    scenario = build_scenario()
    workflows = tuple(
        workflow
        for workflow in scenario.metadata.workflows
        if workflow.anchor_depth == 8_192
        and workflow.fanout == 1
        and workflow.recency_class == "old"
    )
    assert {workflow.steps_to_execution for workflow in workflows} == {1, 3}
    workflow_ids = {workflow.workflow_id for workflow in workflows}
    candidates = tuple(
        candidate
        for candidate in scenario.candidates
        if candidate.workflow_id in workflow_ids
    )
    continuations = tuple(
        continuation
        for continuation in scenario.continuations
        if continuation.workflow_id in workflow_ids
    )
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )

    selected = KVFlowStylePolicy().select(
        continuations,
        candidates,
        1,
        scenario.metadata.steps_to_execution_by_continuation,
        recency,
    ).selected_checkpoint_ids
    selected_workflow = next(
        workflow
        for workflow in workflows
        if f"{workflow.workflow_id}_MAIN" == selected[0]
    )

    assert selected_workflow.steps_to_execution == 1


def test_kvflow_equal_steps_uses_recent_checkpoint() -> None:
    scenario = build_scenario()
    workflows = tuple(
        workflow
        for workflow in scenario.metadata.workflows
        if workflow.anchor_depth == 8_192
        and workflow.fanout == 1
        and workflow.steps_to_execution == 1
    )
    assert {workflow.recency_class for workflow in workflows} == {
        "old",
        "recent",
    }
    workflow_ids = {workflow.workflow_id for workflow in workflows}
    candidates = tuple(
        candidate
        for candidate in scenario.candidates
        if candidate.workflow_id in workflow_ids
    )
    continuations = tuple(
        continuation
        for continuation in scenario.continuations
        if continuation.workflow_id in workflow_ids
    )
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )

    selected = KVFlowStylePolicy().select(
        continuations,
        candidates,
        1,
        scenario.metadata.steps_to_execution_by_continuation,
        recency,
    ).selected_checkpoint_ids
    selected_workflow = next(
        workflow
        for workflow in workflows
        if f"{workflow.workflow_id}_MAIN" == selected[0]
    )

    assert selected_workflow.recency_class == "recent"


def test_marconi_metadata_exposes_both_native_signals() -> None:
    scenario = build_scenario()
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    flop_saved = build_marconi_flop_saved(scenario.candidates)
    workflows_by_id = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }

    for candidate in scenario.candidates:
        workflow = workflows_by_id[candidate.workflow_id]
        counterpart = next(
            other
            for other in scenario.candidates
            if other.workflow_id
            == next(
                item.workflow_id
                for item in scenario.metadata.workflows
                if item.anchor_depth == workflow.anchor_depth
                and item.fanout == workflow.fanout
                and item.steps_to_execution == workflow.steps_to_execution
                and item.recency_class
                != workflow.recency_class
            )
        )
        if workflow.recency_class == "recent":
            assert recency[candidate.checkpoint_id] > recency[
                counterpart.checkpoint_id
            ]
    assert min(
        flop_saved[candidate.checkpoint_id] / candidate.memory_bytes
        for candidate in scenario.candidates
        if candidate.token_pos == 32_768
    ) > max(
        flop_saved[candidate.checkpoint_id] / candidate.memory_bytes
        for candidate in scenario.candidates
        if candidate.token_pos == 8_192
    )


def test_flowstate_selection_does_not_read_steps_or_recency() -> None:
    scenario = build_scenario(8)
    model = RecoveryCostModel()
    original = GlobalOptimizer(model).select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )
    changed_recency = tuple(
        replace(
            item,
            last_access_order=100 - item.last_access_order,
        )
        for item in reversed(scenario.metadata.checkpoint_recency)
    )
    changed_steps = {
        continuation.continuation_id: 99
        for continuation in scenario.continuations
    }
    altered = replace(
        scenario,
        metadata=replace(
            scenario.metadata,
            checkpoint_recency=changed_recency,
            steps_to_execution_by_continuation=changed_steps,
        ),
    )
    altered_result = GlobalOptimizer(model).select(
        altered.continuations,
        altered.candidates,
        altered.budget_bytes,
    )

    assert tuple(
        candidate.checkpoint_id for candidate in original.selected
    ) == tuple(
        candidate.checkpoint_id for candidate in altered_result.selected
    )


def test_analysis_matrix_budget_and_factor_distribution(
    analysis_result: SignalAnalysisResult,
) -> None:
    assert len(analysis_result.rows) == 32
    assert {row.policy_name for row in analysis_result.rows} == set(
        POLICY_NAMES
    )
    assert {row.budget_checkpoints for row in analysis_result.rows} == set(
        BUDGET_CHECKPOINTS
    )
    for row in analysis_result.rows:
        assert row.used_budget_checkpoints <= row.budget_checkpoints
        assert row.used_budget_bytes == (
            row.used_budget_checkpoints
            * build_scenario().metadata.checkpoint_size_bytes
        )
        assert len(row.selected_factor_tuples) == row.used_budget_checkpoints
        distribution = row.factor_distribution
        assert distribution.anchor_8192 + distribution.anchor_32768 == (
            row.used_budget_checkpoints
        )
        assert distribution.fanout_1 + distribution.fanout_4 == (
            row.used_budget_checkpoints
        )
        assert distribution.steps_1 + distribution.steps_3 == (
            row.used_budget_checkpoints
        )
        assert distribution.recency_old + distribution.recency_recent == (
            row.used_budget_checkpoints
        )


def test_policy_native_signal_preferences_at_k4(
    analysis_result: SignalAnalysisResult,
) -> None:
    rows = {
        (row.budget_checkpoints, row.policy_name): row
        for row in analysis_result.rows
    }

    assert rows[(4, "Global-LRU")].factor_distribution.recency_recent == 4
    assert rows[(4, "KVFlow-style")].factor_distribution.steps_1 == 4
    assert rows[(4, "Workflow-Only")].factor_distribution.fanout_4 == 4
    assert rows[(4, "Recovery-Only")].factor_distribution.anchor_32768 == 4


def test_oracle_is_no_worse_and_full_budget_is_complete(
    analysis_result: SignalAnalysisResult,
) -> None:
    rows = {
        (row.budget_checkpoints, row.policy_name): row
        for row in analysis_result.rows
    }

    for budget in BUDGET_CHECKPOINTS:
        oracle_cost = rows[
            (budget, "Oracle")
        ].estimated_recovery_cost_ms
        assert all(
            oracle_cost
            <= rows[(budget, policy_name)].estimated_recovery_cost_ms + 1e-9
            for policy_name in POLICY_NAMES
        )
    assert all(
        rows[(16, policy_name)].total_recovery_gap_tokens == 0
        and rows[(16, policy_name)].planning_executable_prefix_ratio == 1.0
        for policy_name in POLICY_NAMES
    )


def test_analysis_is_deterministic(
    analysis_result: SignalAnalysisResult,
) -> None:
    assert run_offline_analysis() == analysis_result


def test_artifact_round_trip(
    tmp_path,
    analysis_result: SignalAnalysisResult,
) -> None:
    _, json_path = write_offline_artifacts(analysis_result, tmp_path)

    assert load_offline_artifact(json_path) == analysis_result
