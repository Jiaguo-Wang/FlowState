from __future__ import annotations

from collections import Counter

import pytest

from evaluation.scalable_multiworkflow_v2.scenario import (
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_runtime_correctness import (
    EXPECTED_SCALABLE_CASES,
    EXPECTED_SIGNAL_CASES,
    EXPECTED_TOTAL_CASES,
    GPU_POLICY_NAMES,
    EXPECTED_E2E_EQUIVALENCE_CASES,
    SCALABLE_SCENARIO_NAME,
    SIGNAL_SCENARIO_NAME,
    EvaluationPolicyOptimizerAdapter,
    RuntimeCorrectnessCase,
    build_e2e_equivalence_cases,
    build_equivalence_class_report,
    build_flowstate_oracle_report,
    build_representative_cases,
    build_runtime_scenario_view,
    build_snapshot_audit_plans,
    validate_exact_runtime_observation,
)
from evaluation.sota_signal_stress_v1.scenario import (
    build_scenario as build_signal_scenario,
)
from flowstate.recovery_model import RecoveryCostModel


def test_representative_case_plan_is_complete_and_unique() -> None:
    cases = build_representative_cases()

    assert len(cases) == EXPECTED_TOTAL_CASES
    assert Counter(case.scenario_name for case in cases) == {
        SCALABLE_SCENARIO_NAME: EXPECTED_SCALABLE_CASES,
        SIGNAL_SCENARIO_NAME: EXPECTED_SIGNAL_CASES,
    }
    assert Counter(
        (
            case.scenario_name,
            case.budget_checkpoints,
            case.policy_name,
        )
        for case in cases
    ) == {
        (SCALABLE_SCENARIO_NAME, budget, policy_name): 60
        for budget in (4, 12)
        for policy_name in GPU_POLICY_NAMES
    } | {
        (SIGNAL_SCENARIO_NAME, budget, policy_name): 40
        for budget in (4, 8)
        for policy_name in GPU_POLICY_NAMES
    }
    assert len(
        {
            (
                case.scenario_name,
                case.budget_checkpoints,
                case.policy_name,
                case.continuation_id,
            )
            for case in cases
        }
    ) == EXPECTED_TOTAL_CASES


def test_case_plan_uses_core_frontier_and_gap_identity() -> None:
    cases = build_representative_cases()

    assert all(
        case.planning_executable_frontier
        + case.planning_gap_tokens
        == case.planning_target
        for case in cases
    )
    assert all(case.planning_gap_tokens >= 0 for case in cases)


def test_snapshot_audit_plan_covers_all_logical_cases() -> None:
    plans = build_snapshot_audit_plans()

    assert len(plans) == 16
    assert sum(len(plan.logical_cases) for plan in plans) == 800
    assert all(
        all(
            case.selected_checkpoint_ids == plan.selected_checkpoint_ids
            for case in plan.logical_cases
        )
        for plan in plans
    )


def test_equivalence_class_representatives_are_deterministic() -> None:
    cases = build_representative_cases()
    representatives = build_e2e_equivalence_cases(cases)

    assert len(representatives) == 69
    for representative in representatives:
        peers = tuple(
            case
            for case in cases
            if case.scenario_name == representative.scenario_name
            and case.budget_checkpoints
            == representative.budget_checkpoints
            and case.policy_name == representative.policy_name
            and (
                case.planning_target,
                case.planning_executable_frontier,
                case.planning_gap_tokens,
            )
            == (
                representative.planning_target,
                representative.planning_executable_frontier,
                representative.planning_gap_tokens,
            )
        )
        assert representative.continuation_id == min(
            case.continuation_id for case in peers
        )


def test_dry_run_requires_all_frozen_equivalence_classes() -> None:
    report = build_equivalence_class_report()

    assert report["snapshot_runtime_runs"] == 16
    assert report["logical_cases_checked"] == 800
    assert report["total_e2e_cases"] == 69
    assert (
        report["expected_e2e_cases"]
        == EXPECTED_E2E_EQUIVALENCE_CASES
        == 69
    )
    assert report["gpu_allowed_by_case_count"] is True


def test_oracle_report_separates_objective_and_selection() -> None:
    report = build_flowstate_oracle_report()

    assert len(report) == 4
    assert all(
        set(row) == {
            "oracle_objective_match",
            "oracle_selection_match",
            "multiple_optimal_selections",
        }
        for row in report.values()
    )
    assert all(
        row["multiple_optimal_selections"]
        == (
            row["oracle_objective_match"]
            and not row["oracle_selection_match"]
        )
        for row in report.values()
    )


@pytest.mark.parametrize(
    "scenario",
    (
        build_scalable_scenario(16, 4),
        build_signal_scenario(4),
    ),
)
def test_runtime_scenario_view_preserves_frozen_inputs(scenario) -> None:
    view = build_runtime_scenario_view(scenario)

    assert view.continuations is scenario.continuations
    assert view.candidates is scenario.candidates
    assert view.budget_bytes == scenario.budget_bytes
    assert tuple(item.workflow_id for item in view.metadata.workflows) == tuple(
        item.workflow_id for item in scenario.metadata.workflows
    )
    assert all(item.anchor_pos > 0 for item in view.metadata.workflows)
    assert all(item.pending_fanout > 0 for item in view.metadata.workflows)


@pytest.mark.parametrize(
    "scenario",
    (
        build_scalable_scenario(16, 4),
        build_signal_scenario(4),
    ),
)
@pytest.mark.parametrize(
    "policy_name",
    GPU_POLICY_NAMES[:-1],
)
def test_policy_optimizer_adapter_matches_frozen_case_plan(
    scenario,
    policy_name: str,
) -> None:
    model = RecoveryCostModel()
    cases = build_representative_cases(model)
    expected = next(
        case.selected_checkpoint_ids
        for case in cases
        if case.scenario_name
        == (
            SCALABLE_SCENARIO_NAME
            if hasattr(scenario.metadata, "workflow_count")
            else SIGNAL_SCENARIO_NAME
        )
        and case.budget_checkpoints
        == scenario.metadata.budget_checkpoints
        and case.policy_name == policy_name
    )
    result = EvaluationPolicyOptimizerAdapter(
        policy_name,
        scenario,
        model,
    ).select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )

    assert tuple(
        candidate.checkpoint_id for candidate in result.selected
    ) == expected


def test_exact_runtime_observation_requires_token_exact_agreement() -> None:
    case = RuntimeCorrectnessCase(
        scenario_name=SCALABLE_SCENARIO_NAME,
        budget_checkpoints=4,
        policy_name="FlowState",
        continuation_id="W-B1",
        workflow_id="W",
        selected_checkpoint_ids=("W_MAIN",),
        planning_target=32_768,
        planning_executable_frontier=16_384,
        planning_gap_tokens=16_384,
    )

    observation = validate_exact_runtime_observation(
        case,
        {
            "physical_fa_hit": 32_768,
            "executable_prefix": 16_384,
            "replay_gap": 16_384,
        },
    )
    assert observation["runtime_gap_tokens"] == 16_384

    with pytest.raises(RuntimeError, match="不一致"):
        validate_exact_runtime_observation(
            case,
            {
                "physical_fa_hit": 32_769,
                "executable_prefix": 16_384,
                "replay_gap": 16_385,
            },
        )
