from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isclose

from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.controlled_multiworkflow_v1.snapshot_cases import (
    POLICY_NAMES,
    build_planning_summaries,
    build_snapshot_cases,
)
from flowstate.executable_state import recovery_gap
from flowstate.recovery_model import RecoveryCostModel


_CONTINUATION_IDS = (
    "W1-A",
    "W1-B",
    "W2",
    "W3",
    "W4-A",
    "W4-B",
    "W4-C",
)


def test_snapshot_case_matrix_is_complete_and_unique() -> None:
    cases = build_snapshot_cases()

    assert len(cases) == 28
    assert Counter(case.policy_name for case in cases) == {
        policy_name: 7 for policy_name in POLICY_NAMES
    }
    assert Counter(
        (case.policy_name, case.continuation_id) for case in cases
    ) == {
        (policy_name, continuation_id): 1
        for policy_name in POLICY_NAMES
        for continuation_id in _CONTINUATION_IDS
    }


def test_expected_gaps_are_derived_from_core_recovery_gap() -> None:
    scenario = build_scenario()
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    continuations_by_id = {
        continuation.continuation_id: continuation
        for continuation in scenario.continuations
    }

    for case in build_snapshot_cases(scenario):
        selected = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in case.expected_selected_ids
        )
        continuation = continuations_by_id[case.scenario_continuation_id]
        assert case.expected_recovery_gap == recovery_gap(
            continuation,
            selected,
        )


def test_planning_totals_and_estimated_costs() -> None:
    model = RecoveryCostModel()
    scenario = build_scenario()
    summaries = {
        summary.policy_name: summary
        for summary in build_planning_summaries(
            scenario=scenario,
            recovery_cost_model=model,
        )
    }

    assert {
        policy_name: summary.total_recovery_gap
        for policy_name, summary in summaries.items()
    } == {
        "FlowState": 8_192,
        "Global-LRU": 65_536,
        "Equal-Share": 12_288,
        "Recovery-Only": 20_480,
    }
    assert {
        policy_name: dict(summary.recovery_gaps)
        for policy_name, summary in summaries.items()
    } == {
        "FlowState": {
            "W1-A": 0,
            "W1-B": 0,
            "W2": 0,
            "W3": 8_192,
            "W4-A": 0,
            "W4-B": 0,
            "W4-C": 0,
        },
        "Global-LRU": {
            "W1-A": 32_768,
            "W1-B": 32_768,
            "W2": 0,
            "W3": 0,
            "W4-A": 0,
            "W4-B": 0,
            "W4-C": 0,
        },
        "Equal-Share": {
            "W1-A": 0,
            "W1-B": 0,
            "W2": 0,
            "W3": 0,
            "W4-A": 4_096,
            "W4-B": 4_096,
            "W4-C": 4_096,
        },
        "Recovery-Only": {
            "W1-A": 0,
            "W1-B": 0,
            "W2": 0,
            "W3": 8_192,
            "W4-A": 4_096,
            "W4-B": 4_096,
            "W4-C": 4_096,
        },
    }
    for summary in summaries.values():
        expected_cost = sum(
            model.estimate(
                gap,
                continuation.planning_target,
            )
            for continuation, (_, gap) in zip(
                scenario.continuations,
                summary.recovery_gaps,
            )
        )
        assert isclose(
            summary.estimated_recovery_cost_ms,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_snapshot_planning_is_candidate_order_invariant() -> None:
    scenario = build_scenario()
    candidates = scenario.candidates
    candidate_orders = (
        candidates,
        tuple(reversed(candidates)),
        (
            candidates[3],
            candidates[0],
            candidates[4],
            candidates[1],
            candidates[2],
        ),
    )
    expected_summaries = build_planning_summaries(scenario)
    expected_cases = build_snapshot_cases(scenario)

    for order in candidate_orders[1:]:
        reordered = replace(scenario, candidates=order)
        assert build_planning_summaries(reordered) == expected_summaries
        assert build_snapshot_cases(reordered) == expected_cases

    reversed_continuations = replace(
        scenario,
        continuations=tuple(reversed(scenario.continuations)),
    )
    assert build_planning_summaries(
        reversed_continuations
    ) == expected_summaries
    assert build_snapshot_cases(reversed_continuations) == expected_cases
