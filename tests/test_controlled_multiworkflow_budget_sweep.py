from __future__ import annotations

from dataclasses import replace
from math import isclose

from evaluation.controlled_multiworkflow_v1.budget_sweep import (
    BUDGET_SWEEP_POLICY_NAMES,
    DEFAULT_BUDGET_CHECKPOINTS,
    build_budget_sweep,
    format_sanity_table,
)
from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.controlled_multiworkflow_v1.snapshot_cases import (
    POLICY_NAMES as SNAPSHOT_POLICY_NAMES,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.recovery_model import RecoveryCostModel


def test_budget_sweep_contains_complete_matrix() -> None:
    result = build_budget_sweep()

    assert {row.budget_checkpoints for row in result.rows} == set(
        DEFAULT_BUDGET_CHECKPOINTS
    )
    assert len(result.rows) == 40
    for k in DEFAULT_BUDGET_CHECKPOINTS:
        assert tuple(
            row.policy_name
            for row in result.rows
            if row.budget_checkpoints == k
        ) == BUDGET_SWEEP_POLICY_NAMES


def test_selections_obey_budget_uniqueness_and_residency() -> None:
    scenario = build_scenario()
    resident_ids = {
        candidate.checkpoint_id
        for candidate in scenario.candidates
        if candidate.recurrent_resident
    }

    for row in build_budget_sweep(scenario).rows:
        assert len(row.selected_checkpoint_ids) <= row.budget_checkpoints
        assert len(set(row.selected_checkpoint_ids)) == len(
            row.selected_checkpoint_ids
        )
        assert set(row.selected_checkpoint_ids) <= resident_ids


def test_sweep_metrics_reuse_core_frontier_gap_and_recovery_model() -> None:
    scenario = build_scenario()
    model = RecoveryCostModel()
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    total_target = sum(
        continuation.planning_target
        for continuation in scenario.continuations
    )

    for row in build_budget_sweep(scenario, model).rows:
        selected = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in row.selected_checkpoint_ids
        )
        gaps = tuple(
            recovery_gap(continuation, selected)
            for continuation in scenario.continuations
        )
        frontiers = tuple(
            executable_frontier(continuation, selected)
            for continuation in scenario.continuations
        )
        assert row.total_recovery_gap == sum(gaps)
        assert row.mean_recovery_gap_per_request == (
            sum(gaps) / len(gaps)
        )
        assert isclose(
            row.planning_executable_prefix_ratio,
            sum(frontiers) / total_target,
        )
        assert isclose(
            row.estimated_recovery_cost_ms,
            sum(model.estimate(gap) for gap in gaps),
        )


def test_flowstate_cost_and_gap_are_monotonic_non_increasing() -> None:
    flowstate_rows = tuple(
        row
        for row in build_budget_sweep().rows
        if row.policy_name == "FlowState"
    )

    assert all(
        current.total_recovery_gap >= following.total_recovery_gap
        for current, following in zip(flowstate_rows, flowstate_rows[1:])
    )
    assert all(
        current.estimated_recovery_cost_ms
        >= following.estimated_recovery_cost_ms
        for current, following in zip(flowstate_rows, flowstate_rows[1:])
    )


def test_k3_regression_and_k5_full_recovery() -> None:
    rows = {
        (row.budget_checkpoints, row.policy_name): row
        for row in build_budget_sweep().rows
    }

    assert rows[(3, "FlowState")].selected_checkpoint_ids == (
        "W1_PARENT",
        "W2_PARENT",
        "W4_PARENT",
    )
    assert {
        policy_name: rows[(3, policy_name)].total_recovery_gap
        for policy_name in SNAPSHOT_POLICY_NAMES
    } == {
        "FlowState": 8_192,
        "Global-LRU": 65_536,
        "Equal-Share": 12_288,
        "Recovery-Only": 20_480,
    }
    for policy_name in BUDGET_SWEEP_POLICY_NAMES:
        row = rows[(5, policy_name)]
        assert row.total_recovery_gap == 0
        assert row.estimated_recovery_cost_ms == 0.0
        assert row.planning_executable_prefix_ratio == 1.0


def test_existing_four_policy_budget_regression() -> None:
    rows = {
        (row.budget_checkpoints, row.policy_name): (
            row.selected_checkpoint_ids,
            row.total_recovery_gap,
        )
        for row in build_budget_sweep().rows
        if row.policy_name in SNAPSHOT_POLICY_NAMES
    }

    assert rows == {
        (1, "FlowState"): (("W1_PARENT",), 36_864),
        (1, "Global-LRU"): (("W4_PARENT",), 90_112),
        (1, "Equal-Share"): (("W1_PARENT",), 36_864),
        (1, "Recovery-Only"): (("W1_PARENT",), 36_864),
        (2, "FlowState"): (
            ("W1_PARENT", "W2_PARENT"),
            20_480,
        ),
        (2, "Global-LRU"): (
            ("W4_PARENT", "W3_PARENT"),
            81_920,
        ),
        (2, "Equal-Share"): (
            ("W1_PARENT", "W2_PARENT"),
            20_480,
        ),
        (2, "Recovery-Only"): (
            ("W1_PARENT", "W2_PARENT"),
            20_480,
        ),
        (3, "FlowState"): (
            ("W1_PARENT", "W2_PARENT", "W4_PARENT"),
            8_192,
        ),
        (3, "Global-LRU"): (
            ("W4_PARENT", "W3_PARENT", "W2_PARENT"),
            65_536,
        ),
        (3, "Equal-Share"): (
            ("W1_PARENT", "W2_PARENT", "W3_PARENT"),
            12_288,
        ),
        (3, "Recovery-Only"): (
            ("W1_PARENT", "W2_PARENT", "W1_SHALLOW"),
            20_480,
        ),
        (4, "FlowState"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W4_PARENT",
                "W3_PARENT",
            ),
            0,
        ),
        (4, "Global-LRU"): (
            (
                "W4_PARENT",
                "W3_PARENT",
                "W2_PARENT",
                "W1_PARENT",
            ),
            0,
        ),
        (4, "Equal-Share"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W3_PARENT",
                "W4_PARENT",
            ),
            0,
        ),
        (4, "Recovery-Only"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W1_SHALLOW",
                "W3_PARENT",
            ),
            12_288,
        ),
        (5, "FlowState"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W4_PARENT",
                "W3_PARENT",
            ),
            0,
        ),
        (5, "Global-LRU"): (
            (
                "W4_PARENT",
                "W3_PARENT",
                "W2_PARENT",
                "W1_PARENT",
                "W1_SHALLOW",
            ),
            0,
        ),
        (5, "Equal-Share"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W3_PARENT",
                "W4_PARENT",
                "W1_SHALLOW",
            ),
            0,
        ),
        (5, "Recovery-Only"): (
            (
                "W1_PARENT",
                "W2_PARENT",
                "W1_SHALLOW",
                "W3_PARENT",
                "W4_PARENT",
            ),
            0,
        ),
    }


def test_flowstate_baseline_comparisons_cover_every_budget() -> None:
    result = build_budget_sweep()

    assert len(result.comparisons) == 35
    assert {
        (comparison.budget_checkpoints, comparison.baseline_policy_name)
        for comparison in result.comparisons
    } == {
        (k, policy_name)
        for k in DEFAULT_BUDGET_CHECKPOINTS
        for policy_name in BUDGET_SWEEP_POLICY_NAMES[1:]
    }
    assert all(
        comparison.absolute_gap_reduction >= 0
        and comparison.estimated_recovery_cost_reduction_ms >= -1e-9
        for comparison in result.comparisons
    )


def test_budget_sweep_is_candidate_order_invariant() -> None:
    scenario = build_scenario()
    expected = build_budget_sweep(scenario)
    reordered = replace(
        scenario,
        candidates=tuple(reversed(scenario.candidates)),
    )

    assert build_budget_sweep(reordered) == expected


def test_sanity_table_uses_planning_epr_label() -> None:
    table = format_sanity_table(build_budget_sweep())

    assert "Planning EPR" in table
    assert table.count("\n") == 41
