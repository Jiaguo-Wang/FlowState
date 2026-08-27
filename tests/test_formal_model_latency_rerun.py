from __future__ import annotations

import json

import pytest

from evaluation.formal_model_latency_rerun import (
    BASELINE_POLICIES,
    EXPECTED_EQUIVALENCE_CLASSES,
    EXPECTED_MEASURED_CASES,
    EXPECTED_MULTIPLICITY_PER_REPETITION,
    EXPECTED_WARMUP_CASES,
    FORMAL_SELECTION_AUDIT,
    HISTORICAL_ARTIFACT,
    TARGET_POINTS,
    build_selection_manifest,
    build_target_equivalence_classes,
    build_target_schedule,
    build_weighted_summary,
    historical_source_hashes,
    protected_source_hashes,
)
from evaluation.sota_latency_benchmark import (
    POLICY_ORDER_SEED,
    REPRESENTATIVE_POINTS,
    balanced_policy_order,
)
from evaluation.sota_runtime_correctness import GPU_POLICY_NAMES
from flowstate.recovery_model import FORMAL_RECOVERY_MODEL_METADATA


def test_rerun_only_contains_two_preregistered_points() -> None:
    classes = build_target_equivalence_classes()
    schedule = build_target_schedule(classes)

    assert len(classes) == EXPECTED_EQUIVALENCE_CLASSES == 33
    assert sum(item.class_multiplicity for item in classes) == (
        EXPECTED_MULTIPLICITY_PER_REPETITION
    )
    assert {
        (item.scenario_name, item.budget_checkpoints) for item in classes
    } == set(TARGET_POINTS)
    assert {
        (item.scenario_name, item.budget_checkpoints) for item in schedule
    } == set(TARGET_POINTS)
    assert sum(item.is_warmup for item in schedule) == (
        EXPECTED_WARMUP_CASES
    )
    assert sum(not item.is_warmup for item in schedule) == (
        EXPECTED_MEASURED_CASES
    )


def test_formal_flowstate_selections_match_step10d4_audit() -> None:
    audit = json.loads(FORMAL_SELECTION_AUDIT.read_text(encoding="utf-8"))
    expected = {
        (row["scenario"], int(row["budget_checkpoints"])): tuple(
            row["new_formal_selection"]
        )
        for row in audit["points"]
        if (row["scenario"], int(row["budget_checkpoints"]))
        in TARGET_POINTS
    }
    rows = {
        (row.scenario, row.budget_checkpoints): row
        for row in build_selection_manifest()
        if row.policy == "FlowState"
    }

    assert set(rows) == set(TARGET_POINTS)
    assert all(
        rows[point].selected_checkpoint_ids == expected[point]
        for point in TARGET_POINTS
    )
    assert "W16_A32768_F02_MAIN" in rows[TARGET_POINTS[0]].selected_checkpoint_ids
    assert "W16_A16384_F04_MAIN" not in rows[TARGET_POINTS[0]].selected_checkpoint_ids
    assert all(
        "A32768" in checkpoint_id
        for checkpoint_id in rows[TARGET_POINTS[1]].selected_checkpoint_ids
    )


def test_baseline_selections_are_frozen_and_unique() -> None:
    classes = build_target_equivalence_classes()
    manifest = build_selection_manifest(classes)

    for point in TARGET_POINTS:
        for policy in BASELINE_POLICIES:
            row = next(
                item
                for item in manifest
                if (item.scenario, item.budget_checkpoints) == point
                and item.policy == policy
            )
            class_selections = {
                item.selected_checkpoint_ids
                for item in classes
                if (item.scenario_name, item.budget_checkpoints) == point
                and item.policy_name == policy
            }
            assert class_selections == {row.selected_checkpoint_ids}
            assert row.selection_source == "Step 9B 历史制品"


def test_target_schedule_preserves_step9b_policy_order_indices() -> None:
    schedule = build_target_schedule()

    for point in TARGET_POINTS:
        original_index = REPRESENTATIVE_POINTS.index(point)
        for repetition in range(10):
            expected = balanced_policy_order(
                original_index,
                repetition,
                seed=POLICY_ORDER_SEED,
            )
            observed = tuple(
                policy
                for position, policy in sorted(
                    {
                        (item.execution_order_position, item.policy_name)
                        for item in schedule
                        if not item.is_warmup
                        and item.repetition == repetition
                        and (
                            item.scenario_name,
                            item.budget_checkpoints,
                        )
                        == point
                    }
                )
            )
            assert observed == expected


def test_weighted_aggregation_matches_step9b_frozen_summary() -> None:
    records = tuple(
        json.loads(line)
        for line in (
            HISTORICAL_ARTIFACT / "raw_samples.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if (
            json.loads(line)["scenario"],
            int(json.loads(line)["K"]),
        )
        in TARGET_POINTS
    )
    rows = build_weighted_summary(records, build_selection_manifest())
    observed = {
        (row["scenario"], int(row["K"]), row["policy"]): row
        for row in rows
    }
    historical = json.loads(
        (HISTORICAL_ARTIFACT / "summary.json").read_text(encoding="utf-8")
    )
    expected = {
        (row["scenario"], int(row["K"]), row["policy"]): row
        for row in historical["policy_summaries"]
        if (row["scenario"], int(row["K"])) in TARGET_POINTS
    }

    assert set(observed) == set(expected)
    for key in expected:
        for statistic in (
            "weighted_mean",
            "weighted_median",
            "weighted_p95",
        ):
            assert observed[key]["ttft_ms"][statistic] == pytest.approx(
                expected[key]["ttft_ms"][statistic]
            )
            assert observed[key]["request_latency_ms"][
                statistic
            ] == pytest.approx(
                expected[key]["request_latency_ms"][statistic]
            )


def test_plan_build_is_read_only_for_sources_and_protocol() -> None:
    protected_before = protected_source_hashes()
    historical_before = historical_source_hashes()

    build_selection_manifest()
    build_target_schedule()

    assert protected_source_hashes() == protected_before
    assert historical_source_hashes() == historical_before
    assert FORMAL_RECOVERY_MODEL_METADATA.name == (
        "position_aware_quadratic_v1"
    )
    assert tuple(GPU_POLICY_NAMES) == (
        "Global-LRU",
        "KVFlow-style",
        "Marconi-style",
        "FlowState",
    )
