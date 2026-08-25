from __future__ import annotations

from collections import Counter

import pytest

from evaluation.sota_latency_benchmark import (
    MAX_ESTIMATED_GPU_HOURS,
    MEASURED_REPETITIONS,
    POLICY_ORDER_SEED,
    REPRESENTATIVE_POINTS,
    REQUIRED_SAFETY_FLAGS,
    WARMUP_REPETITIONS,
    LatencyBenchmarkCase,
    LatencyCorrectnessError,
    aggregate_latency_records,
    balanced_policy_order,
    build_benchmark_cases,
    build_dry_run_report,
    build_latency_equivalence_classes,
    policy_position_distribution,
    validate_latency_measurement,
    weighted_mean,
    weighted_quantile,
)
from evaluation.sota_runtime_correctness import (
    GPU_POLICY_NAMES,
    build_e2e_equivalence_cases,
    build_representative_cases,
)


def test_frozen_equivalence_classes_and_multiplicity_are_complete() -> None:
    logical_cases = build_representative_cases()
    classes = build_latency_equivalence_classes(logical_cases)
    frozen_representatives = build_e2e_equivalence_cases(logical_cases)

    assert len(classes) == len(frozen_representatives) == 69
    assert sum(item.class_multiplicity for item in classes) == 800
    assert all(item.class_multiplicity > 0 for item in classes)
    assert tuple(
        item.representative_continuation_id for item in classes
    ) == tuple(item.continuation_id for item in frozen_representatives)


def test_each_class_multiplicity_matches_its_frozen_members() -> None:
    logical_cases = build_representative_cases()
    classes = build_latency_equivalence_classes(logical_cases)

    for item in classes:
        members = tuple(
            case
            for case in logical_cases
            if case.scenario_name == item.scenario_name
            and case.budget_checkpoints == item.budget_checkpoints
            and case.policy_name == item.policy_name
            and (
                case.planning_target,
                case.planning_executable_frontier,
                case.planning_gap_tokens,
            )
            == item.equivalence_key
        )
        assert len(members) == item.class_multiplicity
        assert item.representative_continuation_id == min(
            member.continuation_id for member in members
        )


def test_benchmark_schedule_has_frozen_case_counts() -> None:
    schedule = build_benchmark_cases()

    assert sum(item.is_warmup for item in schedule) == 138
    assert sum(not item.is_warmup for item in schedule) == 690
    assert len(schedule) == 828
    assert len({item.case_id for item in schedule}) == len(schedule)
    assert {
        (item.scenario_name, item.budget_checkpoints)
        for item in schedule
    } == set(REPRESENTATIVE_POINTS)
    assert Counter(
        item.policy_name
        for item in schedule
        if not item.is_warmup
    ) == {
        policy_name: sum(
            item.policy_name == policy_name
            for item in build_latency_equivalence_classes()
        )
        * MEASURED_REPETITIONS
        for policy_name in GPU_POLICY_NAMES
    }


def test_policy_order_is_deterministic_and_cyclic() -> None:
    first = tuple(
        balanced_policy_order(point, repetition)
        for point in range(len(REPRESENTATIVE_POINTS))
        for repetition in range(MEASURED_REPETITIONS)
    )
    second = tuple(
        balanced_policy_order(
            point,
            repetition,
            seed=POLICY_ORDER_SEED,
        )
        for point in range(len(REPRESENTATIVE_POINTS))
        for repetition in range(MEASURED_REPETITIONS)
    )

    assert first == second
    assert all(set(order) == set(GPU_POLICY_NAMES) for order in first)
    assert balanced_policy_order(0, 1) == (
        balanced_policy_order(0, 0)[1:]
        + balanced_policy_order(0, 0)[:1]
    )


def test_policy_block_positions_are_balanced() -> None:
    schedule = build_benchmark_cases()
    warmup = policy_position_distribution(schedule, warmup=True)
    measured = policy_position_distribution(schedule, warmup=False)

    assert warmup == {
        policy_name: (2, 2, 2, 2)
        for policy_name in GPU_POLICY_NAMES
    }
    assert measured == {
        policy_name: (10, 10, 10, 10)
        for policy_name in GPU_POLICY_NAMES
    }


def test_weighted_statistics_use_multiplicity() -> None:
    values = (10.0, 20.0)
    weights = (1.0, 3.0)

    assert weighted_mean(values, weights) == pytest.approx(17.5)
    assert weighted_quantile(values, weights, 0.5) == 20.0
    assert weighted_quantile(values, weights, 0.95) == 20.0


def test_warmup_and_failed_cases_do_not_enter_statistics() -> None:
    records = (
        _record("Global-LRU", "lru-a", 10.0, 2),
        _record(
            "Global-LRU",
            "lru-warmup",
            1_000.0,
            100,
            warmup=True,
        ),
        _record("FlowState", "flow-a", 5.0, 2),
        _record(
            "FlowState",
            "flow-failed",
            0.0,
            100,
            correctness_pass=False,
        ),
    )

    rows = aggregate_latency_records(records)
    by_policy = {row["policy"]: row for row in rows}

    assert by_policy["Global-LRU"]["measured_case_count"] == 1
    assert by_policy["FlowState"]["measured_case_count"] == 1
    assert by_policy["Global-LRU"]["ttft_ms"][
        "weighted_mean"
    ] == 10.0
    assert by_policy["FlowState"]["ttft_ms"][
        "weighted_mean"
    ] == 5.0
    assert by_policy["FlowState"]["relative_to_global_lru"][
        "ttft_ms"
    ]["weighted_mean"] == pytest.approx(0.5)


def test_runtime_gap_mismatch_triggers_correctness_failure() -> None:
    case = _first_measured_case()
    item = case.equivalence_class

    with pytest.raises(LatencyCorrectnessError, match="正确性门禁"):
        validate_latency_measurement(
            case,
            runtime_metrics={
                "physical_fa_hit": item.planning_target,
                "executable_prefix": (
                    item.planning_executable_frontier
                ),
                "replay_gap": item.planning_gap_tokens + 1,
            },
            safety={flag: True for flag in REQUIRED_SAFETY_FLAGS},
            ttft_ms=10.0,
            request_latency_ms=20.0,
            snapshot_build_ms=100.0,
            reconcile_ms=5.0,
        )


def test_valid_measurement_keeps_build_and_reconcile_separate() -> None:
    case = _first_measured_case()
    item = case.equivalence_class
    record = validate_latency_measurement(
        case,
        runtime_metrics={
            "physical_fa_hit": item.planning_target,
            "executable_prefix": item.planning_executable_frontier,
            "replay_gap": item.planning_gap_tokens,
        },
        safety={flag: True for flag in REQUIRED_SAFETY_FLAGS},
        ttft_ms=10.0,
        request_latency_ms=20.0,
        snapshot_build_ms=321.0,
        reconcile_ms=7.0,
    )

    assert record["correctness_pass"] is True
    assert record["ttft_ms"] == 10.0
    assert record["request_latency_ms"] == 20.0
    assert record["snapshot_build_ms"] == 321.0
    assert record["reconcile_ms"] == 7.0
    assert record["class_multiplicity"] == item.class_multiplicity


def test_safety_failure_prevents_a_valid_measurement() -> None:
    case = _first_measured_case()
    item = case.equivalence_class
    safety = {flag: True for flag in REQUIRED_SAFETY_FLAGS}
    safety["mamba_safety"] = False

    with pytest.raises(LatencyCorrectnessError, match="安全条件失败"):
        validate_latency_measurement(
            case,
            runtime_metrics={
                "physical_fa_hit": item.planning_target,
                "executable_prefix": item.planning_executable_frontier,
                "replay_gap": item.planning_gap_tokens,
            },
            safety=safety,
            ttft_ms=10.0,
            request_latency_ms=20.0,
            snapshot_build_ms=100.0,
            reconcile_ms=5.0,
        )


def test_dry_run_freezes_scale_balance_and_runtime_gate() -> None:
    report = build_dry_run_report()

    assert report["equivalence_class_count"] == 69
    assert len(report["equivalence_classes"]) == 69
    assert report["multiplicity_sum"] == 800
    assert report["warmup_repetitions"] == WARMUP_REPETITIONS == 2
    assert report["measured_repetitions"] == MEASURED_REPETITIONS == 10
    assert report["warmup_cases"] == 138
    assert report["measured_cases"] == 690
    assert report["policy_order_seed"] == POLICY_ORDER_SEED
    assert report["estimated_gpu_runtime"][
        "within_six_hour_limit"
    ] is True
    assert report["estimated_gpu_runtime"][
        "estimated_hours"
    ] <= MAX_ESTIMATED_GPU_HOURS
    assert all(
        row["oracle_objective_match"] is True
        for row in report["oracle"].values()
    )


def _first_measured_case() -> LatencyBenchmarkCase:
    """返回一个确定性的 measured case。"""
    return next(
        item for item in build_benchmark_cases() if not item.is_warmup
    )


def _record(
    policy: str,
    class_id: str,
    value: float,
    multiplicity: int,
    *,
    warmup: bool = False,
    correctness_pass: bool = True,
) -> dict[str, object]:
    """构造统计测试使用的最小原始样本。"""
    return {
        "scenario": "scenario",
        "K": 4,
        "policy": policy,
        "equivalence_class": class_id,
        "class_multiplicity": multiplicity,
        "ttft_ms": value,
        "request_latency_ms": value * 2.0,
        "is_warmup": warmup,
        "correctness_pass": correctness_pass,
    }
