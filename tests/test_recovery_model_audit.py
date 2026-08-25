from __future__ import annotations

import pytest

from evaluation.recovery_model_audit import (
    DEFAULT_ARTIFACT_DIRECTORY,
    GAP_VALUES,
    build_recovery_model_audit,
    load_measured_records,
    render_markdown,
)


@pytest.fixture(scope="module")
def audit() -> dict[str, object]:
    """构建一次冻结 Step 9B 审计结果。"""
    return build_recovery_model_audit()


def test_frozen_artifact_is_complete() -> None:
    records = load_measured_records()

    assert len(records) == 690
    assert len({record["equivalence_class"] for record in records}) == 69
    assert all(record["correctness_pass"] is True for record in records)
    assert all(record["safety_pass"] is True for record in records)


def test_phi_calibration_uses_zero_gap_baseline(
    audit: dict[str, object],
) -> None:
    calibration = audit["phi_calibration"]
    rows = {row["gap_tokens"]: row for row in calibration["rows"]}

    assert calibration["baseline_ttft_mean_ms"] == pytest.approx(
        39.801623748952885
    )
    assert tuple(rows) == GAP_VALUES
    assert rows[0]["phi_ms"] == 0.0
    assert rows[0]["measured_incremental_ttft_ms"] == 0.0
    assert rows[8192]["phi_ms"] == pytest.approx(378.5894282627851)
    assert rows[8192]["measured_incremental_ttft_ms"] == pytest.approx(
        290.7188608939043
    )
    assert rows[32768]["relative_error_percent"] == pytest.approx(
        0.42658519495278985
    )
    assert calibration["material_drift_gaps"] == (4096, 8192, 16384)


def test_sota_k8_same_total_gap_but_different_distribution(
    audit: dict[str, object],
) -> None:
    rows = audit["sota_signal_k8"]
    marconi = rows["Marconi-style"]
    flowstate = rows["FlowState"]

    assert marconi["gap_histogram"] == {
        0: 20,
        4096: 0,
        8192: 20,
        16384: 0,
        32768: 0,
    }
    assert flowstate["gap_histogram"] == {
        0: 32,
        4096: 0,
        8192: 4,
        16384: 0,
        32768: 4,
    }
    assert marconi["total_gap_tokens"] == flowstate["total_gap_tokens"] == 163840
    assert flowstate["frozen_phi_predicted_total_cost_ms"] < (
        marconi["frozen_phi_predicted_total_cost_ms"]
    )
    assert flowstate["measured_incremental_ttft_total_ms"] > (
        marconi["measured_incremental_ttft_total_ms"]
    )
    assert flowstate["measured_ttft_weighted_mean_ms"] > (
        marconi["measured_ttft_weighted_mean_ms"]
    )


def test_all_policy_histograms_recover_one_logical_workload(
    audit: dict[str, object],
) -> None:
    for row in audit["policy_audits"]:
        expected = 60 if row["scenario"] == "scalable_multiworkflow_v2_n16" else 40
        assert row["logical_request_count"] == expected
        assert sum(row["gap_histogram"].values()) == expected
        assert sum(row["gap_fractions"].values()) == pytest.approx(1.0)


def test_rank_consistency_only_fails_at_sota_k8(
    audit: dict[str, object],
) -> None:
    ranking = {
        (row["scenario"], row["K"]): row
        for row in audit["rank_consistency"]
    }

    assert ranking[("scalable_multiworkflow_v2_n16", 4)]["ranking_same"] is True
    assert ranking[("scalable_multiworkflow_v2_n16", 12)]["ranking_same"] is True
    assert ranking[("sota_signal_stress_v1", 4)]["ranking_same"] is True
    assert ranking[("sota_signal_stress_v1", 8)]["ranking_same"] is False
    assert ranking[("sota_signal_stress_v1", 8)]["inversions"] == (
        ("Marconi-style", "FlowState"),
    )


def test_tail_audit_preserves_mean_and_p95_distinction(
    audit: dict[str, object],
) -> None:
    rows = {
        (row["scenario"], row["K"], row["policy"]): row
        for row in audit["policy_audits"]
    }
    scalable_flow = rows[
        ("scalable_multiworkflow_v2_n16", 4, "FlowState")
    ]
    scalable_lru = rows[
        ("scalable_multiworkflow_v2_n16", 4, "Global-LRU")
    ]
    signal_flow = rows[("sota_signal_stress_v1", 8, "FlowState")]
    signal_marconi = rows[("sota_signal_stress_v1", 8, "Marconi-style")]

    assert scalable_flow["mean_gap_tokens"] < scalable_lru["mean_gap_tokens"]
    assert scalable_flow["measured_ttft_weighted_mean_ms"] < (
        scalable_lru["measured_ttft_weighted_mean_ms"]
    )
    assert scalable_flow["measured_ttft_weighted_p95_ms"] > (
        scalable_lru["measured_ttft_weighted_p95_ms"]
    )
    assert signal_flow["p95_gap_tokens"] == 32768
    assert signal_marconi["p95_gap_tokens"] == 8192


def test_markdown_report_contains_required_diagnosis(
    audit: dict[str, object],
) -> None:
    report = render_markdown(audit)

    assert "calibration drift" in report
    assert "post-hoc refit" in report
    assert "SOTA-signal K8" in report
    assert "ranking_same" in report
    assert "独立 Recovery Profiler recalibration" in report
    assert str(DEFAULT_ARTIFACT_DIRECTORY) in report
