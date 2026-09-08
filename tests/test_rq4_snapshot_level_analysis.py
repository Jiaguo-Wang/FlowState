"""RQ4-D snapshot-level 正式分析测试。"""

from __future__ import annotations

import math

import pytest

from evaluation.rq4_snapshot_level_analysis import (
    BOOTSTRAP_REPETITIONS,
    aggregate_snapshots,
    average_ranks,
    bootstrap_results,
    bootstrap_samples,
    gap_ttft_consistency,
    paired_comparison,
    percentile,
    policy_summary,
    repetition_stability,
    spearman,
)


def _synthetic_runs():
    """构造覆盖四个 round 的确定性三次 repetition 数据。"""
    runs = []
    snapshot_ids = []
    run_index = 0
    for round_id in (2, 3, 4, 5):
        for ordinal in range(6):
            snapshot_id = f"s{round_id}-{ordinal}"
            snapshot_ids.append(snapshot_id)
            for policy, base_ttft, base_gap in (
                ("LRU", 120.0, 1200.0),
                ("Marconi", 100.0, 1000.0),
                ("FlowState", 80.0, 800.0),
            ):
                for repetition, offset in ((1, -1.0), (2, 0.0), (3, 1.0)):
                    run_index += 1
                    runs.append(
                        {
                            "run_index": run_index,
                            "run_id": f"{snapshot_id}-{policy}-{repetition}",
                            "snapshot_id": snapshot_id,
                            "allocation_round": round_id,
                            "policy": policy,
                            "repetition": repetition,
                            "logical_k": round_id,
                            "selected_count": round_id,
                            "mean_ttft_ms": base_ttft + ordinal + offset,
                            "mean_g_tokens": base_gap + ordinal * 10.0,
                            "mean_h_tokens": 2000.0,
                            "mean_e_tokens": 2000.0 - base_gap - ordinal * 10.0,
                        }
                    )
    return runs, snapshot_ids


def test_percentile_uses_linear_interpolation() -> None:
    """确认百分位数采用冻结的相邻秩线性插值。"""
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert percentile([3.0], 0.95) == 3.0


def test_average_ranks_and_spearman_handle_ties() -> None:
    """确认并列秩与斯皮尔曼相关计算稳定。"""
    assert average_ranks([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0, 1.0], [10.0, 20.0, 30.0]) == 0.0


def test_snapshot_hierarchy_produces_exactly_24_observations() -> None:
    """确认四请求之后的 run 均值只在三次 repetition 间聚合。"""
    runs, snapshot_ids = _synthetic_runs()
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    summary = policy_summary(snapshots, "mean_ttft_ms")
    assert len(snapshots) == 24
    assert all(summary[policy]["observations"] == 24 for policy in summary)
    assert snapshots[0]["policies"]["LRU"]["mean_ttft_ms"] == pytest.approx(120.0)


def test_paired_comparison_uses_snapshot_values() -> None:
    """确认配对差值和主 W/T/L 均按 snapshot 直接比较。"""
    runs, snapshot_ids = _synthetic_runs()
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    comparisons = paired_comparison(snapshots, "mean_ttft_ms")
    assert comparisons["LRU"]["win_tie_loss"] == {"win": 24, "tie": 0, "loss": 0}
    assert comparisons["LRU"]["mean_absolute_reduction"] == pytest.approx(40.0)
    assert comparisons["Marconi"]["mean_absolute_reduction"] == pytest.approx(20.0)


def test_round_stratified_bootstrap_is_deterministic() -> None:
    """确认一万次分层 bootstrap 由固定 seed 完全决定。"""
    runs, snapshot_ids = _synthetic_runs()
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    first = bootstrap_samples(snapshots)
    second = bootstrap_samples(snapshots)
    assert len(first) == BOOTSTRAP_REPETITIONS
    assert first == second
    assert all(len(sample) == 24 for sample in first)
    comparisons = paired_comparison(snapshots, "mean_ttft_ms")
    result = bootstrap_results(comparisons, first)
    assert result["comparisons"]["LRU"]["mean_absolute_reduction_95_ci"] == pytest.approx([40.0, 40.0])


def test_gap_ttft_consistency_reports_counterexample_without_removal() -> None:
    """确认 G 更小但 TTFT 更差的反例被保留并报告。"""
    runs, snapshot_ids = _synthetic_runs()
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    snapshots[0]["policies"]["FlowState"]["mean_ttft_ms"] = 130.0
    ttft = paired_comparison(snapshots, "mean_ttft_ms")
    gap = paired_comparison(snapshots, "mean_g_tokens")
    result = gap_ttft_consistency(snapshots, ttft, gap)
    assert result["comparisons"]["LRU"]["smaller_gap_but_worse_ttft_count"] == 1
    assert snapshot_ids[0] in result["comparisons"]["LRU"]["smaller_gap_but_worse_ttft_snapshots"]


def test_repetition_stability_uses_sample_standard_deviation() -> None:
    """确认 CV 使用三次 repetition 的样本标准差。"""
    runs, snapshot_ids = _synthetic_runs()
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    result = repetition_stability(snapshots)
    first = result["entries"][0]
    assert first["sample_standard_deviation_ms"] == pytest.approx(1.0)
    assert first["coefficient_of_variation"] == pytest.approx(1.0 / 120.0)
    assert result["outliers_removed"] is False
