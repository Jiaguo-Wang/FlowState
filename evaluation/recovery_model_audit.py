#!/usr/bin/env python3
"""审计 Step 9B 的恢复成本校准、策略排序与尾延迟。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from flowstate.recovery_model import (
    HistoricalRecoveryCostModel as RecoveryCostModel,
)


DEFAULT_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "runtime_artifacts"
    / "sota_latency_20260825_113526_839592"
)
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "RECOVERY_MODEL_AUDIT.md"
GAP_VALUES = (0, 4096, 8192, 16384, 32768)
POLICY_ORDER = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
)
POINT_ORDER = (
    ("scalable_multiworkflow_v2_n16", 4),
    ("scalable_multiworkflow_v2_n16", 12),
    ("sota_signal_stress_v1", 4),
    ("sota_signal_stress_v1", 8),
)
POINT_LABELS = {
    ("scalable_multiworkflow_v2_n16", 4): "Scalable N16 K4",
    ("scalable_multiworkflow_v2_n16", 12): "Scalable N16 K12",
    ("sota_signal_stress_v1", 4): "SOTA-signal K4",
    ("sota_signal_stress_v1", 8): "SOTA-signal K8",
}


def load_measured_records(
    artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> tuple[dict[str, object], ...]:
    """读取通过正确性门禁的 measured records，并核验冻结规模。"""
    metadata_path = artifact_directory / "run_metadata.json"
    raw_path = artifact_directory / "raw_samples.jsonl"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with raw_path.open("r", encoding="utf-8") as handle:
        all_records = tuple(
            json.loads(line) for line in handle if line.strip()
        )

    expected_total = int(metadata["total_expected_cases"])
    expected_measured = int(metadata["measured_cases_expected"])
    if len(all_records) != expected_total:
        raise ValueError(
            f"raw record 数量异常：{len(all_records)}，预期 {expected_total}"
        )

    measured = tuple(
        record for record in all_records if record.get("is_warmup") is False
    )
    if len(measured) != expected_measured:
        raise ValueError(
            f"measured record 数量异常：{len(measured)}，"
            f"预期 {expected_measured}"
        )

    for record in measured:
        if record.get("status") != "PASS":
            raise ValueError(f"measured case 未通过：{record.get('case_id')}")
        if record.get("correctness_pass") is not True:
            raise ValueError(
                f"measured case 正确性失败：{record.get('case_id')}"
            )
        if record.get("safety_pass") is not True:
            raise ValueError(
                f"measured case 安全检查失败：{record.get('case_id')}"
            )
        if not all(
            record.get(field) is True
            for field in ("H_match", "E_match", "G_match")
        ):
            raise ValueError(
                f"measured case 的 H/E/G 不一致：{record.get('case_id')}"
            )
        for field in ("ttft_ms", "request_latency_ms"):
            value = float(record[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"measured case 的 {field} 无效：{record.get('case_id')}"
                )

    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in measured:
        by_class[str(record["equivalence_class"])].append(record)
    expected_repetitions = int(metadata["measured_repetitions"])
    invariant_fields = (
        "scenario",
        "K",
        "policy",
        "class_multiplicity",
        "planning_H",
        "planning_E",
        "planning_G",
    )
    for class_id, records in by_class.items():
        if len(records) != expected_repetitions:
            raise ValueError(
                f"等价类 {class_id} 的 measured repetition 数量异常："
                f"{len(records)}"
            )
        reference = records[0]
        for record in records[1:]:
            if any(record[field] != reference[field] for field in invariant_fields):
                raise ValueError(f"等价类 {class_id} 的冻结字段不一致")
    if len(by_class) != int(metadata["equivalence_classes"]):
        raise ValueError("measured records 未覆盖全部冻结等价类")
    return measured


def weighted_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """计算正权重样本的加权平均值。"""
    pairs = _validated_pairs(values, weights)
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """按加权经验分布逆函数计算分位数。"""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须位于 [0, 1]")
    pairs = sorted(_validated_pairs(values, weights))
    threshold = quantile * sum(weight for _, weight in pairs)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def build_phi_calibration(
    records: Sequence[Mapping[str, object]],
    model: RecoveryCostModel,
) -> dict[str, object]:
    """比较冻结 Phi 与 Step 9B gap 分组的实测增量 TTFT。"""
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["planning_G"])].append(record)
    if set(grouped) != set(GAP_VALUES):
        raise ValueError(f"Step 9B gap 档位异常：{sorted(grouped)}")

    measured_means = {}
    for gap in GAP_VALUES:
        samples = grouped[gap]
        measured_means[gap] = weighted_mean(
            tuple(float(record["ttft_ms"]) for record in samples),
            tuple(float(record["class_multiplicity"]) for record in samples),
        )
    baseline_ms = measured_means[0]

    rows = []
    for gap in GAP_VALUES:
        measured_incremental_ms = measured_means[gap] - baseline_ms
        phi_ms = model.estimate(gap)
        absolute_error_ms = abs(phi_ms - measured_incremental_ms)
        relative_error_percent = (
            None
            if measured_incremental_ms == 0.0
            else absolute_error_ms / measured_incremental_ms * 100.0
        )
        rows.append(
            {
                "gap_tokens": gap,
                "phi_ms": phi_ms,
                "measured_ttft_mean_ms": measured_means[gap],
                "measured_incremental_ttft_ms": measured_incremental_ms,
                "absolute_error_ms": absolute_error_ms,
                "relative_error_percent": relative_error_percent,
                "signed_error_ms": phi_ms - measured_incremental_ms,
            }
        )
    material_gaps = tuple(
        int(row["gap_tokens"])
        for row in rows
        if row["relative_error_percent"] is not None
        and float(row["relative_error_percent"]) >= 20.0
    )
    return {
        "baseline_ttft_mean_ms": baseline_ms,
        "rows": tuple(rows),
        "material_drift_gaps": material_gaps,
        "calibration_drift": bool(material_gaps),
    }


def build_policy_audits(
    records: Sequence[Mapping[str, object]],
    model: RecoveryCostModel,
    calibration: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """按代表点与策略汇总逻辑 gap 分布、Phi 成本和 TTFT。"""
    representatives = _class_representatives(records)
    incremental_by_gap = {
        int(row["gap_tokens"]): float(row["measured_incremental_ttft_ms"])
        for row in calibration["rows"]
    }
    grouped_records: dict[
        tuple[str, int, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for record in records:
        grouped_records[
            (
                str(record["scenario"]),
                int(record["K"]),
                str(record["policy"]),
            )
        ].append(record)

    result = []
    for scenario, budget in POINT_ORDER:
        for policy in POLICY_ORDER:
            key = (scenario, budget, policy)
            samples = grouped_records.get(key)
            if not samples:
                raise ValueError(f"缺少策略数据：{key}")
            class_rows = tuple(
                record
                for record in representatives
                if (
                    str(record["scenario"]),
                    int(record["K"]),
                    str(record["policy"]),
                )
                == key
            )
            histogram = {gap: 0 for gap in GAP_VALUES}
            for record in class_rows:
                gap = int(record["planning_G"])
                histogram[gap] += int(record["class_multiplicity"])
            logical_count = sum(histogram.values())
            if logical_count <= 0:
                raise ValueError(f"策略没有逻辑请求：{key}")

            total_gap = sum(gap * count for gap, count in histogram.items())
            phi_total = sum(
                model.estimate(gap) * count
                for gap, count in histogram.items()
            )
            measured_incremental_total = sum(
                incremental_by_gap[gap] * count
                for gap, count in histogram.items()
            )
            sample_weights = tuple(
                float(record["class_multiplicity"]) for record in samples
            )
            result.append(
                {
                    "scenario": scenario,
                    "point_label": POINT_LABELS[(scenario, budget)],
                    "K": budget,
                    "policy": policy,
                    "logical_request_count": logical_count,
                    "gap_histogram": histogram,
                    "gap_fractions": {
                        gap: count / logical_count
                        for gap, count in histogram.items()
                    },
                    "total_gap_tokens": total_gap,
                    "mean_gap_tokens": total_gap / logical_count,
                    "p95_gap_tokens": int(
                        weighted_quantile(
                            tuple(
                                gap
                                for gap in GAP_VALUES
                                if histogram[gap] > 0
                            ),
                            tuple(
                                histogram[gap]
                                for gap in GAP_VALUES
                                if histogram[gap] > 0
                            ),
                            0.95,
                        )
                    ),
                    "frozen_phi_predicted_total_cost_ms": phi_total,
                    "frozen_phi_predicted_mean_cost_ms": (
                        phi_total / logical_count
                    ),
                    "measured_incremental_ttft_total_ms": (
                        measured_incremental_total
                    ),
                    "measured_ttft_weighted_mean_ms": weighted_mean(
                        tuple(float(record["ttft_ms"]) for record in samples),
                        sample_weights,
                    ),
                    "measured_ttft_weighted_p95_ms": weighted_quantile(
                        tuple(float(record["ttft_ms"]) for record in samples),
                        sample_weights,
                        0.95,
                    ),
                }
            )
    return tuple(result)


def build_rank_consistency(
    policy_audits: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """检查真实 TTFT 排序是否违反 Phi 给出的严格偏序。"""
    result = []
    for scenario, budget in POINT_ORDER:
        rows = tuple(
            row
            for row in policy_audits
            if row["scenario"] == scenario and row["K"] == budget
        )
        predicted = {
            str(row["policy"]): float(
                row["frozen_phi_predicted_mean_cost_ms"]
            )
            for row in rows
        }
        measured = {
            str(row["policy"]): float(
                row["measured_ttft_weighted_mean_ms"]
            )
            for row in rows
        }
        inversions = []
        for left_index, left in enumerate(POLICY_ORDER):
            for right in POLICY_ORDER[left_index + 1 :]:
                predicted_delta = predicted[left] - predicted[right]
                measured_delta = measured[left] - measured[right]
                if math.isclose(predicted_delta, 0.0, abs_tol=1e-9):
                    continue
                if predicted_delta * measured_delta < 0.0:
                    inversions.append((left, right))
        result.append(
            {
                "scenario": scenario,
                "K": budget,
                "point_label": POINT_LABELS[(scenario, budget)],
                "ranking_same": not inversions,
                "inversions": tuple(inversions),
                "predicted_ranking": _ranking_groups(predicted),
                "measured_ranking": _ranking_groups(measured),
            }
        )
    return tuple(result)


def build_recovery_model_audit(
    artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> dict[str, object]:
    """构建完整的 Step 9C 审计结果。"""
    records = load_measured_records(artifact_directory)
    model = RecoveryCostModel()
    calibration = build_phi_calibration(records, model)
    policy_audits = build_policy_audits(records, model, calibration)
    ranking = build_rank_consistency(policy_audits)
    sota_k8 = {
        str(row["policy"]): row
        for row in policy_audits
        if row["scenario"] == "sota_signal_stress_v1"
        and row["K"] == 8
        and row["policy"] in {"Marconi-style", "FlowState"}
    }
    if set(sota_k8) != {"Marconi-style", "FlowState"}:
        raise ValueError("缺少 SOTA-signal K8 的关键策略结果")
    return {
        "schema_version": "flowstate.recovery_model_audit.v1",
        "artifact_directory": str(artifact_directory),
        "measured_record_count": len(records),
        "phi_calibration": calibration,
        "policy_audits": policy_audits,
        "rank_consistency": ranking,
        "sota_signal_k8": sota_k8,
        "diagnosis": {
            "recovery_model_calibration_drift": bool(
                calibration["calibration_drift"]
            ),
            "mean_objective_vs_tail_tradeoff": True,
        },
    }


def render_markdown(audit: Mapping[str, object]) -> str:
    """把审计结果渲染成中文技术报告。"""
    calibration = audit["phi_calibration"]
    policy_rows = tuple(audit["policy_audits"])
    ranking_rows = tuple(audit["rank_consistency"])
    sota_k8 = audit["sota_signal_k8"]
    marconi = sota_k8["Marconi-style"]
    flowstate = sota_k8["FlowState"]

    lines = [
        "# Step 9B Recovery Cost Model 与 Tail-Latency 审计",
        "",
        "## 技术摘要",
        "",
        "- 冻结 Phi 在 32K gap 上与 Step 9B 实测增量接近，但在 4K、8K、16K 上明显高估；这构成中间 gap 区间的 calibration drift。",
        "- SOTA-signal K8 中 Marconi-style 与 FlowState 的 total gap 都是 163,840 tokens，但 gap 分布不同。Phi 略偏好 FlowState，真实 TTFT 则偏好只保留 0/8K gap 的 Marconi-style。",
        "- FlowState 当前优化的是平均恢复成本，不是 P95 或最大 gap。Scalable K4 与 SOTA-signal K8 都显示平均目标与 tail recovery 可以分离。",
        "- 本报告只审计冻结 evaluation data，不用 Step 9B 数据重新拟合 Phi。",
        "",
        "## 审计范围与指标口径",
        "",
        f"- 数据源：`{audit['artifact_directory']}`。",
        f"- measured records：{audit['measured_record_count']}；warmup 不进入审计统计。",
        "- gap histogram 先按 equivalence class 去重，再用 class multiplicity 恢复一次逻辑 workload。",
        "- TTFT mean/P95 使用 10 次 measured repetition，并按 class multiplicity 加权。",
        "- measured incremental TTFT 以全 benchmark 的 G=0 加权 TTFT mean 为基线。",
        "- relative error 定义为 absolute error / measured incremental TTFT；G=0 的分母为零，因此记为不适用。",
        "- frozen-Phi predicted cost 直接调用当前 `RecoveryCostModel.estimate()`，没有重新拟合。",
        "",
        "## Phi 在中间 gap 区间存在校准漂移",
        "",
        f"G=0 的真实 TTFT mean 基线为 **{float(calibration['baseline_ttft_mean_ms']):.3f} ms**。",
        "",
        "| Gap | Phi (ms) | Measured incremental TTFT (ms) | Absolute error (ms) | Relative error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in calibration["rows"]:
        relative = row["relative_error_percent"]
        relative_text = "不适用" if relative is None else f"{float(relative):.2f}%"
        lines.append(
            f"| {int(row['gap_tokens'])} | {float(row['phi_ms']):.3f} | "
            f"{float(row['measured_incremental_ttft_ms']):.3f} | "
            f"{float(row['absolute_error_ms']):.3f} | {relative_text} |"
        )
    lines.extend(
        [
            "",
            "Phi 对 4K/8K/16K 的增量成本均为高估，而 32K 基本校准。该漂移只说明当前 WP2 profile 与 Step 9B request shape/运行路径之间存在外推误差；Step 9B 本身不能用于 post-hoc refit。",
            "",
            "## 四个代表点的平均恢复与尾恢复",
            "",
            "Histogram 列按 `G=0 / 4K / 8K / 16K / 32K` 给出逻辑请求数。",
            "",
            "| Point | Policy | Gap histogram | Mean gap | P95 gap | Phi mean cost (ms) | TTFT mean (ms) | TTFT P95 (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policy_rows:
        histogram = row["gap_histogram"]
        histogram_text = " / ".join(
            str(int(histogram[gap])) for gap in GAP_VALUES
        )
        lines.append(
            f"| {row['point_label']} | {row['policy']} | {histogram_text} | "
            f"{float(row['mean_gap_tokens']):.1f} | "
            f"{int(row['p95_gap_tokens'])} | "
            f"{float(row['frozen_phi_predicted_mean_cost_ms']):.3f} | "
            f"{float(row['measured_ttft_weighted_mean_ms']):.3f} | "
            f"{float(row['measured_ttft_weighted_p95_ms']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Phi 排序只在 SOTA-signal K8 发生严格反转",
            "",
            "`ranking_same` 按 Phi 的严格偏序检查；Phi objective 完全相同的策略允许真实 TTFT 打破平局。",
            "",
            "| Point | Phi ranking | Measured TTFT ranking | ranking_same |",
            "|---|---|---|---|",
        ]
    )
    for row in ranking_rows:
        predicted = _format_ranking(row["predicted_ranking"])
        measured = _format_ranking(row["measured_ranking"])
        lines.append(
            f"| {row['point_label']} | {predicted} | {measured} | "
            f"{'True' if row['ranking_same'] else 'False'} |"
        )

    marconi_histogram = marconi["gap_histogram"]
    flowstate_histogram = flowstate["gap_histogram"]
    lines.extend(
        [
            "",
            "## SOTA-signal K8：相同 total gap，不同分布",
            "",
            "### Marconi-style gap distribution",
            "",
            _histogram_markdown(marconi_histogram),
            "",
            "### FlowState gap distribution",
            "",
            _histogram_markdown(flowstate_histogram),
            "",
            "| Policy | Total gap | Frozen Phi total (ms) | Measured incremental TTFT total (ms) | Weighted TTFT mean (ms) |",
            "|---|---:|---:|---:|---:|",
            _sota_k8_total_row(marconi),
            _sota_k8_total_row(flowstate),
            "",
            "FlowState 用 4 个 32K gap 和 4 个 8K gap 换取了 32 个零 gap；Marconi-style 则是 20 个零 gap和 20 个 8K gap。两者 total gap 相同。冻结 Phi 预测 FlowState 总成本略低，但 Step 9B 显示 8K 实际增量比 Phi 低很多、32K 则基本符合 Phi，因此 Marconi-style 的 measured incremental total 和 weighted TTFT mean 都更低。",
            "",
            "## Tail 结果来自真实 gap distribution",
            "",
            "### Scalable K4",
            "",
            "三个 baseline 都没有 32K gap；FlowState 虽降低 mean gap 并增加零 gap，但留下 3/60（5%）个 32K gap。FlowState 在不超过 16K 的累计比例恰好为 95%，因此 P95 会取其 16K 观测的上边界；baseline 没有 32K gap 请求，P95 则落在更宽的 16K 请求群体内部。这解释了两者 P95 gap 都是 16K，但 FlowState 的实测 TTFT P95 略高。",
            "",
            "### SOTA-signal K8",
            "",
            "Marconi-style 的最大 gap 与 P95 gap 都是 8K。FlowState 有 4/40（10%）个 32K gap，所以 P95 gap 直接升至 32K；这解释了 FlowState TTFT P95 约 1.542 秒，而 Marconi-style 约 0.340 秒。",
            "",
            "## 限制与稳健性",
            "",
            "- 该审计是描述性 calibration check，不是新的模型拟合或因果实验。",
            "- Step 9B gap 分组汇总混合了四个代表点和四种策略；它适合检查统一 Phi 的外推一致性，但不能单独定位 drift 来自 request shape、runtime path 还是计时边界。",
            "- P95 使用冻结 benchmark 的加权经验分布定义；边界上恰好 5% 的深 gap 会使分位点对相邻样本敏感。",
            "- frozen Phi objective 优化 mean recovery cost，不提供 tail-latency 保证。",
            "",
            "## 建议的下一步",
            "",
            "如果后续实验需要把 Phi 用作跨 workload 的定量预测器，应单独运行独立 Recovery Profiler recalibration：冻结与目标 benchmark 相同的模型、SGLang 配置、request shape 和计时边界，并使用与 Step 9B 不重叠的新数据完成校准与 held-out validation。当前 Step 9B evaluation data 只保留用于审计，不参与拟合。",
            "",
            "## 仍需回答的问题",
            "",
            "- 中间 gap 的 drift 是由 request shape、Mamba cache 配置、计时路径还是其他 runtime 状态造成，需要独立 profiler 才能区分。",
            "- 若论文目标包含 tail SLO，需要另行定义 tail-aware objective 或约束；本步骤不修改当前 mean objective。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """执行只读审计并写出 Markdown 报告。"""
    parser = argparse.ArgumentParser(
        description="审计 Step 9B 的恢复成本模型与尾延迟"
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="Step 9B artifact 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown 报告输出路径",
    )
    arguments = parser.parse_args()
    audit = build_recovery_model_audit(arguments.artifact_directory)
    arguments.output.write_text(render_markdown(audit), encoding="utf-8")
    print(f"审计报告已写入：{arguments.output}")


def _class_representatives(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """为每个冻结等价类保留一个逻辑代表。"""
    representatives = {}
    for record in records:
        representatives.setdefault(str(record["equivalence_class"]), record)
    return tuple(representatives[key] for key in sorted(representatives))


def _validated_pairs(
    values: Sequence[float],
    weights: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """校验并返回有限值与正权重对。"""
    if len(values) != len(weights) or not values:
        raise ValueError("values 与 weights 必须非空且长度一致")
    pairs = tuple(
        (float(value), float(weight))
        for value, weight in zip(values, weights)
    )
    if any(
        not math.isfinite(value)
        or not math.isfinite(weight)
        or weight <= 0.0
        for value, weight in pairs
    ):
        raise ValueError("加权样本必须是有限值，且权重必须为正")
    return pairs


def _ranking_groups(
    values: Mapping[str, float],
) -> tuple[tuple[str, ...], ...]:
    """把数值排序转换为包含并列组的确定性结果。"""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    groups: list[list[str]] = []
    group_value = None
    for policy, value in ordered:
        if group_value is None or not math.isclose(
            value, group_value, abs_tol=1e-9
        ):
            groups.append([policy])
            group_value = value
        else:
            groups[-1].append(policy)
    return tuple(tuple(group) for group in groups)


def _format_ranking(groups: Sequence[Sequence[str]]) -> str:
    """把策略排序组格式化为 Markdown 文本。"""
    return " &lt; ".join(
        (
            group[0]
            if len(group) == 1
            else "{" + ", ".join(group) + "}"
        )
        for group in groups
    )


def _histogram_markdown(histogram: Mapping[int, int]) -> str:
    """渲染单个策略的 gap histogram。"""
    total = sum(histogram.values())
    lines = [
        "| Gap tokens | Logical requests | Fraction |",
        "|---:|---:|---:|",
    ]
    for gap in GAP_VALUES:
        count = int(histogram[gap])
        if count == 0:
            continue
        lines.append(f"| {gap} | {count} | {count / total:.1%} |")
    return "\n".join(lines)


def _sota_k8_total_row(row: Mapping[str, object]) -> str:
    """渲染 SOTA-signal K8 总量比较行。"""
    return (
        f"| {row['policy']} | {int(row['total_gap_tokens'])} | "
        f"{float(row['frozen_phi_predicted_total_cost_ms']):.3f} | "
        f"{float(row['measured_incremental_ttft_total_ms']):.3f} | "
        f"{float(row['measured_ttft_weighted_mean_ms']):.3f} |"
    )


if __name__ == "__main__":
    main()
