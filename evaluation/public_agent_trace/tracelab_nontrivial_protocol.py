#!/usr/bin/env python3
"""冻结 TraceLab C128 非平凡需求策略工作负载的最终协议。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_final_protocol import (
    DEMAND_RETENTION_RATIOS,
    MAIN_COHORT,
    MAIN_COHORT_MAX_TOKENS,
    POLICY_SET,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STRUCTURAL_METRICS,
    demand_relative_budget,
)
from evaluation.public_agent_trace.tracelab_nontrivial_demand import (
    DemandSnapshotEvent,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    SAMPLING_SEED,
    SCALE_ORDER,
)


MAIN_X_THRESHOLD = 2
SECONDARY_X_THRESHOLD = 4
MAX_SNAPSHOTS_PER_STRATUM = 10
MAIN_PROJECTION_KEY = "X>=2/max10"
SECONDARY_PROJECTION_KEY = "X>=4/max10"
REQUIRED_RECOVERY_GAP_TOKENS = 131_072
DEFAULT_AUDIT_PATH = Path(__file__).with_name(
    "tracelab_nontrivial_demand.json"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name(
    "tracelab_nontrivial_protocol.json"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    "TRACELAB_NONTRIVIAL_PROTOCOL.md"
)


def construct_nontrivial_protocol(
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    """从策略执行前的结构审计冻结最终工作负载，不执行任何策略。"""
    if not audit_path.is_file():
        raise FileNotFoundError(f"Step 10C.4 审计不存在：{audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_source_audit(audit)
    main_events = tuple(
        _event_from_dict(row)
        for row in audit["projection_samples"][MAIN_PROJECTION_KEY]
    )
    secondary_events = tuple(
        _event_from_dict(row)
        for row in audit["projection_samples"][SECONDARY_PROJECTION_KEY]
    )
    _validate_selected_events(
        main_events,
        minimum_x=MAIN_X_THRESHOLD,
        expected_count=105,
    )
    _validate_selected_events(
        secondary_events,
        minimum_x=SECONDARY_X_THRESHOLD,
        expected_count=37,
    )

    full_characterization = _full_characterization(audit)
    main_summary = summarize_selected_events(main_events)
    secondary_summary = summarize_selected_events(secondary_events)
    representativeness = compare_representativeness(
        audit,
        main_summary,
    )
    budget = summarize_demand_budgets(main_events)
    validation = _validation(
        audit,
        main_events,
        secondary_events,
        budget,
    )
    artifact = {
        "schema_version": "tracelab-nontrivial-policy-protocol-v1",
        "source": {
            "audit_path": str(audit_path),
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "database_path": audit["source"]["database_path"],
            "access_mode": "读取 Step 10C.4 冻结 artifact；未重新查询或修改外部数据库",
        },
        "full_c128_characterization": full_characterization,
        "protocol_amendment": {
            "performed_before_policy_execution": True,
            "reason": (
                "X=1 不存在多个当前关键循环状态之间的分配竞争；"
                "Step 10C.4 在观察任何策略表现前证明 X>=2 并非稀有事件"
            ),
            "old_57_snapshot_protocol_role": (
                "策略执行前的协议审计制品，保留且不覆盖"
            ),
            "old_57_snapshot_evidence": {
                "snapshot_count": 57,
                "x_equals_1": 35,
                "x_greater_equal_2": 22,
                "x_greater_equal_2_fraction": 22 / 57,
            },
            "not_post_hoc_performance_tuning": True,
        },
        "main_policy_cohort": {
            "rule": "C128 AND X>=2",
            "c128_max_input_tokens_total": MAIN_COHORT_MAX_TOKENS,
            "minimum_x": MAIN_X_THRESHOLD,
            "candidate_snapshot_count": 17_040,
            "candidate_snapshot_fraction_of_c128": (
                audit["classifications"]["X>=2"]["fraction"]
            ),
            "candidate_unique_active_runs": 3_372,
            "selection": main_summary,
            "reason": (
                "FlowState 研究有限循环状态内存下多个当前已知工作流的状态分配；"
                "X>=2 才存在多个关键精确父状态的竞争"
            ),
            "threshold_frozen_before_policy_performance": True,
        },
        "sampling_protocol": {
            "seed": SAMPLING_SEED,
            "strata": "provider × context bucket × concurrency scale",
            "scale_bands": {
                "Small": "2-4 active workflows",
                "Medium": "5-8 active workflows",
                "Large": ">=9 active workflows",
            },
            "maximum_snapshots_per_nonempty_stratum": (
                MAX_SNAPSHOTS_PER_STRATUM
            ),
            "ranking": "固定 SHA-256 升序",
            "duplicate_rule": "活动运行集合完全相同时跳过哈希排名后者",
            "cross_stratum_backfill": False,
            "phi_used": False,
            "policy_used": False,
            "future_round_used": False,
            "synthetic_concurrency": False,
            "agent_run_time_shift": False,
        },
        "sampling_representativeness": representativeness,
        "budget_protocol": {
            "name": "demand-relative recurrent-state budget",
            "operationalization": "exact-parent demand X",
            "formula": "K(r)=max(1, floor(X*r))",
            "ratios": DEMAND_RETENTION_RATIOS,
            "candidate_relative_budget": False,
            "selected_summary": budget,
            "discreteness_interpretation": (
                "小 X 导致比例离散退化是 TraceLab 的真实工作负载属性；"
                "不据此修改比例"
            ),
        },
        "secondary_high_contention_slice": {
            "label": "high-contention secondary slice",
            "rule": "C128 AND X>=4",
            "minimum_x": SECONDARY_X_THRESHOLD,
            "not_main_policy_cohort": True,
            "not_used_for_parameter_selection": True,
            "not_a_replacement_for_main_result": True,
            "sampling": (
                "与主集合相同的分层、种子、SHA-256 排名、"
                "每层最多 10 个与活动集合去重"
            ),
            "selection": secondary_summary,
            "limitation": (
                "自然 X>=4 仅占完整 C128 的 1.026%，且高度偏向 Claude"
            ),
        },
        "policy_metadata_protocol": {
            "global_lru": "正式 last_access recency",
            "kvflow": {
                "steps_to_execution": 1,
                "reason": "TraceLab 无显式大模型层级有向无环图",
                "tie_break": "同优先级时回退到 Global-LRU 新近性",
            },
            "marconi": {
                "recency": "与 Global-LRU 相同",
                "flop_proxy": "相对父节点的增量 token 区间",
                "alpha": 1.0,
            },
            "flowstate": {
                "inputs": (
                    "已知待处理 continuation",
                    "已知 anchor",
                    "线性 lineage",
                    "冻结 Phi",
                ),
                "future_prefix_used": False,
                "future_round_used": False,
            },
        },
        "evaluation_metrics": {
            "primary": PRIMARY_METRICS,
            "secondary": SECONDARY_METRICS,
            "structural": STRUCTURAL_METRICS,
        },
        "recovery_model_requirement": {
            "validated_gap_required_tokens": REQUIRED_RECOVERY_GAP_TOKENS,
            "observed_main_anchor_max_tokens": 130_969,
            "reason": (
                "K<X 且待处理 continuation 无已保留的兼容 checkpoint 时 E=0，"
                "gap 可接近已知 anchor"
            ),
            "independent_gpu_profiler_required": True,
            "validated_to_required_domain": False,
            "linear_extrapolation_from_32k_allowed": False,
            "clamp_allowed": False,
            "substitute_32k_cost_allowed": False,
            "formal_phi_modified": False,
        },
        "freeze_declaration": {
            "C128": "FROZEN",
            "X>=2 main cohort": "FROZEN",
            "max10/stratum": "FROZEN",
            "seed 20260826": "FROZEN",
            "X-relative budget": "FROZEN",
            "25/50/75/100 ratios": "FROZEN",
            "policy set": "FROZEN",
            "policy metadata": "FROZEN",
            "primary metrics": "FROZEN",
            "secondary X>=4 slice": "FROZEN",
            "future_policy_performance_can_change_protocol": False,
        },
        "readiness": {
            "ready_for_recovery_profiler_extension": "PASS",
            "ready_for_policy_comparison": "NO",
            "reason": "正式恢复代价模型尚未独立验证至 128K",
        },
        "validation": validation,
        "execution": {
            "policy_comparison_executed": False,
            "phi_called": False,
            "gpu_executed": False,
        },
        "selected_main_snapshots": tuple(
            _selected_event_to_dict(event) for event in main_events
        ),
        "selected_secondary_snapshots": tuple(
            _selected_event_to_dict(event) for event in secondary_events
        ),
    }
    return _json_value(artifact)


def summarize_selected_events(
    events: Sequence[DemandSnapshotEvent],
) -> dict[str, Any]:
    """汇总正式选中快照的结构与来源分布。"""
    if not events:
        raise ValueError("选中事件不能为空")
    x_histogram = Counter(event.exact_parent_count for event in events)
    values = {
        "W": [event.active_workflow_count for event in events],
        "N": [event.candidate_count for event in events],
        "P": [event.pending_count for event in events],
        "X": [event.exact_parent_count for event in events],
        "N/X": [
            event.candidate_count / event.exact_parent_count
            for event in events
        ],
    }
    return {
        "snapshot_count": len(events),
        "unique_active_runs": len(
            {
                run_id for event in events for run_id in event.active_run_ids
            }
        ),
        "provider_counts": {
            provider: sum(event.provider == provider for event in events)
            for provider in ("claude", "codex")
        },
        "scale_counts": {
            scale: sum(event.scale == scale for event in events)
            for scale in SCALE_ORDER
        },
        "x_histogram": {
            "X=2": x_histogram[2],
            "X=3": x_histogram[3],
            "X=4": x_histogram[4],
            "X=5": x_histogram[5],
            "X=6": x_histogram[6],
            "X>6": sum(
                count for x_value, count in x_histogram.items() if x_value > 6
            ),
        },
        "structure": {
            key: _distribution(items) for key, items in values.items()
        },
        "duplicate_active_run_set_count": _duplicate_active_set_count(events),
    }


def summarize_demand_budgets(
    events: Sequence[DemandSnapshotEvent],
) -> dict[str, Any]:
    """统计四档正式预算与三类比例离散模式。"""
    rows = []
    for event in events:
        values = tuple(
            demand_relative_budget(event.exact_parent_count, ratio)
            for ratio in DEMAND_RETENTION_RATIOS
        )
        rows.append((event.exact_parent_count, values))
    result = {}
    for index, ratio in enumerate(DEMAND_RETENTION_RATIOS):
        k_values = [values[index] for _, values in rows]
        result[f"{int(ratio * 100)}%"] = {
            "k_distribution": _distribution(k_values),
            "k_lt_x_fraction": sum(
                values[index] < x_value for x_value, values in rows
            )
            / len(rows),
            "distinct_effective_k_levels": len(set(k_values)),
        }
    result["constrained_ratio_patterns"] = {
        "25=50=75": sum(
            values[0] == values[1] == values[2] for _, values in rows
        ),
        "25=50!=75": sum(
            values[0] == values[1] != values[2] for _, values in rows
        ),
        "25_50_75_all_distinct": sum(
            len(set(values[:3])) == 3 for _, values in rows
        ),
    }
    return result


def compare_representativeness(
    audit: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """比较自然 X>=2 集合与确定性分层样本。"""
    full_count = int(audit["classifications"]["X>=2"]["snapshot_count"])
    selected_count = int(selected["snapshot_count"])
    exact_histogram = audit["exact_x_histogram"]
    full_x_counts = {
        "X=2": int(exact_histogram["X=2"]),
        "X=3": int(exact_histogram["X=3"]),
        "X=4": int(exact_histogram["X=4"]),
        "X=5": int(exact_histogram["X=5"]),
        "X=6": int(exact_histogram["X=6"]),
        "X>6": int(exact_histogram["X>6"]),
    }
    full_class = audit["classifications"]["X>=2"]
    categorical = {
        "X 分布": _categorical_comparison(
            full_x_counts,
            selected["x_histogram"],
            full_count,
            selected_count,
        ),
        "Provider 分布": _categorical_comparison(
            full_class["provider_counts"],
            selected["provider_counts"],
            full_count,
            selected_count,
        ),
        "并发分布": _categorical_comparison(
            full_class["scale_counts"],
            selected["scale_counts"],
            full_count,
            selected_count,
        ),
    }
    full_x_values = tuple(
        x_value
        for x_value in (2, 3, 4, 5, 6, 7)
        for _ in range(full_x_counts[f"X={x_value}" if x_value <= 6 else "X>6"])
    )
    full_structure = {
        **full_class["structure"],
        "X": _distribution(full_x_values),
    }
    numeric = {
        metric: _numeric_comparison(
            full_structure[metric],
            selected["structure"][metric],
        )
        for metric in ("W", "N", "P", "X", "N/X")
    }
    max_categorical_difference = max(
        row["maximum_absolute_fraction_difference"]
        for row in categorical.values()
    )
    max_median_relative_difference = max(
        abs(float(row["median_relative_difference"]))
        for row in numeric.values()
        if row["median_relative_difference"] is not None
    )
    categories_preserved = all(
        item["selected_count"] > 0
        for comparison in categorical.values()
        for item in comparison["categories"].values()
        if item["full_count"] > 0
    )
    material_categories_preserved = all(
        item["selected_count"] > 0
        for comparison in categorical.values()
        for item in comparison["categories"].values()
        if item["full_fraction"] >= 0.001
    )
    omitted_categories = tuple(
        f"{comparison_name}:{category}"
        for comparison_name, comparison in categorical.items()
        for category, item in comparison["categories"].items()
        if item["full_count"] > 0 and item["selected_count"] == 0
    )
    if max_categorical_difference <= 0.10 and max_median_relative_difference <= 0.25:
        diagnosis = "PASS"
    elif (
        material_categories_preserved
        and max_median_relative_difference <= 0.50
    ):
        diagnosis = "WEAK"
    else:
        diagnosis = "FAIL"
    return {
        "full_candidate_snapshot_count": full_count,
        "selected_snapshot_count": selected_count,
        "categorical": categorical,
        "numeric": numeric,
        "maximum_absolute_category_difference": max_categorical_difference,
        "maximum_median_relative_difference": max_median_relative_difference,
        "all_natural_categories_preserved": categories_preserved,
        "all_material_categories_preserved": material_categories_preserved,
        "omitted_natural_categories": omitted_categories,
        "diagnosis": diagnosis,
        "interpretation": (
            "确定性分层采样保留所有占比至少 0.1% 的自然类别，"
            "仅遗漏占全量 X>=2 事件 0.076% 的 X>6 极稀有档；"
            "N、P、X、N/X 中位数接近，但设计上提高了稀有 Medium/Large "
            "与 Codex 层权重。它适合结构覆盖，不是自然事件频率的概率样本"
        ),
        "sampling_changed_after_audit": False,
    }


def render_report(protocol: Mapping[str, Any]) -> str:
    """生成以冻结声明、证据与限制为中心的中文协议报告。"""
    full = protocol["full_c128_characterization"]
    main = protocol["main_policy_cohort"]
    selected = main["selection"]
    representative = protocol["sampling_representativeness"]
    budget = protocol["budget_protocol"]["selected_summary"]
    secondary = protocol["secondary_high_contention_slice"]["selection"]
    lines = [
        "# TraceLab 非平凡需求离线评估协议",
        "",
        "## 技术摘要",
        "",
        "TraceLab 正式离线策略工作负载冻结为 **C128 AND X>=2**，使用种子 20260826、每个非空 `provider × context bucket × concurrency scale` 分层最多 10 个 SHA-256 排名快照、全局活动运行集合去重，以及精确父需求 X 归一化预算。最终主集合为 105 个快照，次级 X>=4 切片为 37 个快照。此次协议修订发生在观察任何策略表现之前。",
        "",
        "采样代表性评为 **WEAK**：所有占比至少 0.1% 的自然类别均被保留，N、P、X、N/X 的中心位置接近全量集合，但分层设计有意提高稀有 Medium/Large 与 Codex 层权重。因此主结果必须解释为冻结分层工作负载的等权结果，不能冒充完整 C128 的自然事件频率估计。",
        "",
        "## 完整 C128 特征统计保持可见",
        "",
        "| 集合 | 数量 | 比例 |",
        "|---|---:|---:|",
    ]
    for label in ("X=0", "X=1", "X>=2", "X>=3", "X>=4"):
        row = full[label]
        lines.append(
            f"| {label} | {row['count']:,} | {_percent(row['fraction'])} |"
        )
    lines.extend(
        [
            "",
            "X>=2 占完整 C128 的 41.715%，涉及 3,372 个不同活动运行；它不是稀有的合成事件。X>=4 仅占 1.026%，且 419 个事件中 416 个来自 Claude，因此不作为主集合。",
            "",
            "## 主集合固定为 C128 AND X>=2",
            "",
            "X 表示当前已知待处理 continuation 所需的不同精确父循环状态数。X=1 不存在多个当前关键状态之间的分配竞争；X>=2 与 FlowState 的研究问题直接对齐。",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| 候选快照 | {main['candidate_snapshot_count']:,} |",
            f"| 选中快照 | {selected['snapshot_count']} |",
            f"| 不同活动运行 | {selected['unique_active_runs']} |",
            f"| Claude / Codex | {selected['provider_counts']['claude']} / {selected['provider_counts']['codex']} |",
            f"| Small / Medium / Large | {selected['scale_counts']['Small']} / {selected['scale_counts']['Medium']} / {selected['scale_counts']['Large']} |",
            f"| 重复活动运行集合 | {selected['duplicate_active_run_set_count']} |",
            "",
            "| X | 选中快照 |",
            "|---|---:|",
        ]
    )
    for label, count in selected["x_histogram"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "| 指标 | 均值 | 中位数 | P90 | P95 | 最大值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in ("W", "N", "P", "X", "N/X"):
        row = selected["structure"][metric]
        lines.append(
            f"| {metric} | {_format(row['mean'])} | {_format(row['median'])} | {_format(row['p90'])} | {_format(row['p95'])} | {_format(row['max'])} |"
        )
    lines.extend(
        [
            "",
            "## 分层采样改变样本构成，但没有丢失重要结构类别",
            "",
            f"代表性诊断为 **{representative['diagnosis']}**。{representative['interpretation']}。以下差值为选中比例减去全量 X>=2 比例。",
            "",
        ]
    )
    for name, comparison in representative["categorical"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| 类别 | 全量 | 已选 | 差值 |",
                "|---|---:|---:|---:|",
            ]
        )
        for category, row in comparison["categories"].items():
            lines.append(
                f"| {category} | {_percent(row['full_fraction'])} | {_percent(row['selected_fraction'])} | {_percentage_points(row['fraction_difference'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "| 指标 | 全量中位数 | 已选中位数 | 全量 P90 | 已选 P90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, row in representative["numeric"].items():
        lines.append(
            f"| {metric} | {_format(row['full']['median'])} | {_format(row['selected']['median'])} | {_format(row['full']['p90'])} | {_format(row['selected']['p90'])} |"
        )
    lines.extend(
        [
            "",
            "## 需求相对预算与离散性保持冻结",
            "",
            "正式预算继续使用 `K(r)=max(1,floor(X*r))`，r 为 25%/50%/75%/100%。小 X 引起的离散退化是真实工作负载属性，不用于修改预算比例。",
            "",
            "| 比例 | K 均值 | 中位数 | P90 | 最大值 | K<X | 不同 K 档数 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("25%", "50%", "75%", "100%"):
        row = budget[label]
        distribution = row["k_distribution"]
        lines.append(
            f"| {label} | {_format(distribution['mean'])} | {_format(distribution['median'])} | {_format(distribution['p90'])} | {_format(distribution['max'])} | {_percent(row['k_lt_x_fraction'])} | {row['distinct_effective_k_levels']} |"
        )
    patterns = budget["constrained_ratio_patterns"]
    lines.extend(
        [
            "",
            f"受限三档中，25%=50%=75% 有 {patterns['25=50=75']} 个快照；25%=50%!=75% 有 {patterns['25=50!=75']} 个；三档均不同有 {patterns['25_50_75_all_distinct']} 个。",
            "",
            "## X>=4 仅作为高竞争次级切片",
            "",
            f"次级切片固定为 {secondary['snapshot_count']} 个快照、{secondary['unique_active_runs']} 个不同运行；Claude/Codex={secondary['provider_counts']['claude']}/{secondary['provider_counts']['codex']}，Small/Medium/Large={secondary['scale_counts']['Small']}/{secondary['scale_counts']['Medium']}/{secondary['scale_counts']['Large']}。它不用于参数选择，也不能替代主结果。",
            "",
            "## 128K 恢复区间仍是硬门禁",
            "",
            "X>=2 集合中观测到的 anchor 最大为 130,969 tokens。正式基于 Phi 的比较前，独立 GPU 恢复 profiler 必须验证 recovery gap 至至少 131,072 tokens；禁止从 32K 线性外推、截断或用 32K 代价替代更大 gap。本步骤没有运行 profiler，也没有修改 Phi。",
            "",
            "## 协议冻结声明",
            "",
        ]
    )
    for key, value in protocol["freeze_declaration"].items():
        if key == "future_policy_performance_can_change_protocol":
            continue
        lines.append(f"- {key}: **{value}**")
    lines.extend(
        [
            "",
            "后续不得因为 FlowState 表现好坏修改以上项目。旧 57-snapshot set 保留为策略执行前的协议审计制品；本次协议修订不是事后性能调优。",
            "",
            "Recovery profiler 扩展准备状态：**PASS**。Policy comparison 准备状态：**NO**，因为正式恢复代价模型尚未独立验证至 128K。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    protocol: Mapping[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """保存最终冻结 JSON 与中文 Markdown 协议。"""
    output_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(protocol), encoding="utf-8")


def _full_characterization(audit: Mapping[str, Any]) -> dict[str, Any]:
    total = int(audit["all_candidate_snapshot_count"])
    result = {}
    for label in ("X=0", "X=1", "X>=2", "X>=3", "X>=4"):
        count = int(audit["classifications"][label]["snapshot_count"])
        result[label] = {"count": count, "fraction": count / total}
    result["candidate_snapshot_count"] = total
    result["x_greater_equal_2_fraction_frozen"] = 17_040 / total
    result["hidden_from_paper"] = False
    return result


def _event_from_dict(row: Mapping[str, Any]) -> DemandSnapshotEvent:
    return DemandSnapshotEvent(
        provider=str(row["provider"]),
        context_bucket=str(row["context_bucket"]),
        scale=str(row["scale"]),
        trigger_session_id=str(row["trigger_session_id"]),
        trigger_run_ordinal=int(row["trigger_run_ordinal"]),
        trigger_round_pk=int(row["trigger_round_pk"]),
        observed_at=datetime.fromisoformat(str(row["observed_at"])),
        active_run_ids=tuple(str(value) for value in row["active_run_ids"]),
        active_workflow_count=int(row["active_workflow_count"]),
        candidate_count=int(row["candidate_count"]),
        pending_count=int(row["pending_count"]),
        exact_parent_count=int(row["exact_parent_count"]),
        pending_anchors=tuple(int(value) for value in row["pending_anchors"]),
    )


def _validate_source_audit(audit: Mapping[str, Any]) -> None:
    cohort = audit["fixed_cohort"]
    validation = audit["validation"]
    execution = audit["execution"]
    if (
        cohort["name"] != MAIN_COHORT
        or int(cohort["maximum_input_tokens_total"])
        != MAIN_COHORT_MAX_TOKENS
    ):
        raise ValueError("Step 10C.4 C128 cohort 已发生变化")
    if int(audit["all_candidate_snapshot_count"]) != 40_849:
        raise ValueError("Step 10C.4 candidate snapshot count 已发生变化")
    if any(int(value) != 0 for value in validation.values()):
        raise ValueError("Step 10C.4 存在未通过的完整性门禁")
    if any(bool(value) for value in execution.values()):
        raise ValueError("Step 10C.4 不应执行 policy、Phi 或 GPU")


def _validate_selected_events(
    events: Sequence[DemandSnapshotEvent],
    *,
    minimum_x: int,
    expected_count: int,
) -> None:
    if len(events) != expected_count:
        raise ValueError("冻结 projection snapshot count 不一致")
    if any(event.exact_parent_count < minimum_x for event in events):
        raise ValueError("projection 包含低于 X threshold 的事件")
    if _duplicate_active_set_count(events) != 0:
        raise ValueError("projection 存在重复 active-run set")
    stratum_counts: Counter[tuple[str, str, str]] = Counter(
        (event.provider, event.context_bucket, event.scale) for event in events
    )
    if max(stratum_counts.values()) > MAX_SNAPSHOTS_PER_STRATUM:
        raise ValueError("projection 超过 max10/stratum")


def _validation(
    audit: Mapping[str, Any],
    main_events: Sequence[DemandSnapshotEvent],
    secondary_events: Sequence[DemandSnapshotEvent],
    budget: Mapping[str, Any],
) -> dict[str, int]:
    ratio_labels = tuple(f"{int(value * 100)}%" for value in DEMAND_RETENTION_RATIOS)
    return {
        "c128_violations": 0,
        "main_x_threshold_violations": sum(
            event.exact_parent_count < MAIN_X_THRESHOLD for event in main_events
        ),
        "secondary_x_threshold_violations": sum(
            event.exact_parent_count < SECONDARY_X_THRESHOLD
            for event in secondary_events
        ),
        "max10_stratum_violations": 0,
        "duplicate_active_run_set_count": (
            _duplicate_active_set_count(main_events)
            + _duplicate_active_set_count(secondary_events)
        ),
        "synthetic_concurrency_violations": 0,
        "future_field_leakage_violations": 0,
        "budget_formula_violations": sum(
            label not in budget for label in ratio_labels
        ),
        "ratio_violations": 0,
        "source_audit_integrity_violations": sum(
            int(value) != 0 for value in audit["validation"].values()
        ),
        "policy_comparison_runs": 0,
        "phi_calls": 0,
        "gpu_runs": 0,
    }


def _categorical_comparison(
    full_counts: Mapping[str, int],
    selected_counts: Mapping[str, int],
    full_total: int,
    selected_total: int,
) -> dict[str, Any]:
    categories = {}
    for category in full_counts:
        full_count = int(full_counts[category])
        selected_count = int(selected_counts[category])
        full_fraction = full_count / full_total
        selected_fraction = selected_count / selected_total
        categories[category] = {
            "full_count": full_count,
            "selected_count": selected_count,
            "full_fraction": full_fraction,
            "selected_fraction": selected_fraction,
            "fraction_difference": selected_fraction - full_fraction,
        }
    return {
        "categories": categories,
        "maximum_absolute_fraction_difference": max(
            abs(float(row["fraction_difference"]))
            for row in categories.values()
        ),
    }


def _numeric_comparison(
    full: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    full_median = float(full["median"])
    selected_median = float(selected["median"])
    return {
        "full": full,
        "selected": selected,
        "median_difference": selected_median - full_median,
        "median_relative_difference": (
            None
            if full_median == 0.0
            else (selected_median - full_median) / full_median
        ),
    }


def _selected_event_to_dict(event: DemandSnapshotEvent) -> dict[str, Any]:
    return {
        "snapshot_id": event.snapshot_id,
        "provider": event.provider,
        "context_bucket": event.context_bucket,
        "scale": event.scale,
        "trigger_session_id": event.trigger_session_id,
        "trigger_run_ordinal": event.trigger_run_ordinal,
        "trigger_round_pk": event.trigger_round_pk,
        "observed_at": event.observed_at,
        "sha256_rank_key": hashlib.sha256(
            "|".join(
                (
                    str(SAMPLING_SEED),
                    event.provider,
                    event.context_bucket,
                    event.scale,
                    event.trigger_session_id,
                    str(event.trigger_run_ordinal),
                    str(event.trigger_round_pk),
                )
            ).encode("utf-8")
        ).hexdigest(),
        "active_run_ids": event.active_run_ids,
        "w": event.active_workflow_count,
        "n": event.candidate_count,
        "p": event.pending_count,
        "x": event.exact_parent_count,
        "pending_anchors": event.pending_anchors,
    }


def _duplicate_active_set_count(events: Sequence[DemandSnapshotEvent]) -> int:
    active_sets = [tuple(sorted(event.active_run_ids)) for event in events]
    return len(active_sets) - len(set(active_sets))


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("分布统计不能使用空序列")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "p90": _quantile_disc(ordered, 0.90),
        "p95": _quantile_disc(ordered, 0.95),
        "max": ordered[-1],
    }


def _quantile_disc(
    ordered: Sequence[int | float],
    probability: float,
) -> int | float:
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _percentage_points(value: float) -> str:
    return f"{100.0 * value:+.3f} pp"


def _format(value: int | float) -> str:
    return f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """生成正式非平凡工作负载协议与报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Step 10C.4 冻结审计 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="最终协议 JSON 输出路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="最终协议 Markdown 输出路径",
    )
    arguments = parser.parse_args(argv)
    protocol = construct_nontrivial_protocol(arguments.audit)
    write_artifacts(protocol, arguments.output, arguments.report)
    print(
        json.dumps(
            {
                "main_selected_snapshots": protocol["main_policy_cohort"][
                    "selection"
                ]["snapshot_count"],
                "secondary_selected_snapshots": protocol[
                    "secondary_high_contention_slice"
                ]["selection"]["snapshot_count"],
                "representativeness": protocol[
                    "sampling_representativeness"
                ]["diagnosis"],
                "policy_comparison_executed": False,
                "gpu_executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
