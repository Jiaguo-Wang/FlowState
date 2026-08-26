#!/usr/bin/env python3
"""审计 TraceLab 累计 context coverage 与逻辑 recurrent-state pressure。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CONTEXT_BUCKET_ORDER,
    SCALE_ORDER,
    SAMPLING_SEED,
    CompletedRoundFact,
    PendingRoundFact,
    SampledSnapshotEvent,
    TraceSnapshot,
    build_trace_snapshot,
    deterministic_sample_key,
    workflow_id_for,
)
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


CONTEXT_COHORTS = (
    ("C32", 32_768),
    ("C64", 65_536),
    ("C128", 131_072),
    ("C256", 262_144),
)
BUDGET_RATIOS = (0.25, 0.50, 0.75, 1.00)
BUDGET_NORMALIZATIONS = (
    "candidate_relative",
    "pending_relative",
    "exact_parent_relative",
)
GAP_DEPTHS = (1, 2, 4, 8)
GAP_THRESHOLDS = (4_096, 8_192, 16_384, 32_768, 65_536, 131_072)
MATERIAL_GAIN_THRESHOLD = 0.20
DEFAULT_OUTPUT_PATH = Path(__file__).with_name(
    "tracelab_context_pressure.json"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    "TRACELAB_CONTEXT_PRESSURE.md"
)


@dataclass(frozen=True)
class ContextSnapshotEvent:
    """累计 context cohort 中由真实 overlap 产生的采样时点。"""

    cohort: str
    cutoff_tokens: int
    event: SampledSnapshotEvent


def context_cohort_contains(
    max_input_tokens_total: int,
    cutoff_tokens: int,
) -> bool:
    """判断一个完整 Agent Run 是否属于指定累计 context cohort。"""
    if max_input_tokens_total < 0:
        raise ValueError("max_input_tokens_total 必须非负")
    if cutoff_tokens <= 0:
        raise ValueError("cutoff_tokens 必须大于零")
    return max_input_tokens_total <= cutoff_tokens


def context_bucket(max_input_tokens_total: int) -> str:
    """按 Step 10C 的固定边界返回完整 run context bucket。"""
    if max_input_tokens_total < 0:
        raise ValueError("max_input_tokens_total 必须非负")
    if max_input_tokens_total <= 32_768:
        return "<=32K"
    if max_input_tokens_total <= 65_536:
        return "32K-64K"
    if max_input_tokens_total <= 131_072:
        return "64K-128K"
    if max_input_tokens_total <= 262_144:
        return "128K-256K"
    return ">256K"


def choose_context_events(
    events: Sequence[ContextSnapshotEvent],
) -> tuple[ContextSnapshotEvent, ...]:
    """按 Step 10C stratum 和固定哈希键选择确定性真实时点。"""
    selected: dict[tuple[str, str, str, str], ContextSnapshotEvent] = {}
    for item in events:
        event = item.event
        stratum = (
            item.cohort,
            event.scale,
            event.provider,
            event.context_bucket,
        )
        current = selected.get(stratum)
        if current is None or _event_rank(item) < _event_rank(current):
            selected[stratum] = item
    scale_rank = {value: index for index, value in enumerate(SCALE_ORDER)}
    bucket_rank = {
        value: index for index, value in enumerate(CONTEXT_BUCKET_ORDER)
    }
    cohort_rank = {
        value: index for index, (value, _) in enumerate(CONTEXT_COHORTS)
    }
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                cohort_rank[item.cohort],
                scale_rank[item.event.scale],
                item.event.provider,
                bucket_rank[item.event.context_bucket],
                item.event.snapshot_id,
            ),
        )
    )


def budget_k(base_count: int, ratio: float) -> int:
    """按冻结公式计算至少为一的结构预算。"""
    if base_count < 0:
        raise ValueError("base_count 必须非负")
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("ratio 必须位于 (0, 1] 区间")
    return max(1, math.floor(base_count * ratio))


def exact_parent_checkpoint_ids(
    continuation: PendingContinuation,
    candidates: Sequence[CheckpointCandidate],
) -> tuple[str, ...]:
    """返回同 workflow、兼容 lineage 且位于 anchor 的 distinct 候选。"""
    return tuple(
        candidate.checkpoint_id
        for candidate in sorted(candidates, key=lambda item: item.checkpoint_id)
        if candidate.token_pos == continuation.anchor_pos
        and is_compatible(candidate, continuation)
    )


def lineage_recovery_envelope(
    continuation: PendingContinuation,
    candidates: Sequence[CheckpointCandidate],
) -> dict[str, int | None]:
    """计算 exact parent 丢失后第 1、2、4、8 个历史状态对应的 gap。"""
    historical = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.workflow_id == continuation.workflow_id
                and len(candidate.lineage_path) < len(
                    continuation.lineage_path
                )
                and candidate.token_pos <= continuation.anchor_pos
                and is_compatible(candidate, continuation)
            ),
            key=lambda candidate: (
                -len(candidate.lineage_path),
                -candidate.token_pos,
                candidate.checkpoint_id,
            ),
        )
    )
    result: dict[str, int | None] = {}
    for depth in GAP_DEPTHS:
        if len(historical) < depth:
            result[f"G{depth}"] = None
            continue
        gap = continuation.anchor_pos - historical[depth - 1].token_pos
        if gap < 0:
            raise ValueError("结构 recovery gap 不能为负")
        result[f"G{depth}"] = gap
    return result


def analyze_snapshot(snapshot: TraceSnapshot) -> dict[str, Any]:
    """计算单个 snapshot 的状态压力、exact-parent 与 gap envelope。"""
    workflow_count = len(snapshot.active_workflow_ids)
    candidate_count = len(snapshot.candidates)
    pending_count = len(snapshot.continuations)
    if min(workflow_count, candidate_count) <= 0:
        raise ValueError(
            f"{snapshot.snapshot_id} 的 W、N 必须均大于零："
            f"W={workflow_count}, N={candidate_count}, P={pending_count}"
        )

    exact_parent_by_continuation = {
        continuation.continuation_id: exact_parent_checkpoint_ids(
            continuation,
            snapshot.candidates,
        )
        for continuation in snapshot.continuations
    }
    exact_parent_ids = {
        checkpoint_id
        for identifiers in exact_parent_by_continuation.values()
        for checkpoint_id in identifiers
    }
    exact_parent_count = len(exact_parent_ids)
    budget_rows = []
    base_counts = {
        "candidate_relative": candidate_count,
        "pending_relative": pending_count,
        "exact_parent_relative": exact_parent_count,
    }
    for normalization, base_count in base_counts.items():
        for ratio in BUDGET_RATIOS:
            k = budget_k(base_count, ratio)
            budget_rows.append(
                {
                    "normalization": normalization,
                    "ratio": ratio,
                    "base_count": base_count,
                    "k": k,
                    "k_lt_w": k < workflow_count,
                    "k_lt_p": k < pending_count,
                    "k_lt_x": k < exact_parent_count,
                    "k_ge_w": k >= workflow_count,
                    "k_ge_p": k >= pending_count,
                    "k_ge_x": k >= exact_parent_count,
                }
            )
    gaps = {
        continuation.continuation_id: lineage_recovery_envelope(
            continuation,
            snapshot.candidates,
        )
        for continuation in snapshot.continuations
    }
    return {
        "snapshot_id": snapshot.snapshot_id,
        "scale": snapshot.scale,
        "provider": snapshot.time_domain,
        "w": workflow_count,
        "n": candidate_count,
        "p": pending_count,
        "x": exact_parent_count,
        "n_over_w": candidate_count / workflow_count,
        "n_over_p": _divide_or_none(candidate_count, pending_count),
        "n_over_x": _divide_or_none(candidate_count, exact_parent_count),
        "x_over_w": exact_parent_count / workflow_count,
        "x_over_p": _divide_or_none(exact_parent_count, pending_count),
        "exact_parent_by_continuation": exact_parent_by_continuation,
        "exact_parent_available_count": sum(
            bool(value) for value in exact_parent_by_continuation.values()
        ),
        "budget_rows": budget_rows,
        "recovery_envelope_by_continuation": gaps,
        "future_prefix_used": snapshot.future_prefix_used,
        "runtime_residency_inferred": snapshot.runtime_residency_inferred,
        "llm_level_branching_introduced": (
            snapshot.llm_level_branching_introduced
        ),
    }


def summarize_cohort(
    cohort: str,
    cutoff_tokens: int,
    source_counts: Mapping[str, Any],
    snapshots: Sequence[TraceSnapshot],
    analyses: Sequence[Mapping[str, Any]],
    empty_strata: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """聚合一个累计 context cohort 的结构统计。"""
    if not snapshots or len(snapshots) != len(analyses):
        raise ValueError("cohort snapshots 与 analyses 必须非空且一一对应")
    fields = ("w", "n", "p", "x")
    ratio_fields = (
        "n_over_w",
        "n_over_p",
        "n_over_x",
        "x_over_w",
        "x_over_p",
    )
    unique_runs = {
        workflow_id
        for snapshot in snapshots
        for workflow_id in snapshot.active_workflow_ids
    }
    total_pending = sum(int(item["p"]) for item in analyses)
    exact_available = sum(
        int(item["exact_parent_available_count"]) for item in analyses
    )
    gap_values = _collect_gap_values(analyses)
    budget_summary = _summarize_budgets(analyses)
    return {
        "cohort": cohort,
        "cutoff_tokens": cutoff_tokens,
        "eligible_runs": int(source_counts["eligible_runs"]),
        "overlapping_runs": int(source_counts["overlapping_runs"]),
        "provider_distribution": {
            "claude": int(source_counts["claude_runs"]),
            "codex": int(source_counts["codex_runs"]),
        },
        "snapshot_count": len(snapshots),
        "selected_unique_runs": len(unique_runs),
        "scale_snapshot_counts": {
            scale: sum(snapshot.scale == scale for snapshot in snapshots)
            for scale in SCALE_ORDER
        },
        "empty_strata": tuple(empty_strata),
        "structure": {
            field.upper(): _distribution(
                [int(item[field]) for item in analyses],
                include_p99=False,
            )
            for field in fields
        },
        "state_pressure_ratios": {
            field.replace("_over_", "/").upper(): _distribution(
                [
                    float(item[field])
                    for item in analyses
                    if item[field] is not None
                ],
                include_p99=False,
            )
            for field in ratio_fields
        },
        "zero_pending_snapshot_count": sum(
            int(item["p"]) == 0 for item in analyses
        ),
        "exact_parent_availability_ratio": (
            None if total_pending == 0 else exact_available / total_pending
        ),
        "exact_parent_unavailability_reason": (
            None
            if exact_available == total_pending
            else "同 workflow、兼容 lineage 且 token_pos 等于 anchor 的候选缺失"
        ),
        "budget_normalization": budget_summary,
        "recovery_gap_envelope": _summarize_gap_values(gap_values),
        "profiler_structural_coverage": _profiler_coverage(gap_values),
        "integrity": {
            "future_field_leakage_violations": sum(
                bool(item["future_prefix_used"]) for item in analyses
            ),
            "runtime_residency_inference_violations": sum(
                bool(item["runtime_residency_inferred"])
                for item in analyses
            ),
            "synthetic_concurrency_violations": sum(
                not snapshot.trace_observed_concurrency
                for snapshot in snapshots
            ),
            "llm_level_branching_violations": sum(
                bool(item["llm_level_branching_introduced"])
                for item in analyses
            ),
        },
    }


def construct_context_pressure(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """只读构造四个累计 cohort 的结构审计，不运行任何 policy 或 Phi。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    cohort_payloads = []
    all_snapshots: dict[str, tuple[TraceSnapshot, ...]] = {}
    all_analyses: dict[str, tuple[dict[str, Any], ...]] = {}
    try:
        for cohort, cutoff_tokens in CONTEXT_COHORTS:
            source_counts = _single_row(
                connection,
                _COHORT_COUNTS_SQL,
                (cutoff_tokens,),
            )
            event_rows = fetch_dicts(
                connection,
                _SAMPLED_EVENTS_SQL,
                (cutoff_tokens, str(SAMPLING_SEED)),
            )
            events = tuple(
                _context_event_from_row(cohort, cutoff_tokens, row)
                for row in event_rows
            )
            snapshots = []
            for item in events:
                event = item.event
                known_rows = fetch_dicts(
                    connection,
                    _KNOWN_STATE_SQL,
                    (
                        cutoff_tokens,
                        event.observed_at,
                        event.observed_at,
                        event.observed_at,
                        event.observed_at,
                        event.provider,
                    ),
                )
                snapshots.append(_snapshot_from_rows(item, known_rows))
            snapshot_tuple = tuple(snapshots)
            analyses = tuple(analyze_snapshot(item) for item in snapshot_tuple)
            empty_strata = _empty_strata(cutoff_tokens, events)
            summary = summarize_cohort(
                cohort,
                cutoff_tokens,
                source_counts,
                snapshot_tuple,
                analyses,
                empty_strata,
            )
            all_snapshots[cohort] = snapshot_tuple
            all_analyses[cohort] = analyses
            cohort_payloads.append(
                {
                    "summary": summary,
                    "snapshots": tuple(
                        {
                            "event": _event_to_dict(event),
                            "snapshot": _snapshot_structure_to_dict(snapshot),
                            "analysis": analysis,
                        }
                        for event, snapshot, analysis in zip(
                            events,
                            snapshot_tuple,
                            analyses,
                        )
                    ),
                }
            )
    finally:
        connection.close()

    summaries = {
        item["summary"]["cohort"]: item["summary"]
        for item in cohort_payloads
    }
    nested_violations = sum(
        summaries[left]["eligible_runs"] > summaries[right]["eligible_runs"]
        for (left, _), (right, _) in zip(
            CONTEXT_COHORTS,
            CONTEXT_COHORTS[1:],
        )
    )
    coverage_gain = _context_coverage_gain(summaries)
    diagnosis = _diagnose(summaries, coverage_gain)
    total_integrity_violations = sum(
        sum(summary["integrity"].values()) for summary in summaries.values()
    )
    primary_envelope = summaries["C256"]["recovery_gap_envelope"]
    primary_coverage = summaries["C256"]["profiler_structural_coverage"]
    artifact = {
        "schema_version": "tracelab-context-pressure-v1",
        "source": {
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "access_mode": "read_only=True",
            "tables": ("rounds", "timing_events", "tool_calls"),
        },
        "frozen_semantics": {
            "agent_run_boundary": "current_user_message_count > 0",
            "strictly_closed_only": True,
            "known_anchor": "current_round.input_tokens_total",
            "lineage": "按真实 round order 构造的线性 lineage",
            "concurrency": "同 provider 内 trace-observed overlap",
            "sampling_seed": SAMPLING_SEED,
            "sampling_strata": "scale × provider × 完整 run context bucket",
            "full_run_context_role": "仅用于离线累计 cohort 与采样分层",
            "future_round_used_as_online_input": False,
            "synthetic_concurrency": False,
            "kvflow_steps_to_execution": 1,
        },
        "cohort_definitions": tuple(
            {"cohort": cohort, "cutoff_tokens": cutoff}
            for cohort, cutoff in CONTEXT_COHORTS
        ),
        "budget_definitions": {
            "ratios": BUDGET_RATIOS,
            "candidate_relative": "max(1, floor(N * ratio))",
            "pending_relative": "max(1, floor(P * ratio))",
            "exact_parent_relative": "max(1, floor(X * ratio))",
        },
        "recovery_gap_definition": {
            "depths": GAP_DEPTHS,
            "insufficient_history": "unavailable",
            "primary_reporting_cohort": "C256",
            "phi_called": False,
        },
        "cohorts": tuple(cohort_payloads),
        "context_coverage_gain": coverage_gain,
        "primary_recovery_gap_envelope": primary_envelope,
        "primary_profiler_structural_coverage": primary_coverage,
        "diagnosis": diagnosis,
        "validation": {
            "nested_cohort_violations": nested_violations,
            "future_or_synthetic_integrity_violations": (
                total_integrity_violations
            ),
            "policy_comparison_runs": 0,
            "phi_calls": 0,
            "gpu_runs": 0,
        },
    }
    return _json_value(artifact)


def render_report(artifact: Mapping[str, Any]) -> str:
    """生成以结果为先、定义和限制可复核的中文技术报告。"""
    summaries = _summary_by_cohort(artifact)
    diagnosis = artifact["diagnosis"]
    primary_envelope = artifact["primary_recovery_gap_envelope"]
    primary_coverage = artifact["primary_profiler_structural_coverage"]
    lines = [
        "# TraceLab 上下文覆盖与状态压力审计",
        "",
        "## 技术摘要",
        "",
        _diagnosis_summary(diagnosis),
        "",
        "本审计没有运行任何 checkpoint policy、Phi 或 GPU。所有快照只来自同 provider 内真实观测 overlap；完整 run 最大 context 只用于离线累计 cohort 与确定性采样分层，不进入 online anchor、pending 或 checkpoint value。",
        "",
        "## 更长 context cohort 增加了多少真实可用结构",
        "",
        "四个 cohort 均只包含严格闭合 Agent Runs。每个非空 `scale × provider × context bucket` 层固定选择一个 SHA-256 键最小的多轮 tool-call 时点；空层不补造。",
        "",
        "| Cohort | Eligible runs | Overlapping runs | Snapshots | Unique runs | Small | Medium | Large | Exact-parent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort, _ in CONTEXT_COHORTS:
        summary = summaries[cohort]
        scale = summary["scale_snapshot_counts"]
        lines.append(
            f"| {cohort} | {summary['eligible_runs']:,} | {summary['overlapping_runs']:,} | {summary['snapshot_count']} | {summary['selected_unique_runs']} | {scale['Small']} | {scale['Medium']} | {scale['Large']} | {_percent(summary['exact_parent_availability_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "累计 cohort 之间只保证 run membership 按 cutoff 嵌套；每个 cohort 会重新计算真实 concurrency，因此同一个 trigger bucket 的 scale 或入选时点可以随 active set 改变。",
            "",
            "| Context gain | Eligible runs | Snapshots | Medium+Large | Unique runs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for gain in artifact["context_coverage_gain"]:
        lines.append(
            f"| {gain['transition']} | {_change_text(gain['eligible_runs'])} | {_change_text(gain['snapshots'])} | {_change_text(gain['medium_large_snapshots'])} | {_change_text(gain['selected_unique_runs'])} |"
        )
    lines.extend(
        [
            "",
            "## 历史 candidate 数明显大于当前 exact-parent 需求",
            "",
            "W 是 active workflows，N 是历史逻辑 candidates，P 是 known pending，X 是它们所需的 distinct exact parents。P=0 或 X=0 时，相关除法明确记为 unavailable。",
            "",
            "| Cohort | Metric | Mean | Median | P90 | P95 | Max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort, _ in CONTEXT_COHORTS:
        summary = summaries[cohort]
        for metric in ("W", "N", "P", "X"):
            lines.append(
                _report_distribution_row(
                    cohort,
                    metric,
                    summary["structure"][metric],
                )
            )
    lines.extend(
        [
            "",
            "| Cohort | Ratio | Available snapshots | Mean | Median | P90 | P95 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort, _ in CONTEXT_COHORTS:
        for metric in ("N/W", "N/P", "N/X", "X/W", "X/P"):
            row = summaries[cohort]["state_pressure_ratios"][metric]
            lines.append(
                f"| {cohort} | {metric} | {row['count']} | {_decimal_or_na(row['mean'])} | {_decimal_or_na(row['median'])} | {_decimal_or_na(row['p90'])} | {_decimal_or_na(row['p95'])} |"
            )
    lines.extend(
        [
            "",
            "N 记录全部可知历史 checkpoint，而 X 只去重当前 pending 的 exact-parent 需求。两者的偏离决定 candidate-relative K 是否把历史长度误当成当前压力。",
            "",
            "## 三种 budget normalization 呈现不同压力",
            "",
            "`K>=X` 直接表示容量是否足以同时容纳所有 distinct exact parents，不代表任何 policy 能自动找到它们。以下结果逐 cohort 报告，没有用历史较长的 cohort 淹没短 context cohort。",
            "",
            "| Cohort | Normalization | Ratio | Median K | P90 K | Max K | K<W | K<P | K<X | K>=W | K>=P | K>=X |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort, _ in CONTEXT_COHORTS:
        for normalization in BUDGET_NORMALIZATIONS:
            for ratio in BUDGET_RATIOS:
                row = summaries[cohort]["budget_normalization"][
                    normalization
                ][_ratio_label(ratio)]
                lines.append(
                    f"| {cohort} | {_normalization_label(normalization)} | {_ratio_label(ratio)} | {row['k']['median']:.0f} | {row['k']['p90']:.0f} | {row['k']['max']:.0f} | {_percent(row['k_lt_w_fraction'])} | {_percent(row['k_lt_p_fraction'])} | {_percent(row['k_lt_x_fraction'])} | {_percent(row['k_ge_w_fraction'])} | {_percent(row['k_ge_p_fraction'])} | {_percent(row['k_ge_x_fraction'])} |"
                )
    lines.extend(
        [
            "",
            "## C256 的真实 lineage gap envelope",
            "",
            "为避免累计 cohort 重复计数，主 gap envelope 只报告最宽的 C256 snapshots。G1/G2/G4/G8 分别表示 exact parent 丢失后回退到第 1、2、4、8 个兼容历史 checkpoint；历史不足记为 unavailable。",
            "",
            "| Gap | Available | Median | P90 | P95 | P99 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("G1", "G2", "G4", "G8"):
        row = primary_envelope[label]
        lines.append(
            f"| {label} | {row['available_count']} | {_number_or_na(row['median'])} | {_number_or_na(row['p90'])} | {_number_or_na(row['p95'])} | {_number_or_na(row['p99'])} | {_number_or_na(row['max'])} |"
        )
    lines.extend(
        [
            "",
            "所有 available gaps 的累计覆盖如下；分母不包含 unavailable history。",
            "",
            "| Gap 上限 | 累计比例 |",
            "|---:|---:|",
        ]
    )
    all_gap_coverage = primary_envelope["all_available"][
        "cumulative_coverage"
    ]
    for threshold in GAP_THRESHOLDS:
        lines.append(
            f"| <={threshold // 1024}K | {_percent(all_gap_coverage[f'<={threshold}'])} |"
        )
    lines.extend(
        [
            "",
            "## Profiler coverage 只描述结构有效域",
            "",
            "这些比例只回答 structural gaps 是否落在指定 token 上限内；没有估计 64K/128K latency，也没有拟合或修改 Phi。",
            "",
            "| Profiler range | G1 | G2 | G4 | G8 | Overall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold in (32_768, 65_536, 131_072):
        row = primary_coverage[f"<={threshold}"]
        lines.append(
            f"| {threshold // 1024}K | {_percent(row['G1'])} | {_percent(row['G2'])} | {_percent(row['G4'])} | {_percent(row['G8'])} | {_percent(row['overall'])} |"
        )
    lines.extend(
        [
            "",
            "## 诊断与下一步边界",
            "",
        ]
    )
    for key, label in (
        ("profiler_64k_extension", "64K profiler extension value"),
        ("profiler_128k_extension", "128K profiler extension value"),
        ("candidate_relative_budget", "Candidate-relative budget"),
        ("pending_relative_budget", "Pending-relative budget"),
        ("exact_parent_relative_budget", "Exact-parent-relative budget"),
        ("tracelab_policy_suitability", "TraceLab policy suitability"),
    ):
        row = diagnosis[key]
        lines.append(f"- {label}: **{row['rating']}**。{row['reason']}")
    lines.extend(
        [
            "",
            f"推荐下一步：{diagnosis['recommended_next_step']}",
            "",
            "限制：TraceLab 没有显式 LLM-level DAG、token IDs 或 runtime residency truth；KVFlow 的 `steps_to_execution` 仍固定为 1。当前结果是结构审计，不是 policy comparison 或 runtime latency 结果。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    artifact: Mapping[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """保存结构审计 JSON 与中文 Markdown 技术报告。"""
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(artifact), encoding="utf-8")


def _collect_gap_values(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    values = {f"G{depth}": [] for depth in GAP_DEPTHS}
    for analysis in analyses:
        for envelope in analysis["recovery_envelope_by_continuation"].values():
            for label, gap in envelope.items():
                if gap is not None:
                    values[label].append(int(gap))
    return values


def _summarize_gap_values(
    gap_values: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    result = {}
    all_values = []
    for label in ("G1", "G2", "G4", "G8"):
        values = list(gap_values[label])
        all_values.extend(values)
        distribution = _distribution(values, include_p99=True)
        result[label] = {
            "available_count": len(values),
            **distribution,
            "cumulative_coverage": {
                f"<={threshold}": _coverage(values, threshold)
                for threshold in GAP_THRESHOLDS
            },
        }
    result["all_available"] = {
        "available_count": len(all_values),
        **_distribution(all_values, include_p99=True),
        "cumulative_coverage": {
            f"<={threshold}": _coverage(all_values, threshold)
            for threshold in GAP_THRESHOLDS
        },
    }
    return result


def _profiler_coverage(
    gap_values: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, float]]:
    all_values = [
        value
        for label in ("G1", "G2", "G4", "G8")
        for value in gap_values[label]
    ]
    return {
        f"<={threshold}": {
            **{
                label: _coverage(gap_values[label], threshold)
                for label in ("G1", "G2", "G4", "G8")
            },
            "overall": _coverage(all_values, threshold),
        }
        for threshold in (32_768, 65_536, 131_072)
    }


def _summarize_budgets(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for normalization in BUDGET_NORMALIZATIONS:
        result[normalization] = {}
        for ratio in BUDGET_RATIOS:
            rows = tuple(
                row
                for analysis in analyses
                for row in analysis["budget_rows"]
                if row["normalization"] == normalization
                and row["ratio"] == ratio
            )
            result[normalization][_ratio_label(ratio)] = {
                "snapshot_count": len(rows),
                "k_values": tuple(int(row["k"]) for row in rows),
                "k": _distribution(
                    [int(row["k"]) for row in rows],
                    include_p99=False,
                ),
                "k_lt_w_fraction": _true_fraction(rows, "k_lt_w"),
                "k_lt_p_fraction": _true_fraction(rows, "k_lt_p"),
                "k_lt_x_fraction": _true_fraction(rows, "k_lt_x"),
                "k_ge_w_fraction": _true_fraction(rows, "k_ge_w"),
                "k_ge_p_fraction": _true_fraction(rows, "k_ge_p"),
                "k_ge_x_fraction": _true_fraction(rows, "k_ge_x"),
            }
    return result


def _merge_budget_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    k_values = [int(value) for row in rows for value in row["k_values"]]
    count = sum(int(row["snapshot_count"]) for row in rows)
    if count <= 0:
        raise ValueError("合并预算统计至少需要一个 snapshot")
    result = {"k": _distribution(k_values, include_p99=False)}
    for field in (
        "k_lt_w_fraction",
        "k_lt_p_fraction",
        "k_lt_x_fraction",
        "k_ge_w_fraction",
        "k_ge_p_fraction",
        "k_ge_x_fraction",
    ):
        result[field] = sum(
            float(row[field]) * int(row["snapshot_count"]) for row in rows
        ) / count
    return result


def _context_coverage_gain(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for (left, _), (right, _) in zip(
        CONTEXT_COHORTS,
        CONTEXT_COHORTS[1:],
    ):
        before = summaries[left]
        after = summaries[right]
        before_medium_large = (
            before["scale_snapshot_counts"]["Medium"]
            + before["scale_snapshot_counts"]["Large"]
        )
        after_medium_large = (
            after["scale_snapshot_counts"]["Medium"]
            + after["scale_snapshot_counts"]["Large"]
        )
        rows.append(
            {
                "transition": f"{left}->{right}",
                "eligible_runs": _change(
                    before["eligible_runs"], after["eligible_runs"]
                ),
                "snapshots": _change(
                    before["snapshot_count"], after["snapshot_count"]
                ),
                "medium_large_snapshots": _change(
                    before_medium_large, after_medium_large
                ),
                "selected_unique_runs": _change(
                    before["selected_unique_runs"],
                    after["selected_unique_runs"],
                ),
            }
        )
    return tuple(rows)


def _diagnose(
    summaries: Mapping[str, Mapping[str, Any]],
    coverage_gain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gain_by_transition = {
        row["transition"]: row for row in coverage_gain
    }
    gain_64 = gain_by_transition["C32->C64"]
    gain_128 = gain_by_transition["C64->C128"]
    rating_64 = _extension_rating(gain_64)
    rating_128 = _extension_rating(gain_128)
    candidate_50 = [
        summaries[cohort]["budget_normalization"]["candidate_relative"][
            "50%"
        ]["k_ge_x_fraction"]
        for cohort, _ in CONTEXT_COHORTS
    ]
    candidate_75 = [
        summaries[cohort]["budget_normalization"]["candidate_relative"][
            "75%"
        ]["k_ge_x_fraction"]
        for cohort, _ in CONTEXT_COHORTS
    ]
    candidate_not_representative = (
        mean(candidate_50) >= 0.90 and mean(candidate_75) >= 0.90
    )
    all_exact = all(
        summary["exact_parent_availability_ratio"] == 1.0
        for summary in summaries.values()
    )
    return {
        "decision_rules": {
            "material_context_gain": (
                "eligible runs 与 selected unique runs 的相对增长均至少 20% "
                "则为 YES；仅一项达到则为 WEAK；否则为 NO"
            ),
            "candidate_pressure": (
                "若 50% 与 75% candidate-relative budget 的 K>=X "
                "跨 cohort 平均比例均至少 90%，则判为 NO"
            ),
        },
        "profiler_64k_extension": {
            "rating": rating_64,
            "reason": _extension_reason(gain_64),
        },
        "profiler_128k_extension": {
            "rating": rating_128,
            "reason": _extension_reason(gain_128),
        },
        "candidate_relative_budget": {
            "rating": "NO" if candidate_not_representative else "WEAK",
            "reason": (
                "50% 与 75% 档位在绝大多数 snapshots 已足以容纳全部 X，"
                "N 受历史长度放大，弱化了当前 pending state pressure。"
                if candidate_not_representative
                else "candidate 历史长度仍影响 K，且当前证据不足以判为代表性良好。"
            ),
        },
        "pending_relative_budget": {
            "rating": "WEAK",
            "reason": (
                "P 能描述当前 known demand，但 TraceLab 没有 LLM-level branching，"
                "本样本中非零 P 与 X 完全相同，无法验证共享 exact parent 时的去重语义；"
                "P=0 时公式仍因下限产生 K=1。"
            ),
        },
        "exact_parent_relative_budget": {
            "rating": "YES" if all_exact else "WEAK",
            "reason": (
                "X 直接度量保护当前 pending 所需的 distinct exact-parent 状态，"
                "不会被历史 checkpoint 数量放大。"
            ),
        },
        "tracelab_policy_suitability": {
            "rating": "WEAK",
            "reason": (
                "真实 overlap、线性 history 与 exact parents 足以形成结构比较；"
                "但缺少显式 DAG、token IDs 和 runtime residency truth。"
            ),
        },
        "recommended_next_step": (
            "先基于本审计结果单独冻结 demand-relative budget protocol；"
            "仅在独立 profiler 扩展后才把超过其有效域的 cohort 纳入 Phi-based comparison。"
        ),
    }


def _extension_rating(gain: Mapping[str, Any]) -> str:
    material = sum(
        gain[field]["relative"] is not None
        and gain[field]["relative"] >= MATERIAL_GAIN_THRESHOLD
        for field in ("eligible_runs", "selected_unique_runs")
    )
    if material == 2:
        return "YES"
    if material == 1:
        return "WEAK"
    return "NO"


def _extension_reason(gain: Mapping[str, Any]) -> str:
    runs = gain["eligible_runs"]
    unique = gain["selected_unique_runs"]
    return (
        f"eligible runs 增加 {runs['absolute']:+d}（{_percent_or_na(runs['relative'])}），"
        f"selected unique runs 增加 {unique['absolute']:+d}（{_percent_or_na(unique['relative'])}）。"
    )


def _context_event_from_row(
    cohort: str,
    cutoff_tokens: int,
    row: Mapping[str, Any],
) -> ContextSnapshotEvent:
    scale = str(row["scale"])
    provider = str(row["provider"])
    bucket = str(row["context_bucket"])
    safe_bucket = (
        bucket.replace("<=", "le-").replace(">", "gt-").replace("K", "k")
    )
    return ContextSnapshotEvent(
        cohort=cohort,
        cutoff_tokens=cutoff_tokens,
        event=SampledSnapshotEvent(
            snapshot_id=(
                f"{cohort.lower()}-{scale.lower()}-{provider}-{safe_bucket}"
            ),
            scale=scale,
            provider=provider,
            context_bucket=bucket,
            trigger_session_id=str(row["session_id"]),
            trigger_run_ordinal=int(row["run_ordinal"]),
            trigger_round_pk=int(row["round_pk"]),
            observed_at=row["observed_at"],
            trace_observed_active_runs=int(row["active_run_count"]),
        ),
    )


def _snapshot_from_rows(
    item: ContextSnapshotEvent,
    rows: Sequence[Mapping[str, Any]],
) -> TraceSnapshot:
    event = item.event
    active_workflow_ids = tuple(
        sorted(
            {
                workflow_id_for(
                    str(row["provider"]),
                    str(row["session_id"]),
                    int(row["run_ordinal"]),
                )
                for row in rows
            }
        )
    )
    if len(active_workflow_ids) != event.trace_observed_active_runs:
        raise ValueError(
            f"active run 数与真实 overlap 统计不一致：{event.snapshot_id}"
        )
    completed_rounds = []
    pending_rounds = []
    for row in rows:
        workflow_id = workflow_id_for(
            str(row["provider"]),
            str(row["session_id"]),
            int(row["run_ordinal"]),
        )
        if row["known_completion_time"] is not None:
            completed_rounds.append(
                CompletedRoundFact(
                    workflow_id=workflow_id,
                    round_pk=int(row["round_pk"]),
                    round_index=int(row["round_index"]),
                    run_position=int(row["run_position"]),
                    input_tokens_total=int(row["input_tokens_total"]),
                    current_prefix_tokens=int(row["prefix_tokens"]),
                    known_at_time=row["known_completion_time"],
                )
            )
        if (
            bool(row["is_latest_started_round"])
            and int(row["observed_tool_call_count"]) > 0
        ):
            pending_rounds.append(
                PendingRoundFact(
                    workflow_id=workflow_id,
                    round_pk=int(row["round_pk"]),
                    round_index=int(row["round_index"]),
                    run_position=int(row["run_position"]),
                    input_tokens_total=int(row["input_tokens_total"]),
                    current_prefix_tokens=int(row["prefix_tokens"]),
                    known_at_time=row["first_observed_tool_time"],
                    observed_tool_call_ids=tuple(
                        str(value) for value in row["observed_tool_call_ids"]
                    ),
                )
            )
    return build_trace_snapshot(
        snapshot_id=event.snapshot_id,
        scale=event.scale,
        time_domain=event.provider,
        observed_at=event.observed_at,
        active_workflow_ids=active_workflow_ids,
        completed_rounds=completed_rounds,
        pending_rounds=pending_rounds,
    )


def _empty_strata(
    cutoff_tokens: int,
    events: Sequence[ContextSnapshotEvent],
) -> tuple[dict[str, str], ...]:
    allowed_buckets = tuple(
        bucket
        for bucket in CONTEXT_BUCKET_ORDER
        if bucket != ">256K" and _bucket_upper_bound(bucket) <= cutoff_tokens
    )
    expected = {
        (scale, provider, bucket)
        for scale in SCALE_ORDER
        for provider in ("claude", "codex")
        for bucket in allowed_buckets
    }
    observed = {
        (item.event.scale, item.event.provider, item.event.context_bucket)
        for item in events
    }
    return tuple(
        {
            "scale": scale,
            "provider": provider,
            "context_bucket": bucket,
        }
        for scale, provider, bucket in sorted(
            expected - observed,
            key=lambda value: (
                SCALE_ORDER.index(value[0]),
                value[1],
                CONTEXT_BUCKET_ORDER.index(value[2]),
            ),
        )
    )


def _bucket_upper_bound(bucket: str) -> int:
    values = {
        "<=32K": 32_768,
        "32K-64K": 65_536,
        "64K-128K": 131_072,
        "128K-256K": 262_144,
    }
    if bucket not in values:
        raise ValueError(f"不支持的累计 bucket：{bucket}")
    return values[bucket]


def _event_rank(item: ContextSnapshotEvent) -> tuple[str, int]:
    event = item.event
    return (
        deterministic_sample_key(
            seed=SAMPLING_SEED,
            provider=event.provider,
            context_bucket=event.context_bucket,
            scale=event.scale,
            session_id=event.trigger_session_id,
            run_ordinal=event.trigger_run_ordinal,
            round_pk=event.trigger_round_pk,
        ),
        event.trigger_round_pk,
    )


def _event_to_dict(item: ContextSnapshotEvent) -> dict[str, Any]:
    return _json_value(asdict(item))


def _snapshot_structure_to_dict(snapshot: TraceSnapshot) -> dict[str, Any]:
    positions_by_workflow: dict[str, list[int]] = {}
    for candidate in snapshot.candidates:
        positions_by_workflow.setdefault(candidate.workflow_id, []).append(
            candidate.token_pos
        )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "scale": snapshot.scale,
        "provider": snapshot.time_domain,
        "observed_at": snapshot.observed_at,
        "trace_observed_concurrency": snapshot.trace_observed_concurrency,
        "active_workflow_ids": snapshot.active_workflow_ids,
        "candidate_count": len(snapshot.candidates),
        "pending_count": len(snapshot.continuations),
        "candidate_positions_by_workflow": {
            workflow_id: tuple(values)
            for workflow_id, values in sorted(positions_by_workflow.items())
        },
        "continuation_anchors": tuple(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "anchor_pos": continuation.anchor_pos,
            }
            for continuation in snapshot.continuations
        ),
        "future_prefix_used": snapshot.future_prefix_used,
        "runtime_residency_inferred": snapshot.runtime_residency_inferred,
        "llm_level_branching_introduced": (
            snapshot.llm_level_branching_introduced
        ),
    }


def _summary_by_cohort(
    artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        item["summary"]["cohort"]: item["summary"]
        for item in artifact["cohorts"]
    }


def _change(before: int, after: int) -> dict[str, Any]:
    return {
        "before": before,
        "after": after,
        "absolute": after - before,
        "relative": None if before == 0 else (after - before) / before,
    }


def _divide_or_none(numerator: int, denominator: int) -> float | None:
    """对零分母明确返回 unavailable。"""
    if denominator == 0:
        return None
    return numerator / denominator


def _coverage(values: Sequence[int], threshold: int) -> float:
    if not values:
        return 0.0
    return sum(value <= threshold for value in values) / len(values)


def _true_fraction(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    if not rows:
        raise ValueError("比例统计不能使用空 rows")
    return sum(bool(row[field]) for row in rows) / len(rows)


def _distribution(
    values: Sequence[int | float],
    *,
    include_p99: bool,
) -> dict[str, Any]:
    if not values:
        result: dict[str, Any] = {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
        if include_p99:
            result["p99"] = None
        return result
    ordered = sorted(values)
    result = {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": _quantile_disc(ordered, 0.90),
        "p95": _quantile_disc(ordered, 0.95),
        "max": ordered[-1],
    }
    if include_p99:
        result["p99"] = _quantile_disc(ordered, 0.99)
    return result


def _quantile_disc(
    ordered: Sequence[int | float],
    probability: float,
) -> int | float:
    if not ordered:
        raise ValueError("quantile 输入不能为空")
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _single_row(
    connection,
    query: str,
    parameters: Sequence[Any],
) -> dict[str, Any]:
    rows = fetch_dicts(connection, query, parameters)
    if len(rows) != 1:
        raise ValueError("查询应恰好返回一行")
    return rows[0]


def _ratio_label(ratio: float) -> str:
    return f"{int(ratio * 100)}%"


def _normalization_label(value: str) -> str:
    return {
        "candidate_relative": "Candidate-relative",
        "pending_relative": "Pending-relative",
        "exact_parent_relative": "Exact-parent-relative",
    }[value]


def _change_text(row: Mapping[str, Any]) -> str:
    """把绝对和相对增长格式化为一个审计单元格。"""
    return f"{row['absolute']:+d} ({_percent_or_na(row['relative'])})"


def _report_distribution_row(
    cohort: str,
    metric: str,
    row: Mapping[str, Any],
) -> str:
    """格式化 W、N、P、X 的完整分布。"""
    return (
        f"| {cohort} | {metric} | {_decimal_or_na(row['mean'])} | "
        f"{_decimal_or_na(row['median'])} | {_decimal_or_na(row['p90'])} | "
        f"{_decimal_or_na(row['p95'])} | {_decimal_or_na(row['max'])} |"
    )


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _percent_or_na(value: float | None) -> str:
    return "N/A" if value is None else _percent(value)


def _number_or_na(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}"


def _decimal_or_na(value: int | float | None) -> str:
    """按三位小数格式化可缺失的结构指标。"""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _diagnosis_summary(diagnosis: Mapping[str, Any]) -> str:
    return (
        f"64K profiler 扩展价值为 {diagnosis['profiler_64k_extension']['rating']}，"
        f"128K 扩展价值为 {diagnosis['profiler_128k_extension']['rating']}；"
        f"candidate-relative budget 的结构代表性为 "
        f"{diagnosis['candidate_relative_budget']['rating']}。"
        "核心风险是 N 会随历史 checkpoint 数累积，而当前 demand 由 distinct exact parents X 决定。"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


_COHORT_COUNTS_SQL = """
WITH ordered AS (
    SELECT
        r.*,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds r
), assigned AS (
    SELECT * FROM ordered WHERE run_ordinal > 0
), raw_runs AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        max(a.input_tokens_total) AS max_input_tokens_total,
        min(t.timestamp) AS start_time,
        max(t.timestamp) AS end_time
    FROM assigned a
    JOIN timing_events t USING (round_pk)
    GROUP BY a.provider, a.session_id, a.run_ordinal
), classified AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM raw_runs
), eligible AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= ?
), marked AS (
    SELECT
        current.*,
        EXISTS (
            SELECT 1
            FROM eligible other
            WHERE other.provider = current.provider
              AND NOT (
                  other.session_id = current.session_id
                  AND other.run_ordinal = current.run_ordinal
              )
              AND other.start_time <= current.end_time
              AND other.end_time >= current.start_time
        ) AS has_overlap
    FROM eligible current
)
SELECT
    count(*) AS eligible_runs,
    count(*) FILTER (WHERE has_overlap) AS overlapping_runs,
    count(*) FILTER (WHERE provider = 'claude') AS claude_runs,
    count(*) FILTER (WHERE provider = 'codex') AS codex_runs
FROM marked
"""


_SAMPLED_EVENTS_SQL = """
WITH ordered AS (
    SELECT
        r.*,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds r
), assigned AS (
    SELECT * FROM ordered WHERE run_ordinal > 0
), raw_runs AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        count(DISTINCT a.round_pk) AS round_count,
        max(a.input_tokens_total) AS max_input_tokens_total,
        min(t.timestamp) AS start_time,
        max(t.timestamp) AS end_time
    FROM assigned a
    JOIN timing_events t USING (round_pk)
    GROUP BY a.provider, a.session_id, a.run_ordinal
), classified AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM raw_runs
), eligible AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= ?
), tool_events AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        max(tool.emitted_at) AS observed_at
    FROM assigned a
    JOIN tool_calls tool USING (round_pk)
    JOIN eligible run USING (provider, session_id, run_ordinal)
    WHERE run.round_count >= 2 AND tool.emitted_at IS NOT NULL
    GROUP BY a.provider, a.session_id, a.run_ordinal, a.round_pk
), counted AS (
    SELECT
        event.provider,
        event.session_id,
        event.run_ordinal,
        event.round_pk,
        event.observed_at,
        trigger.max_input_tokens_total,
        count(active.session_id) AS active_run_count
    FROM tool_events event
    JOIN eligible trigger USING (provider, session_id, run_ordinal)
    JOIN eligible active
      ON active.provider = event.provider
     AND active.start_time <= event.observed_at
     AND active.end_time >= event.observed_at
    GROUP BY
        event.provider, event.session_id, event.run_ordinal,
        event.round_pk, event.observed_at, trigger.max_input_tokens_total
), labeled AS (
    SELECT
        *,
        CASE
            WHEN active_run_count BETWEEN 2 AND 4 THEN 'Small'
            WHEN active_run_count BETWEEN 5 AND 8 THEN 'Medium'
            WHEN active_run_count >= 9 THEN 'Large'
        END AS scale,
        CASE
            WHEN max_input_tokens_total <= 32768 THEN '<=32K'
            WHEN max_input_tokens_total <= 65536 THEN '32K-64K'
            WHEN max_input_tokens_total <= 131072 THEN '64K-128K'
            WHEN max_input_tokens_total <= 262144 THEN '128K-256K'
            ELSE '>256K'
        END AS context_bucket
    FROM counted
), ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY scale, provider, context_bucket
            ORDER BY sha256(
                concat_ws(
                    '|', ?, provider, context_bucket, scale,
                    session_id, run_ordinal::VARCHAR, round_pk::VARCHAR
                )
            ), round_pk
        ) AS sample_rank
    FROM labeled
    WHERE scale IS NOT NULL
)
SELECT
    scale,
    provider,
    context_bucket,
    session_id,
    run_ordinal,
    round_pk,
    observed_at,
    active_run_count
FROM ranked
WHERE sample_rank = 1
ORDER BY
    CASE scale WHEN 'Small' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    provider,
    CASE context_bucket
        WHEN '<=32K' THEN 1
        WHEN '32K-64K' THEN 2
        WHEN '64K-128K' THEN 3
        ELSE 4
    END
"""


_KNOWN_STATE_SQL = """
WITH ordered AS (
    SELECT
        r.*,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds r
), assigned AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY session_id, run_ordinal ORDER BY round_index
        ) - 1 AS run_position
    FROM ordered
    WHERE run_ordinal > 0
), raw_runs AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        max(a.input_tokens_total) AS max_input_tokens_total,
        min(t.timestamp) AS start_time,
        max(t.timestamp) AS end_time
    FROM assigned a
    JOIN timing_events t USING (round_pk)
    GROUP BY a.provider, a.session_id, a.run_ordinal
), classified AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM raw_runs
), eligible AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= ?
), observed_round_events AS (
    SELECT
        round_pk,
        min(timestamp) AS started_at,
        min(timestamp) FILTER (
            WHERE event_type IN ('tool_call', 'usage_report')
        ) AS completion_marker_at
    FROM timing_events
    WHERE timestamp <= ?
    GROUP BY round_pk
), observed_tools AS (
    SELECT
        round_pk,
        count(*) AS observed_tool_call_count,
        list(tool_call_id ORDER BY emitted_at, tool_call_id)
            AS observed_tool_call_ids,
        min(emitted_at) AS first_observed_tool_time
    FROM tool_calls
    WHERE emitted_at <= ?
    GROUP BY round_pk
), observed_rounds AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        a.round_index,
        a.run_position,
        a.input_tokens_total,
        a.prefix_tokens,
        event.started_at,
        event.completion_marker_at,
        coalesce(tool.observed_tool_call_count, 0)
            AS observed_tool_call_count,
        coalesce(tool.observed_tool_call_ids, [])
            AS observed_tool_call_ids,
        tool.first_observed_tool_time
    FROM assigned a
    JOIN eligible run USING (provider, session_id, run_ordinal)
    JOIN observed_round_events event USING (round_pk)
    LEFT JOIN observed_tools tool USING (round_pk)
    WHERE run.start_time <= ?
      AND run.end_time >= ?
      AND run.provider = ?
), sequenced AS (
    SELECT
        *,
        lead(started_at) OVER (
            PARTITION BY provider, session_id, run_ordinal
            ORDER BY run_position
        ) AS next_observed_round_started_at,
        run_position = max(run_position) OVER (
            PARTITION BY provider, session_id, run_ordinal
        ) AS is_latest_started_round
    FROM observed_rounds
)
SELECT
    *,
    coalesce(completion_marker_at, next_observed_round_started_at)
        AS known_completion_time
FROM sequenced
ORDER BY provider, session_id, run_ordinal, run_position
"""


def main(argv: Sequence[str] | None = None) -> int:
    """执行只读结构审计并写入 JSON 与 Markdown。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="外部 TraceLab DuckDB 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="结构审计 JSON 输出路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown 技术报告输出路径",
    )
    arguments = parser.parse_args(argv)
    artifact = construct_context_pressure(arguments.database)
    write_artifacts(artifact, arguments.output, arguments.report)
    print(
        json.dumps(
            {
                "cohorts": {
                    item["summary"]["cohort"]: item["summary"]
                    for item in artifact["cohorts"]
                },
                "diagnosis": artifact["diagnosis"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
