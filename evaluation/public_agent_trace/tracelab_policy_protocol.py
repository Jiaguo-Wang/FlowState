#!/usr/bin/env python3
"""冻结 TraceLab 离线 policy evaluation 的有效域、预算与元数据协议。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CHECKPOINT_MEMORY_BYTES,
    CompletedRoundFact,
    PendingRoundFact,
    RETENTION_RATIOS,
    SAMPLING_SEED,
    TraceSnapshot,
    build_trace_snapshot,
    workflow_id_for,
)
from evaluation.sota_metadata import build_marconi_flop_saved
from flowstate.state_catalog import CheckpointCandidate, is_lineage_prefix
from flowstate.workflow import PendingContinuation


PROFILER_MAX_GAP_TOKENS = 32_768
MARCONI_ALPHA = 1.0
DEFAULT_PROTOCOL_PATH = Path(__file__).with_name(
    "tracelab_policy_protocol.json"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    "TRACELAB_POLICY_PROTOCOL.md"
)
SCALE_ORDER = ("Small", "Medium", "Large")
POLICIES_FOR_STEP_10D = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
)
PRIMARY_METRICS = (
    "total_predicted_recovery_cost_ms",
    "mean_predicted_recovery_cost_ms_per_pending",
    "total_recovery_gap_tokens",
    "mean_recovery_gap_tokens",
)
SECONDARY_METRICS = (
    "executable_hit_ratio",
    "max_recovery_gap_tokens",
    "p95_recovery_gap_tokens",
)


@dataclass(frozen=True)
class ProtocolSnapshotEvent:
    """由 provider 与 trace-observed concurrency scale 分层选出的时点。"""

    snapshot_id: str
    scale: str
    provider: str
    trigger_session_id: str
    trigger_run_ordinal: int
    trigger_round_pk: int
    observed_at: datetime
    active_run_count: int


@dataclass(frozen=True)
class TraceCheckpointRecency:
    """供 Global-LRU 与 Marconi 共用的确定性 checkpoint recency。"""

    checkpoint_id: str
    creation_order: int
    last_access_order: int
    known_at_time: datetime


@dataclass(frozen=True)
class SnapshotPolicyMetadata:
    """只由 snapshot 当前与历史信息构造的冻结 policy metadata。"""

    steps_to_execution_by_continuation: tuple[tuple[str, int], ...]
    checkpoint_recency: tuple[TraceCheckpointRecency, ...]
    last_access_by_checkpoint: tuple[tuple[str, float], ...]
    marconi_flop_saved_by_checkpoint: tuple[tuple[str, float], ...]
    marconi_alpha: float


def is_profiler_supported(max_input_tokens_total: int) -> bool:
    """判断完整 Agent Run 是否位于正式 profiler 的 32K 有效域。"""
    if max_input_tokens_total < 0:
        raise ValueError("max_input_tokens_total 必须非负")
    return max_input_tokens_total <= PROFILER_MAX_GAP_TOKENS


def concurrency_scale(active_run_count: int) -> str | None:
    """把同 provider active run 数映射到冻结的并发规模。"""
    if active_run_count < 0:
        raise ValueError("active_run_count 必须非负")
    if 2 <= active_run_count <= 4:
        return "Small"
    if 5 <= active_run_count <= 8:
        return "Medium"
    if active_run_count >= 9:
        return "Large"
    return None


def deterministic_event_key(
    event: ProtocolSnapshotEvent,
    seed: int = SAMPLING_SEED,
) -> str:
    """为 snapshot sampling 生成与输入顺序无关的稳定 SHA-256 键。"""
    material = "|".join(
        (
            str(seed),
            event.provider,
            event.scale,
            event.trigger_session_id,
            str(event.trigger_run_ordinal),
            str(event.trigger_round_pk),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def choose_protocol_events(
    events: Iterable[ProtocolSnapshotEvent],
) -> tuple[ProtocolSnapshotEvent, ...]:
    """每个非空 provider × scale stratum 选择固定键最小的事件。"""
    selected: dict[tuple[str, str], ProtocolSnapshotEvent] = {}
    for event in events:
        key = (event.provider, event.scale)
        current = selected.get(key)
        if current is None or (
            deterministic_event_key(event), event.trigger_round_pk
        ) < (
            deterministic_event_key(current), current.trigger_round_pk
        ):
            selected[key] = event
    scale_rank = {scale: index for index, scale in enumerate(SCALE_ORDER)}
    return tuple(
        sorted(
            selected.values(),
            key=lambda event: (
                scale_rank[event.scale],
                event.provider,
                event.snapshot_id,
            ),
        )
    )


def build_snapshot_policy_metadata(
    snapshot: TraceSnapshot,
) -> SnapshotPolicyMetadata:
    """冻结 STE、共享 recency 与 parent-relative FLOP proxy。"""
    checkpoint_metadata = {
        item.checkpoint_id: item for item in snapshot.checkpoint_metadata
    }
    if len(checkpoint_metadata) != len(snapshot.candidates):
        raise ValueError("candidate 与 checkpoint metadata 数量不一致")

    ordered_oldest_first = tuple(
        sorted(
            snapshot.candidates,
            key=lambda candidate: (
                checkpoint_metadata[candidate.checkpoint_id].known_at_time,
                checkpoint_metadata[candidate.checkpoint_id].round_pk,
                candidate.checkpoint_id,
            ),
        )
    )
    recency = tuple(
        TraceCheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=index,
            last_access_order=index,
            known_at_time=checkpoint_metadata[
                candidate.checkpoint_id
            ].known_at_time,
        )
        for index, candidate in enumerate(ordered_oldest_first, start=1)
    )
    last_access = tuple(
        (item.checkpoint_id, float(item.last_access_order))
        for item in recency
    )
    steps = tuple(
        (continuation.continuation_id, 1)
        for continuation in sorted(
            snapshot.continuations,
            key=lambda item: item.continuation_id,
        )
    )
    flop_saved = build_marconi_flop_saved(snapshot.candidates)
    if any(value <= 0.0 for value in flop_saved.values()):
        raise ValueError("Marconi incremental token span 必须严格为正")
    return SnapshotPolicyMetadata(
        steps_to_execution_by_continuation=steps,
        checkpoint_recency=recency,
        last_access_by_checkpoint=last_access,
        marconi_flop_saved_by_checkpoint=tuple(sorted(flop_saved.items())),
        marconi_alpha=MARCONI_ALPHA,
    )


def exact_parent_ids(
    continuation: PendingContinuation,
    candidates: Sequence[CheckpointCandidate],
) -> tuple[str, ...]:
    """返回同 workflow、同完整线性 lineage 且位于 anchor 的候选。"""
    return tuple(
        candidate.checkpoint_id
        for candidate in sorted(
            candidates,
            key=lambda item: item.checkpoint_id,
        )
        if candidate.workflow_id == continuation.workflow_id
        and candidate.lineage_path == continuation.lineage_path
        and candidate.token_pos == continuation.anchor_pos
    )


def immediate_ancestor_gap(
    continuation: PendingContinuation,
    candidates: Sequence[CheckpointCandidate],
) -> int:
    """计算 exact parent 丢失后最近兼容线性祖先到 anchor 的距离。"""
    compatible_ancestors = tuple(
        candidate
        for candidate in candidates
        if candidate.workflow_id == continuation.workflow_id
        and len(candidate.lineage_path) < len(continuation.lineage_path)
        and is_lineage_prefix(
            candidate.lineage_path,
            continuation.lineage_path,
        )
        and candidate.token_pos <= continuation.anchor_pos
    )
    if not compatible_ancestors:
        return continuation.anchor_pos
    nearest = max(
        compatible_ancestors,
        key=lambda candidate: (
            len(candidate.lineage_path),
            candidate.token_pos,
            candidate.checkpoint_id,
        ),
    )
    gap = continuation.anchor_pos - nearest.token_pos
    if gap < 0:
        raise ValueError("immediate ancestor gap 不能为负")
    return gap


def analyze_snapshot_structure(
    snapshot: TraceSnapshot,
    metadata: SnapshotPolicyMetadata,
) -> dict[str, Any]:
    """只做预算与结构检查，不执行任何 checkpoint selection。"""
    candidate_count = len(snapshot.candidates)
    pending_count = len(snapshot.continuations)
    workflow_count = len(snapshot.active_workflow_ids)
    if candidate_count <= 0 or pending_count <= 0 or workflow_count <= 0:
        raise ValueError("protocol snapshot 的 N、P、W 必须均大于零")

    exact_by_continuation = {
        continuation.continuation_id: exact_parent_ids(
            continuation,
            snapshot.candidates,
        )
        for continuation in snapshot.continuations
    }
    exact_parent_id_set = {
        checkpoint_id
        for checkpoint_ids in exact_by_continuation.values()
        for checkpoint_id in checkpoint_ids
    }
    ancestor_gaps = {
        continuation.continuation_id: immediate_ancestor_gap(
            continuation,
            snapshot.candidates,
        )
        for continuation in snapshot.continuations
    }
    budget_rows = []
    for ratio in RETENTION_RATIOS:
        k = max(1, math.floor(candidate_count * ratio))
        budget_rows.append(
            {
                "ratio": ratio,
                "k": k,
                "candidate_count": candidate_count,
                "active_workflow_count": workflow_count,
                "pending_count": pending_count,
                "k_over_n": k / candidate_count,
                "k_over_w": k / workflow_count,
                "k_over_p": k / pending_count,
                "k_lt_w": k < workflow_count,
                "k_lt_p": k < pending_count,
                "k_ge_w": k >= workflow_count,
                "k_ge_p": k >= pending_count,
                "capacity_sufficient_for_all_exact_parents": (
                    k >= len(exact_parent_id_set)
                ),
            }
        )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "scale": snapshot.scale,
        "provider": snapshot.time_domain,
        "candidate_count": candidate_count,
        "pending_count": pending_count,
        "active_workflow_count": workflow_count,
        "exact_parent_by_continuation": exact_by_continuation,
        "exact_parent_available_count": sum(
            bool(ids) for ids in exact_by_continuation.values()
        ),
        "distinct_exact_parent_count": len(exact_parent_id_set),
        "immediate_ancestor_gap_by_continuation": ancestor_gaps,
        "budgets": tuple(budget_rows),
        "kvflow_steps_all_one": all(
            value == 1
            for _, value in metadata.steps_to_execution_by_continuation
        ),
        "marconi_incremental_spans_positive": all(
            value > 0.0
            for _, value in metadata.marconi_flop_saved_by_checkpoint
        ),
    }


def aggregate_protocol(
    snapshots: Sequence[TraceSnapshot],
    snapshot_analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """聚合 profiler cohort snapshots 的竞争性与 spacing 统计。"""
    if not snapshots or len(snapshots) != len(snapshot_analyses):
        raise ValueError("snapshot 与结构分析必须非空且一一对应")
    workflow_counts = [len(item.active_workflow_ids) for item in snapshots]
    candidate_counts = [len(item.candidates) for item in snapshots]
    pending_counts = [len(item.continuations) for item in snapshots]
    gaps = [
        int(gap)
        for analysis in snapshot_analyses
        for gap in analysis["immediate_ancestor_gap_by_continuation"].values()
    ]
    total_pending = sum(pending_counts)
    exact_available = sum(
        int(analysis["exact_parent_available_count"])
        for analysis in snapshot_analyses
    )

    budget_summary = {}
    for ratio in RETENTION_RATIOS:
        rows = tuple(
            budget
            for analysis in snapshot_analyses
            for budget in analysis["budgets"]
            if budget["ratio"] == ratio
        )
        label = f"{int(ratio * 100)}%"
        budget_summary[label] = {
            "snapshot_count": len(rows),
            "k": _distribution([int(row["k"]) for row in rows]),
            "k_over_n": _distribution(
                [float(row["k_over_n"]) for row in rows]
            ),
            "k_over_w": _distribution(
                [float(row["k_over_w"]) for row in rows]
            ),
            "k_over_p": _distribution(
                [float(row["k_over_p"]) for row in rows]
            ),
            "k_lt_w_fraction": _true_fraction(rows, "k_lt_w"),
            "k_lt_p_fraction": _true_fraction(rows, "k_lt_p"),
            "k_ge_w_fraction": _true_fraction(rows, "k_ge_w"),
            "k_ge_p_fraction": _true_fraction(rows, "k_ge_p"),
            "capacity_sufficient_for_all_exact_parents_fraction": (
                _true_fraction(
                    rows,
                    "capacity_sufficient_for_all_exact_parents",
                )
            ),
        }
    return {
        "snapshot_count": len(snapshots),
        "unique_selected_runs": len(
            {
                workflow_id
                for snapshot in snapshots
                for workflow_id in snapshot.active_workflow_ids
            }
        ),
        "scale_counts": {
            scale: sum(snapshot.scale == scale for snapshot in snapshots)
            for scale in SCALE_ORDER
        },
        "active_workflows_per_snapshot": _distribution(workflow_counts),
        "candidates_per_snapshot": _distribution(candidate_counts),
        "pending_per_snapshot": _distribution(pending_counts),
        "exact_parent_availability_fraction": (
            exact_available / total_pending
        ),
        "exact_parent_count_per_snapshot": _distribution(
            [
                int(analysis["distinct_exact_parent_count"])
                for analysis in snapshot_analyses
            ]
        ),
        "immediate_ancestor_gap_tokens": _distribution(gaps),
        "gap_bucket_fractions": {
            "<=4K": _threshold_fraction(gaps, 4096),
            "<=8K": _threshold_fraction(gaps, 8192),
            "<=16K": _threshold_fraction(gaps, 16384),
            "<=32K": _threshold_fraction(gaps, 32768),
        },
        "budget_contention": budget_summary,
    }


def construct_protocol(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """只读构造 profiler-supported policy protocol，不运行任何 policy。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    try:
        cohort_rows = fetch_dicts(connection, _COHORT_STATISTICS_SQL)
        sampled_rows = fetch_dicts(
            connection,
            _SAMPLED_PROTOCOL_EVENTS_SQL,
            (str(SAMPLING_SEED),),
        )
        sampled_events = tuple(_event_from_row(row) for row in sampled_rows)
        snapshots = []
        active_run_limits: dict[str, dict[str, int]] = {}
        for event in sampled_events:
            known_rows = fetch_dicts(
                connection,
                _SUPPORTED_KNOWN_STATE_SQL,
                (
                    event.observed_at,
                    event.observed_at,
                    event.observed_at,
                    event.observed_at,
                    event.provider,
                ),
            )
            snapshot, run_limits = _snapshot_from_rows(event, known_rows)
            snapshots.append(snapshot)
            active_run_limits[event.snapshot_id] = run_limits
    finally:
        connection.close()

    policy_metadata = tuple(
        build_snapshot_policy_metadata(snapshot) for snapshot in snapshots
    )
    analyses = tuple(
        analyze_snapshot_structure(snapshot, metadata)
        for snapshot, metadata in zip(snapshots, policy_metadata)
    )
    summary = aggregate_protocol(snapshots, analyses)
    cohort = {str(row["provider"]): row for row in cohort_rows}
    expected_strata = {
        (provider, scale)
        for provider in ("claude", "codex")
        for scale in SCALE_ORDER
    }
    observed_strata = {
        (event.provider, event.scale) for event in sampled_events
    }
    empty_strata = tuple(
        {
            "provider": provider,
            "scale": scale,
        }
        for provider, scale in sorted(
            expected_strata - observed_strata,
            key=lambda item: (SCALE_ORDER.index(item[1]), item[0]),
        )
    )
    validation = validate_protocol(
        snapshots,
        policy_metadata,
        analyses,
        active_run_limits,
    )
    contention_gate = _contention_gate(summary["budget_contention"])
    gates = {
        "profiler_supported_cohort": (
            "PASS" if validation["profiler_domain_violations"] == 0 else "FAIL"
        ),
        "non_trivial_state_contention": contention_gate,
        "kvflow_metadata_well_defined": (
            "WEAK" if validation["kvflow_ste_violations"] == 0 else "FAIL"
        ),
        "marconi_metadata_well_defined": (
            "PASS" if validation["marconi_metadata_violations"] == 0 else "FAIL"
        ),
        "flowstate_metadata_leakage_free": (
            "PASS" if validation["flowstate_leakage_violations"] == 0 else "FAIL"
        ),
    }
    if "FAIL" in gates.values():
        ready = "FAIL"
    elif contention_gate == "FAIL":
        ready = "FAIL"
    else:
        ready = "WEAK"
    gates["ready_for_step_10d"] = ready

    artifact = {
        "schema_version": "tracelab-policy-protocol-v1",
        "source": {
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "access_mode": "read_only=True",
        },
        "profiler_supported_cohort": {
            "rule": "strictly_closed 且 max_input_tokens_total <= 32768",
            "maximum_recovery_gap_tokens": PROFILER_MAX_GAP_TOKENS,
            "frozen_before_policy_comparison": True,
            "statistics": cohort,
            "greater_than_32k_role": "仅保留用于真实 workload characterization",
        },
        "sampling_protocol": {
            "seed": SAMPLING_SEED,
            "strata": "provider × trace-observed concurrency scale",
            "scale_bands": {
                "Small": "2-4 active runs",
                "Medium": "5-8 active runs",
                "Large": ">=9 active runs",
            },
            "selection": "每个非空 stratum 选择固定 SHA-256 键最小的 tool-call 时点",
            "policy_or_cost_signal_used": False,
            "empty_strata": empty_strata,
        },
        "policy_metadata_protocol": {
            "kvflow": {
                "steps_to_execution": 1,
                "recency_fallback": "与 Global-LRU 完全相同",
                "future_round_count_used": False,
                "tool_count_used_as_steps": False,
                "anchor_depth_used_as_steps": False,
                "future_time_used_as_steps": False,
                "limitation": "TraceLab 不激活 KVFlow 的 DAG-distance 优势",
            },
            "marconi": {
                "recency": "与 Global-LRU 共用 checkpoint known_at_time 全序",
                "flop_proxy": "同 workflow 线性 ancestry 的 parent-relative incremental token span",
                "alpha": MARCONI_ALPHA,
                "future_round_used": False,
                "tuned": False,
            },
            "flowstate": {
                "inputs": (
                    "known pending continuation",
                    "known_anchor",
                    "linear lineage",
                    "正式 frozen Phi",
                ),
                "future_prefix_used": False,
                "future_round_used": False,
                "recency_used": False,
                "steps_to_execution_used": False,
                "phi_modified": False,
            },
        },
        "step_10d_preregistration": {
            "policies": POLICIES_FOR_STEP_10D,
            "primary_metrics": PRIMARY_METRICS,
            "secondary_metrics": SECONDARY_METRICS,
            "result_label": "trace-driven offline policy evaluation",
            "forbidden_result_label": "real runtime latency result",
            "oracle": "仅在 candidate count 足够小时允许 optional exact audit；不作为主 baseline",
        },
        "summary": summary,
        "validation": validation,
        "gates": gates,
        "snapshots": tuple(
            {
                "snapshot": _snapshot_to_dict(snapshot),
                "policy_metadata": _json_value(asdict(metadata)),
                "structural_analysis": analysis,
                "active_run_max_input_tokens": active_run_limits[
                    snapshot.snapshot_id
                ],
            }
            for snapshot, metadata, analysis in zip(
                snapshots,
                policy_metadata,
                analyses,
            )
        ),
    }
    return _json_value(artifact)


def validate_protocol(
    snapshots: Sequence[TraceSnapshot],
    metadata_rows: Sequence[SnapshotPolicyMetadata],
    analyses: Sequence[Mapping[str, Any]],
    active_run_limits: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    """执行 profiler 有效域、metadata 与未来字段的完整性检查。"""
    profiler_domain_violations = 0
    kvflow_ste_violations = 0
    marconi_metadata_violations = 0
    flowstate_leakage_violations = 0
    exact_parent_missing = 0
    for snapshot, metadata, analysis in zip(
        snapshots,
        metadata_rows,
        analyses,
    ):
        profiler_domain_violations += sum(
            value > PROFILER_MAX_GAP_TOKENS
            for value in active_run_limits[snapshot.snapshot_id].values()
        )
        kvflow_ste_violations += sum(
            value != 1
            for _, value in metadata.steps_to_execution_by_continuation
        )
        if metadata.marconi_alpha != MARCONI_ALPHA:
            marconi_metadata_violations += 1
        marconi_metadata_violations += sum(
            value <= 0.0
            for _, value in metadata.marconi_flop_saved_by_checkpoint
        )
        flowstate_leakage_violations += int(snapshot.future_prefix_used)
        flowstate_leakage_violations += int(
            snapshot.runtime_residency_inferred
        )
        flowstate_leakage_violations += int(
            snapshot.llm_level_branching_introduced
        )
        exact_parent_missing += (
            int(analysis["pending_count"])
            - int(analysis["exact_parent_available_count"])
        )
    return {
        "profiler_domain_violations": profiler_domain_violations,
        "kvflow_ste_violations": kvflow_ste_violations,
        "marconi_metadata_violations": marconi_metadata_violations,
        "flowstate_leakage_violations": flowstate_leakage_violations,
        "exact_parent_missing": exact_parent_missing,
        "formal_policy_runs": 0,
        "phi_calls": 0,
        "oracle_runs": 0,
    }


def render_report(protocol: Mapping[str, Any]) -> str:
    """把冻结协议渲染为中文技术报告。"""
    cohort = protocol["profiler_supported_cohort"]["statistics"]
    overall = cohort["全部"]
    summary = protocol["summary"]
    gates = protocol["gates"]
    active = summary["active_workflows_per_snapshot"]
    candidates = summary["candidates_per_snapshot"]
    pending = summary["pending_per_snapshot"]
    spacing = summary["immediate_ancestor_gap_tokens"]
    lines = [
        "# TraceLab 策略评估协议冻结",
        "",
        "## 技术摘要",
        "",
        f"正式 TraceLab policy cohort 冻结为 {overall['run_count']:,} 个严格闭合且完整 run 最大输入不超过 32K 的 Agent Runs。固定 seed `{SAMPLING_SEED}` 生成 {summary['snapshot_count']} 个 provider × concurrency snapshots；本步骤 policy runs=0、Phi calls=0、Oracle runs=0。",
        "",
        f"有效域与无泄漏 gate 均通过，但非平凡竞争性为 {gates['non_trivial_state_contention']}，KVFlow metadata 为 WEAK：TraceLab 只能安全赋予所有已知 pending `steps_to_execution=1`，不能激活 DAG-distance signal。因此 Step 10D readiness={gates['ready_for_step_10d']}，不得据此调整 workload 或预算。",
        "",
        "## 32K 有效域在 policy 之前冻结",
        "",
        "| Provider | Strict runs | Rounds | Pending | Candidates | Mean rounds/run | Median | P90 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in ("全部", "claude", "codex"):
        row = cohort[provider]
        lines.append(
            f"| {provider} | {row['run_count']:,} | {row['round_count']:,} | {row['pending_count']:,} | {row['candidate_count']:,} | {row['mean_rounds_per_run']:.3f} | {row['median_rounds_per_run']:.0f} | {row['p90_rounds_per_run']:.0f} | {row['p95_rounds_per_run']:.0f} | {row['max_rounds_per_run']:,} |"
        )
    lines.extend(
        [
            "",
            "该规则保证任意 planning gap 均位于 `[0, 32768]`。大于 32K 的 runs 仍保留在 Step 10C characterization 中，但不进入当前正式 Phi-based comparison。",
            "",
            "## 实际支持三个非空并发 strata",
            "",
            f"固定 sampling 只按 provider 与 trace-observed concurrency scale 分层，不读取 policy value、Phi 或 gap。最终 Small={summary['scale_counts']['Small']}、Medium={summary['scale_counts']['Medium']}、Large={summary['scale_counts']['Large']}；empty strata={len(protocol['sampling_protocol']['empty_strata'])}。",
            "",
            "| Snapshot 指标 | Mean | Median | P90 | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|",
            _distribution_row("Active workflows", active),
            _distribution_row("Candidates", candidates),
            _distribution_row("Pending", pending),
            "",
            f"共有 {summary['unique_selected_runs']} 个不同 runs 出现在 snapshots 中。跨 provider 时间不混合，并发仍只解释为 trace-observed overlap。",
            "",
            "## Budget contention 先报告、不调参",
            "",
            "| Ratio | K median | K/N mean | K/W mean | K/P mean | K<W | K<P | K>=W | K>=P | Exact-parent capacity sufficient |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("25%", "50%", "75%"):
        row = summary["budget_contention"][label]
        lines.append(
            f"| {label} | {row['k']['median']:.0f} | {row['k_over_n']['mean']:.3f} | {row['k_over_w']['mean']:.3f} | {row['k_over_p']['mean']:.3f} | {_percent(row['k_lt_w_fraction'])} | {_percent(row['k_lt_p_fraction'])} | {_percent(row['k_ge_w_fraction'])} | {_percent(row['k_ge_p_fraction'])} | {_percent(row['capacity_sufficient_for_all_exact_parents_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "`K<W` 表示预算无法为每个 active workflow 各留一个状态；`K<P` 和 exact-parent capacity 表示预算是否足以同时保护所有当前 pending。这里不执行任何 selection。",
            "",
            "## Exact-parent 与 spacing 只做结构审计",
            "",
            f"exact-parent availability={_percent(summary['exact_parent_availability_fraction'])}。每个 pending 只检查同 workflow、相同完整线性 lineage、`token_pos == anchor_pos` 的候选。",
            "",
            f"exact parent 丢失后，最近严格线性兼容祖先的 gap：median={spacing['median']:.0f}、P90={spacing['p90']:.0f}、P95={spacing['p95']:.0f}、max={spacing['max']:.0f} tokens。若没有更早兼容 checkpoint，gap 按 anchor 到零计算。",
            "",
            "| Gap 上限 | 比例 |",
            "|---:|---:|",
        ]
    )
    for label in ("<=4K", "<=8K", "<=16K", "<=32K"):
        lines.append(
            f"| {label} | {_percent(summary['gap_bucket_fractions'][label])} |"
        )
    lines.extend(
        [
            "",
            "spacing 只使用 snapshot 当前与历史 candidates，不调用 Phi。",
            "",
            "## 三类 policy metadata 已预注册",
            "",
            "### KVFlow-style",
            "",
            "所有 known pending 的 `steps_to_execution=1`；相同优先级时使用与 Global-LRU 完全相同的 recency。不得从未来 round 数、tool 数、anchor 深度或真实未来等待时间生成 STE。TraceLab 不验证 KVFlow 的 DAG-distance 优势。",
            "",
            "### Marconi-style",
            "",
            "recency 与 Global-LRU 共用 checkpoint `known_at_time` 全序；FLOP proxy 使用同 workflow 线性 ancestry 上 parent-relative incremental token span；`alpha=1.0`，不搜索、不调参。所有 span 只来自 snapshot 当前与历史 token positions。",
            "",
            "### FlowState",
            "",
            "只允许 known pending、known anchor、linear lineage 和正式 frozen Phi。future prefix、future round、recency 与 STE 不进入 FlowState metadata；本步骤没有读取或调用 Phi。",
            "",
            "## Step 10D 指标与解释边界",
            "",
            "Primary：total/mean predicted recovery cost，以及 total/mean recovery gap。Secondary：executable-hit ratio、max gap、P95 gap。主 baseline 只包含 Global-LRU、KVFlow-style、Marconi-style、FlowState。Oracle 仅允许在小 candidate snapshot 做 optional exact audit，本步骤未运行。",
            "",
            "下一步结果只能称为 `trace-driven offline policy evaluation`，不能称为真实 runtime latency。",
            "",
            "## Gate",
            "",
            "| Gate | 结果 |",
            "|---|---|",
        ]
    )
    for key, label in (
        ("profiler_supported_cohort", "Profiler-supported cohort"),
        ("non_trivial_state_contention", "Non-trivial state contention"),
        ("kvflow_metadata_well_defined", "KVFlow metadata well-defined"),
        ("marconi_metadata_well_defined", "Marconi metadata well-defined"),
        ("flowstate_metadata_leakage_free", "FlowState metadata leakage-free"),
        ("ready_for_step_10d", "Ready for Step 10D"),
    ):
        lines.append(f"| {label} | {gates[key]} |")
    lines.extend(
        [
            "",
            "WEAK 不授权修改 budget、metadata 或 sampling；它只记录 TraceLab 缺少显式 DAG signal，以及当前 cohort 的结构竞争强度。",
            "",
        ]
    )
    return "\n".join(lines)


def write_protocol(
    protocol: Mapping[str, Any],
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """保存冻结 protocol JSON 与 Markdown 技术报告。"""
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(protocol), encoding="utf-8")


def _event_from_row(row: Mapping[str, Any]) -> ProtocolSnapshotEvent:
    scale = str(row["scale"])
    provider = str(row["provider"])
    return ProtocolSnapshotEvent(
        snapshot_id=f"profiler-{scale.lower()}-{provider}",
        scale=scale,
        provider=provider,
        trigger_session_id=str(row["session_id"]),
        trigger_run_ordinal=int(row["run_ordinal"]),
        trigger_round_pk=int(row["round_pk"]),
        observed_at=row["observed_at"],
        active_run_count=int(row["active_run_count"]),
    )


def _snapshot_from_rows(
    event: ProtocolSnapshotEvent,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[TraceSnapshot, dict[str, int]]:
    workflow_ids = tuple(
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
    if len(workflow_ids) != event.active_run_count:
        raise ValueError("active workflow 数与采样时统计不一致")
    run_limits = {
        workflow_id_for(
            str(row["provider"]),
            str(row["session_id"]),
            int(row["run_ordinal"]),
        ): int(row["run_max_input_tokens_total"])
        for row in rows
    }
    completed = []
    pending = []
    for row in rows:
        workflow_id = workflow_id_for(
            str(row["provider"]),
            str(row["session_id"]),
            int(row["run_ordinal"]),
        )
        if row["known_completion_time"] is not None:
            completed.append(
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
            pending.append(
                PendingRoundFact(
                    workflow_id=workflow_id,
                    round_pk=int(row["round_pk"]),
                    round_index=int(row["round_index"]),
                    run_position=int(row["run_position"]),
                    input_tokens_total=int(row["input_tokens_total"]),
                    current_prefix_tokens=int(row["prefix_tokens"]),
                    known_at_time=row["first_observed_tool_time"],
                    observed_tool_call_ids=tuple(
                        str(value)
                        for value in row["observed_tool_call_ids"]
                    ),
                )
            )
    snapshot = build_trace_snapshot(
        snapshot_id=event.snapshot_id,
        scale=event.scale,
        time_domain=event.provider,
        observed_at=event.observed_at,
        active_workflow_ids=workflow_ids,
        completed_rounds=completed,
        pending_rounds=pending,
    )
    return snapshot, run_limits


def _snapshot_to_dict(snapshot: TraceSnapshot) -> dict[str, Any]:
    value = asdict(snapshot)
    value["active_workflow_count"] = len(snapshot.active_workflow_ids)
    value["candidate_count"] = len(snapshot.candidates)
    value["pending_count"] = len(snapshot.continuations)
    return _json_value(value)


def _contention_gate(
    budget_summary: Mapping[str, Mapping[str, Any]],
) -> str:
    contentious_ratios = sum(
        row["k_lt_w_fraction"] > 0.0 or row["k_lt_p_fraction"] > 0.0
        for row in budget_summary.values()
    )
    if contentious_ratios == len(RETENTION_RATIOS):
        return "PASS"
    if contentious_ratios > 0:
        return "WEAK"
    return "FAIL"


def _true_fraction(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    if not rows:
        raise ValueError("比例 rows 不能为空")
    return sum(bool(row[field]) for row in rows) / len(rows)


def _threshold_fraction(values: Sequence[int], threshold: int) -> float:
    if not values:
        raise ValueError("gap values 不能为空")
    return sum(value <= threshold for value in values) / len(values)


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": _quantile_disc(ordered, 0.90),
        "p95": _quantile_disc(ordered, 0.95),
        "max": ordered[-1],
    }


def _quantile_disc(
    ordered: Sequence[float | int],
    probability: float,
) -> float | int:
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _distribution_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {row['mean']:.3f} | {row['median']:.0f} | "
        f"{row['p90']:.0f} | {row['p95']:.0f} | {row['max']:.0f} |"
    )


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


_COHORT_STATISTICS_SQL = """
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
), tool_rounds AS (
    SELECT DISTINCT round_pk FROM tool_calls
), run_statistics AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        count(*) AS round_count,
        max(a.input_tokens_total) AS max_input_tokens_total,
        count(*) FILTER (WHERE t.round_pk IS NOT NULL) AS pending_count
    FROM assigned a
    LEFT JOIN tool_rounds t USING (round_pk)
    GROUP BY a.provider, a.session_id, a.run_ordinal
), classified AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM run_statistics
), supported AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= 32768
), provider_statistics AS (
    SELECT
        provider,
        count(*) AS run_count,
        sum(round_count) AS round_count,
        sum(pending_count) AS pending_count,
        sum(round_count) AS candidate_count,
        avg(round_count) AS mean_rounds_per_run,
        median(round_count) AS median_rounds_per_run,
        quantile_disc(round_count, 0.90) AS p90_rounds_per_run,
        quantile_disc(round_count, 0.95) AS p95_rounds_per_run,
        max(round_count) AS max_rounds_per_run
    FROM supported
    GROUP BY provider
), overall AS (
    SELECT
        '全部' AS provider,
        count(*) AS run_count,
        sum(round_count) AS round_count,
        sum(pending_count) AS pending_count,
        sum(round_count) AS candidate_count,
        avg(round_count) AS mean_rounds_per_run,
        median(round_count) AS median_rounds_per_run,
        quantile_disc(round_count, 0.90) AS p90_rounds_per_run,
        quantile_disc(round_count, 0.95) AS p95_rounds_per_run,
        max(round_count) AS max_rounds_per_run
    FROM supported
)
SELECT * FROM overall
UNION ALL
SELECT * FROM provider_statistics
ORDER BY provider
"""


_SAMPLED_PROTOCOL_EVENTS_SQL = """
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
), supported AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= 32768
), tool_events AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        max(t.emitted_at) AS observed_at
    FROM assigned a
    JOIN tool_calls t USING (round_pk)
    JOIN supported s USING (provider, session_id, run_ordinal)
    WHERE t.emitted_at IS NOT NULL
    GROUP BY a.provider, a.session_id, a.run_ordinal, a.round_pk
), counted AS (
    SELECT
        event.provider,
        event.session_id,
        event.run_ordinal,
        event.round_pk,
        event.observed_at,
        count(active.session_id) AS active_run_count
    FROM tool_events event
    JOIN supported active
      ON active.provider = event.provider
     AND active.start_time <= event.observed_at
     AND active.end_time >= event.observed_at
    GROUP BY
        event.provider, event.session_id, event.run_ordinal,
        event.round_pk, event.observed_at
), labeled AS (
    SELECT
        *,
        CASE
            WHEN active_run_count BETWEEN 2 AND 4 THEN 'Small'
            WHEN active_run_count BETWEEN 5 AND 8 THEN 'Medium'
            WHEN active_run_count >= 9 THEN 'Large'
        END AS scale
    FROM counted
), ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY provider, scale
            ORDER BY sha256(
                concat_ws(
                    '|', ?, provider, scale, session_id,
                    run_ordinal::VARCHAR, round_pk::VARCHAR
                )
            ), round_pk
        ) AS sample_rank
    FROM labeled
    WHERE scale IS NOT NULL
)
SELECT
    provider,
    scale,
    session_id,
    run_ordinal,
    round_pk,
    observed_at,
    active_run_count
FROM ranked
WHERE sample_rank = 1
ORDER BY
    CASE scale WHEN 'Small' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    provider
"""


_SUPPORTED_KNOWN_STATE_SQL = """
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
), supported AS (
    SELECT *
    FROM classified
    WHERE is_strictly_closed AND max_input_tokens_total <= 32768
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
        s.max_input_tokens_total AS run_max_input_tokens_total,
        event.started_at,
        event.completion_marker_at,
        coalesce(tool.observed_tool_call_count, 0)
            AS observed_tool_call_count,
        coalesce(tool.observed_tool_call_ids, [])
            AS observed_tool_call_ids,
        tool.first_observed_tool_time
    FROM assigned a
    JOIN supported s USING (provider, session_id, run_ordinal)
    JOIN observed_round_events event USING (round_pk)
    LEFT JOIN observed_tools tool USING (round_pk)
    WHERE s.start_time <= ?
      AND s.end_time >= ?
      AND s.provider = ?
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
    """生成冻结 policy protocol artifact，不运行任何正式策略。"""
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
        default=DEFAULT_PROTOCOL_PATH,
        help="protocol JSON 输出路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown 报告输出路径",
    )
    arguments = parser.parse_args(argv)
    protocol = construct_protocol(arguments.database)
    write_protocol(protocol, arguments.output, arguments.report)
    print(json.dumps(protocol["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
