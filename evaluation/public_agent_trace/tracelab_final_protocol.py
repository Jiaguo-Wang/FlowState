#!/usr/bin/env python3
"""冻结 TraceLab 最终离线评估的 cohort、采样、预算与元数据协议。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_context_pressure import (
    ContextSnapshotEvent,
    _COHORT_COUNTS_SQL,
    _KNOWN_STATE_SQL,
    _SAMPLED_EVENTS_SQL,
    _snapshot_from_rows,
    analyze_snapshot,
)
from evaluation.public_agent_trace.tracelab_policy_protocol import (
    MARCONI_ALPHA,
    SnapshotPolicyMetadata,
    TraceCheckpointRecency,
)
from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CONTEXT_BUCKET_ORDER,
    SAMPLING_SEED,
    SCALE_ORDER,
    SampledSnapshotEvent,
    TraceSnapshot,
    deterministic_sample_key,
)
from flowstate.state_catalog import is_lineage_prefix


MAIN_COHORT = "C128"
MAIN_COHORT_MAX_TOKENS = 131_072
MAX_SNAPSHOTS_PER_STRATUM = 5
DEMAND_RETENTION_RATIOS = (0.25, 0.50, 0.75, 1.00)
POLICY_SET = (
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
    "p95_recovery_gap_tokens",
    "max_recovery_gap_tokens",
)
STRUCTURAL_METRICS = (
    "distinct_exact_parent_demand_x",
    "candidate_count_n",
    "n_over_x",
    "active_workflow_count",
)
DEFAULT_CONTEXT_EVIDENCE_PATH = Path(__file__).with_name(
    "tracelab_context_pressure.json"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name(
    "tracelab_final_protocol.json"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    "TRACELAB_FINAL_PROTOCOL.md"
)


@dataclass(frozen=True)
class RankedSnapshot:
    """保存候选采样事件、已知快照与其固定哈希排名。"""

    event: ContextSnapshotEvent
    snapshot: TraceSnapshot
    rank_key: str


def demand_relative_budget(exact_parent_count: int, ratio: float) -> int:
    """按 distinct exact-parent demand 计算至少为一的预算。"""
    if exact_parent_count < 0:
        raise ValueError("exact_parent_count 必须非负")
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("ratio 必须位于 (0, 1] 区间")
    return max(1, math.floor(exact_parent_count * ratio))


def active_run_set(snapshot: TraceSnapshot) -> tuple[str, ...]:
    """返回可用于全局重复检查的稳定 active-run 集合。"""
    return tuple(sorted(snapshot.active_workflow_ids))


def choose_unique_ranked_snapshots(
    records: Sequence[RankedSnapshot],
    max_per_stratum: int = MAX_SNAPSHOTS_PER_STRATUM,
) -> tuple[RankedSnapshot, ...]:
    """按固定哈希选择快照，并跳过完全相同的 active-run 集合。"""
    if max_per_stratum <= 0:
        raise ValueError("max_per_stratum 必须大于零")
    selected = []
    counts: dict[tuple[str, str, str], int] = {}
    seen_active_sets: set[tuple[str, ...]] = set()
    for record in sorted(
        records,
        key=lambda item: (
            item.rank_key,
            item.event.event.trigger_round_pk,
            item.snapshot.snapshot_id,
        ),
    ):
        event = record.event.event
        stratum = (event.provider, event.context_bucket, event.scale)
        signature = active_run_set(record.snapshot)
        if counts.get(stratum, 0) >= max_per_stratum:
            continue
        if signature in seen_active_sets:
            continue
        selected.append(record)
        counts[stratum] = counts.get(stratum, 0) + 1
        seen_active_sets.add(signature)
    scale_rank = {value: index for index, value in enumerate(SCALE_ORDER)}
    bucket_rank = {
        value: index for index, value in enumerate(CONTEXT_BUCKET_ORDER)
    }
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                scale_rank[item.snapshot.scale],
                item.snapshot.time_domain,
                bucket_rank[item.event.event.context_bucket],
                item.rank_key,
                item.snapshot.snapshot_id,
            ),
        )
    )


def construct_final_protocol(
    database_path: Path = DEFAULT_DATABASE_PATH,
    context_evidence_path: Path = DEFAULT_CONTEXT_EVIDENCE_PATH,
) -> dict[str, Any]:
    """只读构造最终协议，不执行 policy、Phi 或 GPU。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    if not context_evidence_path.is_file():
        raise FileNotFoundError(
            f"Step 10C.2 证据不存在：{context_evidence_path}"
        )
    context_evidence = json.loads(
        context_evidence_path.read_text(encoding="utf-8")
    )
    cohort_evidence = _cohort_evidence(context_evidence)
    _validate_cohort_rationale(cohort_evidence)

    dense_query = _SAMPLED_EVENTS_SQL.replace(
        "WHERE sample_rank = 1",
        "WHERE sample_rank <= ?",
    )
    if dense_query == _SAMPLED_EVENTS_SQL:
        raise RuntimeError("无法从冻结 Step 10C.2 查询构造密集采样")

    connection = open_database_read_only(database_path)
    try:
        source_counts = _single_row(
            connection,
            _COHORT_COUNTS_SQL,
            (MAIN_COHORT_MAX_TOKENS,),
        )
        event_rows = fetch_dicts(
            connection,
            dense_query,
            (
                MAIN_COHORT_MAX_TOKENS,
                str(SAMPLING_SEED),
                MAX_SNAPSHOTS_PER_STRATUM,
            ),
        )
        records = []
        for row in event_rows:
            event = _event_from_row(row)
            known_rows = fetch_dicts(
                connection,
                _KNOWN_STATE_SQL,
                (
                    MAIN_COHORT_MAX_TOKENS,
                    event.event.observed_at,
                    event.event.observed_at,
                    event.event.observed_at,
                    event.event.observed_at,
                    event.event.provider,
                ),
            )
            snapshot = _snapshot_from_rows(event, known_rows)
            records.append(
                RankedSnapshot(
                    event=event,
                    snapshot=snapshot,
                    rank_key=_rank_key(event.event),
                )
            )
    finally:
        connection.close()

    characterization_records = choose_unique_ranked_snapshots(records)
    characterization_analyses = tuple(
        analyze_snapshot(record.snapshot)
        for record in characterization_records
    )
    formal_rows = tuple(
        (record, analysis)
        for record, analysis in zip(
            characterization_records,
            characterization_analyses,
        )
        if int(analysis["x"]) > 0
    )
    if not formal_rows:
        raise ValueError("C128 密集采样没有 X>0 的正式快照")
    formal_records = tuple(item[0] for item in formal_rows)
    formal_analyses = tuple(item[1] for item in formal_rows)
    metadata_rows = tuple(
        build_final_snapshot_policy_metadata(record.snapshot)
        for record in formal_records
    )

    validation = validate_final_protocol(
        formal_records,
        formal_analyses,
        metadata_rows,
    )
    summary = _summarize(formal_records, formal_analyses)
    gates = {
        "cohort_frozen": (
            "PASS"
            if source_counts["eligible_runs"]
            == cohort_evidence[MAIN_COHORT]["eligible_runs"]
            else "FAIL"
        ),
        "sampling_frozen": (
            "PASS"
            if validation["sampling_violations"] == 0
            else "FAIL"
        ),
        "demand_relative_budget_frozen": (
            "PASS"
            if validation["budget_violations"] == 0
            else "FAIL"
        ),
        "policy_metadata_frozen": (
            "PASS"
            if validation["policy_metadata_violations"] == 0
            else "FAIL"
        ),
        "ready_for_profiler_extension": "PASS",
        "ready_for_policy_comparison": "NO",
    }
    artifact = {
        "schema_version": "tracelab-final-offline-protocol-v1",
        "source": {
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "access_mode": "read_only=True",
            "context_evidence_path": str(context_evidence_path),
        },
        "main_cohort": {
            "name": MAIN_COHORT,
            "rule": (
                "strictly closed Agent Run 且 "
                "max_input_tokens_total <= 131072"
            ),
            "maximum_input_tokens_total": MAIN_COHORT_MAX_TOKENS,
            "eligible_runs": int(source_counts["eligible_runs"]),
            "frozen_before_policy_performance": True,
            "rationale": _cohort_rationale(cohort_evidence),
            "greater_than_128k_role": "保留用于 workload characterization",
        },
        "sampling_protocol": {
            "seed": SAMPLING_SEED,
            "strata": "provider × context bucket × concurrency scale",
            "scale_bands": {
                "Small": "2-4 active runs",
                "Medium": "5-8 active runs",
                "Large": ">=9 active runs",
            },
            "ranking": "stratum 内固定 SHA-256 升序",
            "maximum_snapshots_per_nonempty_stratum": (
                MAX_SNAPSHOTS_PER_STRATUM
            ),
            "duplicate_rule": (
                "active-run set 完全相同时保留哈希排名更前者"
            ),
            "phi_used": False,
            "policy_used": False,
            "recovery_objective_used": False,
            "flowstate_performance_filter_used": False,
        },
        "demand_filter": {
            "formal_workload_requires_x_gt_zero": True,
            "characterization_snapshot_count": len(
                characterization_records
            ),
            "x_zero_snapshots_excluded": (
                len(characterization_records) - len(formal_records)
            ),
            "x_zero_role": "仅保留 workload characterization",
            "other_snapshot_filters": (),
        },
        "candidate_protocol": {
            "rule": "保留 snapshot 时点之前已生成的全部逻辑 recurrent checkpoints",
            "exact_parent_only": False,
            "value_pruning": False,
            "phi_pruning": False,
            "recency_pruning": False,
        },
        "pending_anchor_protocol": {
            "pending_rule": (
                "当前已发出至少一个 tool call 时至多产生一个 LLM-level pending"
            ),
            "known_anchor": "current_round.input_tokens_total",
            "multiple_tool_calls_create_fanout": False,
            "future_fields_used": False,
            "lineage": "按已发生 round execution order 构造的线性 lineage",
        },
        "budget_protocol": {
            "name": "demand-relative recurrent-state budget",
            "operationalization": "exact-parent demand X",
            "x_definition": (
                "所有 known pending 所需的 distinct exact-parent recurrent states"
            ),
            "formula": "K(r) = max(1, floor(X * r))",
            "ratios": DEMAND_RETENTION_RATIOS,
            "candidate_relative_budget_retired": True,
            "x_equals_p": (
                "当前 TraceLab 线性 workflow 的数据性质，不是算法假设"
            ),
            "step_10c2_n_over_x_medians": {
                "C32": 8.0,
                "C64": 4.5,
                "C128": 13.0,
                "C256": 20.5,
            },
            "candidate_relative_50_75_k_ge_x": (
                "Step 10C.2 四个 cohort 均为 100%"
            ),
        },
        "policy_metadata_protocol": {
            "kvflow": {
                "steps_to_execution": 1,
                "recency_fallback": "与 Global-LRU 完全相同",
                "future_round_distance_used": False,
                "tool_count_used": False,
                "elapsed_time_used": False,
                "anchor_depth_used": False,
                "limitation": "TraceLab 不激活 richer DAG-distance signal",
            },
            "marconi": {
                "recency": "与 Global-LRU 共用 known_at_time 全序",
                "flop_proxy": (
                    "同 workflow 线性 ancestry 的 parent-relative incremental token span"
                ),
                "zero_token_checkpoint_rule": (
                    "TraceLab 原始 token_pos=0 checkpoint 保留零 span，不删除 candidate"
                ),
                "alpha": MARCONI_ALPHA,
                "future_round_used": False,
                "tuned": False,
            },
            "flowstate": {
                "inputs": (
                    "known pending continuations",
                    "known anchor",
                    "linear lineage",
                    "frozen recovery model",
                ),
                "recency_used": False,
                "steps_to_execution_used": False,
                "future_prefix_used": False,
                "future_round_used": False,
            },
        },
        "recovery_model_requirement": {
            "maximum_required_validated_gap_tokens": (
                MAIN_COHORT_MAX_TOKENS
            ),
            "reason": (
                "当 K<X 且无 retained compatible checkpoint 时 E_p=0，"
                "所以 G_p 可达到 anchor_pos"
            ),
            "independent_profiler_validation_required": True,
            "validated_to_128k": False,
            "linear_extrapolation_allowed": False,
            "clamp_to_32k_allowed": False,
            "substitute_32k_cost_allowed": False,
            "formal_phi_modified": False,
        },
        "evaluation_protocol": {
            "policies": POLICY_SET,
            "oracle": (
                "仅在 candidate count 足够小时允许 optional exact audit，"
                "不属于主 baseline"
            ),
            "primary_metrics": PRIMARY_METRICS,
            "secondary_metrics": SECONDARY_METRICS,
            "structural_metadata": STRUCTURAL_METRICS,
            "snapshot_weight": "每个 selected snapshot 等权",
            "pending_weighted_secondary_allowed": True,
            "pending_weighted_results_separate": True,
        },
        "summary": summary,
        "validation": validation,
        "gates": gates,
        "execution": {
            "policy_comparison_executed": False,
            "phi_called": False,
            "gpu_executed": False,
        },
        "snapshots": tuple(
            _snapshot_payload(record, analysis, metadata)
            for record, analysis, metadata in zip(
                formal_records,
                formal_analyses,
                metadata_rows,
            )
        ),
    }
    return _json_value(artifact)


def validate_final_protocol(
    records: Sequence[RankedSnapshot],
    analyses: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[SnapshotPolicyMetadata],
) -> dict[str, int]:
    """验证最终协议的采样、预算、无泄漏与元数据约束。"""
    if not records or not (
        len(records) == len(analyses) == len(metadata_rows)
    ):
        raise ValueError("records、analyses 与 metadata 必须非空且一一对应")
    stratum_counts: dict[tuple[str, str, str], int] = {}
    active_sets = []
    budget_violations = 0
    policy_metadata_violations = 0
    marconi_zero_span_count = 0
    future_leakage_violations = 0
    branching_violations = 0
    x_zero_violations = 0
    x_not_equal_p = 0
    for record, analysis, metadata in zip(
        records,
        analyses,
        metadata_rows,
    ):
        event = record.event.event
        stratum = (event.provider, event.context_bucket, event.scale)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        snapshot = record.snapshot
        active_sets.append(active_run_set(snapshot))
        x_value = int(analysis["x"])
        p_value = int(analysis["p"])
        x_zero_violations += x_value <= 0
        x_not_equal_p += x_value != p_value
        for ratio in DEMAND_RETENTION_RATIOS:
            k = demand_relative_budget(x_value, ratio)
            budget_violations += k != max(1, math.floor(x_value * ratio))
        policy_metadata_violations += sum(
            value != 1
            for _, value in metadata.steps_to_execution_by_continuation
        )
        policy_metadata_violations += metadata.marconi_alpha != MARCONI_ALPHA
        policy_metadata_violations += sum(
            value < 0.0
            for _, value in metadata.marconi_flop_saved_by_checkpoint
        )
        marconi_zero_span_count += sum(
            value == 0.0
            for _, value in metadata.marconi_flop_saved_by_checkpoint
        )
        future_leakage_violations += int(snapshot.future_prefix_used)
        future_leakage_violations += int(snapshot.runtime_residency_inferred)
        branching_violations += int(
            snapshot.llm_level_branching_introduced
        )
    duplicate_count = len(active_sets) - len(set(active_sets))
    sampling_violations = duplicate_count + sum(
        count > MAX_SNAPSHOTS_PER_STRATUM
        for count in stratum_counts.values()
    )
    return {
        "sampling_violations": sampling_violations,
        "duplicate_active_run_set_count": duplicate_count,
        "budget_violations": budget_violations,
        "policy_metadata_violations": policy_metadata_violations,
        "marconi_zero_span_count": marconi_zero_span_count,
        "future_field_leakage_violations": future_leakage_violations,
        "llm_level_branching_violations": branching_violations,
        "x_zero_violations": x_zero_violations,
        "x_not_equal_p_count": x_not_equal_p,
        "policy_comparison_runs": 0,
        "phi_calls": 0,
        "gpu_runs": 0,
    }


def build_final_snapshot_policy_metadata(
    snapshot: TraceSnapshot,
) -> SnapshotPolicyMetadata:
    """按冻结规则生成 STE、共享 recency 与 parent-relative span。"""
    trace_metadata = {
        item.checkpoint_id: item for item in snapshot.checkpoint_metadata
    }
    if len(trace_metadata) != len(snapshot.candidates):
        raise ValueError("candidate 与 checkpoint metadata 数量不一致")
    ordered_oldest_first = tuple(
        sorted(
            snapshot.candidates,
            key=lambda candidate: (
                trace_metadata[candidate.checkpoint_id].known_at_time,
                trace_metadata[candidate.checkpoint_id].round_pk,
                candidate.checkpoint_id,
            ),
        )
    )
    recency = tuple(
        TraceCheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=index,
            last_access_order=index,
            known_at_time=trace_metadata[
                candidate.checkpoint_id
            ].known_at_time,
        )
        for index, candidate in enumerate(ordered_oldest_first, start=1)
    )
    flop_saved = []
    for candidate in sorted(
        snapshot.candidates,
        key=lambda item: item.checkpoint_id,
    ):
        parent_positions = tuple(
            ancestor.token_pos
            for ancestor in snapshot.candidates
            if ancestor.checkpoint_id != candidate.checkpoint_id
            and ancestor.workflow_id == candidate.workflow_id
            and ancestor.token_pos < candidate.token_pos
            and is_lineage_prefix(
                ancestor.lineage_path,
                candidate.lineage_path,
            )
        )
        incremental_span = candidate.token_pos - max(
            parent_positions,
            default=0,
        )
        if incremental_span < 0:
            raise ValueError("Marconi parent-relative span 不能为负")
        flop_saved.append((candidate.checkpoint_id, float(incremental_span)))
    return SnapshotPolicyMetadata(
        steps_to_execution_by_continuation=tuple(
            (continuation.continuation_id, 1)
            for continuation in sorted(
                snapshot.continuations,
                key=lambda item: item.continuation_id,
            )
        ),
        checkpoint_recency=recency,
        last_access_by_checkpoint=tuple(
            (item.checkpoint_id, float(item.last_access_order))
            for item in recency
        ),
        marconi_flop_saved_by_checkpoint=tuple(flop_saved),
        marconi_alpha=MARCONI_ALPHA,
    )


def render_report(artifact: Mapping[str, Any]) -> str:
    """生成以冻结决定与可审计门禁为核心的中文协议报告。"""
    summary = artifact["summary"]
    main = artifact["main_cohort"]
    sampling = summary["snapshot_counts"]
    structure = summary["structure"]
    lines = [
        "# TraceLab 最终离线评估协议",
        "",
        "## 冻结结论",
        "",
        "主 cohort 正式冻结为 **C128**。正式 workload 采用每个非空 `provider × context bucket × concurrency scale` 最多 5 个确定性快照、全局 active-run-set 去重、`X>0` demand gate，以及 exact-parent demand X 归一化预算。此决定在任何 policy performance 被观察之前完成。",
        "",
        "当前协议可进入 128K recovery profiler 扩展，但**不能进入正式 policy comparison**：正式 recovery profiler 尚未独立验证到 128K gap。",
        "",
        "## 为什么冻结 C128",
        "",
    ]
    lines.extend(f"- {reason}" for reason in main["rationale"])
    lines.extend(
        [
            "- 完整 TraceLab 中大于 128K 的数据继续保留用于 workload characterization，不从报告中删除。",
            "",
            "## 正式采样结果",
            "",
            f"固定 seed 为 `{artifact['sampling_protocol']['seed']}`，每层最多 {artifact['sampling_protocol']['maximum_snapshots_per_nonempty_stratum']} 个快照。先按 SHA-256 排名，再跳过 active-run set 完全相同的后项；不补造空层，也不使用 Phi、policy 或 recovery objective。",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| C128 eligible runs | {main['eligible_runs']:,} |",
            f"| Characterization snapshots | {artifact['demand_filter']['characterization_snapshot_count']} |",
            f"| X=0 excluded | {artifact['demand_filter']['x_zero_snapshots_excluded']} |",
            f"| Formal snapshots | {sampling['total']} |",
            f"| Small / Medium / Large | {sampling['by_scale']['Small']} / {sampling['by_scale']['Medium']} / {sampling['by_scale']['Large']} |",
            f"| Claude / Codex | {sampling['by_provider']['claude']} / {sampling['by_provider']['codex']} |",
            f"| Unique active runs | {summary['unique_active_runs']} |",
            f"| Duplicate active-run sets | {artifact['validation']['duplicate_active_run_set_count']} |",
            "",
            "## Snapshot 结构",
            "",
            "N 保留 snapshot 之前已经生成的全部逻辑 recurrent checkpoints；X 只表示当前 known pending 所需的 distinct exact-parent states。没有执行 exact-parent、value、Phi 或 recency pruning。",
            "",
            "| 指标 | Mean | Median | P90 | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("W", "N", "P", "X", "N/X"):
        row = structure[key]
        lines.append(
            f"| {key} | {_format(row['mean'])} | {_format(row['median'])} | {_format(row['p90'])} | {_format(row['p95'])} | {_format(row['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Demand-relative budget",
            "",
            "正式公式为 `K(r)=max(1, floor(X*r))`，其中 X 是 exact-parent demand。当前线性 TraceLab workload 中 X=P 是数据性质，不是算法假设。Candidate-relative 预算正式淘汰：历史 checkpoint 会令 N/X 偏大，并使 50%/75% 档缺少有效 state pressure。",
            "",
            "| Ratio | K count | K mean | K median | K P90 | K max | K<X |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in summary["budgets"].items():
        k = row["k_distribution"]
        lines.append(
            f"| {label} | {k['count']} | {_format(k['mean'])} | {_format(k['median'])} | {_format(k['p90'])} | {_format(k['max'])} | {_percent(row['k_lt_x_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## 冻结 policy metadata",
            "",
            "- KVFlow-style：所有 known pending 的 `steps_to_execution=1`；同 STE 时复用 Global-LRU recency。TraceLab 不激活 richer DAG-distance signal，也不从 future round distance、tool count、elapsed time 或 anchor depth 构造 STE。",
            "- Marconi-style：与 Global-LRU 共用 recency；FLOP proxy 使用同 workflow 线性 ancestry 上 parent-relative incremental token span；`alpha=1.0`，不调参、不读取未来 round。TraceLab 原始 `token_pos=0` checkpoint 保留零 span，不据此删除 candidate。",
            "- FlowState：只使用 known pending、known anchor、线性 lineage 与冻结 recovery model；不使用 recency、STE、future prefix 或 future round。",
            "",
            "## Recovery model 有效域门禁",
            "",
            "C128 中 anchor 最大为 131072；当 K<X 且没有 retained compatible checkpoint 时，E 可以为 0，G 因而可达到 anchor。正式 Phi-based comparison 前，独立 recovery profiler 必须验证至 128K。禁止线性外推、把 gap clamp 到 32K，或用 32K cost 替代更大 gap。本步骤没有修改 Phi。",
            "",
            "## 冻结指标与权重",
            "",
            "Primary：",
            "",
        ]
    )
    lines.extend(
        f"- `{value}`" for value in artifact["evaluation_protocol"]["primary_metrics"]
    )
    lines.extend(["", "Secondary：", ""])
    lines.extend(
        f"- `{value}`" for value in artifact["evaluation_protocol"]["secondary_metrics"]
    )
    lines.extend(
        [
            "",
            "Structural metadata：X、N、N/X、active workflows。主聚合对每个 selected snapshot 等权；允许另报 pending-weighted secondary aggregate，但必须分开。主 policy set 为 Global-LRU、KVFlow-style、Marconi-style、FlowState。Oracle 仅可在小 N 时作为 optional exact audit，不能改变主 workload 或参数。",
            "",
            "## 完整性门禁",
            "",
            f"- Cohort frozen: **{artifact['gates']['cohort_frozen']}**",
            f"- Sampling frozen: **{artifact['gates']['sampling_frozen']}**",
            f"- Demand-relative budget frozen: **{artifact['gates']['demand_relative_budget_frozen']}**",
            f"- Policy metadata frozen: **{artifact['gates']['policy_metadata_frozen']}**",
            f"- Future leakage violations: **{artifact['validation']['future_field_leakage_violations']}**",
            f"- LLM-level branching violations: **{artifact['validation']['llm_level_branching_violations']}**",
            f"- Ready for profiler extension: **{artifact['gates']['ready_for_profiler_extension']}**",
            f"- Ready for policy comparison: **{artifact['gates']['ready_for_policy_comparison']}**，因为 recovery profiler 尚未独立验证至 128K。",
            "",
            "本协议没有运行 policy、Phi 或 GPU；未来不得根据 policy performance 反向修改 cohort、采样、预算、metadata、指标或权重。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    artifact: Mapping[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """写入冻结 JSON 与中文 Markdown 协议。"""
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(artifact), encoding="utf-8")


def _event_from_row(row: Mapping[str, Any]) -> ContextSnapshotEvent:
    """把密集采样查询结果转换成具有唯一标识的 C128 事件。"""
    scale = str(row["scale"])
    provider = str(row["provider"])
    bucket = str(row["context_bucket"])
    safe_bucket = (
        bucket.replace("<=", "le-").replace(">", "gt-").replace("K", "k")
    )
    return ContextSnapshotEvent(
        cohort=MAIN_COHORT,
        cutoff_tokens=MAIN_COHORT_MAX_TOKENS,
        event=SampledSnapshotEvent(
            snapshot_id=(
                f"c128-{scale.lower()}-{provider}-{safe_bucket}-"
                f"round-{int(row['round_pk'])}"
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


def _rank_key(event: SampledSnapshotEvent) -> str:
    return deterministic_sample_key(
        seed=SAMPLING_SEED,
        provider=event.provider,
        context_bucket=event.context_bucket,
        scale=event.scale,
        session_id=event.trigger_session_id,
        run_ordinal=event.trigger_run_ordinal,
        round_pk=event.trigger_round_pk,
    )


def _cohort_evidence(
    artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["summary"]["cohort"]): item["summary"]
        for item in artifact["cohorts"]
    }


def _validate_cohort_rationale(
    evidence: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        "C32": (3, 10),
        "C64": (8, 29),
        "C128": (15, 64),
    }
    for cohort, (snapshot_count, unique_runs) in expected.items():
        row = evidence[cohort]
        if (
            int(row["snapshot_count"]) != snapshot_count
            or int(row["selected_unique_runs"]) != unique_runs
        ):
            raise ValueError(f"{cohort} 的 Step 10C.2 冻结证据发生变化")


def _cohort_rationale(
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    return (
        "C32 仅产生 3 个 representative snapshots",
        "C64 产生 8 个 representative snapshots",
        "C128 产生 15 个且已有 Medium/Large workload",
        "相比 C64，C128 eligible runs 增加 162.401%，selected unique runs 增加 120.690%",
        "C256 虽覆盖更多 workload，但要求明显更大的 recovery-model 有效域",
        "该选择在任何 policy performance 被观察之前冻结",
    )


def _summarize(
    records: Sequence[RankedSnapshot],
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    snapshots = tuple(record.snapshot for record in records)
    structure_values = {
        "W": [int(item["w"]) for item in analyses],
        "N": [int(item["n"]) for item in analyses],
        "P": [int(item["p"]) for item in analyses],
        "X": [int(item["x"]) for item in analyses],
        "N/X": [float(item["n_over_x"]) for item in analyses],
    }
    budget_summary = {}
    for ratio in DEMAND_RETENTION_RATIOS:
        x_values = structure_values["X"]
        k_values = [demand_relative_budget(value, ratio) for value in x_values]
        budget_summary[f"{int(ratio * 100)}%"] = {
            "ratio": ratio,
            "k_distribution": _distribution(k_values),
            "k_lt_x_fraction": sum(
                k < x for k, x in zip(k_values, x_values)
            )
            / len(x_values),
        }
    active_sets = [active_run_set(snapshot) for snapshot in snapshots]
    return {
        "snapshot_counts": {
            "total": len(snapshots),
            "by_scale": {
                scale: sum(snapshot.scale == scale for snapshot in snapshots)
                for scale in SCALE_ORDER
            },
            "by_provider": {
                provider: sum(
                    snapshot.time_domain == provider for snapshot in snapshots
                )
                for provider in ("claude", "codex")
            },
        },
        "unique_active_runs": len(
            {
                workflow_id
                for snapshot in snapshots
                for workflow_id in snapshot.active_workflow_ids
            }
        ),
        "duplicate_active_run_set_count": (
            len(active_sets) - len(set(active_sets))
        ),
        "structure": {
            key: _distribution(values)
            for key, values in structure_values.items()
        },
        "budgets": budget_summary,
    }


def _snapshot_payload(
    record: RankedSnapshot,
    analysis: Mapping[str, Any],
    metadata: SnapshotPolicyMetadata,
) -> dict[str, Any]:
    snapshot = record.snapshot
    x_value = int(analysis["x"])
    return {
        "sampling": {
            "event": asdict(record.event),
            "sha256_rank_key": record.rank_key,
            "active_run_set": active_run_set(snapshot),
            "snapshot_equal_weight": 1.0,
        },
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "scale": snapshot.scale,
            "time_domain": snapshot.time_domain,
            "observed_at": snapshot.observed_at,
            "active_workflow_ids": snapshot.active_workflow_ids,
            "candidates": tuple(asdict(item) for item in snapshot.candidates),
            "continuations": tuple(
                asdict(item) for item in snapshot.continuations
            ),
            "checkpoint_metadata": tuple(
                asdict(item) for item in snapshot.checkpoint_metadata
            ),
            "continuation_metadata": tuple(
                asdict(item) for item in snapshot.continuation_metadata
            ),
            "trace_observed_concurrency": (
                snapshot.trace_observed_concurrency
            ),
            "runtime_residency_inferred": snapshot.runtime_residency_inferred,
            "llm_level_branching_introduced": (
                snapshot.llm_level_branching_introduced
            ),
            "future_prefix_used": snapshot.future_prefix_used,
        },
        "structure": {
            "w": int(analysis["w"]),
            "n": int(analysis["n"]),
            "p": int(analysis["p"]),
            "x": x_value,
            "n_over_x": float(analysis["n_over_x"]),
            "exact_parent_by_continuation": analysis[
                "exact_parent_by_continuation"
            ],
        },
        "demand_relative_budgets": tuple(
            {
                "ratio": ratio,
                "k": demand_relative_budget(x_value, ratio),
            }
            for ratio in DEMAND_RETENTION_RATIOS
        ),
        "policy_metadata": asdict(metadata),
    }


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


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


def _format(value: int | float) -> str:
    return f"{value:.3f}"


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main(argv: Sequence[str] | None = None) -> int:
    """生成最终冻结协议与报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="外部 TraceLab DuckDB 路径",
    )
    parser.add_argument(
        "--context-evidence",
        type=Path,
        default=DEFAULT_CONTEXT_EVIDENCE_PATH,
        help="Step 10C.2 冻结证据路径",
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
    artifact = construct_final_protocol(
        arguments.database,
        arguments.context_evidence,
    )
    write_artifacts(artifact, arguments.output, arguments.report)
    print(
        json.dumps(
            {
                "main_cohort": artifact["main_cohort"]["name"],
                "eligible_runs": artifact["main_cohort"]["eligible_runs"],
                "selected_snapshots": artifact["summary"][
                    "snapshot_counts"
                ]["total"],
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
