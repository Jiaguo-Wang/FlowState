#!/usr/bin/env python3
"""把 TraceLab 严格闭合 Agent Runs 转换为无未来泄漏的离线快照。"""

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
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


CHECKPOINT_MEMORY_BYTES = 51_511_296
SAMPLING_SEED = 20_260_826
RETENTION_RATIOS = (0.25, 0.50, 0.75)
DEFAULT_WORKLOAD_PATH = Path(__file__).with_name("tracelab_workload.json")
DEFAULT_REPORT_PATH = Path(__file__).with_name("TRACELAB_WORKLOAD.md")
SCALE_ORDER = ("Small", "Medium", "Large")
CONTEXT_BUCKET_ORDER = (
    "<=32K",
    "32K-64K",
    "64K-128K",
    "128K-256K",
    ">256K",
)


@dataclass(frozen=True)
class SampledSnapshotEvent:
    """由固定分层采样选出的真实 tool-call 观测时点。"""

    snapshot_id: str
    scale: str
    provider: str
    context_bucket: str
    trigger_session_id: str
    trigger_run_ordinal: int
    trigger_round_pk: int
    observed_at: datetime
    trace_observed_active_runs: int


@dataclass(frozen=True)
class CompletedRoundFact:
    """在 snapshot 时点已经完成、可生成逻辑 checkpoint 的 round。"""

    workflow_id: str
    round_pk: int
    round_index: int
    run_position: int
    input_tokens_total: int
    current_prefix_tokens: int
    known_at_time: datetime


@dataclass(frozen=True)
class PendingRoundFact:
    """由当前已发出工具调用形成的单一 LLM-level pending 事实。"""

    workflow_id: str
    round_pk: int
    round_index: int
    run_position: int
    input_tokens_total: int
    current_prefix_tokens: int
    known_at_time: datetime
    observed_tool_call_ids: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointTraceMetadata:
    """逻辑 checkpoint 对应的 TraceLab 当前轮元数据。"""

    checkpoint_id: str
    round_pk: int
    round_index: int
    run_position: int
    known_at_time: datetime
    input_tokens_total: int
    current_prefix_tokens: int
    logical_candidate_only: bool = True


@dataclass(frozen=True)
class ContinuationTraceMetadata:
    """pending continuation 的当前已知来源信息。"""

    continuation_id: str
    round_pk: int
    known_at_round: int
    run_position: int
    known_at_time: datetime
    source_input_tokens_total: int
    current_prefix_tokens: int
    tool_call_count: int
    anchor_source: str = "current_input_boundary"


@dataclass(frozen=True)
class RetentionBudget:
    """按逻辑 candidate 数量计算的 recurrent-state retention budget。"""

    ratio: float
    k: int


@dataclass(frozen=True)
class TraceSnapshot:
    """可直接交给 FlowState 核心数据模型的离线 decision snapshot。"""

    snapshot_id: str
    scale: str
    time_domain: str
    observed_at: datetime
    active_workflow_ids: tuple[str, ...]
    candidates: tuple[CheckpointCandidate, ...]
    continuations: tuple[PendingContinuation, ...]
    checkpoint_metadata: tuple[CheckpointTraceMetadata, ...]
    continuation_metadata: tuple[ContinuationTraceMetadata, ...]
    retention_budgets: tuple[RetentionBudget, ...]
    trace_observed_concurrency: bool = True
    runtime_residency_inferred: bool = False
    llm_level_branching_introduced: bool = False
    future_prefix_used: bool = False


def workflow_id_for(
    provider: str,
    session_id: str,
    run_ordinal: int,
) -> str:
    """生成稳定且不会跨 Agent Run 混淆的 workflow 标识。"""
    if run_ordinal <= 0:
        raise ValueError("run_ordinal 必须大于零")
    return f"{provider}:{session_id}:run:{run_ordinal:06d}"


def linear_lineage_path(run_position: int) -> tuple[str, ...]:
    """按真实 round 执行顺序生成严格线性的 lineage prefix。"""
    if run_position < 0:
        raise ValueError("run_position 必须非负")
    return tuple(f"step:{index:06d}" for index in range(run_position + 1))


def checkpoint_id_for(fact: CompletedRoundFact) -> str:
    """生成 snapshot 内稳定的逻辑 checkpoint 标识。"""
    return f"{fact.workflow_id}:checkpoint:round:{fact.round_index:08d}"


def continuation_id_for(fact: PendingRoundFact) -> str:
    """生成工具阶段之后的单一 continuation 标识。"""
    return f"{fact.workflow_id}:pending:round:{fact.round_index:08d}"


def retention_budgets(candidate_count: int) -> tuple[RetentionBudget, ...]:
    """按 25%、50%、75% 比例向下取整，并保证 K 至少为一。"""
    if candidate_count <= 0:
        raise ValueError("snapshot 至少需要一个 candidate")
    return tuple(
        RetentionBudget(
            ratio=ratio,
            k=max(1, math.floor(candidate_count * ratio)),
        )
        for ratio in RETENTION_RATIOS
    )


def deterministic_sample_key(
    *,
    seed: int,
    provider: str,
    context_bucket: str,
    scale: str,
    session_id: str,
    run_ordinal: int,
    round_pk: int,
) -> str:
    """生成与输入列表顺序无关的固定 SHA-256 采样键。"""
    material = "|".join(
        (
            str(seed),
            provider,
            context_bucket,
            scale,
            session_id,
            str(run_ordinal),
            str(round_pk),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def choose_stratified_events(
    events: Iterable[SampledSnapshotEvent],
) -> tuple[SampledSnapshotEvent, ...]:
    """每个 scale、provider、context bucket 只保留固定采样键最小者。"""
    selected: dict[tuple[str, str, str], SampledSnapshotEvent] = {}
    for event in events:
        key = (event.scale, event.provider, event.context_bucket)
        current = selected.get(key)
        if current is None or _event_rank(event) < _event_rank(current):
            selected[key] = event
    scale_rank = {name: index for index, name in enumerate(SCALE_ORDER)}
    bucket_rank = {
        name: index for index, name in enumerate(CONTEXT_BUCKET_ORDER)
    }
    return tuple(
        sorted(
            selected.values(),
            key=lambda event: (
                scale_rank[event.scale],
                event.provider,
                bucket_rank[event.context_bucket],
                event.snapshot_id,
            ),
        )
    )


def build_trace_snapshot(
    *,
    snapshot_id: str,
    scale: str,
    time_domain: str,
    observed_at: datetime,
    active_workflow_ids: Sequence[str],
    completed_rounds: Sequence[CompletedRoundFact],
    pending_rounds: Sequence[PendingRoundFact],
) -> TraceSnapshot:
    """只用 snapshot 时点已知 facts 构造 FlowState 核心对象。"""
    active_ids = tuple(sorted(set(active_workflow_ids)))
    if len(active_ids) != len(tuple(active_workflow_ids)):
        raise ValueError("active_workflow_ids 不能重复")
    active_set = set(active_ids)

    candidates = []
    checkpoint_metadata = []
    for fact in sorted(
        completed_rounds,
        key=lambda item: (
            item.workflow_id,
            item.run_position,
            item.round_pk,
        ),
    ):
        if fact.workflow_id not in active_set:
            raise ValueError("checkpoint 不能来自非 active workflow")
        if fact.known_at_time > observed_at:
            raise ValueError("checkpoint 不能来自 snapshot 之后完成的 round")
        checkpoint_id = checkpoint_id_for(fact)
        candidates.append(
            CheckpointCandidate(
                checkpoint_id=checkpoint_id,
                workflow_id=fact.workflow_id,
                lineage_path=linear_lineage_path(fact.run_position),
                token_pos=fact.input_tokens_total,
                memory_bytes=CHECKPOINT_MEMORY_BYTES,
                recurrent_resident=True,
                fa_resident=True,
            )
        )
        checkpoint_metadata.append(
            CheckpointTraceMetadata(
                checkpoint_id=checkpoint_id,
                round_pk=fact.round_pk,
                round_index=fact.round_index,
                run_position=fact.run_position,
                known_at_time=fact.known_at_time,
                input_tokens_total=fact.input_tokens_total,
                current_prefix_tokens=fact.current_prefix_tokens,
            )
        )
    validate_unique_checkpoint_ids(candidates)

    continuations = []
    continuation_metadata = []
    seen_pending_rounds: set[tuple[str, int]] = set()
    for fact in sorted(
        pending_rounds,
        key=lambda item: (
            item.workflow_id,
            item.run_position,
            item.round_pk,
        ),
    ):
        if fact.workflow_id not in active_set:
            raise ValueError("pending 不能来自非 active workflow")
        if fact.known_at_time > observed_at:
            raise ValueError("pending 不能来自 snapshot 之后发出的 tool call")
        if not fact.observed_tool_call_ids:
            raise ValueError("pending 必须由至少一个已发出 tool call 产生")
        pending_key = (fact.workflow_id, fact.round_pk)
        if pending_key in seen_pending_rounds:
            raise ValueError("同一个 round 不能生成多个 LLM-level pending")
        seen_pending_rounds.add(pending_key)
        continuation_id = continuation_id_for(fact)
        continuations.append(
            PendingContinuation(
                continuation_id=continuation_id,
                workflow_id=fact.workflow_id,
                lineage_path=linear_lineage_path(fact.run_position),
                anchor_pos=fact.input_tokens_total,
                resident_fa_frontier=fact.input_tokens_total,
            )
        )
        continuation_metadata.append(
            ContinuationTraceMetadata(
                continuation_id=continuation_id,
                round_pk=fact.round_pk,
                known_at_round=fact.round_index,
                run_position=fact.run_position,
                known_at_time=fact.known_at_time,
                source_input_tokens_total=fact.input_tokens_total,
                current_prefix_tokens=fact.current_prefix_tokens,
                tool_call_count=len(fact.observed_tool_call_ids),
            )
        )

    snapshot = TraceSnapshot(
        snapshot_id=snapshot_id,
        scale=scale,
        time_domain=time_domain,
        observed_at=observed_at,
        active_workflow_ids=active_ids,
        candidates=tuple(candidates),
        continuations=tuple(continuations),
        checkpoint_metadata=tuple(checkpoint_metadata),
        continuation_metadata=tuple(continuation_metadata),
        retention_budgets=retention_budgets(len(candidates)),
    )
    violations = validate_snapshot(snapshot)
    if violations:
        raise ValueError("snapshot 完整性失败：" + "; ".join(violations))
    return snapshot


def validate_snapshot(snapshot: TraceSnapshot) -> tuple[str, ...]:
    """验证无未来信息、线性 lineage 与 workflow 隔离约束。"""
    violations: list[str] = []
    active = set(snapshot.active_workflow_ids)
    checkpoint_metadata = {
        item.checkpoint_id: item for item in snapshot.checkpoint_metadata
    }
    continuation_metadata = {
        item.continuation_id: item
        for item in snapshot.continuation_metadata
    }
    if len(checkpoint_metadata) != len(snapshot.checkpoint_metadata):
        violations.append("checkpoint metadata 标识重复")
    if len(continuation_metadata) != len(snapshot.continuation_metadata):
        violations.append("continuation metadata 标识重复")

    for candidate in snapshot.candidates:
        metadata = checkpoint_metadata.get(candidate.checkpoint_id)
        if metadata is None:
            violations.append(f"checkpoint 缺 metadata：{candidate.checkpoint_id}")
            continue
        if candidate.workflow_id not in active:
            violations.append(f"checkpoint 跨 workflow：{candidate.checkpoint_id}")
        if metadata.known_at_time > snapshot.observed_at:
            violations.append(f"checkpoint 来自未来：{candidate.checkpoint_id}")
        if candidate.token_pos != metadata.input_tokens_total:
            violations.append(f"checkpoint token_pos 不一致：{candidate.checkpoint_id}")
        if candidate.lineage_path != linear_lineage_path(
            metadata.run_position
        ):
            violations.append(f"checkpoint lineage 非线性：{candidate.checkpoint_id}")
        if candidate.memory_bytes != CHECKPOINT_MEMORY_BYTES:
            violations.append(f"checkpoint memory 不一致：{candidate.checkpoint_id}")

    pending_round_keys: set[tuple[str, int]] = set()
    for continuation in snapshot.continuations:
        metadata = continuation_metadata.get(continuation.continuation_id)
        if metadata is None:
            violations.append(
                f"continuation 缺 metadata：{continuation.continuation_id}"
            )
            continue
        if continuation.workflow_id not in active:
            violations.append(
                f"continuation 跨 workflow：{continuation.continuation_id}"
            )
        if metadata.known_at_time > snapshot.observed_at:
            violations.append(
                f"continuation 来自未来：{continuation.continuation_id}"
            )
        if metadata.tool_call_count < 1:
            violations.append(
                f"continuation 无 tool call：{continuation.continuation_id}"
            )
        if continuation.anchor_pos != metadata.source_input_tokens_total:
            violations.append(
                f"continuation anchor 不等于当前输入：{continuation.continuation_id}"
            )
        if continuation.resident_fa_frontier != continuation.anchor_pos:
            violations.append(
                f"逻辑 FA frontier 不等于 anchor：{continuation.continuation_id}"
            )
        if continuation.lineage_path != linear_lineage_path(
            metadata.run_position
        ):
            violations.append(
                f"continuation lineage 非线性：{continuation.continuation_id}"
            )
        pending_key = (continuation.workflow_id, metadata.round_pk)
        if pending_key in pending_round_keys:
            violations.append(
                f"同 round 存在多个 pending：{continuation.continuation_id}"
            )
        pending_round_keys.add(pending_key)

    candidate_count = len(snapshot.candidates)
    for budget in snapshot.retention_budgets:
        expected = max(1, math.floor(candidate_count * budget.ratio))
        if budget.k != expected or budget.k > candidate_count:
            violations.append(f"retention budget 非法：{budget.ratio}")
    if snapshot.future_prefix_used:
        violations.append("future prefix 被用于 online mapping")
    if snapshot.runtime_residency_inferred:
        violations.append("从 trace 推断了 runtime residency")
    if snapshot.llm_level_branching_introduced:
        violations.append("引入了 LLM-level branching")
    return tuple(violations)


def construct_workload(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """从外部 DuckDB 只读生成固定采样的 TraceLab snapshot workload。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    try:
        source_cohort = _single_row(connection, _SOURCE_COHORT_SQL)
        context_buckets = fetch_dicts(connection, _CONTEXT_BUCKET_SQL)
        sampled_rows = fetch_dicts(
            connection,
            _SAMPLED_EVENTS_SQL,
            (str(SAMPLING_SEED),),
        )
        sampled_events = tuple(_sampled_event(row) for row in sampled_rows)
        snapshots = []
        for event in sampled_events:
            known_rows = fetch_dicts(
                connection,
                _KNOWN_SNAPSHOT_STATE_SQL,
                (
                    event.observed_at,
                    event.observed_at,
                    event.observed_at,
                    event.observed_at,
                    event.provider,
                ),
            )
            snapshots.append(_snapshot_from_known_rows(event, known_rows))
    finally:
        connection.close()

    all_violations = tuple(
        violation
        for snapshot in snapshots
        for violation in validate_snapshot(snapshot)
    )
    summary = summarize_workload(snapshots)
    expected_strata = {
        (scale, provider, bucket)
        for scale in SCALE_ORDER
        for provider in ("claude", "codex")
        for bucket in CONTEXT_BUCKET_ORDER
    }
    observed_strata = {
        (event.scale, event.provider, event.context_bucket)
        for event in sampled_events
    }
    missing_strata = tuple(
        {
            "scale": scale,
            "provider": provider,
            "context_bucket": bucket,
        }
        for scale, provider, bucket in sorted(
            expected_strata - observed_strata,
            key=lambda item: (
                SCALE_ORDER.index(item[0]),
                item[1],
                CONTEXT_BUCKET_ORDER.index(item[2]),
            ),
        )
    )
    artifact = {
        "schema_version": "tracelab-flowstate-workload-v1",
        "source": {
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "access_mode": "read_only=True",
            "tables": ("rounds", "timing_events", "tool_calls"),
        },
        "frozen_semantics": {
            "agent_run_boundary": "current_user_message_count > 0",
            "known_anchor": "current_round.input_tokens_total",
            "pending_rule": (
                "当前最新 round 已经发出至少一个 tool call 时，"
                "至多创建一个 LLM-level pending continuation"
            ),
            "lineage_rule": "按真实 round_index 顺序构造线性 tuple prefix",
            "checkpoint_memory_bytes": CHECKPOINT_MEMORY_BYTES,
            "resident_fa_frontier_model": (
                "离线逻辑 snapshot 令 resident_fa_frontier 等于 known anchor；"
                "这不是 TraceLab runtime residency truth"
            ),
            "future_prefix_used": False,
            "runtime_residency_inferred": False,
            "llm_level_branching_introduced": False,
        },
        "sampling_protocol": {
            "seed": SAMPLING_SEED,
            "cohort": "仅 strictly closed Agent Runs",
            "time_domain": "同 provider 内 trace-observed relative timing",
            "scale_bands": {
                "Small": "2-4 个同 provider active runs",
                "Medium": "5-8 个同 provider active runs",
                "Large": "至少 9 个同 provider active runs",
            },
            "strata": (
                "scale × provider × 完整 run 最大 input context bucket"
            ),
            "selection": (
                "每个非空 stratum 选择 SHA-256(seed, provider, bucket, scale, "
                "session, run, round) 字典序最小的 tool-call 时点"
            ),
            "trigger_run_requirement": "至少 2 rounds",
            "future_run_context_used_only_for_sampling_stratum": True,
            "policy_or_cost_signal_used": False,
            "missing_strata": missing_strata,
        },
        "source_cohort": source_cohort,
        "context_buckets": context_buckets,
        "summary": {
            **summary,
            "leakage_violation_count": len(all_violations),
            "leakage_violations": all_violations,
        },
        "snapshots": tuple(_snapshot_to_dict(snapshot) for snapshot in snapshots),
        "gates": {
            "leakage_free_construction": not all_violations,
            "deterministic_workload": True,
            "suitable_for_offline_policy_comparison": "WEAK",
        },
    }
    return _json_value(artifact)


def summarize_workload(
    snapshots: Sequence[TraceSnapshot],
) -> dict[str, Any]:
    """计算 snapshot workload 的完整性与规模统计。"""
    if not snapshots:
        raise ValueError("workload 至少需要一个 snapshot")
    active_counts = [len(snapshot.active_workflow_ids) for snapshot in snapshots]
    candidate_counts = [len(snapshot.candidates) for snapshot in snapshots]
    pending_counts = [len(snapshot.continuations) for snapshot in snapshots]
    anchor_depths = [
        continuation.anchor_pos
        for snapshot in snapshots
        for continuation in snapshot.continuations
    ]
    unique_workflows = {
        workflow_id
        for snapshot in snapshots
        for workflow_id in snapshot.active_workflow_ids
    }
    scale_counts = {
        scale: sum(snapshot.scale == scale for snapshot in snapshots)
        for scale in SCALE_ORDER
    }
    budget_stats = {}
    for ratio in RETENTION_RATIOS:
        values = [
            budget.k
            for snapshot in snapshots
            for budget in snapshot.retention_budgets
            if budget.ratio == ratio
        ]
        budget_stats[f"{int(ratio * 100)}%"] = _distribution(values)
    return {
        "snapshot_count": len(snapshots),
        "scale_snapshot_counts": scale_counts,
        "unique_active_runs_selected": len(unique_workflows),
        "logical_checkpoint_instances": sum(candidate_counts),
        "pending_continuation_instances": sum(pending_counts),
        "active_workflows_per_snapshot": _distribution(active_counts),
        "candidates_per_snapshot": _distribution(candidate_counts),
        "pending_per_snapshot": _distribution(pending_counts),
        "anchor_depth_tokens": _distribution(anchor_depths),
        "budget_k": budget_stats,
    }


def render_report(workload: Mapping[str, Any]) -> str:
    """生成带定义、证据、限制和 gate 的中文技术报告。"""
    cohort = workload["source_cohort"]
    summary = workload["summary"]
    sampling = workload["sampling_protocol"]
    active = summary["active_workflows_per_snapshot"]
    candidates = summary["candidates_per_snapshot"]
    pending = summary["pending_per_snapshot"]
    anchors = summary["anchor_depth_tokens"]
    lines = [
        "# TraceLab 到 FlowState 的无未来泄漏离线 Workload",
        "",
        "## 技术摘要",
        "",
        f"本构造只使用 {cohort['strictly_closed_runs']:,} 个严格闭合 Agent Runs，并从真实 tool-call 观测时点生成 {summary['snapshot_count']} 个 snapshot。所有 online anchor 均严格等于当前轮 `input_tokens_total`，future-prefix leakage violations={summary['leakage_violation_count']}。",
        "",
        f"workload 包含 {summary['logical_checkpoint_instances']:,} 个逻辑 checkpoint 实例和 {summary['pending_continuation_instances']:,} 个 LLM-level pending 实例。它可以供现有 FlowState 数据模型做离线方法比较，但评级为 WEAK：TraceLab 不提供 token IDs、runtime checkpoint residency 或 workflow DAG。",
        "",
        "## 冻结语义没有引入未来字段",
        "",
        "- Agent Run 仍由 `current_user_message_count > 0` 启动。",
        "- tool call round 的 anchor 只取当前 `input_tokens_total`。",
        "- 一个 round 无论发出多少 tool calls，最多生成一个 continuation。",
        "- lineage 仅由真实 `round_index` 执行顺序形成线性 tuple prefix。",
        "- 当前 `prefix_tokens` 只保存在 characterization metadata，不参与 anchor 或 planning target。",
        "- round 完成可知性只依赖 snapshot 前已观察到的 `tool_call`/`usage_report`，或同 run 下一轮已经开始；查询不会把未来 timing event 注入 candidate。",
        "- `recurrent_resident=True`、`fa_resident=True` 和 `resident_fa_frontier=anchor` 是逻辑 snapshot 可消费性假设，不是从 TraceLab 推断出的 GPU residency truth。",
        "",
        "## 严格闭合 cohort 排除了右删失状态",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 原始 rounds | {cohort['source_rounds']:,} |",
        f"| 首个 user-message 前孤立 rounds | {cohort['leading_rounds']:,} |",
        f"| 可分配 Agent Runs | {cohort['all_agent_runs']:,} |",
        f"| 严格闭合 Agent Runs | {cohort['strictly_closed_runs']:,} |",
        f"| 右删失 Agent Runs | {cohort['right_censored_runs']:,} |",
        f"| 严格闭合 rounds | {cohort['strictly_closed_rounds']:,} |",
        f"| 右删失 rounds | {cohort['right_censored_rounds']:,} |",
        f"| 缺少完成事件的严格闭合 rounds | {cohort['closed_rounds_without_completion_event']:,} |",
        "",
        "最后一个 Agent Run 因没有 session-end marker 被排除；首个真实 user-message 边界之前的记录不被强行归入 run。过滤只依赖边界完整性，不依赖任何 policy 或 recovery cost。",
        "",
        "## 三种 scale 来自已观察 concurrency 分布",
        "",
        f"固定 seed 为 `{sampling['seed']}`。Small、Medium、Large 分别对应同 provider 内 2–4、5–8、>=9 个 trace-observed active runs；每个非空 `scale × provider × context bucket` stratum 选择一个固定 SHA-256 键最小的多轮 trigger run 时点。",
        "",
        f"最终 Small={summary['scale_snapshot_counts']['Small']}、Medium={summary['scale_snapshot_counts']['Medium']}、Large={summary['scale_snapshot_counts']['Large']}，合计 {summary['snapshot_count']} 个 snapshots。缺失 strata={len(sampling['missing_strata'])}，均因源数据中不存在对应 concurrency/context/provider 组合，而不是人为补齐或按结果改样。",
        "",
        "完整 run 最大 context 只用于客观采样分层，不进入 snapshot 的 anchor、pending、checkpoint value 或后续 policy 输入。跨 provider 不混合时间域；并发只称为 trace-observed concurrency，不声称精确生产 arrival replay。",
        "",
        "## Snapshot 规模与 anchor 分布",
        "",
        "| 指标 | Mean | Median | P90 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
        _distribution_row("Active workflows", active),
        _distribution_row("Candidates", candidates),
        _distribution_row("Pending", pending),
        _distribution_row("Anchor depth tokens", anchors),
        "",
        f"共有 {summary['unique_active_runs_selected']:,} 个不同 Agent Runs 出现在 sampled snapshots 中。candidate 只能来自 `known_at_time <= snapshot.observed_at` 的完成 rounds；pending 只能来自最新已开始 round 中截至 snapshot 已经观察到的 tool calls。",
        "",
        "## Context buckets 保留全部长上下文 cohort",
        "",
        "| Bucket | Runs | Rounds | Pending signals | Logical candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in workload["context_buckets"]:
        lines.append(
            f"| {bucket['context_bucket']} | {bucket['run_count']:,} | {bucket['round_count']:,} | {bucket['pending_continuation_count']:,} | {bucket['checkpoint_candidate_count']:,} |"
        )
    lines.extend(
        [
            "",
            "`>256K` bucket 没有被删除、裁剪或 downscale。表中的 candidate 是“每个完成 round 一个逻辑 recurrent checkpoint”的 source-cohort 计数，不是 GPU 上实际存在的状态。",
            "",
            "## Budget 只按 retention ratio 生成元数据",
            "",
            "| Ratio | K mean | K median | K P90 | K P95 | K max |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in ("25%", "50%", "75%"):
        row = summary["budget_k"][ratio]
        lines.append(
            f"| {ratio} | {row['mean']:.3f} | {row['median']:.0f} | {row['p90']:.0f} | {row['p95']:.0f} | {row['max']:.0f} |"
        )
    lines.extend(
        [
            "",
            "每个 snapshot 对 N 个 candidates 使用 `max(1, floor(N × ratio))`。本步骤只保存 K，不调用 FlowState、KVFlow、Marconi 或 Oracle。",
            "",
            "## 完整性 Gate 全部通过",
            "",
            "- future-field leakage violations：0",
            "- snapshot 之后完成的 checkpoint：0",
            "- 无已发出 tool call 的 pending：0",
            "- 同一 round 生成多个 LLM-level pending：0",
            "- 非线性或跨 workflow lineage：0",
            "- anchor 与当前轮 input 不一致：0",
            "- future prefix 影响 online mapping：0",
            "- runtime residency inference：0",
            "",
            "## 限制与下一步边界",
            "",
            "该 workload 的 workflow identity、线性 lineage 和 logical checkpoint catalog 是 TraceLab 事实上的确定性离线映射，不是显式 DAG 或 SGLang runtime observation。当前 prefix metadata 可用于描述真实 provider cache reuse，但不能改写 FlowState planning target。",
            "",
            "下一步若进行离线 policy comparison，必须读取已冻结 JSON，不得重采样；应把结果描述为 trace-derived logical snapshot comparison，而不是生产 arrival replay 或 GPU correctness。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    workload: Mapping[str, Any],
    workload_path: Path = DEFAULT_WORKLOAD_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """写入冻结 workload JSON 与中文 Markdown 报告。"""
    workload_path.write_text(
        json.dumps(workload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(workload), encoding="utf-8")


def _event_rank(event: SampledSnapshotEvent) -> tuple[str, int]:
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


def _sampled_event(row: Mapping[str, Any]) -> SampledSnapshotEvent:
    scale = str(row["scale"])
    provider = str(row["provider"])
    bucket = str(row["context_bucket"])
    safe_bucket = (
        bucket.replace("<=", "le-")
        .replace(">", "gt-")
        .replace("K", "k")
    )
    return SampledSnapshotEvent(
        snapshot_id=f"{scale.lower()}-{provider}-{safe_bucket}",
        scale=scale,
        provider=provider,
        context_bucket=bucket,
        trigger_session_id=str(row["session_id"]),
        trigger_run_ordinal=int(row["run_ordinal"]),
        trigger_round_pk=int(row["round_pk"]),
        observed_at=row["observed_at"],
        trace_observed_active_runs=int(row["active_run_count"]),
    )


def _snapshot_from_known_rows(
    event: SampledSnapshotEvent,
    rows: Sequence[Mapping[str, Any]],
) -> TraceSnapshot:
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
            f"active run 数与采样统计不一致：{event.snapshot_id}"
        )
    completed_rounds = []
    pending_rounds = []
    for row in rows:
        workflow_id = workflow_id_for(
            str(row["provider"]),
            str(row["session_id"]),
            int(row["run_ordinal"]),
        )
        known_completion_time = row["known_completion_time"]
        if known_completion_time is not None:
            completed_rounds.append(
                CompletedRoundFact(
                    workflow_id=workflow_id,
                    round_pk=int(row["round_pk"]),
                    round_index=int(row["round_index"]),
                    run_position=int(row["run_position"]),
                    input_tokens_total=int(row["input_tokens_total"]),
                    current_prefix_tokens=int(row["prefix_tokens"]),
                    known_at_time=known_completion_time,
                )
            )
        tool_call_count = int(row["observed_tool_call_count"])
        if bool(row["is_latest_started_round"]) and tool_call_count > 0:
            tool_ids = tuple(str(value) for value in row["observed_tool_call_ids"])
            pending_rounds.append(
                PendingRoundFact(
                    workflow_id=workflow_id,
                    round_pk=int(row["round_pk"]),
                    round_index=int(row["round_index"]),
                    run_position=int(row["run_position"]),
                    input_tokens_total=int(row["input_tokens_total"]),
                    current_prefix_tokens=int(row["prefix_tokens"]),
                    known_at_time=row["first_observed_tool_time"],
                    observed_tool_call_ids=tool_ids,
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


def _snapshot_to_dict(snapshot: TraceSnapshot) -> dict[str, Any]:
    value = asdict(snapshot)
    value["candidate_count"] = len(snapshot.candidates)
    value["pending_continuation_count"] = len(snapshot.continuations)
    value["active_workflow_count"] = len(snapshot.active_workflow_ids)
    return _json_value(value)


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0,
            "p95": 0,
            "max": 0,
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


def _quantile_disc(ordered: Sequence[int], probability: float) -> int:
    if not ordered:
        raise ValueError("quantile 输入不能为空")
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _distribution_row(label: str, values: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {values['mean']:.3f} | {values['median']:.0f} | "
        f"{values['p90']:.0f} | {values['p95']:.0f} | {values['max']:.0f} |"
    )


def _single_row(connection, query: str) -> dict[str, Any]:
    rows = fetch_dicts(connection, query)
    if len(rows) != 1:
        raise ValueError("预期查询恰好返回一行")
    return rows[0]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


_ASSIGNED_RUNS_CTE = """
WITH ordered AS (
    SELECT
        r.*,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id
                ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds r
), assigned AS (
    SELECT * FROM ordered WHERE run_ordinal > 0
), run_counts AS (
    SELECT
        provider,
        session_id,
        run_ordinal,
        count(*) AS round_count,
        max(input_tokens_total) AS max_input_tokens_total
    FROM assigned
    GROUP BY provider, session_id, run_ordinal
), classified_runs AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM run_counts
)
"""


_SOURCE_COHORT_SQL = _ASSIGNED_RUNS_CTE + """
, output_events AS (
    SELECT DISTINCT round_pk
    FROM timing_events
    WHERE event_type IN ('reasoning', 'text', 'tool_call', 'usage_report')
), session_counts AS (
    SELECT
        session_id,
        count(*) FILTER (WHERE current_user_message_count > 0) AS run_count
    FROM rounds
    GROUP BY session_id
)
SELECT
    (SELECT count(*) FROM rounds) AS source_rounds,
    (SELECT count(*) FROM ordered WHERE run_ordinal = 0) AS leading_rounds,
    (SELECT count(*) FROM classified_runs) AS all_agent_runs,
    (SELECT count(*) FROM classified_runs WHERE is_strictly_closed)
        AS strictly_closed_runs,
    (SELECT count(*) FROM classified_runs WHERE NOT is_strictly_closed)
        AS right_censored_runs,
    (
        SELECT count(*) FROM assigned a
        JOIN classified_runs c USING (provider, session_id, run_ordinal)
        WHERE c.is_strictly_closed
    ) AS strictly_closed_rounds,
    (
        SELECT count(*) FROM assigned a
        JOIN classified_runs c USING (provider, session_id, run_ordinal)
        WHERE NOT c.is_strictly_closed
    ) AS right_censored_rounds,
    (
        SELECT count(*) FROM assigned a
        JOIN classified_runs c USING (provider, session_id, run_ordinal)
        LEFT JOIN output_events o USING (round_pk)
        WHERE c.is_strictly_closed AND o.round_pk IS NULL
    ) AS closed_rounds_without_completion_event,
    (SELECT count(*) FROM session_counts WHERE run_count = 0)
        AS sessions_without_runs
"""


_CONTEXT_BUCKET_SQL = _ASSIGNED_RUNS_CTE + """
, tool_rounds AS (
    SELECT DISTINCT round_pk FROM tool_calls
), bucketed AS (
    SELECT
        c.provider,
        c.session_id,
        c.run_ordinal,
        c.round_count,
        c.max_input_tokens_total,
        count(*) FILTER (WHERE t.round_pk IS NOT NULL)
            AS pending_continuation_count
    FROM classified_runs c
    JOIN assigned a USING (provider, session_id, run_ordinal)
    LEFT JOIN tool_rounds t USING (round_pk)
    WHERE c.is_strictly_closed
    GROUP BY
        c.provider, c.session_id, c.run_ordinal,
        c.round_count, c.max_input_tokens_total
), labeled AS (
    SELECT
        *,
        CASE
            WHEN max_input_tokens_total <= 32768 THEN '<=32K'
            WHEN max_input_tokens_total <= 65536 THEN '32K-64K'
            WHEN max_input_tokens_total <= 131072 THEN '64K-128K'
            WHEN max_input_tokens_total <= 262144 THEN '128K-256K'
            ELSE '>256K'
        END AS context_bucket
    FROM bucketed
)
SELECT
    context_bucket,
    count(*) AS run_count,
    sum(round_count) AS round_count,
    sum(pending_continuation_count) AS pending_continuation_count,
    sum(round_count) AS checkpoint_candidate_count
FROM labeled
GROUP BY context_bucket
ORDER BY CASE context_bucket
    WHEN '<=32K' THEN 1
    WHEN '32K-64K' THEN 2
    WHEN '64K-128K' THEN 3
    WHEN '128K-256K' THEN 4
    ELSE 5
END
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
), raw_run_bounds AS (
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
), run_bounds AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM raw_run_bounds
), tool_events AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        max(tc.emitted_at) AS observed_at
    FROM assigned a
    JOIN tool_calls tc USING (round_pk)
    JOIN run_bounds r USING (provider, session_id, run_ordinal)
    WHERE r.is_strictly_closed AND tc.emitted_at IS NOT NULL
    GROUP BY a.provider, a.session_id, a.run_ordinal, a.round_pk
), event_concurrency AS (
    SELECT
        e.provider,
        e.session_id,
        e.run_ordinal,
        e.round_pk,
        e.observed_at,
        trigger.round_count,
        trigger.max_input_tokens_total,
        count(active.session_id) AS active_run_count
    FROM tool_events e
    JOIN run_bounds trigger
      USING (provider, session_id, run_ordinal)
    JOIN run_bounds active
      ON active.provider = e.provider
     AND active.is_strictly_closed
     AND active.start_time <= e.observed_at
     AND active.end_time >= e.observed_at
    WHERE trigger.round_count >= 2
    GROUP BY
        e.provider, e.session_id, e.run_ordinal, e.round_pk,
        e.observed_at, trigger.round_count, trigger.max_input_tokens_total
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
    FROM event_concurrency
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
        WHEN '128K-256K' THEN 4
        ELSE 5
    END
"""


_KNOWN_SNAPSHOT_STATE_SQL = """
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
), raw_run_bounds AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        min(t.timestamp) AS start_time,
        max(t.timestamp) AS end_time
    FROM assigned a
    JOIN timing_events t USING (round_pk)
    GROUP BY a.provider, a.session_id, a.run_ordinal
), run_bounds AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM raw_run_bounds
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
        rt.started_at,
        rt.completion_marker_at,
        coalesce(ot.observed_tool_call_count, 0)
            AS observed_tool_call_count,
        coalesce(ot.observed_tool_call_ids, [])
            AS observed_tool_call_ids,
        ot.first_observed_tool_time
    FROM assigned a
    JOIN run_bounds rb USING (provider, session_id, run_ordinal)
    JOIN observed_round_events rt USING (round_pk)
    LEFT JOIN observed_tools ot USING (round_pk)
    WHERE rb.is_strictly_closed
      AND rb.start_time <= ?
      AND rb.end_time >= ?
      AND rb.provider = ?
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
), marked AS (
    SELECT
        *,
        coalesce(completion_marker_at, next_observed_round_started_at)
            AS known_completion_time
    FROM sequenced
)
SELECT *
FROM marked
ORDER BY provider, session_id, run_ordinal, run_position
"""


def main(argv: Sequence[str] | None = None) -> int:
    """执行只读构造并写入冻结 workload 与技术报告。"""
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
        default=DEFAULT_WORKLOAD_PATH,
        help="workload JSON 输出路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown 报告输出路径",
    )
    arguments = parser.parse_args(argv)
    workload = construct_workload(arguments.database)
    write_artifacts(workload, arguments.output, arguments.report)
    print(json.dumps(workload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
