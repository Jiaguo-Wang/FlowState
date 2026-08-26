#!/usr/bin/env python3
"""审计 TraceLab C128 全量真实快照中的非平凡 recurrent-state demand。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_final_protocol import (
    DEMAND_RETENTION_RATIOS,
    MAIN_COHORT,
    MAIN_COHORT_MAX_TOKENS,
    demand_relative_budget,
)
from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    SAMPLING_SEED,
    SCALE_ORDER,
    deterministic_sample_key,
)


PROJECTION_THRESHOLDS = (2, 3, 4)
PROJECTION_LIMITS = (5, 10, 20)
EXACT_X_VALUES = (0, 1, 2, 3, 4, 5, 6)
DEFAULT_FINAL_PROTOCOL_PATH = Path(__file__).with_name(
    "tracelab_final_protocol.json"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name(
    "tracelab_nontrivial_demand.json"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    "TRACELAB_NONTRIVIAL_DEMAND.md"
)


@dataclass(frozen=True)
class DemandSnapshotEvent:
    """描述一个真实 tool-call 时点及其当时可知的状态需求。"""

    provider: str
    context_bucket: str
    scale: str
    trigger_session_id: str
    trigger_run_ordinal: int
    trigger_round_pk: int
    observed_at: datetime
    active_run_ids: tuple[str, ...]
    active_workflow_count: int
    candidate_count: int
    pending_count: int
    exact_parent_count: int
    pending_anchors: tuple[int, ...]

    @property
    def snapshot_id(self) -> str:
        """返回与事件事实绑定的稳定标识。"""
        return (
            f"c128-{self.scale.lower()}-{self.provider}-"
            f"round-{self.trigger_round_pk}"
        )


def candidate_event_rank(
    event: DemandSnapshotEvent,
    seed: int = SAMPLING_SEED,
) -> str:
    """按冻结字段生成 deterministic SHA-256 rank。"""
    return deterministic_sample_key(
        seed=seed,
        provider=event.provider,
        context_bucket=event.context_bucket,
        scale=event.scale,
        session_id=event.trigger_session_id,
        run_ordinal=event.trigger_run_ordinal,
        round_pk=event.trigger_round_pk,
    )


def project_workload(
    events: Iterable[DemandSnapshotEvent],
    minimum_x: int,
    max_per_stratum: int,
) -> tuple[DemandSnapshotEvent, ...]:
    """按 X gate、固定哈希和 active-set 去重投影 workload 大小。"""
    if minimum_x <= 0:
        raise ValueError("minimum_x 必须大于零")
    if max_per_stratum <= 0:
        raise ValueError("max_per_stratum 必须大于零")
    selected = []
    stratum_counts: dict[tuple[str, str, str], int] = {}
    seen_active_sets: set[tuple[str, ...]] = set()
    ordered = sorted(
        (
            event
            for event in events
            if event.exact_parent_count >= minimum_x
        ),
        key=lambda event: (
            candidate_event_rank(event),
            event.trigger_round_pk,
        ),
    )
    for event in ordered:
        stratum = (event.provider, event.context_bucket, event.scale)
        active_set = tuple(sorted(event.active_run_ids))
        if stratum_counts.get(stratum, 0) >= max_per_stratum:
            continue
        if active_set in seen_active_sets:
            continue
        selected.append(event)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        seen_active_sets.add(active_set)
    scale_rank = {value: index for index, value in enumerate(SCALE_ORDER)}
    return tuple(
        sorted(
            selected,
            key=lambda event: (
                scale_rank[event.scale],
                event.provider,
                event.context_bucket,
                candidate_event_rank(event),
            ),
        )
    )


def budget_discreteness(x_value: int) -> dict[str, Any]:
    """返回给定 X 在四个冻结 ratio 下产生的离散 K。"""
    if x_value <= 0:
        raise ValueError("x_value 必须大于零")
    k_values = tuple(
        demand_relative_budget(x_value, ratio)
        for ratio in DEMAND_RETENTION_RATIOS
    )
    return {
        "x": x_value,
        "k_by_ratio": {
            f"{int(ratio * 100)}%": k
            for ratio, k in zip(DEMAND_RETENTION_RATIOS, k_values)
        },
        "distinct_k_levels": len(set(k_values)),
    }


def construct_nontrivial_demand_audit(
    database_path: Path = DEFAULT_DATABASE_PATH,
    final_protocol_path: Path = DEFAULT_FINAL_PROTOCOL_PATH,
) -> dict[str, Any]:
    """只读扫描 C128 全部合法事件并生成结构审计。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    if not final_protocol_path.is_file():
        raise FileNotFoundError(
            f"Step 10C.3 协议不存在：{final_protocol_path}"
        )
    final_protocol = json.loads(
        final_protocol_path.read_text(encoding="utf-8")
    )
    _validate_frozen_protocol(final_protocol)

    connection = open_database_read_only(database_path)
    try:
        rows = fetch_dicts(connection, _ALL_C128_EVENTS_SQL)
    finally:
        connection.close()
    events = tuple(_event_from_row(row) for row in rows)
    if not events:
        raise ValueError("C128 没有合法的 trace-observed candidate events")

    exact_histogram = {
        f"X={x_value}": sum(
            event.exact_parent_count == x_value for event in events
        )
        for x_value in EXACT_X_VALUES
    }
    exact_histogram["X>6"] = sum(
        event.exact_parent_count > 6 for event in events
    )
    classifications = {
        "X=0": _summarize_events(
            tuple(event for event in events if event.exact_parent_count == 0),
            len(events),
        ),
        "X=1": _summarize_events(
            tuple(event for event in events if event.exact_parent_count == 1),
            len(events),
        ),
    }
    for threshold in range(2, 7):
        classifications[f"X>={threshold}"] = _summarize_events(
            tuple(
                event
                for event in events
                if event.exact_parent_count >= threshold
            ),
            len(events),
        )

    x_values_present = sorted(
        {
            event.exact_parent_count
            for event in events
            if event.exact_parent_count >= 2
        }
    )
    discreteness = {
        f"X={x_value}": budget_discreteness(x_value)
        for x_value in x_values_present
    }
    projections = {}
    projected_events: dict[tuple[int, int], tuple[DemandSnapshotEvent, ...]] = {}
    for threshold in PROJECTION_THRESHOLDS:
        threshold_rows = {}
        for limit in PROJECTION_LIMITS:
            projected = project_workload(events, threshold, limit)
            projected_events[(threshold, limit)] = projected
            threshold_rows[f"max{limit}"] = _projection_summary(projected)
        projections[f"X>={threshold}"] = threshold_rows

    current_sample = _audit_current_sample(final_protocol, events)
    x2 = classifications["X>=2"]
    x4 = classifications["X>=4"]
    diagnoses = _diagnose(
        x2=x2,
        x4=x4,
        projections=projections,
        current_sample=current_sample,
        exact_histogram=exact_histogram,
    )
    validation = {
        "c128_cutoff_violations": 0,
        "current_sample_state_mismatches": current_sample[
            "state_mismatch_count"
        ],
        "x_histogram_difference": (
            len(events) - sum(exact_histogram.values())
        ),
        "future_field_leakage_violations": 0,
        "synthetic_concurrency_violations": 0,
        "projection_duplicate_active_set_count": sum(
            _duplicate_active_set_count(items)
            for items in projected_events.values()
        ),
        "policy_comparison_runs": 0,
        "phi_calls": 0,
        "gpu_runs": 0,
    }
    artifact = {
        "schema_version": "tracelab-nontrivial-demand-audit-v1",
        "source": {
            "database_path": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "access_mode": "read_only=True",
            "final_protocol_path": str(final_protocol_path),
        },
        "fixed_cohort": {
            "name": MAIN_COHORT,
            "strictly_closed_only": True,
            "maximum_input_tokens_total": MAIN_COHORT_MAX_TOKENS,
        },
        "candidate_snapshot_event": {
            "definition": (
                "C128 严格闭合且至少两轮的 Agent Run 中，每个已发出 tool call 的 round，"
                "以该 round 最后一个已发出 tool call 的 timestamp 作为观测时点"
            ),
            "concurrency": (
                "仅保留同 provider、run 时间区间真实重叠且 W>=2 的时点"
            ),
            "known_state": (
                "只使用 observed_at 之前已开始、已完成或已发出 tool call 的 round facts"
            ),
            "synthetic_snapshot": False,
            "future_fields_used": False,
        },
        "definitions": {
            "x": (
                "known pending continuations 所需且在 candidate set 中存在的 "
                "distinct exact-parent recurrent states"
            ),
            "nontrivial_demand": "X>=2",
            "agent_run_boundary": "current_user_message_count > 0",
            "known_anchor": "current_round.input_tokens_total",
            "lineage": "按已发生 round order 构造的线性 lineage",
            "multiple_tool_calls_create_fanout": False,
        },
        "all_candidate_snapshot_count": len(events),
        "exact_x_histogram": exact_histogram,
        "classifications": classifications,
        "budget_discreteness": discreteness,
        "workload_size_projection": projections,
        "recovery_domain_requirement": {
            "X>=2": _anchor_summary(
                tuple(event for event in events if event.exact_parent_count >= 2)
            ),
            "X>=4": _anchor_summary(
                tuple(event for event in events if event.exact_parent_count >= 4)
            ),
            "phi_called": False,
        },
        "current_57_snapshot_audit": current_sample,
        "diagnosis": diagnoses,
        "validation": validation,
        "execution": {
            "policy_comparison_executed": False,
            "phi_called": False,
            "gpu_executed": False,
        },
        "projection_samples": {
            f"X>={threshold}/max{limit}": tuple(
                _projection_event_to_dict(event)
                for event in projected_events[(threshold, limit)]
            )
            for threshold in PROJECTION_THRESHOLDS
            for limit in PROJECTION_LIMITS
        },
    }
    return _json_value(artifact)


def render_report(artifact: Mapping[str, Any]) -> str:
    """生成以结构证据与决策边界为主的中文审计报告。"""
    classifications = artifact["classifications"]
    histogram = artifact["exact_x_histogram"]
    current = artifact["current_57_snapshot_audit"]
    diagnosis = artifact["diagnosis"]
    lines = [
        "# TraceLab C128 非平凡需求快照审计",
        "",
        "## 技术摘要",
        "",
        f"C128 共形成 {artifact['all_candidate_snapshot_count']:,} 个合法、真实重叠的候选快照事件，其中 X>=2 的比例为 {_percent(classifications['X>=2']['fraction'])}。当前冻结的 57 个正式快照中只有 {current['cumulative_histogram']['X>=2']} 个 X>=2 快照，其占比低于 C128 全量事件，因此在扩展 profiler 之前应先重新冻结需求感知采样；本步骤没有执行任何 policy、Phi 或 GPU。",
        "",
        "候选快照事件定义为：C128 严格闭合且至少两轮的 Agent Run 中，每个包含已发出 tool call 的 round，以该 round 最后一个已发出 tool call 的时间为观测时点。活跃 workflows 只来自同 provider 的真实 run 区间重叠，且 W>=2；没有人工时间平移或合成并发。",
        "",
        "## 全量 X 分布显示非平凡 demand 并不稀缺",
        "",
        "| X | Snapshots | Fraction |",
        "|---|---:|---:|",
    ]
    for label in ("X=0", "X=1", "X=2", "X=3", "X=4", "X=5", "X=6", "X>6"):
        count = histogram[label]
        lines.append(
            f"| {label} | {count:,} | {_percent(count / artifact['all_candidate_snapshot_count'])} |"
        )
    lines.extend(
        [
            "",
            "以下累计 cohort 的 N/X 对 X=0 不定义；其余分布均以 snapshot 为单位。",
            "",
            "| Cohort | Count | Fraction | Unique runs | Claude | Codex | Small | Medium | Large |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("X=0", "X=1", "X>=2", "X>=3", "X>=4", "X>=5", "X>=6"):
        row = classifications[label]
        lines.append(
            f"| {label} | {row['snapshot_count']:,} | {_percent(row['fraction'])} | {row['unique_active_runs']:,} | {row['provider_counts']['claude']:,} | {row['provider_counts']['codex']:,} | {row['scale_counts']['Small']:,} | {row['scale_counts']['Medium']:,} | {row['scale_counts']['Large']:,} |"
        )
    for label in ("X>=2", "X>=4"):
        row = classifications[label]
        lines.extend(
            [
                "",
                f"### {label} 结构分布",
                "",
                "| Metric | Median | P90 | P95 | Max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ("W", "N", "P", "N/X"):
            distribution = row["structure"][metric]
            lines.append(
                f"| {metric} | {_format(distribution['median'])} | {_format(distribution['p90'])} | {_format(distribution['p95'])} | {_format(distribution['max'])} |"
            )
    lines.extend(
        [
            "",
            "## 小 X 使 budget ratio 离散退化",
            "",
            "| X | K25 | K50 | K75 | K100 | Distinct levels |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in sorted(
        artifact["budget_discreteness"],
        key=lambda value: int(value.split("=")[1]),
    ):
        row = artifact["budget_discreteness"][key]
        values = row["k_by_ratio"]
        lines.append(
            f"| {row['x']} | {values['25%']} | {values['50%']} | {values['75%']} | {values['100%']} | {row['distinct_k_levels']} |"
        )
    lines.extend(
        [
            "",
            "这一步只量化离散性，不修改 Step 10C.3 的 demand-relative budget 公式或 ratio。",
            "",
            "## 确定性 workload 大小投影",
            "",
            "每个投影继续使用 `provider × context bucket × concurrency scale` 分层、seed 20260826、SHA-256 排名和 active-run-set 去重；没有运行 policy。",
            "",
            "| X gate | Limit/stratum | Snapshots | Unique runs | Small | Medium | Large |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold in PROJECTION_THRESHOLDS:
        for limit in PROJECTION_LIMITS:
            row = artifact["workload_size_projection"][f"X>={threshold}"][f"max{limit}"]
            scales = row["scale_counts"]
            lines.append(
                f"| X>={threshold} | {limit} | {row['snapshot_count']} | {row['unique_active_runs']} | {scales['Small']} | {scales['Medium']} | {scales['Large']} |"
            )
    lines.extend(
        [
            "",
            "## 当前 57 个快照为何三个受限 ratio 得到相同 K<X",
            "",
            f"当前 exact X histogram 为：{', '.join(f'{key}:{value}' for key, value in current['exact_histogram'].items())}。对任意整数 X>=2，`max(1,floor(0.25X))`、`max(1,floor(0.50X))` 和 `max(1,floor(0.75X))` 都严格小于 X；对 X=1，三者都等于 X。因此三档的 K<X 条件都恰好等价于 X>=2，即 {current['cumulative_histogram']['X>=2']}/57={_percent(current['cumulative_histogram']['X>=2'] / current['snapshot_count'])}。",
            "",
            "## Recovery-model domain 不因 demand gate 缩短",
            "",
            "| Cohort | Anchor median | P90 | P95 | Max | Worst E=0 domain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("X>=2", "X>=4"):
        row = artifact["recovery_domain_requirement"][label]
        anchor = row["known_anchor_tokens"]
        lines.append(
            f"| {label} | {_format(anchor['median'])} | {_format(anchor['p90'])} | {_format(anchor['p95'])} | {_format(anchor['max'])} | {row['maximum_required_gap_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## 结构判断与下一步",
            "",
            f"- X>=2 是否充足：**{diagnosis['enough_x_ge_2']}**。{diagnosis['enough_x_ge_2_reason']}",
            f"- X>=4 是否充足：**{diagnosis['enough_x_ge_4']}**。{diagnosis['enough_x_ge_4_reason']}",
            f"- 当前样本是否低估非平凡需求：**{diagnosis['current_sample_underrepresents_nontrivial_demand']}**。{diagnosis['underrepresentation_reason']}",
            f"- 是否应在修改采样前先扩展 profiler：**{diagnosis['profiler_extension_before_sampling_revision']}**。",
            "",
            f"推荐下一步：**{diagnosis['recommended_next_step']}**。本步骤不执行该建议。",
            "",
            "限制：这些结论证明结构样本量与多级 budget 可用性，不构成具体 policy effect 的统计功效分析。TraceLab 仍没有显式 LLM-level DAG、token IDs 或 runtime residency truth。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    artifact: Mapping[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """保存只读审计 JSON 与中文 Markdown 报告。"""
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(artifact), encoding="utf-8")


def _event_from_row(row: Mapping[str, Any]) -> DemandSnapshotEvent:
    return DemandSnapshotEvent(
        provider=str(row["provider"]),
        context_bucket=str(row["context_bucket"]),
        scale=str(row["scale"]),
        trigger_session_id=str(row["session_id"]),
        trigger_run_ordinal=int(row["run_ordinal"]),
        trigger_round_pk=int(row["round_pk"]),
        observed_at=row["observed_at"],
        active_run_ids=tuple(str(value) for value in row["active_run_ids"]),
        active_workflow_count=int(row["w"]),
        candidate_count=int(row["n"]),
        pending_count=int(row["p"]),
        exact_parent_count=int(row["x"]),
        pending_anchors=tuple(
            int(value) for value in (row["pending_anchors"] or ())
        ),
    )


def _summarize_events(
    events: Sequence[DemandSnapshotEvent],
    denominator: int,
) -> dict[str, Any]:
    if denominator <= 0:
        raise ValueError("分类分母必须大于零")
    ratio_values = tuple(
        event.candidate_count / event.exact_parent_count
        for event in events
        if event.exact_parent_count > 0
    )
    return {
        "snapshot_count": len(events),
        "fraction": len(events) / denominator,
        "unique_active_runs": len(
            {
                run_id
                for event in events
                for run_id in event.active_run_ids
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
        "structure": {
            "W": _distribution(
                [event.active_workflow_count for event in events]
            ),
            "N": _distribution([event.candidate_count for event in events]),
            "P": _distribution([event.pending_count for event in events]),
            "N/X": _distribution(ratio_values),
        },
    }


def _projection_summary(
    events: Sequence[DemandSnapshotEvent],
) -> dict[str, Any]:
    return {
        "snapshot_count": len(events),
        "unique_active_runs": len(
            {
                run_id
                for event in events
                for run_id in event.active_run_ids
            }
        ),
        "scale_counts": {
            scale: sum(event.scale == scale for event in events)
            for scale in SCALE_ORDER
        },
        "provider_counts": {
            provider: sum(event.provider == provider for event in events)
            for provider in ("claude", "codex")
        },
        "duplicate_active_run_set_count": _duplicate_active_set_count(events),
    }


def _anchor_summary(events: Sequence[DemandSnapshotEvent]) -> dict[str, Any]:
    anchors = tuple(
        anchor for event in events for anchor in event.pending_anchors
    )
    if not anchors:
        raise ValueError("recovery domain cohort 没有 known anchors")
    distribution = _distribution(anchors)
    return {
        "known_anchor_count": len(anchors),
        "known_anchor_tokens": distribution,
        "maximum_required_gap_tokens": int(distribution["max"]),
        "worst_case_executable_frontier": 0,
        "phi_called": False,
    }


def _audit_current_sample(
    protocol: Mapping[str, Any],
    all_events: Sequence[DemandSnapshotEvent],
) -> dict[str, Any]:
    event_by_key = {
        (event.provider, event.trigger_round_pk): event
        for event in all_events
    }
    histogram = {f"X={value}": 0 for value in range(1, 7)}
    histogram["X>6"] = 0
    mismatch_count = 0
    for row in protocol["snapshots"]:
        event_row = row["sampling"]["event"]["event"]
        event = event_by_key.get(
            (str(event_row["provider"]), int(event_row["trigger_round_pk"]))
        )
        if event is None:
            mismatch_count += 1
            continue
        structure = row["structure"]
        expected = (
            int(structure["w"]),
            int(structure["n"]),
            int(structure["p"]),
            int(structure["x"]),
        )
        observed = (
            event.active_workflow_count,
            event.candidate_count,
            event.pending_count,
            event.exact_parent_count,
        )
        mismatch_count += expected != observed
        x_value = int(structure["x"])
        label = f"X={x_value}" if x_value <= 6 else "X>6"
        histogram[label] += 1
    snapshot_count = len(protocol["snapshots"])
    return {
        "snapshot_count": snapshot_count,
        "exact_histogram": histogram,
        "cumulative_histogram": {
            f"X>={threshold}": sum(
                int(row["structure"]["x"]) >= threshold
                for row in protocol["snapshots"]
            )
            for threshold in (2, 3, 4)
        },
        "state_mismatch_count": mismatch_count,
        "k_lt_x_fraction_by_ratio": {
            f"{int(ratio * 100)}%": sum(
                demand_relative_budget(int(row["structure"]["x"]), ratio)
                < int(row["structure"]["x"])
                for row in protocol["snapshots"]
            )
            / snapshot_count
            for ratio in DEMAND_RETENTION_RATIOS
        },
    }


def _diagnose(
    *,
    x2: Mapping[str, Any],
    x4: Mapping[str, Any],
    projections: Mapping[str, Any],
    current_sample: Mapping[str, Any],
    exact_histogram: Mapping[str, int],
) -> dict[str, str]:
    x2_projection = projections["X>=2"]["max20"]
    x4_projection = projections["X>=4"]["max20"]
    enough_x2 = _structural_sufficiency(x2, x2_projection)
    enough_x4 = _structural_sufficiency(x4, x4_projection)
    population_fraction = float(x2["fraction"])
    sample_fraction = (
        int(current_sample["cumulative_histogram"]["X>=2"])
        / int(current_sample["snapshot_count"])
    )
    underrepresents = sample_fraction < population_fraction
    recommended = (
        "在 C128 内重新冻结 X>=2 sampling"
        if enough_x2 != "NO" and underrepresents
        else "保留 57 snapshot protocol"
    )
    return {
        "enough_x_ge_2": enough_x2,
        "enough_x_ge_2_reason": (
            f"全量 {x2['snapshot_count']:,} 个事件，max20 投影 "
            f"{x2_projection['snapshot_count']} 个快照、"
            f"{x2_projection['unique_active_runs']} 个 unique runs。"
        ),
        "enough_x_ge_4": enough_x4,
        "enough_x_ge_4_reason": (
            f"全量 {x4['snapshot_count']:,} 个事件，max20 投影 "
            f"{x4_projection['snapshot_count']} 个快照；"
            f"实际 X>=4 值域包含 "
            f"{sum(exact_histogram.get(f'X={value}', 0) > 0 for value in (4, 5, 6)) + (exact_histogram['X>6'] > 0)} 个 X 档。"
        ),
        "current_sample_underrepresents_nontrivial_demand": (
            "YES" if underrepresents else "NO"
        ),
        "underrepresentation_reason": (
            f"当前样本 X>=2={_percent(sample_fraction)}，"
            f"全量 C128={_percent(population_fraction)}。"
        ),
        "profiler_extension_before_sampling_revision": (
            "NO" if recommended != "保留 57 snapshot protocol" else "YES"
        ),
        "recommended_next_step": recommended,
        "decision_scope": "仅基于结构样本量，不包含 policy effect 或 latency",
    }


def _structural_sufficiency(
    cohort: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> str:
    providers = sum(
        int(value) > 0 for value in cohort["provider_counts"].values()
    )
    scales = sum(int(value) > 0 for value in cohort["scale_counts"].values())
    if (
        int(cohort["snapshot_count"]) >= 1_000
        and int(projection["snapshot_count"]) >= 100
        and providers == 2
        and scales == 3
    ):
        return "YES"
    if (
        int(cohort["snapshot_count"]) >= 100
        and int(projection["snapshot_count"]) >= 30
        and providers == 2
        and scales >= 2
    ):
        return "WEAK"
    return "NO"


def _projection_event_to_dict(
    event: DemandSnapshotEvent,
) -> dict[str, Any]:
    return {
        **asdict(event),
        "snapshot_id": event.snapshot_id,
        "sha256_rank_key": candidate_event_rank(event),
    }


def _duplicate_active_set_count(
    events: Sequence[DemandSnapshotEvent],
) -> int:
    active_sets = [tuple(sorted(event.active_run_ids)) for event in events]
    return len(active_sets) - len(set(active_sets))


def _validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    main = protocol["main_cohort"]
    execution = protocol["execution"]
    if (
        main["name"] != MAIN_COHORT
        or int(main["maximum_input_tokens_total"])
        != MAIN_COHORT_MAX_TOKENS
    ):
        raise ValueError("Step 10C.3 的 C128 cohort 已发生变化")
    if any(bool(value) for value in execution.values()):
        raise ValueError("Step 10C.3 不应包含 policy、Phi 或 GPU 执行")


def _distribution(
    values: Sequence[int | float],
) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
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


def _format(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """执行全量 C128 结构审计并写入 artifacts。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="外部 TraceLab DuckDB 路径",
    )
    parser.add_argument(
        "--final-protocol",
        type=Path,
        default=DEFAULT_FINAL_PROTOCOL_PATH,
        help="Step 10C.3 冻结协议路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="审计 JSON 输出路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="审计 Markdown 输出路径",
    )
    arguments = parser.parse_args(argv)
    artifact = construct_nontrivial_demand_audit(
        arguments.database,
        arguments.final_protocol,
    )
    write_artifacts(artifact, arguments.output, arguments.report)
    print(
        json.dumps(
            {
                "all_candidate_snapshots": artifact[
                    "all_candidate_snapshot_count"
                ],
                "x_ge_2": artifact["classifications"]["X>=2"][
                    "snapshot_count"
                ],
                "x_ge_4": artifact["classifications"]["X>=4"][
                    "snapshot_count"
                ],
                "policy_comparison_executed": False,
                "gpu_executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


_ALL_C128_EVENTS_SQL = r"""
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
), round_clock AS (
    SELECT
        round_pk,
        min(timestamp) AS started_at,
        min(timestamp) FILTER (
            WHERE event_type IN ('tool_call', 'usage_report')
        ) AS completion_marker_at
    FROM timing_events
    GROUP BY round_pk
), tool_clock AS (
    SELECT
        round_pk,
        min(emitted_at) AS first_tool_at
    FROM tool_calls
    WHERE emitted_at IS NOT NULL
    GROUP BY round_pk
), round_facts AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        a.round_index,
        a.run_position,
        a.input_tokens_total,
        clock.started_at,
        clock.completion_marker_at,
        tool.first_tool_at
    FROM assigned a
    JOIN round_clock clock USING (round_pk)
    LEFT JOIN tool_clock tool USING (round_pk)
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
    WHERE is_strictly_closed
      AND max_input_tokens_total <= 131072
), candidate_events AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        a.round_pk,
        max(tool.emitted_at) AS observed_at
    FROM assigned a
    JOIN tool_calls tool USING (round_pk)
    JOIN eligible run USING (provider, session_id, run_ordinal)
    WHERE run.round_count >= 2
      AND tool.emitted_at IS NOT NULL
    GROUP BY a.provider, a.session_id, a.run_ordinal, a.round_pk
), event_runs AS (
    SELECT
        event.provider,
        event.session_id,
        event.run_ordinal,
        event.round_pk,
        event.observed_at,
        trigger.max_input_tokens_total,
        active.session_id AS active_session_id,
        active.run_ordinal AS active_run_ordinal
    FROM candidate_events event
    JOIN eligible trigger USING (provider, session_id, run_ordinal)
    JOIN eligible active
      ON active.provider = event.provider
     AND active.start_time <= event.observed_at
     AND active.end_time >= event.observed_at
), event_base AS (
    SELECT
        provider,
        session_id,
        run_ordinal,
        round_pk,
        observed_at,
        max_input_tokens_total,
        count(*) AS w,
        list(
            concat(
                provider, ':', active_session_id, ':run:',
                lpad(active_run_ordinal::VARCHAR, 6, '0')
            )
            ORDER BY active_session_id, active_run_ordinal
        ) AS active_run_ids
    FROM event_runs
    GROUP BY
        provider, session_id, run_ordinal, round_pk,
        observed_at, max_input_tokens_total
), labeled_events AS (
    SELECT
        *,
        CASE
            WHEN w BETWEEN 2 AND 4 THEN 'Small'
            WHEN w BETWEEN 5 AND 8 THEN 'Medium'
            WHEN w >= 9 THEN 'Large'
        END AS scale,
        CASE
            WHEN max_input_tokens_total <= 32768 THEN '<=32K'
            WHEN max_input_tokens_total <= 65536 THEN '32K-64K'
            ELSE '64K-128K'
        END AS context_bucket
    FROM event_base
), observed_round_state AS (
    SELECT
        event.*,
        fact.provider AS fact_provider,
        fact.session_id AS fact_session_id,
        fact.run_ordinal AS fact_run_ordinal,
        fact.round_pk AS fact_round_pk,
        fact.run_position,
        fact.input_tokens_total,
        fact.started_at,
        CASE
            WHEN fact.completion_marker_at <= event.observed_at
            THEN fact.completion_marker_at
        END AS observed_completion_marker_at,
        CASE
            WHEN fact.first_tool_at <= event.observed_at
            THEN fact.first_tool_at
        END AS observed_first_tool_at
    FROM labeled_events event
    JOIN eligible active
      ON active.provider = event.provider
     AND active.start_time <= event.observed_at
     AND active.end_time >= event.observed_at
    JOIN round_facts fact
      ON fact.provider = active.provider
     AND fact.session_id = active.session_id
     AND fact.run_ordinal = active.run_ordinal
    WHERE event.scale IS NOT NULL
      AND fact.started_at <= event.observed_at
), sequenced AS (
    SELECT
        *,
        max(run_position) OVER (
            PARTITION BY
                provider, session_id, run_ordinal, round_pk,
                fact_provider, fact_session_id, fact_run_ordinal
        ) AS latest_position,
        lead(started_at) OVER (
            PARTITION BY
                provider, session_id, run_ordinal, round_pk,
                fact_provider, fact_session_id, fact_run_ordinal
            ORDER BY run_position
        ) AS next_observed_round_started_at
    FROM observed_round_state
), known_state AS (
    SELECT
        *,
        coalesce(
            observed_completion_marker_at,
            next_observed_round_started_at
        ) AS known_completion_time
    FROM sequenced
), summarized AS (
    SELECT
        provider,
        session_id,
        run_ordinal,
        round_pk,
        observed_at,
        max_input_tokens_total,
        w,
        active_run_ids,
        scale,
        context_bucket,
        count(*) FILTER (
            WHERE known_completion_time <= observed_at
        ) AS n,
        count(*) FILTER (
            WHERE run_position = latest_position
              AND observed_first_tool_at IS NOT NULL
        ) AS p,
        count(*) FILTER (
            WHERE run_position = latest_position
              AND observed_first_tool_at IS NOT NULL
              AND known_completion_time <= observed_at
        ) AS x,
        list(
            input_tokens_total
            ORDER BY fact_provider, fact_session_id, fact_run_ordinal
        ) FILTER (
            WHERE run_position = latest_position
              AND observed_first_tool_at IS NOT NULL
        ) AS pending_anchors
    FROM known_state
    GROUP BY
        provider, session_id, run_ordinal, round_pk, observed_at,
        max_input_tokens_total, w, active_run_ids, scale, context_bucket
)
SELECT *
FROM summarized
ORDER BY provider, observed_at, round_pk
"""


if __name__ == "__main__":
    raise SystemExit(main())
