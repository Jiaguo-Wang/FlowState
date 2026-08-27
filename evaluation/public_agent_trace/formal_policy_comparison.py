#!/usr/bin/env python3
"""执行冻结 TraceLab 快照上的正式离线策略比较。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.public_agent_trace.tracelab_context_pressure import (
    ContextSnapshotEvent,
    _KNOWN_STATE_SQL,
    _snapshot_from_rows,
    analyze_snapshot,
)
from evaluation.public_agent_trace.tracelab_final_protocol import (
    MAIN_COHORT_MAX_TOKENS,
    build_final_snapshot_policy_metadata,
    demand_relative_budget,
)
from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CHECKPOINT_MEMORY_BYTES,
    SampledSnapshotEvent,
    TraceSnapshot,
    validate_snapshot,
)
from evaluation.sota_policies import KVFlowStylePolicy, MarconiStylePolicy
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import (
    FORMAL_RECOVERY_MODEL_METADATA,
    RecoveryCostModel,
)
from flowstate.state_catalog import CheckpointCandidate


POLICY_NAMES = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
)
BUDGET_RATIOS = (0.25, 0.50, 0.75, 1.00)
EXPECTED_MAIN_X_HISTOGRAM = {2: 83, 3: 9, 4: 4, 5: 1, 6: 8}
EXPECTED_MAIN_SNAPSHOT_COUNT = 105
EXPECTED_SECONDARY_SNAPSHOT_COUNT = 37
EXPECTED_MAIN_UNIQUE_RUNS = 284
EXPECTED_MAIN_SCALE_COUNTS = {"Small": 60, "Medium": 38, "Large": 7}
EXPECTED_MAIN_PROVIDER_COUNTS = {"claude": 67, "codex": 38}
EXPECTED_COLLAPSED_BUDGET_COUNT = 83
BOOTSTRAP_SEED = 20_260_827
BOOTSTRAP_ITERATIONS = 10_000
FLOAT_TOLERANCE_MS = 1e-9
DEFAULT_PROTOCOL_PATH = Path(__file__).with_name(
    "tracelab_nontrivial_protocol.json"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
PROTECTED_PATHS = (
    Path("flowstate/recovery_model.py"),
    Path("flowstate/optimizer.py"),
    Path("evaluation/controlled_multiworkflow_v1/policies.py"),
    Path("evaluation/sota_policies.py"),
    Path("evaluation/public_agent_trace/tracelab_final_protocol.json"),
    Path("evaluation/public_agent_trace/tracelab_nontrivial_protocol.json"),
    Path("evaluation/public_agent_trace/tracelab_nontrivial_demand.json"),
    Path("evaluation/public_agent_trace/tracelab_to_flowstate.py"),
)


@dataclass(frozen=True)
class FrozenSnapshot:
    """保存冻结事件、重建快照及其正式策略元数据。"""

    cohort: str
    source_row: Mapping[str, Any]
    snapshot: TraceSnapshot
    exact_parent_count: int
    policy_metadata: Any


@dataclass(frozen=True)
class PolicyResult:
    """记录单个快照、预算和策略的完整离线结果。"""

    cohort: str
    snapshot_id: str
    provider: str
    concurrency_bucket: str
    x: int
    candidate_count: int
    pending_count: int
    budget_ratio: float
    budget_k: int
    policy: str
    selected_checkpoint_ids: tuple[str, ...]
    selected_count: int
    total_recovery_gap_tokens: int
    mean_recovery_gap_tokens: float
    total_formal_recovery_cost_ms: float
    mean_formal_recovery_cost_ms: float
    executable_hit_count: int
    executable_hit_ratio: float
    selection_overhead_ms: float
    continuation_results: tuple[Mapping[str, Any], ...]


def load_frozen_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    """读取并验证 Step 10C.5 冻结 protocol。"""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != (
        "tracelab-nontrivial-policy-protocol-v1"
    ):
        raise ValueError("Step 10C.5 protocol schema 不匹配")
    main = protocol.get("selected_main_snapshots", ())
    secondary = protocol.get("selected_secondary_snapshots", ())
    if len(main) != EXPECTED_MAIN_SNAPSHOT_COUNT:
        raise ValueError(f"正式快照数量不是 {EXPECTED_MAIN_SNAPSHOT_COUNT}")
    if len(secondary) != EXPECTED_SECONDARY_SNAPSHOT_COUNT:
        raise ValueError(
            f"X>=4 次级快照数量不是 {EXPECTED_SECONDARY_SNAPSHOT_COUNT}"
        )

    histogram: dict[int, int] = {}
    for row in main:
        value = int(row["x"])
        histogram[value] = histogram.get(value, 0) + 1
    if histogram != EXPECTED_MAIN_X_HISTOGRAM:
        raise ValueError(f"正式 X histogram 不一致：{histogram}")

    active_runs = {
        str(run_id)
        for row in main
        for run_id in row["active_run_ids"]
    }
    if len(active_runs) != EXPECTED_MAIN_UNIQUE_RUNS:
        raise ValueError("正式不同 active run 数量不一致")
    _validate_category_counts(
        main,
        "scale",
        EXPECTED_MAIN_SCALE_COUNTS,
        "并发分层",
    )
    _validate_category_counts(
        main,
        "provider",
        EXPECTED_MAIN_PROVIDER_COUNTS,
        "provider 分层",
    )
    if any(int(row["x"]) < 4 for row in secondary):
        raise ValueError("次级切片包含 X<4 快照")
    if len({row["snapshot_id"] for row in main}) != len(main):
        raise ValueError("正式快照标识重复")
    if len({tuple(row["active_run_ids"]) for row in main}) != len(main):
        raise ValueError("正式快照活动运行集合重复")

    ratios = tuple(float(value) for value in protocol["budget_protocol"]["ratios"])
    if ratios != BUDGET_RATIOS:
        raise ValueError("冻结预算比例发生变化")
    if protocol["sampling_protocol"]["seed"] != 20_260_826:
        raise ValueError("冻结采样种子发生变化")
    if protocol["policy_metadata_protocol"]["marconi"]["alpha"] != 1.0:
        raise ValueError("Marconi alpha 不是冻结值 1.0")
    if protocol["policy_metadata_protocol"]["kvflow"][
        "steps_to_execution"
    ] != 1:
        raise ValueError("KVFlow STE 不是冻结值 1")
    return protocol


def reconstruct_frozen_snapshots(
    protocol: Mapping[str, Any],
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> tuple[tuple[FrozenSnapshot, ...], tuple[FrozenSnapshot, ...]]:
    """按冻结 trigger event 从只读数据库重建主集合与次级切片。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    cache: dict[tuple[Any, ...], tuple[TraceSnapshot, int]] = {}
    try:
        main = tuple(
            _reconstruct_one(connection, "main", row, cache)
            for row in protocol["selected_main_snapshots"]
        )
        secondary = tuple(
            _reconstruct_one(connection, "secondary_x4", row, cache)
            for row in protocol["selected_secondary_snapshots"]
        )
    finally:
        connection.close()
    return main, secondary


def select_checkpoint_ids(
    policy_name: str,
    frozen: FrozenSnapshot,
    budget_k: int,
    optimizer: GlobalOptimizer,
) -> tuple[str, ...]:
    """只向每个策略传入冻结协议允许读取的输入。"""
    snapshot = frozen.snapshot
    metadata = frozen.policy_metadata
    budget_bytes = budget_k * CHECKPOINT_MEMORY_BYTES
    if policy_name == "Global-LRU":
        return select_global_lru(
            snapshot.candidates,
            metadata.checkpoint_recency,
            budget_bytes,
        )
    if policy_name == "KVFlow-style":
        return KVFlowStylePolicy().select(
            snapshot.continuations,
            snapshot.candidates,
            budget_k,
            dict(metadata.steps_to_execution_by_continuation),
            dict(metadata.last_access_by_checkpoint),
        ).selected_checkpoint_ids
    if policy_name == "Marconi-style":
        return MarconiStylePolicy().select(
            snapshot.candidates,
            budget_k,
            dict(metadata.last_access_by_checkpoint),
            dict(metadata.marconi_flop_saved_by_checkpoint),
            metadata.marconi_alpha,
        ).selected_checkpoint_ids
    if policy_name == "FlowState":
        return tuple(
            candidate.checkpoint_id
            for candidate in optimizer.select(
                snapshot.continuations,
                snapshot.candidates,
                budget_bytes,
            ).selected
        )
    raise ValueError(f"未知正式策略：{policy_name}")


def evaluate_selection(
    frozen: FrozenSnapshot,
    budget_ratio: float,
    budget_k: int,
    policy_name: str,
    selected_checkpoint_ids: Sequence[str],
    recovery_model: RecoveryCostModel,
    selection_overhead_ms: float = 0.0,
) -> PolicyResult:
    """使用同一正式位置感知模型评价任意策略的选择。"""
    snapshot = frozen.snapshot
    candidate_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in snapshot.candidates
    }
    selected_ids = tuple(selected_checkpoint_ids)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("策略选择包含重复 checkpoint")
    missing = tuple(
        checkpoint_id
        for checkpoint_id in selected_ids
        if checkpoint_id not in candidate_by_id
    )
    if missing:
        raise ValueError(f"策略选择了未知 checkpoint：{missing}")
    if len(selected_ids) > budget_k:
        raise ValueError("策略选择数量超过 K")
    selected = tuple(candidate_by_id[item] for item in selected_ids)
    if any(not candidate.recurrent_resident for candidate in selected):
        raise ValueError("策略选择了非 resident checkpoint")

    continuation_results = []
    for continuation in snapshot.continuations:
        target = continuation.planning_target
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        if gap > target:
            raise ValueError("恢复间隔超过 planning target")
        cost = recovery_model.estimate(gap, target)
        continuation_results.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "planning_target_tokens": target,
                "executable_frontier_tokens": frontier,
                "recovery_gap_tokens": gap,
                "formal_recovery_cost_ms": cost,
                "executable_hit": gap == 0,
            }
        )
    pending_count = len(continuation_results)
    if pending_count <= 0:
        raise ValueError("正式快照必须包含 pending continuation")
    total_gap = sum(item["recovery_gap_tokens"] for item in continuation_results)
    total_cost = sum(
        item["formal_recovery_cost_ms"] for item in continuation_results
    )
    hit_count = sum(item["executable_hit"] for item in continuation_results)
    return PolicyResult(
        cohort=frozen.cohort,
        snapshot_id=snapshot.snapshot_id,
        provider=snapshot.time_domain,
        concurrency_bucket=snapshot.scale,
        x=frozen.exact_parent_count,
        candidate_count=len(snapshot.candidates),
        pending_count=pending_count,
        budget_ratio=budget_ratio,
        budget_k=budget_k,
        policy=policy_name,
        selected_checkpoint_ids=selected_ids,
        selected_count=len(selected_ids),
        total_recovery_gap_tokens=total_gap,
        mean_recovery_gap_tokens=total_gap / pending_count,
        total_formal_recovery_cost_ms=total_cost,
        mean_formal_recovery_cost_ms=total_cost / pending_count,
        executable_hit_count=hit_count,
        executable_hit_ratio=hit_count / pending_count,
        selection_overhead_ms=selection_overhead_ms,
        continuation_results=tuple(continuation_results),
    )


def evaluate_cohort(
    snapshots: Sequence[FrozenSnapshot],
    recovery_model: RecoveryCostModel | None = None,
) -> tuple[PolicyResult, ...]:
    """执行完整 budget × policy 矩阵，不改变冻结输入。"""
    model = recovery_model or RecoveryCostModel()
    optimizer = GlobalOptimizer(model)
    rows = []
    for frozen in snapshots:
        candidate_count = len(frozen.snapshot.candidates)
        for ratio in BUDGET_RATIOS:
            budget_k = demand_relative_budget(frozen.exact_parent_count, ratio)
            if budget_k > candidate_count:
                raise ValueError("冻结预算超过 candidate count")
            for policy_name in POLICY_NAMES:
                started = perf_counter()
                selected_ids = select_checkpoint_ids(
                    policy_name,
                    frozen,
                    budget_k,
                    optimizer,
                )
                overhead_ms = (perf_counter() - started) * 1_000.0
                rows.append(
                    evaluate_selection(
                        frozen,
                        ratio,
                        budget_k,
                        policy_name,
                        selected_ids,
                        model,
                        overhead_ms,
                    )
                )
    return tuple(rows)


def aggregate_by_budget(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """按预算与策略计算快照等权主指标。"""
    result = []
    for ratio in BUDGET_RATIOS:
        for policy_name in POLICY_NAMES:
            group = tuple(
                row
                for row in rows
                if row.budget_ratio == ratio and row.policy == policy_name
            )
            if not group:
                raise ValueError("聚合分组不能为空")
            pending_count = sum(row.pending_count for row in group)
            total_cost = sum(
                row.total_formal_recovery_cost_ms for row in group
            )
            total_gap = sum(row.total_recovery_gap_tokens for row in group)
            hit_count = sum(row.executable_hit_count for row in group)
            result.append(
                {
                    "budget_ratio": ratio,
                    "policy": policy_name,
                    "snapshot_count": len(group),
                    "pending_count": pending_count,
                    "mean_total_recovery_cost_ms_per_snapshot": mean(
                        row.total_formal_recovery_cost_ms for row in group
                    ),
                    "median_total_recovery_cost_ms_per_snapshot": median(
                        row.total_formal_recovery_cost_ms for row in group
                    ),
                    "mean_recovery_cost_ms_per_pending": (
                        total_cost / pending_count
                    ),
                    "total_recovery_gap_tokens": total_gap,
                    "mean_recovery_gap_tokens": total_gap / pending_count,
                    "executable_hit_count": hit_count,
                    "executable_hit_ratio": hit_count / pending_count,
                    "mean_selection_overhead_ms": mean(
                        row.selection_overhead_ms for row in group
                    ),
                }
            )
    return tuple(result)


def paired_comparisons(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """以相同 snapshot 为单位比较 FlowState 与三个 baseline。"""
    index = {
        (row.snapshot_id, row.budget_ratio, row.policy): row for row in rows
    }
    snapshot_ids = sorted({row.snapshot_id for row in rows})
    result = []
    for ratio in BUDGET_RATIOS:
        for baseline in POLICY_NAMES[:-1]:
            pairs = tuple(
                (
                    index[(snapshot_id, ratio, baseline)],
                    index[(snapshot_id, ratio, "FlowState")],
                )
                for snapshot_id in snapshot_ids
            )
            wins, ties, losses = _win_tie_loss(pairs)
            baseline_total = sum(
                item[0].total_formal_recovery_cost_ms for item in pairs
            )
            flow_total = sum(
                item[1].total_formal_recovery_cost_ms for item in pairs
            )
            result.append(
                {
                    "budget_ratio": ratio,
                    "baseline": baseline,
                    "snapshot_count": len(pairs),
                    "win_count": wins,
                    "tie_count": ties,
                    "loss_count": losses,
                    "win_fraction": wins / len(pairs),
                    "tie_fraction": ties / len(pairs),
                    "loss_fraction": losses / len(pairs),
                    "mean_absolute_cost_reduction_ms": mean(
                        baseline_row.total_formal_recovery_cost_ms
                        - flow_row.total_formal_recovery_cost_ms
                        for baseline_row, flow_row in pairs
                    ),
                    "aggregate_relative_cost_reduction": (
                        _relative_reduction(baseline_total, flow_total)
                    ),
                }
            )
    return tuple(result)


def improvement_distributions(
    rows: Sequence[PolicyResult],
    baseline: str = "Marconi-style",
) -> tuple[dict[str, Any], ...]:
    """汇总逐快照相对改善，并显式处理零成本情况。"""
    index = {
        (row.snapshot_id, row.budget_ratio, row.policy): row for row in rows
    }
    snapshot_ids = sorted({row.snapshot_id for row in rows})
    summaries = []
    for ratio in BUDGET_RATIOS:
        values = []
        both_zero = 0
        baseline_zero_flow_nonzero = 0
        baseline_nonzero_flow_zero = 0
        for snapshot_id in snapshot_ids:
            baseline_cost = index[
                (snapshot_id, ratio, baseline)
            ].total_formal_recovery_cost_ms
            flow_cost = index[
                (snapshot_id, ratio, "FlowState")
            ].total_formal_recovery_cost_ms
            if abs(baseline_cost) <= FLOAT_TOLERANCE_MS:
                if abs(flow_cost) <= FLOAT_TOLERANCE_MS:
                    both_zero += 1
                else:
                    baseline_zero_flow_nonzero += 1
                continue
            if abs(flow_cost) <= FLOAT_TOLERANCE_MS:
                baseline_nonzero_flow_zero += 1
            values.append((baseline_cost - flow_cost) / baseline_cost)
        ordered = sorted(values)
        summaries.append(
            {
                "budget_ratio": ratio,
                "baseline": baseline,
                "defined_snapshot_count": len(ordered),
                "mean_relative_improvement": (
                    mean(ordered) if ordered else None
                ),
                "median_relative_improvement": (
                    median(ordered) if ordered else None
                ),
                "p25_relative_improvement": _quantile(ordered, 0.25),
                "p75_relative_improvement": _quantile(ordered, 0.75),
                "p90_relative_improvement": _quantile(ordered, 0.90),
                "minimum_relative_improvement": (
                    ordered[0] if ordered else None
                ),
                "maximum_relative_improvement": (
                    ordered[-1] if ordered else None
                ),
                "both_zero": both_zero,
                "baseline_zero_flow_nonzero": baseline_zero_flow_nonzero,
                "baseline_nonzero_flow_zero": baseline_nonzero_flow_zero,
            }
        )
    return tuple(summaries)


def bootstrap_flow_vs_marconi(
    rows: Sequence[PolicyResult],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """以 snapshot 为抽样单位生成均值成本改善的确定性区间。"""
    if iterations <= 0:
        raise ValueError("bootstrap iterations 必须大于零")
    index = {
        (row.snapshot_id, row.budget_ratio, row.policy): row for row in rows
    }
    snapshot_ids = sorted({row.snapshot_id for row in rows})
    generator = random.Random(seed)
    result: dict[str, Any] = {
        "seed": seed,
        "iterations": iterations,
        "sampling_unit": "snapshot",
        "budget_ratios": {},
    }
    for ratio in BUDGET_RATIOS:
        pairs = tuple(
            (
                index[(snapshot_id, ratio, "Marconi-style")]
                .total_formal_recovery_cost_ms,
                index[(snapshot_id, ratio, "FlowState")]
                .total_formal_recovery_cost_ms,
            )
            for snapshot_id in snapshot_ids
        )
        observed = _relative_reduction(
            sum(item[0] for item in pairs),
            sum(item[1] for item in pairs),
        )
        samples = []
        for _ in range(iterations):
            draw = tuple(
                pairs[generator.randrange(len(pairs))]
                for _ in range(len(pairs))
            )
            value = _relative_reduction(
                sum(item[0] for item in draw),
                sum(item[1] for item in draw),
            )
            if value is not None:
                samples.append(value)
        ordered = sorted(samples)
        result["budget_ratios"][_ratio_label(ratio)] = {
            "snapshot_count": len(pairs),
            "observed_relative_reduction": observed,
            "valid_bootstrap_iterations": len(ordered),
            "ci95_lower": _quantile(ordered, 0.025),
            "ci95_upper": _quantile(ordered, 0.975),
        }
    return result


def build_x_bucket_analysis(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """按预注册 X=2、X=3、X>=4 分层计算成对结果。"""
    return _build_segment_analysis(
        rows,
        (
            ("X=2", lambda row: row.x == 2),
            ("X=3", lambda row: row.x == 3),
            ("X>=4", lambda row: row.x >= 4),
        ),
        include_mean_n_k=True,
    )


def build_provider_analysis(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """按 provider 生成描述性诊断。"""
    return _build_segment_analysis(
        rows,
        tuple(
            (provider, lambda row, value=provider: row.provider == value)
            for provider in ("claude", "codex")
        ),
    )


def build_concurrency_analysis(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """按冻结并发档位生成描述性诊断。"""
    return _build_segment_analysis(
        rows,
        tuple(
            (
                scale,
                lambda row, value=scale: row.concurrency_bucket == value,
            )
            for scale in ("Small", "Medium", "Large")
        ),
    )


def build_secondary_analysis(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    """汇总冻结 X>=4 高竞争次级切片。"""
    aggregates = aggregate_by_budget(rows)
    paired = {
        (item["budget_ratio"], item["baseline"]): item
        for item in paired_comparisons(rows)
    }
    result = []
    for item in aggregates:
        row = dict(item)
        if item["policy"] == "FlowState":
            comparison = paired[(item["budget_ratio"], "Marconi-style")]
            row["flow_vs_marconi_relative_cost_reduction"] = comparison[
                "aggregate_relative_cost_reduction"
            ]
            row["flow_vs_marconi_win_count"] = comparison["win_count"]
            row["flow_vs_marconi_tie_count"] = comparison["tie_count"]
            row["flow_vs_marconi_loss_count"] = comparison["loss_count"]
        else:
            row["flow_vs_marconi_relative_cost_reduction"] = None
            row["flow_vs_marconi_win_count"] = None
            row["flow_vs_marconi_tie_count"] = None
            row["flow_vs_marconi_loss_count"] = None
        result.append(row)
    return tuple(result)


def validate_results(
    protocol: Mapping[str, Any],
    main_snapshots: Sequence[FrozenSnapshot],
    secondary_snapshots: Sequence[FrozenSnapshot],
    main_rows: Sequence[PolicyResult],
    secondary_rows: Sequence[PolicyResult],
) -> dict[str, Any]:
    """验证正式输入、选择、成本与无未来信息边界。"""
    expected_main_rows = (
        EXPECTED_MAIN_SNAPSHOT_COUNT * len(BUDGET_RATIOS) * len(POLICY_NAMES)
    )
    expected_secondary_rows = (
        EXPECTED_SECONDARY_SNAPSHOT_COUNT
        * len(BUDGET_RATIOS)
        * len(POLICY_NAMES)
    )
    if len(main_rows) != expected_main_rows:
        raise ValueError("主策略矩阵行数不完整")
    if len(secondary_rows) != expected_secondary_rows:
        raise ValueError("次级策略矩阵行数不完整")
    all_snapshots = tuple(main_snapshots) + tuple(secondary_snapshots)
    all_rows = tuple(main_rows) + tuple(secondary_rows)
    future_violations = sum(
        len(validate_snapshot(item.snapshot)) for item in all_snapshots
    )
    gap_violations = sum(
        continuation["recovery_gap_tokens"]
        > continuation["planning_target_tokens"]
        for row in all_rows
        for continuation in row.continuation_results
    )
    budget_violations = sum(
        row.budget_k > row.candidate_count
        or row.selected_count > row.budget_k
        for row in all_rows
    )
    selection_uniqueness_violations = sum(
        len(set(row.selected_checkpoint_ids))
        != len(row.selected_checkpoint_ids)
        for row in all_rows
    )
    phi_validity_violations = sum(
        not math.isfinite(row.total_formal_recovery_cost_ms)
        or row.total_formal_recovery_cost_ms < 0.0
        for row in all_rows
    )
    collapsed_count = sum(
        len(
            {
                demand_relative_budget(item.exact_parent_count, ratio)
                for ratio in BUDGET_RATIOS[:3]
            }
        )
        == 1
        for item in main_snapshots
    )
    if collapsed_count != EXPECTED_COLLAPSED_BUDGET_COUNT:
        raise ValueError("X=2 budget collapse 数量不一致")
    if protocol["validation"]["future_field_leakage_violations"] != 0:
        raise ValueError("冻结 protocol 已记录 future leakage")
    return {
        "status": (
            "PASS"
            if not any(
                (
                    future_violations,
                    gap_violations,
                    budget_violations,
                    selection_uniqueness_violations,
                    phi_validity_violations,
                )
            )
            else "FAIL"
        ),
        "formal_snapshot_count": len(main_snapshots),
        "secondary_snapshot_count": len(secondary_snapshots),
        "main_policy_result_count": len(main_rows),
        "secondary_policy_result_count": len(secondary_rows),
        "future_information_violations": future_violations,
        "gap_domain_violations": gap_violations,
        "budget_violations": budget_violations,
        "selection_uniqueness_violations": (
            selection_uniqueness_violations
        ),
        "formal_phi_validity_violations": phi_validity_violations,
        "formal_phi_name": FORMAL_RECOVERY_MODEL_METADATA.name,
        "formal_phi_correct": phi_validity_violations == 0,
        "policy_metadata_correct": all(
            item.policy_metadata.marconi_alpha == 1.0
            and all(
                value == 1
                for _, value in item.policy_metadata.steps_to_execution_by_continuation
            )
            for item in all_snapshots
        ),
        "baseline_selection_phi_dependency": False,
        "flowstate_uses_formal_phi_g_t": True,
        "x2_collapsed_budget_snapshots": collapsed_count,
        "gpu_executed": False,
        "sglang_runtime_calls": 0,
        "database_access": "read_only=True",
    }


def run_comparison(
    database_path: Path = DEFAULT_DATABASE_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_directory: Path | None = None,
) -> Path:
    """执行冻结比较、写入全部可审计 artifact 并返回目录。"""
    repository_root = Path(__file__).resolve().parents[2]
    protected_before = _hash_paths(repository_root, PROTECTED_PATHS)
    protocol = load_frozen_protocol(protocol_path)
    _validate_protocol_source_hash(protocol, repository_root)
    main_snapshots, secondary_snapshots = reconstruct_frozen_snapshots(
        protocol,
        database_path,
    )
    model = RecoveryCostModel()
    main_rows = evaluate_cohort(main_snapshots, model)
    secondary_rows = evaluate_cohort(secondary_snapshots, model)
    aggregates = aggregate_by_budget(main_rows)
    paired = paired_comparisons(main_rows)
    distributions = improvement_distributions(main_rows)
    bootstrap = bootstrap_flow_vs_marconi(main_rows)
    x_analysis = build_x_bucket_analysis(main_rows)
    provider_analysis = build_provider_analysis(main_rows)
    concurrency_analysis = build_concurrency_analysis(main_rows)
    secondary_analysis = build_secondary_analysis(secondary_rows)
    correctness = validate_results(
        protocol,
        main_snapshots,
        secondary_snapshots,
        main_rows,
        secondary_rows,
    )
    protected_after = _hash_paths(repository_root, PROTECTED_PATHS)
    correctness["protected_sources_unchanged"] = (
        protected_before == protected_after
    )
    if protected_before != protected_after:
        correctness["status"] = "FAIL"
        raise RuntimeError("冻结模型、策略或 protocol 在评估期间发生变化")

    if output_directory is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_directory = DEFAULT_OUTPUT_ROOT / f"formal_policy_results_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_artifacts(
        output_directory,
        database_path,
        protocol_path,
        protocol,
        main_snapshots,
        secondary_snapshots,
        main_rows,
        secondary_rows,
        aggregates,
        paired,
        distributions,
        bootstrap,
        x_analysis,
        provider_analysis,
        concurrency_analysis,
        secondary_analysis,
        correctness,
        protected_before,
    )
    return output_directory


def _reconstruct_one(
    connection: Any,
    cohort: str,
    row: Mapping[str, Any],
    cache: dict[tuple[Any, ...], tuple[TraceSnapshot, int]],
) -> FrozenSnapshot:
    observed_at = datetime.fromisoformat(str(row["observed_at"]))
    key = (
        row["provider"],
        row["trigger_session_id"],
        int(row["trigger_run_ordinal"]),
        int(row["trigger_round_pk"]),
        observed_at,
    )
    cached = cache.get(key)
    if cached is None:
        event = SampledSnapshotEvent(
            snapshot_id=str(row["snapshot_id"]),
            scale=str(row["scale"]),
            provider=str(row["provider"]),
            context_bucket=str(row["context_bucket"]),
            trigger_session_id=str(row["trigger_session_id"]),
            trigger_run_ordinal=int(row["trigger_run_ordinal"]),
            trigger_round_pk=int(row["trigger_round_pk"]),
            observed_at=observed_at,
            trace_observed_active_runs=int(row["w"]),
        )
        item = ContextSnapshotEvent(
            cohort="C128",
            cutoff_tokens=MAIN_COHORT_MAX_TOKENS,
            event=event,
        )
        known_rows = fetch_dicts(
            connection,
            _KNOWN_STATE_SQL,
            (
                MAIN_COHORT_MAX_TOKENS,
                observed_at,
                observed_at,
                observed_at,
                observed_at,
                row["provider"],
            ),
        )
        snapshot = _snapshot_from_rows(item, known_rows)
        analysis = analyze_snapshot(snapshot)
        expected = (
            int(row["w"]),
            int(row["n"]),
            int(row["p"]),
            int(row["x"]),
        )
        observed = (
            int(analysis["w"]),
            int(analysis["n"]),
            int(analysis["p"]),
            int(analysis["x"]),
        )
        if observed != expected:
            raise ValueError(
                f"{row['snapshot_id']} 的 W/N/P/X 重建不一致："
                f"expected={expected}, observed={observed}"
            )
        cached = (snapshot, int(analysis["x"]))
        cache[key] = cached
    snapshot, exact_parent_count = cached
    return FrozenSnapshot(
        cohort=cohort,
        source_row=row,
        snapshot=snapshot,
        exact_parent_count=exact_parent_count,
        policy_metadata=build_final_snapshot_policy_metadata(snapshot),
    )


def _build_segment_analysis(
    rows: Sequence[PolicyResult],
    segments: Sequence[tuple[str, Any]],
    *,
    include_mean_n_k: bool = False,
) -> tuple[dict[str, Any], ...]:
    result = []
    for label, predicate in segments:
        for ratio in BUDGET_RATIOS:
            marconi = tuple(
                row
                for row in rows
                if row.budget_ratio == ratio
                and row.policy == "Marconi-style"
                and predicate(row)
            )
            flow = tuple(
                row
                for row in rows
                if row.budget_ratio == ratio
                and row.policy == "FlowState"
                and predicate(row)
            )
            if not marconi or len(marconi) != len(flow):
                raise ValueError(f"分层 {label} 的 paired rows 不完整")
            flow_by_id = {row.snapshot_id: row for row in flow}
            pairs = tuple((row, flow_by_id[row.snapshot_id]) for row in marconi)
            wins, ties, losses = _win_tie_loss(pairs)
            record = {
                "segment": label,
                "budget_ratio": ratio,
                "snapshot_count": len(pairs),
                "flowstate_mean_cost_ms": mean(
                    item[1].total_formal_recovery_cost_ms for item in pairs
                ),
                "marconi_mean_cost_ms": mean(
                    item[0].total_formal_recovery_cost_ms for item in pairs
                ),
                "flow_vs_marconi_relative_cost_reduction": _relative_reduction(
                    sum(item[0].total_formal_recovery_cost_ms for item in pairs),
                    sum(item[1].total_formal_recovery_cost_ms for item in pairs),
                ),
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
            }
            if include_mean_n_k:
                record["mean_candidate_count"] = mean(
                    item[0].candidate_count for item in pairs
                )
                record["mean_budget_k"] = mean(
                    item[0].budget_k for item in pairs
                )
            result.append(record)
    return tuple(result)


def _win_tie_loss(
    pairs: Sequence[tuple[PolicyResult, PolicyResult]],
) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for baseline, flow in pairs:
        difference = (
            baseline.total_formal_recovery_cost_ms
            - flow.total_formal_recovery_cost_ms
        )
        if difference > FLOAT_TOLERANCE_MS:
            wins += 1
        elif difference < -FLOAT_TOLERANCE_MS:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def _relative_reduction(
    baseline_cost: float,
    flow_cost: float,
) -> float | None:
    if abs(baseline_cost) <= FLOAT_TOLERANCE_MS:
        return None
    return (baseline_cost - flow_cost) / baseline_cost


def _quantile(values: Sequence[float], probability: float) -> float | None:
    """使用确定性的线性插值分位数。"""
    if not values:
        return None
    if probability < 0.0 or probability > 1.0:
        raise ValueError("分位数概率必须位于 [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _validate_category_counts(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    expected: Mapping[str, int],
    label: str,
) -> None:
    observed = {
        value: sum(str(row[field]) == value for row in rows)
        for value in expected
    }
    if observed != dict(expected):
        raise ValueError(f"{label}数量不一致：{observed}")


def _validate_protocol_source_hash(
    protocol: Mapping[str, Any],
    repository_root: Path,
) -> None:
    source = protocol["source"]
    audit_path = Path(str(source["audit_path"]))
    if not audit_path.is_absolute():
        audit_path = repository_root / audit_path
    actual = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if actual != source["audit_sha256"]:
        raise ValueError("Step 10C.4 冻结输入 hash 不一致")


def _hash_paths(
    repository_root: Path,
    paths: Sequence[Path],
) -> dict[str, str]:
    return {
        str(path): hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _snapshot_manifest_row(frozen: FrozenSnapshot) -> dict[str, Any]:
    snapshot = frozen.snapshot
    policy_metadata = frozen.policy_metadata
    recency_by_id = {
        item.checkpoint_id: item for item in policy_metadata.checkpoint_recency
    }
    last_access = dict(policy_metadata.last_access_by_checkpoint)
    flop_saved = dict(policy_metadata.marconi_flop_saved_by_checkpoint)
    checkpoint_trace = {
        item.checkpoint_id: item for item in snapshot.checkpoint_metadata
    }
    steps = dict(policy_metadata.steps_to_execution_by_continuation)
    return {
        "cohort": frozen.cohort,
        "snapshot_id": snapshot.snapshot_id,
        "provider": snapshot.time_domain,
        "concurrency_bucket": snapshot.scale,
        "observed_at": snapshot.observed_at.isoformat(),
        "active_workflow_ids": snapshot.active_workflow_ids,
        "x": frozen.exact_parent_count,
        "candidate_count": len(snapshot.candidates),
        "pending_count": len(snapshot.continuations),
        "candidates": [
            {
                **asdict(candidate),
                "lineage_path": candidate.lineage_path,
                "creation_order": recency_by_id[
                    candidate.checkpoint_id
                ].creation_order,
                "last_access": last_access[candidate.checkpoint_id],
                "known_at_time": checkpoint_trace[
                    candidate.checkpoint_id
                ].known_at_time.isoformat(),
                "incremental_flop_proxy": flop_saved[
                    candidate.checkpoint_id
                ],
            }
            for candidate in snapshot.candidates
        ],
        "continuations": [
            {
                **asdict(continuation),
                "lineage_path": continuation.lineage_path,
                "planning_target": continuation.planning_target,
                "steps_to_execution": steps[continuation.continuation_id],
            }
            for continuation in snapshot.continuations
        ],
        "marconi_alpha": policy_metadata.marconi_alpha,
        "future_prefix_used": snapshot.future_prefix_used,
        "runtime_residency_inferred": snapshot.runtime_residency_inferred,
        "llm_level_branching_introduced": (
            snapshot.llm_level_branching_introduced
        ),
    }


def _policy_result_to_csv(row: PolicyResult) -> dict[str, Any]:
    values = asdict(row)
    values["selected_checkpoint_ids"] = json.dumps(
        row.selected_checkpoint_ids,
        ensure_ascii=False,
    )
    values["continuation_results"] = json.dumps(
        row.continuation_results,
        ensure_ascii=False,
        sort_keys=True,
    )
    return values


def _write_artifacts(
    output: Path,
    database_path: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    main_snapshots: Sequence[FrozenSnapshot],
    secondary_snapshots: Sequence[FrozenSnapshot],
    main_rows: Sequence[PolicyResult],
    secondary_rows: Sequence[PolicyResult],
    aggregates: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    distributions: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    x_analysis: Sequence[Mapping[str, Any]],
    provider_analysis: Sequence[Mapping[str, Any]],
    concurrency_analysis: Sequence[Mapping[str, Any]],
    secondary_analysis: Sequence[Mapping[str, Any]],
    correctness: Mapping[str, Any],
    protected_hashes: Mapping[str, str],
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": "flowstate.tracelab.formal_policy_comparison.v1",
        "created_at": created_at,
        "database_path": str(database_path),
        "database_access": "read_only=True",
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_audit_sha256": protocol["source"]["audit_sha256"],
        "formal_model": asdict(FORMAL_RECOVERY_MODEL_METADATA),
        "policies": POLICY_NAMES,
        "budget_formula": "K(r)=max(1,floor(X*r))",
        "budget_ratios": BUDGET_RATIOS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_sampling_unit": "snapshot",
        "checkpoint_memory_bytes": CHECKPOINT_MEMORY_BYTES,
        "executable_hit_definition": "G_p=0",
        "main_snapshot_weight": "equal",
        "protected_source_hashes": protected_hashes,
        "gpu_executed": False,
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "flowstate.tracelab.snapshot_manifest.v1",
        "main": [_snapshot_manifest_row(item) for item in main_snapshots],
        "secondary_x4": [
            _snapshot_manifest_row(item) for item in secondary_snapshots
        ],
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "raw_policy_results.csv",
        tuple(_policy_result_to_csv(row) for row in (*main_rows, *secondary_rows)),
    )
    _write_csv(output / "aggregate_by_budget.csv", aggregates)
    _write_csv(output / "paired_comparison.csv", paired)
    _write_csv(output / "improvement_distribution.csv", distributions)
    _write_csv(output / "x_bucket_analysis.csv", x_analysis)
    _write_csv(output / "provider_analysis.csv", provider_analysis)
    _write_csv(output / "concurrency_analysis.csv", concurrency_analysis)
    _write_csv(output / "secondary_x4_analysis.csv", secondary_analysis)
    (output / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "correctness_audit.json").write_text(
        json.dumps(correctness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_csv(output / "plot_budget_cost.csv", aggregates)
    _write_csv(
        output / "plot_budget_ehr.csv",
        tuple(
            {
                "budget_ratio": row["budget_ratio"],
                "policy": row["policy"],
                "executable_hit_ratio": row["executable_hit_ratio"],
            }
            for row in aggregates
        ),
    )
    _write_csv(
        output / "plot_improvement_distribution.csv",
        _raw_improvement_rows(main_rows),
    )
    _write_csv(output / "plot_x_bucket.csv", x_analysis)
    (output / "README.md").write_text(
        _render_readme(
            aggregates,
            paired,
            bootstrap,
            correctness,
        ),
        encoding="utf-8",
    )


def _raw_improvement_rows(
    rows: Sequence[PolicyResult],
) -> tuple[dict[str, Any], ...]:
    index = {
        (row.snapshot_id, row.budget_ratio, row.policy): row for row in rows
    }
    result = []
    for snapshot_id in sorted({row.snapshot_id for row in rows}):
        for ratio in BUDGET_RATIOS:
            baseline = index[(snapshot_id, ratio, "Marconi-style")]
            flow = index[(snapshot_id, ratio, "FlowState")]
            relative = _relative_reduction(
                baseline.total_formal_recovery_cost_ms,
                flow.total_formal_recovery_cost_ms,
            )
            if baseline.total_formal_recovery_cost_ms <= FLOAT_TOLERANCE_MS:
                zero_category = (
                    "both_zero"
                    if flow.total_formal_recovery_cost_ms <= FLOAT_TOLERANCE_MS
                    else "baseline_zero_flow_nonzero"
                )
            elif flow.total_formal_recovery_cost_ms <= FLOAT_TOLERANCE_MS:
                zero_category = "baseline_nonzero_flow_zero"
            else:
                zero_category = "none"
            result.append(
                {
                    "snapshot_id": snapshot_id,
                    "budget_ratio": ratio,
                    "marconi_cost_ms": baseline.total_formal_recovery_cost_ms,
                    "flowstate_cost_ms": flow.total_formal_recovery_cost_ms,
                    "relative_improvement": relative,
                    "zero_category": zero_category,
                }
            )
    return tuple(result)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV 不能写入空结果：{path.name}")
    fieldnames = tuple(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_readme(
    aggregates: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> str:
    aggregate_index = {
        (row["budget_ratio"], row["policy"]): row for row in aggregates
    }
    paired_index = {
        (row["budget_ratio"], row["baseline"]): row for row in paired
    }
    lines = [
        "# TraceLab 正式离线策略比较",
        "",
        "本目录只使用 Step 10C.5 冻结的 105 个 C128 AND X>=2 快照及其 37 个 X>=4 次级切片。数据库以只读方式按冻结 trigger event 重建；没有重新采样、没有未来字段、没有 GPU 或 SGLang runtime 调用。",
        "",
        "所有策略选择后的恢复成本统一由 `position_aware_quadratic_v1` 计算。Global-LRU、KVFlow-style 与 Marconi-style 的选择过程不读取该模型；FlowState 使用既有 `GlobalOptimizer` 的 set-dependent marginal recovery reduction。",
        "",
        "## 主结果",
        "",
        "| Budget | Policy | Mean total cost/snapshot (ms) | Gap tokens | EHR |",
        "|---:|---|---:|---:|---:|",
    ]
    for ratio in BUDGET_RATIOS:
        for policy in POLICY_NAMES:
            row = aggregate_index[(ratio, policy)]
            lines.append(
                f"| {_ratio_label(ratio)} | {policy} | {row['mean_total_recovery_cost_ms_per_snapshot']:.6f} | {row['total_recovery_gap_tokens']} | {row['executable_hit_ratio']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## FlowState 与 Marconi 的成对结果",
            "",
            "| Budget | Reduction | Win | Tie | Loss | Bootstrap 95% CI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in BUDGET_RATIOS:
        pair = paired_index[(ratio, "Marconi-style")]
        ci = bootstrap["budget_ratios"][_ratio_label(ratio)]
        lines.append(
            f"| {_ratio_label(ratio)} | {_percent_or_na(pair['aggregate_relative_cost_reduction'])} | {pair['win_count']} | {pair['tie_count']} | {pair['loss_count']} | [{_percent_or_na(ci['ci95_lower'])}, {_percent_or_na(ci['ci95_upper'])}] |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "主集合是预注册的结构覆盖型分层样本，对每个 snapshot 等权；它不是完整 C128 自然事件频率的概率估计。X>=4 次级切片 provider/concurrency 分布偏斜，只用于高竞争描述。TraceLab 不提供显式 LLM-level DAG、token IDs 或真实 runtime residency，本结果是 leakage-free 逻辑 checkpoint snapshot 上的正式离线比较。",
            "",
            f"Correctness gate：**{correctness['status']}**；future-information violations：{correctness['future_information_violations']}。",
            "",
        ]
    )
    return "\n".join(lines)


def _ratio_label(ratio: float) -> str:
    return f"{int(round(ratio * 100))}%"


def _percent_or_na(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.3f}%"


def main(argv: Sequence[str] | None = None) -> int:
    """从命令行运行冻结离线比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = run_comparison(
        database_path=arguments.database,
        protocol_path=arguments.protocol,
        output_directory=arguments.output,
    )
    correctness = json.loads(
        (output / "correctness_audit.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "status": correctness["status"],
                "formal_snapshots": correctness["formal_snapshot_count"],
                "future_information_violations": correctness[
                    "future_information_violations"
                ],
                "gpu_executed": correctness["gpu_executed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
