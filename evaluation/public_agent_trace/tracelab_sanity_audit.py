#!/usr/bin/env python3
"""只读审计冻结 TraceLab 正式策略结果的机制与解释边界。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from evaluation.sota_policies import _min_max_normalize
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


FORMAL_RESULT_DIRECTORY = Path(__file__).with_name(
    "formal_policy_results_20260827_075548_356403"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
FROZEN_SANITY_AUDIT_DIRECTORY = Path(__file__).with_name(
    "tracelab_sanity_audit_20260827_082236_084252"
)
BUDGET_RATIOS = (0.25, 0.50, 0.75, 1.00)
CONSTRAINED_BUDGET_RATIOS = BUDGET_RATIOS[:3]
INTERPRETABLE_CANDIDATE_LIMIT = 24
POLICY_NAMES = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
)
FLOAT_TOLERANCE_MS = 1e-9
PROTECTED_PATHS = (
    Path("flowstate/recovery_model.py"),
    Path("flowstate/optimizer.py"),
    Path("evaluation/controlled_multiworkflow_v1/policies.py"),
    Path("evaluation/sota_policies.py"),
    Path("evaluation/public_agent_trace/tracelab_nontrivial_protocol.json"),
)


@dataclass(frozen=True)
class AuditSnapshot:
    """保存从冻结 manifest 重建的逻辑快照与策略元数据。"""

    snapshot_id: str
    cohort: str
    provider: str
    concurrency_bucket: str
    x: int
    candidates: tuple[CheckpointCandidate, ...]
    continuations: tuple[PendingContinuation, ...]
    candidate_metadata: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class FrozenPolicyResult:
    """保存 Step 10E 已计算完成的一条只读策略结果。"""

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
    total_recovery_gap_tokens: int
    mean_recovery_gap_tokens: float
    total_formal_recovery_cost_ms: float
    executable_hit_count: int
    executable_hit_ratio: float
    continuation_results: tuple[Mapping[str, Any], ...]


def load_formal_results(
    result_directory: Path = FORMAL_RESULT_DIRECTORY,
) -> tuple[
    dict[str, AuditSnapshot],
    dict[str, AuditSnapshot],
    tuple[FrozenPolicyResult, ...],
]:
    """读取冻结 manifest 与 raw policy results，不执行任何策略。"""
    required = (
        "snapshot_manifest.json",
        "raw_policy_results.csv",
        "correctness_audit.json",
        "config.json",
    )
    missing = tuple(name for name in required if not (result_directory / name).is_file())
    if missing:
        raise FileNotFoundError(f"Step 10E artifact 缺少文件：{missing}")
    correctness = json.loads(
        (result_directory / "correctness_audit.json").read_text(encoding="utf-8")
    )
    if correctness.get("status") != "PASS":
        raise ValueError("Step 10E correctness gate 不是 PASS")
    if correctness.get("future_information_violations") != 0:
        raise ValueError("Step 10E artifact 存在未来信息违规")
    manifest = json.loads(
        (result_directory / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    main = {
        item["snapshot_id"]: _snapshot_from_manifest(item)
        for item in manifest["main"]
    }
    secondary = {
        item["snapshot_id"]: _snapshot_from_manifest(item)
        for item in manifest["secondary_x4"]
    }
    if len(main) != 105 or len(secondary) != 37:
        raise ValueError("Step 10E 快照数量不一致")
    with (result_directory / "raw_policy_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = tuple(_policy_result_from_row(row) for row in csv.DictReader(handle))
    if len(rows) != 2_272:
        raise ValueError("Step 10E raw policy result 行数不一致")
    return main, secondary, rows


def verify_frozen_results(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
    model: RecoveryCostModel | None = None,
) -> dict[str, Any]:
    """复核冻结选择的指标与三种 baseline 排序公式。"""
    recovery_model = model or RecoveryCostModel()
    metric_mismatches = 0
    lru_selection_mismatches = 0
    kvflow_selection_mismatches = 0
    marconi_selection_mismatches = 0
    keys = set()
    for row in rows:
        if row.cohort != "main":
            continue
        snapshot = snapshots[row.snapshot_id]
        key = (row.snapshot_id, row.budget_ratio, row.policy)
        if key in keys:
            raise ValueError(f"冻结结果主键重复：{key}")
        keys.add(key)
        recalculated = calculate_selection_effect(
            snapshot,
            row.selected_checkpoint_ids,
            recovery_model,
        )
        if (
            recalculated["total_gap_tokens"] != row.total_recovery_gap_tokens
            or abs(
                recalculated["total_cost_ms"]
                - row.total_formal_recovery_cost_ms
            )
            > FLOAT_TOLERANCE_MS
            or recalculated["executable_hit_count"] != row.executable_hit_count
        ):
            metric_mismatches += 1
        if row.policy == "Global-LRU" and tuple(
            _expected_lru_selection(snapshot, row.budget_k)
        ) != row.selected_checkpoint_ids:
            lru_selection_mismatches += 1
        if row.policy == "KVFlow-style" and tuple(
            _expected_kvflow_selection(snapshot, row.budget_k)
        ) != row.selected_checkpoint_ids:
            kvflow_selection_mismatches += 1
        if row.policy == "Marconi-style" and tuple(
            _expected_marconi_selection(snapshot, row.budget_k)
        ) != row.selected_checkpoint_ids:
            marconi_selection_mismatches += 1
    return {
        "metric_mismatches": metric_mismatches,
        "lru_selection_mismatches": lru_selection_mismatches,
        "kvflow_selection_mismatches": kvflow_selection_mismatches,
        "marconi_selection_mismatches": marconi_selection_mismatches,
        "implementation_bug_found": any(
            (
                metric_mismatches,
                lru_selection_mismatches,
                kvflow_selection_mismatches,
                marconi_selection_mismatches,
            )
        ),
    }


def calculate_selection_effect(
    snapshot: AuditSnapshot,
    selected_checkpoint_ids: Sequence[str],
    model: RecoveryCostModel,
) -> dict[str, Any]:
    """从冻结 selected set 复算 E、G 与正式成本。"""
    candidate_by_id = {
        candidate.checkpoint_id: candidate for candidate in snapshot.candidates
    }
    selected = tuple(candidate_by_id[item] for item in selected_checkpoint_ids)
    continuation_rows = []
    for continuation in snapshot.continuations:
        target = continuation.planning_target
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        continuation_rows.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "T": target,
                "E": frontier,
                "G": gap,
                "cost_ms": model.estimate(gap, target),
                "exact_parent_checkpoint_ids": exact_parent_ids(
                    snapshot,
                    continuation,
                ),
            }
        )
    return {
        "continuations": tuple(continuation_rows),
        "total_gap_tokens": sum(item["G"] for item in continuation_rows),
        "total_cost_ms": sum(item["cost_ms"] for item in continuation_rows),
        "executable_hit_count": sum(item["G"] == 0 for item in continuation_rows),
        "mean_executable_frontier_tokens": mean(
            item["E"] for item in continuation_rows
        ),
        "mean_gap_tokens": mean(item["G"] for item in continuation_rows),
    }


def exact_parent_ids(
    snapshot: AuditSnapshot,
    continuation: PendingContinuation,
) -> tuple[str, ...]:
    """返回当前 continuation 的 distinct exact-parent candidates。"""
    return tuple(
        candidate.checkpoint_id
        for candidate in snapshot.candidates
        if candidate.token_pos == continuation.planning_target
        and is_compatible(candidate, continuation)
    )


def select_representative_cases(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
) -> tuple[dict[str, Any], ...]:
    """按固定排序选取明显收益、tie 与 X>=4 紧预算案例。"""
    index = _main_result_index(rows)
    comparisons = []
    for snapshot_id, snapshot in snapshots.items():
        for ratio in BUDGET_RATIOS:
            marconi = index[(snapshot_id, ratio, "Marconi-style")]
            flow = index[(snapshot_id, ratio, "FlowState")]
            comparisons.append(
                {
                    "snapshot_id": snapshot_id,
                    "budget_ratio": ratio,
                    "x": snapshot.x,
                    "cost_difference_ms": (
                        marconi.total_formal_recovery_cost_ms
                        - flow.total_formal_recovery_cost_ms
                    ),
                }
            )
    used: set[tuple[str, float]] = set()
    selected = []
    clear = sorted(
        (
            item
            for item in comparisons
            if item["cost_difference_ms"] > FLOAT_TOLERANCE_MS
        ),
        key=lambda item: (
            -item["cost_difference_ms"],
            item["snapshot_id"],
            item["budget_ratio"],
        ),
    )[:3]
    for item in clear:
        selected.append({**item, "category": "clear_advantage"})
        used.add((item["snapshot_id"], item["budget_ratio"]))

    ties = sorted(
        (
            item
            for item in comparisons
            if abs(item["cost_difference_ms"]) <= FLOAT_TOLERANCE_MS
            and (item["snapshot_id"], item["budget_ratio"]) not in used
        ),
        key=lambda item: (item["snapshot_id"], item["budget_ratio"]),
    )[:3]
    for item in ties:
        selected.append({**item, "category": "tie_or_close"})
        used.add((item["snapshot_id"], item["budget_ratio"]))

    x4_tight = sorted(
        (
            item
            for item in comparisons
            if item["x"] >= 4
            and item["budget_ratio"] == 0.25
            and (item["snapshot_id"], item["budget_ratio"]) not in used
        ),
        key=lambda item: item["snapshot_id"],
    )[:2]
    for item in x4_tight:
        selected.append({**item, "category": "x4_tight_budget"})
    if sum(item["category"] == "clear_advantage" for item in selected) < 3:
        raise ValueError("明显收益案例不足三个")
    if sum(item["category"] == "tie_or_close" for item in selected) < 3:
        raise ValueError("tie 案例不足三个")
    if sum(item["category"] == "x4_tight_budget" for item in selected) < 2:
        raise ValueError("X>=4 紧预算案例不足两个")
    return tuple(selected)


def select_constrained_representative_cases(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
) -> tuple[dict[str, Any], ...]:
    """从冻结结果中确定性选择四个受限预算解释案例。"""
    index = _main_result_index(rows)
    comparisons = []
    for snapshot_id, snapshot in snapshots.items():
        for ratio in CONSTRAINED_BUDGET_RATIOS:
            marconi = index[(snapshot_id, ratio, "Marconi-style")]
            flow = index[(snapshot_id, ratio, "FlowState")]
            difference = (
                marconi.total_formal_recovery_cost_ms
                - flow.total_formal_recovery_cost_ms
            )
            if difference <= FLOAT_TOLERANCE_MS:
                continue
            if set(marconi.selected_checkpoint_ids) == set(
                flow.selected_checkpoint_ids
            ):
                continue
            comparisons.append(
                {
                    "snapshot_id": snapshot_id,
                    "budget_ratio": ratio,
                    "x": snapshot.x,
                    "candidate_count": len(snapshot.candidates),
                    "pending_count": len(snapshot.continuations),
                    "budget_k": flow.budget_k,
                    "cost_difference_ms": difference,
                }
            )

    requests = (
        ("25% budget", 0.25, None),
        ("50% budget", 0.50, 4),
        ("75% budget", 0.75, 4),
        ("X>=4 constrained budget", 0.25, 4),
    )
    selected = []
    used_snapshots: set[str] = set()
    for category, ratio, minimum_x in requests:
        eligible = tuple(
            item
            for item in comparisons
            if item["budget_ratio"] == ratio
            and item["snapshot_id"] not in used_snapshots
            and (minimum_x is None or item["x"] >= minimum_x)
        )
        if not eligible:
            raise ValueError(f"缺少受限预算解释案例：{category}")
        chosen = min(eligible, key=_constrained_case_rank)
        selected.append({**chosen, "category": category})
        used_snapshots.add(chosen["snapshot_id"])
    return tuple(selected)


def _constrained_case_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """优先选择规模适中且绝对成本差较大的案例。"""
    interpretable_tier = int(
        item["candidate_count"] > INTERPRETABLE_CANDIDATE_LIMIT
        or item["pending_count"] > 6
    )
    return (
        interpretable_tier,
        -float(item["cost_difference_ms"]),
        int(item["candidate_count"]),
        int(item["pending_count"]),
        str(item["snapshot_id"]),
    )


def audit_constrained_representative_case(
    descriptor: Mapping[str, Any],
    snapshot: AuditSnapshot,
    result_index: Mapping[tuple[str, float, str], FrozenPolicyResult],
) -> dict[str, Any]:
    """从冻结 Step 10E 行构造一个受限预算案例的解释记录。"""
    ratio = float(descriptor["budget_ratio"])
    marconi = result_index[(snapshot.snapshot_id, ratio, "Marconi-style")]
    flow = result_index[(snapshot.snapshot_id, ratio, "FlowState")]
    marconi_effects = {
        item["continuation_id"]: item for item in marconi.continuation_results
    }
    flow_effects = {
        item["continuation_id"]: item for item in flow.continuation_results
    }
    continuation_ids = tuple(
        continuation.continuation_id for continuation in snapshot.continuations
    )
    if set(marconi_effects) != set(continuation_ids):
        raise ValueError("Marconi 冻结 continuation 记录不完整")
    if set(flow_effects) != set(continuation_ids):
        raise ValueError("FlowState 冻结 continuation 记录不完整")

    pending_rows = []
    comparison_rows = []
    exact_parent_gain_count = 0
    deeper_frontier_gain_count = 0
    lower_frontier_count = 0
    for continuation in snapshot.continuations:
        continuation_id = continuation.continuation_id
        marconi_item = marconi_effects[continuation_id]
        flow_item = flow_effects[continuation_id]
        target = continuation.planning_target
        if int(marconi_item["planning_target_tokens"]) != target:
            raise ValueError("Marconi 冻结 planning target 不一致")
        if int(flow_item["planning_target_tokens"]) != target:
            raise ValueError("FlowState 冻结 planning target 不一致")
        marconi_frontier = int(marconi_item["executable_frontier_tokens"])
        flow_frontier = int(flow_item["executable_frontier_tokens"])
        exact_parent_gain_count += flow_frontier == target and marconi_frontier < target
        deeper_frontier_gain_count += flow_frontier > marconi_frontier
        lower_frontier_count += flow_frontier < marconi_frontier
        pending_rows.append(
            {
                "continuation_id": continuation_id,
                "workflow_id": continuation.workflow_id,
                "lineage_path": continuation.lineage_path,
                "T": target,
                "exact_parent_checkpoint_ids": exact_parent_ids(
                    snapshot,
                    continuation,
                ),
            }
        )
        comparison_rows.append(
            {
                "continuation_id": continuation_id,
                "T": target,
                "marconi_E": marconi_frontier,
                "marconi_G": int(marconi_item["recovery_gap_tokens"]),
                "marconi_Phi_ms": float(
                    marconi_item["formal_recovery_cost_ms"]
                ),
                "flowstate_E": flow_frontier,
                "flowstate_G": int(flow_item["recovery_gap_tokens"]),
                "flowstate_Phi_ms": float(flow_item["formal_recovery_cost_ms"]),
            }
        )

    marconi_redundant = redundant_compatible_checkpoint_ids(
        snapshot,
        marconi.selected_checkpoint_ids,
    )
    flow_redundant = redundant_compatible_checkpoint_ids(
        snapshot,
        flow.selected_checkpoint_ids,
    )
    redundant_avoidance_count = max(
        0,
        len(marconi_redundant) - len(flow_redundant),
    )
    mechanisms = []
    if exact_parent_gain_count:
        mechanisms.append("exact-parent coverage")
    if deeper_frontier_gain_count:
        mechanisms.append("deeper compatible checkpoint")
    if redundant_avoidance_count:
        mechanisms.append("redundant checkpoint avoidance")
    absolute_reduction = (
        marconi.total_formal_recovery_cost_ms
        - flow.total_formal_recovery_cost_ms
    )
    relative_reduction = absolute_reduction / marconi.total_formal_recovery_cost_ms
    explanation = _constrained_case_explanation(
        exact_parent_gain_count,
        deeper_frontier_gain_count,
        redundant_avoidance_count,
        lower_frontier_count,
    )
    return {
        "category": descriptor["category"],
        "snapshot_id": snapshot.snapshot_id,
        "X": snapshot.x,
        "N": len(snapshot.candidates),
        "K": flow.budget_k,
        "budget_ratio": ratio,
        "pending_continuations": tuple(pending_rows),
        "marconi_selection": marconi.selected_checkpoint_ids,
        "flowstate_selection": flow.selected_checkpoint_ids,
        "per_pending_comparison": tuple(comparison_rows),
        "marconi_total_recovery_cost_ms": marconi.total_formal_recovery_cost_ms,
        "flowstate_total_recovery_cost_ms": flow.total_formal_recovery_cost_ms,
        "absolute_reduction_ms": absolute_reduction,
        "relative_reduction": relative_reduction,
        "relative_reduction_percent": relative_reduction * 100.0,
        "mechanisms": tuple(mechanisms),
        "mechanism_explanation": explanation,
        "source": "Step 10E 冻结 formal policy results",
        "policy_rerun": False,
    }


def _constrained_case_explanation(
    exact_parent_gain_count: int,
    deeper_frontier_gain_count: int,
    redundant_avoidance_count: int,
    lower_frontier_count: int,
) -> str:
    """生成只描述冻结 selection 差异的中文机制说明。"""
    parts = []
    if exact_parent_gain_count:
        parts.append(f"多覆盖 {exact_parent_gain_count} 个 exact-parent demand")
    if deeper_frontier_gain_count:
        parts.append(f"让 {deeper_frontier_gain_count} 个 pending 获得更深 E")
    if redundant_avoidance_count:
        parts.append(f"少保留 {redundant_avoidance_count} 个无 E 增量的冗余 checkpoint")
    if lower_frontier_count:
        parts.append(f"同时有 {lower_frontier_count} 个 pending 的 E 较低")
    return "冻结 selection 显示 FlowState " + "，".join(parts) + "；这是结构性描述，不是因果结论。"


def audit_representative_case(
    descriptor: Mapping[str, Any],
    snapshot: AuditSnapshot,
    result_index: Mapping[tuple[str, float, str], FrozenPolicyResult],
    model: RecoveryCostModel,
) -> dict[str, Any]:
    """输出一个代表案例的全部选择信号、边际收益与 E/G/Phi。"""
    ratio = float(descriptor["budget_ratio"])
    score_rows = marconi_candidate_scores(snapshot)
    score_by_id = {item["checkpoint_id"]: item for item in score_rows}
    policy_effects = {}
    policy_selections = {}
    for policy in POLICY_NAMES:
        result = result_index[(snapshot.snapshot_id, ratio, policy)]
        policy_selections[policy] = result.selected_checkpoint_ids
        policy_effects[policy] = calculate_selection_effect(
            snapshot,
            result.selected_checkpoint_ids,
            model,
        )["continuations"]
    flow_ids = policy_selections["FlowState"]
    marconi_ids = policy_selections["Marconi-style"]
    return {
        "category": descriptor["category"],
        "snapshot_id": snapshot.snapshot_id,
        "provider": snapshot.provider,
        "concurrency_bucket": snapshot.concurrency_bucket,
        "x": snapshot.x,
        "candidate_count": len(snapshot.candidates),
        "pending_count": len(snapshot.continuations),
        "budget_ratio": ratio,
        "budget_k": result_index[
            (snapshot.snapshot_id, ratio, "FlowState")
        ].budget_k,
        "marconi_minus_flowstate_cost_ms": descriptor["cost_difference_ms"],
        "pending_continuations": tuple(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "lineage_path": continuation.lineage_path,
                "T": continuation.planning_target,
                "exact_parent_checkpoint_ids": exact_parent_ids(
                    snapshot,
                    continuation,
                ),
            }
            for continuation in snapshot.continuations
        ),
        "candidate_checkpoints": tuple(
            {
                "checkpoint_id": candidate.checkpoint_id,
                "workflow_id": candidate.workflow_id,
                "lineage_path": candidate.lineage_path,
                "token_pos": candidate.token_pos,
                "last_access": snapshot.candidate_metadata[
                    candidate.checkpoint_id
                ]["last_access"],
                "incremental_flop_proxy": snapshot.candidate_metadata[
                    candidate.checkpoint_id
                ]["incremental_flop_proxy"],
                "compatible_pending_ids": tuple(
                    continuation.continuation_id
                    for continuation in snapshot.continuations
                    if is_compatible(candidate, continuation)
                ),
            }
            for candidate in snapshot.candidates
        ),
        "policy_selections": policy_selections,
        "marconi_selected_scores": tuple(
            score_by_id[checkpoint_id] for checkpoint_id in marconi_ids
        ),
        "flowstate_selected_marginal_benefits": flowstate_marginal_benefits(
            snapshot,
            flow_ids,
            model,
        ),
        "policy_continuation_effects": policy_effects,
    }


def marconi_candidate_scores(
    snapshot: AuditSnapshot,
) -> tuple[dict[str, Any], ...]:
    """按冻结 metadata 复算 Marconi 的两项归一化分数与 utility。"""
    recency = {
        candidate.checkpoint_id: float(
            snapshot.candidate_metadata[candidate.checkpoint_id]["last_access"]
        )
        for candidate in snapshot.candidates
    }
    efficiency = {
        candidate.checkpoint_id: float(
            snapshot.candidate_metadata[candidate.checkpoint_id][
                "incremental_flop_proxy"
            ]
        )
        / candidate.memory_bytes
        for candidate in snapshot.candidates
    }
    normalized_recency = _min_max_normalize(recency)
    normalized_efficiency = _min_max_normalize(efficiency)
    return tuple(
        {
            "checkpoint_id": candidate.checkpoint_id,
            "last_access": recency[candidate.checkpoint_id],
            "raw_flop_efficiency": efficiency[candidate.checkpoint_id],
            "normalized_recency": normalized_recency[candidate.checkpoint_id],
            "normalized_flop_efficiency": normalized_efficiency[
                candidate.checkpoint_id
            ],
            "utility": normalized_recency[candidate.checkpoint_id]
            + normalized_efficiency[candidate.checkpoint_id],
        }
        for candidate in sorted(snapshot.candidates, key=lambda item: item.checkpoint_id)
    )


def flowstate_marginal_benefits(
    snapshot: AuditSnapshot,
    selected_checkpoint_ids: Sequence[str],
    model: RecoveryCostModel,
) -> tuple[dict[str, Any], ...]:
    """按冻结 FlowState 选择顺序计算每个已选项的边际恢复收益。"""
    candidate_by_id = {
        candidate.checkpoint_id: candidate for candidate in snapshot.candidates
    }
    selected: tuple[CheckpointCandidate, ...] = ()
    current_cost = _selection_cost(snapshot, selected, model)
    rows = []
    for checkpoint_id in selected_checkpoint_ids:
        candidate = candidate_by_id[checkpoint_id]
        following = selected + (candidate,)
        next_cost = _selection_cost(snapshot, following, model)
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "marginal_recovery_benefit_ms": current_cost - next_cost,
                "cost_before_ms": current_cost,
                "cost_after_ms": next_cost,
            }
        )
        selected = following
        current_cost = next_cost
    return tuple(rows)


def benefit_decomposition(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
    model: RecoveryCostModel,
) -> dict[str, Any]:
    """对不同 selection 做可重叠的描述性收益来源分解。"""
    index = _main_result_index(rows)
    counts = {
        "different_selection_cases": 0,
        "flowstate_cost_advantage_cases": 0,
        "objective_tie_cases": 0,
        "exact_parent_coverage_gain_cases": 0,
        "compatible_checkpoint_depth_gain_cases": 0,
        "redundant_checkpoint_avoidance_cases": 0,
        "no_listed_factor_cases": 0,
    }
    by_budget = {
        _ratio_label(ratio): {key: 0 for key in counts}
        for ratio in BUDGET_RATIOS
    }
    combinations: dict[str, int] = {}
    total_deeper_continuations = 0
    total_exact_parent_gain = 0
    total_redundant_reduction = 0
    for snapshot_id, snapshot in snapshots.items():
        for ratio in BUDGET_RATIOS:
            marconi = index[(snapshot_id, ratio, "Marconi-style")]
            flow = index[(snapshot_id, ratio, "FlowState")]
            if set(marconi.selected_checkpoint_ids) == set(
                flow.selected_checkpoint_ids
            ):
                continue
            counts["different_selection_cases"] += 1
            budget_counts = by_budget[_ratio_label(ratio)]
            budget_counts["different_selection_cases"] += 1
            cost_difference = (
                marconi.total_formal_recovery_cost_ms
                - flow.total_formal_recovery_cost_ms
            )
            if cost_difference > FLOAT_TOLERANCE_MS:
                counts["flowstate_cost_advantage_cases"] += 1
                budget_counts["flowstate_cost_advantage_cases"] += 1
            else:
                counts["objective_tie_cases"] += 1
                budget_counts["objective_tie_cases"] += 1

            marconi_effect = calculate_selection_effect(
                snapshot, marconi.selected_checkpoint_ids, model
            )
            flow_effect = calculate_selection_effect(
                snapshot, flow.selected_checkpoint_ids, model
            )
            marconi_by_continuation = {
                item["continuation_id"]: item
                for item in marconi_effect["continuations"]
            }
            flow_by_continuation = {
                item["continuation_id"]: item
                for item in flow_effect["continuations"]
            }
            exact_gain = sum(
                flow_by_continuation[key]["G"] == 0
                and marconi_by_continuation[key]["G"] > 0
                for key in flow_by_continuation
            )
            depth_gain = sum(
                flow_by_continuation[key]["E"]
                > marconi_by_continuation[key]["E"]
                for key in flow_by_continuation
            )
            marconi_redundant = redundant_compatible_checkpoint_ids(
                snapshot, marconi.selected_checkpoint_ids
            )
            flow_redundant = redundant_compatible_checkpoint_ids(
                snapshot, flow.selected_checkpoint_ids
            )
            redundant_gain = max(0, len(marconi_redundant) - len(flow_redundant))
            labels = []
            if exact_gain > 0:
                labels.append("exact_parent_coverage_gain")
                counts["exact_parent_coverage_gain_cases"] += 1
                budget_counts["exact_parent_coverage_gain_cases"] += 1
            if depth_gain > 0:
                labels.append("compatible_checkpoint_depth_gain")
                counts["compatible_checkpoint_depth_gain_cases"] += 1
                budget_counts["compatible_checkpoint_depth_gain_cases"] += 1
            if redundant_gain > 0:
                labels.append("redundant_checkpoint_avoidance")
                counts["redundant_checkpoint_avoidance_cases"] += 1
                budget_counts["redundant_checkpoint_avoidance_cases"] += 1
            if not labels:
                labels.append("no_listed_factor")
                counts["no_listed_factor_cases"] += 1
                budget_counts["no_listed_factor_cases"] += 1
            combination = "+".join(labels)
            combinations[combination] = combinations.get(combination, 0) + 1
            total_exact_parent_gain += exact_gain
            total_deeper_continuations += depth_gain
            total_redundant_reduction += redundant_gain
    denominator = counts["different_selection_cases"]
    return {
        "definitions": {
            "exact_parent_coverage_gain": (
                "FlowState 使更多 continuation 达到 E=T，而 Marconi 对应 G>0"
            ),
            "compatible_checkpoint_depth_gain": (
                "至少一个 continuation 的 FlowState E 高于 Marconi E"
            ),
            "redundant_checkpoint_avoidance": (
                "Marconi 中对当前 pending 兼容但移除后所有 E 均不变的 selected checkpoint 更多"
            ),
            "overlap_allowed": True,
        },
        "counts": counts,
        "fractions_among_different_selections": {
            "exact_parent_coverage_gain": counts[
                "exact_parent_coverage_gain_cases"
            ]
            / denominator,
            "compatible_checkpoint_depth_gain": counts[
                "compatible_checkpoint_depth_gain_cases"
            ]
            / denominator,
            "redundant_checkpoint_avoidance": counts[
                "redundant_checkpoint_avoidance_cases"
            ]
            / denominator,
        },
        "total_exact_parent_demands_gained": total_exact_parent_gain,
        "total_continuations_with_deeper_frontier": total_deeper_continuations,
        "total_redundant_selected_checkpoints_avoided": total_redundant_reduction,
        "overlap_combinations": combinations,
        "by_budget": by_budget,
        "interpretation": (
            "该分解只描述冻结 selection 的结构差异，不是重新运行策略或因果 ablation"
        ),
    }


def redundant_compatible_checkpoint_ids(
    snapshot: AuditSnapshot,
    selected_checkpoint_ids: Sequence[str],
) -> tuple[str, ...]:
    """返回对当前 pending 兼容但移除后全部 E 均不变的 selected checkpoint。"""
    candidate_by_id = {
        candidate.checkpoint_id: candidate for candidate in snapshot.candidates
    }
    selected = tuple(candidate_by_id[item] for item in selected_checkpoint_ids)
    frontiers = tuple(
        executable_frontier(continuation, selected)
        for continuation in snapshot.continuations
    )
    redundant = []
    for candidate in selected:
        if not any(
            is_compatible(candidate, continuation)
            for continuation in snapshot.continuations
        ):
            continue
        remaining = tuple(item for item in selected if item != candidate)
        if tuple(
            executable_frontier(continuation, remaining)
            for continuation in snapshot.continuations
        ) == frontiers:
            redundant.append(candidate.checkpoint_id)
    return tuple(redundant)


def x4_tight_budget_analysis(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
    model: RecoveryCostModel,
) -> dict[str, Any]:
    """审计主集合 X>=4 在三个受限预算下的状态覆盖。"""
    index = _main_result_index(rows)
    selected_snapshots = tuple(item for item in snapshots.values() if item.x >= 4)
    result = {
        "snapshot_count": len(selected_snapshots),
        "mean_x": mean(item.x for item in selected_snapshots),
        "budgets": {},
    }
    for ratio in BUDGET_RATIOS[:3]:
        budget_rows = []
        for snapshot in selected_snapshots:
            marconi = index[(snapshot.snapshot_id, ratio, "Marconi-style")]
            flow = index[(snapshot.snapshot_id, ratio, "FlowState")]
            marconi_effect = calculate_selection_effect(
                snapshot, marconi.selected_checkpoint_ids, model
            )
            flow_effect = calculate_selection_effect(
                snapshot, flow.selected_checkpoint_ids, model
            )
            budget_rows.append(
                {
                    "x": snapshot.x,
                    "k": flow.budget_k,
                    "marconi": marconi_effect,
                    "flowstate": flow_effect,
                }
            )
        pending_count = sum(item["x"] for item in budget_rows)
        marconi_hits = sum(
            item["marconi"]["executable_hit_count"] for item in budget_rows
        )
        flow_hits = sum(
            item["flowstate"]["executable_hit_count"] for item in budget_rows
        )
        marconi_cost = mean(
            item["marconi"]["total_cost_ms"] for item in budget_rows
        )
        flow_cost = mean(
            item["flowstate"]["total_cost_ms"] for item in budget_rows
        )
        result["budgets"][_ratio_label(ratio)] = {
            "mean_k": mean(item["k"] for item in budget_rows),
            "mean_k_over_x": mean(item["k"] / item["x"] for item in budget_rows),
            "marconi_exact_parent_coverage": marconi_hits / pending_count,
            "flowstate_exact_parent_coverage": flow_hits / pending_count,
            "marconi_mean_executable_frontier_tokens": _pending_weighted_mean(
                budget_rows, "marconi", "continuations", "E"
            ),
            "flowstate_mean_executable_frontier_tokens": _pending_weighted_mean(
                budget_rows, "flowstate", "continuations", "E"
            ),
            "marconi_mean_gap_tokens": _pending_weighted_mean(
                budget_rows, "marconi", "continuations", "G"
            ),
            "flowstate_mean_gap_tokens": _pending_weighted_mean(
                budget_rows, "flowstate", "continuations", "G"
            ),
            "marconi_mean_total_cost_ms_per_snapshot": marconi_cost,
            "flowstate_mean_total_cost_ms_per_snapshot": flow_cost,
            "relative_cost_reduction": (
                (marconi_cost - flow_cost) / marconi_cost
            ),
        }
    result["interpretation"] = (
        "25% 时所有 X>=4 快照均只有 K=1，平均 K/X 很低；两策略都只能覆盖少量需求，"
        "主要总成本由共同未覆盖需求决定。预算增加后，FlowState 可按位置感知边际成本连续覆盖"
        "更昂贵需求，而 Marconi 仍按 recency 与 FLOP efficiency 分配。"
    )
    return result


def full_budget_analysis(
    snapshots: Mapping[str, AuditSnapshot],
    rows: Sequence[FrozenPolicyResult],
) -> dict[str, Any]:
    """验证 K=X 的 demand-sufficient 语义及 baseline 漏覆盖原因。"""
    index = _main_result_index(rows)
    policies = {}
    flow_exact_parent_set_violations = 0
    for policy in POLICY_NAMES:
        snapshot_full_coverage = 0
        selected_exact_parent = 0
        selected_compatible_non_exact = 0
        selected_without_pending_compatibility = 0
        missing_exact_parent_demands = 0
        total_selected = 0
        total_cost = 0.0
        for snapshot in snapshots.values():
            row = index[(snapshot.snapshot_id, 1.0, policy)]
            selected_ids = set(row.selected_checkpoint_ids)
            selected_candidates = tuple(
                candidate
                for candidate in snapshot.candidates
                if candidate.checkpoint_id in selected_ids
            )
            all_exact_ids = {
                checkpoint_id
                for continuation in snapshot.continuations
                for checkpoint_id in exact_parent_ids(snapshot, continuation)
            }
            if policy == "FlowState" and not all_exact_ids.issubset(selected_ids):
                flow_exact_parent_set_violations += 1
            covered = 0
            for continuation in snapshot.continuations:
                if any(
                    candidate.checkpoint_id in selected_ids
                    for candidate in snapshot.candidates
                    if candidate.checkpoint_id
                    in exact_parent_ids(snapshot, continuation)
                ):
                    covered += 1
            if covered == len(snapshot.continuations):
                snapshot_full_coverage += 1
            missing_exact_parent_demands += len(snapshot.continuations) - covered
            for candidate in selected_candidates:
                compatible = tuple(
                    continuation
                    for continuation in snapshot.continuations
                    if is_compatible(candidate, continuation)
                )
                if candidate.checkpoint_id in all_exact_ids:
                    selected_exact_parent += 1
                elif compatible:
                    selected_compatible_non_exact += 1
                else:
                    selected_without_pending_compatibility += 1
            total_selected += len(selected_candidates)
            total_cost += row.total_formal_recovery_cost_ms
        policies[policy] = {
            "snapshots_with_all_exact_parent_demands_covered": snapshot_full_coverage,
            "missing_exact_parent_demands": missing_exact_parent_demands,
            "selected_checkpoint_count": total_selected,
            "selected_exact_parent_count": selected_exact_parent,
            "selected_compatible_non_exact_count": selected_compatible_non_exact,
            "selected_without_pending_compatibility_count": (
                selected_without_pending_compatibility
            ),
            "mean_total_cost_ms_per_snapshot": total_cost / len(snapshots),
        }
    return {
        "snapshot_count": len(snapshots),
        "condition": "K=X",
        "flowstate_exact_parent_set_violations": flow_exact_parent_set_violations,
        "flowstate_all_gaps_zero": policies["FlowState"][
            "mean_total_cost_ms_per_snapshot"
        ]
        == 0.0,
        "policies": policies,
        "interpretation": (
            "K=X 恰好足以保留每个 distinct exact-parent demand。FlowState 的正式目标使其选择"
            "这些 demand，因而 E=T、G=0、EHR=100%。baseline 可能把相同 K 用于较浅兼容状态"
            "或与当前 pending 无关的历史状态。该点验证 demand sufficiency，不是独立 latency 性能结论。"
        ),
        "claim_boundary": "DEMAND-SUFFICIENT SANITY POINT, NOT CORE PERFORMANCE CLAIM",
    }


def run_audit(
    result_directory: Path = FORMAL_RESULT_DIRECTORY,
    output_directory: Path | None = None,
) -> Path:
    """执行只读解释审计并写入独立 artifact。"""
    repository_root = Path(__file__).resolve().parents[2]
    result_hashes_before = _hash_directory(result_directory)
    protected_before = _hash_paths(repository_root, PROTECTED_PATHS)
    main_snapshots, _, all_rows = load_formal_results(result_directory)
    main_rows = tuple(row for row in all_rows if row.cohort == "main")
    model = RecoveryCostModel()
    verification = verify_frozen_results(main_snapshots, main_rows, model)
    result_index = _main_result_index(main_rows)
    representative_descriptors = select_representative_cases(
        main_snapshots, main_rows
    )
    representative_cases = tuple(
        audit_representative_case(
            descriptor,
            main_snapshots[descriptor["snapshot_id"]],
            result_index,
            model,
        )
        for descriptor in representative_descriptors
    )
    decomposition = benefit_decomposition(main_snapshots, main_rows, model)
    x4_analysis = x4_tight_budget_analysis(main_snapshots, main_rows, model)
    full_analysis = full_budget_analysis(main_snapshots, main_rows)
    marconi_sanity = _build_marconi_sanity(
        main_rows,
        verification,
        representative_cases,
    )
    result_hashes_after = _hash_directory(result_directory)
    protected_after = _hash_paths(repository_root, PROTECTED_PATHS)
    if result_hashes_before != result_hashes_after:
        raise RuntimeError("Step 10E 正式结果在审计期间发生变化")
    if protected_before != protected_after:
        raise RuntimeError("正式模型、策略或 protocol 在审计期间发生变化")
    marconi_sanity["formal_results_read_only"] = True
    marconi_sanity["protected_sources_unchanged"] = True
    marconi_sanity["gpu_executed"] = False
    marconi_sanity["resampling_executed"] = False
    marconi_sanity["new_policy_sweep_executed"] = False

    if output_directory is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_directory = DEFAULT_OUTPUT_ROOT / f"tracelab_sanity_audit_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_audit_artifacts(
        output_directory,
        result_directory,
        representative_cases,
        marconi_sanity,
        decomposition,
        x4_analysis,
        full_analysis,
        result_hashes_before,
        protected_before,
    )
    return output_directory


def write_constrained_representative_artifacts(
    result_directory: Path = FORMAL_RESULT_DIRECTORY,
    audit_directory: Path = FROZEN_SANITY_AUDIT_DIRECTORY,
) -> tuple[dict[str, Any], ...]:
    """在既有 Step 10F artifact 中写入受限预算案例，不运行策略。"""
    required_audit_files = (
        "marconi_sanity.json",
        "selected_case_audit.csv",
        "full_budget_analysis.json",
    )
    missing = tuple(
        name for name in required_audit_files if not (audit_directory / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Step 10F artifact 缺少文件：{missing}")
    sanity = json.loads(
        (audit_directory / "marconi_sanity.json").read_text(encoding="utf-8")
    )
    if Path(sanity["source_result_directory"]).resolve() != result_directory.resolve():
        raise ValueError("Step 10F 与 Step 10E 来源目录不一致")
    if sanity["status"] != "PASS":
        raise ValueError("Step 10F sanity gate 不是 PASS")

    target_names = {
        "constrained_representative_cases.csv",
        "constrained_representative_cases.md",
    }
    existing_targets = tuple(
        name for name in sorted(target_names) if (audit_directory / name).exists()
    )
    if existing_targets:
        raise FileExistsError(f"受限预算案例 artifact 已存在：{existing_targets}")
    formal_hashes_before = _hash_directory(result_directory)
    audit_hashes_before = {
        key: value
        for key, value in _hash_directory(audit_directory).items()
        if key not in target_names
    }
    main_snapshots, _, all_rows = load_formal_results(result_directory)
    main_rows = tuple(row for row in all_rows if row.cohort == "main")
    result_index = _main_result_index(main_rows)
    descriptors = select_constrained_representative_cases(
        main_snapshots,
        main_rows,
    )
    cases = tuple(
        audit_constrained_representative_case(
            descriptor,
            main_snapshots[descriptor["snapshot_id"]],
            result_index,
        )
        for descriptor in descriptors
    )
    _write_selected_case_csv(
        audit_directory / "constrained_representative_cases.csv",
        cases,
    )
    (audit_directory / "constrained_representative_cases.md").write_text(
        _render_constrained_cases(cases, sanity),
        encoding="utf-8",
    )
    if formal_hashes_before != _hash_directory(result_directory):
        raise RuntimeError("Step 10E 正式结果在案例选择期间发生变化")
    audit_hashes_after = {
        key: value
        for key, value in _hash_directory(audit_directory).items()
        if key not in target_names
    }
    if audit_hashes_before != audit_hashes_after:
        raise RuntimeError("Step 10F 原有 artifact 在案例选择期间发生变化")
    return cases


def _build_marconi_sanity(
    rows: Sequence[FrozenPolicyResult],
    verification: Mapping[str, Any],
    representative_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = _main_result_index(rows)
    snapshot_ids = sorted({row.snapshot_id for row in rows if row.cohort == "main"})
    comparisons = {}
    for baseline in ("Global-LRU", "KVFlow-style"):
        budget_rows = {}
        for ratio in BUDGET_RATIOS:
            marconi_wins = ties = losses = 0
            for snapshot_id in snapshot_ids:
                marconi = index[(snapshot_id, ratio, "Marconi-style")]
                other = index[(snapshot_id, ratio, baseline)]
                difference = (
                    other.total_formal_recovery_cost_ms
                    - marconi.total_formal_recovery_cost_ms
                )
                if difference > FLOAT_TOLERANCE_MS:
                    marconi_wins += 1
                elif difference < -FLOAT_TOLERANCE_MS:
                    losses += 1
                else:
                    ties += 1
            budget_rows[_ratio_label(ratio)] = {
                "marconi_win": marconi_wins,
                "tie": ties,
                "marconi_loss": losses,
            }
        comparisons[baseline] = budget_rows
    return {
        "status": "PASS" if not verification["implementation_bug_found"] else "FAIL",
        "verification": dict(verification),
        "marconi_vs_lru_kvflow": comparisons,
        "representative_case_ids": tuple(
            {
                "category": item["category"],
                "snapshot_id": item["snapshot_id"],
                "budget_ratio": item["budget_ratio"],
            }
            for item in representative_cases
        ),
        "diagnosis": (
            "Marconi-style 的冻结 utility 优化全局 recency 与 parent-relative incremental FLOP efficiency，"
            "不读取当前 pending coverage、E/G 或 Phi(G,T)。因此它可把有限 K 分配给较新或高 FLOP-efficiency、"
            "但对当前 pending 较浅、冗余或无关的历史状态。排序复核为零 mismatch，现有证据支持语义差异，"
            "不支持实现 bug 解释。"
        ),
        "fairness_boundary": (
            "这是预注册的 Marconi-style snapshot adaptation，不是原系统完整运行环境复现；alpha、recency、"
            "FLOP proxy 均在观察结果前冻结，未做事后调参。"
        ),
    }


def _expected_lru_selection(
    snapshot: AuditSnapshot,
    budget_k: int,
) -> tuple[str, ...]:
    ordered = sorted(
        snapshot.candidates,
        key=lambda candidate: (
            -float(snapshot.candidate_metadata[candidate.checkpoint_id]["last_access"]),
            -int(snapshot.candidate_metadata[candidate.checkpoint_id]["creation_order"]),
            candidate.checkpoint_id,
        ),
    )
    return tuple(candidate.checkpoint_id for candidate in ordered[:budget_k])


def _expected_kvflow_selection(
    snapshot: AuditSnapshot,
    budget_k: int,
) -> tuple[str, ...]:
    ordered = sorted(
        snapshot.candidates,
        key=lambda candidate: (
            1.0
            if any(
                is_compatible(candidate, continuation)
                for continuation in snapshot.continuations
            )
            else math.inf,
            -float(snapshot.candidate_metadata[candidate.checkpoint_id]["last_access"]),
            candidate.checkpoint_id,
        ),
    )
    return tuple(candidate.checkpoint_id for candidate in ordered[:budget_k])


def _expected_marconi_selection(
    snapshot: AuditSnapshot,
    budget_k: int,
) -> tuple[str, ...]:
    scores = {item["checkpoint_id"]: item["utility"] for item in marconi_candidate_scores(snapshot)}
    ordered = sorted(
        snapshot.candidates,
        key=lambda candidate: (-scores[candidate.checkpoint_id], candidate.checkpoint_id),
    )
    return tuple(candidate.checkpoint_id for candidate in ordered[:budget_k])


def _selection_cost(
    snapshot: AuditSnapshot,
    selected: Sequence[CheckpointCandidate],
    model: RecoveryCostModel,
) -> float:
    return sum(
        model.estimate(
            recovery_gap(continuation, selected),
            continuation.planning_target,
        )
        for continuation in snapshot.continuations
    )


def _pending_weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    collection: str,
    field: str,
) -> float:
    values = tuple(
        item[field]
        for row in rows
        for item in row[policy][collection]
    )
    return mean(values)


def _main_result_index(
    rows: Sequence[FrozenPolicyResult],
) -> dict[tuple[str, float, str], FrozenPolicyResult]:
    return {
        (row.snapshot_id, row.budget_ratio, row.policy): row
        for row in rows
        if row.cohort == "main"
    }


def _snapshot_from_manifest(item: Mapping[str, Any]) -> AuditSnapshot:
    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=str(row["checkpoint_id"]),
            workflow_id=str(row["workflow_id"]),
            lineage_path=tuple(str(value) for value in row["lineage_path"]),
            token_pos=int(row["token_pos"]),
            memory_bytes=int(row["memory_bytes"]),
            recurrent_resident=bool(row["recurrent_resident"]),
            fa_resident=bool(row["fa_resident"]),
        )
        for row in item["candidates"]
    )
    continuations = tuple(
        PendingContinuation(
            continuation_id=str(row["continuation_id"]),
            workflow_id=str(row["workflow_id"]),
            lineage_path=tuple(str(value) for value in row["lineage_path"]),
            anchor_pos=int(row["anchor_pos"]),
            resident_fa_frontier=int(row["resident_fa_frontier"]),
        )
        for row in item["continuations"]
    )
    metadata = {str(row["checkpoint_id"]): row for row in item["candidates"]}
    if any(int(row["steps_to_execution"]) != 1 for row in item["continuations"]):
        raise ValueError("冻结 KVFlow STE 不为一")
    if float(item["marconi_alpha"]) != 1.0:
        raise ValueError("冻结 Marconi alpha 不为一")
    if any(
        (
            bool(item["future_prefix_used"]),
            bool(item["runtime_residency_inferred"]),
            bool(item["llm_level_branching_introduced"]),
        )
    ):
        raise ValueError("冻结 manifest 违反 TraceLab 信息边界")
    return AuditSnapshot(
        snapshot_id=str(item["snapshot_id"]),
        cohort=str(item["cohort"]),
        provider=str(item["provider"]),
        concurrency_bucket=str(item["concurrency_bucket"]),
        x=int(item["x"]),
        candidates=candidates,
        continuations=continuations,
        candidate_metadata=metadata,
    )


def _policy_result_from_row(row: Mapping[str, str]) -> FrozenPolicyResult:
    return FrozenPolicyResult(
        cohort=row["cohort"],
        snapshot_id=row["snapshot_id"],
        provider=row["provider"],
        concurrency_bucket=row["concurrency_bucket"],
        x=int(row["x"]),
        candidate_count=int(row["candidate_count"]),
        pending_count=int(row["pending_count"]),
        budget_ratio=float(row["budget_ratio"]),
        budget_k=int(row["budget_k"]),
        policy=row["policy"],
        selected_checkpoint_ids=tuple(json.loads(row["selected_checkpoint_ids"])),
        total_recovery_gap_tokens=int(row["total_recovery_gap_tokens"]),
        mean_recovery_gap_tokens=float(row["mean_recovery_gap_tokens"]),
        total_formal_recovery_cost_ms=float(row["total_formal_recovery_cost_ms"]),
        executable_hit_count=int(row["executable_hit_count"]),
        executable_hit_ratio=float(row["executable_hit_ratio"]),
        continuation_results=tuple(json.loads(row["continuation_results"])),
    )


def _hash_directory(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _ratio_label(ratio: float) -> str:
    return f"{int(round(ratio * 100))}%"


def _write_audit_artifacts(
    output: Path,
    result_directory: Path,
    representative_cases: Sequence[Mapping[str, Any]],
    marconi_sanity: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    x4_analysis: Mapping[str, Any],
    full_analysis: Mapping[str, Any],
    result_hashes: Mapping[str, str],
    protected_hashes: Mapping[str, str],
) -> None:
    _write_selected_case_csv(output / "selected_case_audit.csv", representative_cases)
    payloads = {
        "marconi_sanity.json": {
            **marconi_sanity,
            "source_result_directory": str(result_directory),
            "source_result_hashes": result_hashes,
            "protected_source_hashes": protected_hashes,
        },
        "benefit_decomposition.json": decomposition,
        "x4_tight_budget_analysis.json": x4_analysis,
        "full_budget_analysis.json": full_analysis,
    }
    for name, payload in payloads.items():
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "reviewer_attack.md").write_text(
        _render_reviewer_attack(marconi_sanity, x4_analysis, full_analysis),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _render_readme(marconi_sanity, decomposition, x4_analysis, full_analysis),
        encoding="utf-8",
    )


def _write_selected_case_csv(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
) -> None:
    rows = []
    for item in cases:
        rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in item.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_constrained_cases(
    cases: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, Any],
) -> str:
    """渲染受限预算案例说明与逐 pending 对照。"""
    lines = [
        "# TraceLab 受限预算代表案例",
        "",
        "本文件只读取 Step 10E 冻结结果与 Step 10F 审计 artifact，没有重新运行策略、修改模型、重采样或调用 GPU。",
        "",
        "## 确定性选择规则",
        "",
        "- 只考虑 25%、50%、75% 预算中 FlowState 成本严格低于 Marconi 且 selection 不同的冻结记录。",
        f"- 优先使用 N<={INTERPRETABLE_CANDIDATE_LIMIT} 且 pending<=6 的规模适中案例，再按 absolute reduction 降序、N、pending 数和 snapshot_id 排序。",
        "- 四个 snapshot 不重复；50% 与 75% 案例要求 X>=4，额外 X>=4 案例固定来自 25% 紧预算。",
        "- 该规则只用于解释案例，不改变 Step 10E aggregate、policy selection 或 protocol。",
        "",
        "## 案例摘要",
        "",
        "| 类别 | Snapshot | X | N | K | Budget | Marconi cost | FlowState cost | Absolute reduction | Relative reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cases:
        lines.append(
            "| {category} | `{snapshot_id}` | {X} | {N} | {K} | {budget:.0%} | "
            "{marconi:.3f} ms | {flowstate:.3f} ms | {absolute:.3f} ms | {relative:.3f}% |".format(
                category=item["category"],
                snapshot_id=item["snapshot_id"],
                X=item["X"],
                N=item["N"],
                K=item["K"],
                budget=item["budget_ratio"],
                marconi=item["marconi_total_recovery_cost_ms"],
                flowstate=item["flowstate_total_recovery_cost_ms"],
                absolute=item["absolute_reduction_ms"],
                relative=item["relative_reduction_percent"],
            )
        )
    for item in cases:
        lines.extend(
            [
                "",
                f"## {item['category']}：`{item['snapshot_id']}`",
                "",
                f"- X={item['X']}，N={item['N']}，K={item['K']}，budget={item['budget_ratio']:.0%}",
                "- Marconi selection：" + ", ".join(
                    f"`{value}`" for value in item["marconi_selection"]
                ),
                "- FlowState selection：" + ", ".join(
                    f"`{value}`" for value in item["flowstate_selection"]
                ),
                f"- Marconi total cost：{item['marconi_total_recovery_cost_ms']:.3f} ms",
                f"- FlowState total cost：{item['flowstate_total_recovery_cost_ms']:.3f} ms",
                f"- Absolute reduction：{item['absolute_reduction_ms']:.3f} ms",
                f"- Relative reduction：{item['relative_reduction_percent']:.3f}%",
                f"- 机制说明：{item['mechanism_explanation']}",
                "",
                "| Continuation | T | Marconi E | Marconi G | Marconi Phi | FlowState E | FlowState G | FlowState Phi |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["per_pending_comparison"]:
            lines.append(
                "| `{continuation_id}` | {T} | {marconi_E} | {marconi_G} | "
                "{marconi_Phi_ms:.3f} ms | {flowstate_E} | {flowstate_G} | "
                "{flowstate_Phi_ms:.3f} ms |".format(**row)
            )

    old_examples = tuple(
        item
        for item in sanity["representative_case_ids"]
        if item["category"] == "clear_advantage"
        and float(item["budget_ratio"]) == 1.0
    )
    lines.extend(
        [
            "",
            "## 原 Step 10F 的 100% 案例角色",
            "",
            "以下原案例不删除，但统一标记为 **demand-sufficient sanity examples**，不作为 main benefit examples：",
            "",
        ]
    )
    lines.extend(
        f"- `{item['snapshot_id']}`，budget=100%，K=X" for item in old_examples
    )
    lines.extend(
        [
            "",
            "这些案例验证预算足以覆盖全部 distinct exact-parent demands 时 FlowState 应得到 G=0；它们不代表受限预算下的核心收益。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_readme(
    marconi: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    x4: Mapping[str, Any],
    full: Mapping[str, Any],
) -> str:
    counts = decomposition["counts"]
    lines = [
        "# TraceLab 结果 sanity 与 reviewer-attack 审计",
        "",
        "本审计只读取 Step 10E 冻结 artifact。没有重新采样、运行策略、调用 GPU、修改 recovery model 或 protocol。",
        "",
        "## 核心结论",
        "",
        f"- Marconi 实现复核：**{marconi['status']}**。排序与冻结 recency、parent-relative FLOP efficiency、alpha=1.0 完全一致。",
        "- Marconi 在部分 TraceLab 快照弱于 LRU/KVFlow，来自它与 current-pending recovery objective 的信号错位，不是已发现的实现错误。",
        f"- FlowState 与 Marconi selection 不同的案例为 {counts['different_selection_cases']}；其中 exact-parent coverage、兼容深度和 lineage redundancy reduction 可以重叠。",
        f"- 主集合 X>=4 共 {x4['snapshot_count']} 个快照；25% 时 mean K={x4['budgets']['25%']['mean_k']:.3f}，mean K/X={x4['budgets']['25%']['mean_k_over_x']:.3f}。",
        f"- K=X 时 FlowState exact-parent set 违规为 {full['flowstate_exact_parent_set_violations']}，全部 gap 为零：{full['flowstate_all_gaps_zero']}。",
        "",
        "## 解释边界",
        "",
        "100% 是 demand-sufficient sanity point，不是核心性能卖点。TraceLab 成本是独立校准 Phi(G,T) 的离线估计，不等于该 trace 上直接测得的 TTFT。TraceLab 没有显式 LLM-level DAG，因此 KVFlow 的 richer STE 信号没有被激活。",
        "",
    ]
    return "\n".join(lines)


def _render_reviewer_attack(
    marconi: Mapping[str, Any],
    x4: Mapping[str, Any],
    full: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Reviewer-attack questions",
            "",
            "## 1. 为什么 Marconi-style 弱于 LRU？",
            "",
            "冻结 Marconi utility 同时奖励 recency 与 parent-relative FLOP efficiency，但不读取当前 pending coverage、E/G 或 Phi(G,T)。它可能保留计算跨度较大却不服务当前 demand 的状态。排序复核零 mismatch，因此现有证据指向目标语义差异，而非实现错误。",
            "",
            "## 2. 这个 Marconi 比较是否不公平？",
            "",
            "它应被明确称为 Marconi-style snapshot adaptation，而非原系统端到端复现。recency 与 LRU 共用，FLOP proxy、alpha=1.0 均在观察结果前冻结，未调参；公平性来自相同 candidate、K 和统一 evaluator，外部有效性限制仍需披露。",
            "",
            "## 3. KVFlow 为什么与 LRU 接近？",
            "",
            "TraceLab 没有显式 LLM-level DAG，冻结协议令全部 known pending 的 STE=1。同 priority 内 KVFlow 回退到 LRU recency，仅会把无 compatible future 的 candidate 排到 priority=1 candidate 之后，因此大量退化为 LRU。",
            "",
            "## 4. 为什么 K=X 时 FlowState cost=0？",
            "",
            f"X 是 distinct exact-parent demands 数，K=X 恰好 demand-sufficient。审计中 FlowState exact-parent set 违规为 {full['flowstate_exact_parent_set_violations']}，因此每个 continuation 都有 E=T、G=0。",
            "",
            "## 5. TraceLab 是否天然偏向 FlowState？",
            "",
            "主 cohort 预注册为 X>=2，确实聚焦多状态竞争而非自然事件频率；但采样、X、预算和 metadata 在 policy comparison 前冻结，且未用 policy outcome 采样。结果适用于该结构覆盖 cohort，不能外推为完整 TraceLab population average。",
            "",
            "## 6. 为什么 X>=4 紧预算优势变小？",
            "",
            f"25% 时 mean K/X={x4['budgets']['25%']['mean_k_over_x']:.3f}，每个快照实际只有 K=1。两策略都只能覆盖少量 demand，共同未覆盖项主导总成本；到 50%/75% 后 FlowState 才有多个 slot 按位置感知边际收益连续分配。",
            "",
            "## 7. estimated recovery cost 是否等于真实 TTFT？",
            "",
            "不等于。它是独立 H100 profiler 校准的增量 recovery estimate；Step 10E 没有 GPU 或真实 TraceLab token replay。它支持 objective-level 离线比较，不应被表述为该 trace 上实测 latency speedup。",
            "",
            "## 8. TraceLab 没有 DAG 是否削弱结论？",
            "",
            "是。它削弱对 branching workflow 和 richer KVFlow STE 的结论。当前证据只覆盖由真实 round 顺序构造的线性 lineage 与 immediate tool-call continuation。",
            "",
            "## 9. FlowState 使用同一个 evaluator，胜出是否是同义反复？",
            "",
            "Step 10E 直接检验的是 allocator 与正式 executable-state objective 的对齐，不能单独证明真实 latency superiority。独立 held-out profiler 与受控 H100 runtime evidence提供外部支撑，但 TraceLab offline 结果本身必须标为 modeled objective comparison。",
            "",
            "## 10. 为什么 baseline 在 K=X 时仍会漏掉 exact parent？",
            "",
            "K=X 只等于 demand 数，不等于全部历史 candidate 数 N。baseline 可能把 slot 用于较浅兼容 checkpoint 或当前无 pending dependency 的历史状态；它们没有读取 exact-parent recovery demand。",
            "",
            "## 11. 105 个快照足以做 population claim 吗？",
            "",
            "不足。105-set 是 deterministic stratified structural sample，稀有 Codex、Medium/Large 和高 X 被有意提高权重。bootstrap 量化该样本内部不确定性，不修复相对于自然事件频率的 sampling bias。",
            "",
            "## 12. 100% 点能否作为核心性能提升？",
            "",
            "不能。它是 demand-sufficient correctness sanity point，展示当预算足够覆盖所有已知 exact-parent demand 时目标应归零；它不是有限资源区间中的主要 tradeoff，也不是实测 GPU speedup。",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """运行只读 sanity audit。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=FORMAL_RESULT_DIRECTORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--constrained-existing-audit", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.constrained_existing_audit is not None:
        cases = write_constrained_representative_artifacts(
            arguments.results,
            arguments.constrained_existing_audit,
        )
        print(
            json.dumps(
                {
                    "artifact": str(arguments.constrained_existing_audit),
                    "constrained_cases": len(cases),
                    "policy_rerun": False,
                    "gpu_executed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    output = run_audit(arguments.results, arguments.output)
    sanity = json.loads((output / "marconi_sanity.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(output),
                "marconi_sanity": sanity["status"],
                "implementation_bug_found": sanity["verification"][
                    "implementation_bug_found"
                ],
                "gpu_executed": sanity["gpu_executed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
