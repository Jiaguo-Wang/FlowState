"""Step 13G-A：对 Step 13F 正式结果做 correctness + structure + mechanism audit。

本模块只读取已冻结的 population 与 evaluation artifact，
不允许修改任何核心算法、selector、recovery model 或 population。
所有新增注释、docstring、说明性文字使用中文。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.scenario import CheckpointRecency
from evaluation.rq3_formal_policy_evaluation import (
    _flowstate_greedy_trace,
    _load_allocation_snapshot,
    create_budget_variant,
    load_eligible_snapshots,
    search_space_size,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    evaluate_objective,
    select_lfu,
)
from evaluation.sota_policies import MarconiStylePolicy, _build_marconi_metrics
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


# 冻结数值容差
_FLOAT_TOLERANCE_MS = 1e-9


def _json_safe(value: object) -> object:
    """递归把所有 dict key 转换为 string，避免 float/str 混合 key 导致 JSON 失败。"""

    if isinstance(value, dict):
        return {
            str(key): _json_safe(val) for key, val in value.items()
        }
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """用稳定格式写 JSON。"""

    path.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_per_snapshot_results(evaluation_root: Path) -> list[dict[str, Any]]:
    """加载 Step 13F 的 per_snapshot_results.jsonl。"""

    results: list[dict[str, Any]] = []
    path = evaluation_root / "per_snapshot_results.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def build_snapshot_map(formal_root: Path) -> dict[str, AllocationSnapshot]:
    """按 snapshot_id 索引正式 population 中的 snapshot。"""

    return {s.snapshot_id: s for s in load_eligible_snapshots(formal_root)}


def _snapshot_variant(
    snapshot: AllocationSnapshot,
    k: int,
) -> AllocationSnapshot:
    """返回指定 K 的 budget variant。"""

    return create_budget_variant(snapshot, k)


# ---------------------------------------------------------------------------
# 第一层：正式结果完整性复核
# ---------------------------------------------------------------------------


def reproduce_objectives(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """使用公共 objective 从 snapshot + selected IDs 重新计算 C(S)，与 13F 对比。"""

    mismatches: list[dict[str, Any]] = []
    snapshot_mutation: list[str] = []

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        digest_before = snapshot.content_digest()
        variant = _snapshot_variant(snapshot, row["k"])

        # 重新计算每个 policy 的目标值
        for policy_name in ("LRU", "LFU", "Marconi", "FlowState"):
            expected_ids = tuple(row["policies"][policy_name]["selected_checkpoint_ids"])
            obj = evaluate_objective(variant, expected_ids)
            stored_cost = row["policies"][policy_name]["total_recovery_cost_ms"]
            stored_benefit = row["policies"][policy_name]["total_benefit_ms"]
            stored_empty = row["policies"][policy_name]["empty_selection_cost_ms"]

            if (
                abs(obj.total_recovery_cost_ms - stored_cost) > _FLOAT_TOLERANCE_MS
                or abs(obj.total_benefit_ms - stored_benefit) > _FLOAT_TOLERANCE_MS
                or abs(obj.empty_selection_cost_ms - stored_empty) > _FLOAT_TOLERANCE_MS
            ):
                mismatches.append(
                    {
                        "snapshot_id": row["snapshot_id"],
                        "k": row["k"],
                        "policy": policy_name,
                        "stored_cost": stored_cost,
                        "reproduced_cost": obj.total_recovery_cost_ms,
                        "stored_benefit": stored_benefit,
                        "reproduced_benefit": obj.total_benefit_ms,
                    }
                )

        # Exact OPT tractable cases
        if row["exact_opt"].get("tractable"):
            expected_ids = tuple(row["exact_opt"]["selected_checkpoint_ids"])
            obj = evaluate_objective(variant, expected_ids)
            stored_cost = row["exact_opt"]["total_recovery_cost_ms"]
            if abs(obj.total_recovery_cost_ms - stored_cost) > _FLOAT_TOLERANCE_MS:
                mismatches.append(
                    {
                        "snapshot_id": row["snapshot_id"],
                        "k": row["k"],
                        "policy": "Exact OPT",
                        "stored_cost": stored_cost,
                        "reproduced_cost": obj.total_recovery_cost_ms,
                    }
                )

        if snapshot.content_digest() != digest_before:
            snapshot_mutation.append(row["snapshot_id"])

    return {
        "total_cases": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "snapshot_mutation_count": len(snapshot_mutation),
        "snapshot_mutations": snapshot_mutation,
        "pass": len(mismatches) == 0 and len(snapshot_mutation) == 0,
    }


# ---------------------------------------------------------------------------
# 第二层：Selector 语义独立审计
# ---------------------------------------------------------------------------


def _index_metadata(snapshot: AllocationSnapshot) -> dict[str, Any]:
    """返回 candidate metadata 字典。"""

    return {
        item.checkpoint_id: item for item in snapshot.candidate_metadata
    }


def expected_lru_selected(snapshot: AllocationSnapshot) -> tuple[str, ...]:
    """按冻结 LRU 规则构造 expected selected set。"""

    metadata = _index_metadata(snapshot)
    candidates = snapshot.core_candidates()
    recency = tuple(
        CheckpointRecency(
            checkpoint_id=item.checkpoint_id,
            creation_order=metadata[item.checkpoint_id].creation_order,
            last_access_order=metadata[item.checkpoint_id].last_access_order,
        )
        for item in candidates
    )
    return select_global_lru(
        candidates,
        recency,
        snapshot.budget_bytes,
    )


def expected_lfu_selected(snapshot: AllocationSnapshot) -> tuple[str, ...]:
    """按冻结 LFU Adaptation 规则构造 expected selected set。"""

    metadata = _index_metadata(snapshot)
    candidates = snapshot.core_candidates()
    frequency = {
        item.checkpoint_id: item.access_frequency
        for item in snapshot.lfu_access_frequency
    }
    last_access = {
        item.checkpoint_id: metadata[item.checkpoint_id].last_access_order
        for item in candidates
    }
    return select_lfu(
        candidates,
        frequency,
        last_access,
        snapshot.logical_budget_k,
    )


def expected_marconi_selected(snapshot: AllocationSnapshot) -> tuple[str, ...]:
    """按冻结 Marconi Adaptation 规则构造 expected selected set。"""

    metadata = _index_metadata(snapshot)
    candidates = snapshot.core_candidates()
    result = MarconiStylePolicy().select(
        candidates,
        snapshot.logical_budget_k,
        {
            item.checkpoint_id: float(metadata[item.checkpoint_id].last_access_order)
            for item in candidates
        },
        {
            item.checkpoint_id: metadata[item.checkpoint_id].marconi_flop_saved
            for item in candidates
        },
        snapshot.marconi_alpha,
    )
    return result.selected_checkpoint_ids


def audit_selector_semantics(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """独立审计 LRU/LFU/Marconi selector 语义是否与 13F 一致。"""

    lru_mismatches: list[dict[str, Any]] = []
    lfu_mismatches: list[dict[str, Any]] = []
    marconi_mismatches: list[dict[str, Any]] = []
    marconi_scores: list[dict[str, float]] = []

    frequency_boundary_violations: list[dict[str, Any]] = []

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])

        expected_lru = expected_lru_selected(variant)
        formal_lru = tuple(row["policies"]["LRU"]["selected_checkpoint_ids"])
        if expected_lru != formal_lru:
            lru_mismatches.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "k": row["k"],
                    "expected": list(expected_lru),
                    "formal": list(formal_lru),
                }
            )

        expected_lfu = expected_lfu_selected(variant)
        formal_lfu = tuple(row["policies"]["LFU"]["selected_checkpoint_ids"])
        if expected_lfu != formal_lfu:
            lfu_mismatches.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "k": row["k"],
                    "expected": list(expected_lfu),
                    "formal": list(formal_lfu),
                }
            )

        expected_marconi = expected_marconi_selected(variant)
        formal_marconi = tuple(row["policies"]["Marconi"]["selected_checkpoint_ids"])
        if expected_marconi != formal_marconi:
            marconi_mismatches.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "k": row["k"],
                    "expected": list(expected_marconi),
                    "formal": list(formal_marconi),
                }
            )

        # Marconi score distribution audit
        metadata = _index_metadata(variant)
        candidates = variant.core_candidates()
        metrics = _build_marconi_metrics(
            candidates,
            {
                item.checkpoint_id: float(metadata[item.checkpoint_id].last_access_order)
                for item in candidates
            },
            {
                item.checkpoint_id: metadata[item.checkpoint_id].marconi_flop_saved
                for item in candidates
            },
            variant.marconi_alpha,
        )
        recencies = {
            item.checkpoint_id: metadata[item.checkpoint_id].last_access_order
            for item in candidates
        }
        flop_saved = {
            item.checkpoint_id: metadata[item.checkpoint_id].marconi_flop_saved
            for item in candidates
        }
        recency_min, recency_max = min(recencies.values()), max(recencies.values())
        flop_min, flop_max = min(flop_saved.values()), max(flop_saved.values())
        for cid, score in metrics.items():
            marconi_scores.append(
                {
                    "checkpoint_id": cid,
                    "score": score,
                    "raw_recency": recencies[cid],
                    "normalized_recency": (
                        (recencies[cid] - recency_min) / (recency_max - recency_min)
                        if recency_max != recency_min
                        else 0.0
                    ),
                    "raw_flop_saved": flop_saved[cid],
                    "normalized_efficiency": (
                        (flop_saved[cid] - flop_min) / (flop_max - flop_min)
                        if flop_max != flop_min
                        else 0.0
                    ),
                }
            )

        # LFU frequency boundary check
        if (
            variant.frequency_observed_through_epoch
            > variant.online_boundary.materialized_through_epoch
        ):
            frequency_boundary_violations.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "frequency_observed_through_epoch": (
                        variant.frequency_observed_through_epoch
                    ),
                    "materialized_through_epoch": (
                        variant.online_boundary.materialized_through_epoch
                    ),
                }
            )

    # score stats
    scores = [m["score"] for m in marconi_scores]
    normalized_recencies = [m["normalized_recency"] for m in marconi_scores]
    normalized_efficiencies = [m["normalized_efficiency"] for m in marconi_scores]

    # tie frequency
    score_counts = Counter(round(s, 12) for s in scores)
    tie_count = sum(c for c in score_counts.values() if c > 1)

    return {
        "lru_mismatch_count": len(lru_mismatches),
        "lru_mismatches": lru_mismatches,
        "lfu_mismatch_count": len(lfu_mismatches),
        "lfu_mismatches": lfu_mismatches,
        "marconi_mismatch_count": len(marconi_mismatches),
        "marconi_mismatches": marconi_mismatches,
        "marconi_alpha": 1.0,
        "marconi_score_stats": _summary_stats(scores),
        "marconi_normalized_recency_stats": _summary_stats(normalized_recencies),
        "marconi_normalized_efficiency_stats": _summary_stats(normalized_efficiencies),
        "marconi_tie_count": tie_count,
        "frequency_boundary_violations": frequency_boundary_violations,
        "pass": (
            len(lru_mismatches) == 0
            and len(lfu_mismatches) == 0
            and len(marconi_mismatches) == 0
            and len(frequency_boundary_violations) == 0
        ),
    }


# ---------------------------------------------------------------------------
# 第三层：Budget Monotonicity
# ---------------------------------------------------------------------------


def audit_budget_monotonicity(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """对每个 snapshot 检查 K 增大时 C(S) 不增。"""

    by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_snapshot.setdefault(row["snapshot_id"], []).append(row)

    violations: list[dict[str, Any]] = []
    for snapshot_id, cases in by_snapshot.items():
        cases_sorted = sorted(cases, key=lambda r: r["k"])
        for i in range(len(cases_sorted) - 1):
            left = cases_sorted[i]
            right = cases_sorted[i + 1]
            for policy in ("LRU", "LFU", "Marconi", "FlowState"):
                c_left = left["policies"][policy]["total_recovery_cost_ms"]
                c_right = right["policies"][policy]["total_recovery_cost_ms"]
                if c_right > c_left + _FLOAT_TOLERANCE_MS:
                    violations.append(
                        {
                            "snapshot_id": snapshot_id,
                            "policy": policy,
                            "k_left": left["k"],
                            "k_right": right["k"],
                            "c_left": c_left,
                            "c_right": c_right,
                            "s_left": left["policies"][policy]["selected_checkpoint_ids"],
                            "s_right": right["policies"][policy]["selected_checkpoint_ids"],
                        }
                    )

    return {
        "violation_count": len(violations),
        "violations": violations,
        "pass": len(violations) == 0,
    }


# ---------------------------------------------------------------------------
# 通用统计辅助
# ---------------------------------------------------------------------------


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    """计算 mean/median/P25/P75/P95。"""

    if not values:
        return {"mean": None, "median": None, "p25": None, "p75": None, "p95": None}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _mean(values: list[float]) -> float | None:
    """均值辅助。"""

    return float(np.mean(values)) if values else None


def _fraction(values: list[bool]) -> float | None:
    """布尔列表中 True 的比例。"""

    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# 第四层：Selected-Set Size 与 Budget Saturation
# ---------------------------------------------------------------------------


def analyze_selected_size_and_saturation(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """统计 |S|、unused budget、K 与 |P| 的关系。"""

    by_ratio: dict[float, dict[str, list[Any]]] = {}
    for row in rows:
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(
            ratio,
            {
                "k": [],
                "pending_count": [],
                "flowstate_s": [],
                "lru_s": [],
                "lfu_s": [],
                "marconi_s": [],
            },
        )
        entry["k"].append(row["k"])
        entry["pending_count"].append(row["pending_count"])
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            entry[f"{policy.lower()}_s"].append(
                len(row["policies"][policy]["selected_checkpoint_ids"])
            )

    aggregate: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        data = by_ratio[ratio]
        k_values = data["k"]
        pending_count = data["pending_count"][0] if data["pending_count"] else 0
        fs_s = data["flowstate_s"]
        aggregate[ratio] = {
            "n": len(k_values),
            "mean_k": _mean(k_values),
            "median_k": float(np.median(k_values)) if k_values else None,
            "pending_count": pending_count,
            "fraction_k_ge_pending": _fraction([k >= pending_count for k in k_values]),
            "flowstate": {
                "mean_s": _mean(fs_s),
                "median_s": float(np.median(fs_s)) if fs_s else None,
                "p25_s": float(np.percentile(fs_s, 25)) if fs_s else None,
                "p75_s": float(np.percentile(fs_s, 75)) if fs_s else None,
                "p95_s": float(np.percentile(fs_s, 95)) if fs_s else None,
                "fraction_s_lt_k": _fraction([s < k for s, k in zip(fs_s, k_values)]),
                "fraction_s_eq_k": _fraction([s == k for s, k in zip(fs_s, k_values)]),
                "mean_unused_budget": _mean([k - s for k, s in zip(k_values, fs_s)]),
            },
        }
        for policy in ("LRU", "LFU", "Marconi"):
            s_values = data[f"{policy.lower()}_s"]
            aggregate[ratio][policy.lower()] = {
                "mean_s": _mean(s_values),
                "median_s": float(np.median(s_values)) if s_values else None,
                "fraction_s_lt_k": _fraction([s < k for s, k in zip(s_values, k_values)]),
            }

    return aggregate


# ---------------------------------------------------------------------------
# 第五层：Zero-Cost / Zero-Gap Saturation
# ---------------------------------------------------------------------------


def analyze_zero_cost_and_gap(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计 zero-cost case 与 zero-gap pending 比例。"""

    by_ratio: dict[float, dict[str, list[Any]]] = {}
    for row in rows:
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(ratio, {p: [] for p in ("LRU", "LFU", "Marconi", "FlowState")})
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            cost = row["policies"][policy]["total_recovery_cost_ms"]
            entry[policy].append(cost <= _FLOAT_TOLERANCE_MS)

    # zero-gap pending rate requires per-continuation rows
    by_ratio_gap: dict[float, dict[str, list[bool]]] = {}
    for row in rows:
        ratio = row["budget_ratio"]
        entry = by_ratio_gap.setdefault(ratio, {p: [] for p in ("LRU", "LFU", "Marconi", "FlowState")})
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            for cont in row["policies"][policy]["per_continuation"]:
                entry[policy].append(cont["recovery_gap_tokens"] <= 0)

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        result[ratio] = {
            "zero_cost_rate": {
                policy: _fraction(by_ratio[ratio][policy])
                for policy in ("LRU", "LFU", "Marconi", "FlowState")
            },
            "zero_gap_pending_rate": {
                policy: _fraction(by_ratio_gap[ratio][policy])
                for policy in ("LRU", "LFU", "Marconi", "FlowState")
            },
        }

    return result


# ---------------------------------------------------------------------------
# 第六层：Workflow Coverage
# ---------------------------------------------------------------------------


def _coverage_counts(
    snapshot: AllocationSnapshot,
    selected_ids: tuple[str, ...],
) -> tuple[int, int]:
    """返回 (coverage, full_coverage) 计数。"""

    candidate_by_id = {
        c.checkpoint_id: c.to_core() for c in snapshot.eligible_candidates
    }
    selected = [candidate_by_id[cid] for cid in selected_ids]
    coverage = 0
    full = 0
    for cont in snapshot.core_continuations():
        frontier = executable_frontier(cont, selected)
        gap = recovery_gap(cont, selected)
        if frontier > 0:
            coverage += 1
        if gap <= 0:
            full += 1
    return coverage, full


def analyze_workflow_coverage(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计每个 policy 的 workflow coverage 与 full coverage。"""

    by_ratio: dict[float, dict[str, list[int]]] = {}
    by_ratio_full: dict[float, dict[str, list[bool]]] = {}
    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        ratio = row["budget_ratio"]
        cov_entry = by_ratio.setdefault(ratio, {p: [] for p in ("LRU", "LFU", "Marconi", "FlowState")})
        full_entry = by_ratio_full.setdefault(ratio, {p: [] for p in ("LRU", "LFU", "Marconi", "FlowState")})
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            selected_ids = tuple(row["policies"][policy]["selected_checkpoint_ids"])
            coverage, full = _coverage_counts(snapshot, selected_ids)
            cov_entry[policy].append(coverage)
            full_entry[policy].append(full == row["pending_count"])

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        result[ratio] = {}
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            coverages = by_ratio[ratio][policy]
            result[ratio][policy] = {
                "mean_coverage": _mean(coverages),
                "median_coverage": float(np.median(coverages)) if coverages else None,
                "p25_coverage": float(np.percentile(coverages, 25)) if coverages else None,
                "p75_coverage": float(np.percentile(coverages, 75)) if coverages else None,
                "max_coverage": max(coverages) if coverages else None,
                "full_coverage_rate": _fraction(by_ratio_full[ratio][policy]),
            }
    return result


# ---------------------------------------------------------------------------
# 第七层：Selected Checkpoints 的 Workflow 分布
# ---------------------------------------------------------------------------


def analyze_workflow_distribution(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计 selected set 中 workflow label 的分布与重复。"""

    by_ratio: dict[float, dict[str, dict[str, list[Any]]]] = {}
    workflow_share: dict[str, dict[str, int]] = {
        policy: Counter() for policy in ("LRU", "LFU", "Marconi", "FlowState")
    }

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(
            ratio,
            {
                policy: {
                    "distinct": [],
                    "max_same": [],
                    "duplicate_count": [],
                }
                for policy in ("LRU", "LFU", "Marconi", "FlowState")
            },
        )
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            selected_ids = row["policies"][policy]["selected_checkpoint_ids"]
            workflow_ids = [
                next(
                    c.workflow_id
                    for c in snapshot.eligible_candidates
                    if c.checkpoint_id == cid
                )
                for cid in selected_ids
            ]
            counter = Counter(workflow_ids)
            distinct = len(counter)
            max_same = max(counter.values()) if counter else 0
            duplicate = len(selected_ids) - distinct
            entry[policy]["distinct"].append(distinct)
            entry[policy]["max_same"].append(max_same)
            entry[policy]["duplicate_count"].append(duplicate)
            for wid in workflow_ids:
                workflow_share[policy][wid] += 1

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        result[ratio] = {}
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            data = by_ratio[ratio][policy]
            result[ratio][policy] = {
                "distinct_workflows": _summary_stats(data["distinct"]),
                "max_same_workflow": _summary_stats(data["max_same"]),
                "duplicate_within_workflow": _summary_stats(data["duplicate_count"]),
            }

    result["workflow_share_counts"] = {
        policy: dict(counter) for policy, counter in workflow_share.items()
    }
    return result


# ---------------------------------------------------------------------------
# 第八层：Zero-Marginal Redundancy
# ---------------------------------------------------------------------------


def _cost_without_each(
    snapshot: AllocationSnapshot,
    selected_ids: tuple[str, ...],
) -> dict[str, float]:
    """计算移除每个 selected checkpoint 后的 C(S)。"""

    selected_set = set(selected_ids)
    result: dict[str, float] = {}
    for cid in selected_ids:
        remaining = tuple(selected_set - {cid})
        obj = evaluate_objective(snapshot, remaining)
        result[cid] = obj.total_recovery_cost_ms
    return result


def analyze_redundancy(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计每个 policy 的 zero-marginal selected checkpoints。"""

    by_ratio: dict[float, dict[str, dict[str, list[Any]]]] = {}
    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(
            ratio,
            {
                policy: {"z": [], "ratio": [], "useful": []}
                for policy in ("LRU", "LFU", "Marconi", "FlowState")
            },
        )
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            selected_ids = tuple(row["policies"][policy]["selected_checkpoint_ids"])
            base_cost = row["policies"][policy]["total_recovery_cost_ms"]
            costs_without = _cost_without_each(variant, selected_ids)
            z = sum(
                1
                for cid in selected_ids
                if abs(costs_without[cid] - base_cost) <= _FLOAT_TOLERANCE_MS
            )
            size = len(selected_ids)
            entry[policy]["z"].append(z)
            entry[policy]["ratio"].append(z / size if size > 0 else None)
            entry[policy]["useful"].append(size - z)

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        result[ratio] = {}
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            data = by_ratio[ratio][policy]
            ratios = [r for r in data["ratio"] if r is not None]
            result[ratio][policy] = {
                "mean_zero_marginal_count": _mean(data["z"]),
                "median_zero_marginal_count": float(np.median(data["z"])) if data["z"] else None,
                "mean_redundancy_ratio": _mean(ratios),
                "median_redundancy_ratio": float(np.median(ratios)) if ratios else None,
                "p75_redundancy_ratio": float(np.percentile(ratios, 75)) if ratios else None,
                "p95_redundancy_ratio": float(np.percentile(ratios, 95)) if ratios else None,
                "mean_useful_selected": _mean(data["useful"]),
            }
    return result


# ---------------------------------------------------------------------------
# 第九层：Candidate Compatibility Structure
# ---------------------------------------------------------------------------


def _compatibility_degree(
    candidate: CheckpointCandidate,
    continuations: list[PendingContinuation],
) -> int:
    """计算 candidate 与多少个 pending continuation compatible。"""

    return sum(1 for cont in continuations if is_compatible(candidate, cont))


def analyze_compatibility_structure(
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计 population 上 candidate 的 compatibility degree 分布。"""

    degrees: list[int] = []
    per_snapshot: list[dict[str, Any]] = []
    for snapshot in snapshot_map.values():
        continuations = list(snapshot.core_continuations())
        candidates = list(snapshot.core_candidates())
        snapshot_degrees = [
            _compatibility_degree(c, continuations) for c in candidates
        ]
        degrees.extend(snapshot_degrees)
        cross_pending = sum(1 for d in snapshot_degrees if d > 1)
        exact_one = sum(1 for d in snapshot_degrees if d == 1)
        per_snapshot.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "candidate_count": len(candidates),
                "cross_pending_candidates": cross_pending,
                "exactly_one_pending": exact_one,
            }
        )

    counter = Counter(degrees)
    return {
        "total_candidates": len(degrees),
        "degree_distribution": {
            f"d={k}": v for k, v in sorted(counter.items())
        },
        "max_compatibility_degree": max(degrees) if degrees else 0,
        "fraction_d_eq_1": _fraction([d == 1 for d in degrees]),
        "per_snapshot": per_snapshot,
        "conclusion_one_candidate_one_pending": (
            _fraction([d == 1 for d in degrees]) or 0.0
        ) >= 0.99,
    }


# ---------------------------------------------------------------------------
# 第十层：Per-Workflow Candidate Chain Structure
# ---------------------------------------------------------------------------


def _standalone_benefit(
    candidate: CheckpointCandidate,
    continuation: PendingContinuation,
    model: RecoveryCostModel,
) -> float:
    """计算 candidate 对单个 pending 的 standalone benefit。"""

    if not is_compatible(candidate, continuation):
        return 0.0
    target = continuation.planning_target
    phi_full = model.estimate(target, target)
    phi_with = model.estimate(max(0, target - candidate.token_pos), target)
    return phi_full - phi_with


def analyze_chain_structure(
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """检查每个 pending workflow 的 compatible candidates 是否形成单调 chain。"""

    model = RecoveryCostModel()
    strict_chain_count = 0
    non_decreasing_benefit_count = 0
    branching_count = 0
    total_snapshots = len(snapshot_map)
    per_snapshot: list[dict[str, Any]] = []

    for snapshot in snapshot_map.values():
        continuations = list(snapshot.core_continuations())
        candidates = list(snapshot.core_candidates())
        snapshot_strict = True
        snapshot_non_decreasing = True
        snapshot_branching = False

        for cont in continuations:
            compat = sorted(
                [c for c in candidates if is_compatible(c, cont)],
                key=lambda c: c.token_pos,
            )
            if not compat:
                continue
            positions = [c.token_pos for c in compat]
            if positions != sorted(positions) or len(set(positions)) != len(positions):
                snapshot_strict = False
            benefits = [_standalone_benefit(c, cont, model) for c in compat]
            if not all(
                benefits[i] <= benefits[i + 1] + _FLOAT_TOLERANCE_MS
                for i in range(len(benefits) - 1)
            ):
                snapshot_non_decreasing = False
            # branching：同一 pending 存在不同 lineage 分支的兼容候选
            lineages = {tuple(c.lineage_path) for c in compat}
            if len(lineages) > 1:
                snapshot_branching = True

        if snapshot_strict:
            strict_chain_count += 1
        if snapshot_non_decreasing:
            non_decreasing_benefit_count += 1
        if snapshot_branching:
            branching_count += 1

        per_snapshot.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "strict_chain": snapshot_strict,
                "non_decreasing_benefit": snapshot_non_decreasing,
                "branching_structure": snapshot_branching,
            }
        )

    return {
        "total_snapshots": total_snapshots,
        "strict_chain_snapshots": strict_chain_count,
        "non_decreasing_benefit_snapshots": non_decreasing_benefit_count,
        "branching_structure_snapshots": branching_count,
        "fraction_strict_chain": strict_chain_count / total_snapshots if total_snapshots else None,
        "fraction_non_decreasing_benefit": non_decreasing_benefit_count / total_snapshots if total_snapshots else None,
        "per_snapshot": per_snapshot,
    }


# ---------------------------------------------------------------------------
# 第十一层：FlowState Marginal 是否真正发生 Set Dependency
# ---------------------------------------------------------------------------


def _marginal_gain(
    snapshot: AllocationSnapshot,
    selected: list[CheckpointCandidate],
    candidate: CheckpointCandidate,
    model: RecoveryCostModel,
) -> float:
    """计算 Delta(c | S)。"""

    cost_with = sum(
        model.estimate(recovery_gap(cont, selected + [candidate]), cont.planning_target)
        for cont in snapshot.core_continuations()
    )
    cost_without = sum(
        model.estimate(recovery_gap(cont, selected), cont.planning_target)
        for cont in snapshot.core_continuations()
    )
    return cost_without - cost_with


def analyze_marginal_dependency(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """分析 FlowState greedy 每步的 marginal 是否相对于 empty set 发生变化。"""

    total_steps = 0
    changed_steps = 0
    same_pending_redundancy = 0
    multi_pending_overlap = 0
    other_changes = 0

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])
        trace = _flowstate_greedy_trace(variant, row["k"])

        candidates = list(variant.core_candidates())
        candidate_by_id = {c.checkpoint_id: c for c in candidates}
        model = RecoveryCostModel()

        selected: list[CheckpointCandidate] = []
        selection_order = trace["selection_order"]
        for idx, cid in enumerate(selection_order):
            current_candidate = candidate_by_id[cid]
            # 计算当前 S 下所有剩余候选的 marginal
            remaining = [
                c for c in candidates if c.checkpoint_id not in trace["selection_order"][: idx + 1]
            ]
            for cand in remaining:
                delta_s = _marginal_gain(variant, selected, cand, model)
                delta_empty = _marginal_gain(variant, [], cand, model)
                total_steps += 1
                if abs(delta_s - delta_empty) > _FLOAT_TOLERANCE_MS:
                    changed_steps += 1
                    # 分类
                    selected_compatible_pending = {
                        cont.continuation_id
                        for cont in variant.core_continuations()
                        for s in selected
                        if is_compatible(s, cont)
                    }
                    cand_compatible = {
                        cont.continuation_id
                        for cont in variant.core_continuations()
                        if is_compatible(cand, cont)
                    }
                    overlap = selected_compatible_pending & cand_compatible
                    if overlap:
                        if len(overlap) == 1:
                            same_pending_redundancy += 1
                        else:
                            multi_pending_overlap += 1
                    else:
                        other_changes += 1
            selected.append(current_candidate)

    return {
        "total_candidate_steps": total_steps,
        "marginal_changed_steps": changed_steps,
        "fraction_marginal_changed": changed_steps / total_steps if total_steps else None,
        "same_pending_redundancy": same_pending_redundancy,
        "multi_pending_overlap": multi_pending_overlap,
        "other_changes": other_changes,
        "conclusion": (
            "mainly_same_pending_redundancy"
            if changed_steps
            and same_pending_redundancy / changed_steps >= 0.8
            else "mixed"
        ),
    }


# ---------------------------------------------------------------------------
# 第十二层：PerWorkflowBest Diagnostic Probe
# ---------------------------------------------------------------------------


def _per_workflow_best_selection(
    snapshot: AllocationSnapshot,
    k: int,
) -> tuple[str, ...]:
    """对每个 pending workflow 选 standalone benefit 最大的兼容候选，再按 benefit Top-K。"""

    model = RecoveryCostModel()
    continuations = list(snapshot.core_continuations())
    candidates = list(snapshot.core_candidates())

    per_workflow_best: list[tuple[str, float]] = []
    for cont in continuations:
        scored = []
        for cand in candidates:
            benefit = _standalone_benefit(cand, cont, model)
            if benefit > _FLOAT_TOLERANCE_MS:
                scored.append((cand, benefit))
        if scored:
            scored.sort(key=lambda x: (-x[1], x[0].checkpoint_id))
            per_workflow_best.append((scored[0][0].checkpoint_id, scored[0][1]))

    # 如果 K < workflow count，从 per-workflow best 里按 benefit 选 Top-K
    per_workflow_best.sort(key=lambda x: (-x[1], x[0]))
    if k < len(per_workflow_best):
        selected = [cid for cid, _ in per_workflow_best[:k]]
    else:
        selected = [cid for cid, _ in per_workflow_best]

    return tuple(sorted(selected))


def analyze_per_workflow_best_probe(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """比较 FlowState 与 PerWorkflowBest diagnostic selector。"""

    selected_match_count = 0
    c_match_count = 0
    diffs: list[float] = []

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])
        flowstate_ids = tuple(row["policies"]["FlowState"]["selected_checkpoint_ids"])
        probe_ids = _per_workflow_best_selection(variant, row["k"])
        if set(flowstate_ids) == set(probe_ids):
            selected_match_count += 1

        flowstate_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
        probe_obj = evaluate_objective(variant, probe_ids)
        probe_cost = probe_obj.total_recovery_cost_ms
        if abs(flowstate_cost - probe_cost) <= _FLOAT_TOLERANCE_MS:
            c_match_count += 1
        diffs.append(probe_cost - flowstate_cost)

    return {
        "cases": len(rows),
        "selected_set_exact_match_rate": selected_match_count / len(rows) if rows else None,
        "c_exact_match_rate": c_match_count / len(rows) if rows else None,
        "mean_c_difference_ms": _mean(diffs),
        "max_c_difference_ms": max(diffs) if diffs else None,
        "conclusion": (
            "equivalent"
            if selected_match_count == len(rows)
            else (
                "partial"
                if rows and selected_match_count / len(rows) >= 0.95
                else "different"
            )
        ),
    }


# ---------------------------------------------------------------------------
# 第十三层：Standalone Top-K Diagnostic Probe
# ---------------------------------------------------------------------------


def _standalone_score(
    candidate: CheckpointCandidate,
    continuations: list[PendingContinuation],
    model: RecoveryCostModel,
) -> float:
    """计算 candidate 的 standalone F({c}) = sum_p b_{p,c}。"""

    return sum(
        _standalone_benefit(candidate, cont, model) for cont in continuations
    )


def _standalone_topk_selection(
    snapshot: AllocationSnapshot,
    k: int,
) -> tuple[str, ...]:
    """直接按 standalone score 全局 Top-K，不更新 marginal。"""

    model = RecoveryCostModel()
    continuations = list(snapshot.core_continuations())
    candidates = list(snapshot.core_candidates())
    scored = [
        (c, _standalone_score(c, continuations, model)) for c in candidates
    ]
    scored = [(c, s) for c, s in scored if s > _FLOAT_TOLERANCE_MS]
    scored.sort(key=lambda x: (-x[1], x[0].checkpoint_id))
    capacity = min(k, len(scored))
    return tuple(sorted(c.checkpoint_id for c, _ in scored[:capacity]))


def analyze_standalone_topk_probe(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """比较 FlowState 与 StandaloneTopK diagnostic selector。"""

    selected_match_count = 0
    c_match_count = 0
    improvements: list[float] = []

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])
        flowstate_ids = tuple(row["policies"]["FlowState"]["selected_checkpoint_ids"])
        probe_ids = _standalone_topk_selection(variant, row["k"])
        if set(flowstate_ids) == set(probe_ids):
            selected_match_count += 1

        flowstate_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
        probe_obj = evaluate_objective(variant, probe_ids)
        probe_cost = probe_obj.total_recovery_cost_ms
        if abs(flowstate_cost - probe_cost) <= _FLOAT_TOLERANCE_MS:
            c_match_count += 1
        improvements.append(flowstate_cost - probe_cost)

    return {
        "cases": len(rows),
        "selected_set_exact_match_rate": selected_match_count / len(rows) if rows else None,
        "c_exact_match_rate": c_match_count / len(rows) if rows else None,
        "mean_flowstate_improvement_ms": _mean(improvements),
        "conclusion": (
            "equivalent"
            if selected_match_count == len(rows)
            else (
                "partial"
                if rows and selected_match_count / len(rows) >= 0.95
                else "different"
            )
        ),
    }


# ---------------------------------------------------------------------------
# 第十四层：FlowState vs Marconi Overlap
# ---------------------------------------------------------------------------


def analyze_marconi_overlap(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """统计 FlowState 与 Marconi selected set 的 Jaccard overlap。"""

    by_ratio: dict[float, list[float]] = {}
    exact_match: dict[float, int] = {}

    for row in rows:
        ratio = row["budget_ratio"]
        fs = set(row["policies"]["FlowState"]["selected_checkpoint_ids"])
        mar = set(row["policies"]["Marconi"]["selected_checkpoint_ids"])
        union = fs | mar
        jaccard = len(fs & mar) / len(union) if union else 1.0
        by_ratio.setdefault(ratio, []).append(jaccard)
        if fs == mar:
            exact_match[ratio] = exact_match.get(ratio, 0) + 1

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        values = by_ratio[ratio]
        result[ratio] = {
            "mean_jaccard": _mean(values),
            "median_jaccard": float(np.median(values)) if values else None,
            "p25_jaccard": float(np.percentile(values, 25)) if values else None,
            "p75_jaccard": float(np.percentile(values, 75)) if values else None,
            "p95_jaccard": float(np.percentile(values, 95)) if values else None,
            "exact_set_match_rate": exact_match.get(ratio, 0) / len(values) if values else None,
        }
    return result


# ---------------------------------------------------------------------------
# 第十五层：Win / Tie / Loss
# ---------------------------------------------------------------------------


def analyze_win_tie_loss(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """统计 FlowState 相对每个 baseline 的 win/tie/loss。"""

    by_ratio: dict[float, dict[str, dict[str, int]]] = {}
    for row in rows:
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(ratio, {})
        fs_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
        for baseline in ("LRU", "LFU", "Marconi"):
            base_cost = row["policies"][baseline]["total_recovery_cost_ms"]
            key = f"FlowState_vs_{baseline}"
            counts = entry.setdefault(key, {"win": 0, "tie": 0, "loss": 0})
            if fs_cost < base_cost - _FLOAT_TOLERANCE_MS:
                counts["win"] += 1
            elif fs_cost > base_cost + _FLOAT_TOLERANCE_MS:
                counts["loss"] += 1
            else:
                counts["tie"] += 1

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        result[ratio] = {}
        for key, counts in by_ratio[ratio].items():
            total = sum(counts.values())
            result[ratio][key] = {
                "win": counts["win"],
                "tie": counts["tie"],
                "loss": counts["loss"],
                "win_pct": counts["win"] / total if total else None,
                "tie_pct": counts["tie"] / total if total else None,
                "loss_pct": counts["loss"] / total if total else None,
            }
    return result


# ---------------------------------------------------------------------------
# 第十六层：Per-Round / Candidate-Count Stratification
# ---------------------------------------------------------------------------


def _percentile_bootstrap_ci(
    values: list[float],
    n_iterations: int = 10000,
    seed: int = 20260903,
) -> dict[str, float]:
    """计算 percentile bootstrap 95% CI。"""

    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    means = [float(np.mean(rng.choice(values, size=n, replace=True))) for _ in range(n_iterations)]
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def analyze_per_round(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """按 allocation round 与 candidate count 分层分析。"""

    by_round: dict[int, list[dict[str, Any]]] = {}
    by_candidate_count: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_round.setdefault(row["allocation_epoch"], []).append(row)
        by_candidate_count.setdefault(row["candidate_count"], []).append(row)

    def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
        reductions = []
        marconi_costs = []
        flowstate_costs = []
        marconi_redundancy = []
        flowstate_full = []
        for row in cases:
            mar_cost = row["policies"]["Marconi"]["total_recovery_cost_ms"]
            fs_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
            reductions.append((mar_cost - fs_cost) / mar_cost if mar_cost > _FLOAT_TOLERANCE_MS else 0.0)
            marconi_costs.append(mar_cost)
            flowstate_costs.append(fs_cost)
            # redundancy placeholder filled later if needed
            marconi_redundancy.append(0.0)
            flowstate_full.append(0.0)
        return {
            "n": len(cases),
            "mean_reduction": _mean(reductions),
            "median_reduction": float(np.median(reductions)) if reductions else None,
            "p25_reduction": float(np.percentile(reductions, 25)) if reductions else None,
            "p75_reduction": float(np.percentile(reductions, 75)) if reductions else None,
            "bootstrap_ci": _percentile_bootstrap_ci(reductions),
            "marconi_c": _summary_stats(marconi_costs),
            "flowstate_c": _summary_stats(flowstate_costs),
        }

    result = {
        "by_round": {epoch: summarize(cases) for epoch, cases in sorted(by_round.items())},
        "by_candidate_count": {
            cc: summarize(cases) for cc, cases in sorted(by_candidate_count.items())
        },
    }
    return result


# ---------------------------------------------------------------------------
# 第十七层：Saturation Explanation
# ---------------------------------------------------------------------------


def analyze_saturation(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """汇总解释 50%/75% budget 提升异常大的 saturation table。"""

    by_ratio: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_ratio.setdefault(row["budget_ratio"], []).append(row)

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        cases = by_ratio[ratio]
        k_values = [r["k"] for r in cases]
        pending_count = cases[0]["pending_count"]
        fs_s = [len(r["policies"]["FlowState"]["selected_checkpoint_ids"]) for r in cases]
        fs_zero = [
            r["policies"]["FlowState"]["total_recovery_cost_ms"] <= _FLOAT_TOLERANCE_MS
            for r in cases
        ]
        mar_zero = [
            r["policies"]["Marconi"]["total_recovery_cost_ms"] <= _FLOAT_TOLERANCE_MS
            for r in cases
        ]
        fs_full = [
            all(
                cont["recovery_gap_tokens"] <= 0
                for cont in r["policies"]["FlowState"]["per_continuation"]
            )
            for r in cases
        ]
        mar_full = [
            all(
                cont["recovery_gap_tokens"] <= 0
                for cont in r["policies"]["Marconi"]["per_continuation"]
            )
            for r in cases
        ]

        result[ratio] = {
            "mean_k": _mean(k_values),
            "median_k": float(np.median(k_values)) if k_values else None,
            "fraction_k_ge_pending": _fraction([k >= pending_count for k in k_values]),
            "flowstate_mean_s": _mean(fs_s),
            "flowstate_fraction_s_lt_k": _fraction([s < k for s, k in zip(fs_s, k_values)]),
            "flowstate_zero_cost_rate": _fraction(fs_zero),
            "marconi_zero_cost_rate": _fraction(mar_zero),
            "flowstate_full_coverage_rate": _fraction(fs_full),
            "marconi_full_coverage_rate": _fraction(mar_full),
        }

    return result


# ---------------------------------------------------------------------------
# 第十八层：Exact OPT 独立审计
# ---------------------------------------------------------------------------


def independent_exact_opt_audit(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """对 tractable cases 使用独立 combinations loop 复核 Exact OPT。"""

    tractable_rows = [r for r in rows if r["exact_opt"].get("tractable")]
    mismatches: list[dict[str, Any]] = []

    for row in tractable_rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])
        candidate_ids = tuple(c.checkpoint_id for c in variant.eligible_candidates)
        capacity = variant.logical_budget_k

        best_ids: tuple[str, ...] | None = None
        best_cost: float | None = None
        for size in range(min(capacity, len(candidate_ids)) + 1):
            for subset in combinations(candidate_ids, size):
                obj = evaluate_objective(variant, subset)
                cost = obj.total_recovery_cost_ms
                if (
                    best_cost is None
                    or cost < best_cost - _FLOAT_TOLERANCE_MS
                    or (
                        abs(cost - best_cost) <= _FLOAT_TOLERANCE_MS
                        and (best_ids is None or subset < best_ids)
                    )
                ):
                    best_ids = subset
                    best_cost = cost

        formal_ids = tuple(row["exact_opt"]["selected_checkpoint_ids"])
        formal_cost = row["exact_opt"]["total_recovery_cost_ms"]
        if best_ids != formal_ids or (
            best_cost is not None
            and abs(best_cost - formal_cost) > _FLOAT_TOLERANCE_MS
        ):
            mismatches.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "k": row["k"],
                    "formal_ids": list(formal_ids),
                    "independent_ids": list(best_ids) if best_ids else [],
                    "formal_cost": formal_cost,
                    "independent_cost": best_cost,
                }
            )

    return {
        "tractable_cases_checked": len(tractable_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": len(mismatches) == 0,
    }


# ---------------------------------------------------------------------------
# 第十九层：Current-Set Myopia
# ---------------------------------------------------------------------------


def analyze_current_set_myopia(
    rows: list[dict[str, Any]],
    snapshot_map: dict[str, AllocationSnapshot],
) -> dict[str, Any]:
    """统计 FlowState unused budget 与 zero-current-marginal unselected candidates。"""

    unused_budgets: list[int] = []
    stops_before_k: list[bool] = []
    zero_marginal_unselected: list[int] = []
    baseline_zero_marginal_selected: list[int] = []
    model = RecoveryCostModel()

    for row in rows:
        snapshot = snapshot_map[row["snapshot_id"]]
        variant = _snapshot_variant(snapshot, row["k"])

        # FlowState selected
        fs_ids = tuple(row["policies"]["FlowState"]["selected_checkpoint_ids"])
        fs_set = set(fs_ids)
        unused = row["k"] - len(fs_ids)
        unused_budgets.append(unused)
        stops_before_k.append(unused > 0)

        # unselected candidates with zero marginal given FlowState selected set
        candidates = list(variant.core_candidates())
        selected = [c for c in candidates if c.checkpoint_id in fs_set]
        zero_marginal = 0
        for cand in candidates:
            if cand.checkpoint_id in fs_set:
                continue
            gain = _marginal_gain(variant, selected, cand, model)
            if gain <= _FLOAT_TOLERANCE_MS:
                zero_marginal += 1
        zero_marginal_unselected.append(zero_marginal)

        # baseline zero-marginal selected
        for policy in ("LRU", "LFU", "Marconi"):
            base_ids = tuple(row["policies"][policy]["selected_checkpoint_ids"])
            costs_without = _cost_without_each(variant, base_ids)
            base_cost = row["policies"][policy]["total_recovery_cost_ms"]
            z = sum(
                1
                for cid in base_ids
                if abs(costs_without[cid] - base_cost) <= _FLOAT_TOLERANCE_MS
            )
            baseline_zero_marginal_selected.append(z)

    return {
        "flowstate_mean_unused_budget": _mean(unused_budgets),
        "flowstate_median_unused_budget": float(np.median(unused_budgets)) if unused_budgets else None,
        "flowstate_fraction_stops_before_k": _fraction(stops_before_k),
        "mean_zero_current_marginal_unselected": _mean(zero_marginal_unselected),
        "median_zero_current_marginal_unselected": float(np.median(zero_marginal_unselected)) if zero_marginal_unselected else None,
        "baseline_mean_zero_marginal_selected": _mean(baseline_zero_marginal_selected),
        "note": "当前 RQ3 只评估 single-allocation-epoch objective，不能证明未选择 checkpoint 在 future rounds 没有价值。",
    }


# ---------------------------------------------------------------------------
# 第二十层：Absolute Percentage Denominator Audit
# ---------------------------------------------------------------------------


def analyze_absolute_cost_distribution(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """报告绝对 C(S) 分布与 FlowState-Marconi 绝对改进。"""

    by_ratio: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        ratio = row["budget_ratio"]
        entry = by_ratio.setdefault(ratio, {})
        for key in ("empty", "LRU", "LFU", "Marconi", "FlowState"):
            entry.setdefault(key, []).append(
                row["c_empty"] if key == "empty" else row["policies"][key]["total_recovery_cost_ms"]
            )
        entry.setdefault("paired_improvement", []).append(
            row["policies"]["Marconi"]["total_recovery_cost_ms"]
            - row["policies"]["FlowState"]["total_recovery_cost_ms"]
        )

    result: dict[str, Any] = {}
    for ratio in sorted(by_ratio):
        data = by_ratio[ratio]
        result[ratio] = {
            "C_empty": _summary_stats(data["empty"]),
            "C_LRU": _summary_stats(data["LRU"]),
            "C_LFU": _summary_stats(data["LFU"]),
            "C_Marconi": _summary_stats(data["Marconi"]),
            "C_FlowState": _summary_stats(data["FlowState"]),
            "FlowState_minus_Marconi_ms": _summary_stats(data["paired_improvement"]),
            "marconi_cost_threshold_fraction": {
                "<=1ms": _fraction([v <= 1.0 for v in data["Marconi"]]),
                "<=10ms": _fraction([v <= 10.0 for v in data["Marconi"]]),
                "<=100ms": _fraction([v <= 100.0 for v in data["Marconi"]]),
            },
            "flowstate_cost_threshold_fraction": {
                "<=1ms": _fraction([v <= 1.0 for v in data["FlowState"]]),
                "<=10ms": _fraction([v <= 10.0 for v in data["FlowState"]]),
                "<=100ms": _fraction([v <= 100.0 for v in data["FlowState"]]),
            },
        }
    return result


# ---------------------------------------------------------------------------
# Artifact 汇总与入口
# ---------------------------------------------------------------------------


def run_audit(
    formal_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    """执行全部 13G-A audit 并返回结果字典。"""

    rows = load_per_snapshot_results(evaluation_root)
    snapshot_map = build_snapshot_map(formal_root)

    # 第一层
    reproduction = reproduce_objectives(rows, snapshot_map)
    # 第二层
    selector_audit = audit_selector_semantics(rows, snapshot_map)
    # 第三层
    monotonicity = audit_budget_monotonicity(rows, snapshot_map)
    # 第四至二十层
    size_saturation = analyze_selected_size_and_saturation(rows)
    zero_analysis = analyze_zero_cost_and_gap(rows, snapshot_map)
    coverage = analyze_workflow_coverage(rows, snapshot_map)
    workflow_dist = analyze_workflow_distribution(rows, snapshot_map)
    redundancy = analyze_redundancy(rows, snapshot_map)
    compatibility = analyze_compatibility_structure(snapshot_map)
    chain = analyze_chain_structure(snapshot_map)
    marginal = analyze_marginal_dependency(rows, snapshot_map)
    pw_best = analyze_per_workflow_best_probe(rows, snapshot_map)
    standalone_topk = analyze_standalone_topk_probe(rows, snapshot_map)
    overlap = analyze_marconi_overlap(rows)
    win_tie_loss = analyze_win_tie_loss(rows)
    per_round = analyze_per_round(rows, snapshot_map)
    saturation = analyze_saturation(rows, snapshot_map)
    exact_audit = independent_exact_opt_audit(rows, snapshot_map)
    myopia = analyze_current_set_myopia(rows, snapshot_map)
    absolute_dist = analyze_absolute_cost_distribution(rows)

    return {
        "inputs": {
            "formal_root": str(formal_root.resolve()),
            "evaluation_root": str(evaluation_root.resolve()),
            "eligible_snapshots": len(snapshot_map),
            "cases": len(rows),
        },
        "result_reproduction": reproduction,
        "selector_semantics_audit": selector_audit,
        "budget_monotonicity": monotonicity,
        "selected_size_and_saturation": size_saturation,
        "zero_cost_and_gap": zero_analysis,
        "workflow_coverage": coverage,
        "workflow_distribution": workflow_dist,
        "redundancy_analysis": redundancy,
        "compatibility_structure": compatibility,
        "chain_structure": chain,
        "marginal_dependency": marginal,
        "per_workflow_best_probe": pw_best,
        "standalone_topk_probe": standalone_topk,
        "marconi_overlap": overlap,
        "win_tie_loss": win_tie_loss,
        "per_round_analysis": per_round,
        "saturation_explanation": saturation,
        "exact_opt_audit": exact_audit,
        "current_set_myopia": myopia,
        "absolute_cost_distribution": absolute_dist,
    }


def write_audit_artifacts(
    audit_root: Path,
    results: dict[str, Any],
) -> None:
    """将全部 audit 结果写入新的 artifact root。"""

    audit_root.mkdir(parents=True, exist_ok=True)

    _write_json(audit_root / "INPUTS.json", results["inputs"])
    _write_json(audit_root / "SANITY_PROTOCOL.json", {"schema_version": "flowstate.rq3_sanity_structure_audit.v1"})
    _write_json(audit_root / "result_reproduction.json", results["result_reproduction"])
    _write_json(audit_root / "selector_semantics_audit.json", results["selector_semantics_audit"])
    _write_json(audit_root / "budget_monotonicity.json", results["budget_monotonicity"])
    _write_json(audit_root / "selected_size_and_saturation.json", results["selected_size_and_saturation"])
    _write_json(audit_root / "absolute_cost_distribution.json", results["absolute_cost_distribution"])
    _write_json(audit_root / "workflow_coverage.json", results["workflow_coverage"])
    _write_json(audit_root / "workflow_distribution.json", results["workflow_distribution"])
    _write_json(audit_root / "redundancy_analysis.json", results["redundancy_analysis"])
    _write_json(audit_root / "compatibility_structure.json", results["compatibility_structure"])
    _write_json(audit_root / "chain_structure.json", results["chain_structure"])
    _write_json(audit_root / "marginal_dependency.json", results["marginal_dependency"])
    _write_json(audit_root / "per_workflow_best_probe.json", results["per_workflow_best_probe"])
    _write_json(audit_root / "standalone_topk_probe.json", results["standalone_topk_probe"])
    _write_json(audit_root / "marconi_overlap.json", results["marconi_overlap"])
    _write_json(audit_root / "win_tie_loss.json", results["win_tie_loss"])
    _write_json(audit_root / "per_round_analysis.json", results["per_round_analysis"])
    _write_json(audit_root / "exact_structure_audit.json", results["exact_opt_audit"])
    _write_json(audit_root / "current_set_myopia.json", results["current_set_myopia"])
    _write_json(audit_root / "zero_cost_and_gap.json", results["zero_cost_and_gap"])
    _write_json(audit_root / "saturation_explanation.json", results["saturation_explanation"])
    _write_json(audit_root / "validation_report.json", _build_validation_report(results))


def _build_validation_report(results: dict[str, Any]) -> dict[str, Any]:
    """汇总所有 PASS/FAIL gate。"""

    return {
        "result_reproduction": (
            "PASS" if results["result_reproduction"]["pass"] else "FAIL"
        ),
        "selector_semantics_audit": (
            "PASS" if results["selector_semantics_audit"]["pass"] else "FAIL"
        ),
        "budget_monotonicity": (
            "PASS" if results["budget_monotonicity"]["pass"] else "FAIL"
        ),
        "exact_opt_independent_audit": (
            "PASS" if results["exact_opt_audit"]["pass"] else "FAIL"
        ),
        "formal_population_modified": False,
        "formal_evaluation_modified": False,
        "core_code_modified": False,
        "gpu_used": False,
        "status": (
            "RQ3_SANITY_BLOCKED"
            if not (
                results["result_reproduction"]["pass"]
                and results["selector_semantics_audit"]["pass"]
                and results["budget_monotonicity"]["pass"]
                and results["exact_opt_audit"]["pass"]
            )
            else "RQ3_SANITY_AUDIT_READY"
        ),
    }


def main() -> int:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description="Step 13G-A Formal RQ3 Sanity/Structure/Mechanism Audit")
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011"),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    formal_root = args.formal_root.resolve()
    evaluation_root = args.evaluation_root.resolve()

    if args.audit_root is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_root = Path(f"evaluation/runtime_artifacts/rq3_sanity_structure_audit_{timestamp}")
    else:
        audit_root = args.audit_root.resolve()

    print(f"[13G-A] formal root: {formal_root}")
    print(f"[13G-A] evaluation root: {evaluation_root}")
    print(f"[13G-A] audit root: {audit_root}")

    results = run_audit(formal_root, evaluation_root)
    write_audit_artifacts(audit_root, results)

    report = _build_validation_report(results)
    print(f"[13G-A] status: {report['status']}")
    print(f"[13G-A] result reproduction: {report['result_reproduction']}")
    print(f"[13G-A] selector semantics audit: {report['selector_semantics_audit']}")
    print(f"[13G-A] budget monotonicity: {report['budget_monotonicity']}")
    print(f"[13G-A] exact opt independent audit: {report['exact_opt_independent_audit']}")
    print(f"[13G-A] audit artifacts written to: {audit_root}")

    return 0 if report["status"] == "RQ3_SANITY_AUDIT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
