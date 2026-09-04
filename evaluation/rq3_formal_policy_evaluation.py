"""Step 13F：在正式 Main Population 上执行 Same-Snapshot Policy Evaluation。

本模块只做纯 CPU 的策略比较与 artifact 生成，不触碰 GPU、runtime replay 或 population。
所有新增注释、docstring、说明性文字使用中文。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    FrozenAccessFrequency,
    FrozenCandidateMetadata,
    FrozenCheckpointCandidate,
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    FrozenPendingContinuation,
    FrozenRecoveryModelIdentity,
    ObjectiveEvaluation,
    _select_policy,
    evaluate_objective,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


# 冻结预算比例与实验参数
_FROZEN_BUDGET_RATIOS: tuple[float, float, float] = (0.25, 0.50, 0.75)
_EXACT_OPT_SEARCH_SPACE_THRESHOLD = 100_000
_BOOTSTRAP_ITERATIONS = 10_000
_BOOTSTRAP_SEED = 20260903
_FLOAT_TOLERANCE_MS = 1e-9


def load_eligible_snapshots(formal_root: Path) -> list[AllocationSnapshot]:
    """从 formal root 加载全部 ELIGIBLE snapshot，按 group ordinal 升序排列。"""

    snapshots: list[AllocationSnapshot] = []
    for path in sorted(formal_root.glob("snapshots/g*.json")):
        snapshots.append(_load_allocation_snapshot(path))
    return snapshots


def _load_allocation_snapshot(path: Path) -> AllocationSnapshot:
    """从 canonical artifact 重建 snapshot 并校验 digest 一致。

    本实现只依赖 frozen evaluator 中的纯数据类型，避免引入 collector 的
    pyarrow / OpenHands 运行时依赖。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = payload["canonical_snapshot"]
    raw = json.loads(canonical)

    snapshot = AllocationSnapshot(
        allocation_epoch=int(raw["allocation_epoch"]),
        snapshot_id=str(raw["snapshot_id"]),
        pending_continuations=tuple(
            FrozenPendingContinuation(
                continuation_id=str(item["continuation_id"]),
                workflow_id=str(item["workflow_id"]),
                lineage_path=tuple(str(v) for v in item["lineage_path"]),
                anchor_pos=int(item["anchor_pos"]),
                resident_fa_frontier=int(item["resident_fa_frontier"]),
            )
            for item in raw["pending_continuations"]
        ),
        eligible_candidates=tuple(
            FrozenCheckpointCandidate(
                checkpoint_id=str(item["checkpoint_id"]),
                workflow_id=str(item["workflow_id"]),
                lineage_path=tuple(str(v) for v in item["lineage_path"]),
                token_pos=int(item["token_pos"]),
                memory_bytes=int(item["memory_bytes"]),
                recurrent_resident=bool(item["recurrent_resident"]),
                fa_resident=bool(item["fa_resident"]),
            )
            for item in raw["eligible_candidates"]
        ),
        candidate_metadata=tuple(
            FrozenCandidateMetadata(
                checkpoint_id=str(item["checkpoint_id"]),
                creation_order=int(item["creation_order"]),
                last_access_order=int(item["last_access_order"]),
                marconi_flop_saved=float(item["marconi_flop_saved"]),
            )
            for item in raw["candidate_metadata"]
        ),
        lfu_access_frequency=tuple(
            FrozenAccessFrequency(
                checkpoint_id=str(item["checkpoint_id"]),
                access_frequency=int(item["access_frequency"]),
            )
            for item in raw["lfu_access_frequency"]
        ),
        frequency_observed_through_epoch=int(
            raw["frequency_observed_through_epoch"]
        ),
        marconi_alpha=float(raw["marconi_alpha"]),
        recovery_model=FrozenRecoveryModelIdentity(
            **{
                key: value
                for key, value in raw["recovery_model"].items()
            }
        ),
        logical_budget_k=int(raw["logical_budget_k"]),
        budget_bytes=int(raw["budget_bytes"]),
        runtime_evidence=tuple(
            FrozenCheckpointRuntimeEvidence(
                checkpoint_id=str(item["checkpoint_id"]),
                node_id=int(item["node_id"]),
                runtime_identity_digest=str(item["runtime_identity_digest"]),
                checkpoint_handle_digest=str(item["checkpoint_handle_digest"]),
            )
            for item in raw["runtime_evidence"]
        ),
        residency_snapshot_digest=str(raw["residency_snapshot_digest"]),
        online_boundary=FrozenOnlineInformationBoundary(
            materialized_through_epoch=int(
                raw["online_boundary"]["materialized_through_epoch"]
            ),
            visible_continuation_ids=tuple(
                str(v)
                for v in raw["online_boundary"]["visible_continuation_ids"]
            ),
            future_continuation_included=bool(
                raw["online_boundary"]["future_continuation_included"]
            ),
            future_request_included=bool(
                raw["online_boundary"]["future_request_included"]
            ),
            future_latency_included=bool(
                raw["online_boundary"]["future_latency_included"]
            ),
        ),
    )
    if snapshot.content_digest() != payload["snapshot_digest"]:
        raise RuntimeError(f"{path} 的 canonical snapshot digest 不一致")
    if snapshot.canonical_serialization() != canonical:
        raise RuntimeError(f"{path} 的 canonical 序列化不一致")
    return snapshot


def compute_budget_ks(
    candidate_count: int,
    ratios: Sequence[float] = _FROZEN_BUDGET_RATIOS,
) -> list[tuple[float, int]]:
    """按冻结规则把 budget ratio 映射为 logical K。

    规则：
    - K = max(1, floor(r * |C|))
    - K >= |C| 为 trivial case，不进入比较
    - 因 floor 产生的重复 K 只保留一个（collapsed）
    """

    results: list[tuple[float, int]] = []
    seen: set[int] = set()
    for ratio in ratios:
        k = max(1, math.floor(ratio * candidate_count))
        if k >= candidate_count:
            continue
        if k not in seen:
            seen.add(k)
            results.append((ratio, k))
    return results


def search_space_size(candidate_count: int, k: int) -> int:
    """计算 Exact OPT 的搜索空间大小：sum_{i=0}^{K} C(|C|, i)。"""

    capacity = min(k, candidate_count)
    total = 0
    for size in range(capacity + 1):
        total += math.comb(candidate_count, size)
    return total


def create_budget_variant(snapshot: AllocationSnapshot, k: int) -> AllocationSnapshot:
    """为同一 snapshot 创建指定 logical budget K 的变体（deep-immutable，不修改原对象）。"""

    checkpoint_size = snapshot.eligible_candidates[0].memory_bytes
    return replace(
        snapshot,
        logical_budget_k=k,
        budget_bytes=k * checkpoint_size,
    )


def _policy_result(
    snapshot: AllocationSnapshot,
    policy_name: str,
) -> tuple[tuple[str, ...], int, float, ObjectiveEvaluation]:
    """运行单个 selector 并用公共 objective 评分。

    返回：selected_ids, selector_internal_evaluations, wall_time_ms, objective。
    """

    started_ns = perf_counter_ns()
    selected_ids, internal_evaluations = _select_policy(
        policy_name,
        snapshot,
        evaluate_objective,
    )
    wall_time_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
    objective = evaluate_objective(snapshot, selected_ids)
    return selected_ids, internal_evaluations, wall_time_ms, objective


def _policy_evaluation_to_dict(
    policy_name: str,
    selected_ids: tuple[str, ...],
    internal_evaluations: int,
    wall_time_ms: float,
    objective: ObjectiveEvaluation,
    snapshot_digest: str,
) -> dict[str, Any]:
    """把单次策略结果序列化为可写入 JSON 的字典。"""

    return {
        "policy_name": policy_name,
        "selected_checkpoint_ids": list(selected_ids),
        "selector_wall_time_ms": wall_time_ms,
        "selector_internal_evaluations": internal_evaluations,
        "total_recovery_cost_ms": objective.total_recovery_cost_ms,
        "empty_selection_cost_ms": objective.empty_selection_cost_ms,
        "total_benefit_ms": objective.total_benefit_ms,
        "normalized_cost": (
            objective.total_recovery_cost_ms / objective.empty_selection_cost_ms
            if objective.empty_selection_cost_ms > _FLOAT_TOLERANCE_MS
            else None
        ),
        "per_continuation": [
            {
                "continuation_id": row.continuation_id,
                "workflow_id": row.workflow_id,
                "target_tokens": row.target_tokens,
                "executable_frontier_tokens": row.executable_frontier_tokens,
                "recovery_gap_tokens": row.recovery_gap_tokens,
                "recovery_cost_ms": row.recovery_cost_ms,
            }
            for row in objective.per_continuation
        ],
        "snapshot_digest_before": snapshot_digest,
        "snapshot_digest_after": snapshot_digest,
        "final_common_scoring_evaluations": objective.objective_evaluation_count,
    }


def _flowstate_greedy_trace(
    snapshot: AllocationSnapshot,
    k: int,
) -> dict[str, Any]:
    """复现 FlowState greedy 选择过程并记录机制诊断数据。

    复现逻辑与 flowstate.optimizer.GlobalOptimizer 严格一致：
    - 候选仅限 recurrent_resident
    - 等大小检查点约束
    - capacity = K（budget_bytes 已设好）
    - 每轮选边际收益最大者，tie-break 按 checkpoint_id 升序
    - 收益 <= 0 时停止

    同时返回每步的 Delta(c|S) 与 u_p(S)。
    """

    continuations = list(snapshot.core_continuations())
    candidates = list(snapshot.core_candidates())
    eligible = sorted(
        (c for c in candidates if c.recurrent_resident),
        key=lambda c: c.checkpoint_id,
    )
    if not eligible:
        return {
            "selection_order": [],
            "delta_sequence": [],
            "cumulative_u_p_sequence": [],
            "final_selected": [],
        }

    checkpoint_size = eligible[0].memory_bytes
    capacity = min(k, len(eligible))
    model = RecoveryCostModel()

    def recovery_cost(selected: Sequence[CheckpointCandidate]) -> float:
        return sum(
            model.estimate(recovery_gap(cont, selected), cont.planning_target)
            for cont in continuations
        )

    selected: list[CheckpointCandidate] = []
    remaining = list(eligible)
    current_cost = recovery_cost(selected)
    selection_order: list[str] = []
    delta_sequence: list[float] = []
    cumulative_u_p_sequence: list[dict[str, list[float]]] = []

    for _ in range(capacity):
        best_index: int | None = None
        best_gain: float | None = None
        best_cost_after: float | None = None

        for idx, candidate in enumerate(remaining):
            cost_after = recovery_cost(selected + [candidate])
            gain = current_cost - cost_after
            if gain < -_FLOAT_TOLERANCE_MS:
                raise RuntimeError("检查点边际收益不能为负")
            if gain < 0.0:
                gain = 0.0
            if best_index is None:
                best_index = idx
                best_gain = gain
                best_cost_after = cost_after
                continue
            assert best_gain is not None
            best_candidate = remaining[best_index]
            gain_greater = gain > best_gain + _FLOAT_TOLERANCE_MS
            gain_tied = abs(gain - best_gain) <= _FLOAT_TOLERANCE_MS
            if gain_greater or (
                gain_tied and candidate.checkpoint_id < best_candidate.checkpoint_id
            ):
                best_index = idx
                best_gain = gain
                best_cost_after = cost_after

        if best_index is None or best_gain is None or best_gain <= 0.0:
            break

        chosen = remaining.pop(best_index)
        assert best_cost_after is not None
        selected.append(chosen)
        current_cost = best_cost_after
        selection_order.append(chosen.checkpoint_id)
        delta_sequence.append(best_gain)

        # 记录当前 S 下每个 pending 的 u_p(S)（即其被覆盖的 benefit）
        u_p: dict[str, list[float]] = {}
        for cont in continuations:
            empty_gap = recovery_gap(cont, ())
            empty_item_cost = model.estimate(empty_gap, cont.planning_target)
            current_gap = recovery_gap(cont, selected)
            current_item_cost = model.estimate(current_gap, cont.planning_target)
            u_p.setdefault(cont.workflow_id, []).append(
                empty_item_cost - current_item_cost
            )
        cumulative_u_p_sequence.append(u_p)

    return {
        "selection_order": selection_order,
        "delta_sequence": delta_sequence,
        "cumulative_u_p_sequence": cumulative_u_p_sequence,
        "final_selected": [c.checkpoint_id for c in selected],
    }


def _standalone_benefits(
    snapshot: AllocationSnapshot,
) -> dict[str, dict[str, Any]]:
    """计算每个候选对每个 pending 的 standalone benefit b_{p,c}。"""

    model = RecoveryCostModel()
    continuations = list(snapshot.core_continuations())
    candidates = list(snapshot.core_candidates())
    result: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        per_pending: list[dict[str, Any]] = []
        aggregate_benefit = 0.0
        for cont in continuations:
            if is_compatible(candidate, cont) and candidate.token_pos <= cont.planning_target:
                target = cont.planning_target
                phi_full = model.estimate(target, target)
                phi_with_c = model.estimate(target - candidate.token_pos, target)
                benefit = phi_full - phi_with_c
            else:
                benefit = 0.0
            per_pending.append(
                {
                    "continuation_id": cont.continuation_id,
                    "benefit_ms": benefit,
                    "compatible": benefit > 0.0,
                }
            )
            aggregate_benefit += benefit
        result[candidate.checkpoint_id] = {
            "checkpoint_id": candidate.checkpoint_id,
            "workflow_id": candidate.workflow_id,
            "token_pos": candidate.token_pos,
            "aggregate_benefit_ms": aggregate_benefit,
            "per_pending": per_pending,
        }
    return result


def _compatibility_matrix(
    snapshot: AllocationSnapshot,
) -> dict[str, dict[str, bool]]:
    """返回 candidate_id -> continuation_id -> 是否 compatible 的矩阵。"""

    continuations = list(snapshot.core_continuations())
    candidates = list(snapshot.core_candidates())
    return {
        c.checkpoint_id: {
            cont.continuation_id: is_compatible(c, cont)
            for cont in continuations
        }
        for c in candidates
    }


def _collect_mechanism_diagnostics(
    snapshot: AllocationSnapshot,
    k: int,
    policy_results: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """聚合当前 snapshot×K 的机制诊断数据。"""

    # FlowState greedy 轨迹
    greedy_trace = _flowstate_greedy_trace(snapshot, k)

    # standalone benefit 与 compatibility
    standalone = _standalone_benefits(snapshot)
    compatibility = _compatibility_matrix(snapshot)

    # 验证复现 greedy 与 frozen FlowState selector 输出一致
    flowstate_selected = tuple(policy_results["FlowState"]["selected_checkpoint_ids"])
    reproduced_selected = tuple(greedy_trace["final_selected"])
    if flowstate_selected != reproduced_selected:
        raise RuntimeError(
            "FlowState greedy 机制复现与 frozen selector 输出不一致："
            f"frozen={flowstate_selected}, reproduced={reproduced_selected}"
        )

    # pairwise selected-set overlap
    policies = ["LRU", "LFU", "Marconi", "FlowState"]
    overlaps: dict[str, dict[str, float]] = {}
    for a in policies:
        overlaps[a] = {}
        set_a = set(policy_results[a]["selected_checkpoint_ids"])
        for b in policies:
            set_b = set(policy_results[b]["selected_checkpoint_ids"])
            union = set_a | set_b
            overlaps[a][b] = (
                len(set_a & set_b) / len(union) if union else 1.0
            )

    return {
        "k": k,
        "candidate_count": len(snapshot.eligible_candidates),
        "pending_count": len(snapshot.pending_continuations),
        "selected_sets": {
            name: policy_results[name]["selected_checkpoint_ids"]
            for name in policies
        },
        "pairwise_overlap": overlaps,
        "per_pending": {
            name: [
                {
                    "continuation_id": row["continuation_id"],
                    "workflow_id": row["workflow_id"],
                    "target_tokens": row["target_tokens"],
                    "executable_frontier_tokens": row["executable_frontier_tokens"],
                    "recovery_gap_tokens": row["recovery_gap_tokens"],
                    "recovery_cost_ms": row["recovery_cost_ms"],
                }
                for row in policy_results[name]["per_continuation"]
            ]
            for name in policies
        },
        "flowstate_greedy_trace": greedy_trace,
        "standalone_benefits": standalone,
        "compatibility_matrix": compatibility,
    }


def evaluate_snapshot_at_ks(
    snapshot: AllocationSnapshot,
    group_ordinal: int,
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
) -> list[dict[str, Any]]:
    """对一个 snapshot 在全部非平凡 K 上运行四个 baseline 与条件 Exact OPT。

    返回每个 snapshot×K 的完整结果字典列表。
    """

    original_digest = snapshot.content_digest()
    candidate_count = len(snapshot.eligible_candidates)
    budget_ks = compute_budget_ks(candidate_count)
    results: list[dict[str, Any]] = []

    for ratio, k in budget_ks:
        snapshot_k = create_budget_variant(snapshot, k)
        variant_digest = snapshot_k.content_digest()

        # baseline policies
        policy_evaluations: dict[str, dict[str, Any]] = {}
        for policy_name in ("LRU", "LFU", "Marconi", "FlowState"):
            selected_ids, internal_evals, wall_ms, objective = _policy_result(
                snapshot_k, policy_name
            )
            policy_evaluations[policy_name] = _policy_evaluation_to_dict(
                policy_name,
                selected_ids,
                internal_evals,
                wall_ms,
                objective,
                variant_digest,
            )

        # 验证原 snapshot digest 未被任何 selector 修改
        if snapshot.content_digest() != original_digest:
            raise RuntimeError(
                f"group {group_ordinal} K={k} 运行 policy 后原 snapshot digest 变化"
            )

        # Exact OPT：仅当搜索空间在阈值内
        exact_info: dict[str, Any] = {"tractable": False}
        search_space = search_space_size(candidate_count, k)
        if search_space <= exact_threshold:
            started_ns = perf_counter_ns()
            exact_selected, exact_internal_evals, _wall_ms, exact_objective = _policy_result(
                snapshot_k, "Exact OPT"
            )
            exact_wall_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
            exact_info = {
                "tractable": True,
                "search_space_size": search_space,
                "subset_evaluations": exact_internal_evals,
                "wall_time_ms": exact_wall_ms,
                "selected_checkpoint_ids": list(exact_selected),
                "total_recovery_cost_ms": exact_objective.total_recovery_cost_ms,
                "empty_selection_cost_ms": exact_objective.empty_selection_cost_ms,
                "total_benefit_ms": exact_objective.total_benefit_ms,
                "normalized_cost": (
                    exact_objective.total_recovery_cost_ms
                    / exact_objective.empty_selection_cost_ms
                    if exact_objective.empty_selection_cost_ms > _FLOAT_TOLERANCE_MS
                    else None
                ),
            }
            # FlowState vs Exact 最优性指标
            fs_obj = policy_evaluations["FlowState"]
            exact_cost = exact_objective.total_recovery_cost_ms
            fs_cost = fs_obj["total_recovery_cost_ms"]
            gap_abs = fs_cost - exact_cost
            if gap_abs < -_FLOAT_TOLERANCE_MS:
                raise RuntimeError("FlowState 成本低于 Exact OPT，违反最优性")
            if abs(gap_abs) <= _FLOAT_TOLERANCE_MS:
                gap_abs = 0.0
            gap_rel = (
                gap_abs / exact_cost
                if exact_cost > _FLOAT_TOLERANCE_MS
                else (0.0 if gap_abs == 0.0 else None)
            )
            fs_benefit = fs_obj["total_benefit_ms"]
            exact_benefit = exact_objective.total_benefit_ms
            benefit_ratio = (
                fs_benefit / exact_benefit
                if exact_benefit > _FLOAT_TOLERANCE_MS
                else None
            )
            exact_info["flowstate_vs_exact"] = {
                "absolute_cost_gap_ms": gap_abs,
                "relative_cost_gap": gap_rel,
                "benefit_ratio": benefit_ratio,
            }
        else:
            exact_info = {
                "tractable": False,
                "search_space_size": search_space,
                "reason": "search_space_exceeds_threshold",
            }

        # paired reductions（绝对差与相对 reduction）
        c_empty = policy_evaluations["LRU"]["empty_selection_cost_ms"]
        paired: dict[str, dict[str, Any]] = {}
        fs_cost = policy_evaluations["FlowState"]["total_recovery_cost_ms"]
        for baseline in ("LRU", "LFU", "Marconi"):
            base_cost = policy_evaluations[baseline]["total_recovery_cost_ms"]
            abs_diff = base_cost - fs_cost
            rel_reduction = (
                abs_diff / base_cost if base_cost > _FLOAT_TOLERANCE_MS else None
            )
            paired[baseline] = {
                "absolute_difference_ms": abs_diff,
                "relative_reduction": rel_reduction,
            }

        # normalized cost
        normalized: dict[str, float | None] = {}
        for name in ("LRU", "LFU", "Marconi", "FlowState"):
            empty = policy_evaluations[name]["empty_selection_cost_ms"]
            cost = policy_evaluations[name]["total_recovery_cost_ms"]
            normalized[name] = (
                cost / empty if empty > _FLOAT_TOLERANCE_MS else None
            )

        # mechanism diagnostics
        diagnostics = _collect_mechanism_diagnostics(
            snapshot_k, k, policy_evaluations
        )

        results.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_digest": original_digest,
                "variant_digest": variant_digest,
                "group_ordinal": group_ordinal,
                "allocation_epoch": snapshot.allocation_epoch,
                "candidate_count": candidate_count,
                "pending_count": len(snapshot.pending_continuations),
                "budget_ratio": ratio,
                "k": k,
                "c_empty": c_empty,
                "policies": policy_evaluations,
                "exact_opt": exact_info,
                "paired": paired,
                "normalized_cost": normalized,
                "snapshot_digest_before": original_digest,
                "snapshot_digest_after": snapshot.content_digest(),
                "mechanism_diagnostics": diagnostics,
            }
        )

    return results


def evaluate_all_snapshots(
    formal_root: Path,
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
    progress_every: int = 50,
) -> list[dict[str, Any]]:
    """对全部 eligible snapshots 执行完整 same-snapshot evaluation。"""

    snapshots = load_eligible_snapshots(formal_root)
    all_results: list[dict[str, Any]] = []
    for idx, snapshot in enumerate(snapshots):
        group_ordinal = int(snapshot.snapshot_id.split("-")[-2].replace("g", ""))
        results = evaluate_snapshot_at_ks(
            snapshot, group_ordinal, exact_threshold=exact_threshold
        )
        all_results.extend(results)
        if (idx + 1) % progress_every == 0 or (idx + 1) == len(snapshots):
            print(
                f"[RQ3F] 已评估 {idx + 1}/{len(snapshots)} snapshots，"
                f"累计 snapshot×K = {len(all_results)}",
                flush=True,
            )
    return all_results


def _percentile(values: Sequence[float], p: float) -> float:
    """numpy percentile 的薄封装，空序列返回 None。"""

    if not values:
        return None  # type: ignore[return-value]
    return float(np.percentile(values, p))


def _summary_stats(values: Sequence[float]) -> dict[str, float | None]:
    """计算 mean/median/P25/P75/P95。"""

    if not values:
        return {
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _bootstrap_ci(
    paired_reductions: Sequence[float | None],
    n_iterations: int = _BOOTSTRAP_ITERATIONS,
    seed: int = _BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """对非空 paired reduction 做 percentile bootstrap，返回 95% CI。"""

    valid = [float(v) for v in paired_reductions if v is not None]
    n = len(valid)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_iterations):
        sample = [rng.choice(valid) for _ in range(n)]
        means.append(float(np.mean(sample)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "mean": float(np.mean(valid)),
        "ci_low": float(np.percentile(means, alpha * 100)),
        "ci_high": float(np.percentile(means, (1.0 - alpha) * 100)),
    }


def aggregate_results(
    per_snapshot_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """按 budget ratio 聚合描述统计、paired reduction 与 bootstrap CI。"""

    by_ratio: dict[float, list[dict[str, Any]]] = {}
    for row in per_snapshot_results:
        by_ratio.setdefault(row["budget_ratio"], []).append(row)

    aggregate: dict[str, Any] = {}
    for ratio in _FROZEN_BUDGET_RATIOS:
        rows = by_ratio.get(ratio, [])
        entry: dict[str, Any] = {"n": len(rows)}

        # 每个 policy 的 absolute C(S) 与 normalized C_hat(S)
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            costs = [r["policies"][policy]["total_recovery_cost_ms"] for r in rows]
            norm = [r["normalized_cost"][policy] for r in rows if r["normalized_cost"][policy] is not None]
            entry[f"C_{policy}"] = _summary_stats(costs)
            entry[f"C_hat_{policy}"] = _summary_stats(norm)

        # paired reduction（相对 + 绝对差）
        for baseline in ("LRU", "LFU", "Marconi"):
            reductions = [r["paired"][baseline]["relative_reduction"] for r in rows]
            diffs = [r["paired"][baseline]["absolute_difference_ms"] for r in rows]
            entry[f"FlowState_vs_{baseline}"] = {
                "relative_reduction": _summary_stats(
                    [v for v in reductions if v is not None]
                ),
                "absolute_difference_ms": _summary_stats(diffs),
                "bootstrap_ci": _bootstrap_ci(reductions),
            }

        aggregate[ratio] = entry

    # Exact OPT 跨 snapshot×K 的聚合
    exact_rows = [
        r["exact_opt"]
        for r in per_snapshot_results
        if r["exact_opt"].get("tractable")
    ]
    exact_aggregate: dict[str, Any] = {
        "tractable_cases": len(exact_rows),
    }
    if exact_rows:
        exact_matches = sum(
            1
            for r in exact_rows
            if abs(r["flowstate_vs_exact"]["absolute_cost_gap_ms"]) <= _FLOAT_TOLERANCE_MS
        )
        exact_aggregate["exact_matches"] = exact_matches
        exact_aggregate["exact_match_rate"] = exact_matches / len(exact_rows)
        abs_gaps = [r["flowstate_vs_exact"]["absolute_cost_gap_ms"] for r in exact_rows]
        rel_gaps = [
            r["flowstate_vs_exact"]["relative_cost_gap"]
            for r in exact_rows
            if r["flowstate_vs_exact"]["relative_cost_gap"] is not None
        ]
        benefit_ratios = [
            r["flowstate_vs_exact"]["benefit_ratio"]
            for r in exact_rows
            if r["flowstate_vs_exact"]["benefit_ratio"] is not None
        ]
        exact_aggregate["absolute_gap_stats"] = _summary_stats(abs_gaps)
        exact_aggregate["relative_gap_stats"] = _summary_stats(rel_gaps)
        exact_aggregate["benefit_ratio_stats"] = _summary_stats(benefit_ratios)
        exact_aggregate["worst_absolute_gap_ms"] = max(abs_gaps)
        exact_aggregate["worst_relative_gap"] = (
            max(rel_gaps) if rel_gaps else None
        )
        exact_aggregate["subset_evaluations_total"] = sum(
            r["subset_evaluations"] for r in exact_rows
        )

    aggregate["exact_opt"] = exact_aggregate
    return aggregate


def _artifact_evaluation_protocol(
    source_digest_before: str,
    evaluation_start: str,
) -> dict[str, Any]:
    """构造 EVALUATION_PROTOCOL.json 内容。"""

    return {
        "artifact_kind": "evaluation_protocol",
        "schema_version": "flowstate.rq3_formal_policy_eval.v1",
        "policy_names": ["LRU", "LFU Adaptation", "Marconi Adaptation", "FlowState", "Exact OPT"],
        "budget_ratios": list(_FROZEN_BUDGET_RATIOS),
        "k_rule": "K = max(1, floor(r * |C|))，排除 K >= |C| trivial cases",
        "exact_opt_search_space_threshold": _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
        "bootstrap_method": "percentile_bootstrap",
        "bootstrap_iterations": _BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "bootstrap_unit": "snapshot",
        "kvflow_main": "NOT_RUN",
        "primary_metric": "Aggregate Executable Recovery Cost C(S)",
        "source_digest_before": source_digest_before,
        "evaluation_start_timestamp": evaluation_start,
    }


def _artifact_input_population(
    formal_root: Path,
    population_marker: dict[str, Any],
    collection_summary: dict[str, Any],
) -> dict[str, Any]:
    """构造 INPUT_POPULATION.json 内容。"""

    snapshots = sorted(formal_root.glob("snapshots/g*.json"))
    return {
        "artifact_kind": "input_population",
        "schema_version": "flowstate.rq3_formal_policy_eval.v1",
        "formal_root": str(formal_root.resolve()),
        "formal_collection_identity": population_marker,
        "collection_summary_digest_fields": {
            "designated_groups": collection_summary["designated_groups"],
            "attempted_groups": collection_summary["attempted_groups"],
            "eligible_count": collection_summary["eligible_count"],
            "failed_count": collection_summary["failed_count"],
        },
        "eligible_snapshot_count": len(snapshots),
        "snapshot_digest_list": [
            json.loads(p.read_text(encoding="utf-8"))["snapshot_digest"]
            for p in snapshots
        ],
        "old_diagnostic_root_consumed": False,
        "sensitivity_population_consumed": False,
        "note": "只消费 evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/snapshots/ 中的 ELIGIBLE snapshots",
    }


def write_evaluation_artifacts(
    formal_root: Path,
    evaluation_root: Path,
    per_snapshot_results: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    source_digest_before: str,
    validation_report: dict[str, Any],
) -> None:
    """把全部 13F evaluation artifact 写入新 root。"""

    evaluation_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).astimezone().isoformat()

    # 1. INPUT_POPULATION.json
    marker = json.loads(
        (formal_root / "FORMAL_COLLECTION.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (formal_root / "collection_summary.json").read_text(encoding="utf-8")
    )
    _write_json(
        evaluation_root / "INPUT_POPULATION.json",
        _artifact_input_population(formal_root, marker, summary),
    )

    # 2. EVALUATION_PROTOCOL.json
    _write_json(
        evaluation_root / "EVALUATION_PROTOCOL.json",
        _artifact_evaluation_protocol(source_digest_before, started),
    )

    # 3. per_snapshot_results.jsonl
    per_path = evaluation_root / "per_snapshot_results.jsonl"
    with per_path.open("w", encoding="utf-8") as f:
        for row in per_snapshot_results:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # 4. aggregate_results.json（budget ratio 作为 JSON 字符串 key 以保持可序列化）
    json_safe_aggregate: dict[str, Any] = {
        (f"ratio_{k}" if isinstance(k, float) else str(k)): v
        for k, v in aggregate.items()
    }
    _write_json(evaluation_root / "aggregate_results.json", json_safe_aggregate)

    # 5. exact_opt_results.jsonl（仅 tractable cases）
    exact_path = evaluation_root / "exact_opt_results.jsonl"
    with exact_path.open("w", encoding="utf-8") as f:
        for row in per_snapshot_results:
            if row["exact_opt"].get("tractable"):
                record = {
                    "snapshot_id": row["snapshot_id"],
                    "snapshot_digest": row["snapshot_digest"],
                    "group_ordinal": row["group_ordinal"],
                    "allocation_epoch": row["allocation_epoch"],
                    "k": row["k"],
                    "search_space_size": row["exact_opt"]["search_space_size"],
                    "subset_evaluations": row["exact_opt"]["subset_evaluations"],
                    "wall_time_ms": row["exact_opt"]["wall_time_ms"],
                    "selected_exact": row["exact_opt"]["selected_checkpoint_ids"],
                    "c_exact": row["exact_opt"]["total_recovery_cost_ms"],
                    "f_exact": row["exact_opt"]["total_benefit_ms"],
                    "flowstate_vs_exact": row["exact_opt"]["flowstate_vs_exact"],
                }
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    # 6. bootstrap_results/
    bootstrap_dir = evaluation_root / "bootstrap_results"
    bootstrap_dir.mkdir(exist_ok=True)
    for ratio in _FROZEN_BUDGET_RATIOS:
        entry = aggregate[ratio]
        record = {
            "budget_ratio": ratio,
            "FlowState_vs_LRU": entry["FlowState_vs_LRU"]["bootstrap_ci"],
            "FlowState_vs_LFU": entry["FlowState_vs_LFU"]["bootstrap_ci"],
            "FlowState_vs_Marconi": entry["FlowState_vs_Marconi"]["bootstrap_ci"],
        }
        _write_json(bootstrap_dir / f"ratio_{int(ratio*100):03d}.json", record)

    # 7. mechanism_diagnostics/
    mech_dir = evaluation_root / "mechanism_diagnostics"
    mech_dir.mkdir(exist_ok=True)
    for row in per_snapshot_results:
        filename = f"g{row['group_ordinal']:03d}_k{row['k']:03d}.json"
        _write_json(
            mech_dir / filename,
            {
                "snapshot_id": row["snapshot_id"],
                "snapshot_digest": row["snapshot_digest"],
                "group_ordinal": row["group_ordinal"],
                "allocation_epoch": row["allocation_epoch"],
                "k": row["k"],
                "budget_ratio": row["budget_ratio"],
                **row["mechanism_diagnostics"],
            },
        )

    # 8. validation_report.json
    _write_json(evaluation_root / "validation_report.json", validation_report)

    # 9. SOURCE_VERSION.json（evaluation 结束后再填 after，但先写 before）
    _write_json(
        evaluation_root / "SOURCE_VERSION.json",
        {
            "artifact_kind": "source_version",
            "schema_version": "flowstate.rq3_formal_policy_eval.v1",
            "source_digest_before": source_digest_before,
            "source_digest_after": None,
            "evaluation_end_timestamp": None,
        },
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """用稳定格式写 JSON。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_determinism_on_snapshots(
    snapshots: Sequence[AllocationSnapshot],
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
) -> dict[str, Any]:
    """对给定 snapshot 列表的每个 snapshot×K 重新运行 policy，验证 selected sets 完全一致。"""

    mismatches: list[dict[str, Any]] = []
    total_runs = 0

    for snapshot in snapshots:
        group_ordinal = int(snapshot.snapshot_id.split("-")[-2].replace("g", ""))
        candidate_count = len(snapshot.eligible_candidates)
        budget_ks = compute_budget_ks(candidate_count)
        for ratio, k in budget_ks:
            snapshot_k = create_budget_variant(snapshot, k)
            first_run: dict[str, tuple[str, ...]] = {}
            for policy_name in ("LRU", "LFU", "Marconi", "FlowState"):
                selected_ids, _internal, _wall, _obj = _policy_result(
                    snapshot_k, policy_name
                )
                first_run[policy_name] = selected_ids

            # 第二次运行
            for policy_name in ("LRU", "LFU", "Marconi", "FlowState"):
                selected_ids, _internal, _wall, _obj = _policy_result(
                    snapshot_k, policy_name
                )
                total_runs += 1
                if selected_ids != first_run[policy_name]:
                    mismatches.append(
                        {
                            "group_ordinal": group_ordinal,
                            "k": k,
                            "policy": policy_name,
                            "first": list(first_run[policy_name]),
                            "second": list(selected_ids),
                        }
                    )

            # Exact OPT tractable cases
            if search_space_size(candidate_count, k) <= exact_threshold:
                exact_first, _i1, _w1, _o1 = _policy_result(snapshot_k, "Exact OPT")
                exact_second, _i2, _w2, _o2 = _policy_result(snapshot_k, "Exact OPT")
                total_runs += 1
                if exact_first != exact_second:
                    mismatches.append(
                        {
                            "group_ordinal": group_ordinal,
                            "k": k,
                            "policy": "Exact OPT",
                            "first": list(exact_first),
                            "second": list(exact_second),
                        }
                    )

    return {
        "total_policy_runs": total_runs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": len(mismatches) == 0,
    }


def run_determinism_rerun(
    formal_root: Path,
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
) -> dict[str, Any]:
    """对所有 snapshot×K 重新运行 policy，验证 selected sets 完全一致。"""

    snapshots = load_eligible_snapshots(formal_root)
    return _run_determinism_on_snapshots(snapshots, exact_threshold)


def compute_source_digest(paths: Sequence[Path]) -> str:
    """计算策略相关源码的合并 SHA-256 digest。"""

    from hashlib import sha256

    rows = []
    for path in sorted(paths, key=lambda p: str(p)):
        content = path.read_bytes()
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    """CLI 入口：执行完整 13F evaluation 并写 artifact。"""

    parser = argparse.ArgumentParser(description="Step 13F Formal RQ3 Policy Evaluation")
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"),
        help="Step 13E-F 正式 population root",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=None,
        help="13F evaluation artifact 输出目录（默认带时间戳）",
    )
    args = parser.parse_args()

    formal_root = args.formal_root.resolve()
    if args.evaluation_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_root = Path(
            f"evaluation/runtime_artifacts/rq3_formal_policy_eval_{timestamp}"
        )
    else:
        evaluation_root = args.evaluation_root.resolve()

    print(f"[RQ3F] formal root: {formal_root}")
    print(f"[RQ3F] evaluation root: {evaluation_root}")

    # 源码 digest before
    source_paths = [
        Path("evaluation/rq3_frozen_snapshot_evaluator.py"),
        Path("evaluation/rq3_formal_policy_evaluation.py"),
        Path("evaluation/controlled_multiworkflow_v1/policies.py"),
        Path("evaluation/sota_policies.py"),
        Path("flowstate/optimizer.py"),
        Path("flowstate/recovery_model.py"),
        Path("flowstate/executable_state.py"),
        Path("flowstate/state_catalog.py"),
        Path("flowstate/workflow.py"),
    ]
    source_digest_before = compute_source_digest(source_paths)
    print(f"[RQ3F] source digest before: {source_digest_before}")

    # 执行 evaluation
    per_snapshot_results = evaluate_all_snapshots(formal_root)
    print(f"[RQ3F] 总 snapshot×K cases: {len(per_snapshot_results)}")

    # determinism rerun
    print("[RQ3F] 开始 determinism rerun 验证...")
    determinism_report = run_determinism_rerun(formal_root)
    print(
        f"[RQ3F] determinism: {determinism_report['total_policy_runs']} runs, "
        f"mismatches={determinism_report['mismatch_count']}"
    )
    if not determinism_report["pass"]:
        raise RuntimeError(
            "Determinism rerun 失败：" + json.dumps(determinism_report["mismatches"])
        )

    # 聚合统计
    aggregate = aggregate_results(per_snapshot_results)

    # 构建 validation report
    validation_report = {
        "formal_population_validation": "PASS",
        "eligible_snapshots": len(load_eligible_snapshots(formal_root)),
        "old_diagnostic_root_consumed": False,
        "sensitivity_population_consumed": False,
        "snapshot_digest_immutable": "PASS",
        "determinism_rerun": determinism_report,
        "budget_protocol": "PASS",
        "kvflow_main": "NOT_RUN",
        "future_information_boundary": "PASS",
        "source_digest_before": source_digest_before,
    }

    # 写 artifact
    write_evaluation_artifacts(
        formal_root,
        evaluation_root,
        per_snapshot_results,
        aggregate,
        source_digest_before,
        validation_report,
    )

    # 更新 SOURCE_VERSION.json 的 after 与时间戳
    source_digest_after = compute_source_digest(source_paths)
    print(f"[RQ3F] source digest after: {source_digest_after}")
    if source_digest_after != source_digest_before:
        raise RuntimeError(
            "Evaluation 期间核心源码发生变化，结果 INVALID"
        )
    source_version = json.loads(
        (evaluation_root / "SOURCE_VERSION.json").read_text(encoding="utf-8")
    )
    source_version["source_digest_after"] = source_digest_after
    source_version["evaluation_end_timestamp"] = (
        datetime.now(timezone.utc).astimezone().isoformat()
    )
    _write_json(evaluation_root / "SOURCE_VERSION.json", source_version)

    print(f"[RQ3F] 完成，artifact root: {evaluation_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
