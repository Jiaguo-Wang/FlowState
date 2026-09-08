"""RQ5-A：在冻结 allocation snapshot 上执行设计消融。

本模块只进行 CPU 计算。它复用 RQ3 的快照、预算协议、兼容语义、正式
FlowState optimizer 与公共 M2 恢复成本目标，不读取未来轨迹，也不修改输入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.recovery_model_freeze import predict_model
from evaluation.rq3_agentx_formal_policy_evaluation import (
    DEFAULT_B1_ROOT,
    DEFAULT_B2_ROOT,
    FROZEN_AGENTX_PATH,
    build_agentx_formal_snapshots,
)
from evaluation.rq3_formal_policy_evaluation import (
    _BOOTSTRAP_ITERATIONS,
    _BOOTSTRAP_SEED,
    _FLOAT_TOLERANCE_MS,
    _FROZEN_BUDGET_RATIOS,
    _bootstrap_ci,
    _summary_stats,
    compute_budget_ks,
    create_budget_variant,
    load_eligible_snapshots,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    evaluate_objective,
)
from evaluation.rq3_sanity_structure_audit import _standalone_topk_selection
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENHANDS_ROOT = REPOSITORY_ROOT / (
    "evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"
)
DEFAULT_OPENHANDS_RESULTS = REPOSITORY_ROOT / (
    "evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011"
)
DEFAULT_AGENTX_RESULTS = REPOSITORY_ROOT / "rq3_agentx_policy_eval_20260906_214956"
DEFAULT_GAP_MODEL_ROOT = REPOSITORY_ROOT / (
    "evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020"
)

EXPECTED_OPENHANDS_SNAPSHOTS = 168
EXPECTED_AGENTX_SNAPSHOTS = 23
VARIANTS = ("full", "without_position_aware", "without_set_dependent")


@dataclass(frozen=True)
class FrozenGapOnlyCostModel:
    """封装冻结 M0 gap-only 插值模型，接口与正式 optimizer 兼容。"""

    model: Mapping[str, object]

    def estimate(self, gap_tokens: int, target_tokens: int) -> float:
        """返回冻结 M0 的恢复成本，不使用绝对位置 T 参与数值计算。"""
        return predict_model("M0", self.model, target_tokens, gap_tokens)


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    """以稳定格式写入 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """以稳定格式写入 JSONL。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    """计算可序列化对象的规范摘要。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_gap_only_model(
    model_root: Path = DEFAULT_GAP_MODEL_ROOT,
) -> tuple[FrozenGapOnlyCostModel, dict[str, Any]]:
    """加载冻结 M0，并验证它确为 held-out 前冻结的 gap-only 候选。"""
    candidates_path = model_root / "candidate_models.json"
    selection_path = model_root / "model_selection.json"
    candidates = _read_json(candidates_path)
    selection = _read_json(selection_path)
    model = candidates["M0"]
    before_hash = selection["candidate_models_hash_before_heldout"]
    after_hash = selection["candidate_models_hash_after_heldout"]
    if before_hash != after_hash:
        raise RuntimeError("M0 参数在 held-out 评估后发生变化")
    if candidates.get("heldout_used_for_fit") is not False:
        raise RuntimeError("M0 冻结记录未明确排除 held-out 拟合")
    if "gap-only" not in str(model["formula"]):
        raise RuntimeError("M0 不是冻结的 gap-only 候选")
    evidence = {
        "model_name": "M0",
        "formula": model["formula"],
        "parameters": model["parameters"],
        "heldout_used_for_fit": False,
        "candidate_models_sha256": _sha256_file(candidates_path),
        "model_selection_sha256": _sha256_file(selection_path),
        "candidate_models_hash_before_heldout": before_hash,
        "candidate_models_hash_after_heldout": after_hash,
        "frozen_model_status": selection["model_statuses"]["M0"],
        "note": "M0 是冻结消融模型；其 held-out 质量门禁为失败，不会替代正式 M2。",
    }
    return FrozenGapOnlyCostModel(model), evidence


def _model_cost(
    snapshot: AllocationSnapshot,
    selected_ids: Sequence[str],
    model: object,
) -> float:
    """按指定模型计算 C(S)，仅供 selector 内部模型与诊断使用。"""
    selected_set = set(selected_ids)
    candidate_by_id = {
        item.checkpoint_id: item for item in snapshot.core_candidates()
    }
    if not selected_set <= set(candidate_by_id):
        raise ValueError("选择集合含未知候选")
    selected = tuple(candidate_by_id[value] for value in sorted(selected_set))
    return sum(
        model.estimate(recovery_gap(pending, selected), pending.planning_target)
        for pending in snapshot.core_continuations()
    )


def _select_greedy(
    snapshot: AllocationSnapshot,
    model: object,
) -> tuple[tuple[str, ...], float]:
    """调用冻结 GlobalOptimizer，并保留确定性的贪心顺序。"""
    result = GlobalOptimizer(model).select(
        snapshot.core_continuations(),
        snapshot.core_candidates(),
        snapshot.budget_bytes,
    )
    return (
        tuple(item.checkpoint_id for item in result.selected),
        result.recovery_cost_after_ms,
    )


def _standalone_ranking(
    snapshot: AllocationSnapshot,
    model: object,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """按 empty-set 独立收益构造固定排序，并采用冻结字典序兜底。"""
    empty_cost = _model_cost(snapshot, (), model)
    benefits = {
        candidate.checkpoint_id: empty_cost
        - _model_cost(snapshot, (candidate.checkpoint_id,), model)
        for candidate in snapshot.core_candidates()
    }
    ranking = tuple(
        identifier
        for identifier, value in sorted(
            benefits.items(), key=lambda item: (-item[1], item[0])
        )
        if value > _FLOAT_TOLERANCE_MS
    )
    return ranking, benefits


def _select_standalone_topk(
    snapshot: AllocationSnapshot,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, float]]:
    """复用冻结 StandaloneTopK 结果，并另外保留其固定排序顺序。"""
    selected = _standalone_topk_selection(snapshot, snapshot.logical_budget_k)
    ranking, benefits = _standalone_ranking(snapshot, RecoveryCostModel())
    order = ranking[: min(snapshot.logical_budget_k, len(ranking))]
    if set(order) != set(selected):
        raise RuntimeError("StandaloneTopK 固定实现与独立排序不一致")
    return tuple(sorted(selected)), order, benefits


def _selection_diagnostics(
    snapshot: AllocationSnapshot,
    selection_order: Sequence[str],
) -> dict[str, Any]:
    """在公共 M2 目标下统计逐步零边际与最终可删除冗余。"""
    selected: list[str] = []
    gains: list[dict[str, Any]] = []
    zero_current = 0
    current_cost = evaluate_objective(snapshot, ()).total_recovery_cost_ms
    for identifier in selection_order:
        after = evaluate_objective(
            snapshot, tuple(selected + [identifier])
        ).total_recovery_cost_ms
        gain = current_cost - after
        if gain < -_FLOAT_TOLERANCE_MS:
            raise RuntimeError("公共目标中出现负边际收益")
        is_zero = gain <= _FLOAT_TOLERANCE_MS
        zero_current += int(is_zero)
        gains.append(
            {
                "checkpoint_id": identifier,
                "common_marginal_ms": max(0.0, gain),
                "zero_current_marginal": is_zero,
            }
        )
        selected.append(identifier)
        current_cost = after

    full_cost = evaluate_objective(snapshot, selected).total_recovery_cost_ms
    removable_zero = 0
    for identifier in selected:
        remaining = tuple(value for value in selected if value != identifier)
        without_cost = evaluate_objective(snapshot, remaining).total_recovery_cost_ms
        if abs(without_cost - full_cost) <= _FLOAT_TOLERANCE_MS:
            removable_zero += 1
    return {
        "selected_count": len(selected),
        "zero_current_marginal_count": zero_current,
        "zero_current_marginal_rate": (
            zero_current / len(selected) if selected else 0.0
        ),
        "final_removable_zero_count": removable_zero,
        "final_removable_zero_rate": (
            removable_zero / len(selected) if selected else 0.0
        ),
        "selection_steps": gains,
    }


def _compatible_pending_ids(
    snapshot: AllocationSnapshot,
) -> dict[str, set[str]]:
    """从冻结 snapshot 的显式 workflow lineage 计算兼容 pending 集合。"""
    return {
        candidate.checkpoint_id: {
            pending.continuation_id
            for pending in snapshot.core_continuations()
            if is_compatible(candidate, pending)
        }
        for candidate in snapshot.core_candidates()
    }


def _marginal_structure(
    snapshot: AllocationSnapshot,
    full_order: Sequence[str],
) -> dict[str, Any]:
    """把 set-dependent 边际变化分成单 pending 与跨 pending 重叠。"""
    model = RecoveryCostModel()
    candidates = {
        item.checkpoint_id: item for item in snapshot.core_candidates()
    }
    compatibility = _compatible_pending_ids(snapshot)
    selected: list[CheckpointCandidate] = []
    changed_events = 0
    same_pending_events = 0
    cross_pending_events = 0
    other_events = 0
    comparisons = 0
    for chosen_id in full_order:
        already = {item.checkpoint_id for item in selected}
        remaining = [
            item
            for identifier, item in candidates.items()
            if identifier not in already and identifier != chosen_id
        ]
        selected_coverage = set().union(
            *(compatibility[item.checkpoint_id] for item in selected)
        ) if selected else set()
        for candidate in remaining:
            comparisons += 1
            empty_gain = _model_cost(
                snapshot, (), model
            ) - _model_cost(snapshot, (candidate.checkpoint_id,), model)
            selected_ids = tuple(item.checkpoint_id for item in selected)
            current_gain = _model_cost(
                snapshot, selected_ids, model
            ) - _model_cost(
                snapshot, selected_ids + (candidate.checkpoint_id,), model
            )
            if abs(empty_gain - current_gain) <= _FLOAT_TOLERANCE_MS:
                continue
            changed_events += 1
            overlap_count = len(
                selected_coverage & compatibility[candidate.checkpoint_id]
            )
            if overlap_count == 1:
                same_pending_events += 1
            elif overlap_count > 1:
                cross_pending_events += 1
            else:
                other_events += 1
        selected.append(candidates[chosen_id])
    return {
        "candidate_stage_comparisons": comparisons,
        "marginal_changed_events": changed_events,
        "same_pending_redundancy_events": same_pending_events,
        "cross_pending_overlap_events": cross_pending_events,
        "other_events": other_events,
    }


def _load_reference_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """按 snapshot 与 K 索引冻结 RQ3 Full FlowState 结果。"""
    rows = _read_jsonl(path)
    return {(row["snapshot_id"], int(row["k"])): row for row in rows}


def _evaluate_case(
    population: str,
    snapshot: AllocationSnapshot,
    ratio: float,
    k: int,
    gap_model: FrozenGapOnlyCostModel,
    reference: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """运行单个 snapshot×budget 消融，并执行确定性与 RQ3 复现门禁。"""
    variant = create_budget_variant(snapshot, k)
    full_order, full_native_cost = _select_greedy(variant, RecoveryCostModel())
    gap_order, gap_native_cost = _select_greedy(variant, gap_model)
    standalone_ids, standalone_order, standalone_benefits = (
        _select_standalone_topk(variant)
    )

    repeat_full, repeat_full_cost = _select_greedy(
        variant, RecoveryCostModel()
    )
    repeat_gap, repeat_gap_cost = _select_greedy(variant, gap_model)
    repeat_standalone, repeat_standalone_order, _ = _select_standalone_topk(
        variant
    )
    deterministic = (
        full_order == repeat_full
        and abs(full_native_cost - repeat_full_cost) <= _FLOAT_TOLERANCE_MS
        and gap_order == repeat_gap
        and abs(gap_native_cost - repeat_gap_cost) <= _FLOAT_TOLERANCE_MS
        and standalone_ids == repeat_standalone
        and standalone_order == repeat_standalone_order
    )
    if not deterministic:
        raise RuntimeError(f"{snapshot.snapshot_id} K={k} selector 不确定")

    reference_row = reference[(snapshot.snapshot_id, k)]
    reference_full = reference_row["policies"]["FlowState"]
    reference_ids = tuple(reference_full["selected_checkpoint_ids"])
    full_objective = evaluate_objective(variant, full_order)
    if (
        full_order != reference_ids
        or abs(
            full_objective.total_recovery_cost_ms
            - float(reference_full["total_recovery_cost_ms"])
        )
        > _FLOAT_TOLERANCE_MS
    ):
        raise RuntimeError(
            f"{snapshot.snapshot_id} K={k} 未复现冻结 RQ3 Full FlowState"
        )

    selections = {
        "full": (tuple(sorted(full_order)), full_order),
        "without_position_aware": (tuple(sorted(gap_order)), gap_order),
        "without_set_dependent": (standalone_ids, standalone_order),
    }
    variants: dict[str, Any] = {}
    empty_cost = evaluate_objective(variant, ()).total_recovery_cost_ms
    for name, (selected_ids, order) in selections.items():
        objective = evaluate_objective(variant, selected_ids)
        diagnostics = _selection_diagnostics(variant, order)
        variants[name] = {
            "selected_checkpoint_ids": list(selected_ids),
            "selection_order": list(order),
            "common_cost_ms": objective.total_recovery_cost_ms,
            "common_empty_cost_ms": objective.empty_selection_cost_ms,
            "common_normalized_cost": (
                objective.total_recovery_cost_ms / empty_cost
                if empty_cost > _FLOAT_TOLERANCE_MS
                else None
            ),
            "full_coverage": (
                objective.total_recovery_cost_ms <= _FLOAT_TOLERANCE_MS
            ),
            **diagnostics,
        }
    variants["full"]["selector_native_cost_ms"] = full_native_cost
    variants["without_position_aware"][
        "selector_native_gap_only_cost_ms"
    ] = gap_native_cost
    variants["without_set_dependent"][
        "empty_set_standalone_benefit_ms"
    ] = {
        identifier: standalone_benefits[identifier]
        for identifier in standalone_order
    }

    comparisons: dict[str, Any] = {}
    full_set = set(full_order)
    full_cost = full_objective.total_recovery_cost_ms
    for ablation in ("without_position_aware", "without_set_dependent"):
        ablation_set = set(variants[ablation]["selected_checkpoint_ids"])
        ablation_cost = float(variants[ablation]["common_cost_ms"])
        absolute = ablation_cost - full_cost
        relative = (
            absolute / ablation_cost
            if ablation_cost > _FLOAT_TOLERANCE_MS
            else (0.0 if abs(absolute) <= _FLOAT_TOLERANCE_MS else None)
        )
        comparisons[ablation] = {
            "selected_set_different": full_set != ablation_set,
            "selected_set_symmetric_difference_count": len(
                full_set ^ ablation_set
            ),
            "selection_order_different": tuple(full_order)
            != tuple(variants[ablation]["selection_order"]),
            "ablation_minus_full_cost_ms": absolute,
            "full_relative_reduction": relative,
            "outcome": (
                "win"
                if absolute > _FLOAT_TOLERANCE_MS
                else "loss"
                if absolute < -_FLOAT_TOLERANCE_MS
                else "tie"
            ),
        }

    full_static_ranking, _ = _standalone_ranking(
        variant, RecoveryCostModel()
    )
    gap_static_ranking, _ = _standalone_ranking(variant, gap_model)
    structure = _marginal_structure(variant, full_order)
    row = {
        "population": population,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_digest": snapshot.content_digest(),
        "variant_digest": variant.content_digest(),
        "budget_ratio": ratio,
        "k": k,
        "candidate_count": len(snapshot.eligible_candidates),
        "pending_count": len(snapshot.pending_continuations),
        "variants": variants,
        "comparisons": comparisons,
        "position_ranking": {
            "static_positive_ranking_different": (
                full_static_ranking != gap_static_ranking
            ),
            "static_top_candidate_different": (
                bool(full_static_ranking)
                and bool(gap_static_ranking)
                and full_static_ranking[0] != gap_static_ranking[0]
            ),
            "full_first_candidate": (
                full_order[0] if full_order else None
            ),
            "gap_only_first_candidate": (
                gap_order[0] if gap_order else None
            ),
        },
        "marginal_structure": structure,
        "rq3_full_reproduced": True,
        "deterministic": True,
    }
    gate = {
        "population": population,
        "snapshot_id": snapshot.snapshot_id,
        "k": k,
        "deterministic": deterministic,
        "rq3_full_reproduced": True,
    }
    return row, gate


def _aggregate_comparison(
    rows: Sequence[Mapping[str, Any]],
    ablation: str,
) -> dict[str, Any]:
    """聚合 Full 相对指定消融的 paired 指标。"""
    if not rows:
        empty_summary = _summary_stats([])
        empty_ci = _bootstrap_ci([])
        return {
            "mean_absolute_reduction_ms": None,
            "absolute_reduction_ms": empty_summary,
            "absolute_reduction_bootstrap_95_ci": empty_ci,
            "mean_relative_reduction": None,
            "relative_reduction": empty_summary,
            "relative_reduction_bootstrap_95_ci": empty_ci,
            "win_tie_loss": {"win": 0, "tie": 0, "loss": 0},
            "selected_set_difference_count": 0,
            "selected_set_difference_rate": None,
            "selection_order_difference_rate": None,
        }
    absolute = [
        float(row["comparisons"][ablation]["ablation_minus_full_cost_ms"])
        for row in rows
    ]
    relative = [
        row["comparisons"][ablation]["full_relative_reduction"]
        for row in rows
    ]
    outcomes = [row["comparisons"][ablation]["outcome"] for row in rows]
    different = [
        bool(row["comparisons"][ablation]["selected_set_different"])
        for row in rows
    ]
    return {
        "mean_absolute_reduction_ms": statistics.fmean(absolute),
        "absolute_reduction_ms": _summary_stats(absolute),
        "absolute_reduction_bootstrap_95_ci": _bootstrap_ci(absolute),
        "mean_relative_reduction": statistics.fmean(
            value for value in relative if value is not None
        ) if any(value is not None for value in relative) else None,
        "relative_reduction": _summary_stats(
            [float(value) for value in relative if value is not None]
        ),
        "relative_reduction_bootstrap_95_ci": _bootstrap_ci(relative),
        "win_tie_loss": {
            "win": outcomes.count("win"),
            "tie": outcomes.count("tie"),
            "loss": outcomes.count("loss"),
        },
        "selected_set_difference_count": sum(different),
        "selected_set_difference_rate": (
            sum(different) / len(different) if different else None
        ),
        "selection_order_difference_rate": sum(
            bool(row["comparisons"][ablation]["selection_order_different"])
            for row in rows
        ) / len(rows) if rows else None,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按 population 与预算汇总主指标和机制指标。"""
    result: dict[str, Any] = {}
    for population in ("OpenHands", "AgentX"):
        population_rows = [row for row in rows if row["population"] == population]
        budgets: dict[str, Any] = {}
        for ratio in _FROZEN_BUDGET_RATIOS:
            ratio_rows = [
                row for row in population_rows if row["budget_ratio"] == ratio
            ]
            variants: dict[str, Any] = {}
            for variant in VARIANTS:
                costs = [
                    float(row["variants"][variant]["common_cost_ms"])
                    for row in ratio_rows
                ]
                zero_count = sum(
                    int(row["variants"][variant]["zero_current_marginal_count"])
                    for row in ratio_rows
                )
                selected_count = sum(
                    int(row["variants"][variant]["selected_count"])
                    for row in ratio_rows
                )
                coverage_count = sum(
                    bool(row["variants"][variant]["full_coverage"])
                    for row in ratio_rows
                )
                variants[variant] = {
                    "common_cost_ms": _summary_stats(costs),
                    "zero_current_marginal_selection_count": zero_count,
                    "selected_count": selected_count,
                    "zero_current_marginal_selection_rate": (
                        zero_count / selected_count if selected_count else 0.0
                    ),
                    "full_coverage_count": coverage_count,
                    "full_coverage_ratio": (
                        coverage_count / len(ratio_rows) if ratio_rows else None
                    ),
                }
            structure_keys = (
                "candidate_stage_comparisons",
                "marginal_changed_events",
                "same_pending_redundancy_events",
                "cross_pending_overlap_events",
                "other_events",
            )
            budgets[f"{int(ratio * 100)}%"] = {
                "n": len(ratio_rows),
                "variants": variants,
                "full_vs_without_position_aware": _aggregate_comparison(
                    ratio_rows, "without_position_aware"
                ),
                "full_vs_without_set_dependent": _aggregate_comparison(
                    ratio_rows, "without_set_dependent"
                ),
                "position_ranking": {
                    "static_ranking_difference_rate": sum(
                        bool(row["position_ranking"]["static_positive_ranking_different"])
                        for row in ratio_rows
                    ) / len(ratio_rows) if ratio_rows else None,
                    "static_top_candidate_difference_rate": sum(
                        bool(row["position_ranking"]["static_top_candidate_different"])
                        for row in ratio_rows
                    ) / len(ratio_rows) if ratio_rows else None,
                },
                "marginal_structure": {
                    key: sum(int(row["marginal_structure"][key]) for row in ratio_rows)
                    for key in structure_keys
                },
            }
        result[population] = {
            "snapshots": len({row["snapshot_id"] for row in population_rows}),
            "cases": len(population_rows),
            "budgets": budgets,
        }
    return result


def _overall_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总跨预算机制结果，并区分单 pending 与跨 pending 结构。"""
    result: dict[str, Any] = {}
    for population in ("OpenHands", "AgentX"):
        population_rows = [row for row in rows if row["population"] == population]
        structure = {
            key: sum(int(row["marginal_structure"][key]) for row in population_rows)
            for key in (
                "candidate_stage_comparisons",
                "marginal_changed_events",
                "same_pending_redundancy_events",
                "cross_pending_overlap_events",
                "other_events",
            )
        }
        cross_cases = [
            row
            for row in population_rows
            if row["marginal_structure"]["cross_pending_overlap_events"] > 0
        ]
        same_only_cases = [
            row
            for row in population_rows
            if row["marginal_structure"]["same_pending_redundancy_events"] > 0
            and row["marginal_structure"]["cross_pending_overlap_events"] == 0
        ]
        structure["cross_pending_case_count"] = len(cross_cases)
        structure["same_pending_only_case_count"] = len(same_only_cases)
        structure["set_dependent_reduction_in_cross_pending_cases"] = (
            _aggregate_comparison(cross_cases, "without_set_dependent")
            if cross_cases else None
        )
        structure["set_dependent_reduction_in_same_pending_only_cases"] = (
            _aggregate_comparison(same_only_cases, "without_set_dependent")
            if same_only_cases else None
        )
        result[population] = structure
    return result


def _source_paths() -> list[Path]:
    """列出本实验不允许变化的冻结实现。"""
    return [
        REPOSITORY_ROOT / "flowstate/optimizer.py",
        REPOSITORY_ROOT / "flowstate/recovery_model.py",
        REPOSITORY_ROOT / "flowstate/executable_state.py",
        REPOSITORY_ROOT / "flowstate/state_catalog.py",
        REPOSITORY_ROOT / "flowstate/workflow.py",
        REPOSITORY_ROOT / "evaluation/rq3_frozen_snapshot_evaluator.py",
        REPOSITORY_ROOT / "evaluation/rq3_formal_policy_evaluation.py",
        REPOSITORY_ROOT / "evaluation/rq3_sanity_structure_audit.py",
        REPOSITORY_ROOT / "evaluation/rq3_agentx_formal_policy_evaluation.py",
    ]


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    """记录一组输入文件摘要。"""
    return {str(path): _sha256_file(path) for path in paths}


def _input_paths(
    openhands_root: Path,
    openhands_results: Path,
    agentx_results: Path,
    gap_model_root: Path,
) -> list[Path]:
    """列出所有会被读取且不得改写的正式输入。"""
    paths = sorted((openhands_root / "snapshots").glob("g*.json"))
    paths.extend(
        [
            openhands_results / "per_snapshot_results.jsonl",
            agentx_results / "per_case_results.jsonl",
            gap_model_root / "candidate_models.json",
            gap_model_root / "model_selection.json",
            DEFAULT_B1_ROOT / "FORMAL_CANDIDATE_POPULATION.json",
            DEFAULT_B2_ROOT / "runtime_snapshots.jsonl",
            DEFAULT_B2_ROOT / "runtime_correctness.json",
            FROZEN_AGENTX_PATH,
        ]
    )
    return paths


def _report(
    status: str,
    aggregate: Mapping[str, Any],
    effects: Mapping[str, Any],
    output_root: Path,
) -> str:
    """生成可直接审阅的中文报告。"""
    lines = [
        "# RQ5-A 冻结 allocation snapshot 设计消融",
        "",
        f"- 状态：`{status}`",
        "- 执行方式：纯 CPU；没有启动 GPU。",
        "- 公共评分：全部 selected set 均以正式位置感知 M2 的 `C(S)` 评分。",
        "- 预算：`25% / 50% / 75%`，完全复用 RQ3 的 K 与并列规则。",
        "- 无位置感知：selector 内替换为冻结 M0；没有重新拟合。",
        "- 无集合边际：复用冻结 StandaloneTopK 的 empty-set 独立收益排序。",
        "- 无 workflow conditioning：`MISSING`；仓库中没有可靠冻结语义，未临时构造。",
        "",
        "## 主结果",
        "",
    ]
    for population in ("OpenHands", "AgentX"):
        lines.extend(
            [
                f"### {population}",
                "",
                f"- snapshots：`{aggregate[population]['snapshots']}`；cases：`{aggregate[population]['cases']}`。",
            ]
        )
        for ratio, entry in aggregate[population]["budgets"].items():
            position = entry["full_vs_without_position_aware"]
            marginal = entry["full_vs_without_set_dependent"]
            lines.append(
                f"- {ratio}：Full mean/median C={entry['variants']['full']['common_cost_ms']['mean']:.6f}/"
                f"{entry['variants']['full']['common_cost_ms']['median']:.6f} ms；"
                f"相对无位置感知 mean reduction={position['mean_relative_reduction']:.6%}，"
                f"W/T/L={position['win_tie_loss']['win']}/{position['win_tie_loss']['tie']}/{position['win_tie_loss']['loss']}；"
                f"相对无集合边际 mean reduction={marginal['mean_relative_reduction']:.6%}，"
                f"W/T/L={marginal['win_tie_loss']['win']}/{marginal['win_tie_loss']['tie']}/{marginal['win_tie_loss']['loss']}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 机制结论",
            "",
            f"- OpenHands 跨 pending 边际事件：`{effects['OpenHands']['cross_pending_overlap_events']}`；"
            f"单 pending 冗余事件：`{effects['OpenHands']['same_pending_redundancy_events']}`。",
            f"- AgentX 跨 pending 边际事件：`{effects['AgentX']['cross_pending_overlap_events']}`；"
            f"单 pending 冗余事件：`{effects['AgentX']['same_pending_redundancy_events']}`。",
            "- 结构解释：OpenHands 的候选基本各自只服务一个 pending，集合边际主要消除同一 pending 内重复覆盖；AgentX 的 inherited-FORK 候选可同时覆盖多个 pending，因此额外出现跨 pending overlap。",
            "- 以上是冻结 workload 上的 paired allocation 结果，不使用 future information，也不把 M0 升级为正式模型。",
            "",
            f"- artifact root：`{output_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_ablation(
    output_root: Path,
    *,
    openhands_root: Path = DEFAULT_OPENHANDS_ROOT,
    openhands_results: Path = DEFAULT_OPENHANDS_RESULTS,
    agentx_results: Path = DEFAULT_AGENTX_RESULTS,
    gap_model_root: Path = DEFAULT_GAP_MODEL_ROOT,
) -> Path:
    """执行 RQ5-A、写入新 artifact，并返回输出目录。"""
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"输出目录已存在且非空：{output_root}")
    source_paths = _source_paths()
    input_paths = _input_paths(
        openhands_root, openhands_results, agentx_results, gap_model_root
    )
    missing = [str(path) for path in source_paths + input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少冻结输入：{missing}")
    source_before = _hash_paths(source_paths)
    input_before = _hash_paths(input_paths)
    gap_model, gap_evidence = load_gap_only_model(gap_model_root)

    openhands = load_eligible_snapshots(openhands_root)
    agentx_formal, agentx_audit = build_agentx_formal_snapshots()
    if len(openhands) != EXPECTED_OPENHANDS_SNAPSHOTS:
        raise RuntimeError("OpenHands snapshot 数量不是 168")
    if len(agentx_formal) != EXPECTED_AGENTX_SNAPSHOTS:
        raise RuntimeError("AgentX snapshot 数量不是 23")
    if agentx_audit["b2_raw_compatibility_used"] is not False:
        raise RuntimeError("AgentX 意外使用 B2 raw compatibility")

    references = {
        "OpenHands": _load_reference_rows(
            openhands_results / "per_snapshot_results.jsonl"
        ),
        "AgentX": _load_reference_rows(
            agentx_results / "per_case_results.jsonl"
        ),
    }
    populations = {
        "OpenHands": openhands,
        "AgentX": [item.snapshot for item in agentx_formal],
    }
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    snapshot_digests_before = {
        f"{population}:{snapshot.snapshot_id}": snapshot.content_digest()
        for population, snapshots in populations.items()
        for snapshot in snapshots
    }
    for population, snapshots in populations.items():
        for snapshot in snapshots:
            for ratio, k in compute_budget_ks(len(snapshot.eligible_candidates)):
                row, gate = _evaluate_case(
                    population,
                    snapshot,
                    ratio,
                    k,
                    gap_model,
                    references[population],
                )
                rows.append(row)
                gates.append(gate)

    snapshot_digests_after = {
        f"{population}:{snapshot.snapshot_id}": snapshot.content_digest()
        for population, snapshots in populations.items()
        for snapshot in snapshots
    }
    source_after = _hash_paths(source_paths)
    input_after = _hash_paths(input_paths)
    aggregate = _aggregate_rows(rows)
    effects = _overall_effects(rows)
    expected_cases = EXPECTED_OPENHANDS_SNAPSHOTS * 3 + EXPECTED_AGENTX_SNAPSHOTS * 3
    all_pass = (
        len(rows) == expected_cases
        and all(gate["deterministic"] for gate in gates)
        and all(gate["rq3_full_reproduced"] for gate in gates)
        and snapshot_digests_before == snapshot_digests_after
        and source_before == source_after
        and input_before == input_after
    )
    status = "RQ5_ABLATION_READY" if all_pass else "RQ5_ABLATION_BLOCKED"
    determinism = {
        "status": "PASS" if all(gate["deterministic"] for gate in gates) else "FAIL",
        "case_count": len(gates),
        "selector_repeat_count": len(gates) * len(VARIANTS),
        "mismatch_count": sum(not gate["deterministic"] for gate in gates),
        "rq3_full_reproduction": (
            "PASS" if all(gate["rq3_full_reproduced"] for gate in gates) else "FAIL"
        ),
    }
    integrity = {
        "status": "PASS" if source_before == source_after and input_before == input_after else "FAIL",
        "frozen_source_unchanged": source_before == source_after,
        "formal_inputs_unchanged": input_before == input_after,
        "snapshots_unchanged": snapshot_digests_before == snapshot_digests_after,
        "source_before": source_before,
        "source_after": source_after,
        "input_before": input_before,
        "input_after": input_after,
    }
    manifest = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_started": False,
        "populations": {
            "OpenHands": EXPECTED_OPENHANDS_SNAPSHOTS,
            "AgentX": EXPECTED_AGENTX_SNAPSHOTS,
        },
        "cases": len(rows),
        "budgets": list(_FROZEN_BUDGET_RATIOS),
        "budget_protocol": "RQ3 冻结协议：K=max(1,floor(r*|C|))，排除 K>=|C|，重复 K 折叠。",
        "common_objective": "正式位置感知 M2 C(S)",
        "bootstrap": {
            "method": "以 snapshot 为 paired unit 的 percentile bootstrap",
            "iterations": _BOOTSTRAP_ITERATIONS,
            "seed": _BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "variants": {
            "full": "冻结 FlowState optimizer 与正式 M2",
            "without_position_aware": "冻结 FlowState optimizer 与冻结 M0 gap-only",
            "without_set_dependent": "冻结 StandaloneTopK，按 empty-set 独立收益固定排序",
            "without_workflow_conditioning": "MISSING",
        },
        "workflow_conditioning_ablation": {
            "status": "MISSING",
            "reason": "现有 artifact 与实现没有可靠冻结语义；按协议未临时发明。",
        },
        "future_information_used": False,
        "agentx_compatibility": {
            "source": agentx_audit["formal_compatibility_source"],
            "formal_max_d_t_c": agentx_audit["formal_max_d_t_c"],
            "b2_raw_compatibility_used": agentx_audit["b2_raw_compatibility_used"],
        },
        "gap_only_model": gap_evidence,
        "artifact_digest_scope": "manifest 自身不进入摘要；其余输入与冻结源逐文件留档。",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "per_case_results.jsonl", rows)
    _write_json(output_root / "aggregate_results.json", aggregate)
    _write_json(output_root / "structural_effects.json", effects)
    _write_json(output_root / "determinism.json", determinism)
    _write_json(output_root / "source_integrity.json", integrity)
    (output_root / "final_report.md").write_text(
        _report(status, aggregate, effects, output_root) + "\n",
        encoding="utf-8",
    )
    return output_root


def _default_output_root() -> Path:
    """构造不会覆盖既有产物的新目录名。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY_ROOT / "evaluation/runtime_artifacts" / (
        f"rq5_frozen_snapshot_ablation_{timestamp}"
    )


def main() -> int:
    """解析命令行并执行消融。"""
    parser = argparse.ArgumentParser(description="执行 RQ5-A 冻结快照设计消融")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or _default_output_root()
    print(run_ablation(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
