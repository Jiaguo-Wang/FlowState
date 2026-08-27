#!/usr/bin/env python3
"""审计正式位置感知恢复模型对受控实验与历史制品的影响。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.budget_sweep import (
    BUDGET_SWEEP_POLICY_NAMES,
    build_budget_sweep,
)
from evaluation.controlled_multiworkflow_v1.policies import (
    select_equal_share,
    select_global_lru,
    select_recovery_only,
    select_workflow_only,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    build_scenario as build_controlled_scenario,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    BUDGETS_BY_WORKFLOW_COUNT,
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_metadata import (
    build_kvflow_steps,
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import KVFlowStylePolicy, MarconiStylePolicy
from evaluation.sota_signal_stress_v1.scenario import (
    BUDGET_CHECKPOINTS as SIGNAL_BUDGETS,
    build_scenario as build_signal_scenario,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import (
    FORMAL_RECOVERY_MODEL_METADATA,
    HistoricalRecoveryCostModel,
    RecoveryCostModel,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "evaluation" / "formal_model_regression"
SCALABLE_OLD_ARTIFACT = (
    ROOT / "evaluation" / "scalable_multiworkflow_v2" / "offline_summary.json"
)
SIGNAL_OLD_ARTIFACT = (
    ROOT / "evaluation" / "sota_signal_stress_v1" / "offline_summary.json"
)
H100_CORRECTNESS_ARTIFACT = (
    ROOT
    / "evaluation"
    / "runtime_artifacts"
    / "sota_correctness_20260825_104513_833972"
    / "summary.json"
)
H100_LATENCY_ARTIFACT = (
    ROOT
    / "evaluation"
    / "runtime_artifacts"
    / "sota_latency_20260825_113526_839592"
)
TRACELAB_FORMAL_WORKLOAD = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_final_protocol.json"
)

POLICY_NAMES = BUDGET_SWEEP_POLICY_NAMES
BASELINE_SELECTION_INDEPENDENT_POLICIES = (
    "Global-LRU",
    "Equal-Share",
    "Workflow-Only",
    "KVFlow-style",
    "Marconi-style",
)
H100_POINTS = (
    ("Scalable N16 K4", "scalable_multiworkflow_v2_n16", 4),
    ("Scalable N16 K12", "scalable_multiworkflow_v2_n16", 12),
    ("SOTA-signal K4", "sota_signal_stress_v1", 4),
    ("SOTA-signal K8", "sota_signal_stress_v1", 8),
)
_FLOAT_TOLERANCE_MS = 1e-9


@dataclass(frozen=True)
class SelectionMetric:
    """记录一个选择集合在正式模型下的统一指标。"""

    total_recovery_cost_ms: float
    mean_recovery_cost_ms: float
    total_recovery_gap_tokens: int
    executable_hit_ratio: float


@dataclass(frozen=True)
class SelectionDiff:
    """记录一个 workload、预算与策略的旧新选择差异。"""

    workload: str
    budget_checkpoints: int
    policy_name: str
    old_selection: tuple[str, ...]
    new_selection: tuple[str, ...]
    selection_changed: bool
    selection_jaccard: float
    old_metric: SelectionMetric
    new_metric: SelectionMetric


@dataclass(frozen=True)
class OracleRegression:
    """记录正式模型下 FlowState 相对 exact Oracle 的差距。"""

    workload: str
    budget_checkpoints: int
    flowstate_selection: tuple[str, ...]
    oracle_selection: tuple[str, ...]
    flowstate_objective_ms: float
    oracle_objective_ms: float
    absolute_regret_ms: float
    relative_regret: float


def exact_oracle_selection(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    budget_checkpoints: int,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """利用工作流隔离性做精确动态规划，求解预算内全局最优子集。"""
    if budget_checkpoints < 0:
        raise ValueError("预算检查点数量必须大于等于零")
    workflow_ids = tuple(
        sorted(
            {
                continuation.workflow_id for continuation in continuations
            }
            | {candidate.workflow_id for candidate in candidates}
        )
    )
    dynamic: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for workflow_id in workflow_ids:
        workflow_candidates = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.workflow_id == workflow_id
                    and candidate.recurrent_resident
                ),
                key=lambda candidate: candidate.checkpoint_id,
            )
        )
        workflow_continuations = tuple(
            continuation
            for continuation in continuations
            if continuation.workflow_id == workflow_id
        )
        local_options = []
        for subset_size in range(len(workflow_candidates) + 1):
            for subset in combinations(workflow_candidates, subset_size):
                checkpoint_ids = tuple(
                    candidate.checkpoint_id for candidate in subset
                )
                local_options.append(
                    (
                        subset_size,
                        _recovery_cost(
                            workflow_continuations,
                            subset,
                            recovery_cost_model,
                        ),
                        checkpoint_ids,
                    )
                )

        following: dict[int, tuple[float, tuple[str, ...]]] = {}
        for used, (prefix_cost, prefix_ids) in dynamic.items():
            for local_size, local_cost, local_ids in local_options:
                total_size = used + local_size
                if total_size > budget_checkpoints:
                    continue
                ids = tuple(sorted(prefix_ids + local_ids))
                cost = prefix_cost + local_cost
                current = following.get(total_size)
                if current is None or _is_better(cost, ids, current):
                    following[total_size] = (cost, ids)
        dynamic = following

    best: tuple[float, tuple[str, ...]] | None = None
    for used in sorted(dynamic):
        candidate_result = dynamic[used]
        if best is None or _is_better(
            candidate_result[0],
            candidate_result[1],
            best,
        ):
            best = candidate_result
    return () if best is None else best[1]


def build_controlled_regression() -> tuple[
    tuple[SelectionDiff, ...],
    tuple[OracleRegression, ...],
]:
    """重新计算全部冻结受控 workload，并与旧模型选择逐项比较。"""
    formal_model = RecoveryCostModel()
    historical_model = HistoricalRecoveryCostModel()
    old_selections = _load_old_selections()
    selection_diffs = []
    oracle_rows = []
    for workload, budget, scenario in _build_scenario_points():
        for policy_name in POLICY_NAMES:
            new_selection = _select_policy(
                policy_name,
                scenario,
                budget,
                formal_model,
            )
            old_selection = old_selections[(workload, budget, policy_name)]
            selection_diffs.append(
                SelectionDiff(
                    workload=workload,
                    budget_checkpoints=budget,
                    policy_name=policy_name,
                    old_selection=old_selection,
                    new_selection=new_selection,
                    selection_changed=(
                        set(old_selection) != set(new_selection)
                    ),
                    selection_jaccard=_jaccard(old_selection, new_selection),
                    old_metric=evaluate_selection(
                        scenario.continuations,
                        scenario.candidates,
                        old_selection,
                        historical_model,
                    ),
                    new_metric=evaluate_selection(
                        scenario.continuations,
                        scenario.candidates,
                        new_selection,
                        formal_model,
                    ),
                )
            )

        flowstate = next(
            row
            for row in selection_diffs
            if row.workload == workload
            and row.budget_checkpoints == budget
            and row.policy_name == "FlowState"
        )
        oracle = next(
            row
            for row in selection_diffs
            if row.workload == workload
            and row.budget_checkpoints == budget
            and row.policy_name == "Oracle"
        )
        regret = (
            flowstate.new_metric.total_recovery_cost_ms
            - oracle.new_metric.total_recovery_cost_ms
        )
        if regret < -_FLOAT_TOLERANCE_MS:
            raise RuntimeError("FlowState 正式目标不能优于 exact Oracle")
        regret = max(0.0, regret)
        oracle_rows.append(
            OracleRegression(
                workload=workload,
                budget_checkpoints=budget,
                flowstate_selection=flowstate.new_selection,
                oracle_selection=oracle.new_selection,
                flowstate_objective_ms=(
                    flowstate.new_metric.total_recovery_cost_ms
                ),
                oracle_objective_ms=oracle.new_metric.total_recovery_cost_ms,
                absolute_regret_ms=regret,
                relative_regret=(
                    regret / oracle.new_metric.total_recovery_cost_ms
                    if oracle.new_metric.total_recovery_cost_ms > 0.0
                    else 0.0
                ),
            )
        )

    for policy_name in BASELINE_SELECTION_INDEPENDENT_POLICIES:
        changed = tuple(
            row
            for row in selection_diffs
            if row.policy_name == policy_name and row.selection_changed
        )
        if changed:
            raise RuntimeError(
                f"{policy_name} 的冻结选择意外依赖正式恢复模型"
            )
    return tuple(selection_diffs), tuple(oracle_rows)


def evaluate_selection(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    selected_ids: Sequence[str],
    recovery_cost_model: RecoveryCostModel,
) -> SelectionMetric:
    """用正式位置感知模型计算一个固定选择集合的统一指标。"""
    candidates_by_id = {
        candidate.checkpoint_id: candidate for candidate in candidates
    }
    selected = tuple(candidates_by_id[checkpoint_id] for checkpoint_id in selected_ids)
    gaps = tuple(
        recovery_gap(continuation, selected) for continuation in continuations
    )
    frontiers = tuple(
        executable_frontier(continuation, selected)
        for continuation in continuations
    )
    total_target = sum(
        continuation.planning_target for continuation in continuations
    )
    total_cost = sum(
        recovery_cost_model.estimate(gap, continuation.planning_target)
        for continuation, gap in zip(continuations, gaps)
    )
    return SelectionMetric(
        total_recovery_cost_ms=total_cost,
        mean_recovery_cost_ms=(
            total_cost / len(continuations) if continuations else 0.0
        ),
        total_recovery_gap_tokens=sum(gaps),
        executable_hit_ratio=(
            sum(frontiers) / total_target if total_target else 0.0
        ),
    )


def build_h100_selection_audit() -> dict[str, object]:
    """比较历史 H100 实际选择与正式模型重新计算的选择。"""
    with H100_CORRECTNESS_ARTIFACT.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    formal_model = RecoveryCostModel()
    rows = []
    rerun_points = []
    for label, scenario_name, budget in H100_POINTS:
        scenario = _build_h100_scenario(scenario_name, budget)
        groups = artifact["scenario_summaries"][scenario_name]["groups"]
        flowstate_group = next(
            group
            for group in groups
            if group["policy"] == "FlowState"
            and int(group["budget_checkpoints"]) == budget
        )
        old_selection = tuple(flowstate_group["selected_checkpoint_ids"])
        new_selection = _select_policy(
            "FlowState",
            scenario,
            budget,
            formal_model,
        )
        old_metric = evaluate_selection(
            scenario.continuations,
            scenario.candidates,
            old_selection,
            formal_model,
        )
        new_metric = evaluate_selection(
            scenario.continuations,
            scenario.candidates,
            new_selection,
            formal_model,
        )
        if set(old_selection) == set(new_selection):
            classification = "IDENTICAL"
            reusability = "REUSABLE"
        elif abs(
            old_metric.total_recovery_cost_ms
            - new_metric.total_recovery_cost_ms
        ) <= _FLOAT_TOLERANCE_MS:
            classification = "MULTIPLE_OPTIMUM_EQUIVALENT"
            reusability = "OBJECTIVE-EQUIVALENT BUT RUNTIME-RERUN RECOMMENDED"
            rerun_points.append(label)
        else:
            classification = "CHANGED"
            reusability = "NOT REUSABLE FOR FINAL FLOWSTATE CLAIM"
            rerun_points.append(label)
        rows.append(
            {
                "point": label,
                "scenario": scenario_name,
                "budget_checkpoints": budget,
                "old_h100_selection": old_selection,
                "new_formal_selection": new_selection,
                "classification": classification,
                "existing_artifact": reusability,
                "old_selection_formal_metric": asdict(old_metric),
                "new_selection_formal_metric": asdict(new_metric),
                "measured_ttft_unchanged": True,
            }
        )
    return {
        "source_correctness_artifact": str(H100_CORRECTNESS_ARTIFACT),
        "source_latency_artifact": str(H100_LATENCY_ARTIFACT),
        "points": rows,
        "gpu_rerun_required": bool(rerun_points),
        "gpu_rerun_points": rerun_points,
    }


def build_trace_model_compatibility(sample_snapshot_count: int = 12) -> dict[str, object]:
    """只验证冻结 TraceLab 样本可进入正式模型，不执行任何策略。"""
    with TRACELAB_FORMAL_WORKLOAD.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ordered = sorted(
        payload["snapshots"],
        key=lambda row: row["snapshot"]["snapshot_id"],
    )
    sampled = ordered[:sample_snapshot_count]
    model = RecoveryCostModel()
    checked = 0
    violations = []
    sample_ids = []
    for row in sampled:
        snapshot = row["snapshot"]
        sample_ids.append(snapshot["snapshot_id"])
        for item in snapshot["continuations"]:
            target = min(
                int(item["anchor_pos"]),
                int(item["resident_fa_frontier"]),
            )
            try:
                model.estimate(0, target)
                model.estimate(target, target)
            except (TypeError, ValueError) as error:
                violations.append(
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "continuation_id": item["continuation_id"],
                        "target_tokens": target,
                        "error": str(error),
                    }
                )
            checked += 1
    return {
        "source": str(TRACELAB_FORMAL_WORKLOAD),
        "sampling": "按 snapshot_id 字典序取前十二个快照",
        "sample_snapshot_ids": sample_ids,
        "continuations_checked": checked,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
        "policy_comparison_executed": False,
    }


def write_artifacts(
    selection_diffs: Sequence[SelectionDiff],
    oracle_rows: Sequence[OracleRegression],
    h100_audit: Mapping[str, object],
    trace_compatibility: Mapping[str, object],
) -> tuple[Path, ...]:
    """写出正式模型集成的独立回归制品。"""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    controlled_path = OUTPUT_DIRECTORY / "controlled_regression.json"
    selection_path = OUTPUT_DIRECTORY / "selection_diff.csv"
    oracle_path = OUTPUT_DIRECTORY / "oracle_regression.csv"
    h100_path = OUTPUT_DIRECTORY / "h100_selection_audit.json"
    trace_path = OUTPUT_DIRECTORY / "trace_model_compatibility.json"

    payload = {
        "schema_version": "flowstate.formal_model_regression.v1",
        "formal_model": asdict(FORMAL_RECOVERY_MODEL_METADATA),
        "selection_diffs": [asdict(row) for row in selection_diffs],
        "policy_rankings": _build_policy_rankings(selection_diffs),
        "oracle_regression": [asdict(row) for row in oracle_rows],
        "baseline_selection_independence": True,
        "trace_policy_comparison_executed": False,
        "gpu_executed": False,
    }
    _write_json(controlled_path, payload)
    _write_json(h100_path, dict(h100_audit))
    _write_json(trace_path, dict(trace_compatibility))

    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "workload",
            "budget_checkpoints",
            "policy_name",
            "old_selection",
            "new_selection",
            "selection_changed",
            "selection_jaccard",
            "old_total_recovery_cost_ms",
            "new_total_recovery_cost_ms",
            "new_mean_recovery_cost_ms",
            "new_total_recovery_gap_tokens",
            "new_executable_hit_ratio",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in selection_diffs:
            writer.writerow(
                {
                    "workload": row.workload,
                    "budget_checkpoints": row.budget_checkpoints,
                    "policy_name": row.policy_name,
                    "old_selection": ";".join(row.old_selection),
                    "new_selection": ";".join(row.new_selection),
                    "selection_changed": row.selection_changed,
                    "selection_jaccard": row.selection_jaccard,
                    "old_total_recovery_cost_ms": row.old_metric.total_recovery_cost_ms,
                    "new_total_recovery_cost_ms": row.new_metric.total_recovery_cost_ms,
                    "new_mean_recovery_cost_ms": row.new_metric.mean_recovery_cost_ms,
                    "new_total_recovery_gap_tokens": row.new_metric.total_recovery_gap_tokens,
                    "new_executable_hit_ratio": row.new_metric.executable_hit_ratio,
                }
            )

    with oracle_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = tuple(asdict(oracle_rows[0]).keys()) if oracle_rows else ()
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in oracle_rows:
            values = asdict(row)
            values["flowstate_selection"] = ";".join(row.flowstate_selection)
            values["oracle_selection"] = ";".join(row.oracle_selection)
            writer.writerow(values)
    return controlled_path, selection_path, oracle_path, h100_path, trace_path


def _build_scenario_points():
    """按冻结顺序产生全部受控 workload 与预算点。"""
    for budget in (1, 2, 3, 4, 5):
        yield "controlled_v1", budget, build_controlled_scenario()
    for workflow_count in (8, 16):
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]:
            yield (
                f"scalable_n{workflow_count}",
                budget,
                build_scalable_scenario(workflow_count, budget),
            )
    for budget in SIGNAL_BUDGETS:
        yield "sota_signal", budget, build_signal_scenario(budget)


def _select_policy(
    policy_name: str,
    scenario,
    budget_checkpoints: int,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """用冻结策略信号选择检查点；只有正式目标策略读取恢复模型。"""
    budget_bytes = budget_checkpoints * scenario.metadata.checkpoint_size_bytes
    if policy_name == "FlowState":
        result = GlobalOptimizer(recovery_cost_model).select(
            scenario.continuations,
            scenario.candidates,
            budget_bytes,
        )
        return tuple(candidate.checkpoint_id for candidate in result.selected)
    if policy_name == "Global-LRU":
        return select_global_lru(
            scenario.candidates,
            scenario.metadata.checkpoint_recency,
            budget_bytes,
        )
    if policy_name == "Equal-Share":
        return select_equal_share(
            scenario.continuations,
            scenario.candidates,
            scenario.metadata.workflow_order,
            budget_bytes,
        )
    if policy_name == "Recovery-Only":
        return select_recovery_only(
            scenario.continuations,
            scenario.candidates,
            budget_bytes,
            recovery_cost_model,
        )
    if policy_name == "Workflow-Only":
        return select_workflow_only(
            scenario.continuations,
            scenario.candidates,
            budget_bytes,
        )

    last_access = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    if policy_name == "KVFlow-style":
        steps = getattr(
            scenario.metadata,
            "steps_to_execution_by_continuation",
            None,
        )
        if steps is None:
            steps = build_kvflow_steps(scenario.continuations)
        return KVFlowStylePolicy().select(
            scenario.continuations,
            scenario.candidates,
            budget_checkpoints,
            steps,
            last_access,
        ).selected_checkpoint_ids
    if policy_name == "Marconi-style":
        return MarconiStylePolicy().select(
            scenario.candidates,
            budget_checkpoints,
            last_access,
            build_marconi_flop_saved(scenario.candidates),
            getattr(scenario.metadata, "marconi_alpha", 1.0),
        ).selected_checkpoint_ids
    if policy_name == "Oracle":
        return exact_oracle_selection(
            scenario.continuations,
            scenario.candidates,
            budget_checkpoints,
            recovery_cost_model,
        )
    raise ValueError(f"未知策略：{policy_name}")


def _load_old_selections() -> dict[tuple[str, int, str], tuple[str, ...]]:
    """从冻结旧制品或历史模型复现读取旧选择，不覆盖任何制品。"""
    old: dict[tuple[str, int, str], tuple[str, ...]] = {}
    controlled = build_budget_sweep(
        recovery_cost_model=HistoricalRecoveryCostModel(),
    )
    for row in controlled.rows:
        old[("controlled_v1", row.budget_checkpoints, row.policy_name)] = (
            row.selected_checkpoint_ids
        )
    _load_old_json_rows(SCALABLE_OLD_ARTIFACT, old, "scalable")
    _load_old_json_rows(SIGNAL_OLD_ARTIFACT, old, "sota_signal")
    return old


def _load_old_json_rows(
    path: Path,
    destination: dict[tuple[str, int, str], tuple[str, ...]],
    workload_prefix: str,
) -> None:
    """读取旧离线行，不调用其已过期的正式模型校验。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload["rows"]:
        workload = workload_prefix
        if workload_prefix == "scalable":
            workload = f"scalable_n{int(row['workflow_count'])}"
        destination[
            (
                workload,
                int(row["budget_checkpoints"]),
                row["policy_name"],
            )
        ] = tuple(row["selected_checkpoint_ids"])


def _build_h100_scenario(scenario_name: str, budget: int):
    """按历史 H100 代表点标识重建冻结逻辑场景。"""
    if scenario_name == "scalable_multiworkflow_v2_n16":
        return build_scalable_scenario(16, budget)
    if scenario_name == "sota_signal_stress_v1":
        return build_signal_scenario(budget)
    raise ValueError(f"未知 H100 场景：{scenario_name}")


def _recovery_cost(
    continuations: Sequence[PendingContinuation],
    selected: Sequence[CheckpointCandidate],
    recovery_cost_model: RecoveryCostModel,
) -> float:
    """显式传递每个待续请求固定的规划目标位置。"""
    return sum(
        recovery_cost_model.estimate(
            recovery_gap(continuation, selected),
            continuation.planning_target,
        )
        for continuation in continuations
    )


def _is_better(
    cost: float,
    ids: tuple[str, ...],
    current: tuple[float, tuple[str, ...]],
) -> bool:
    """按目标值优先、检查点标识字典序次优先比较两个解。"""
    return cost < current[0] - _FLOAT_TOLERANCE_MS or (
        abs(cost - current[0]) <= _FLOAT_TOLERANCE_MS and ids < current[1]
    )


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """计算两个检查点集合的 Jaccard 相似度。"""
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _write_json(path: Path, payload: object) -> None:
    """以稳定排序写出 UTF-8 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _build_policy_rankings(
    selection_diffs: Sequence[SelectionDiff],
) -> list[dict[str, object]]:
    """按旧新统一目标分别排序策略，显式记录相对顺序变化。"""
    keys = tuple(
        sorted(
            {
                (row.workload, row.budget_checkpoints)
                for row in selection_diffs
            }
        )
    )
    rankings = []
    for workload, budget in keys:
        rows = tuple(
            row
            for row in selection_diffs
            if row.workload == workload
            and row.budget_checkpoints == budget
        )
        old_ranking = tuple(
            row.policy_name
            for row in sorted(
                rows,
                key=lambda row: (
                    row.old_metric.total_recovery_cost_ms,
                    row.policy_name,
                ),
            )
        )
        new_ranking = tuple(
            row.policy_name
            for row in sorted(
                rows,
                key=lambda row: (
                    row.new_metric.total_recovery_cost_ms,
                    row.policy_name,
                ),
            )
        )
        rankings.append(
            {
                "workload": workload,
                "budget_checkpoints": budget,
                "old_ranking": old_ranking,
                "new_ranking": new_ranking,
                "ranking_changed": old_ranking != new_ranking,
            }
        )
    return rankings


def main() -> None:
    """执行纯离线回归审计并写出独立制品。"""
    selection_diffs, oracle_rows = build_controlled_regression()
    h100_audit = build_h100_selection_audit()
    trace_compatibility = build_trace_model_compatibility()
    paths = write_artifacts(
        selection_diffs,
        oracle_rows,
        h100_audit,
        trace_compatibility,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
