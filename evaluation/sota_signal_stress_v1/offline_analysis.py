#!/usr/bin/env python3
"""执行 SOTA 信号受控 workload 的八策略离线分析。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Sequence

from evaluation.controlled_multiworkflow_v1.policies import (
    select_equal_share,
    select_global_lru,
    select_recovery_only,
    select_workflow_only,
)
from evaluation.sota_metadata import (
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import KVFlowStylePolicy, MarconiStylePolicy
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate

from .scenario import (
    ANCHOR_DEPTHS,
    BUDGET_CHECKPOINTS,
    FANOUTS,
    RECENCY_CLASSES,
    STEPS_TO_EXECUTION,
    SignalScenario,
    SignalWorkflowSpec,
    build_scenario,
)


POLICY_NAMES = (
    "Global-LRU",
    "Equal-Share",
    "Workflow-Only",
    "Recovery-Only",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
    "Oracle",
)
_OUTPUT_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PolicyFactorDistribution:
    """记录所选 workflow 在四个因素各层级上的数量。"""

    anchor_8192: int
    anchor_32768: int
    fanout_1: int
    fanout_4: int
    steps_1: int
    steps_3: int
    recency_old: int
    recency_recent: int


@dataclass(frozen=True)
class SignalSummaryRow:
    """记录一个预算与策略的选择、规划指标和因素分布。"""

    budget_checkpoints: int
    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    selected_factor_tuples: tuple[tuple[int, int, int, str], ...]
    total_recovery_gap_tokens: int
    planning_executable_prefix_ratio: float
    estimated_recovery_cost_ms: float
    used_budget_checkpoints: int
    used_budget_bytes: int
    factor_distribution: PolicyFactorDistribution


@dataclass(frozen=True)
class SignalAnalysisResult:
    """汇总全部预算点和八个策略的离线结果。"""

    rows: tuple[SignalSummaryRow, ...]


def run_offline_analysis(
    recovery_cost_model: RecoveryCostModel | None = None,
) -> SignalAnalysisResult:
    """预先冻结场景后完成四个预算点的确定性离线比较。"""
    model = recovery_cost_model or RecoveryCostModel()
    base_scenario = build_scenario(BUDGET_CHECKPOINTS[0])
    validate_factorial_scenario(base_scenario)
    oracle_selections = _build_exact_oracle_selections(
        base_scenario,
        model,
    )

    rows = []
    for budget_checkpoints in BUDGET_CHECKPOINTS:
        scenario = build_scenario(budget_checkpoints)
        last_access = build_marconi_recency(
            scenario.candidates,
            scenario.metadata.checkpoint_recency,
        )
        flop_saved = build_marconi_flop_saved(scenario.candidates)
        for policy_name in POLICY_NAMES:
            selected_ids = _select_checkpoint_ids(
                policy_name,
                scenario,
                model,
                last_access,
                flop_saved,
                oracle_selections[budget_checkpoints],
            )
            rows.append(
                _build_summary_row(
                    scenario,
                    policy_name,
                    selected_ids,
                    model,
                )
            )
    result = SignalAnalysisResult(rows=tuple(rows))
    validate_analysis(result)
    return result


def validate_factorial_scenario(scenario: SignalScenario) -> None:
    """验证完整阶乘、核心兼容性隔离和固定 metadata 规则。"""
    expected = {
        (anchor, fanout, steps, recency)
        for anchor in ANCHOR_DEPTHS
        for fanout in FANOUTS
        for steps in STEPS_TO_EXECUTION
        for recency in RECENCY_CLASSES
    }
    observed = {
        workflow.factor_tuple for workflow in scenario.metadata.workflows
    }
    if observed != expected or len(scenario.metadata.workflows) != 16:
        raise RuntimeError("四因素 workload 不是完整且唯一的 2×2×2×2 阶乘")
    if len(scenario.candidates) != 16:
        raise RuntimeError("每个工作流必须恰好有一个 main candidate")
    if len(scenario.continuations) != 40:
        raise RuntimeError("完整阶乘 workload 必须有四十个待续请求")

    workflows_by_id = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }
    for candidate in scenario.candidates:
        workflow = workflows_by_id[candidate.workflow_id]
        if (
            candidate.token_pos != workflow.anchor_depth
            or candidate.lineage_path != ("P",)
            or not candidate.recurrent_resident
            or not candidate.fa_resident
        ):
            raise RuntimeError("main candidate 未遵循固定构造规则")
    for continuation in scenario.continuations:
        workflow = workflows_by_id[continuation.workflow_id]
        if continuation.planning_target != workflow.anchor_depth:
            raise RuntimeError("待续请求 planning target 与 anchor 不一致")
        if scenario.metadata.steps_to_execution_by_continuation[
            continuation.continuation_id
        ] != workflow.steps_to_execution:
            raise RuntimeError("待续请求没有继承工作流的执行距离")

    recency_by_id = {
        item.checkpoint_id: item.last_access_order
        for item in scenario.metadata.checkpoint_recency
    }
    old_ranks = tuple(
        recency_by_id[candidate.checkpoint_id]
        for candidate in scenario.candidates
        if workflows_by_id[candidate.workflow_id].recency_class == "old"
    )
    recent_ranks = tuple(
        recency_by_id[candidate.checkpoint_id]
        for candidate in scenario.candidates
        if workflows_by_id[candidate.workflow_id].recency_class == "recent"
    )
    if max(old_ranks) >= min(recent_ranks):
        raise RuntimeError("recent checkpoint 必须严格晚于全部 old checkpoint")


def validate_analysis(result: SignalAnalysisResult) -> None:
    """验证预算、安全、Oracle 最优性和 full-budget sanity。"""
    if len(result.rows) != len(BUDGET_CHECKPOINTS) * len(POLICY_NAMES):
        raise RuntimeError("离线结果没有覆盖完整预算与策略矩阵")
    rows_by_key = {
        (row.budget_checkpoints, row.policy_name): row
        for row in result.rows
    }
    for budget in BUDGET_CHECKPOINTS:
        oracle = rows_by_key[(budget, "Oracle")]
        for policy_name in POLICY_NAMES:
            row = rows_by_key[(budget, policy_name)]
            if row.used_budget_checkpoints > budget:
                raise RuntimeError(f"{policy_name} 超出 K={budget} 预算")
            if len(set(row.selected_checkpoint_ids)) != len(
                row.selected_checkpoint_ids
            ):
                raise RuntimeError(f"{policy_name} 返回重复检查点")
            if (
                oracle.estimated_recovery_cost_ms
                > row.estimated_recovery_cost_ms + 1e-9
            ):
                raise RuntimeError(f"K={budget} 的 Oracle 目标值不是最优")
    for policy_name in POLICY_NAMES:
        full = rows_by_key[(16, policy_name)]
        if (
            full.total_recovery_gap_tokens != 0
            or full.planning_executable_prefix_ratio != 1.0
        ):
            raise RuntimeError(f"{policy_name} 未通过 K=16 满预算检查")


def write_offline_artifacts(
    result: SignalAnalysisResult,
    output_directory: Path = _OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """写出可复核的 CSV 与 JSON 离线 artifact。"""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "offline_summary.csv"
    json_path = output_directory / "offline_summary.json"
    fieldnames = (
        "budget_checkpoints",
        "policy_name",
        "selected_checkpoint_ids",
        "selected_factor_tuples",
        "total_recovery_gap_tokens",
        "planning_executable_prefix_ratio",
        "estimated_recovery_cost_ms",
        "used_budget_checkpoints",
        "used_budget_bytes",
        "anchor_8192",
        "anchor_32768",
        "fanout_1",
        "fanout_4",
        "steps_1",
        "steps_3",
        "recency_old",
        "recency_recent",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in result.rows:
            distribution = asdict(row.factor_distribution)
            writer.writerow(
                {
                    "budget_checkpoints": row.budget_checkpoints,
                    "policy_name": row.policy_name,
                    "selected_checkpoint_ids": ";".join(
                        row.selected_checkpoint_ids
                    ),
                    "selected_factor_tuples": ";".join(
                        "|".join(str(value) for value in factor_tuple)
                        for factor_tuple in row.selected_factor_tuples
                    ),
                    "total_recovery_gap_tokens": (
                        row.total_recovery_gap_tokens
                    ),
                    "planning_executable_prefix_ratio": (
                        row.planning_executable_prefix_ratio
                    ),
                    "estimated_recovery_cost_ms": (
                        row.estimated_recovery_cost_ms
                    ),
                    "used_budget_checkpoints": row.used_budget_checkpoints,
                    "used_budget_bytes": row.used_budget_bytes,
                    **distribution,
                }
            )

    payload = {
        "schema_version": "flowstate.sota_signal_stress.offline.v1",
        "workload_construction": "FIXED BEFORE POLICY EVALUATION",
        "rows": [asdict(row) for row in result.rows],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return csv_path, json_path


def load_offline_artifact(
    json_path: Path = _OUTPUT_DIRECTORY / "offline_summary.json",
) -> SignalAnalysisResult:
    """读取并验证已保存的 SOTA 信号离线 artifact。"""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != (
        "flowstate.sota_signal_stress.offline.v1"
    ):
        raise ValueError("SOTA 信号 artifact schema 不匹配")
    rows = tuple(
        SignalSummaryRow(
            **{
                **row,
                "selected_checkpoint_ids": tuple(
                    row["selected_checkpoint_ids"]
                ),
                "selected_factor_tuples": tuple(
                    tuple(factor_tuple)
                    for factor_tuple in row["selected_factor_tuples"]
                ),
                "factor_distribution": PolicyFactorDistribution(
                    **row["factor_distribution"]
                ),
            }
        )
        for row in payload["rows"]
    )
    result = SignalAnalysisResult(rows=rows)
    validate_analysis(result)
    return result


def _select_checkpoint_ids(
    policy_name: str,
    scenario: SignalScenario,
    recovery_cost_model: RecoveryCostModel,
    last_access_by_checkpoint: dict[str, float],
    flop_saved_by_checkpoint: dict[str, float],
    oracle_selection: tuple[str, ...],
) -> tuple[str, ...]:
    """只向每个策略传递其允许读取的固定输入。"""
    if policy_name == "Global-LRU":
        return select_global_lru(
            scenario.candidates,
            scenario.metadata.checkpoint_recency,
            scenario.budget_bytes,
        )
    if policy_name == "Equal-Share":
        return select_equal_share(
            scenario.continuations,
            scenario.candidates,
            scenario.metadata.workflow_order,
            scenario.budget_bytes,
        )
    if policy_name == "Workflow-Only":
        return select_workflow_only(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
        )
    if policy_name == "Recovery-Only":
        return select_recovery_only(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
            recovery_cost_model,
        )
    if policy_name == "KVFlow-style":
        return KVFlowStylePolicy().select(
            scenario.continuations,
            scenario.candidates,
            scenario.metadata.budget_checkpoints,
            scenario.metadata.steps_to_execution_by_continuation,
            last_access_by_checkpoint,
        ).selected_checkpoint_ids
    if policy_name == "Marconi-style":
        return MarconiStylePolicy().select(
            scenario.candidates,
            scenario.metadata.budget_checkpoints,
            last_access_by_checkpoint,
            flop_saved_by_checkpoint,
            scenario.metadata.marconi_alpha,
        ).selected_checkpoint_ids
    if policy_name == "FlowState":
        allocation = GlobalOptimizer(recovery_cost_model).select(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
        )
        return tuple(
            candidate.checkpoint_id for candidate in allocation.selected
        )
    if policy_name == "Oracle":
        return oracle_selection
    raise ValueError(f"未知策略：{policy_name}")


def _build_summary_row(
    scenario: SignalScenario,
    policy_name: str,
    selected_ids: tuple[str, ...],
    recovery_cost_model: RecoveryCostModel,
) -> SignalSummaryRow:
    """使用核心 E、G 与 Phi 计算统一指标和因素分布。"""
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    workflows_by_id = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }
    selected = _resolve_selected(selected_ids, candidates_by_id)
    selected_workflows = tuple(
        workflows_by_id[candidate.workflow_id] for candidate in selected
    )
    frontiers = tuple(
        executable_frontier(continuation, selected)
        for continuation in scenario.continuations
    )
    gaps = tuple(
        recovery_gap(continuation, selected)
        for continuation in scenario.continuations
    )
    total_target = sum(
        continuation.planning_target
        for continuation in scenario.continuations
    )
    if sum(frontiers) + sum(gaps) != total_target:
        raise RuntimeError("规划前沿与恢复间隔未分解完整目标")
    return SignalSummaryRow(
        budget_checkpoints=scenario.metadata.budget_checkpoints,
        policy_name=policy_name,
        selected_checkpoint_ids=selected_ids,
        selected_factor_tuples=tuple(
            workflow.factor_tuple for workflow in selected_workflows
        ),
        total_recovery_gap_tokens=sum(gaps),
        planning_executable_prefix_ratio=(
            sum(frontiers) / total_target if total_target else 0.0
        ),
        estimated_recovery_cost_ms=sum(
            recovery_cost_model.estimate(
                gap,
                continuation.planning_target,
            )
            for continuation, gap in zip(
                scenario.continuations,
                gaps,
            )
        ),
        used_budget_checkpoints=len(selected),
        used_budget_bytes=sum(
            candidate.memory_bytes for candidate in selected
        ),
        factor_distribution=_build_factor_distribution(
            selected_workflows
        ),
    )


def _resolve_selected(
    selected_ids: Sequence[str],
    candidates_by_id: dict[str, CheckpointCandidate],
) -> tuple[CheckpointCandidate, ...]:
    """解析策略选择，并拒绝未知、重复或非驻留状态。"""
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("策略选择包含重复检查点")
    selected = []
    for checkpoint_id in selected_ids:
        candidate = candidates_by_id.get(checkpoint_id)
        if candidate is None:
            raise RuntimeError(f"策略选择未知检查点：{checkpoint_id}")
        if not candidate.recurrent_resident:
            raise RuntimeError(f"策略选择非驻留检查点：{checkpoint_id}")
        selected.append(candidate)
    return tuple(selected)


def _build_factor_distribution(
    workflows: Sequence[SignalWorkflowSpec],
) -> PolicyFactorDistribution:
    """统计选中工作流在每个二值因素上的数量。"""
    return PolicyFactorDistribution(
        anchor_8192=sum(
            workflow.anchor_depth == 8_192 for workflow in workflows
        ),
        anchor_32768=sum(
            workflow.anchor_depth == 32_768 for workflow in workflows
        ),
        fanout_1=sum(workflow.fanout == 1 for workflow in workflows),
        fanout_4=sum(workflow.fanout == 4 for workflow in workflows),
        steps_1=sum(
            workflow.steps_to_execution == 1 for workflow in workflows
        ),
        steps_3=sum(
            workflow.steps_to_execution == 3 for workflow in workflows
        ),
        recency_old=sum(
            workflow.recency_class == "old" for workflow in workflows
        ),
        recency_recent=sum(
            workflow.recency_class == "recent" for workflow in workflows
        ),
    )


def _build_exact_oracle_selections(
    scenario: SignalScenario,
    recovery_cost_model: RecoveryCostModel,
) -> dict[int, tuple[str, ...]]:
    """枚举全部子集，一次性求出每个固定预算的 exact objective。"""
    candidates = tuple(
        sorted(
            (
                candidate
                for candidate in scenario.candidates
                if candidate.recurrent_resident
            ),
            key=lambda candidate: candidate.checkpoint_id,
        )
    )
    best_by_size: dict[int, tuple[float, tuple[str, ...]]] = {}
    for subset_size in range(len(candidates) + 1):
        for subset in combinations(candidates, subset_size):
            selected_ids = tuple(
                candidate.checkpoint_id for candidate in subset
            )
            cost = sum(
                recovery_cost_model.estimate(
                    recovery_gap(continuation, subset),
                    continuation.planning_target,
                )
                for continuation in scenario.continuations
            )
            best = best_by_size.get(subset_size)
            if (
                best is None
                or cost < best[0] - 1e-9
                or (
                    abs(cost - best[0]) <= 1e-9
                    and selected_ids < best[1]
                )
            ):
                best_by_size[subset_size] = (cost, selected_ids)

    selections = {}
    for budget in BUDGET_CHECKPOINTS:
        best_cost: float | None = None
        best_ids: tuple[str, ...] | None = None
        for subset_size in range(budget + 1):
            cost, selected_ids = best_by_size[subset_size]
            if (
                best_cost is None
                or cost < best_cost - 1e-9
                or (
                    abs(cost - best_cost) <= 1e-9
                    and (best_ids is None or selected_ids < best_ids)
                )
            ):
                best_cost = cost
                best_ids = selected_ids
        selections[budget] = best_ids or ()
    return selections


def main() -> None:
    """运行完整离线分析并保存固定 artifact。"""
    result = run_offline_analysis()
    csv_path, json_path = write_offline_artifacts(result)
    print(f"离线 CSV：{csv_path}")
    print(f"离线 JSON：{json_path}")


if __name__ == "__main__":
    main()
