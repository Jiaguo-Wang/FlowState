#!/usr/bin/env python3
"""执行可扩展受控多工作流 v2 的六策略离线分析。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from evaluation.controlled_multiworkflow_v1.policies import (
    select_equal_share,
    select_global_lru,
    select_oracle,
    select_recovery_only,
    select_workflow_only,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    BUDGETS_BY_WORKFLOW_COUNT,
    FANOUTS_BY_WORKFLOW_COUNT,
    ScalableScenario,
    build_scenario,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate


POLICY_NAMES = (
    "FlowState",
    "Global-LRU",
    "Equal-Share",
    "Recovery-Only",
    "Workflow-Only",
    "Oracle",
)
_OUTPUT_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OfflineSummaryRow:
    """记录一个 workload、预算和策略对应的规划指标。"""

    workflow_count: int
    budget_checkpoints: int
    budget_ratio: float
    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    total_recovery_gap: int
    mean_recovery_gap_per_request: float
    planning_executable_prefix_ratio: float
    estimated_recovery_cost_ms: float
    selection_runtime_ms: float


@dataclass(frozen=True)
class FlowStateOracleDifference:
    """记录 FlowState 相对 exact Oracle 的目标差距。"""

    workflow_count: int
    budget_checkpoints: int
    absolute_objective_gap_ms: float
    relative_objective_gap: float | None
    flowstate_exact_optimal: bool


@dataclass(frozen=True)
class OfflineAnalysisResult:
    """汇总所有预算点、Oracle 差值和求解耗时。"""

    rows: tuple[OfflineSummaryRow, ...]
    flowstate_oracle_differences: tuple[FlowStateOracleDifference, ...]
    oracle_runtime_ms_by_workload: tuple[tuple[int, float], ...]


def analyze_workload(
    workflow_count: int,
    recovery_cost_model: RecoveryCostModel | None = None,
    budget_options: Sequence[int] | None = None,
) -> tuple[OfflineSummaryRow, ...]:
    """计算一个 workflow 规模下指定预算点的六策略结果。"""
    model = recovery_cost_model or RecoveryCostModel()
    allowed_budgets = BUDGETS_BY_WORKFLOW_COUNT.get(workflow_count)
    if allowed_budgets is None:
        raise ValueError("workflow_count 只支持 8 或 16")
    active_budgets = (
        allowed_budgets
        if budget_options is None
        else tuple(budget_options)
    )
    if not active_budgets or any(
        budget not in allowed_budgets for budget in active_budgets
    ):
        raise ValueError("预算必须来自 workload 的固定归一化预算集合")
    if len(set(active_budgets)) != len(active_budgets):
        raise ValueError("预算集合不能包含重复值")

    rows = []
    for budget_checkpoints in sorted(active_budgets):
        scenario = build_scenario(workflow_count, budget_checkpoints)
        for policy_name in POLICY_NAMES:
            started = perf_counter()
            selected_ids = _select_checkpoint_ids(
                policy_name,
                scenario,
                model,
            )
            selection_runtime_ms = (perf_counter() - started) * 1_000.0
            rows.append(
                _build_summary_row(
                    scenario,
                    policy_name,
                    selected_ids,
                    model,
                    selection_runtime_ms,
                )
            )
    return tuple(rows)


def run_offline_analysis(
    recovery_cost_model: RecoveryCostModel | None = None,
) -> OfflineAnalysisResult:
    """完成 N=8 与 N=16 的全部固定预算扫描并验证性质。"""
    model = recovery_cost_model or RecoveryCostModel()
    rows = tuple(
        row
        for workflow_count in (8, 16)
        for row in analyze_workload(workflow_count, model)
    )
    differences = _build_oracle_differences(rows)
    result = OfflineAnalysisResult(
        rows=rows,
        flowstate_oracle_differences=differences,
        oracle_runtime_ms_by_workload=tuple(
            (
                workflow_count,
                sum(
                    row.selection_runtime_ms
                    for row in rows
                    if row.workflow_count == workflow_count
                    and row.policy_name == "Oracle"
                ),
            )
            for workflow_count in (8, 16)
        ),
    )
    validate_analysis(result)
    return result


def validate_analysis(result: OfflineAnalysisResult) -> None:
    """验证最优性、单调性、满覆盖和基线分化等离线性质。"""
    rows_by_key = {
        (
            row.workflow_count,
            row.budget_checkpoints,
            row.policy_name,
        ): row
        for row in result.rows
    }
    for workflow_count in (8, 16):
        budgets = BUDGETS_BY_WORKFLOW_COUNT[workflow_count]
        flowstate_rows = tuple(
            rows_by_key[(workflow_count, budget, "FlowState")]
            for budget in budgets
        )
        if any(
            following.estimated_recovery_cost_ms
            > current.estimated_recovery_cost_ms + 1e-9
            for current, following in zip(
                flowstate_rows,
                flowstate_rows[1:],
            )
        ):
            raise RuntimeError(
                f"N={workflow_count} 的 FlowState 恢复成本非单调"
            )

        for policy_name in ("FlowState", "Oracle"):
            full = rows_by_key[
                (workflow_count, workflow_count, policy_name)
            ]
            if full.total_recovery_gap != 0:
                raise RuntimeError(
                    f"N={workflow_count} 的 {policy_name} 未达到满覆盖"
                )

        for budget in budgets:
            oracle = rows_by_key[(workflow_count, budget, "Oracle")]
            for policy_name in POLICY_NAMES:
                policy = rows_by_key[
                    (workflow_count, budget, policy_name)
                ]
                if (
                    oracle.estimated_recovery_cost_ms
                    > policy.estimated_recovery_cost_ms + 1e-9
                ):
                    raise RuntimeError(
                        f"N={workflow_count}, K={budget} 的 Oracle 非最优"
                    )

        scenario = build_scenario(workflow_count)
        _validate_factorial_independence(scenario)

    differentiated = any(
        rows_by_key[(workflow_count, budget, "Workflow-Only")]
        .selected_checkpoint_ids
        != rows_by_key[(workflow_count, budget, "Recovery-Only")]
        .selected_checkpoint_ids
        for workflow_count in (8, 16)
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]
    )
    if not differentiated:
        raise RuntimeError("Workflow-Only 与 Recovery-Only 没有产生选择分化")


def write_offline_artifacts(
    result: OfflineAnalysisResult,
    output_directory: Path = _OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """写出固定文件名的 CSV 与 JSON 离线 artifact。"""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "offline_summary.csv"
    json_path = output_directory / "offline_summary.json"
    fieldnames = (
        "workflow_count",
        "budget_checkpoints",
        "budget_ratio",
        "policy_name",
        "selected_checkpoint_ids",
        "total_recovery_gap",
        "mean_recovery_gap_per_request",
        "planning_executable_prefix_ratio",
        "estimated_recovery_cost_ms",
        "selection_runtime_ms",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            values = asdict(row)
            values["selected_checkpoint_ids"] = ";".join(
                row.selected_checkpoint_ids
            )
            writer.writerow(values)

    payload = {
        "schema_version": "flowstate.scalable_multiworkflow.offline.v2",
        "workload_construction": "FIXED BEFORE POLICY EVALUATION",
        "rows": [asdict(row) for row in result.rows],
        "flowstate_oracle_differences": [
            asdict(item)
            for item in result.flowstate_oracle_differences
        ],
        "oracle_runtime_ms_by_workload": dict(
            result.oracle_runtime_ms_by_workload
        ),
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
) -> OfflineAnalysisResult:
    """读取已冻结的离线 JSON，并重建可验证的分析结果。"""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != (
        "flowstate.scalable_multiworkflow.offline.v2"
    ):
        raise ValueError("离线 artifact schema 不匹配")
    rows = tuple(
        OfflineSummaryRow(
            **{
                **row,
                "selected_checkpoint_ids": tuple(
                    row["selected_checkpoint_ids"]
                ),
            }
        )
        for row in payload["rows"]
    )
    differences = tuple(
        FlowStateOracleDifference(**item)
        for item in payload["flowstate_oracle_differences"]
    )
    runtimes = tuple(
        sorted(
            (
                int(workflow_count),
                float(runtime_ms),
            )
            for workflow_count, runtime_ms in payload[
                "oracle_runtime_ms_by_workload"
            ].items()
        )
    )
    result = OfflineAnalysisResult(
        rows=rows,
        flowstate_oracle_differences=differences,
        oracle_runtime_ms_by_workload=runtimes,
    )
    validate_analysis(result)
    return result


def _select_checkpoint_ids(
    policy_name: str,
    scenario: ScalableScenario,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """把六个策略名称分派到现有 evaluation 策略实现。"""
    if policy_name == "FlowState":
        result = GlobalOptimizer(recovery_cost_model).select(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
        )
        return tuple(
            candidate.checkpoint_id for candidate in result.selected
        )
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
    if policy_name == "Recovery-Only":
        return select_recovery_only(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
            recovery_cost_model,
        )
    if policy_name == "Workflow-Only":
        return select_workflow_only(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
        )
    if policy_name == "Oracle":
        return select_oracle(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
            recovery_cost_model,
        )
    raise ValueError(f"未知策略：{policy_name}")


def _build_summary_row(
    scenario: ScalableScenario,
    policy_name: str,
    selected_ids: tuple[str, ...],
    recovery_cost_model: RecoveryCostModel,
    selection_runtime_ms: float,
) -> OfflineSummaryRow:
    """用核心 E、G 与 Phi 计算一个策略的规划指标。"""
    if len(selected_ids) > scenario.metadata.budget_checkpoints:
        raise RuntimeError(f"{policy_name} 超出检查点预算")
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError(f"{policy_name} 返回重复检查点")
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    selected = _resolve_selected(selected_ids, candidates_by_id)
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
        raise RuntimeError("可执行前沿与恢复间隔未能分解全部规划目标")
    return OfflineSummaryRow(
        workflow_count=scenario.metadata.workflow_count,
        budget_checkpoints=scenario.metadata.budget_checkpoints,
        budget_ratio=(
            scenario.metadata.budget_checkpoints
            / scenario.metadata.workflow_count
        ),
        policy_name=policy_name,
        selected_checkpoint_ids=selected_ids,
        total_recovery_gap=sum(gaps),
        mean_recovery_gap_per_request=(
            sum(gaps) / len(scenario.continuations)
        ),
        planning_executable_prefix_ratio=(
            sum(frontiers) / total_target if total_target else 0.0
        ),
        estimated_recovery_cost_ms=sum(
            recovery_cost_model.estimate(gap) for gap in gaps
        ),
        selection_runtime_ms=selection_runtime_ms,
    )


def _resolve_selected(
    selected_ids: Sequence[str],
    candidates_by_id: dict[str, CheckpointCandidate],
) -> tuple[CheckpointCandidate, ...]:
    """解析策略选择并拒绝未知或非驻留检查点。"""
    selected = []
    for checkpoint_id in selected_ids:
        candidate = candidates_by_id.get(checkpoint_id)
        if candidate is None:
            raise RuntimeError(f"策略选择未知检查点：{checkpoint_id}")
        if not candidate.recurrent_resident:
            raise RuntimeError(f"策略选择非驻留检查点：{checkpoint_id}")
        selected.append(candidate)
    return tuple(selected)


def _build_oracle_differences(
    rows: Sequence[OfflineSummaryRow],
) -> tuple[FlowStateOracleDifference, ...]:
    """计算所有规模与预算下的 FlowState/Oracle 目标差距。"""
    rows_by_key = {
        (
            row.workflow_count,
            row.budget_checkpoints,
            row.policy_name,
        ): row
        for row in rows
    }
    differences = []
    for workflow_count in (8, 16):
        for budget in BUDGETS_BY_WORKFLOW_COUNT[workflow_count]:
            flowstate = rows_by_key[
                (workflow_count, budget, "FlowState")
            ]
            oracle = rows_by_key[(workflow_count, budget, "Oracle")]
            absolute_gap = (
                flowstate.estimated_recovery_cost_ms
                - oracle.estimated_recovery_cost_ms
            )
            if absolute_gap < -1e-9:
                raise RuntimeError("FlowState 目标值不能优于 exact Oracle")
            if abs(absolute_gap) <= 1e-9:
                absolute_gap = 0.0
            relative_gap = (
                absolute_gap / oracle.estimated_recovery_cost_ms
                if oracle.estimated_recovery_cost_ms > 0.0
                else 0.0 if absolute_gap == 0.0 else None
            )
            differences.append(
                FlowStateOracleDifference(
                    workflow_count=workflow_count,
                    budget_checkpoints=budget,
                    absolute_objective_gap_ms=absolute_gap,
                    relative_objective_gap=relative_gap,
                    flowstate_exact_optimal=absolute_gap == 0.0,
                )
            )
    return tuple(differences)


def _validate_factorial_independence(
    scenario: ScalableScenario,
) -> None:
    """确认每个锚点与 fanout 组合恰好出现一次。"""
    observed = {
        (workflow.anchor_pos, workflow.pending_fanout)
        for workflow in scenario.metadata.workflows
    }
    expected = {
        (anchor_pos, fanout)
        for anchor_pos in scenario.metadata.anchor_depths
        for fanout in FANOUTS_BY_WORKFLOW_COUNT[
            scenario.metadata.workflow_count
        ]
    }
    if observed != expected or len(observed) != len(
        scenario.metadata.workflows
    ):
        raise RuntimeError("锚点深度与 fanout 未形成完整独立阶乘组合")


def main() -> None:
    """运行完整离线分析并保存固定 artifact。"""
    result = run_offline_analysis()
    csv_path, json_path = write_offline_artifacts(result)
    print(f"离线 CSV：{csv_path}")
    print(f"离线 JSON：{json_path}")


if __name__ == "__main__":
    main()
