"""计算受控多工作流场景的规划阶段预算扫描。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate

from .policies import (
    select_equal_share,
    select_global_lru,
    select_recovery_only,
)
from .scenario import ControlledScenario, build_scenario
from .snapshot_cases import POLICY_NAMES


DEFAULT_BUDGET_CHECKPOINTS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class BudgetSweepRow:
    """记录一个预算与一个策略对应的规划阶段指标。"""

    budget_checkpoints: int
    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    total_recovery_gap: int
    mean_recovery_gap_per_request: float
    planning_executable_prefix_ratio: float
    estimated_recovery_cost_ms: float


@dataclass(frozen=True)
class FlowStateBaselineComparison:
    """记录同一预算下 FlowState 相对一个基线的差值。"""

    budget_checkpoints: int
    baseline_policy_name: str
    absolute_gap_reduction: int
    relative_gap_reduction: float | None
    estimated_recovery_cost_reduction_ms: float


@dataclass(frozen=True)
class BudgetSweepResult:
    """汇总完整预算扫描和逐基线对比。"""

    rows: tuple[BudgetSweepRow, ...]
    comparisons: tuple[FlowStateBaselineComparison, ...]


def build_budget_sweep(
    scenario: ControlledScenario | None = None,
    recovery_cost_model: RecoveryCostModel | None = None,
    budget_checkpoints: Sequence[int] = DEFAULT_BUDGET_CHECKPOINTS,
) -> BudgetSweepResult:
    """对指定 K 值计算四个策略的离线规划结果。"""
    active_scenario = scenario or build_scenario()
    model = recovery_cost_model or RecoveryCostModel()
    k_values = tuple(int(value) for value in budget_checkpoints)
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("预算检查点数量必须是非空的正整数序列")
    if len(set(k_values)) != len(k_values):
        raise ValueError("预算检查点数量不能重复")

    rows = tuple(
        _build_row(
            budget_checkpoints=k,
            policy_name=policy_name,
            scenario=active_scenario,
            recovery_cost_model=model,
        )
        for k in sorted(k_values)
        for policy_name in POLICY_NAMES
    )
    return BudgetSweepResult(
        rows=rows,
        comparisons=_build_comparisons(rows),
    )


def format_sanity_table(result: BudgetSweepResult) -> str:
    """把预算扫描结果格式化为紧凑的 Markdown 表格。"""
    lines = [
        "| K | Policy | Selected | Total Gap | Planning EPR | Estimated Cost |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in result.rows:
        selected = ", ".join(row.selected_checkpoint_ids)
        lines.append(
            f"| {row.budget_checkpoints} | {row.policy_name} | "
            f"{selected} | {row.total_recovery_gap} | "
            f"{row.planning_executable_prefix_ratio:.4f} | "
            f"{row.estimated_recovery_cost_ms:.3f} ms |"
        )
    return "\n".join(lines)


def _build_row(
    *,
    budget_checkpoints: int,
    policy_name: str,
    scenario: ControlledScenario,
    recovery_cost_model: RecoveryCostModel,
) -> BudgetSweepRow:
    """计算一个预算与策略组合的选择及规划指标。"""
    budget_bytes = (
        budget_checkpoints * scenario.metadata.checkpoint_size_bytes
    )
    selected_ids = _select_checkpoint_ids(
        policy_name=policy_name,
        scenario=scenario,
        budget_bytes=budget_bytes,
        recovery_cost_model=recovery_cost_model,
    )
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
        raise RuntimeError("规划前沿与恢复间隔未能分解全部 planning target")

    return BudgetSweepRow(
        budget_checkpoints=budget_checkpoints,
        policy_name=policy_name,
        selected_checkpoint_ids=selected_ids,
        total_recovery_gap=sum(gaps),
        mean_recovery_gap_per_request=(
            sum(gaps) / len(scenario.continuations)
        ),
        planning_executable_prefix_ratio=(
            sum(frontiers) / total_target if total_target > 0 else 0.0
        ),
        estimated_recovery_cost_ms=sum(
            recovery_cost_model.estimate(gap) for gap in gaps
        ),
    )


def _select_checkpoint_ids(
    *,
    policy_name: str,
    scenario: ControlledScenario,
    budget_bytes: int,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """调用现有 FlowState 或冻结基线策略实现。"""
    if policy_name == "FlowState":
        allocation = GlobalOptimizer(recovery_cost_model).select(
            scenario.continuations,
            scenario.candidates,
            budget_bytes,
        )
        return tuple(
            candidate.checkpoint_id for candidate in allocation.selected
        )
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
    raise ValueError(f"未知策略：{policy_name}")


def _resolve_selected(
    selected_ids: tuple[str, ...],
    candidates_by_id: dict[str, CheckpointCandidate],
) -> tuple[CheckpointCandidate, ...]:
    """解析并验证策略选择的候选标识。"""
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("策略选择包含重复检查点标识")
    selected = []
    for checkpoint_id in selected_ids:
        candidate = candidates_by_id.get(checkpoint_id)
        if candidate is None:
            raise ValueError(f"策略选择了未知检查点：{checkpoint_id}")
        if not candidate.recurrent_resident:
            raise ValueError(f"策略选择了非驻留检查点：{checkpoint_id}")
        selected.append(candidate)
    return tuple(selected)


def _build_comparisons(
    rows: Sequence[BudgetSweepRow],
) -> tuple[FlowStateBaselineComparison, ...]:
    """计算每个 K 下 FlowState 相对三个基线的 gap 与成本下降。"""
    rows_by_key = {
        (row.budget_checkpoints, row.policy_name): row for row in rows
    }
    k_values = sorted({row.budget_checkpoints for row in rows})
    comparisons = []
    for k in k_values:
        flowstate = rows_by_key[(k, "FlowState")]
        for baseline_name in POLICY_NAMES[1:]:
            baseline = rows_by_key[(k, baseline_name)]
            absolute_gap_reduction = (
                baseline.total_recovery_gap
                - flowstate.total_recovery_gap
            )
            relative_gap_reduction = (
                absolute_gap_reduction / baseline.total_recovery_gap
                if baseline.total_recovery_gap > 0
                else None
            )
            comparisons.append(
                FlowStateBaselineComparison(
                    budget_checkpoints=k,
                    baseline_policy_name=baseline_name,
                    absolute_gap_reduction=absolute_gap_reduction,
                    relative_gap_reduction=relative_gap_reduction,
                    estimated_recovery_cost_reduction_ms=(
                        baseline.estimated_recovery_cost_ms
                        - flowstate.estimated_recovery_cost_ms
                    ),
                )
            )
    return tuple(comparisons)


def main() -> None:
    """打印由实际策略与模型计算出的离线 sanity 表。"""
    print(format_sanity_table(build_budget_sweep()))


if __name__ == "__main__":
    main()
