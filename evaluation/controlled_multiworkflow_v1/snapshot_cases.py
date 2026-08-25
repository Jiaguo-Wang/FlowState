"""规划固定决策时点下的隔离评估用例与期望结果。"""

from __future__ import annotations

from dataclasses import dataclass

from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation

from .policies import (
    select_equal_share,
    select_global_lru,
    select_recovery_only,
)
from .scenario import ControlledScenario, build_scenario


POLICY_NAMES = (
    "FlowState",
    "Global-LRU",
    "Equal-Share",
    "Recovery-Only",
)


@dataclass(frozen=True)
class SnapshotEvaluationCase:
    """描述一个策略与一个待续请求组成的独立运行时用例。"""

    policy_name: str
    continuation_id: str
    expected_selected_ids: tuple[str, ...]
    workflow_id: str
    expected_recovery_gap: int
    scenario_continuation_id: str


@dataclass(frozen=True)
class PolicyPlanningSummary:
    """汇总一个策略在固定快照上的规划阶段结果。"""

    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    recovery_gaps: tuple[tuple[str, int], ...]
    total_recovery_gap: int
    estimated_recovery_cost_ms: float


def build_snapshot_cases(
    scenario: ControlledScenario | None = None,
    recovery_cost_model: RecoveryCostModel | None = None,
) -> tuple[SnapshotEvaluationCase, ...]:
    """为四个策略和七个待续请求构造二十八个隔离用例。"""
    active_scenario = scenario or build_scenario()
    summaries = build_planning_summaries(
        active_scenario,
        recovery_cost_model,
    )
    continuations = _ordered_continuations(active_scenario)

    cases = []
    for summary in summaries:
        gaps_by_id = dict(summary.recovery_gaps)
        for continuation_id, continuation in continuations:
            cases.append(
                SnapshotEvaluationCase(
                    policy_name=summary.policy_name,
                    continuation_id=continuation_id,
                    expected_selected_ids=summary.selected_checkpoint_ids,
                    workflow_id=continuation.workflow_id,
                    expected_recovery_gap=gaps_by_id[continuation_id],
                    scenario_continuation_id=continuation.continuation_id,
                )
            )
    return tuple(cases)


def build_planning_summaries(
    scenario: ControlledScenario | None = None,
    recovery_cost_model: RecoveryCostModel | None = None,
) -> tuple[PolicyPlanningSummary, ...]:
    """计算四个策略在同一逻辑快照上的选择、间隔与估计成本。"""
    active_scenario = scenario or build_scenario()
    model = recovery_cost_model or RecoveryCostModel()
    selections = _build_policy_selections(active_scenario, model)
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in active_scenario.candidates
    }

    summaries = []
    for policy_name in POLICY_NAMES:
        selected_ids = selections[policy_name]
        selected = _resolve_selected_candidates(
            selected_ids,
            candidates_by_id,
        )
        gaps = tuple(
            (
                continuation_id,
                recovery_gap(continuation, selected),
            )
            for continuation_id, continuation in _ordered_continuations(
                active_scenario
            )
        )
        summaries.append(
            PolicyPlanningSummary(
                policy_name=policy_name,
                selected_checkpoint_ids=selected_ids,
                recovery_gaps=gaps,
                total_recovery_gap=sum(gap for _, gap in gaps),
                estimated_recovery_cost_ms=sum(
                    model.estimate(gap) for _, gap in gaps
                ),
            )
        )
    return tuple(summaries)


def _build_policy_selections(
    scenario: ControlledScenario,
    recovery_cost_model: RecoveryCostModel,
) -> dict[str, tuple[str, ...]]:
    """调用冻结策略实现，生成同一快照上的四组选择。"""
    flowstate_result = GlobalOptimizer(recovery_cost_model).select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )
    return {
        "FlowState": tuple(
            candidate.checkpoint_id
            for candidate in flowstate_result.selected
        ),
        "Global-LRU": select_global_lru(
            scenario.candidates,
            scenario.metadata.checkpoint_recency,
            scenario.budget_bytes,
        ),
        "Equal-Share": select_equal_share(
            scenario.continuations,
            scenario.candidates,
            scenario.metadata.workflow_order,
            scenario.budget_bytes,
        ),
        "Recovery-Only": select_recovery_only(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
            recovery_cost_model,
        ),
    }


def _resolve_selected_candidates(
    selected_ids: tuple[str, ...],
    candidates_by_id: dict[str, CheckpointCandidate],
) -> tuple[CheckpointCandidate, ...]:
    """把策略输出的标识解析为核心候选对象。"""
    missing = sorted(
        checkpoint_id
        for checkpoint_id in selected_ids
        if checkpoint_id not in candidates_by_id
    )
    if missing:
        raise ValueError(f"策略选择了未知检查点：{', '.join(missing)}")
    return tuple(candidates_by_id[checkpoint_id] for checkpoint_id in selected_ids)


def _evaluation_continuation_id(
    continuation: PendingContinuation,
    scenario: ControlledScenario,
) -> str:
    """为单分支工作流生成协议中使用的简洁请求标识。"""
    workflow = next(
        (
            workflow
            for workflow in scenario.metadata.workflows
            if workflow.workflow_id == continuation.workflow_id
        ),
        None,
    )
    if workflow is None:
        raise ValueError(f"缺少工作流元数据：{continuation.workflow_id}")
    if workflow.pending_fanout == 1:
        return continuation.workflow_id
    return continuation.continuation_id


def _ordered_continuations(
    scenario: ControlledScenario,
) -> tuple[tuple[str, PendingContinuation], ...]:
    """按公开请求标识排序，消除输入序列顺序对计划的影响。"""
    continuations = tuple(
        (
            _evaluation_continuation_id(continuation, scenario),
            continuation,
        )
        for continuation in scenario.continuations
    )
    continuation_ids = tuple(
        continuation_id for continuation_id, _ in continuations
    )
    if len(set(continuation_ids)) != len(continuation_ids):
        raise ValueError("隔离评估中的待续请求标识必须唯一")
    return tuple(sorted(continuations, key=lambda item: item[0]))
