"""定义受控多工作流实验使用的冻结基线策略。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from flowstate.executable_state import recovery_gap
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import (
    CheckpointCandidate,
    is_compatible,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation

from .scenario import CheckpointRecency


def select_global_lru(
    candidates: Sequence[CheckpointCandidate],
    checkpoint_recency: Sequence[CheckpointRecency],
    budget_bytes: int,
) -> tuple[str, ...]:
    """在全局预算内保留最近访问的常驻循环状态。"""
    eligible, capacity = _eligible_candidates(candidates, budget_bytes)
    recency_by_id = _index_recency(checkpoint_recency)
    missing = sorted(
        candidate.checkpoint_id
        for candidate in eligible
        if candidate.checkpoint_id not in recency_by_id
    )
    if missing:
        raise ValueError(f"缺少检查点时序元数据：{', '.join(missing)}")

    ordered = sorted(
        eligible,
        key=lambda candidate: (
            -recency_by_id[candidate.checkpoint_id].last_access_order,
            -recency_by_id[candidate.checkpoint_id].creation_order,
            candidate.checkpoint_id,
        ),
    )
    return tuple(
        candidate.checkpoint_id for candidate in ordered[:capacity]
    )


def select_equal_share(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    workflow_order: Sequence[str],
    budget_bytes: int,
) -> tuple[str, ...]:
    """按公开工作流顺序轮转分配，并在工作流内优先最深兼容状态。"""
    eligible, capacity = _eligible_candidates(candidates, budget_bytes)
    ordered_workflows = _validate_workflow_order(workflow_order)
    candidates_by_workflow: dict[str, list[CheckpointCandidate]] = {
        workflow_id: [] for workflow_id in ordered_workflows
    }

    for candidate in eligible:
        if candidate.workflow_id not in candidates_by_workflow:
            raise ValueError(
                "工作流顺序缺少候选所属工作流："
                f"{candidate.workflow_id}"
            )
        if any(
            is_compatible(candidate, continuation)
            for continuation in continuations
        ):
            candidates_by_workflow[candidate.workflow_id].append(candidate)

    for workflow_candidates in candidates_by_workflow.values():
        workflow_candidates.sort(
            key=lambda candidate: (
                -candidate.token_pos,
                candidate.checkpoint_id,
            )
        )

    selected: list[str] = []
    round_index = 0
    while len(selected) < capacity:
        added = False
        for workflow_id in ordered_workflows:
            workflow_candidates = candidates_by_workflow[workflow_id]
            if round_index >= len(workflow_candidates):
                continue
            selected.append(workflow_candidates[round_index].checkpoint_id)
            added = True
            if len(selected) == capacity:
                break
        if not added:
            break
        round_index += 1

    return tuple(selected)


def select_recovery_only(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    budget_bytes: int,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """按固定的单分支最大恢复收益排序，不累计待续分支覆盖。"""
    eligible, capacity = _eligible_candidates(candidates, budget_bytes)
    scored = tuple(
        (
            _maximum_single_continuation_benefit(
                candidate,
                continuations,
                recovery_cost_model,
            ),
            candidate,
        )
        for candidate in eligible
    )
    ordered = sorted(
        scored,
        key=lambda item: (-item[0], item[1].checkpoint_id),
    )
    return tuple(
        candidate.checkpoint_id
        for _, candidate in ordered[:capacity]
    )


def select_workflow_only(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    budget_bytes: int,
) -> tuple[str, ...]:
    """只按尚未覆盖的兼容待续请求数量执行集合依赖贪心。"""
    eligible, capacity = _eligible_candidates(candidates, budget_bytes)
    ordered_candidates = tuple(
        sorted(eligible, key=lambda candidate: candidate.checkpoint_id)
    )
    covered: set[int] = set()
    selected: list[CheckpointCandidate] = []

    while len(selected) < capacity:
        best_candidate = None
        best_new_coverage: tuple[int, ...] = ()
        for candidate in ordered_candidates:
            if candidate in selected:
                continue
            new_coverage = tuple(
                index
                for index, continuation in enumerate(continuations)
                if index not in covered
                and is_compatible(candidate, continuation)
            )
            if len(new_coverage) > len(best_new_coverage):
                best_candidate = candidate
                best_new_coverage = new_coverage

        if best_candidate is None or not best_new_coverage:
            break
        selected.append(best_candidate)
        covered.update(best_new_coverage)

    return tuple(candidate.checkpoint_id for candidate in selected)


def select_oracle(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    budget_bytes: int,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[str, ...]:
    """精确搜索预算内恢复成本最低的常驻检查点子集。"""
    eligible, capacity = _eligible_candidates(candidates, budget_bytes)
    ordered_candidates = tuple(
        sorted(eligible, key=lambda candidate: candidate.checkpoint_id)
    )
    best_ids: tuple[str, ...] | None = None
    best_cost: float | None = None

    for subset_size in range(capacity + 1):
        for subset in combinations(ordered_candidates, subset_size):
            checkpoint_ids = tuple(
                candidate.checkpoint_id for candidate in subset
            )
            cost = sum(
                recovery_cost_model.estimate(
                    recovery_gap(continuation, subset),
                    continuation.planning_target,
                )
                for continuation in continuations
            )
            if (
                best_cost is None
                or cost < best_cost - 1e-9
                or (
                    abs(cost - best_cost) <= 1e-9
                    and (
                        best_ids is None
                        or checkpoint_ids < best_ids
                    )
                )
            ):
                best_ids = checkpoint_ids
                best_cost = cost

    return best_ids or ()


def _eligible_candidates(
    candidates: Sequence[CheckpointCandidate],
    budget_bytes: int,
) -> tuple[tuple[CheckpointCandidate, ...], int]:
    """验证等大小预算并返回本轮可由策略保留的候选。"""
    if budget_bytes < 0:
        raise ValueError("内存预算必须大于等于零")
    validate_unique_checkpoint_ids(candidates)
    eligible = tuple(
        candidate for candidate in candidates if candidate.recurrent_resident
    )
    if not eligible:
        return (), 0

    checkpoint_size = eligible[0].memory_bytes
    for candidate in eligible[1:]:
        if candidate.memory_bytes != checkpoint_size:
            raise ValueError("受控基线仅支持等大小检查点")
    return eligible, min(len(eligible), budget_bytes // checkpoint_size)


def _index_recency(
    checkpoint_recency: Sequence[CheckpointRecency],
) -> dict[str, CheckpointRecency]:
    """建立时序元数据索引，并拒绝重复检查点标识。"""
    indexed: dict[str, CheckpointRecency] = {}
    for item in checkpoint_recency:
        if item.checkpoint_id in indexed:
            raise ValueError(f"检查点时序元数据重复：{item.checkpoint_id}")
        indexed[item.checkpoint_id] = item
    return indexed


def _validate_workflow_order(workflow_order: Sequence[str]) -> tuple[str, ...]:
    """验证公开工作流顺序没有重复标识。"""
    ordered = tuple(workflow_order)
    if len(set(ordered)) != len(ordered):
        raise ValueError("工作流确定性顺序不能包含重复标识")
    return ordered


def _maximum_single_continuation_benefit(
    candidate: CheckpointCandidate,
    continuations: Sequence[PendingContinuation],
    recovery_cost_model: RecoveryCostModel,
) -> float:
    """计算候选对任一单独待续分支的最大固定恢复收益。"""
    maximum_benefit = 0.0
    for continuation in continuations:
        gap_before = recovery_gap(continuation, ())
        gap_after = recovery_gap(continuation, (candidate,))
        benefit = recovery_cost_model.estimate(
            gap_before,
            continuation.planning_target,
        ) - recovery_cost_model.estimate(
            gap_after,
            continuation.planning_target,
        )
        if benefit < -1e-9:
            raise ValueError(
                f"检查点 {candidate.checkpoint_id} 的单分支恢复收益为负"
            )
        maximum_benefit = max(maximum_benefit, benefit)
    return maximum_benefit
