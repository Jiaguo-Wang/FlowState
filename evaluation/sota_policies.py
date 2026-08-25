"""定义可与 FlowState 共享候选、预算和执行器的 SOTA-style 策略。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

from flowstate.state_catalog import (
    CheckpointCandidate,
    is_compatible,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


@dataclass(frozen=True)
class PolicySelection:
    """记录一次策略选择的确定性结果。"""

    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    budget_k: int


class KVFlowStylePolicy:
    """实现 KVFlow-style adaptation 的未来执行距离保留原则。"""

    policy_name = "KVFlow-style"

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_k: int,
        steps_to_execution_by_continuation: Mapping[str, int],
        last_access_by_checkpoint: Mapping[str, float],
    ) -> PolicySelection:
        """按未来执行步数优先、时序次优先选择常驻循环状态。"""
        capacity = _validate_budget_k(budget_k)
        validate_unique_checkpoint_ids(candidates)
        _validate_steps_to_execution(
            continuations,
            steps_to_execution_by_continuation,
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.recurrent_resident
        )
        recencies = _validate_last_access_metadata(
            eligible,
            last_access_by_checkpoint,
        )
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                _kvflow_priority(
                    candidate,
                    continuations,
                    steps_to_execution_by_continuation,
                ),
                -recencies[candidate.checkpoint_id],
                candidate.checkpoint_id,
            ),
        )
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in ordered[: min(capacity, len(ordered))]
        )
        return PolicySelection(
            policy_name=self.policy_name,
            selected_checkpoint_ids=selected_ids,
            budget_k=capacity,
        )

    def priority(
        self,
        candidate: CheckpointCandidate,
        continuations: Sequence[PendingContinuation],
        steps_to_execution_by_continuation: Mapping[str, int],
    ) -> float:
        """返回候选在当前未来执行快照中的保留优先级。"""
        _validate_steps_to_execution(
            continuations,
            steps_to_execution_by_continuation,
        )
        return _kvflow_priority(
            candidate,
            continuations,
            steps_to_execution_by_continuation,
        )


class MarconiStylePolicy:
    """实现 Marconi-style FLOP-aware eviction adaptation。"""

    policy_name = "Marconi-style"

    def select(
        self,
        candidates: Sequence[CheckpointCandidate],
        budget_k: int,
        last_access_by_checkpoint: Mapping[str, float],
        flop_saved_by_checkpoint: Mapping[str, float],
        alpha: float,
    ) -> PolicySelection:
        """按归一化时序与增量计算效率的组合 utility 选择状态。"""
        capacity = _validate_budget_k(budget_k)
        validate_unique_checkpoint_ids(candidates)
        validated_alpha = _validate_finite_number(
            alpha,
            "alpha",
            nonnegative=True,
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.recurrent_resident
        )
        metrics = _build_marconi_metrics(
            eligible,
            last_access_by_checkpoint,
            flop_saved_by_checkpoint,
            validated_alpha,
        )
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                -metrics[candidate.checkpoint_id],
                candidate.checkpoint_id,
            ),
        )
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in ordered[: min(capacity, len(ordered))]
        )
        return PolicySelection(
            policy_name=self.policy_name,
            selected_checkpoint_ids=selected_ids,
            budget_k=capacity,
        )


def _validate_budget_k(budget_k: int) -> int:
    """验证以检查点数量表示的预算。"""
    if isinstance(budget_k, bool) or not isinstance(budget_k, int):
        raise ValueError("budget_k 必须是整数")
    if budget_k < 0:
        raise ValueError("budget_k 必须大于等于零")
    return budget_k


def _validate_steps_to_execution(
    continuations: Sequence[PendingContinuation],
    steps_by_continuation: Mapping[str, int],
) -> None:
    """验证所有活动待续请求都有合法的未来执行步数。"""
    missing = sorted(
        continuation.continuation_id
        for continuation in continuations
        if continuation.continuation_id not in steps_by_continuation
    )
    if missing:
        raise ValueError(
            "缺少 steps-to-execution 元数据：" + ", ".join(missing)
        )

    for continuation in continuations:
        continuation_id = continuation.continuation_id
        value = steps_by_continuation[continuation_id]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"待续请求 {continuation_id} 的 steps-to-execution 必须是整数"
            )
        if value < 0:
            raise ValueError(
                f"待续请求 {continuation_id} 的 steps-to-execution 必须大于等于零"
            )


def _kvflow_priority(
    candidate: CheckpointCandidate,
    continuations: Sequence[PendingContinuation],
    steps_by_continuation: Mapping[str, int],
) -> float:
    """计算候选所覆盖待续请求中的最小未来执行步数。"""
    compatible_steps = tuple(
        steps_by_continuation[continuation.continuation_id]
        for continuation in continuations
        if is_compatible(candidate, continuation)
    )
    if not compatible_steps:
        return math.inf
    return float(min(compatible_steps))


def _validate_last_access_metadata(
    candidates: Sequence[CheckpointCandidate],
    last_access_by_checkpoint: Mapping[str, float],
) -> dict[str, float]:
    """验证 eligible 候选具有有限的最近访问 metadata。"""
    recencies: dict[str, float] = {}
    for candidate in candidates:
        checkpoint_id = candidate.checkpoint_id
        if checkpoint_id not in last_access_by_checkpoint:
            raise ValueError(f"检查点 {checkpoint_id} 缺少 last_access 元数据")
        recencies[checkpoint_id] = _validate_finite_number(
            last_access_by_checkpoint[checkpoint_id],
            f"检查点 {checkpoint_id} 的 last_access",
        )
    return recencies


def _build_marconi_metrics(
    candidates: Sequence[CheckpointCandidate],
    last_access_by_checkpoint: Mapping[str, float],
    flop_saved_by_checkpoint: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    """计算 eligible 候选的归一化 Marconi utility。"""
    if not candidates:
        return {}

    recencies: dict[str, float] = {}
    raw_efficiencies: dict[str, float] = {}
    for candidate in candidates:
        checkpoint_id = candidate.checkpoint_id
        if checkpoint_id not in last_access_by_checkpoint:
            raise ValueError(f"检查点 {checkpoint_id} 缺少 last_access 元数据")
        if checkpoint_id not in flop_saved_by_checkpoint:
            raise ValueError(f"检查点 {checkpoint_id} 缺少 flop_saved 元数据")
        if candidate.memory_bytes <= 0:
            raise ValueError(f"检查点 {checkpoint_id} 的 memory_bytes 必须大于零")

        recencies[checkpoint_id] = _validate_finite_number(
            last_access_by_checkpoint[checkpoint_id],
            f"检查点 {checkpoint_id} 的 last_access",
        )
        flop_saved = _validate_finite_number(
            flop_saved_by_checkpoint[checkpoint_id],
            f"检查点 {checkpoint_id} 的 flop_saved",
            nonnegative=True,
        )
        raw_efficiencies[checkpoint_id] = (
            flop_saved / candidate.memory_bytes
        )

    normalized_recencies = _min_max_normalize(recencies)
    normalized_efficiencies = _min_max_normalize(raw_efficiencies)
    return {
        candidate.checkpoint_id: (
            normalized_recencies[candidate.checkpoint_id]
            + alpha * normalized_efficiencies[candidate.checkpoint_id]
        )
        for candidate in candidates
    }


def _validate_finite_number(
    value: float,
    field_name: str,
    *,
    nonnegative: bool = False,
) -> float:
    """验证 metadata 数值有限，并按需要求非负。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} 必须是有限数值")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} 必须是有限数值")
    if nonnegative and converted < 0.0:
        raise ValueError(f"{field_name} 必须大于等于零")
    return converted


def _min_max_normalize(values: Mapping[str, float]) -> dict[str, float]:
    """在当前 eligible 候选内执行确定性的最小最大归一化。"""
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {checkpoint_id: 0.0 for checkpoint_id in values}
    scale = maximum - minimum
    return {
        checkpoint_id: (value - minimum) / scale
        for checkpoint_id, value in values.items()
    }
