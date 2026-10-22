"""为等大小检查点执行全局贪心分配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .executable_state import recovery_gap
from .recovery_model import RecoveryCostModel
from .state_catalog import CheckpointCandidate, validate_unique_checkpoint_ids
from .workflow import PendingContinuation


_FLOAT_TOLERANCE_MS = 1e-9


@dataclass(frozen=True)
class AllocationResult:
    """记录一次检查点分配的选择结果与恢复成本变化。"""

    selected: tuple[CheckpointCandidate, ...]
    total_benefit_ms: float
    recovery_cost_before_ms: float
    recovery_cost_after_ms: float
    used_bytes: int


class GlobalOptimizer:
    """在等大小检查点约束下按边际收益执行全局贪心选择。"""

    def __init__(self, recovery_cost_model: RecoveryCostModel) -> None:
        self._recovery_cost_model = recovery_cost_model

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """在给定内存预算内选择能最大幅度降低恢复成本的检查点。"""
        if budget_bytes < 0:
            raise ValueError("内存预算必须大于等于零")
        validate_unique_checkpoint_ids(candidates)

        selected: tuple[CheckpointCandidate, ...] = ()
        recovery_cost_before_ms = self._recovery_cost(continuations, selected)

        eligible = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.recurrent_resident
                ),
                key=lambda candidate: candidate.checkpoint_id,
            )
        )
        if not eligible:
            return self._build_result(
                selected,
                recovery_cost_before_ms,
                recovery_cost_before_ms,
            )

        checkpoint_size = eligible[0].memory_bytes
        for candidate in eligible[1:]:
            if candidate.memory_bytes != checkpoint_size:
                raise ValueError(
                    "当前版本只支持等大小检查点："
                    f"{eligible[0].checkpoint_id} 为 {checkpoint_size} 字节，"
                    f"{candidate.checkpoint_id} 为 {candidate.memory_bytes} 字节"
                )

        capacity = budget_bytes // checkpoint_size
        if capacity == 0:
            return self._build_result(
                selected,
                recovery_cost_before_ms,
                recovery_cost_before_ms,
            )

        remaining = list(eligible)
        current_cost_ms = recovery_cost_before_ms

        for _ in range(min(capacity, len(remaining))):
            best_index = None
            best_gain_ms = None
            best_cost_after_ms = None

            for index, candidate in enumerate(remaining):
                cost_after_ms = self._recovery_cost(
                    continuations,
                    selected + (candidate,),
                )
                gain_ms = current_cost_ms - cost_after_ms
                if gain_ms < -_FLOAT_TOLERANCE_MS:
                    raise ValueError(
                        "检查点边际收益不能为负："
                        f"{candidate.checkpoint_id} 的收益为 {gain_ms} ms"
                    )
                if gain_ms < 0.0:
                    gain_ms = 0.0

                if best_index is None:
                    best_index = index
                    best_gain_ms = gain_ms
                    best_cost_after_ms = cost_after_ms
                    continue

                assert best_gain_ms is not None
                best_candidate = remaining[best_index]
                gain_is_greater = gain_ms > best_gain_ms + _FLOAT_TOLERANCE_MS
                gain_is_tied = abs(gain_ms - best_gain_ms) <= _FLOAT_TOLERANCE_MS
                if gain_is_greater or (
                    gain_is_tied
                    and candidate.checkpoint_id < best_candidate.checkpoint_id
                ):
                    best_index = index
                    best_gain_ms = gain_ms
                    best_cost_after_ms = cost_after_ms

            if best_index is None or best_gain_ms is None:
                break
            if best_gain_ms <= 0.0:
                break

            selected = selected + (remaining.pop(best_index),)
            assert best_cost_after_ms is not None
            current_cost_ms = best_cost_after_ms

        return self._build_result(
            selected,
            recovery_cost_before_ms,
            current_cost_ms,
        )

    def _recovery_cost(
        self,
        continuations: Sequence[PendingContinuation],
        selected: Sequence[CheckpointCandidate],
    ) -> float:
        """计算所有待续分支在当前选择下的总恢复成本。"""
        return sum(
            self._recovery_cost_model.estimate(
                recovery_gap(continuation, selected),
                continuation.planning_target,
            )
            for continuation in continuations
        )

    @staticmethod
    def _build_result(
        selected: tuple[CheckpointCandidate, ...],
        recovery_cost_before_ms: float,
        recovery_cost_after_ms: float,
    ) -> AllocationResult:
        """根据选择和成本构造不可变分配结果。"""
        total_benefit_ms = recovery_cost_before_ms - recovery_cost_after_ms
        if total_benefit_ms < -_FLOAT_TOLERANCE_MS:
            raise ValueError(f"总收益不能为负：{total_benefit_ms} ms")
        if total_benefit_ms < 0.0:
            total_benefit_ms = 0.0

        return AllocationResult(
            selected=selected,
            total_benefit_ms=total_benefit_ms,
            recovery_cost_before_ms=recovery_cost_before_ms,
            recovery_cost_after_ms=recovery_cost_after_ms,
            used_bytes=sum(candidate.memory_bytes for candidate in selected),
        )
