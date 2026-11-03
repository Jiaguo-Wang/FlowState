"""等大小检查点的分配求解：严格入口按规模门禁在 Exact 枚举与 Marginal Greedy 间选择，Legacy 入口仅 Greedy。"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .strict_epoch import StrictAllocationEpoch, StrictAllocationPlan, PendingPlan

from .executable_state import recovery_gap
from .recovery_model import RecoveryCostModel
from .state_catalog import CheckpointCandidate, validate_unique_checkpoint_ids
from .workflow import PendingContinuation


_FLOAT_TOLERANCE_MS = 1e-9

# Exact 准入门禁：严格沿用已验证边界（b5d1 合同），只做规模判定，不扩大支持范围。
_EXACT_MAX_CANDIDATES = 12
_EXACT_MAX_PENDINGS = 4
_EXACT_MAX_BUDGET_K = 4
_EXACT_MAX_SUBSETS = 794


def _exact_admission(candidate_count: int, pending_count: int, budget_k: int) -> tuple[bool, dict]:
    """判定当前规模是否落在已验证 Exact 枚举域内；返回 (是否准许, 判定明细)。

    仅规模超限才允许切换 Greedy；本函数不承担输入合法性检查（由 Epoch 合同负责）。
    """
    subsets = sum(
        math.comb(candidate_count, size)
        for size in range(min(candidate_count, budget_k) + 1)
    )
    allowed = (
        candidate_count <= _EXACT_MAX_CANDIDATES
        and pending_count <= _EXACT_MAX_PENDINGS
        and budget_k <= _EXACT_MAX_BUDGET_K
        and subsets <= _EXACT_MAX_SUBSETS
    )
    return allowed, {
        "candidates": candidate_count,
        "pendings": pending_count,
        "budget_k": budget_k,
        "enumerable_subsets": subsets,
    }


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
        # 最近一次 plan_strict 的决策记录（求解器、门禁判定、选集、目标值、纯求解时间），供审计。
        self.last_strict_decision: dict | None = None

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """在给定内存预算内选择能最大幅度降低恢复成本的检查点。"""
        if any(not isinstance(p, PendingContinuation) for p in continuations) or any(
            not isinstance(c, CheckpointCandidate) for c in candidates
        ):
            raise TypeError("Legacy 入口不能混用 Strict Epoch 对象")
        return self._select_candidates(
            candidates, budget_bytes,
            lambda selected: self._recovery_cost(continuations, selected),
        )

    def plan_strict(self, epoch: StrictAllocationEpoch) -> StrictAllocationPlan:
        """严格入口的混合求解：规模门禁内 Exact 枚举，超域走原有 Marginal Greedy。

        仅当规模超出已验证 Exact 准入门禁时才使用 Greedy；Epoch 合同错误、
        输入不完整或求解器异常一律向上抛出，绝不静默回退。Exact 与 Greedy
        共用同一 frontiers 评分闭包（同一冻结收益语义与共同目标 J）。
        """
        if type(epoch) is not StrictAllocationEpoch:
            raise TypeError("严格入口只接受 StrictAllocationEpoch")
        epoch.validate()
        candidates = tuple(c.candidate for c in epoch.candidates)

        def score_ids(selected_ids):
            frontiers = epoch.frontiers(selected_ids)
            return sum(self._recovery_cost_model.estimate(p.target - e, p.target)
                       for p, e in zip(epoch.pendings, frontiers))

        allowed, gate = _exact_admission(
            len(candidates), len(epoch.pendings), epoch.K)
        started_ns = time.perf_counter_ns()
        if allowed:
            ids = self._enumerate_exact_subsets(
                tuple(sorted(c.checkpoint_id for c in candidates)), epoch.K, score_ids)
            solver = "Exact"
            cost_after_ms = score_ids(ids)
            total_benefit_ms = score_ids(()) - cost_after_ms
            if total_benefit_ms < -_FLOAT_TOLERANCE_MS:
                raise ValueError(f"总收益不能为负：{total_benefit_ms} ms")
            if total_benefit_ms < 0.0:
                total_benefit_ms = 0.0
        else:
            size = candidates[0].memory_bytes if candidates else 1
            allocation = self._select_candidates(
                candidates, epoch.K * size,
                lambda selected: score_ids(tuple(c.checkpoint_id for c in selected)))
            ids = tuple(c.checkpoint_id for c in allocation.selected)
            solver = "Marginal Greedy"
            cost_after_ms = allocation.recovery_cost_after_ms
            total_benefit_ms = allocation.total_benefit_ms
        pure_solve_ms = (time.perf_counter_ns() - started_ns) / 1e6
        self.last_strict_decision = {
            "solver": solver,
            "gate_allowed": allowed,
            "gate": gate,
            "selected_ids": list(ids),
            "objective_ms": cost_after_ms,
            "pure_solve_ms": pure_solve_ms,
        }
        frontiers = epoch.frontiers(ids)
        pending = tuple(PendingPlan(p.pending_id, p.target, e, p.target - e,
                         self._recovery_cost_model.estimate(p.target - e, p.target))
                        for p, e in zip(epoch.pendings, frontiers))
        return StrictAllocationPlan(
            epoch.epoch_id, epoch.snapshot_sha256, epoch.content_digest, epoch.K,
            ids, len(ids), pending, cost_after_ms,
            total_benefit_ms,
            tuple(sorted(c.checkpoint_id for c in candidates if c.checkpoint_id not in ids)),
        )

    def _enumerate_exact_subsets(
        self,
        sorted_ids: tuple[str, ...],
        budget_k: int,
        score_ids: Callable[[tuple[str, ...]], float],
    ) -> tuple[str, ...]:
        """迁移自已验证 Exact 实现：穷举 |S|≤K 子集按共同目标评分。

        同分规则与既有合同一致：容差 1e-9 内取最少项、再按 ID 字典序。
        """
        scored = [
            (score_ids(subset), subset)
            for size in range(min(budget_k, len(sorted_ids)) + 1)
            for subset in itertools.combinations(sorted_ids, size)
        ]
        best_ms = min(value for value, _ in scored)
        return min(
            (subset for value, subset in scored
             if abs(value - best_ms) <= _FLOAT_TOLERANCE_MS),
            key=lambda subset: (len(subset), subset),
        )

    def _select_candidates(
        self, candidates: Sequence[CheckpointCandidate], budget_bytes: int,
        cost: Callable[[Sequence[CheckpointCandidate]], float],
    ) -> AllocationResult:
        """两种入口共用原有 Greedy 循环，评分函数由入口明确指定。"""
        if budget_bytes < 0:
            raise ValueError("内存预算必须大于等于零")
        validate_unique_checkpoint_ids(candidates)

        selected: tuple[CheckpointCandidate, ...] = ()
        recovery_cost_before_ms = cost(selected)

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
                cost_after_ms = cost(selected + (candidate,))
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
