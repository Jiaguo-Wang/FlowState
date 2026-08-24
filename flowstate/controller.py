"""连接全局优化结果与运行时状态操作。"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .adapters.sglang import RuntimeCheckpointHandle
from .optimizer import AllocationResult, GlobalOptimizer
from .state_catalog import CheckpointCandidate, validate_unique_checkpoint_ids
from .workflow import PendingContinuation


class RuntimeAdapter(Protocol):
    """描述控制器执行状态驱逐所需的最小运行时能力。"""

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """仅驱逐句柄指向的 Mamba 状态。"""


class ReconcileExecutionError(RuntimeError):
    """表示协调过程中已有部分驱逐成功、随后某个驱逐失败。"""

    def __init__(
        self,
        completed_evictions: tuple[str, ...],
        failed_checkpoint_id: str,
        cause: Exception,
    ) -> None:
        self.completed_evictions = completed_evictions
        self.failed_checkpoint_id = failed_checkpoint_id
        self.cause = cause
        completed = ", ".join(completed_evictions) or "无"
        super().__init__(
            "状态协调执行失败："
            f"已完成驱逐 {completed}；"
            f"失败检查点 {failed_checkpoint_id}；"
            f"原因：{cause}"
        )


class StateController:
    """将优化器选择转换为确定性的保留与驱逐动作。"""

    def __init__(
        self,
        optimizer: GlobalOptimizer,
        runtime_adapter: RuntimeAdapter,
    ) -> None:
        self._optimizer = optimizer
        self._runtime_adapter = runtime_adapter

    def reconcile(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        handles: Mapping[str, RuntimeCheckpointHandle],
        budget_bytes: int,
    ) -> AllocationResult:
        """根据本轮事实快照选择状态，并驱逐未被保留的驻留状态。

        每次调用传入的 ``recurrent_resident`` 必须代表当前决策时点的事实；
        控制器不保存跨轮次的长期驻留状态。
        """
        validate_unique_checkpoint_ids(candidates)
        self._validate_handles(handles)
        allocation = self._optimizer.select(
            continuations,
            candidates,
            budget_bytes,
        )
        selected_ids = {
            candidate.checkpoint_id for candidate in allocation.selected
        }
        eviction_candidates = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.recurrent_resident
                    and candidate.checkpoint_id not in selected_ids
                ),
                key=lambda candidate: candidate.checkpoint_id,
            )
        )

        missing_handle_ids = tuple(
            candidate.checkpoint_id
            for candidate in eviction_candidates
            if candidate.checkpoint_id not in handles
        )
        if missing_handle_ids:
            missing = ", ".join(missing_handle_ids)
            raise ValueError(f"待驱逐检查点缺少运行时句柄：{missing}")

        completed_evictions = []
        for candidate in eviction_candidates:
            try:
                self._runtime_adapter.evict_mamba_only(
                    handles[candidate.checkpoint_id]
                )
            except Exception as cause:
                raise ReconcileExecutionError(
                    completed_evictions=tuple(completed_evictions),
                    failed_checkpoint_id=candidate.checkpoint_id,
                    cause=cause,
                ) from cause
            completed_evictions.append(candidate.checkpoint_id)

        return allocation

    @staticmethod
    def _validate_handles(
        handles: Mapping[str, RuntimeCheckpointHandle],
    ) -> None:
        """在策略计算和运行时变更前验证句柄映射的一致性。"""
        seen_handles: dict[str, RuntimeCheckpointHandle] = {}
        for mapping_key, handle in handles.items():
            if not isinstance(handle, RuntimeCheckpointHandle):
                raise TypeError(
                    f"运行时句柄 {mapping_key} 必须是 RuntimeCheckpointHandle"
                )
            if mapping_key != handle.checkpoint_id:
                raise ValueError(
                    "运行时句柄字典键与检查点标识不一致："
                    f"键 {mapping_key}，句柄 {handle.checkpoint_id}"
                )

            previous = seen_handles.get(handle.checkpoint_id)
            if previous is not None and previous != handle:
                raise ValueError(
                    "同一 checkpoint_id 对应多个冲突句柄："
                    f"{handle.checkpoint_id}"
                )
            seen_handles[handle.checkpoint_id] = handle
