"""连接全局优化结果与运行时状态操作。"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .adapters.sglang import RuntimeCheckpointHandle
from .optimizer import AllocationResult, GlobalOptimizer
from .state_catalog import CheckpointCandidate
from .workflow import PendingContinuation


class RuntimeAdapter(Protocol):
    """描述控制器执行状态驱逐所需的最小运行时能力。"""

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """仅驱逐句柄指向的 Mamba 状态。"""


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
        """运行全局选择，并驱逐未被保留的现有驻留状态。"""
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

        for candidate in eviction_candidates:
            handle = handles[candidate.checkpoint_id]
            if handle.checkpoint_id != candidate.checkpoint_id:
                raise ValueError(
                    "运行时句柄与检查点标识不一致："
                    f"期望 {candidate.checkpoint_id}，实际 {handle.checkpoint_id}"
                )

        for candidate in eviction_candidates:
            self._runtime_adapter.evict_mamba_only(
                handles[candidate.checkpoint_id]
            )

        return allocation
