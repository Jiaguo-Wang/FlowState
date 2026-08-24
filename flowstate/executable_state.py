"""计算规划阶段的可执行状态前沿与恢复间隔。"""

from __future__ import annotations

from typing import Iterable

from .state_catalog import CheckpointCandidate, is_compatible
from .workflow import PendingContinuation


def executable_frontier(
    continuation: PendingContinuation,
    selected: Iterable[CheckpointCandidate],
) -> int:
    """返回所选兼容检查点中最深的令牌位置，无兼容项时返回零。"""

    compatible_positions = (
        checkpoint.token_pos
        for checkpoint in selected
        if is_compatible(checkpoint, continuation)
    )
    return max(compatible_positions, default=0)


def recovery_gap(
    continuation: PendingContinuation,
    selected: Iterable[CheckpointCandidate],
) -> int:
    """返回规划目标与可执行状态前沿之间的恢复间隔。"""

    frontier = executable_frontier(continuation, selected)
    gap = continuation.planning_target - frontier
    if gap < 0:
        raise ValueError("恢复间隔不能为负数")
    return gap

