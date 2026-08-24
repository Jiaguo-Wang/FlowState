"""Checkpoint catalog data structures and compatibility rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .workflow import PendingContinuation


@dataclass(frozen=True)
class CheckpointCandidate:
    """A recurrent checkpoint that may serve a pending continuation."""

    checkpoint_id: str
    workflow_id: str
    lineage_path: tuple[str, ...]
    token_pos: int
    memory_bytes: int
    recurrent_resident: bool = True
    fa_resident: bool = True

    def __post_init__(self) -> None:
        if self.token_pos < 0:
            raise ValueError("token_pos must be non-negative")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")


def validate_unique_checkpoint_ids(
    candidates: Sequence[CheckpointCandidate],
) -> None:
    """确认一个决策快照中的检查点标识没有重复。"""
    seen = set()
    duplicates = set()
    for candidate in candidates:
        checkpoint_id = candidate.checkpoint_id
        if checkpoint_id in seen:
            duplicates.add(checkpoint_id)
        seen.add(checkpoint_id)

    if duplicates:
        duplicate_ids = ", ".join(sorted(duplicates))
        raise ValueError(f"checkpoint_id 必须唯一，重复标识：{duplicate_ids}")


def is_lineage_prefix(
    candidate_path: tuple[str, ...],
    continuation_path: tuple[str, ...],
) -> bool:
    """Return whether a candidate lies on the continuation's ancestry path."""

    candidate_length = len(candidate_path)
    return (
        candidate_length <= len(continuation_path)
        and continuation_path[:candidate_length] == candidate_path
    )


def is_compatible(
    checkpoint: CheckpointCandidate,
    continuation: PendingContinuation,
) -> bool:
    """Return whether a checkpoint can resume the given continuation."""

    return (
        checkpoint.workflow_id == continuation.workflow_id
        and is_lineage_prefix(
            checkpoint.lineage_path,
            continuation.lineage_path,
        )
        and checkpoint.token_pos <= continuation.planning_target
    )
