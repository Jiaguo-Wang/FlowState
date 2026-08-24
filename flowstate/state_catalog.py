"""Checkpoint catalog data structures and compatibility rules."""

from __future__ import annotations

from dataclasses import dataclass

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
