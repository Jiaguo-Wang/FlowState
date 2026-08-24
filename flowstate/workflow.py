"""Workflow-side data structures used for checkpoint planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingContinuation:
    """A workflow continuation waiting to resume from compatible state."""

    continuation_id: str
    workflow_id: str
    lineage_path: tuple[str, ...]
    anchor_pos: int
    resident_fa_frontier: int

    def __post_init__(self) -> None:
        if self.anchor_pos < 0:
            raise ValueError("anchor_pos must be non-negative")
        if self.resident_fa_frontier < 0:
            raise ValueError("resident_fa_frontier must be non-negative")

    @property
    def planning_target(self) -> int:
        """Return the deepest position currently useful to this continuation."""

        return min(self.anchor_pos, self.resident_fa_frontier)
