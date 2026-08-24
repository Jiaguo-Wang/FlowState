"""Core data model for FlowState."""

from .state_catalog import CheckpointCandidate, is_compatible
from .workflow import PendingContinuation

__all__ = [
    "CheckpointCandidate",
    "PendingContinuation",
    "is_compatible",
]

