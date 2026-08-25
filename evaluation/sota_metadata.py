"""构造受控实验中预先冻结的 SOTA-style 策略元数据。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from flowstate.state_catalog import (
    CheckpointCandidate,
    is_lineage_prefix,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


CONTROLLED_MARCONI_ALPHA = 1.0


@dataclass(frozen=True)
class ControlledSOTAMetadata:
    """冻结一个受控决策快照使用的全部 SOTA-style metadata。"""

    kvflow_steps: Mapping[str, int]
    last_access_by_checkpoint: Mapping[str, float]
    marconi_flop_saved: Mapping[str, float]
    marconi_alpha: float

    @property
    def marconi_last_access(self) -> Mapping[str, float]:
        """返回与 Global-LRU 共用的最近访问 metadata。"""
        return self.last_access_by_checkpoint


class CheckpointRecencyRecord(Protocol):
    """描述现有全局 LRU 策略使用的时序字段。"""

    checkpoint_id: str
    creation_order: int
    last_access_order: int


def build_controlled_sota_metadata(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    checkpoint_recency: Sequence[CheckpointRecencyRecord],
) -> ControlledSOTAMetadata:
    """在策略比较开始前一次性冻结全部受控 metadata。"""
    return ControlledSOTAMetadata(
        kvflow_steps=MappingProxyType(build_kvflow_steps(continuations)),
        last_access_by_checkpoint=MappingProxyType(
            build_marconi_recency(candidates, checkpoint_recency)
        ),
        marconi_flop_saved=MappingProxyType(
            build_marconi_flop_saved(candidates)
        ),
        marconi_alpha=CONTROLLED_MARCONI_ALPHA,
    )


def build_kvflow_steps(
    continuations: Sequence[PendingContinuation],
) -> dict[str, int]:
    """为受控 workload 的所有直接下一步分支固定执行距离一。"""
    continuation_ids = tuple(
        continuation.continuation_id for continuation in continuations
    )
    if len(set(continuation_ids)) != len(continuation_ids):
        raise ValueError("待续请求标识必须唯一")
    return {
        continuation_id: 1
        for continuation_id in sorted(continuation_ids)
    }


def build_marconi_recency(
    candidates: Sequence[CheckpointCandidate],
    checkpoint_recency: Sequence[CheckpointRecencyRecord],
) -> dict[str, float]:
    """把现有全局 LRU 全序转换为数值越大越新的确定性 rank。"""
    validate_unique_checkpoint_ids(candidates)
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.recurrent_resident
    )
    if not eligible:
        return {}

    ordered_newest_first = select_global_lru(
        candidates,
        checkpoint_recency,
        len(eligible) * eligible[0].memory_bytes,
    )
    if len(ordered_newest_first) != len(eligible):
        raise RuntimeError("全局 LRU 全序未包含全部 eligible 候选")
    candidate_count = len(ordered_newest_first)
    return {
        checkpoint_id: float(candidate_count - index)
        for index, checkpoint_id in enumerate(ordered_newest_first)
    }


def build_marconi_flop_saved(
    candidates: Sequence[CheckpointCandidate],
) -> dict[str, float]:
    """用父候选相对的增量 replay-token span 构造 FLOP 比例代理。"""
    validate_unique_checkpoint_ids(candidates)
    result: dict[str, float] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: item.checkpoint_id,
    ):
        parent_positions = tuple(
            ancestor.token_pos
            for ancestor in candidates
            if ancestor.checkpoint_id != candidate.checkpoint_id
            and ancestor.workflow_id == candidate.workflow_id
            and ancestor.token_pos < candidate.token_pos
            and is_lineage_prefix(
                ancestor.lineage_path,
                candidate.lineage_path,
            )
        )
        parent_pos = max(parent_positions, default=0)
        incremental_tokens = candidate.token_pos - parent_pos
        if incremental_tokens <= 0:
            raise ValueError(
                f"检查点 {candidate.checkpoint_id} 的增量 token span 必须大于零"
            )
        result[candidate.checkpoint_id] = float(incremental_tokens)
    return result
