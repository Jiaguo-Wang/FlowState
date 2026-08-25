"""定义四因素完整交叉的 SOTA 信号受控 workload。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    CheckpointRecency,
)
from evaluation.sota_metadata import CONTROLLED_MARCONI_ALPHA
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


ANCHOR_DEPTHS = (8_192, 32_768)
FANOUTS = (1, 4)
STEPS_TO_EXECUTION = (1, 3)
RECENCY_CLASSES = ("old", "recent")
BUDGET_CHECKPOINTS = (4, 8, 12, 16)


@dataclass(frozen=True)
class SignalWorkflowSpec:
    """记录一个工作流的四个独立实验因素。"""

    workflow_id: str
    anchor_depth: int
    fanout: int
    steps_to_execution: int
    recency_class: str

    @property
    def factor_tuple(self) -> tuple[int, int, int, str]:
        """返回用于完整阶乘审计的固定因素元组。"""
        return (
            self.anchor_depth,
            self.fanout,
            self.steps_to_execution,
            self.recency_class,
        )


@dataclass(frozen=True)
class SignalWorkloadMetadata:
    """保存策略比较开始前已经冻结的 workload metadata。"""

    workflows: tuple[SignalWorkflowSpec, ...]
    workflow_order: tuple[str, ...]
    checkpoint_recency: tuple[CheckpointRecency, ...]
    steps_to_execution_by_continuation: Mapping[str, int]
    checkpoint_size_bytes: int
    budget_checkpoints: int
    budget_options: tuple[int, ...]
    marconi_alpha: float


@dataclass(frozen=True)
class SignalScenario:
    """汇总一个预算点的核心输入与显式实验 metadata。"""

    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]
    budget_bytes: int
    metadata: SignalWorkloadMetadata


def build_scenario(budget_checkpoints: int = 4) -> SignalScenario:
    """按预注册顺序构造十六个独立工作流的完整阶乘场景。"""
    if budget_checkpoints not in BUDGET_CHECKPOINTS:
        raise ValueError(
            f"预算 K 必须属于固定集合 {BUDGET_CHECKPOINTS}"
        )

    workflows = tuple(
        SignalWorkflowSpec(
            workflow_id=(
                f"W_A{anchor_depth:05d}_F{fanout}_"
                f"S{steps_to_execution}_R{recency_class}"
            ),
            anchor_depth=anchor_depth,
            fanout=fanout,
            steps_to_execution=steps_to_execution,
            recency_class=recency_class,
        )
        for anchor_depth in ANCHOR_DEPTHS
        for fanout in FANOUTS
        for steps_to_execution in STEPS_TO_EXECUTION
        for recency_class in RECENCY_CLASSES
    )
    continuations = tuple(
        PendingContinuation(
            continuation_id=f"{workflow.workflow_id}-B{branch_index}",
            workflow_id=workflow.workflow_id,
            lineage_path=("P", f"B{branch_index}"),
            anchor_pos=workflow.anchor_depth,
            resident_fa_frontier=workflow.anchor_depth,
        )
        for workflow in workflows
        for branch_index in range(1, workflow.fanout + 1)
    )
    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=f"{workflow.workflow_id}_MAIN",
            workflow_id=workflow.workflow_id,
            lineage_path=("P",),
            token_pos=workflow.anchor_depth,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
            recurrent_resident=True,
            fa_resident=True,
        )
        for workflow in workflows
    )

    workflows_by_id = {
        workflow.workflow_id: workflow for workflow in workflows
    }
    access_order = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                RECENCY_CLASSES.index(
                    workflows_by_id[candidate.workflow_id].recency_class
                ),
                candidate.checkpoint_id,
            ),
        )
    )
    last_access_by_id = {
        candidate.checkpoint_id: rank
        for rank, candidate in enumerate(access_order, start=1)
    }
    checkpoint_recency = tuple(
        CheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=index,
            last_access_order=last_access_by_id[candidate.checkpoint_id],
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    steps_by_continuation = MappingProxyType(
        {
            continuation.continuation_id: workflows_by_id[
                continuation.workflow_id
            ].steps_to_execution
            for continuation in continuations
        }
    )
    metadata = SignalWorkloadMetadata(
        workflows=workflows,
        workflow_order=tuple(
            workflow.workflow_id for workflow in workflows
        ),
        checkpoint_recency=checkpoint_recency,
        steps_to_execution_by_continuation=steps_by_continuation,
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        budget_checkpoints=budget_checkpoints,
        budget_options=BUDGET_CHECKPOINTS,
        marconi_alpha=CONTROLLED_MARCONI_ALPHA,
    )
    return SignalScenario(
        continuations=continuations,
        candidates=candidates,
        budget_bytes=budget_checkpoints * CHECKPOINT_SIZE_BYTES,
        metadata=metadata,
    )
