"""定义可扩展受控多工作流的固定阶乘场景。"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    CheckpointRecency,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


ANCHOR_DEPTHS = (4_096, 8_192, 16_384, 32_768)
FANOUTS_BY_WORKFLOW_COUNT = {
    8: (1, 4),
    16: (1, 2, 4, 8),
}
BUDGETS_BY_WORKFLOW_COUNT = {
    8: (2, 4, 6, 8),
    16: (4, 8, 12, 16),
}


@dataclass(frozen=True)
class ScalableWorkflowSpec:
    """记录一个阶乘工作流的锚点深度与待续分支数量。"""

    workflow_id: str
    anchor_pos: int
    pending_fanout: int


@dataclass(frozen=True)
class ScalableWorkloadMetadata:
    """保存策略无关的固定 workload 元数据。"""

    workflow_count: int
    workflows: tuple[ScalableWorkflowSpec, ...]
    checkpoint_recency: tuple[CheckpointRecency, ...]
    workflow_order: tuple[str, ...]
    checkpoint_size_bytes: int
    budget_checkpoints: int
    budget_options: tuple[int, ...]
    anchor_depths: tuple[int, ...]
    fanouts: tuple[int, ...]


@dataclass(frozen=True)
class ScalableScenario:
    """汇总一个预算决策点所需的逻辑输入。"""

    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]
    budget_bytes: int
    metadata: ScalableWorkloadMetadata


def build_scenario(
    workflow_count: int,
    budget_checkpoints: int | None = None,
) -> ScalableScenario:
    """按固定阶乘规则构造 N=8 或 N=16 的场景。"""
    fanouts = FANOUTS_BY_WORKFLOW_COUNT.get(workflow_count)
    budget_options = BUDGETS_BY_WORKFLOW_COUNT.get(workflow_count)
    if fanouts is None or budget_options is None:
        raise ValueError("workflow_count 只支持 8 或 16")
    active_budget = (
        budget_options[0]
        if budget_checkpoints is None
        else budget_checkpoints
    )
    if active_budget not in budget_options:
        raise ValueError(
            f"N={workflow_count} 不支持预算 K={active_budget}"
        )

    workflows = tuple(
        ScalableWorkflowSpec(
            workflow_id=(
                f"W{workflow_count:02d}_A{anchor_pos:05d}_"
                f"F{fanout:02d}"
            ),
            anchor_pos=anchor_pos,
            pending_fanout=fanout,
        )
        for anchor_pos in ANCHOR_DEPTHS
        for fanout in fanouts
    )
    continuations = tuple(
        PendingContinuation(
            continuation_id=f"{workflow.workflow_id}-B{branch_index}",
            workflow_id=workflow.workflow_id,
            lineage_path=("P", f"B{branch_index}"),
            anchor_pos=workflow.anchor_pos,
            resident_fa_frontier=workflow.anchor_pos,
        )
        for workflow in workflows
        for branch_index in range(1, workflow.pending_fanout + 1)
    )

    candidates = []
    for workflow in workflows:
        candidates.append(
            CheckpointCandidate(
                checkpoint_id=f"{workflow.workflow_id}_MAIN",
                workflow_id=workflow.workflow_id,
                lineage_path=("P",),
                token_pos=workflow.anchor_pos,
                memory_bytes=CHECKPOINT_SIZE_BYTES,
                recurrent_resident=True,
                fa_resident=True,
            )
        )
        if workflow.pending_fanout == fanouts[0]:
            candidates.append(
                CheckpointCandidate(
                    checkpoint_id=f"{workflow.workflow_id}_SHALLOW",
                    workflow_id=workflow.workflow_id,
                    lineage_path=("P",),
                    token_pos=workflow.anchor_pos // 2,
                    memory_bytes=CHECKPOINT_SIZE_BYTES,
                    recurrent_resident=True,
                    fa_resident=True,
                )
            )
    candidate_tuple = tuple(candidates)
    checkpoint_recency = tuple(
        CheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=index,
            last_access_order=index,
        )
        for index, candidate in enumerate(candidate_tuple, start=1)
    )
    metadata = ScalableWorkloadMetadata(
        workflow_count=workflow_count,
        workflows=workflows,
        checkpoint_recency=checkpoint_recency,
        workflow_order=tuple(
            workflow.workflow_id for workflow in workflows
        ),
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        budget_checkpoints=active_budget,
        budget_options=budget_options,
        anchor_depths=ANCHOR_DEPTHS,
        fanouts=fanouts,
    )
    return ScalableScenario(
        continuations=continuations,
        candidates=candidate_tuple,
        budget_bytes=active_budget * CHECKPOINT_SIZE_BYTES,
        metadata=metadata,
    )
