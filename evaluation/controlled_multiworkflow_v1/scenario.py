"""定义共享循环状态预算下的受控多工作流场景。"""

from __future__ import annotations

from dataclasses import dataclass

from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_CHECKPOINT_SIZE_VALUE = 49.125 * 1024 * 1024
if not _CHECKPOINT_SIZE_VALUE.is_integer():
    raise RuntimeError("检查点大小无法精确表示为整数个字节")

CHECKPOINT_SIZE_BYTES = int(_CHECKPOINT_SIZE_VALUE)
BUDGET_CHECKPOINTS = 3


@dataclass(frozen=True)
class WorkflowSpec:
    """记录一个受控工作流的锚点与待续分支。"""

    workflow_id: str
    root_lineage: str
    anchor_pos: int
    pending_branches: tuple[str, ...]

    @property
    def pending_fanout(self) -> int:
        """返回当前工作流的待续分支数量。"""
        return len(self.pending_branches)


@dataclass(frozen=True)
class WorkloadMetadata:
    """保存受控 workload 的固定配置。"""

    workflows: tuple[WorkflowSpec, ...]
    checkpoint_recency: tuple["CheckpointRecency", ...]
    workflow_order: tuple[str, ...]
    checkpoint_size_bytes: int
    budget_checkpoints: int


@dataclass(frozen=True)
class CheckpointRecency:
    """记录 evaluation 层使用的检查点创建与最近访问顺序。"""

    checkpoint_id: str
    creation_order: int
    last_access_order: int


@dataclass(frozen=True)
class ControlledScenario:
    """汇总一次离线受控场景所需的逻辑输入。"""

    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]
    budget_bytes: int
    metadata: WorkloadMetadata


_WORKFLOWS = (
    WorkflowSpec("W1", "ROOT1", 32_768, ("A", "B")),
    WorkflowSpec("W2", "ROOT2", 16_384, ("B",)),
    WorkflowSpec("W3", "ROOT3", 8_192, ("B",)),
    WorkflowSpec("W4", "ROOT4", 4_096, ("A", "B", "C")),
)

_CANDIDATE_SPECS = (
    ("W1_PARENT", "W1", 32_768),
    ("W1_SHALLOW", "W1", 16_384),
    ("W2_PARENT", "W2", 16_384),
    ("W3_PARENT", "W3", 8_192),
    ("W4_PARENT", "W4", 4_096),
)

_CHECKPOINT_RECENCY = (
    CheckpointRecency("W1_SHALLOW", 1, 1),
    CheckpointRecency("W1_PARENT", 2, 2),
    CheckpointRecency("W2_PARENT", 3, 3),
    CheckpointRecency("W3_PARENT", 4, 4),
    CheckpointRecency("W4_PARENT", 5, 5),
)


def build_scenario() -> ControlledScenario:
    """构造固定的七个待续分支、五个候选和共享预算。"""
    continuations = tuple(
        PendingContinuation(
            continuation_id=f"{workflow.workflow_id}-{branch}",
            workflow_id=workflow.workflow_id,
            lineage_path=(workflow.root_lineage, branch),
            anchor_pos=workflow.anchor_pos,
            resident_fa_frontier=workflow.anchor_pos,
        )
        for workflow in _WORKFLOWS
        for branch in workflow.pending_branches
    )

    workflows_by_id = {
        workflow.workflow_id: workflow for workflow in _WORKFLOWS
    }
    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            lineage_path=(workflows_by_id[workflow_id].root_lineage,),
            token_pos=token_pos,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
            recurrent_resident=True,
            fa_resident=True,
        )
        for checkpoint_id, workflow_id, token_pos in _CANDIDATE_SPECS
    )

    metadata = WorkloadMetadata(
        workflows=_WORKFLOWS,
        checkpoint_recency=_CHECKPOINT_RECENCY,
        workflow_order=tuple(
            workflow.workflow_id for workflow in _WORKFLOWS
        ),
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        budget_checkpoints=BUDGET_CHECKPOINTS,
    )
    return ControlledScenario(
        continuations=continuations,
        candidates=candidates,
        budget_bytes=BUDGET_CHECKPOINTS * CHECKPOINT_SIZE_BYTES,
        metadata=metadata,
    )
