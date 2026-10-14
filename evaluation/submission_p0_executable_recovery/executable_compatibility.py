#!/usr/bin/env python3
"""Executable Compatibility 谓词：离线准入合同（Task A1）。

区分四个层次，不允许互相替代：
1. 逻辑 Workflow 祖先（B1 lineage）——不进入本谓词；
2. Token Prefix 相同（逐 token 比较，C2）；
3. Runtime Checkpoint 合法且有效（物化证据 + 物理身份 + 保护，C1/C3–C6）；
4. 实际发生 Recurrent State 恢复并通过正确性测试（C7，动态认证，
   只能由 runtime 阶段写入，离线阶段恒为 UNKNOWN）。

本模块只读、无副作用；不修改任何正式矩阵或数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ConditionStatus(str, Enum):
    """单个准入条件的判定状态。"""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckpointRecord:
    """一次真实物化 checkpoint 的运行时观测（来自 inspect 通道）。"""

    checkpoint_id: str
    model_identity: str
    tokenizer_identity: str
    token_prefix: tuple[int, ...]
    state_position: int
    node_id: int | None
    prefix_digest: str
    resolution_source: str
    recurrent_resident: bool
    fa_resident: bool
    fa_frontier: int
    generation: str | None = None
    lock_refs: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationRecord:
    """一个待恢复 continuation 的完整输入描述。"""

    continuation_id: str
    model_identity: str
    tokenizer_identity: str
    token_ids: tuple[int, ...]


@dataclass
class ConditionResult:
    """单个条件的结论与证据描述。"""

    condition_id: str
    status: ConditionStatus
    detail: str


@dataclass
class CompatibilityVerdict:
    """完整判定结果：整体结论 = 最弱条件。"""

    checkpoint_id: str
    continuation_id: str
    conditions: list[ConditionResult]
    is_admissible: bool

    def reasons(self) -> list[str]:
        """列出全部非 PASS 条件，便于审计输出。"""
        return [
            f"{item.condition_id}={item.status.value}: {item.detail}"
            for item in self.conditions
            if item.status is not ConditionStatus.PASS
        ]


def _identity_condition(
    checkpoint: CheckpointRecord,
    continuation: ContinuationRecord,
) -> ConditionResult:
    """C1：模型、Tokenizer 与状态配置身份一致。"""
    fields = []
    mismatch = []
    for name, left, right in (
        ("model_identity", checkpoint.model_identity, continuation.model_identity),
        (
            "tokenizer_identity",
            checkpoint.tokenizer_identity,
            continuation.tokenizer_identity,
        ),
    ):
        fields.append(name)
        if left != right:
            mismatch.append(f"{name}: {left!r} != {right!r}")
    if mismatch:
        return ConditionResult(
            condition_id="C1_identity",
            status=ConditionStatus.FAIL,
            detail="；".join(mismatch),
        )
    return ConditionResult(
        condition_id="C1_identity",
        status=ConditionStatus.PASS,
        detail=f"{'/'.join(fields)} 一致",
    )


def _token_history_condition(
    checkpoint: CheckpointRecord,
    continuation: ContinuationRecord,
    state_position: int,
) -> ConditionResult:
    """C2：checkpoint 记录的 token 历史与 continuation 前缀逐 token 一致。"""
    if state_position > len(checkpoint.token_prefix):
        return ConditionResult(
            condition_id="C2_token_history",
            status=ConditionStatus.FAIL,
            detail=(
                f"状态位置 L={state_position} 超过 checkpoint 记录历史长度 "
                f"{len(checkpoint.token_prefix)}"
            ),
        )
    if state_position > len(continuation.token_ids):
        return ConditionResult(
            condition_id="C2_token_history",
            status=ConditionStatus.FAIL,
            detail=(
                f"状态位置 L={state_position} 超过 continuation 输入长度 "
                f"{len(continuation.token_ids)}"
            ),
        )
    left = checkpoint.token_prefix[:state_position]
    right = continuation.token_ids[:state_position]
    if left != right:
        first_diff = next(
            index
            for index, (a, b) in enumerate(zip(left, right))
            if a != b
        )
        return ConditionResult(
            condition_id="C2_token_history",
            status=ConditionStatus.FAIL,
            detail=(
                f"token 历史在位置 {first_diff} 首次分歧 "
                f"({left[first_diff]} != {right[first_diff]})"
            ),
        )
    return ConditionResult(
        condition_id="C2_token_history",
        status=ConditionStatus.PASS,
        detail=f"前 {state_position} 个 token 逐位一致",
    )


def _position_condition(
    state_position: int,
    continuation: ContinuationRecord,
    continuation_fa_frontier: int | None,
) -> ConditionResult:
    """C3：L 合法且不超过目标可恢复前沿 T。"""
    if state_position <= 0:
        return ConditionResult(
            condition_id="C3_position",
            status=ConditionStatus.FAIL,
            detail=f"L={state_position} 必须为正",
        )
    frontier = (
        len(continuation.token_ids)
        if continuation_fa_frontier is None
        else min(len(continuation.token_ids), continuation_fa_frontier)
    )
    if state_position > frontier:
        return ConditionResult(
            condition_id="C3_position",
            status=ConditionStatus.FAIL,
            detail=f"L={state_position} > T={frontier}",
        )
    return ConditionResult(
        condition_id="C3_position",
        status=ConditionStatus.PASS,
        detail=f"0 < L={state_position} <= T={frontier}",
    )


def _residency_condition(checkpoint: CheckpointRecord) -> ConditionResult:
    """C4：FA-KV 与 Recurrent State 满足恢复条件（当前都要求驻留设备）。"""
    missing = []
    if not checkpoint.fa_resident:
        missing.append("FA-KV")
    if not checkpoint.recurrent_resident:
        missing.append("Recurrent(Mamba)")
    if missing:
        return ConditionResult(
            condition_id="C4_state_residency",
            status=ConditionStatus.FAIL,
            detail=f"未驻留：{'+'.join(missing)}",
        )
    if checkpoint.resolution_source == "exact":
        note = "exact 节点"
    else:
        note = (
            f"fallback({checkpoint.resolution_source})——物理前缀短于请求前缀，"
            "恢复深度按物化位置计算"
        )
    return ConditionResult(
        condition_id="C4_state_residency",
        status=ConditionStatus.PASS,
        detail=f"FA 与 Recurrent 均驻留；{note}；FA frontier={checkpoint.fa_frontier}",
    )


def _physical_identity_condition(checkpoint: CheckpointRecord) -> ConditionResult:
    """C5：物理身份与生命周期证据（node、digest、generation）。"""
    unknown = []
    if checkpoint.node_id is None:
        unknown.append("node_id")
    if not checkpoint.prefix_digest:
        unknown.append("prefix_digest")
    if checkpoint.generation is None:
        # slot generation 账本当前 runtime 不提供，如实标 UNKNOWN，不伪称通过。
        unknown.append("slot_generation")
    if unknown:
        return ConditionResult(
            condition_id="C5_physical_identity",
            status=ConditionStatus.UNKNOWN,
            detail=f"缺失：{'+'.join(unknown)}",
        )
    return ConditionResult(
        condition_id="C5_physical_identity",
        status=ConditionStatus.PASS,
        detail=(
            f"node={checkpoint.node_id} digest={checkpoint.prefix_digest[:16]}… "
            f"generation={checkpoint.generation}"
        ),
    )


def _protection_condition(checkpoint: CheckpointRecord) -> ConditionResult:
    """C6：Active/COW/Reference 保护检查（读取时无未释放引用）。"""
    held = {
        name: int(value)
        for name, value in (checkpoint.lock_refs or {}).items()
        if int(value) != 0
    }
    if held:
        return ConditionResult(
            condition_id="C6_protection",
            status=ConditionStatus.FAIL,
            detail=f"仍持有引用：{held}",
        )
    if not checkpoint.lock_refs:
        return ConditionResult(
            condition_id="C6_protection",
            status=ConditionStatus.UNKNOWN,
            detail="未提供引用计数证据",
        )
    return ConditionResult(
        condition_id="C6_protection",
        status=ConditionStatus.PASS,
        detail="无未释放引用（读取时点）",
    )


def _dynamic_condition(runtime_evidence: Mapping[str, object] | None) -> ConditionResult:
    """C7：实际 Resume 的动态正确性认证（只能由 runtime 阶段写入）。"""
    if not runtime_evidence:
        return ConditionResult(
            condition_id="C7_resume_certified",
            status=ConditionStatus.UNKNOWN,
            detail="离线阶段不可认证",
        )
    ok = bool(runtime_evidence.get("resume_verified"))
    if not ok:
        return ConditionResult(
            condition_id="C7_resume_certified",
            status=ConditionStatus.FAIL,
            detail="runtime 阶段报告恢复未通过",
        )
    return ConditionResult(
        condition_id="C7_resume_certified",
        status=ConditionStatus.PASS,
        detail=str(runtime_evidence.get("detail", "已认证")),
    )


def evaluate_executable_compatibility(
    checkpoint: CheckpointRecord,
    continuation: ContinuationRecord,
    *,
    state_position: int | None = None,
    continuation_fa_frontier: int | None = None,
    runtime_evidence: Mapping[str, object] | None = None,
) -> CompatibilityVerdict:
    """计算 ExecutableCompatible(c, p) 的完整判定。

    state_position 缺省取 checkpoint 记录的物化位置（C2/C3 会再校验其
    不超过 checkpoint 历史长度与 continuation 可恢复前沿 T）；
    continuation_fa_frontier 为目标请求查询到的驻留 FA 前沿，缺省按输入长度。
    """
    resolved_position = (
        int(state_position)
        if state_position is not None
        else int(checkpoint.state_position)
    )
    conditions = [
        _identity_condition(checkpoint, continuation),
        _token_history_condition(checkpoint, continuation, resolved_position),
        _position_condition(
            resolved_position,
            continuation,
            continuation_fa_frontier,
        ),
        _residency_condition(checkpoint),
        _physical_identity_condition(checkpoint),
        _protection_condition(checkpoint),
        _dynamic_condition(runtime_evidence),
    ]
    # 准入（可进入 runtime 恢复）= C1–C6 无 FAIL；C7 单独作为认证层。
    admission_conditions = [
        item for item in conditions if item.condition_id != "C7_resume_certified"
    ]
    admissible = all(
        item.status is ConditionStatus.PASS or item.status is ConditionStatus.UNKNOWN
        for item in admission_conditions
    )
    return CompatibilityVerdict(
        checkpoint_id=checkpoint.checkpoint_id,
        continuation_id=continuation.continuation_id,
        conditions=conditions,
        is_admissible=admissible,
    )
