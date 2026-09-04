"""为正式 RQ3 提供不可变分配快照与统一五策略离线评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from time import perf_counter_ns
from typing import Callable, Iterable, Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.scenario import CheckpointRecency
from evaluation.sota_metadata import CONTROLLED_MARCONI_ALPHA
from evaluation.sota_policies import MarconiStylePolicy
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_FLOAT_TOLERANCE_MS = 1e-9


@dataclass(frozen=True)
class FrozenPendingContinuation:
    """保存一个待续请求的深度不可变规划字段。"""

    continuation_id: str
    workflow_id: str
    lineage_path: tuple[str, ...]
    anchor_pos: int
    resident_fa_frontier: int

    @property
    def planning_target(self) -> int:
        """返回当前待续请求可利用的最深物理前沿。"""

        return min(self.anchor_pos, self.resident_fa_frontier)

    def to_core(self) -> PendingContinuation:
        """返回与核心选择器隔离的新待续请求对象。"""

        return PendingContinuation(
            continuation_id=self.continuation_id,
            workflow_id=self.workflow_id,
            lineage_path=tuple(self.lineage_path),
            anchor_pos=self.anchor_pos,
            resident_fa_frontier=self.resident_fa_frontier,
        )


@dataclass(frozen=True)
class FrozenCheckpointCandidate:
    """保存一个统一 eligible 候选的深度不可变字段。"""

    checkpoint_id: str
    workflow_id: str
    lineage_path: tuple[str, ...]
    token_pos: int
    memory_bytes: int
    recurrent_resident: bool
    fa_resident: bool

    def to_core(self) -> CheckpointCandidate:
        """返回与核心选择器隔离的新候选对象。"""

        return CheckpointCandidate(
            checkpoint_id=self.checkpoint_id,
            workflow_id=self.workflow_id,
            lineage_path=tuple(self.lineage_path),
            token_pos=self.token_pos,
            memory_bytes=self.memory_bytes,
            recurrent_resident=self.recurrent_resident,
            fa_resident=self.fa_resident,
        )


@dataclass(frozen=True)
class FrozenCandidateMetadata:
    """冻结基线选择器使用的创建、访问与重算效率字段。"""

    checkpoint_id: str
    creation_order: int
    last_access_order: int
    marconi_flop_saved: float


@dataclass(frozen=True)
class FrozenAccessFrequency:
    """冻结一个 checkpoint 的 LFU Adaptation 访问频率。

    频率语义（Step 13D 冻结，忠实映射 SGLang v0.5.17 native hit_count）：
    一次 access 指一个在观测时点之前已完成写回 insert 的请求，
    其最终 token span 覆盖该 checkpoint 在其 lineage 上的位置；
    创建该 checkpoint 的请求写回即其首次 access，
    因此任何已物化候选的频率至少为 1；
    纯 prefix match 不计频率；同一请求对同一 checkpoint 最多计一次。
    """

    checkpoint_id: str
    access_frequency: int


@dataclass(frozen=True)
class FrozenRecoveryModelIdentity:
    """冻结正式恢复模型的身份、系数、单位与有效域。"""

    name: str
    coefficient_a: float
    coefficient_b: float
    coefficient_c: float
    gap_unit: str
    target_unit: str
    output_unit: str
    calibration_artifact: str
    minimum_gap_tokens: int
    maximum_target_tokens: int


@dataclass(frozen=True)
class FrozenCheckpointRuntimeEvidence:
    """冻结 checkpoint 与真实 runtime 身份之间的稳定摘要。"""

    checkpoint_id: str
    node_id: int
    runtime_identity_digest: str
    checkpoint_handle_digest: str


@dataclass(frozen=True)
class FrozenOnlineInformationBoundary:
    """声明快照可见时点并显式拒绝任何未来信息。"""

    materialized_through_epoch: int
    visible_continuation_ids: tuple[str, ...]
    future_continuation_included: bool = False
    future_request_included: bool = False
    future_latency_included: bool = False


@dataclass(frozen=True)
class AllocationSnapshot:
    """原子冻结五种策略共同消费的正式分配输入。"""

    allocation_epoch: int
    snapshot_id: str
    pending_continuations: tuple[FrozenPendingContinuation, ...]
    eligible_candidates: tuple[FrozenCheckpointCandidate, ...]
    candidate_metadata: tuple[FrozenCandidateMetadata, ...]
    lfu_access_frequency: tuple[FrozenAccessFrequency, ...]
    frequency_observed_through_epoch: int
    marconi_alpha: float
    recovery_model: FrozenRecoveryModelIdentity
    logical_budget_k: int
    budget_bytes: int
    runtime_evidence: tuple[FrozenCheckpointRuntimeEvidence, ...]
    residency_snapshot_digest: str
    online_boundary: FrozenOnlineInformationBoundary

    def canonical_serialization(self) -> str:
        """返回字段排序固定且禁止非有限数值的规范 JSON。"""

        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def content_digest(self) -> str:
        """返回规范序列化内容的确定性 SHA-256 摘要。"""

        return sha256(
            self.canonical_serialization().encode("utf-8")
        ).hexdigest()

    def core_continuations(self) -> tuple[PendingContinuation, ...]:
        """为一次策略调用重建互不共享的核心待续请求对象。"""

        return tuple(item.to_core() for item in self.pending_continuations)

    def core_candidates(self) -> tuple[CheckpointCandidate, ...]:
        """为一次策略调用重建互不共享的核心候选对象。"""

        return tuple(item.to_core() for item in self.eligible_candidates)


@dataclass(frozen=True)
class ContinuationObjective:
    """记录一个待续请求在指定选择集合下的正式目标分解。"""

    continuation_id: str
    workflow_id: str
    target_tokens: int
    executable_frontier_tokens: int
    recovery_gap_tokens: int
    recovery_cost_ms: float


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """记录公共目标评分器对一个选择集合的完整结果。"""

    selected_checkpoint_ids: tuple[str, ...]
    total_recovery_cost_ms: float
    empty_selection_cost_ms: float
    total_benefit_ms: float
    per_continuation: tuple[ContinuationObjective, ...]
    objective_evaluation_count: int


@dataclass(frozen=True)
class ExactSelection:
    """记录组合枚举的确定性最优选择与内部评价次数。"""

    selected_checkpoint_ids: tuple[str, ...]
    selector_internal_evaluations: int


@dataclass(frozen=True)
class PolicyEvaluation:
    """记录单个策略的选择、统一评分和无副作用证据。"""

    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    selector_wall_time_ms: float
    total_recovery_cost_ms: float
    empty_selection_cost_ms: float
    total_benefit_ms: float
    per_continuation: tuple[ContinuationObjective, ...]
    candidate_count: int
    pending_count: int
    selector_internal_evaluations: int
    final_common_scoring_evaluations: int
    snapshot_digest_before: str
    snapshot_digest_after: str


@dataclass(frozen=True)
class GreedyExactMetrics:
    """记录 FlowState greedy 相对 Exact OPT 的正式差距。"""

    absolute_cost_gap_ms: float
    benefit_ratio: float | None
    relative_cost_gap: float | None


@dataclass(frozen=True)
class FivePolicyEvaluation:
    """保存一个不可变快照上的五策略统一评价结果。"""

    snapshot_id: str
    snapshot_digest: str
    policy_results: tuple[PolicyEvaluation, ...]
    flowstate_vs_exact: GreedyExactMetrics


ObjectiveFunction = Callable[
    [AllocationSnapshot, Sequence[str]],
    ObjectiveEvaluation,
]


class _CountingRecoveryCostModel(RecoveryCostModel):
    """在不改变正式 Phi 数值的前提下统计内部模型调用。"""

    def __init__(self) -> None:
        self.estimate_calls = 0

    def estimate(self, gap_tokens: int, target_tokens: int) -> float:
        """记录一次 Phi 调用并委托给冻结正式实现。"""

        self.estimate_calls += 1
        return super().estimate(gap_tokens, target_tokens)


def recovery_model_identity(
    model: RecoveryCostModel | None = None,
) -> FrozenRecoveryModelIdentity:
    """从正式恢复模型构造可进入 snapshot digest 的完整身份。"""

    active = model or RecoveryCostModel()
    metadata = active.metadata
    return FrozenRecoveryModelIdentity(
        name=str(metadata.name),
        coefficient_a=float(metadata.coefficient_a),
        coefficient_b=float(metadata.coefficient_b),
        coefficient_c=float(metadata.coefficient_c),
        gap_unit=str(metadata.gap_unit),
        target_unit=str(metadata.target_unit),
        output_unit=str(metadata.output_unit),
        calibration_artifact=str(metadata.calibration_artifact),
        minimum_gap_tokens=int(metadata.minimum_gap_tokens),
        maximum_target_tokens=int(metadata.maximum_target_tokens),
    )


def build_allocation_snapshot(
    *,
    allocation_epoch: int,
    snapshot_id: str,
    pending_continuations: Sequence[PendingContinuation],
    eligible_candidates: Sequence[CheckpointCandidate],
    creation_order_by_checkpoint: Mapping[str, int],
    last_access_order_by_checkpoint: Mapping[str, int],
    marconi_flop_saved_by_checkpoint: Mapping[str, float],
    access_frequency_by_checkpoint: Mapping[str, int],
    frequency_observed_through_epoch: int,
    marconi_alpha: float,
    logical_budget_k: int,
    budget_bytes: int,
    runtime_evidence: Sequence[FrozenCheckpointRuntimeEvidence],
    residency_snapshot_digest: str,
    online_boundary: FrozenOnlineInformationBoundary,
    recovery_model: RecoveryCostModel | None = None,
) -> AllocationSnapshot:
    """复制、验证并规范化一次正式 allocation snapshot。"""

    _validate_nonnegative_integer(allocation_epoch, "allocation_epoch")
    _validate_nonempty_text(snapshot_id, "snapshot_id")
    _validate_nonnegative_integer(logical_budget_k, "logical_budget_k")
    _validate_nonnegative_integer(budget_bytes, "budget_bytes")
    _validate_digest(residency_snapshot_digest, "residency_snapshot_digest")
    if not math.isfinite(float(marconi_alpha)):
        raise ValueError("Marconi alpha 必须是有限数值")
    if float(marconi_alpha) != CONTROLLED_MARCONI_ALPHA:
        raise ValueError("Marconi alpha 必须保持冻结值 1.0")

    pending = tuple(
        sorted(
            (
                FrozenPendingContinuation(
                    continuation_id=str(item.continuation_id),
                    workflow_id=str(item.workflow_id),
                    lineage_path=tuple(str(value) for value in item.lineage_path),
                    anchor_pos=int(item.anchor_pos),
                    resident_fa_frontier=int(item.resident_fa_frontier),
                )
                for item in pending_continuations
            ),
            key=lambda item: item.continuation_id,
        )
    )
    if not pending:
        raise ValueError("allocation snapshot 必须包含 pending continuation")
    _validate_unique(
        (item.continuation_id for item in pending),
        "continuation_id",
    )
    for item in pending:
        _validate_nonempty_text(item.continuation_id, "continuation_id")
        _validate_nonempty_text(item.workflow_id, "pending workflow_id")
        if not item.lineage_path:
            raise ValueError("pending lineage_path 不能为空")
        _validate_nonnegative_integer(item.anchor_pos, "anchor_pos")
        _validate_nonnegative_integer(
            item.resident_fa_frontier,
            "resident_fa_frontier",
        )
        if item.resident_fa_frontier > item.anchor_pos:
            raise ValueError("resident_fa_frontier 不能超过 anchor_pos")

    candidates = tuple(
        sorted(
            (
                FrozenCheckpointCandidate(
                    checkpoint_id=str(item.checkpoint_id),
                    workflow_id=str(item.workflow_id),
                    lineage_path=tuple(str(value) for value in item.lineage_path),
                    token_pos=int(item.token_pos),
                    memory_bytes=int(item.memory_bytes),
                    recurrent_resident=bool(item.recurrent_resident),
                    fa_resident=bool(item.fa_resident),
                )
                for item in eligible_candidates
            ),
            key=lambda item: item.checkpoint_id,
        )
    )
    if not candidates:
        raise ValueError("allocation snapshot 必须包含 eligible candidate")
    _validate_unique(
        (item.checkpoint_id for item in candidates),
        "checkpoint_id",
    )
    checkpoint_ids = tuple(item.checkpoint_id for item in candidates)
    for item in candidates:
        _validate_nonempty_text(item.checkpoint_id, "checkpoint_id")
        _validate_nonempty_text(item.workflow_id, "candidate workflow_id")
        if not item.lineage_path:
            raise ValueError("candidate lineage_path 不能为空")
        _validate_nonnegative_integer(item.token_pos, "token_pos")
        if item.memory_bytes <= 0:
            raise ValueError("candidate memory_bytes 必须大于零")
        if not item.recurrent_resident:
            raise ValueError(
                f"candidate {item.checkpoint_id} 的循环状态未驻留"
            )
        if not item.fa_resident:
            raise ValueError(f"candidate {item.checkpoint_id} 的 FA 状态未驻留")
    checkpoint_sizes = {item.memory_bytes for item in candidates}
    if len(checkpoint_sizes) != 1:
        raise ValueError("正式 snapshot 只接受等大小 checkpoint")
    checkpoint_size = next(iter(checkpoint_sizes))
    if budget_bytes != logical_budget_k * checkpoint_size:
        raise ValueError("budget_bytes 必须严格等于 K 乘以 checkpoint 大小")

    _validate_exact_keys(
        checkpoint_ids,
        creation_order_by_checkpoint,
        "creation_order",
    )
    _validate_exact_keys(
        checkpoint_ids,
        last_access_order_by_checkpoint,
        "last_access_order",
    )
    _validate_exact_keys(
        checkpoint_ids,
        marconi_flop_saved_by_checkpoint,
        "marconi_flop_saved",
    )
    metadata = []
    for checkpoint_id in checkpoint_ids:
        creation_order = creation_order_by_checkpoint[checkpoint_id]
        last_access_order = last_access_order_by_checkpoint[checkpoint_id]
        _validate_nonnegative_integer(creation_order, "creation_order")
        _validate_nonnegative_integer(last_access_order, "last_access_order")
        flop_saved = float(marconi_flop_saved_by_checkpoint[checkpoint_id])
        if not math.isfinite(flop_saved) or flop_saved <= 0.0:
            raise ValueError("Marconi flop_saved 必须是有限正数")
        metadata.append(
            FrozenCandidateMetadata(
                checkpoint_id=checkpoint_id,
                creation_order=int(creation_order),
                last_access_order=int(last_access_order),
                marconi_flop_saved=flop_saved,
            )
        )

    _validate_exact_keys(
        checkpoint_ids,
        access_frequency_by_checkpoint,
        "access_frequency",
    )
    _validate_nonnegative_integer(
        frequency_observed_through_epoch,
        "frequency_observed_through_epoch",
    )
    frequency = []
    for checkpoint_id in checkpoint_ids:
        access_frequency = access_frequency_by_checkpoint[checkpoint_id]
        _validate_nonnegative_integer(access_frequency, "access_frequency")
        if access_frequency < 1:
            raise ValueError(
                "access_frequency 必须至少为 1：checkpoint 创建即首次访问"
            )
        frequency.append(
            FrozenAccessFrequency(
                checkpoint_id=checkpoint_id,
                access_frequency=int(access_frequency),
            )
        )

    evidence = tuple(
        sorted(
            (
                FrozenCheckpointRuntimeEvidence(
                    checkpoint_id=str(item.checkpoint_id),
                    node_id=int(item.node_id),
                    runtime_identity_digest=str(item.runtime_identity_digest),
                    checkpoint_handle_digest=str(item.checkpoint_handle_digest),
                )
                for item in runtime_evidence
            ),
            key=lambda item: item.checkpoint_id,
        )
    )
    if tuple(item.checkpoint_id for item in evidence) != checkpoint_ids:
        raise ValueError("runtime evidence 必须与 eligible candidate 一一对应")
    for item in evidence:
        _validate_nonnegative_integer(item.node_id, "runtime node_id")
        _validate_digest(
            item.runtime_identity_digest,
            "runtime_identity_digest",
        )
        _validate_digest(
            item.checkpoint_handle_digest,
            "checkpoint_handle_digest",
        )

    copied_boundary = FrozenOnlineInformationBoundary(
        materialized_through_epoch=int(
            online_boundary.materialized_through_epoch
        ),
        visible_continuation_ids=tuple(
            sorted(str(value) for value in online_boundary.visible_continuation_ids)
        ),
        future_continuation_included=bool(
            online_boundary.future_continuation_included
        ),
        future_request_included=bool(online_boundary.future_request_included),
        future_latency_included=bool(online_boundary.future_latency_included),
    )
    _validate_nonnegative_integer(
        copied_boundary.materialized_through_epoch,
        "materialized_through_epoch",
    )
    if copied_boundary.materialized_through_epoch > allocation_epoch:
        raise ValueError("online boundary 不能物化 allocation epoch 之后的信息")
    if (
        copied_boundary.future_continuation_included
        or copied_boundary.future_request_included
        or copied_boundary.future_latency_included
    ):
        raise ValueError("allocation snapshot 禁止包含未来信息")
    if copied_boundary.visible_continuation_ids != tuple(
        item.continuation_id for item in pending
    ):
        raise ValueError("online boundary 必须精确覆盖当前 pending set")

    if frequency_observed_through_epoch > allocation_epoch:
        raise ValueError(
            "access frequency 观测时点超过 allocation epoch，禁止使用未来信息"
        )
    if (
        frequency_observed_through_epoch
        > copied_boundary.materialized_through_epoch
    ):
        raise ValueError(
            "access frequency 观测时点不能超过 online boundary 物化时点"
        )

    return AllocationSnapshot(
        allocation_epoch=allocation_epoch,
        snapshot_id=snapshot_id,
        pending_continuations=pending,
        eligible_candidates=candidates,
        candidate_metadata=tuple(metadata),
        lfu_access_frequency=tuple(frequency),
        frequency_observed_through_epoch=int(
            frequency_observed_through_epoch
        ),
        marconi_alpha=float(marconi_alpha),
        recovery_model=recovery_model_identity(recovery_model),
        logical_budget_k=logical_budget_k,
        budget_bytes=budget_bytes,
        runtime_evidence=evidence,
        residency_snapshot_digest=str(residency_snapshot_digest),
        online_boundary=copied_boundary,
    )


def evaluate_objective(
    snapshot: AllocationSnapshot,
    selected_checkpoint_ids: Sequence[str],
) -> ObjectiveEvaluation:
    """用唯一正式 C(S) 定义评价任意预算内候选集合。"""

    _validate_recovery_model_identity(snapshot)
    selected_ids = tuple(sorted(str(value) for value in selected_checkpoint_ids))
    _validate_unique(selected_ids, "selected_checkpoint_ids")
    candidate_by_id = {
        item.checkpoint_id: item.to_core()
        for item in snapshot.eligible_candidates
    }
    missing = tuple(value for value in selected_ids if value not in candidate_by_id)
    if missing:
        raise ValueError(f"选择集合包含未知 checkpoint：{missing}")
    if len(selected_ids) > snapshot.logical_budget_k:
        raise ValueError("选择集合超过 logical budget K")
    selected = tuple(candidate_by_id[value] for value in selected_ids)
    continuations = snapshot.core_continuations()
    model = RecoveryCostModel()
    empty_cost = 0.0
    selected_cost = 0.0
    rows = []
    for continuation in continuations:
        target = continuation.planning_target
        empty_gap = recovery_gap(continuation, ())
        empty_item_cost = model.estimate(empty_gap, target)
        empty_cost += empty_item_cost
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        cost = (
            empty_item_cost
            if not selected
            else model.estimate(gap, target)
        )
        selected_cost += cost
        rows.append(
            ContinuationObjective(
                continuation_id=continuation.continuation_id,
                workflow_id=continuation.workflow_id,
                target_tokens=target,
                executable_frontier_tokens=frontier,
                recovery_gap_tokens=gap,
                recovery_cost_ms=cost,
            )
        )
    benefit = empty_cost - selected_cost
    if benefit < -_FLOAT_TOLERANCE_MS:
        raise RuntimeError("正式 objective 出现负收益")
    if benefit < 0.0:
        benefit = 0.0
    return ObjectiveEvaluation(
        selected_checkpoint_ids=selected_ids,
        total_recovery_cost_ms=selected_cost,
        empty_selection_cost_ms=empty_cost,
        total_benefit_ms=benefit,
        per_continuation=tuple(rows),
        objective_evaluation_count=1,
    )


def select_lfu(
    candidates: Sequence[CheckpointCandidate],
    access_frequency_by_checkpoint: Mapping[str, int],
    last_access_order_by_checkpoint: Mapping[str, int],
    logical_budget_k: int,
) -> tuple[str, ...]:
    """按冻结 LFU Adaptation 语义在预算内保留高访问频率候选。

    主序为访问频率降序；频率相同时按最近访问顺序降序，
    忠实映射 SGLang v0.5.17 native LFUStrategy 的淘汰优先级
    (hit_count, last_access_time) 最小堆语义（同频率下最久未访问者
    优先被淘汰，即最近访问者优先被保留）；再并列时按 checkpoint_id
    字典序升序，这是 LFU Adaptation 的确定性兜底规则。
    只消费冻结 metadata，不查询 runtime，不修改任何输入。
    """

    frequency = {
        str(checkpoint_id): value
        for checkpoint_id, value in access_frequency_by_checkpoint.items()
    }
    last_access = {
        str(checkpoint_id): value
        for checkpoint_id, value in last_access_order_by_checkpoint.items()
    }
    missing_frequency = sorted(
        item.checkpoint_id for item in candidates if item.checkpoint_id not in frequency
    )
    if missing_frequency:
        raise ValueError(
            f"缺少 checkpoint 访问频率 metadata：{', '.join(missing_frequency)}"
        )
    missing_access = sorted(
        item.checkpoint_id for item in candidates if item.checkpoint_id not in last_access
    )
    if missing_access:
        raise ValueError(
            f"缺少 checkpoint 最近访问 metadata：{', '.join(missing_access)}"
        )
    ordered = sorted(
        candidates,
        key=lambda item: (
            -frequency[item.checkpoint_id],
            -last_access[item.checkpoint_id],
            item.checkpoint_id,
        ),
    )
    capacity = min(logical_budget_k, len(ordered))
    return tuple(item.checkpoint_id for item in ordered[:capacity])


def select_exact_opt(
    snapshot: AllocationSnapshot,
    *,
    objective_function: ObjectiveFunction = evaluate_objective,
) -> ExactSelection:
    """用公共 objective 枚举所有不超过 K 的组合并确定最优集合。"""

    candidate_ids = tuple(
        item.checkpoint_id for item in snapshot.eligible_candidates
    )
    capacity = min(snapshot.logical_budget_k, len(candidate_ids))
    best_ids: tuple[str, ...] | None = None
    best_cost: float | None = None
    evaluations = 0
    for subset_size in range(capacity + 1):
        for subset in combinations(candidate_ids, subset_size):
            objective = objective_function(snapshot, subset)
            evaluations += objective.objective_evaluation_count
            cost = objective.total_recovery_cost_ms
            if (
                best_cost is None
                or cost < best_cost - _FLOAT_TOLERANCE_MS
                or (
                    abs(cost - best_cost) <= _FLOAT_TOLERANCE_MS
                    and (best_ids is None or subset < best_ids)
                )
            ):
                best_ids = subset
                best_cost = cost
    return ExactSelection(
        selected_checkpoint_ids=best_ids or (),
        selector_internal_evaluations=evaluations,
    )


def evaluate_allocation_snapshot(
    snapshot: AllocationSnapshot,
    *,
    objective_function: ObjectiveFunction = evaluate_objective,
) -> FivePolicyEvaluation:
    """按冻结顺序选择五种方法并用同一个公共 evaluator 最终评分。"""

    initial_digest = snapshot.content_digest()
    results = []
    for policy_name in ("LRU", "LFU", "Marconi", "FlowState", "Exact OPT"):
        digest_before = snapshot.content_digest()
        if digest_before != initial_digest:
            raise RuntimeError("策略运行前 allocation snapshot digest 已变化")
        started_ns = perf_counter_ns()
        selected_ids, internal_evaluations = _select_policy(
            policy_name,
            snapshot,
            objective_function,
        )
        selector_wall_time_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        objective = objective_function(snapshot, selected_ids)
        digest_after = snapshot.content_digest()
        if digest_after != digest_before:
            raise RuntimeError(
                f"{policy_name} 改变了 allocation snapshot digest"
            )
        results.append(
            PolicyEvaluation(
                policy_name=policy_name,
                selected_checkpoint_ids=objective.selected_checkpoint_ids,
                selector_wall_time_ms=selector_wall_time_ms,
                total_recovery_cost_ms=objective.total_recovery_cost_ms,
                empty_selection_cost_ms=objective.empty_selection_cost_ms,
                total_benefit_ms=objective.total_benefit_ms,
                per_continuation=objective.per_continuation,
                candidate_count=len(snapshot.eligible_candidates),
                pending_count=len(snapshot.pending_continuations),
                selector_internal_evaluations=internal_evaluations,
                final_common_scoring_evaluations=(
                    objective.objective_evaluation_count
                ),
                snapshot_digest_before=digest_before,
                snapshot_digest_after=digest_after,
            )
        )
    by_name = {item.policy_name: item for item in results}
    metrics = greedy_exact_metrics(
        by_name["FlowState"],
        by_name["Exact OPT"],
    )
    return FivePolicyEvaluation(
        snapshot_id=snapshot.snapshot_id,
        snapshot_digest=initial_digest,
        policy_results=tuple(results),
        flowstate_vs_exact=metrics,
    )


def greedy_exact_metrics(
    greedy: PolicyEvaluation,
    exact: PolicyEvaluation,
) -> GreedyExactMetrics:
    """按冻结零分母规则计算 greedy 相对 Exact OPT 的指标。"""

    absolute_gap = (
        greedy.total_recovery_cost_ms - exact.total_recovery_cost_ms
    )
    if absolute_gap < -_FLOAT_TOLERANCE_MS:
        raise ValueError("greedy 成本不能低于 Exact OPT")
    if abs(absolute_gap) <= _FLOAT_TOLERANCE_MS:
        absolute_gap = 0.0
    benefit_ratio = (
        greedy.total_benefit_ms / exact.total_benefit_ms
        if exact.total_benefit_ms > _FLOAT_TOLERANCE_MS
        else None
    )
    if exact.total_recovery_cost_ms > _FLOAT_TOLERANCE_MS:
        relative_gap = absolute_gap / exact.total_recovery_cost_ms
    elif absolute_gap == 0.0:
        relative_gap = 0.0
    else:
        relative_gap = None
    return GreedyExactMetrics(
        absolute_cost_gap_ms=absolute_gap,
        benefit_ratio=benefit_ratio,
        relative_cost_gap=relative_gap,
    )


def _select_policy(
    policy_name: str,
    snapshot: AllocationSnapshot,
    objective_function: ObjectiveFunction,
) -> tuple[tuple[str, ...], int]:
    """把统一 snapshot 转换为现有 selector 的隔离输入。"""

    candidates = snapshot.core_candidates()
    metadata = {
        item.checkpoint_id: item for item in snapshot.candidate_metadata
    }
    if policy_name == "LRU":
        recency = tuple(
            CheckpointRecency(
                checkpoint_id=item.checkpoint_id,
                creation_order=metadata[item.checkpoint_id].creation_order,
                last_access_order=metadata[item.checkpoint_id].last_access_order,
            )
            for item in candidates
        )
        selected = select_global_lru(
            candidates,
            recency,
            snapshot.budget_bytes,
        )
        return tuple(selected), 0
    if policy_name == "LFU":
        selected = select_lfu(
            candidates,
            {
                item.checkpoint_id: item.access_frequency
                for item in snapshot.lfu_access_frequency
            },
            {
                item.checkpoint_id: metadata[
                    item.checkpoint_id
                ].last_access_order
                for item in candidates
            },
            snapshot.logical_budget_k,
        )
        return tuple(selected), 0
    if policy_name == "Marconi":
        result = MarconiStylePolicy().select(
            candidates,
            snapshot.logical_budget_k,
            {
                item.checkpoint_id: float(
                    metadata[item.checkpoint_id].last_access_order
                )
                for item in candidates
            },
            {
                item.checkpoint_id: metadata[
                    item.checkpoint_id
                ].marconi_flop_saved
                for item in candidates
            },
            snapshot.marconi_alpha,
        )
        return tuple(result.selected_checkpoint_ids), 0
    if policy_name == "FlowState":
        model = _CountingRecoveryCostModel()
        result = GlobalOptimizer(model).select(
            snapshot.core_continuations(),
            candidates,
            snapshot.budget_bytes,
        )
        pending_count = len(snapshot.pending_continuations)
        if model.estimate_calls % pending_count != 0:
            raise RuntimeError("FlowState 内部 objective 计数无法整除 pending 数")
        return (
            tuple(item.checkpoint_id for item in result.selected),
            model.estimate_calls // pending_count,
        )
    if policy_name == "Exact OPT":
        result = select_exact_opt(
            snapshot,
            objective_function=objective_function,
        )
        return (
            result.selected_checkpoint_ids,
            result.selector_internal_evaluations,
        )
    raise ValueError(f"未知策略：{policy_name}")


def _validate_recovery_model_identity(snapshot: AllocationSnapshot) -> None:
    """拒绝使用与 snapshot 冻结身份不同的恢复模型进行评分。"""

    if snapshot.recovery_model != recovery_model_identity():
        raise ValueError("snapshot recovery model 身份与当前正式模型不一致")


def _validate_unique(values: Iterable[str], field_name: str) -> None:
    """验证一个可迭代标识集合没有重复值。"""

    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} 不能重复")


def _validate_exact_keys(
    checkpoint_ids: Sequence[str],
    values: Mapping[str, object],
    field_name: str,
) -> None:
    """验证按 checkpoint 索引的 metadata 没有缺项或多项。"""

    expected = set(checkpoint_ids)
    observed = {str(value) for value in values}
    if observed != expected:
        raise ValueError(f"{field_name} metadata 与 candidate set 不一致")


def _validate_nonnegative_integer(value: object, field_name: str) -> None:
    """验证字段是非布尔的非负整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数")


def _validate_nonempty_text(value: object, field_name: str) -> None:
    """验证字段是去除空白后仍非空的文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")


def _validate_digest(value: str, field_name: str) -> None:
    """验证稳定摘要采用六十四位十六进制 SHA-256 表示。"""

    if len(value) != 64 or any(
        character not in "0123456789abcdef"
        for character in value.lower()
    ):
        raise ValueError(f"{field_name} 必须是 SHA-256 摘要")
