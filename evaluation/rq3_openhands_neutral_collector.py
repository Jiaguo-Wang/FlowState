"""按冻结 protocol 从 OpenHands trajectories 中立采集 Main AllocationSnapshots。

本模块是 Step 13E 的 neutral collector：

- 每个 workflow group 使用独立 fresh Engine，组间不继承任何 runtime 状态；
- 只按冻结 replay protocol 把 A/B/C/D 执行到指定 round，然后只读冻结 snapshot；
- 不执行任何 policy selector、不执行 Exact OPT、不执行逻辑驱逐；
- 所有 runtime I/O 都经过 GroupRuntime 适配器接口，离线装配核为纯函数，
  使 collector correctness 可以在无 GPU 环境下完整测试。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Callable, Mapping, Protocol, Sequence


# 直接以脚本方式启动时，先把仓库根目录插入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation

from evaluation.openhands_common_barrier_snapshot_gate import (
    lineage_path,
    token_digest,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    build_allocation_snapshot,
)
from evaluation.rq3_openhands_population import (
    ALLOCATION_ROUNDS,
    WorkflowGroup,
    k_sweep_for_candidate_count,
    reference_budget_for_candidate_count,
)
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_marconi_flop_saved,
)


COLLECTOR_VERSION = "rq3-openhands-neutral-collector.v1"
ARTIFACT_SCHEMA = "flowstate.rq3_openhands_main_snapshots.v1"

# 失败主原因的全集；装配与驱动只产生这里列出的确定性原因。
FAILURE_REASONS = (
    "dataset_load_failed",
    "replay_input_incomplete",
    "non_monotonic_inputs",
    "engine_startup_failed",
    "request_failed",
    "runtime_metrics_invalid",
    "token_count_mismatch",
    "oom",
    "native_mamba_eviction",
    "fa_kv_cascade",
    "materialization_not_unique",
    "checkpoint_not_resident_at_barrier",
    "fa_frontier_query_side_effect",
    "future_boundary_violation",
    "candidate_count_below_8",
    "pending_without_compatible_candidate",
    "competition_insufficient",
    "lfu_provenance_mismatch",
    "snapshot_validation_failed",
    "duplicate_snapshot_digest",
    "duplicate_workflow_set",
    "worker_process_died",
    "worker_timeout",
    "unexpected_error",
)


class CollectionAbort(RuntimeError):
    """携带确定性主失败原因中断单个 group 的采集。"""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in FAILURE_REASONS:
            raise ValueError(f"未知失败原因：{reason}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def canonical_json(value: object) -> str:
    """返回与 Step 13B 一致风格的规范 JSON 文本。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_of_canonical(value: object) -> str:
    """计算规范 JSON 内容的 SHA-256 摘要。"""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


class GroupRuntime(Protocol):
    """collector 与 runtime 之间的最小交互面。"""

    def census(
        self,
        label: str,
        *,
        ordinal: int,
        request: Mapping[str, object] | None,
        previous: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        """返回一次 compact census（含 native eviction / FA cascade 推断）。"""

    def execute(
        self,
        request: Mapping[str, object],
        ordinal: int,
    ) -> Mapping[str, object]:
        """顺序执行一个 replay 请求并返回请求级记录。"""

    def inspect_checkpoint(
        self,
        probe_id: str,
        token_ids: Sequence[int],
    ) -> Mapping[str, object]:
        """返回一次 checkpoint path 检查响应。"""

    def inspect_fa_frontier(
        self,
        token_ids: Sequence[int],
        *,
        nonce: str,
    ) -> Mapping[str, object]:
        """返回一次无副作用 FA frontier 查询响应。"""

    def shutdown(self) -> None:
        """释放本 group 独占的全部 runtime 资源。"""


@dataclass(frozen=True)
class CheckpointObservation:
    """保存一个 checkpoint 截至 barrier 的全部中立观测事实。"""

    checkpoint_id: str
    workflow_label: str
    workflow_id: str
    turn: int
    token_pos: int
    node_id: int
    slots: tuple[int, ...]
    prefix_digest: str
    creation_order: int
    last_access_order: int
    fa_resident: bool
    recurrent_resident: bool
    contributing_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class PendingObservation:
    """保存一个 r+1 pending continuation 的只读构造证据。"""

    continuation_id: str
    workflow_label: str
    workflow_id: str
    anchor_pos: int
    resident_fa_frontier: int
    input_token_digest: str
    query_state_equal: bool
    scope_stable: bool
    traversed_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class GroupReplayTrace:
    """保存一个 group 截至 allocation round 的完整中立 replay 观测。"""

    group: WorkflowGroup
    executed_round: int
    checkpoints: tuple[CheckpointObservation, ...]
    pendings: tuple[PendingObservation, ...]
    residency_snapshot_digest: str
    request_rows: tuple[Mapping[str, object], ...]
    census_rows: tuple[Mapping[str, object], ...]
    boundary_audit: tuple[Mapping[str, object], ...]
    native_mamba_eviction: bool
    fa_cascade: bool
    oom: bool
    truncation: bool
    fa_query_side_effect_free: bool


@dataclass(frozen=True)
class AssemblyResult:
    """保存一个 group 的离线装配结论：成功快照或确定性失败。"""

    status: str
    primary_reason: str | None
    diagnostics: Mapping[str, object]
    snapshot: AllocationSnapshot | None
    lfu_provenance: tuple[Mapping[str, object], ...]
    k_sweep: tuple[int, ...]


def checkpoint_id_for(group_ordinal: int, label: str, turn: int) -> str:
    """生成确定且全局唯一的 checkpoint 标识。"""

    return f"RQ3_G{group_ordinal:03d}_{label}_TURN_{turn:03d}"


def continuation_id_for(group_ordinal: int, label: str, pending_turn: int) -> str:
    """生成确定且全局唯一的 pending continuation 标识。"""

    return f"RQ3_G{group_ordinal:03d}_{label}_PENDING_TURN_{pending_turn:03d}"


def request_id_for(group_ordinal: int, label: str, turn: int) -> str:
    """生成确定且全局唯一的 replay 请求标识。"""

    return f"rq3e-g{group_ordinal:03d}-{label.lower()}-turn-{turn:03d}"


def materialize_group_requests(
    tokenizer: object,
    messages_by_label: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    group: WorkflowGroup,
    normalize_message: Callable[[Mapping[str, object]], dict[str, object]],
    template_input_ids: Callable[[object], list[int]],
) -> tuple[dict[tuple[str, int], dict[str, object]], list[dict[str, object]]]:
    """只物化 <= round 的已执行输入与 round+1 的 pending 输入。

    round+1 的 assistant 输出内容不进入任何输入；
    round+2 及之后的消息完全不被消费。
    """

    pending_turn = group.allocation_round + 1
    requests: dict[tuple[str, int], dict[str, object]] = {}
    audits = []
    for label, session_id in group.session_by_label.items():
        raw_messages = messages_by_label[label]
        history: list[dict[str, object]] = []
        assistant_turn = 0
        raw_items_iterated = 0
        produced: list[int] = []
        for raw_message in raw_messages:
            raw_items_iterated += 1
            if raw_message.get("role") == "assistant":
                assistant_turn += 1
                if assistant_turn <= pending_turn:
                    output = tokenizer.apply_chat_template(
                        list(history),
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    requests[(label, assistant_turn)] = {
                        "workflow_label": label,
                        "workflow_id": session_id,
                        "turn": assistant_turn,
                        "rid": request_id_for(
                            group.group_ordinal,
                            label,
                            assistant_turn,
                        ),
                        "input_ids": template_input_ids(output),
                    }
                    produced.append(assistant_turn)
                if assistant_turn == pending_turn:
                    break
            history.append(normalize_message(raw_message))
        expected = tuple(range(1, pending_turn + 1))
        if tuple(produced) != expected:
            raise CollectionAbort(
                "replay_input_incomplete",
                f"group {group.group_ordinal} 的 {label} 缺少 turn {expected}",
            )
        audits.append(
            {
                "workflow_label": label,
                "session_id": session_id,
                "maximum_assistant_turn_consumed": pending_turn,
                "raw_items_iterated_through_pending_marker": raw_items_iterated,
                "pending_turn_output_read": False,
                "r_plus_2_message_consumed": False,
                "r_plus_2_request_materialized": False,
                "future_timing_read": False,
                "future_checkpoint_read": False,
            }
        )
    return requests, audits


def census_projection(census: Mapping[str, object]) -> Mapping[str, object]:
    """提取 census 中参与驻留摘要的稳定语义字段。"""

    return {
        "resident_mamba_nodes": [
            {
                "node_id": int(row["node_id"]),
                "slots": [int(value) for value in row["slots"]],
                "token_position": int(row["token_position"]),
            }
            for row in sorted(
                census["resident_mamba_nodes"],
                key=lambda item: int(item["node_id"]),
            )
        ],
        "full_device_node_ids": [
            int(value) for value in census["full_device_node_ids"]
        ],
        "mamba_available_slots": int(census["mamba_available_slots"]),
        "mamba_evictable_slots": int(census["mamba_evictable_slots"]),
        "mamba_protected_slots": int(census["mamba_protected_slots"]),
        "full_evictable_tokens": int(census["full_evictable_tokens"]),
        "full_protected_tokens": int(census["full_protected_tokens"]),
        "tree_node_count": int(census["tree_node_count"]),
    }


def replay_group_to_barrier(
    runtime: GroupRuntime,
    group: WorkflowGroup,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    partial_rows: dict[str, list] | None = None,
) -> GroupReplayTrace:
    """把四个 workflow 执行到 allocation round 并做 barrier 只读检查。

    partial_rows 非空时，逐请求证据会持续写入该 dict，
    使中途 CollectionAbort 也能保留完整诊断。
    """

    executed_round = group.allocation_round
    baseline = runtime.census(
        f"rq3e:g{group.group_ordinal:03d}:baseline",
        ordinal=0,
        request=None,
        previous=None,
    )
    if int(baseline["mamba_node_count"]) != 0:
        raise CollectionAbort(
            "engine_startup_failed",
            "fresh Engine 初始 census 含非空 Mamba checkpoint",
        )
    census_rows: list[Mapping[str, object]] = [
        {"event": "baseline", **dict(baseline)}
    ]
    request_rows: list[Mapping[str, object]] = []
    if partial_rows is not None:
        partial_rows["request_rows"] = request_rows
        partial_rows["census_rows"] = census_rows
    checkpoints: dict[str, dict[str, object]] = {}
    ordinal = 0
    previous = baseline
    for turn in range(1, executed_round + 1):
        for label in group.session_by_label:
            ordinal += 1
            request = requests[(label, turn)]
            input_ids = request["input_ids"]
            if not isinstance(input_ids, list):
                raise CollectionAbort(
                    "replay_input_incomplete",
                    f"{label}{turn} 的 input_ids 不是列表",
                )
            try:
                record = runtime.execute(request, ordinal)
            except Exception as error:
                lowered = repr(error).lower()
                if "out of memory" in lowered or "oom" in lowered:
                    raise CollectionAbort(
                        "oom", f"{label}{turn} 请求发生 OOM"
                    ) from error
                raise CollectionAbort(
                    "request_failed",
                    f"{label}{turn} 请求异常：{error!r}",
                ) from error
            census = runtime.census(
                f"rq3e:g{group.group_ordinal:03d}:after:{label}{turn}",
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            request_rows.append(dict(record))
            census_rows.append({"event": f"after_{label}{turn}", **dict(census)})
            if not record.get("request_completed"):
                lowered = repr(record.get("error")).lower()
                if "out of memory" in lowered or "oom" in lowered:
                    raise CollectionAbort(
                        "oom", f"{label}{turn} 请求发生 OOM"
                    )
                raise CollectionAbort(
                    "request_failed",
                    f"{label}{turn} 请求失败：{record.get('error')!r}",
                )
            if record.get("oom"):
                raise CollectionAbort("oom", f"{label}{turn} 请求报告 OOM")
            if not record.get("token_count_exact"):
                raise CollectionAbort(
                    "token_count_mismatch",
                    f"{label}{turn} 服务端 prompt_tokens 与离线长度不一致",
                )
            if record.get("runtime_metrics_valid") is not True:
                raise CollectionAbort(
                    "runtime_metrics_invalid",
                    f"{label}{turn} 的 H/E/G 指标无效",
                )
            if census["native_mamba_capacity_eviction_inferred"]:
                raise CollectionAbort(
                    "native_mamba_eviction",
                    f"{label}{turn} 执行中发生原生 Mamba 驱逐",
                )
            if census["fa_kv_cascade_eviction_inferred"]:
                raise CollectionAbort(
                    "fa_kv_cascade",
                    f"{label}{turn} 执行中发生 FA 级联",
                )
            added_ids = [
                int(value) for value in census["added_mamba_node_ids"]
            ]
            if len(added_ids) != 1:
                raise CollectionAbort(
                    "materialization_not_unique",
                    f"{label}{turn} 新增 Mamba 节点数不是 1：{added_ids}",
                )
            node_id = added_ids[0]
            resident = {
                int(row["node_id"]): row
                for row in census["resident_mamba_nodes"]
            }
            if node_id not in resident:
                raise CollectionAbort(
                    "materialization_not_unique",
                    f"新增节点 {node_id} 不在驻留集合",
                )
            token_pos = int(resident[node_id]["token_position"])
            if not 0 < token_pos <= len(input_ids):
                raise CollectionAbort(
                    "materialization_not_unique",
                    f"{label}{turn} checkpoint position 非法：{token_pos}",
                )
            same_workflow_positions = [
                int(row["token_pos"])
                for row in checkpoints.values()
                if row["workflow_label"] == label
            ]
            if same_workflow_positions and token_pos <= max(
                same_workflow_positions
            ):
                raise CollectionAbort(
                    "non_monotonic_inputs",
                    f"{label} 的 checkpoint position 未严格递增",
                )
            prefix_ids = tuple(int(value) for value in input_ids[:token_pos])
            prefix_digest = token_digest(prefix_ids)
            checkpoint_id = checkpoint_id_for(
                group.group_ordinal, label, turn
            )
            path = runtime.inspect_checkpoint(
                f"{checkpoint_id}_CREATE",
                prefix_ids,
            )["after"]["path"]
            if (
                int(path["node_id"]) != node_id
                or int(path["prefix_tokens"]) != token_pos
                or str(path["prefix_sha256"]) != prefix_digest
            ):
                raise CollectionAbort(
                    "materialization_not_unique",
                    f"{label}{turn} census 与 inspect 不一致",
                )
            executable_frontier = int(record.get("e") or 0)
            for row in checkpoints.values():
                if (
                    row["workflow_label"] == label
                    and int(row["token_pos"]) == executable_frontier
                ):
                    row["last_access_order"] = ordinal
            rid = str(request["rid"])
            for row in checkpoints.values():
                if (
                    row["workflow_label"] == label
                    and int(row["token_pos"]) <= token_pos
                ):
                    row["contributing_request_ids"].append(rid)
            checkpoints[checkpoint_id] = {
                "checkpoint_id": checkpoint_id,
                "workflow_label": label,
                "workflow_id": str(request["workflow_id"]),
                "turn": turn,
                "token_pos": token_pos,
                "node_id": node_id,
                "slots": tuple(
                    int(value) for value in resident[node_id]["slots"]
                ),
                "prefix_ids": prefix_ids,
                "prefix_digest": prefix_digest,
                "creation_order": ordinal,
                "last_access_order": ordinal,
                "fa_resident": bool(
                    path["target_full_present"]
                    and path["path_full_all_present"]
                ),
                "recurrent_resident": bool(path["target_mamba_present"]),
                "contributing_request_ids": [rid],
            }
            previous = census

    for row in checkpoints.values():
        path = runtime.inspect_checkpoint(
            f"{row['checkpoint_id']}_BARRIER",
            row["prefix_ids"],
        )["after"]["path"]
        if (
            int(path["prefix_tokens"]) != int(row["token_pos"])
            or str(path["prefix_sha256"]) != str(row["prefix_digest"])
        ):
            raise CollectionAbort(
                "checkpoint_not_resident_at_barrier",
                f"{row['checkpoint_id']} 的前缀 identity 发生变化",
            )
        row["node_id"] = int(path["node_id"])
        row["fa_resident"] = bool(
            path["target_full_present"] and path["path_full_all_present"]
        )
        row["recurrent_resident"] = bool(path["target_mamba_present"])
        raw_slots = path.get("target_mamba_slots") or ()
        row["slots"] = tuple(int(value) for value in raw_slots)

    final_census = runtime.census(
        f"rq3e:g{group.group_ordinal:03d}:barrier-final",
        ordinal=ordinal,
        request=None,
        previous=previous,
    )
    census_rows.append({"event": "barrier_final", **dict(final_census)})
    residency_snapshot_digest = digest_of_canonical(
        census_projection(final_census)
    )

    pending_turn = executed_round + 1
    pendings = []
    fa_query_clean = True
    for label, session_id in group.session_by_label.items():
        request = requests[(label, pending_turn)]
        input_ids = request["input_ids"]
        lookup = runtime.inspect_fa_frontier(
            input_ids,
            nonce=(
                f"rq3e:g{group.group_ordinal:03d}:{label}:"
                f"turn-{pending_turn:03d}"
            ),
        )
        state_equal = bool(lookup.get("state_equal"))
        scope_stable = lookup.get("scope_before") == lookup.get("scope_after")
        if not state_equal or not scope_stable:
            fa_query_clean = False
        pendings.append(
            PendingObservation(
                continuation_id=continuation_id_for(
                    group.group_ordinal,
                    label,
                    pending_turn,
                ),
                workflow_label=label,
                workflow_id=session_id,
                anchor_pos=len(input_ids),
                resident_fa_frontier=int(lookup["resident_fa_frontier"]),
                input_token_digest=token_digest(input_ids),
                query_state_equal=state_equal,
                scope_stable=bool(scope_stable),
                traversed_node_ids=tuple(
                    int(value) for value in lookup["traversed_node_ids"]
                ),
            )
        )
    observations = tuple(
        CheckpointObservation(
            checkpoint_id=str(row["checkpoint_id"]),
            workflow_label=str(row["workflow_label"]),
            workflow_id=str(row["workflow_id"]),
            turn=int(row["turn"]),
            token_pos=int(row["token_pos"]),
            node_id=int(row["node_id"]),
            slots=tuple(row["slots"]),
            prefix_digest=str(row["prefix_digest"]),
            creation_order=int(row["creation_order"]),
            last_access_order=int(row["last_access_order"]),
            fa_resident=bool(row["fa_resident"]),
            recurrent_resident=bool(row["recurrent_resident"]),
            contributing_request_ids=tuple(row["contributing_request_ids"]),
        )
        for _, row in sorted(checkpoints.items())
    )
    return GroupReplayTrace(
        group=group,
        executed_round=executed_round,
        checkpoints=observations,
        pendings=tuple(pendings),
        residency_snapshot_digest=residency_snapshot_digest,
        request_rows=tuple(request_rows),
        census_rows=tuple(census_rows),
        boundary_audit=(),
        native_mamba_eviction=False,
        fa_cascade=False,
        oom=False,
        truncation=False,
        fa_query_side_effect_free=fa_query_clean,
    )


def assemble_group_snapshot(
    trace: GroupReplayTrace,
    *,
    checkpoint_size_bytes: int,
) -> AssemblyResult:
    """把中立 replay 观测装配为 Step 13B/13D AllocationSnapshot。

    本函数为纯函数：不接触 runtime，不调用任何 policy selector，
    不调用 Exact OPT，不执行任何逻辑驱逐。
    """

    group = trace.group
    diagnostics: dict[str, object] = {
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "session_ids": list(group.session_ids),
    }

    def fail(reason: str, extra: Mapping[str, object]) -> AssemblyResult:
        if reason not in FAILURE_REASONS:
            raise ValueError(f"未知失败原因：{reason}")
        return AssemblyResult(
            status="BUILD_FAILED",
            primary_reason=reason,
            diagnostics={**diagnostics, **dict(extra)},
            snapshot=None,
            lfu_provenance=(),
            k_sweep=(),
        )

    if trace.native_mamba_eviction:
        return fail("native_mamba_eviction", {})
    if trace.fa_cascade:
        return fail("fa_kv_cascade", {})
    if trace.oom:
        return fail("oom", {})
    if trace.truncation:
        return fail("token_count_mismatch", {})
    if not trace.fa_query_side_effect_free:
        return fail("fa_frontier_query_side_effect", {})

    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=item.checkpoint_id,
            workflow_id=item.workflow_id,
            lineage_path=lineage_path(item.workflow_id),
            token_pos=item.token_pos,
            memory_bytes=checkpoint_size_bytes,
            recurrent_resident=item.recurrent_resident,
            fa_resident=item.fa_resident,
        )
        for item in trace.checkpoints
    )
    not_resident = sorted(
        item.checkpoint_id
        for item in trace.checkpoints
        if not item.recurrent_resident or not item.fa_resident
    )
    if not_resident:
        return fail(
            "checkpoint_not_resident_at_barrier",
            {"not_resident_checkpoint_ids": not_resident},
        )
    if len(candidates) < 8:
        return fail(
            "candidate_count_below_8",
            {"candidate_count": len(candidates)},
        )

    pendings = tuple(
        PendingContinuation(
            continuation_id=item.continuation_id,
            workflow_id=item.workflow_id,
            lineage_path=lineage_path(item.workflow_id),
            anchor_pos=item.anchor_pos,
            resident_fa_frontier=item.resident_fa_frontier,
        )
        for item in trace.pendings
    )
    candidate_by_workflow: dict[str, list[CheckpointCandidate]] = {}
    for candidate in candidates:
        candidate_by_workflow.setdefault(candidate.workflow_id, []).append(
            candidate
        )
    incompatible = []
    for pending in pendings:
        target = pending.planning_target
        compatible = [
            candidate
            for candidate in candidate_by_workflow.get(pending.workflow_id, [])
            if 0 < candidate.token_pos <= target
        ]
        if not compatible:
            incompatible.append(pending.continuation_id)
    if incompatible:
        return fail(
            "pending_without_compatible_candidate",
            {"incompatible_continuation_ids": incompatible},
        )
    competing_workflows = sum(
        1
        for pending in pendings
        if any(
            0 < candidate.token_pos <= pending.planning_target
            for candidate in candidate_by_workflow.get(pending.workflow_id, [])
        )
    )
    if competing_workflows < 2:
        return fail(
            "competition_insufficient",
            {"competing_workflow_count": competing_workflows},
        )

    provenance = []
    mismatch = []
    for item in trace.checkpoints:
        frequency = len(item.contributing_request_ids)
        if frequency < 1:
            mismatch.append(item.checkpoint_id)
        provenance.append(
            {
                "checkpoint_id": item.checkpoint_id,
                "workflow_label": item.workflow_label,
                "access_frequency": frequency,
                "contributing_request_ids": list(
                    item.contributing_request_ids
                ),
                "contributing_event_count": len(
                    item.contributing_request_ids
                ),
                "frequency_observed_through_epoch": trace.executed_round,
                "matches_frozen_frequency": frequency >= 1,
            }
        )
    if mismatch:
        return fail(
            "lfu_provenance_mismatch",
            {"zero_frequency_checkpoint_ids": mismatch},
        )

    snapshot_id = (
        f"rq3-openhands-main-g{group.group_ordinal:03d}-"
        f"round{group.allocation_round}"
    )
    reference_k = reference_budget_for_candidate_count(len(candidates))
    try:
        k_sweep = k_sweep_for_candidate_count(len(candidates))
    except ValueError as error:
        return fail("candidate_count_below_8", {"error": repr(error)})
    runtime_evidence = tuple(
        FrozenCheckpointRuntimeEvidence(
            checkpoint_id=item.checkpoint_id,
            node_id=item.node_id,
            runtime_identity_digest=item.prefix_digest,
            checkpoint_handle_digest=digest_of_canonical(
                {
                    "checkpoint_id": item.checkpoint_id,
                    "expected_node_id": item.node_id,
                    "expected_prefix_digest": item.prefix_digest,
                }
            ),
        )
        for item in trace.checkpoints
    )
    boundary = FrozenOnlineInformationBoundary(
        materialized_through_epoch=trace.executed_round,
        visible_continuation_ids=tuple(
            item.continuation_id for item in trace.pendings
        ),
    )
    try:
        snapshot = build_allocation_snapshot(
            allocation_epoch=trace.executed_round,
            snapshot_id=snapshot_id,
            pending_continuations=pendings,
            eligible_candidates=candidates,
            creation_order_by_checkpoint={
                item.checkpoint_id: item.creation_order
                for item in trace.checkpoints
            },
            last_access_order_by_checkpoint={
                item.checkpoint_id: item.last_access_order
                for item in trace.checkpoints
            },
            marconi_flop_saved_by_checkpoint=build_marconi_flop_saved(
                candidates
            ),
            access_frequency_by_checkpoint={
                item.checkpoint_id: len(item.contributing_request_ids)
                for item in trace.checkpoints
            },
            frequency_observed_through_epoch=trace.executed_round,
            marconi_alpha=CONTROLLED_MARCONI_ALPHA,
            logical_budget_k=reference_k,
            budget_bytes=reference_k * checkpoint_size_bytes,
            runtime_evidence=runtime_evidence,
            residency_snapshot_digest=trace.residency_snapshot_digest,
            online_boundary=boundary,
        )
    except ValueError as error:
        return fail("snapshot_validation_failed", {"error": repr(error)})
    return AssemblyResult(
        status="ELIGIBLE",
        primary_reason=None,
        diagnostics={
            **diagnostics,
            "candidate_count": len(candidates),
            "pending_count": len(pendings),
            "k_sweep": list(k_sweep),
            "reference_budget_k": reference_k,
        },
        snapshot=snapshot,
        lfu_provenance=tuple(provenance),
        k_sweep=k_sweep,
    )


def write_snapshot_artifact(
    directory: Path,
    group: WorkflowGroup,
    snapshot: AllocationSnapshot,
) -> Path:
    """写出一个 canonical snapshot artifact 并返回路径。"""

    directory.mkdir(parents=True, exist_ok=True)
    digest = snapshot.content_digest()
    path = directory / f"g{group.group_ordinal:03d}_{digest[:12]}.json"
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "allocation_snapshot",
        "group_ordinal": group.group_ordinal,
        "population_segment": group.population_segment,
        "allocation_round": group.allocation_round,
        "session_ids": list(group.session_ids),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_digest": digest,
        "canonical_snapshot": snapshot.canonical_serialization(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_allocation_snapshot(path: Path) -> AllocationSnapshot:
    """从 canonical artifact 重建 snapshot 并校验 digest 一致。"""

    from evaluation.rq3_frozen_snapshot_evaluator import (
        FrozenAccessFrequency,
        FrozenCandidateMetadata,
        FrozenCheckpointCandidate,
        FrozenCheckpointRuntimeEvidence,
        FrozenOnlineInformationBoundary,
        FrozenPendingContinuation,
        FrozenRecoveryModelIdentity,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = payload["canonical_snapshot"]
    raw = json.loads(canonical)
    snapshot = AllocationSnapshot(
        allocation_epoch=int(raw["allocation_epoch"]),
        snapshot_id=str(raw["snapshot_id"]),
        pending_continuations=tuple(
            FrozenPendingContinuation(
                continuation_id=str(item["continuation_id"]),
                workflow_id=str(item["workflow_id"]),
                lineage_path=tuple(str(v) for v in item["lineage_path"]),
                anchor_pos=int(item["anchor_pos"]),
                resident_fa_frontier=int(item["resident_fa_frontier"]),
            )
            for item in raw["pending_continuations"]
        ),
        eligible_candidates=tuple(
            FrozenCheckpointCandidate(
                checkpoint_id=str(item["checkpoint_id"]),
                workflow_id=str(item["workflow_id"]),
                lineage_path=tuple(str(v) for v in item["lineage_path"]),
                token_pos=int(item["token_pos"]),
                memory_bytes=int(item["memory_bytes"]),
                recurrent_resident=bool(item["recurrent_resident"]),
                fa_resident=bool(item["fa_resident"]),
            )
            for item in raw["eligible_candidates"]
        ),
        candidate_metadata=tuple(
            FrozenCandidateMetadata(
                checkpoint_id=str(item["checkpoint_id"]),
                creation_order=int(item["creation_order"]),
                last_access_order=int(item["last_access_order"]),
                marconi_flop_saved=float(item["marconi_flop_saved"]),
            )
            for item in raw["candidate_metadata"]
        ),
        lfu_access_frequency=tuple(
            FrozenAccessFrequency(
                checkpoint_id=str(item["checkpoint_id"]),
                access_frequency=int(item["access_frequency"]),
            )
            for item in raw["lfu_access_frequency"]
        ),
        frequency_observed_through_epoch=int(
            raw["frequency_observed_through_epoch"]
        ),
        marconi_alpha=float(raw["marconi_alpha"]),
        recovery_model=FrozenRecoveryModelIdentity(
            **{
                key: value
                for key, value in raw["recovery_model"].items()
            }
        ),
        logical_budget_k=int(raw["logical_budget_k"]),
        budget_bytes=int(raw["budget_bytes"]),
        runtime_evidence=tuple(
            FrozenCheckpointRuntimeEvidence(
                checkpoint_id=str(item["checkpoint_id"]),
                node_id=int(item["node_id"]),
                runtime_identity_digest=str(item["runtime_identity_digest"]),
                checkpoint_handle_digest=str(item["checkpoint_handle_digest"]),
            )
            for item in raw["runtime_evidence"]
        ),
        residency_snapshot_digest=str(raw["residency_snapshot_digest"]),
        online_boundary=FrozenOnlineInformationBoundary(
            materialized_through_epoch=int(
                raw["online_boundary"]["materialized_through_epoch"]
            ),
            visible_continuation_ids=tuple(
                str(v)
                for v in raw["online_boundary"]["visible_continuation_ids"]
            ),
            future_continuation_included=bool(
                raw["online_boundary"]["future_continuation_included"]
            ),
            future_request_included=bool(
                raw["online_boundary"]["future_request_included"]
            ),
            future_latency_included=bool(
                raw["online_boundary"]["future_latency_included"]
            ),
        ),
    )
    if snapshot.content_digest() != payload["snapshot_digest"]:
        raise RuntimeError(f"{path} 的 canonical snapshot digest 不一致")
    if snapshot.canonical_serialization() != canonical:
        raise RuntimeError(f"{path} 的 canonical 序列化不一致")
    return snapshot


def write_failure_artifact(
    directory: Path,
    group: WorkflowGroup,
    reason: str,
    diagnostics: Mapping[str, object],
) -> Path:
    """写出一个 BUILD_FAILED 诊断 artifact。"""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"g{group.group_ordinal:03d}.json"
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "build_failed",
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "session_ids": list(group.session_ids),
        "status": "BUILD_FAILED",
        "primary_reason": reason,
        "diagnostics": dict(diagnostics),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def write_provenance_artifact(
    directory: Path,
    group: WorkflowGroup,
    snapshot: AllocationSnapshot,
    provenance: Sequence[Mapping[str, object]],
) -> Path:
    """写出一个 snapshot 对应的 LFU frequency provenance artifact。"""

    directory.mkdir(parents=True, exist_ok=True)
    digest = snapshot.content_digest()
    path = directory / f"g{group.group_ordinal:03d}_{digest[:12]}.json"
    frozen_frequency = {
        item.checkpoint_id: item.access_frequency
        for item in snapshot.lfu_access_frequency
    }
    rows = []
    for row in provenance:
        frozen = frozen_frequency[str(row["checkpoint_id"])]
        rows.append(
            {
                **dict(row),
                "frozen_access_frequency": frozen,
                "matches_frozen_frequency": bool(
                    int(row["access_frequency"]) == frozen
                    and int(row["contributing_event_count"]) == frozen
                ),
            }
        )
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "lfu_provenance",
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_digest": digest,
        "frequency_observed_through_epoch": (
            snapshot.frequency_observed_through_epoch
        ),
        "rows": rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def write_runtime_correctness_artifact(
    directory: Path,
    group: WorkflowGroup,
    trace: GroupReplayTrace | None,
    extra: Mapping[str, object],
) -> Path:
    """写出一个 group 的 runtime correctness 检查 artifact。"""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"g{group.group_ordinal:03d}.json"
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "runtime_correctness",
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "request_rows": (
            [dict(row) for row in trace.request_rows]
            if trace is not None
            else []
        ),
        "census_rows": (
            [dict(row) for row in trace.census_rows]
            if trace is not None
            else []
        ),
        "boundary_audit": (
            [dict(row) for row in trace.boundary_audit]
            if trace is not None
            else []
        ),
        **dict(extra),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def collect_group_verdict(
    group: WorkflowGroup,
    *,
    trace: GroupReplayTrace,
    checkpoint_size_bytes: int,
) -> Mapping[str, object]:
    """对一个已完成 replay 的 group 给出 ELIGIBLE / BUILD_FAILED 判定。"""

    result = assemble_group_snapshot(
        trace,
        checkpoint_size_bytes=checkpoint_size_bytes,
    )
    verdict: dict[str, object] = {
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "session_ids": list(group.session_ids),
        "status": result.status,
        "primary_reason": result.primary_reason,
        "diagnostics": dict(result.diagnostics),
    }
    if result.snapshot is not None:
        verdict["snapshot_id"] = result.snapshot.snapshot_id
        verdict["snapshot_digest"] = result.snapshot.content_digest()
        verdict["candidate_count"] = len(result.snapshot.eligible_candidates)
        verdict["pending_count"] = len(result.snapshot.pending_continuations)
        verdict["k_sweep"] = list(result.k_sweep)
    return verdict


def run_designated_groups(
    groups: Sequence[WorkflowGroup],
    collect_one: Callable[[WorkflowGroup], Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """严格按冻结顺序 attempt 全部 designated groups，不做任何补位。"""

    verdicts = []
    for group in groups:
        verdict = collect_one(group)
        if int(verdict["group_ordinal"]) != group.group_ordinal:
            raise RuntimeError("collector 返回了错误的 group ordinal")
        verdicts.append(dict(verdict))
    attempted = tuple(int(row["group_ordinal"]) for row in verdicts)
    designated = tuple(group.group_ordinal for group in groups)
    if attempted != designated:
        raise RuntimeError("attempted groups 与 designated groups 不一致")
    return verdicts


def summarize_population(
    verdicts: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """汇总 policy-independent 的 population 统计。"""

    eligible = [row for row in verdicts if row["status"] == "ELIGIBLE"]
    failed = [row for row in verdicts if row["status"] == "BUILD_FAILED"]
    reason_distribution: dict[str, int] = {}
    for row in failed:
        reason = str(row["primary_reason"])
        reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
    epoch_distribution = {str(round_id): 0 for round_id in ALLOCATION_ROUNDS}
    for row in verdicts:
        epoch_distribution[str(int(row["allocation_round"]))] += 1
    digests = [
        str(row["snapshot_digest"]) for row in eligible if row.get("snapshot_digest")
    ]
    workflow_sets = [
        tuple(sorted(row["session_ids"])) for row in verdicts
    ]
    return {
        "designated_groups": len(verdicts),
        "attempted_groups": len(verdicts),
        "eligible_count": len(eligible),
        "failed_count": len(failed),
        "failure_reason_distribution": reason_distribution,
        "epoch_distribution": epoch_distribution,
        "snapshot_digest_unique": len(set(digests)) == len(digests),
        "workflow_set_unique": len(set(workflow_sets)) == len(workflow_sets),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """按线性插值计算一个确定性分位数。"""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Mapping[str, object]:
    """返回 min/mean/median/p25/p75/p95/max 的统一摘要。"""

    materialized = [float(value) for value in values]
    if not materialized:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": min(materialized),
        "mean": sum(materialized) / len(materialized),
        "median": _percentile(materialized, 0.50),
        "p25": _percentile(materialized, 0.25),
        "p75": _percentile(materialized, 0.75),
        "p95": _percentile(materialized, 0.95),
        "max": max(materialized),
    }


def population_statistics(
    verdicts: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """计算任务要求的全部 policy-independent population 统计。"""

    eligible = [row for row in verdicts if row["status"] == "ELIGIBLE"]
    candidate_counts = [
        float(row["candidate_count"])
        for row in eligible
        if row.get("candidate_count") is not None
    ]
    pending_counts = [
        float(row["pending_count"])
        for row in eligible
        if row.get("pending_count") is not None
    ]
    workflow_counts = [
        float(len({str(value) for value in row["session_ids"]}))
        for row in eligible
    ]
    frequencies = [
        float(value)
        for row in eligible
        for value in dict(row.get("frequency_by_checkpoint", {})).values()
    ]
    frequency_summary = _distribution(frequencies)
    pending_summary = _distribution(pending_counts)
    workflow_summary = _distribution(workflow_counts)
    return {
        "candidate_count_distribution": _distribution(candidate_counts),
        "pending_count_distribution": {
            "min": pending_summary["min"],
            "median": pending_summary["median"],
            "max": pending_summary["max"],
        },
        "workflow_count_distribution": {
            "min": workflow_summary["min"],
            "median": workflow_summary["median"],
            "max": workflow_summary["max"],
        },
        "lfu_frequency_distribution": {
            "min": frequency_summary["min"],
            "median": frequency_summary["median"],
            "p75": frequency_summary["p75"],
            "p95": frequency_summary["p95"],
            "max": frequency_summary["max"],
        },
    }


def protocol_payload(
    *,
    environment: Mapping[str, object],
    engine_configuration: Mapping[str, object],
    checkpoint_size_bytes: int,
) -> Mapping[str, object]:
    """完整记录当前冻结 protocol，供后续正式评价离线恢复。"""

    return {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "frozen_protocol",
        "protocol_version": "rq3-openhands-v1",
        "seed": 20260903,
        "dataset": {
            "name": "nebius-swe-rebench-openhands / openhands",
            "local_path": (
                "/home/wjg/data/agentic_coding_trajectories/sessions.parquet"
            ),
            "source_dataset_filter": "nebius-swe-rebench-openhands",
            "eligibility": {
                "min_n_turns": 60,
                "max_replay_input_tokens": 131_072,
                "required_assistant_turns": 6,
            },
        },
        "sampling": {
            "ordering": (
                "SHA256('rq3-openhands-v1|20260903|' + session_id) 升序"
            ),
            "group_size": 4,
            "group_labeling": "组内按 hash 顺序标记 A/B/C/D",
            "main_groups": 200,
            "reserved_sensitivity_groups": 100,
            "no_substitution": (
                "指定 group 失败时不得用第 201+ group 补位"
            ),
        },
        "allocation_epoch": {
            "rule": "group ordinal 确定性轮转 round 2/3/4/5",
            "distribution": {"2": 50, "3": 50, "4": 50, "5": 50},
            "snapshot_timing": (
                "round r 全部完成后、round r+1 请求执行前冻结"
            ),
        },
        "k_protocol": {
            "main_relative_budgets": [0.25, 0.50, 0.75],
            "k_rule": "K = max(1, floor(r * |C_t|))，相同 K 只评价一次",
            "sensitivity_point": 2,
            "trivial_exclusion": "K >= |C_t| 不进入主比较",
            "reference_budget_in_snapshot": (
                "logical_budget_k = max(1, floor(0.75 * |C_t|))，"
                "正式 K sweep 在评价阶段按 K 派生快照"
            ),
        },
        "runtime": {
            "engine_configuration": dict(engine_configuration),
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "physical_mamba_pool": int(
                engine_configuration["max_mamba_cache_size"]
            ),
            "fresh_engine_per_group": True,
            "neutral_collection": (
                "不执行任何 policy retention / 逻辑驱逐 / Exact OPT"
            ),
        },
        "online_information_boundary": {
            "executed_history": "<= round r 的已完成 workflow 历史",
            "pending": "round r+1 的当前 input ids 与 dependency",
            "forbidden": (
                "r+2+ 请求、branch、output、tool result、access、"
                "timing、checkpoint、TTFT"
            ),
        },
        "lfu_adaptation": {
            "positioning": (
                "LFU Adaptation（generic frequency-based retention "
                "baseline），非 SGLang-native recurrent LFU"
            ),
            "access_event": (
                "观测时点前已完成 write-back insert 的请求，其最终 "
                "token span 覆盖 checkpoint 在自身 lineage 上的 token_pos"
            ),
            "initial_frequency": "创建前为 0，创建请求写回即首次 access",
            "increment": "同一请求对同一 checkpoint 最多 +1",
            "ancestor": "后续请求 insert path 覆盖祖先时祖先 +1",
            "pure_prefix_match": "不计 frequency",
            "observation_boundary": (
                "frequency_observed_through_epoch <= "
                "materialized_through_epoch <= allocation_epoch"
            ),
        },
        "eligibility_rules": [
            "|P_t| = 4 且分属四个 active workflows",
            "|C_t| >= 8",
            "每个 pending 至少一个 compatible checkpoint",
            "至少两个 workflow 存在 allocation competition",
            "candidates 全部 recurrent_resident 且 fa_resident",
            "runtime evidence 完整（node/handle/prefix digest/residency）",
            "checkpoint equal-size",
            "access_frequency >= 1 且 provenance 一致",
            "resident_fa_frontier 来自无副作用 barrier introspection",
            "native Mamba eviction / rematerialization / FA cascade / "
            "OOM / truncation 全为 0",
            "future-information boundary PASS",
            "workflow set 与 snapshot digest 不重复",
        ],
        "collector_version": COLLECTOR_VERSION,
        "environment": dict(environment),
    }


def allocation_round_distribution(
    groups: Sequence[WorkflowGroup],
) -> Mapping[str, int]:
    """统计 designated groups 的 allocation round 分布。"""

    distribution = {str(round_id): 0 for round_id in ALLOCATION_ROUNDS}
    for group in groups:
        distribution[str(group.allocation_round)] += 1
    return distribution


@dataclass(frozen=True)
class ArtifactDirectories:
    """保存一次正式采集的全部 artifact 子目录。"""

    root: Path
    snapshots: Path
    failures: Path
    lfu_provenance: Path
    runtime_correctness: Path
    eligibility: Path
    verdicts: Path
    workers: Path


def prepare_artifact_directories(root: Path) -> ArtifactDirectories:
    """创建正式采集的 artifact 目录结构。"""

    root.mkdir(parents=True, exist_ok=True)
    directories = ArtifactDirectories(
        root=root,
        snapshots=root / "snapshots",
        failures=root / "failures",
        lfu_provenance=root / "lfu_provenance",
        runtime_correctness=root / "runtime_correctness",
        eligibility=root / "eligibility",
        verdicts=root / "verdicts",
        workers=root / "workers",
    )
    for directory in (
        directories.snapshots,
        directories.failures,
        directories.lfu_provenance,
        directories.runtime_correctness,
        directories.eligibility,
        directories.verdicts,
        directories.workers,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _boundary_audit_has_leakage(
    audits: Sequence[Mapping[str, object]],
) -> bool:
    """判断 request 物化边界是否消费了 r+2 或 pending 输出。"""

    return any(
        audit.get("pending_turn_output_read") is not False
        or audit.get("r_plus_2_message_consumed") is not False
        or audit.get("r_plus_2_request_materialized") is not False
        or audit.get("future_timing_read") is not False
        or audit.get("future_checkpoint_read") is not False
        for audit in audits
    )


def collect_group(
    group: WorkflowGroup,
    *,
    runtime_factory: Callable[[], GroupRuntime],
    messages_by_label: Mapping[str, Sequence[Mapping[str, object]]],
    tokenizer: object,
    normalize_message: Callable[[Mapping[str, object]], dict[str, object]],
    template_input_ids: Callable[[object], list[int]],
    checkpoint_size_bytes: int,
    directories: ArtifactDirectories,
) -> Mapping[str, object]:
    """在独立 fresh runtime 中采集一个 group 并写出全部 artifact。

    任何失败都只产生 BUILD_FAILED 判定，不做补位、不改 epoch、
    不删 candidate、不调整任何配置。
    """

    runtime = None
    trace: GroupReplayTrace | None = None
    shutdown_error = None
    partial_rows: dict[str, list] = {}
    try:
        requests, audits = materialize_group_requests(
            tokenizer,
            messages_by_label,
            group=group,
            normalize_message=normalize_message,
            template_input_ids=template_input_ids,
        )
        if _boundary_audit_has_leakage(audits):
            raise CollectionAbort(
                "future_boundary_violation",
                "request 物化消费了 r+2 或 pending 输出",
            )
        try:
            runtime = runtime_factory()
        except CollectionAbort:
            raise
        except Exception as error:
            raise CollectionAbort(
                "engine_startup_failed", repr(error)
            ) from error
        trace = replay_group_to_barrier(
            runtime,
            group,
            requests,
            partial_rows,
        )
        trace = replace(trace, boundary_audit=tuple(audits))
    except CollectionAbort as abort:
        diagnostics: dict[str, object] = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "detail": abort.detail,
        }
        write_runtime_correctness_artifact(
            directories.runtime_correctness,
            group,
            trace,
            {
                "abort_reason": abort.reason,
                "abort_detail": abort.detail,
                "partial_request_rows": [
                    dict(row) for row in partial_rows.get("request_rows", ())
                ],
                "partial_census_rows": [
                    dict(row) for row in partial_rows.get("census_rows", ())
                ],
            },
        )
        write_failure_artifact(
            directories.failures,
            group,
            abort.reason,
            diagnostics,
        )
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": abort.reason,
            "diagnostics": diagnostics,
        }
    except Exception as error:
        traceback.print_exc()
        diagnostics = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "detail": repr(error),
        }
        write_runtime_correctness_artifact(
            directories.runtime_correctness,
            group,
            trace,
            {"abort_reason": "unexpected_error", "abort_detail": repr(error)},
        )
        write_failure_artifact(
            directories.failures,
            group,
            "unexpected_error",
            diagnostics,
        )
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": "unexpected_error",
            "diagnostics": diagnostics,
        }
    finally:
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    result = assemble_group_snapshot(
        trace,
        checkpoint_size_bytes=checkpoint_size_bytes,
    )
    if result.status != "ELIGIBLE" or result.snapshot is None:
        reason = result.primary_reason or "unexpected_error"
        write_runtime_correctness_artifact(
            directories.runtime_correctness,
            group,
            trace,
            {"abort_reason": reason, "shutdown_error": shutdown_error},
        )
        write_failure_artifact(
            directories.failures,
            group,
            reason,
            result.diagnostics,
        )
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": reason,
            "diagnostics": dict(result.diagnostics),
        }

    snapshot_path = write_snapshot_artifact(
        directories.snapshots,
        group,
        result.snapshot,
    )
    provenance_path = write_provenance_artifact(
        directories.lfu_provenance,
        group,
        result.snapshot,
        result.lfu_provenance,
    )
    write_runtime_correctness_artifact(
        directories.runtime_correctness,
        group,
        trace,
        {"abort_reason": None, "shutdown_error": shutdown_error},
    )
    reloaded = load_allocation_snapshot(snapshot_path)
    if reloaded.content_digest() != result.snapshot.content_digest():
        raise RuntimeError("snapshot artifact 写读后 digest 不一致")
    verdict: dict[str, object] = {
        "group_ordinal": group.group_ordinal,
        "allocation_round": group.allocation_round,
        "session_ids": list(group.session_ids),
        "status": "ELIGIBLE",
        "primary_reason": None,
        "diagnostics": dict(result.diagnostics),
        "snapshot_id": result.snapshot.snapshot_id,
        "snapshot_digest": result.snapshot.content_digest(),
        "candidate_count": len(result.snapshot.eligible_candidates),
        "pending_count": len(result.snapshot.pending_continuations),
        "k_sweep": list(result.k_sweep),
        "snapshot_artifact": str(snapshot_path),
        "lfu_provenance_artifact": str(provenance_path),
        "shutdown_error": shutdown_error,
    }
    return verdict


class SGLangGroupRuntime:
    """包装 frozen Engine 与控制端口的真实 runtime 适配器。"""

    def __init__(self, engine: object, client: object) -> None:
        from evaluation.barrier_fa_frontier_control import (
            BarrierFAControlClient,
        )

        self._engine = engine
        self._client = client
        self._barrier_client = BarrierFAControlClient(client)

    def census(
        self,
        label: str,
        *,
        ordinal: int,
        request: Mapping[str, object] | None,
        previous: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        from evaluation.openhands_4workflow_occupancy_calibration import (
            compact_census,
        )

        return compact_census(
            self._client.census(label),
            ordinal=ordinal,
            request=request,
            previous=previous,
        )

    def execute(
        self,
        request: Mapping[str, object],
        ordinal: int,
    ) -> Mapping[str, object]:
        from evaluation.openhands_4workflow_occupancy_calibration import (
            execute_request,
        )

        return execute_request(self._engine, self._client, request, ordinal)

    def inspect_checkpoint(
        self,
        probe_id: str,
        token_ids: Sequence[int],
    ) -> Mapping[str, object]:
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            inspect_checkpoint,
        )

        return inspect_checkpoint(
            self._client,
            probe_id,
            tuple(int(value) for value in token_ids),
        )

    def inspect_fa_frontier(
        self,
        token_ids: Sequence[int],
        *,
        nonce: str,
    ) -> Mapping[str, object]:
        return self._barrier_client.inspect_fa_frontier(
            [int(value) for value in token_ids],
            extra_key=None,
            limit=None,
            nonce=nonce,
        )

    def shutdown(self) -> None:
        self._engine.shutdown()


def sglang_runtime_factory(
    engine_configuration: Mapping[str, object],
    control_port: int,
) -> Callable[[], GroupRuntime]:
    """构造每个 group 独立 fresh Engine 的真实工厂。"""

    def factory() -> GroupRuntime:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import FormalEndToEndGateEngine

        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            wait_for_transport,
        )

        engine = FormalEndToEndGateEngine(**dict(engine_configuration))
        client = ControlClient(control_port)
        wait_for_transport(client)
        return SGLangGroupRuntime(engine, client)

    return factory


GPU_STABLE_POLL_INTERVAL_S = 5.0
GPU_STABLE_REQUIRED_OBSERVATIONS = 3
GPU_STABLE_THRESHOLD_MIB = 4096
GPU_STABLE_TIMEOUT_S = 300.0
WORKER_TIMEOUT_S = 1800.0


def query_gpu_memory_used_mib(gpu_index: int = 0) -> int:
    """通过 nvidia-smi 读取当前可见 GPU 的已用显存。"""

    import subprocess

    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(output.stdout.strip().splitlines()[0])


def query_gpu_compute_processes(gpu_index: int = 0) -> list[dict[str, object]]:
    """读取当前可见 GPU 上的 compute 进程列表。"""

    import subprocess

    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, object]] = []
    text = output.stdout.strip()
    if text:
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "pid": int(parts[0]),
                    "used_mib": (
                        int(parts[1]) if len(parts) > 1 else None
                    ),
                }
            )
    return rows


def wait_gpu_stable(
    *,
    gpu_index: int = 0,
    threshold_mib: int = GPU_STABLE_THRESHOLD_MIB,
    required_observations: int = GPU_STABLE_REQUIRED_OBSERVATIONS,
    interval_s: float = GPU_STABLE_POLL_INTERVAL_S,
    timeout_s: float = GPU_STABLE_TIMEOUT_S,
) -> Mapping[str, object]:
    """连续多次观测显存与实验进程均稳定干净后才返回。

    只依赖固定秒数 sleep 不足以证明前序 Engine 已释放；
    必须连续 required_observations 次同时满足：
    显存低于阈值且 compute 进程列表为空。
    """

    import time

    observations: list[dict[str, object]] = []
    streak = 0
    started = time.monotonic()
    while True:
        used = query_gpu_memory_used_mib(gpu_index)
        processes = query_gpu_compute_processes(gpu_index)
        clean = used <= threshold_mib and not processes
        streak = streak + 1 if clean else 0
        observations.append(
            {
                "used_mib": used,
                "compute_processes": processes,
                "clean": clean,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        if streak >= required_observations:
            return {
                "stable": True,
                "observations": observations,
                "final_used_mib": used,
            }
        if time.monotonic() - started > timeout_s:
            raise RuntimeError(
                "等待 GPU 稳定干净状态超时："
                f"used={used}MiB processes={processes}"
            )
        time.sleep(interval_s)


def build_worker_command(
    artifact_root: Path,
    group_ordinal: int,
) -> list[str]:
    """构造启动单个 group worker 子进程的命令。"""

    import sys as _sys

    return [
        _sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--artifact-root",
        str(artifact_root),
        "--group-ordinal",
        str(group_ordinal),
    ]


def run_group_via_worker(
    group: WorkflowGroup,
    *,
    directories: ArtifactDirectories,
    gpu_wait_record: Mapping[str, object],
    worker_command: Sequence[str] | None = None,
    worker_timeout_s: float = WORKER_TIMEOUT_S,
) -> Mapping[str, object]:
    """用独立子进程采集一个 group，父进程安全回收任何 worker 结局。

    worker 独占进程组（start_new_session），其内部 Engine 崩溃、
    kill_process_tree 或超时都不会波及 parent collector。
    """

    import os
    import signal
    import subprocess

    verdict_path = directories.verdicts / f"g{group.group_ordinal:03d}.json"
    if verdict_path.exists():
        verdict_path.unlink()
    log_path = directories.workers / f"g{group.group_ordinal:03d}.log"
    command = list(worker_command) if worker_command else build_worker_command(
        directories.root,
        group.group_ordinal,
    )
    lifecycle: dict[str, object] = {
        "group_ordinal": group.group_ordinal,
        "gpu_wait": dict(gpu_wait_record),
    }
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    timed_out = False
    try:
        exit_code = process.wait(timeout=worker_timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        exit_code = process.returncode
    lifecycle["timed_out"] = timed_out
    lifecycle["exit_code"] = exit_code

    verdict: dict[str, object] | None = None
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    elif not timed_out and exit_code == 0:
        for failure_path in directories.failures.glob(
            f"g{group.group_ordinal:03d}.json"
        ):
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            verdict = {
                "group_ordinal": group.group_ordinal,
                "allocation_round": group.allocation_round,
                "session_ids": list(group.session_ids),
                "status": "BUILD_FAILED",
                "primary_reason": payload["primary_reason"],
                "diagnostics": payload.get("diagnostics", {}),
            }
            break
    if verdict is None:
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(
                encoding="utf-8", errors="replace"
            )[-20000:]
        if timed_out:
            reason = "worker_timeout"
        elif (
            "leave no GPU memory" in log_text
            or "scheduler died" in log_text
            or "Received sigquit" in log_text
        ):
            reason = "engine_startup_failed"
        else:
            reason = "worker_process_died"
        diagnostics = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "worker_log": str(log_path),
        }
        write_failure_artifact(
            directories.failures,
            group,
            reason,
            diagnostics,
        )
        verdict = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": reason,
            "diagnostics": diagnostics,
        }
    verdict["worker_lifecycle"] = lifecycle
    _write_json(directories.verdicts / f"g{group.group_ordinal:03d}.json", verdict)
    return verdict


def run_group_worker(
    group_ordinal: int,
    artifact_root: Path,
) -> int:
    """worker 入口：采集单个 group 并把 verdict 写回 artifact root。"""

    from transformers import AutoTokenizer

    from evaluation.openhands_common_barrier_snapshot_gate import (
        ENGINE_CONFIGURATION_COMMON_BARRIER,
    )
    from evaluation.openhands_single_workflow_smoke import (
        DATASET_PATH,
        TOKENIZER_PATH,
        _template_input_ids,
        normalize_message,
    )
    from evaluation.controlled_multiworkflow_v1.scenario import (
        CHECKPOINT_SIZE_BYTES,
    )
    from evaluation.rq3_openhands_population import load_session_messages

    directories = prepare_artifact_directories(artifact_root)
    manifest = json.loads(
        (artifact_root / "population_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    group_row = next(
        row
        for row in manifest["groups"]
        if int(row["group_ordinal"]) == group_ordinal
    )
    group = WorkflowGroup(
        group_ordinal=int(group_row["group_ordinal"]),
        population_segment=str(group_row["population_segment"]),
        allocation_round=int(group_row["allocation_round"]),
        session_ids=tuple(str(value) for value in group_row["session_ids"]),
    )
    control_port = int(os.environ.get("FLOWSTATE_STEP5D_PORT", "49937"))
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    try:
        messages_by_label = {
            label: load_session_messages(session_id, DATASET_PATH)
            for label, session_id in group.session_by_label.items()
        }
    except Exception as error:
        verdict = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": "dataset_load_failed",
            "diagnostics": {"error": repr(error)},
        }
        write_failure_artifact(
            directories.failures,
            group,
            "dataset_load_failed",
            {"error": repr(error)},
        )
        _write_json(
            directories.verdicts / f"g{group.group_ordinal:03d}.json",
            verdict,
        )
        return 0
    try:
        verdict = collect_group(
            group,
            runtime_factory=sglang_runtime_factory(
                ENGINE_CONFIGURATION_COMMON_BARRIER,
                control_port,
            ),
            messages_by_label=messages_by_label,
            tokenizer=tokenizer,
            normalize_message=normalize_message,
            template_input_ids=_template_input_ids,
            checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
            directories=directories,
        )
    except Exception as error:
        traceback.print_exc()
        verdict = {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": "unexpected_error",
            "diagnostics": {"error": repr(error)},
        }
        write_failure_artifact(
            directories.failures,
            group,
            "unexpected_error",
            {"error": repr(error)},
        )
    verdict = dict(verdict)
    _write_json(
        directories.verdicts / f"g{group.group_ordinal:03d}.json",
        verdict,
    )
    print(
        "[RQ3E] group={ordinal:03d} round={round_id} status={status} "
        "reason={reason}".format(
            ordinal=group.group_ordinal,
            round_id=group.allocation_round,
            status=verdict["status"],
            reason=verdict.get("primary_reason"),
        ),
        flush=True,
    )
    return 0


def write_interrupted_marker(
    root: Path,
    *,
    attempted_groups: int,
    eligible_groups: int,
    failed_groups: int,
    failure_reason_distribution: Mapping[str, int],
    collector_versions: Sequence[str],
    reason: str,
) -> Path:
    """在被中断的旧 artifact root 中写入永久诊断标记。"""

    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "interrupted_collection",
        "status": "DIAGNOSTIC_ONLY",
        "reason": reason,
        "attempted_groups": attempted_groups,
        "eligible_groups": eligible_groups,
        "build_failed_groups": failed_groups,
        "failure_reason_distribution": dict(failure_reason_distribution),
        "collector_versions_before_interruption": list(collector_versions),
        "formal_evaluation_may_consume": False,
        "note": (
            "该 root 的采集曾因 Engine lifecycle crash 中断；"
            "其 snapshots 仅可用于 debugging，"
            "禁止进入正式 RQ3 Main Population 统计。"
        ),
    }
    path = root / "INTERRUPTED_COLLECTION.json"
    _write_json(path, payload)
    return path


def read_population_root_status(root: Path) -> str:
    """识别一个 artifact root 的当前身份。"""

    marker = root / "INTERRUPTED_COLLECTION.json"
    if marker.exists():
        return "diagnostic_only"
    if (root / "collection_summary.json").exists():
        return "formal"
    snapshots = root / "snapshots"
    if snapshots.exists() and any(snapshots.glob("g*.json")):
        return "unmarked_partial"
    return "empty"


def assert_formal_restart_root(root: Path) -> None:
    """正式重跑必须使用的全新 artifact root 校验。"""

    status = read_population_root_status(root)
    if status == "diagnostic_only":
        raise RuntimeError(
            "该 artifact root 已标记 DIAGNOSTIC_ONLY，禁止作为正式 population"
        )
    if status != "empty":
        raise RuntimeError(
            f"正式重跑必须从空 artifact root 开始，当前状态：{status}"
        )


def collector_source_digest(paths: Sequence[Path]) -> str:
    """对 collector 相关源码文件计算组合 SHA-256 版本摘要。"""

    rows = []
    for path in sorted(paths, key=lambda item: str(item)):
        content = path.read_bytes()
        rows.append(
            {
                "path": str(path),
                "sha256": sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    return digest_of_canonical(rows)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """用稳定格式写出一个 JSON 文件。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _eligibility_cache_payload(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_path: Path,
    tokenizer_identity: str,
) -> Mapping[str, object]:
    """构造 dataset eligibility 缓存的稳定内容。"""

    return {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "dataset_eligibility",
        "dataset_path": str(dataset_path),
        "tokenizer_identity": tokenizer_identity,
        "session_rows": [dict(row) for row in rows],
    }


def compute_dataset_eligibility(
    *,
    tokenizer: object,
    dataset_path: Path,
    normalize_message: Callable[[Mapping[str, object]], dict[str, object]],
    template_input_ids: Callable[[object], list[int]],
    cache_path: Path,
    tokenizer_identity: str,
) -> list[Mapping[str, object]]:
    """计算或复用全部源 session 的 dataset-level eligibility。"""

    from concurrent.futures import ThreadPoolExecutor

    from evaluation.rq3_openhands_population import (
        MIN_N_TURNS,
        REQUIRED_ASSISTANT_TURNS,
        count_assistant_turns,
        evaluate_session_eligibility,
        load_session_messages,
        load_source_session_rows,
        replay_input_token_lengths,
    )

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("dataset_path") == str(dataset_path)
            and cached.get("tokenizer_identity") == tokenizer_identity
            and isinstance(cached.get("session_rows"), list)
        ):
            return [
                dict(row) for row in cached["session_rows"]
            ]
        raise RuntimeError("dataset eligibility 缓存的身份不一致")

    session_rows = load_source_session_rows(dataset_path)

    def evaluate_one(row: Mapping[str, object]) -> Mapping[str, object]:
        session_id = str(row["session_id"])
        n_turns = int(row["n_turns"])
        if n_turns < MIN_N_TURNS:
            return {
                "session_id": session_id,
                "n_turns": n_turns,
                "assistant_turns": None,
                "replay_input_tokens": [],
                "eligible": False,
                "reason": "n_turns_below_60",
            }
        raw_messages = load_session_messages(session_id, dataset_path)
        assistant_turns = count_assistant_turns(raw_messages)
        if assistant_turns < REQUIRED_ASSISTANT_TURNS:
            lengths: tuple[int, ...] = ()
        else:
            lengths, _ = replay_input_token_lengths(
                tokenizer,
                raw_messages,
                normalize_message=normalize_message,
                template_input_ids=template_input_ids,
            )
        verdict = evaluate_session_eligibility(
            session_id,
            n_turns,
            assistant_turns,
            lengths,
        )
        return {
            "session_id": verdict.session_id,
            "n_turns": verdict.n_turns,
            "assistant_turns": verdict.assistant_turns,
            "replay_input_tokens": list(verdict.replay_input_tokens),
            "eligible": verdict.eligible,
            "reason": verdict.reason,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate_one, session_rows))
    payload = _eligibility_cache_payload(
        results,
        dataset_path=dataset_path,
        tokenizer_identity=tokenizer_identity,
    )
    _write_json(cache_path, payload)
    return results


def _completed_group_ordinals(directories: ArtifactDirectories) -> set[int]:
    """读取已经完成（ELIGIBLE 或 BUILD_FAILED）的 group ordinal 集合。"""

    completed: set[int] = set()
    for directory in (directories.snapshots, directories.failures):
        for path in directory.glob("g*.json"):
            ordinal_text = path.name.split("_")[0].lstrip("g")
            if ordinal_text.isdigit():
                completed.add(int(ordinal_text))
    return completed


def main() -> int:
    """按冻结 protocol 采集全部 200 个 Main groups。"""

    import argparse
    import importlib.metadata
    import os

    parser = argparse.ArgumentParser(
        description="RQ3 OpenHands neutral snapshot collection"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="正式 artifact 根目录（已存在则按已完成状态续跑）",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只运行冻结顺序最前面 4 个 Main groups 的 preflight",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="以单 group worker 子进程模式运行（仅供 parent 调用）",
    )
    parser.add_argument(
        "--group-ordinal",
        type=int,
        default=None,
        help="worker 模式采集的 group ordinal",
    )
    parser.add_argument(
        "--require-fresh-root",
        action="store_true",
        help="要求 artifact root 必须是未被标记、无既有内容的全新目录",
    )
    args = parser.parse_args()
    if args.worker:
        if args.group_ordinal is None:
            raise SystemExit("worker 模式必须指定 --group-ordinal")
        return run_group_worker(args.group_ordinal, args.artifact_root)

    from transformers import AutoTokenizer

    from evaluation.openhands_common_barrier_snapshot_gate import (
        ENGINE_CONFIGURATION_COMMON_BARRIER,
    )
    from evaluation.openhands_single_workflow_smoke import (
        DATASET_PATH,
        TOKENIZER_PATH,
        _template_input_ids,
        normalize_message,
    )
    from evaluation.controlled_multiworkflow_v1.scenario import (
        CHECKPOINT_SIZE_BYTES,
    )
    from evaluation.rq3_openhands_population import (
        DEFAULT_DATASET_PATH,
        MAIN_GROUP_COUNT,
        POPULATION_SEED,
        PROTOCOL_VERSION,
        SOURCE_DATASET,
        build_workflow_groups,
        designated_main_groups,
        order_sessions_by_digest,
        reserved_sensitivity_groups,
    )

    if DATASET_PATH != DEFAULT_DATASET_PATH:
        raise RuntimeError("collector 数据路径与冻结路径不一致")
    control_port = int(os.environ.get("FLOWSTATE_STEP5D_PORT", "49937"))
    if args.require_fresh_root:
        assert_formal_restart_root(args.artifact_root)
    directories = prepare_artifact_directories(args.artifact_root)
    collector_files = [
        Path(__file__).resolve(),
        (Path(__file__).resolve().parent / "rq3_openhands_population.py"),
        (
            Path(__file__).resolve().parent
            / "rq3_frozen_snapshot_evaluator.py"
        ),
    ]
    source_digest = collector_source_digest(collector_files)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "sglang_version": importlib.metadata.version("sglang"),
        "transformers_version": importlib.metadata.version("transformers"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "collector_version": COLLECTOR_VERSION,
    }
    _write_json(
        directories.root / "COLLECTOR_VERSION.json",
        {
            "schema_version": ARTIFACT_SCHEMA,
            "artifact_kind": "collector_version",
            "collector_version": COLLECTOR_VERSION,
            "source_digest": source_digest,
            "source_files": [
                {
                    "path": str(path),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(collector_files, key=str)
            ],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    tokenizer_identity = (
        f"{TOKENIZER_PATH}@transformers-{environment['transformers_version']}"
    )

    eligibility_rows = compute_dataset_eligibility(
        tokenizer=tokenizer,
        dataset_path=DATASET_PATH,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        cache_path=directories.eligibility / "dataset_eligibility.json",
        tokenizer_identity=tokenizer_identity,
    )
    eligible_session_ids = [
        str(row["session_id"]) for row in eligibility_rows if row["eligible"]
    ]
    ordered = order_sessions_by_digest(eligible_session_ids)
    groups = build_workflow_groups(ordered)
    main_groups = designated_main_groups(groups)
    sensitivity_groups = reserved_sensitivity_groups(groups)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "population_manifest",
        "protocol_version": PROTOCOL_VERSION,
        "dataset": {
            "path": str(DATASET_PATH),
            "source_dataset": SOURCE_DATASET,
            "unique_sessions": len(eligibility_rows),
            "eligible_sessions": len(eligible_session_ids),
        },
        "seed": POPULATION_SEED,
        "ordering": "SHA256(rq3-openhands-v1|20260903|session_id) 升序",
        "ordered_session_ids": list(ordered),
        "main_group_count": len(main_groups),
        "sensitivity_group_count": len(sensitivity_groups),
        "groups": [
            {
                "group_ordinal": group.group_ordinal,
                "population_segment": group.population_segment,
                "allocation_round": group.allocation_round,
                "session_ids": list(group.session_ids),
                "session_by_label": group.session_by_label,
            }
            for group in groups
        ],
        "engine_configuration": dict(ENGINE_CONFIGURATION_COMMON_BARRIER),
        "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
        "physical_mamba_pool": int(
            ENGINE_CONFIGURATION_COMMON_BARRIER["max_mamba_cache_size"]
        ),
        "collector_version": COLLECTOR_VERSION,
        "environment": environment,
    }
    _write_json(directories.root / "population_manifest.json", manifest)

    completed = _completed_group_ordinals(directories)

    def collect_one(group: WorkflowGroup) -> Mapping[str, object]:
        if group.group_ordinal in completed:
            return _reload_completed_verdict(directories, group)
        gpu_wait = wait_gpu_stable(gpu_index=0)
        verdict = run_group_via_worker(
            group,
            directories=directories,
            gpu_wait_record=gpu_wait,
        )
        print(
            "[RQ3E] group={ordinal:03d} round={round_id} status={status} "
            "reason={reason}".format(
                ordinal=group.group_ordinal,
                round_id=group.allocation_round,
                status=verdict["status"],
                reason=verdict.get("primary_reason"),
            ),
            flush=True,
        )
        return verdict

    if args.preflight_only:
        designated = main_groups[:4]
    else:
        designated = main_groups
    _write_json(
        directories.root / "protocol.json",
        protocol_payload(
            environment=environment,
            engine_configuration=ENGINE_CONFIGURATION_COMMON_BARRIER,
            checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        ),
    )
    verdicts = run_designated_groups(designated, collect_one)
    _attach_frequency_statistics(directories, verdicts)
    duplicate_updates = _enforce_digest_uniqueness(directories, verdicts)
    if duplicate_updates:
        verdicts = duplicate_updates
    summary = summarize_population(verdicts)
    summary_payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_kind": "collection_summary",
        "preflight_only": bool(args.preflight_only),
        "designated_groups_total": len(designated),
        **dict(summary),
        **dict(population_statistics(verdicts)),
        "verdicts": verdicts,
    }
    _write_json(
        directories.root / "collection_summary.json",
        summary_payload,
    )
    print(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )
    return 0


def _attach_frequency_statistics(
    directories: ArtifactDirectories,
    verdicts: Sequence[Mapping[str, object]],
) -> None:
    """从 provenance artifact 为 ELIGIBLE verdict 回填 frequency 分布数据。"""

    for verdict in verdicts:
        if verdict["status"] != "ELIGIBLE":
            continue
        ordinal = int(verdict["group_ordinal"])
        matches = list(directories.lfu_provenance.glob(f"g{ordinal:03d}_*.json"))
        if len(matches) != 1:
            continue
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        verdict["frequency_by_checkpoint"] = {
            str(row["checkpoint_id"]): int(row["access_frequency"])
            for row in payload.get("rows", ())
        }


def _enforce_digest_uniqueness(
    directories: ArtifactDirectories,
    verdicts: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]] | None:
    """若出现重复 digest，把 ordinal 靠后的 group 记为 BUILD_FAILED。"""

    seen: dict[str, int] = {}
    updated: list[Mapping[str, object]] | None = None
    for index, verdict in enumerate(verdicts):
        if verdict["status"] != "ELIGIBLE":
            continue
        digest = str(verdict.get("snapshot_digest") or "")
        if not digest:
            continue
        if digest not in seen:
            seen[digest] = int(verdict["group_ordinal"])
            continue
        if updated is None:
            updated = [dict(row) for row in verdicts]
        first_ordinal = seen[digest]
        duplicate = dict(updated[index])
        duplicate["status"] = "BUILD_FAILED"
        duplicate["primary_reason"] = "duplicate_snapshot_digest"
        duplicate["diagnostics"] = {
            **dict(duplicate.get("diagnostics", {})),
            "duplicate_of_group_ordinal": first_ordinal,
        }
        updated[index] = duplicate
    return updated


def _reload_completed_verdict(
    directories: ArtifactDirectories,
    group: WorkflowGroup,
) -> Mapping[str, object]:
    """在续跑时从既有 artifact 重建一个 group 的 verdict。"""

    for path in directories.failures.glob(f"g{group.group_ordinal:03d}.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "BUILD_FAILED",
            "primary_reason": payload["primary_reason"],
            "diagnostics": payload.get("diagnostics", {}),
        }
    for path in directories.snapshots.glob(
        f"g{group.group_ordinal:03d}_*.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "ELIGIBLE",
            "primary_reason": None,
            "diagnostics": {},
            "snapshot_id": payload["snapshot_id"],
            "snapshot_digest": payload["snapshot_digest"],
        }
    raise RuntimeError(
        f"group {group.group_ordinal} 标记为已完成但找不到 artifact"
    )


if __name__ == "__main__":
    raise SystemExit(main())
