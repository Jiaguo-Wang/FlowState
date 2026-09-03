#!/usr/bin/env python3
"""验证 Round 2 后的动态候选注册表与第二次 K=2 在线选择。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    SchedulerRuntimeAdapter,
    inspect_checkpoint,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.scenario import CheckpointRecency
from evaluation.openhands_4workflow_occupancy_calibration import (
    WORKFLOWS,
    _environment,
    _failure_record,
    compact_census,
    execute_request,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    CHECKPOINT_SIZE_BYTES,
    LOGICAL_K,
    lineage_path,
    locate_materialized_candidate,
    token_digest,
    validate_candidate_at_barrier,
)
from evaluation.openhands_frozen_barrier_k2_selection_gate import (
    flowstate_continuation_rows,
    marconi_score_rows,
)
from evaluation.openhands_policy_runtime_heg_outcome_gate import (
    ENGINE_CONFIGURATION_HEG_OUTCOME,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
    inspect_candidate_states,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    _template_input_ids,
    normalize_message,
)
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_marconi_flop_saved,
)
from evaluation.sota_policies import MarconiStylePolicy
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
POLICY_ORDER = ("LRU", "Marconi", "FlowState")
ROUND_ONE_TURN = 1
ROUND_TWO_TURN = 2
ROUND_THREE_PENDING_TURN = 3
ROUND_ONE_SCHEDULE = tuple((label, ROUND_ONE_TURN) for label in WORKFLOWS)
ROUND_TWO_SCHEDULE = tuple((label, ROUND_TWO_TURN) for label in WORKFLOWS)
ENGINE_CONFIGURATION_DYNAMIC_REGISTRY = dict(
    ENGINE_CONFIGURATION_HEG_OUTCOME
)
EXPECTED_BARRIER_ONE_SELECTION = {
    "LRU": (
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
    "Marconi": (
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
    "FlowState": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
    ),
}


@dataclass
class DynamicRegistryEntry:
    """保存一个历史检查点及其当前物理驻留与在线时序。"""

    checkpoint_id: str
    workflow_label: str
    workflow_id: str
    turn: int
    lineage_path: tuple[str, ...]
    token_pos: int
    memory_bytes: int
    recurrent_resident: bool
    fa_resident: bool
    handle: RuntimeCheckpointHandle
    creation_order: int
    last_access_order: int
    node_id: int
    slots: tuple[int, ...]
    materialization_events: list[dict[str, object]] = field(
        default_factory=list
    )

    def candidate(self) -> CheckpointCandidate:
        """返回当前决策快照使用的不可变候选。"""
        return CheckpointCandidate(
            checkpoint_id=self.checkpoint_id,
            workflow_id=self.workflow_id,
            lineage_path=self.lineage_path,
            token_pos=self.token_pos,
            memory_bytes=self.memory_bytes,
            recurrent_resident=self.recurrent_resident,
            fa_resident=self.fa_resident,
        )

    def row(self) -> dict[str, object]:
        """返回包含运行时句柄和在线时序的审计行。"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_label": self.workflow_label,
            "workflow_id": self.workflow_id,
            "turn": self.turn,
            "lineage_path": list(self.lineage_path),
            "token_pos": self.token_pos,
            "memory_bytes": self.memory_bytes,
            "recurrent_resident": self.recurrent_resident,
            "fa_resident": self.fa_resident,
            "creation_order": self.creation_order,
            "last_access_order": self.last_access_order,
            "node_id": self.node_id,
            "slots": list(self.slots),
            "runtime_handle": {
                "checkpoint_id": self.handle.checkpoint_id,
                "token_ids": list(self.handle.token_ids),
                "extra_key": self.handle.extra_key,
                "expected_node_id": self.handle.expected_node_id,
                "expected_prefix_digest": (
                    self.handle.expected_prefix_digest
                ),
            },
            "materialization_events": list(self.materialization_events),
        }


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的时间戳产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_round2_dynamic_registry_selection_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def materialize_round2_visible_requests(
    tokenizer: object,
    raw_messages: Sequence[Mapping[str, object]],
    *,
    workflow_label: str,
    workflow_id: str,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    """只物化已执行 Round 1、2 与当前 Round 3 pending。"""
    history: list[dict[str, object]] = []
    requests: dict[int, dict[str, object]] = {}
    assistant_turn = 0
    raw_items_iterated = 0
    for raw_message in raw_messages:
        raw_items_iterated += 1
        role = raw_message.get("role")
        if role == "assistant":
            assistant_turn += 1
            if assistant_turn <= ROUND_THREE_PENDING_TURN:
                output = tokenizer.apply_chat_template(
                    list(history),
                    tokenize=True,
                    add_generation_prompt=True,
                )
                requests[assistant_turn] = {
                    "workflow_label": workflow_label,
                    "workflow_id": workflow_id,
                    "turn": assistant_turn,
                    "rid": (
                        f"openhands-dynamic-{workflow_label.lower()}-"
                        f"turn-{assistant_turn:03d}"
                    ),
                    "input_ids": _template_input_ids(output),
                }
            if assistant_turn == ROUND_THREE_PENDING_TURN:
                break
        history.append(normalize_message(raw_message))
    if tuple(sorted(requests)) != (1, 2, 3):
        raise RuntimeError(
            f"workflow {workflow_label} 缺少 Round 1、2 或 pending Round 3"
        )
    return requests, {
        "workflow": workflow_label,
        "maximum_assistant_turn_consumed": ROUND_THREE_PENDING_TURN,
        "raw_items_iterated_through_pending_marker": raw_items_iterated,
        "round_3_request_materialized": True,
        "round_3_assistant_output_read": False,
        "round_4_message_consumed": False,
        "round_4_request_materialized": False,
        "future_timing_read": False,
        "future_checkpoint_read": False,
    }


def load_round2_visible_requests(
    tokenizer: object,
) -> tuple[dict[tuple[str, int], dict[str, object]], list[dict[str, object]]]:
    """从四条 trajectory 仅读取到 A3、B3、C3、D3 输入边界。"""
    requests: dict[tuple[str, int], dict[str, object]] = {}
    audits = []
    for label, workflow_id in WORKFLOWS.items():
        table = pq.read_table(
            DATASET_PATH,
            filters=[("session_id", "=", workflow_id)],
            columns=["session_id", "messages_json"],
        )
        if table.num_rows != 1:
            raise RuntimeError(
                f"workflow {label} 应唯一命中一行，实际为 {table.num_rows}"
            )
        raw_messages = json.loads(
            table.column("messages_json")[0].as_py()
        )
        if not isinstance(raw_messages, list):
            raise TypeError("messages_json 反序列化后必须是列表")
        visible, audit = materialize_round2_visible_requests(
            tokenizer,
            raw_messages,
            workflow_label=label,
            workflow_id=workflow_id,
        )
        for turn, request in visible.items():
            requests[(label, turn)] = request
        audits.append(audit)
    return requests, audits


def registry_entry_from_round_one(
    candidate: CheckpointCandidate,
    handle: RuntimeCheckpointHandle,
    row: Mapping[str, object],
) -> DynamicRegistryEntry:
    """把 Round 1 的真实 materialization 纳入动态注册表。"""
    order = int(row["creation_order"])
    return DynamicRegistryEntry(
        checkpoint_id=candidate.checkpoint_id,
        workflow_label=str(row["workflow_label"]),
        workflow_id=candidate.workflow_id,
        turn=ROUND_ONE_TURN,
        lineage_path=candidate.lineage_path,
        token_pos=candidate.token_pos,
        memory_bytes=candidate.memory_bytes,
        recurrent_resident=candidate.recurrent_resident,
        fa_resident=candidate.fa_resident,
        handle=handle,
        creation_order=order,
        last_access_order=int(row["last_access_order"]),
        node_id=int(row["node_id"]),
        slots=tuple(int(value) for value in row["slots"]),
        materialization_events=[
            {
                "turn": ROUND_ONE_TURN,
                "event_order": order,
                "kind": "首次创建",
            }
        ],
    )


def apply_materialization_observation(
    registry: dict[str, DynamicRegistryEntry],
    *,
    request: Mapping[str, object],
    event_order: int,
    executable_frontier: int,
    node_id: int,
    token_pos: int,
    slots: Sequence[int],
    handle: RuntimeCheckpointHandle,
    fa_resident: bool,
    recurrent_resident: bool,
    previously_resident_ids: set[str],
) -> tuple[DynamicRegistryEntry, str]:
    """按真实新增节点区分旧检查点重建与新检查点创建。"""
    workflow_id = str(request["workflow_id"])
    digest = handle.expected_prefix_digest
    existing = next(
        (
            entry
            for entry in registry.values()
            if entry.workflow_id == workflow_id
            and entry.token_pos == token_pos
            and entry.handle.expected_prefix_digest == digest
        ),
        None,
    )
    for entry in registry.values():
        if (
            entry.checkpoint_id in previously_resident_ids
            and entry.workflow_id == workflow_id
            and entry.token_pos == executable_frontier
        ):
            entry.last_access_order = event_order
    if existing is not None:
        existing.recurrent_resident = recurrent_resident
        existing.fa_resident = fa_resident
        existing.handle = RuntimeCheckpointHandle(
            checkpoint_id=existing.checkpoint_id,
            token_ids=handle.token_ids,
            extra_key=handle.extra_key,
            expected_node_id=node_id,
            expected_prefix_digest=digest,
        )
        existing.node_id = node_id
        existing.slots = tuple(int(value) for value in slots)
        existing.creation_order = event_order
        existing.last_access_order = event_order
        existing.materialization_events.append(
            {
                "turn": int(request["turn"]),
                "event_order": event_order,
                "kind": "删除后重建",
            }
        )
        return existing, "REMATERIALIZED"
    label = str(request["workflow_label"])
    checkpoint_id = (
        f"OPENHANDS_BARRIER_{label}_TURN_{int(request['turn']):03d}"
    )
    if checkpoint_id in registry:
        raise RuntimeError(f"新检查点标识冲突：{checkpoint_id}")
    entry = DynamicRegistryEntry(
        checkpoint_id=checkpoint_id,
        workflow_label=label,
        workflow_id=workflow_id,
        turn=int(request["turn"]),
        lineage_path=lineage_path(workflow_id),
        token_pos=token_pos,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=recurrent_resident,
        fa_resident=fa_resident,
        handle=RuntimeCheckpointHandle(
            checkpoint_id=checkpoint_id,
            token_ids=handle.token_ids,
            extra_key=handle.extra_key,
            expected_node_id=node_id,
            expected_prefix_digest=digest,
        ),
        creation_order=event_order,
        last_access_order=event_order,
        node_id=node_id,
        slots=tuple(int(value) for value in slots),
        materialization_events=[
            {
                "turn": int(request["turn"]),
                "event_order": event_order,
                "kind": "Round 2 新建",
            }
        ],
    )
    registry[checkpoint_id] = entry
    return entry, "CREATED"


def refresh_registry(
    client: object,
    registry: Mapping[str, DynamicRegistryEntry],
    *,
    phase: str,
) -> list[dict[str, object]]:
    """用 checkpoint inspect 刷新每个历史检查点的当前驻留事实。"""
    rows = []
    for checkpoint_id in sorted(registry):
        entry = registry[checkpoint_id]
        response = inspect_checkpoint(
            client,
            f"{checkpoint_id}_{phase}",
            entry.handle.token_ids,
        )
        path = response["after"]["path"]
        if (
            int(path["prefix_tokens"]) != entry.token_pos
            or str(path["prefix_sha256"])
            != entry.handle.expected_prefix_digest
        ):
            raise RuntimeError(f"{checkpoint_id} 的前缀 identity 发生变化")
        entry.node_id = int(path["node_id"])
        entry.handle = RuntimeCheckpointHandle(
            checkpoint_id=entry.checkpoint_id,
            token_ids=entry.handle.token_ids,
            extra_key=entry.handle.extra_key,
            expected_node_id=entry.node_id,
            expected_prefix_digest=entry.handle.expected_prefix_digest,
        )
        entry.fa_resident = bool(
            path["target_full_present"] and path["path_full_all_present"]
        )
        entry.recurrent_resident = bool(path["target_mamba_present"])
        raw_slots = path.get("target_mamba_slots")
        entry.slots = tuple(
            int(value) for value in (raw_slots or ())
        )
        rows.append(
            {
                "phase": phase,
                "checkpoint_id": checkpoint_id,
                "node_id": entry.node_id,
                "token_pos": entry.token_pos,
                "fa_resident": entry.fa_resident,
                "recurrent_resident": entry.recurrent_resident,
                "slots": list(entry.slots),
            }
        )
    return rows


def register_round_two_materialization(
    client: object,
    registry: dict[str, DynamicRegistryEntry],
    request: Mapping[str, object],
    record: Mapping[str, object],
    census: Mapping[str, object],
    *,
    event_order: int,
    previously_resident_ids: set[str],
) -> dict[str, object]:
    """把 Round 2 请求后唯一新增的物理 Mamba 节点写入 registry。"""
    added_ids = [int(value) for value in census["added_mamba_node_ids"]]
    if len(added_ids) != 1:
        raise RuntimeError(
            f"{request['workflow_label']}2 未产生唯一 materialization：{added_ids}"
        )
    node_id = added_ids[0]
    resident = {
        int(row["node_id"]): row
        for row in census["resident_mamba_nodes"]
    }
    if node_id not in resident:
        raise RuntimeError(f"新增 Mamba 节点 {node_id} 不在驻留集合")
    token_pos = int(resident[node_id]["token_position"])
    raw_input_ids = request["input_ids"]
    if not isinstance(raw_input_ids, list):
        raise TypeError("Round 2 input_ids 必须是列表")
    prefix_ids = tuple(int(value) for value in raw_input_ids[:token_pos])
    digest = token_digest(prefix_ids)
    probe_id = (
        f"OPENHANDS_DYNAMIC_{request['workflow_label']}_"
        f"TURN_{int(request['turn']):03d}"
    )
    response = inspect_checkpoint(client, probe_id, prefix_ids)
    path = response["after"]["path"]
    if (
        int(path["node_id"]) != node_id
        or int(path["prefix_tokens"]) != token_pos
        or str(path["prefix_sha256"]) != digest
    ):
        raise RuntimeError("Round 2 census 与 checkpoint inspect 不一致")
    probe_handle = RuntimeCheckpointHandle(
        checkpoint_id=probe_id,
        token_ids=prefix_ids,
        extra_key=None,
        expected_node_id=node_id,
        expected_prefix_digest=digest,
    )
    entry, kind = apply_materialization_observation(
        registry,
        request=request,
        event_order=event_order,
        executable_frontier=int(record["e"]),
        node_id=node_id,
        token_pos=token_pos,
        slots=resident[node_id]["slots"],
        handle=probe_handle,
        fa_resident=bool(
            path["target_full_present"] and path["path_full_all_present"]
        ),
        recurrent_resident=bool(path["target_mamba_present"]),
        previously_resident_ids=previously_resident_ids,
    )
    return {
        "workflow_label": request["workflow_label"],
        "turn": int(request["turn"]),
        "event_order": event_order,
        "kind": kind,
        "checkpoint_id": entry.checkpoint_id,
        "node_id": node_id,
        "token_pos": token_pos,
        "executable_frontier": int(record["e"]),
    }


def registry_candidates(
    registry: Mapping[str, DynamicRegistryEntry],
) -> tuple[CheckpointCandidate, ...]:
    """按稳定标识返回 registry 的完整历史候选快照。"""
    candidates = tuple(
        registry[checkpoint_id].candidate()
        for checkpoint_id in sorted(registry)
    )
    validate_unique_checkpoint_ids(candidates)
    return candidates


def build_dynamic_metadata(
    registry: Mapping[str, DynamicRegistryEntry],
    candidates: Sequence[CheckpointCandidate],
) -> tuple[list[dict[str, object]], tuple[CheckpointRecency, ...]]:
    """用截至 Round 2 的真实创建和访问历史构造 policy metadata。"""
    recency = tuple(
        CheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=registry[candidate.checkpoint_id].creation_order,
            last_access_order=registry[
                candidate.checkpoint_id
            ].last_access_order,
        )
        for candidate in candidates
    )
    incremental_spans = build_marconi_flop_saved(candidates)
    rows = []
    for candidate in candidates:
        entry = registry[candidate.checkpoint_id]
        parent_positions = [
            other.token_pos
            for other in candidates
            if other.checkpoint_id != candidate.checkpoint_id
            and other.workflow_id == candidate.workflow_id
            and other.token_pos < candidate.token_pos
        ]
        parent_pos = max(parent_positions, default=0)
        span = float(incremental_spans[candidate.checkpoint_id])
        rows.append(
            {
                "checkpoint_id": candidate.checkpoint_id,
                "creation_order": entry.creation_order,
                "last_access_order": entry.last_access_order,
                "marconi_recency": float(entry.last_access_order),
                "marconi_incremental_span": span,
                "marconi_raw_flop_efficiency": (
                    span / candidate.memory_bytes
                ),
                "marconi_first_checkpoint": parent_pos == 0,
                "marconi_parent_position": parent_pos,
                "marconi_alpha": CONTROLLED_MARCONI_ALPHA,
                "eligible": candidate.recurrent_resident,
            }
        )
    return rows, recency


def build_round_three_pending(
    barrier_client: BarrierFAControlClient,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    policy: str,
) -> tuple[tuple[PendingContinuation, ...], list[dict[str, object]]]:
    """只用当前 A3、B3、C3、D3 输入构造第二个 barrier pending。"""
    continuations = []
    rows = []
    for label, workflow_id in WORKFLOWS.items():
        request = requests[(label, ROUND_THREE_PENDING_TURN)]
        input_ids = request["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("Round 3 pending input_ids 必须是列表")
        lookup = barrier_client.inspect_fa_frontier(
            input_ids,
            extra_key=None,
            limit=None,
            nonce=f"openhands-dynamic:{policy}:{label}:turn-003",
        )
        if not lookup.get("state_equal"):
            raise RuntimeError(
                f"{policy} 的 {label}3 FA 查询产生副作用"
            )
        if lookup["scope_before"] != lookup["scope_after"]:
            raise RuntimeError(f"{policy} 的 {label}3 runtime scope 变化")
        continuation = PendingContinuation(
            continuation_id=f"OPENHANDS_BARRIER_{label}_TURN_003",
            workflow_id=workflow_id,
            lineage_path=lineage_path(workflow_id),
            anchor_pos=len(input_ids),
            resident_fa_frontier=int(lookup["resident_fa_frontier"]),
        )
        continuations.append(continuation)
        rows.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_label": label,
                "workflow_id": workflow_id,
                "lineage_path": list(continuation.lineage_path),
                "anchor_pos": continuation.anchor_pos,
                "resident_fa_frontier": (
                    continuation.resident_fa_frontier
                ),
                "planning_target": continuation.planning_target,
                "input_token_digest": token_digest(input_ids),
                "query_state_equal": True,
                "query_changed_fields": list(lookup["changed_fields"]),
                "traversed_node_ids": list(lookup["traversed_node_ids"]),
            }
        )
    return tuple(continuations), rows


def run_policy_selector(
    policy: str,
    candidates: Sequence[CheckpointCandidate],
    continuations: Sequence[PendingContinuation],
    metadata_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """调用指定的现有 selector，并记录其完整在线输入与结果。"""
    candidate_tuple = tuple(candidates)
    continuation_tuple = tuple(continuations)
    validate_unique_checkpoint_ids(candidate_tuple)
    metadata_by_id = {
        str(row["checkpoint_id"]): row for row in metadata_rows
    }
    eligible_ids = {
        candidate.checkpoint_id
        for candidate in candidate_tuple
        if candidate.recurrent_resident
    }
    if policy == "LRU":
        recency = tuple(
            CheckpointRecency(
                checkpoint_id=candidate.checkpoint_id,
                creation_order=int(
                    metadata_by_id[candidate.checkpoint_id][
                        "creation_order"
                    ]
                ),
                last_access_order=int(
                    metadata_by_id[candidate.checkpoint_id][
                        "last_access_order"
                    ]
                ),
            )
            for candidate in candidate_tuple
        )
        selected = select_global_lru(
            candidate_tuple,
            recency,
            BUDGET_BYTES,
        )
        details = {
            "ranking_metadata": [
                {
                    "checkpoint_id": item.checkpoint_id,
                    "creation_order": item.creation_order,
                    "last_access_order": item.last_access_order,
                }
                for item in recency
            ]
        }
    elif policy == "Marconi":
        last_access = {
            candidate.checkpoint_id: float(
                metadata_by_id[candidate.checkpoint_id][
                    "marconi_recency"
                ]
            )
            for candidate in candidate_tuple
        }
        spans = {
            candidate.checkpoint_id: float(
                metadata_by_id[candidate.checkpoint_id][
                    "marconi_incremental_span"
                ]
            )
            for candidate in candidate_tuple
        }
        result = MarconiStylePolicy().select(
            candidate_tuple,
            LOGICAL_K,
            last_access,
            spans,
            CONTROLLED_MARCONI_ALPHA,
        )
        selected = result.selected_checkpoint_ids
        eligible = tuple(
            candidate
            for candidate in candidate_tuple
            if candidate.recurrent_resident
        )
        details = {
            "alpha": CONTROLLED_MARCONI_ALPHA,
            "scores": marconi_score_rows(eligible, metadata_rows),
        }
    elif policy == "FlowState":
        model = RecoveryCostModel()
        result = GlobalOptimizer(model).select(
            continuation_tuple,
            candidate_tuple,
            BUDGET_BYTES,
        )
        selected = tuple(
            candidate.checkpoint_id for candidate in result.selected
        )
        details = {
            "recovery_cost_before_ms": result.recovery_cost_before_ms,
            "recovery_cost_after_ms": result.recovery_cost_after_ms,
            "total_benefit_ms": result.total_benefit_ms,
            "used_bytes": result.used_bytes,
            "continuations": flowstate_continuation_rows(
                continuation_tuple,
                result,
                model,
            ),
        }
    else:
        raise ValueError(f"未知 policy：{policy}")
    valid = bool(
        len(selected) <= LOGICAL_K
        and len(set(selected)) == len(selected)
        and set(selected).issubset(eligible_ids)
    )
    return {
        "policy": policy,
        "eligible_candidate_ids": sorted(eligible_ids),
        "selected_checkpoint_ids": list(selected),
        "selected_count": len(selected),
        "selection_valid": valid,
        **details,
    }


@dataclass(frozen=True)
class PolicyPaths:
    """保存一个独立 policy 生命周期的完整产物路径。"""

    requests: Path
    censuses: Path
    registry: Path
    pending: Path
    selections: Path
    controller: Path


def _policy_paths(directory: Path) -> PolicyPaths:
    """创建一个 policy 独占的产物目录与空 JSONL 文件。"""
    directory.mkdir(parents=True, exist_ok=False)
    paths = PolicyPaths(
        requests=directory / "requests.jsonl",
        censuses=directory / "census.jsonl",
        registry=directory / "registry.jsonl",
        pending=directory / "pending.jsonl",
        selections=directory / "selections.json",
        controller=directory / "controller.json",
    )
    for path in (
        paths.requests,
        paths.censuses,
        paths.registry,
        paths.pending,
    ):
        path.write_text("", encoding="utf-8")
    return paths


def _boundary_has_future_leakage(
    audits: Sequence[Mapping[str, object]],
) -> bool:
    """判断是否读取 Round 4 或 Round 3 assistant output。"""
    return any(
        audit.get("round_3_assistant_output_read") is not False
        or audit.get("round_4_message_consumed") is not False
        or audit.get("round_4_request_materialized") is not False
        or audit.get("future_timing_read") is not False
        or audit.get("future_checkpoint_read") is not False
        for audit in audits
    )


def run_policy_lifecycle(
    *,
    policy: str,
    engine_ordinal: int,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    paths: PolicyPaths,
) -> dict[str, object]:
    """用 fresh Engine 执行至 Barrier 2 的 selector-only 终点。"""
    engine = None
    records = []
    registry: dict[str, DynamicRegistryEntry] = {}
    registry_events = []
    first_selection = None
    second_selection = None
    controller_report = None
    mapping_invariants = None
    pending_rows = []
    metadata_rows = []
    native_eviction = False
    fa_cascade = False
    fatal_error = None
    shutdown_error = None
    try:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_DYNAMIC_REGISTRY
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"openhands-dynamic:{policy}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["event"] = "baseline"
        _append_jsonl(paths.censuses, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("fresh Engine 初始含循环检查点")
        previous = baseline

        round_one_candidates = []
        round_one_handles = {}
        round_one_rows = []
        for ordinal, (label, turn) in enumerate(
            ROUND_ONE_SCHEDULE,
            start=1,
        ):
            request = requests[(label, turn)]
            record = execute_request(engine, client, request, ordinal)
            census = compact_census(
                client.census(
                    f"openhands-dynamic:{policy}:after:{label}{turn}"
                ),
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            census["event"] = f"after_{label}{turn}"
            candidate, handle, row = locate_materialized_candidate(
                client,
                request,
                census,
                event_order=ordinal,
            )
            entry = registry_entry_from_round_one(candidate, handle, row)
            registry[entry.checkpoint_id] = entry
            round_one_candidates.append(candidate)
            round_one_handles[candidate.checkpoint_id] = handle
            round_one_rows.append(row)
            records.append({**record, "policy": policy})
            _append_jsonl(paths.requests, records[-1])
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{policy} 的 {label}1 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError("Round 1 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("Round 1 发生 FA 级联")

        for candidate in round_one_candidates:
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                round_one_handles[candidate.checkpoint_id],
            )
            if not validation["consistent"]:
                raise RuntimeError("Barrier 1 candidate residency 不一致")
        barrier_client = BarrierFAControlClient(client)
        pending_two, _ = build_round_three_pending_compatible(
            barrier_client,
            requests,
            turn=ROUND_TWO_TURN,
            policy=policy,
        )
        candidates_one = registry_candidates(registry)
        metadata_one, _ = build_dynamic_metadata(
            registry,
            candidates_one,
        )
        first_selection = run_policy_selector(
            policy,
            candidates_one,
            pending_two,
            metadata_one,
        )
        if set(first_selection["selected_checkpoint_ids"]) != set(
            EXPECTED_BARRIER_ONE_SELECTION[policy]
        ):
            raise RuntimeError(f"{policy} 的 Barrier 1 selected set 异常")

        before_census = compact_census(
            client.census(f"openhands-dynamic:{policy}:before-reconcile"),
            ordinal=4,
            request=None,
            previous=previous,
        )
        before_states, _ = inspect_candidate_states(
            client,
            candidates_one,
            round_one_handles,
            phase=f"{policy}_DYNAMIC_BEFORE",
        )
        recording_adapter = RecordingRuntimeAdapter(
            SchedulerRuntimeAdapter(client)
        )
        controller = StateController(
            FrozenSelectedSetOptimizer(
                first_selection["selected_checkpoint_ids"]
            ),
            recording_adapter,
        )
        allocation = controller.reconcile(
            pending_two,
            candidates_one,
            round_one_handles,
            BUDGET_BYTES,
        )
        controller_report = build_controller_report(
            allocation=allocation,
            adapter=recording_adapter,
        )
        after_census = compact_census(
            client.census(f"openhands-dynamic:{policy}:after-reconcile"),
            ordinal=4,
            request=None,
            previous=before_census,
        )
        after_states, _ = inspect_candidate_states(
            client,
            candidates_one,
            round_one_handles,
            phase=f"{policy}_DYNAMIC_AFTER",
        )
        expected_evicted = sorted(
            set(round_one_handles)
            - set(first_selection["selected_checkpoint_ids"])
        )
        mapping_invariants = evaluate_mapping_invariants(
            candidate_ids=list(round_one_handles),
            selected_ids=first_selection["selected_checkpoint_ids"],
            expected_evicted_ids=expected_evicted,
            handles=round_one_handles,
            before_states=before_states,
            after_states=after_states,
            before_census=before_census,
            after_census=after_census,
            controller_report=controller_report,
        )
        if mapping_invariants["status"] != "PASS":
            raise RuntimeError(f"{policy} 的 Barrier 1 reconcile 失败")
        refresh_registry(
            client,
            registry,
            phase=f"{policy}_POST_BARRIER1",
        )
        post_eviction_residency = {
            checkpoint_id: entry.recurrent_resident
            for checkpoint_id, entry in registry.items()
        }
        previous = after_census

        for offset, (label, turn) in enumerate(
            ROUND_TWO_SCHEDULE,
            start=1,
        ):
            ordinal = len(ROUND_ONE_SCHEDULE) + offset
            request = requests[(label, turn)]
            previously_resident = {
                checkpoint_id
                for checkpoint_id, entry in registry.items()
                if entry.recurrent_resident
            }
            record = execute_request(engine, client, request, ordinal)
            census = compact_census(
                client.census(
                    f"openhands-dynamic:{policy}:after:{label}{turn}"
                ),
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            census["event"] = f"after_{label}{turn}"
            event = register_round_two_materialization(
                client,
                registry,
                request,
                record,
                census,
                event_order=ordinal,
                previously_resident_ids=previously_resident,
            )
            registry_events.append(event)
            refresh_registry(
                client,
                registry,
                phase=f"{policy}_{label}{turn}",
            )
            records.append({**record, "policy": policy})
            _append_jsonl(paths.requests, records[-1])
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{policy} 的 {label}2 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError("Round 2 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("Round 2 发生 FA 级联")

        refresh_registry(
            client,
            registry,
            phase=f"{policy}_BARRIER2",
        )
        candidates_two = registry_candidates(registry)
        metadata_rows, _ = build_dynamic_metadata(
            registry,
            candidates_two,
        )
        pending_three, pending_rows = build_round_three_pending(
            barrier_client,
            requests,
            policy=policy,
        )
        post_query = compact_census(
            client.census(f"openhands-dynamic:{policy}:after-barrier2-query"),
            ordinal=8,
            request=None,
            previous=previous,
        )
        query_mutation = bool(
            post_query["added_mamba_node_ids"]
            or post_query["removed_mamba_node_ids"]
            or post_query["changed_existing_mamba_node_ids"]
            or post_query["removed_full_device_node_ids"]
            or post_query["removed_structure_node_ids"]
        )
        if query_mutation:
            raise RuntimeError("Barrier 2 FA frontier 查询改变 runtime")
        second_selection = run_policy_selector(
            policy,
            candidates_two,
            pending_three,
            metadata_rows,
        )
        if not second_selection["selection_valid"]:
            raise RuntimeError("Barrier 2 selector 输出非法")

        rematerialized_ids = {
            event["checkpoint_id"]
            for event in registry_events
            if event["kind"] == "REMATERIALIZED"
        }
        evicted_history_valid = all(
            post_eviction_residency[checkpoint_id]
            or not registry[checkpoint_id].recurrent_resident
            or checkpoint_id in rematerialized_ids
            for checkpoint_id in post_eviction_residency
        )
        new_round_two_ids = [
            checkpoint_id
            for checkpoint_id, entry in registry.items()
            if entry.turn == ROUND_TWO_TURN
        ]
        if not evicted_history_valid:
            raise RuntimeError("已驱逐 checkpoint 被错误标成 resident")
        if not new_round_two_ids:
            raise RuntimeError("Round 2 新 materialized checkpoint 未进入 registry")
        for entry in registry.values():
            row = entry.row()
            row["policy"] = policy
            _append_jsonl(paths.registry, row)
        for row in pending_rows:
            _append_jsonl(paths.pending, {**row, "policy": policy})
        _write_json(
            paths.selections,
            {
                "barrier_1": first_selection,
                "barrier_2": second_selection,
            },
        )
        _write_json(paths.controller, controller_report)
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    future_leakage = _boundary_has_future_leakage(boundary_audit)
    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and len(records) == 8
        and all(record.get("status") == "PASS" for record in records)
        and mapping_invariants is not None
        and mapping_invariants.get("status") == "PASS"
        and second_selection is not None
        and second_selection.get("selection_valid") is True
        and len(pending_rows) == 4
        and not native_eviction
        and not fa_cascade
        and not future_leakage
    )
    return {
        "policy": policy,
        "engine_ordinal": engine_ordinal,
        "engine_lifecycle": "independent_fresh",
        "status": "PASS" if passed else "FAIL",
        "round_1_selection": first_selection,
        "round_1_mapping_invariants": mapping_invariants,
        "round_2_requests": [
            record for record in records if record.get("turn") == 2
        ],
        "barrier_1_post_eviction_residency": (
            post_eviction_residency
            if "post_eviction_residency" in locals()
            else {}
        ),
        "registry_events": registry_events,
        "barrier_2_registry": [
            entry.row() for entry in registry.values()
        ],
        "barrier_2_metadata": metadata_rows,
        "barrier_2_pending": pending_rows,
        "round_2_selection": second_selection,
        "barrier_2_selection_applied": False,
        "round_3_requests_executed": False,
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
        "future_leakage": future_leakage,
        "fatal_error": fatal_error,
        "engine_shutdown_error": shutdown_error,
    }


def build_round_three_pending_compatible(
    barrier_client: BarrierFAControlClient,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    turn: int,
    policy: str,
) -> tuple[tuple[PendingContinuation, ...], list[dict[str, object]]]:
    """为 Barrier 1 复用同一无副作用查询逻辑构造 Round 2 pending。"""
    continuations = []
    rows = []
    for label, workflow_id in WORKFLOWS.items():
        request = requests[(label, turn)]
        input_ids = request["input_ids"]
        lookup = barrier_client.inspect_fa_frontier(
            input_ids,
            extra_key=None,
            limit=None,
            nonce=f"openhands-dynamic:{policy}:{label}:turn-{turn:03d}",
        )
        if not lookup.get("state_equal"):
            raise RuntimeError(f"{label}{turn} FA 查询产生副作用")
        if lookup["scope_before"] != lookup["scope_after"]:
            raise RuntimeError(f"{label}{turn} FA 查询前后 scope 不一致")
        continuation = PendingContinuation(
            continuation_id=f"OPENHANDS_BARRIER_{label}_TURN_{turn:03d}",
            workflow_id=workflow_id,
            lineage_path=lineage_path(workflow_id),
            anchor_pos=len(input_ids),
            resident_fa_frontier=int(lookup["resident_fa_frontier"]),
        )
        continuations.append(continuation)
        rows.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_label": label,
                "anchor_pos": continuation.anchor_pos,
                "resident_fa_frontier": (
                    continuation.resident_fa_frontier
                ),
                "planning_target": continuation.planning_target,
            }
        )
    return tuple(continuations), rows


def build_summary(
    *,
    artifact: Path,
    runs: Sequence[Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """汇总三个 path-dependent policy 的 Barrier 2 selector 门禁。"""
    complete = bool(
        len(runs) == len(POLICY_ORDER)
        and [run.get("policy") for run in runs] == list(POLICY_ORDER)
        and all(run.get("status") == "PASS" for run in runs)
    )
    registry_by_policy = {
        str(run["policy"]): [
            row["checkpoint_id"]
            for row in run.get("barrier_2_registry", ())
            if row.get("recurrent_resident")
        ]
        for run in runs
    }
    future_leakage = _boundary_has_future_leakage(boundary_audit) or any(
        run.get("future_leakage") is True for run in runs
    )
    selections_valid = bool(
        complete
        and all(
            run.get("round_2_selection", {}).get("selection_valid") is True
            and run.get("round_2_selection", {}).get("selected_count", 99)
            <= LOGICAL_K
            for run in runs
        )
    )
    passed = bool(
        complete
        and selections_valid
        and not future_leakage
        and not any(
            run.get("native_mamba_capacity_eviction") for run in runs
        )
        and not any(run.get("fa_kv_cascade") for run in runs)
    )
    return {
        "schema_version": "flowstate.openhands_round2_dynamic_registry.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_DYNAMIC_REGISTRY,
        "policy_order": list(POLICY_ORDER),
        "engine_lifecycle_count": len(runs),
        "fresh_engine_per_policy": True,
        "round_1_schedule": [f"{label}1" for label in WORKFLOWS],
        "round_2_schedule": [f"{label}2" for label in WORKFLOWS],
        "barrier_2_pending_schedule": [
            f"{label}3" for label in WORKFLOWS
        ],
        "logical_k": LOGICAL_K,
        "budget_bytes": BUDGET_BYTES,
        "runs": list(runs),
        "eligible_registry_by_policy": registry_by_policy,
        "candidate_universe_equality_required_at_barrier_2": False,
        "candidate_universe_path_dependent": True,
        "selections_valid": selections_valid,
        "barrier_2_selection_applied": False,
        "round_3_requests_executed": False,
        "native_mamba_capacity_eviction": any(
            run.get("native_mamba_capacity_eviction") for run in runs
        ),
        "fa_kv_cascade": any(
            run.get("fa_kv_cascade") for run in runs
        ),
        "future_leakage": future_leakage,
        "online_information_boundary": list(boundary_audit),
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """顺序执行三个 fresh policy 生命周期并停在第二次 selection。"""
    environment = _environment()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    requests, boundary_audit = load_round2_visible_requests(tokenizer)
    _write_json(
        artifact / "config.json",
        {
            "policies": list(POLICY_ORDER),
            "engine_lifecycle_count": len(POLICY_ORDER),
            "fresh_engine_per_policy": True,
            "engine_configuration": ENGINE_CONFIGURATION_DYNAMIC_REGISTRY,
            "sampling_parameters": SAMPLING_PARAMETERS,
            "logical_k": LOGICAL_K,
            "budget_bytes": BUDGET_BYTES,
            "dataset_path": str(DATASET_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "visible_turns": [1, 2, 3],
            "round_3_assistant_output_read": False,
            "round_4_or_later_materialized": False,
            "barrier_2_selection_applied": False,
            "round_3_requests_executed": False,
            "online_information_boundary": boundary_audit,
            "environment": environment,
        },
    )
    runs = []
    for ordinal, policy in enumerate(POLICY_ORDER, start=1):
        paths = _policy_paths(
            artifact / f"policy_{ordinal:02d}_{policy.lower()}"
        )
        result = run_policy_lifecycle(
            policy=policy,
            engine_ordinal=ordinal,
            requests=requests,
            boundary_audit=boundary_audit,
            paths=paths,
        )
        runs.append(result)
        if result["status"] != "PASS":
            break
    summary = build_summary(
        artifact=artifact,
        runs=runs,
        boundary_audit=boundary_audit,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次 Barrier 2 selector-only 门禁。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
