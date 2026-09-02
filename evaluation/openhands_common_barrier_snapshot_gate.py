#!/usr/bin/env python3
"""验证四工作流共同 barrier 的在线 policy 输入能否完整构造。"""

from __future__ import annotations

from array import array
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    inspect_checkpoint,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    CheckpointRecency,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    PHYSICAL_MAX_MAMBA_CACHE_SIZE,
    WORKFLOWS,
    _environment,
    _failure_record,
    compact_census,
    execute_request,
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
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_marconi_flop_saved,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
EXECUTED_TURN = 1
PENDING_TURN = 2
SCHEDULE = tuple((label, EXECUTED_TURN) for label in WORKFLOWS)
RID_TEMPLATE = "openhands-policy-input-{label}-turn-{turn:03d}"
LOGICAL_K = 2
BUDGET_BYTES = LOGICAL_K * CHECKPOINT_SIZE_BYTES
ENGINE_CONFIGURATION_COMMON_BARRIER = {
    **ENGINE_CONFIGURATION_128K,
    "max_mamba_cache_size": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
}


def _artifact_directory() -> Path:
    """创建不会覆盖已有结果的独立产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_common_barrier_policy_input_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径使用便于审计的相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def token_digest(token_ids: Sequence[int]) -> str:
    """计算运行时句柄采用的稳定令牌摘要。"""
    values = array("q", (int(value) for value in token_ids))
    return hashlib.sha256(values.tobytes()).hexdigest()


def lineage_path(workflow_id: str) -> tuple[str, ...]:
    """返回冻结 OpenHands workflow 的单 lineage 标识。"""
    return ("openhands", workflow_id)


def materialize_visible_requests(
    tokenizer: object,
    raw_messages: Sequence[Mapping[str, object]],
    *,
    workflow_label: str,
    workflow_id: str,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    """只消费已执行 turn 1 与当前 pending turn 2 所需的消息。"""
    history: list[dict[str, object]] = []
    requests: dict[int, dict[str, object]] = {}
    assistant_turn = 0
    raw_items_iterated = 0
    assistant_two_output_read = False

    for raw_message in raw_messages:
        raw_items_iterated += 1
        role = raw_message.get("role")
        if role == "assistant":
            assistant_turn += 1
            if assistant_turn in (EXECUTED_TURN, PENDING_TURN):
                output = tokenizer.apply_chat_template(
                    list(history),
                    tokenize=True,
                    add_generation_prompt=True,
                )
                input_ids = _template_input_ids(output)
                requests[assistant_turn] = {
                    "workflow_label": workflow_label,
                    "workflow_id": workflow_id,
                    "turn": assistant_turn,
                    "rid": RID_TEMPLATE.format(
                        label=workflow_label.lower(),
                        turn=assistant_turn,
                    ),
                    "input_ids": input_ids,
                }
            if assistant_turn == PENDING_TURN:
                break

        history.append(normalize_message(raw_message))

    if tuple(sorted(requests)) != (EXECUTED_TURN, PENDING_TURN):
        raise RuntimeError(
            f"workflow {workflow_label} 缺少 turn 1 或 turn 2 请求"
        )
    return requests, {
        "workflow": workflow_label,
        "maximum_assistant_turn_consumed": PENDING_TURN,
        "raw_items_iterated_through_pending_marker": raw_items_iterated,
        "pending_assistant_output_read": assistant_two_output_read,
        "r_plus_2_message_consumed": False,
        "r_plus_2_request_materialized": False,
    }


def load_barrier_requests(
    tokenizer: object,
) -> tuple[dict[tuple[str, int], dict[str, object]], list[dict[str, object]]]:
    """从每条 trajectory 只物化 A1/B1/C1/D1 与 A2/B2/C2/D2。"""
    requests: dict[tuple[str, int], dict[str, object]] = {}
    boundary_audit = []
    for label, workflow_id in WORKFLOWS.items():
        table = pq.read_table(
            DATASET_PATH,
            filters=[("session_id", "=", workflow_id)],
            columns=["session_id", "messages_json"],
        )
        if table.num_rows != 1:
            raise RuntimeError(
                f"workflow {label} 应唯一命中一行，实际为 {table.num_rows} 行"
            )
        serialized = table.column("messages_json")[0].as_py()
        raw_messages = json.loads(serialized)
        if not isinstance(raw_messages, list):
            raise TypeError("messages_json 反序列化后必须是列表")
        visible, audit = materialize_visible_requests(
            tokenizer,
            raw_messages,
            workflow_label=label,
            workflow_id=workflow_id,
        )
        for turn, request in visible.items():
            requests[(label, turn)] = request
        boundary_audit.append(audit)
    return requests, boundary_audit


def locate_materialized_candidate(
    client: object,
    request: Mapping[str, object],
    census: Mapping[str, object],
    *,
    event_order: int,
) -> tuple[CheckpointCandidate, RuntimeCheckpointHandle, dict[str, object]]:
    """由本次请求后唯一新增节点构造真实 checkpoint candidate。"""
    added_ids = [int(value) for value in census["added_mamba_node_ids"]]
    if len(added_ids) != 1:
        raise RuntimeError(
            f"{request['workflow_label']}1 未产生唯一新增 Mamba 节点：{added_ids}"
        )
    node_id = added_ids[0]
    resident_nodes = {
        int(item["node_id"]): item
        for item in census["resident_mamba_nodes"]
    }
    if node_id not in resident_nodes:
        raise RuntimeError(f"新增节点 {node_id} 未处于 Mamba 驻留集合")
    position = int(resident_nodes[node_id]["token_position"])
    raw_input_ids = request["input_ids"]
    if not isinstance(raw_input_ids, list):
        raise TypeError("input_ids 必须是列表")
    if not 0 < position <= len(raw_input_ids):
        raise RuntimeError(f"checkpoint position 非法：{position}")
    prefix_ids = tuple(int(value) for value in raw_input_ids[:position])
    checkpoint_id = (
        f"OPENHANDS_BARRIER_{request['workflow_label']}_TURN_001"
    )
    digest = token_digest(prefix_ids)
    response = inspect_checkpoint(client, checkpoint_id, prefix_ids)
    path = response["after"]["path"]
    fa_resident = bool(
        path["target_full_present"] and path["path_full_all_present"]
    )
    recurrent_resident = bool(path["target_mamba_present"])
    if int(path["node_id"]) != node_id:
        raise RuntimeError(
            f"census 与 inspect 节点不一致：{node_id} != {path['node_id']}"
        )
    if int(path["prefix_tokens"]) != position:
        raise RuntimeError("census 与 inspect token position 不一致")
    if str(path["prefix_sha256"]) != digest:
        raise RuntimeError("运行时 checkpoint 前缀摘要不一致")
    if not fa_resident or not recurrent_resident:
        raise RuntimeError("真实 checkpoint 的 FA 或 Mamba 组件未驻留")

    workflow_id = str(request["workflow_id"])
    candidate = CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path(workflow_id),
        token_pos=position,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=recurrent_resident,
        fa_resident=fa_resident,
    )
    handle = RuntimeCheckpointHandle(
        checkpoint_id=checkpoint_id,
        token_ids=prefix_ids,
        extra_key=None,
        expected_node_id=node_id,
        expected_prefix_digest=digest,
    )
    row = {
        "checkpoint_id": checkpoint_id,
        "workflow_label": request["workflow_label"],
        "workflow_id": workflow_id,
        "lineage_path": list(candidate.lineage_path),
        "token_pos": position,
        "memory_bytes": candidate.memory_bytes,
        "recurrent_resident": recurrent_resident,
        "fa_resident": fa_resident,
        "node_id": node_id,
        "slots": [int(value) for value in resident_nodes[node_id]["slots"]],
        "creation_order": event_order,
        "last_access_order": event_order,
        "runtime_handle": {
            "checkpoint_id": handle.checkpoint_id,
            "token_ids": list(handle.token_ids),
            "extra_key": handle.extra_key,
            "expected_node_id": handle.expected_node_id,
            "expected_prefix_digest": handle.expected_prefix_digest,
        },
        "materialization_proof": {
            "request_ordinal": event_order,
            "added_mamba_node_id": node_id,
            "inspect_node_id": int(path["node_id"]),
            "inspect_prefix_tokens": int(path["prefix_tokens"]),
            "target_full_present": bool(path["target_full_present"]),
            "path_full_all_present": bool(path["path_full_all_present"]),
            "target_mamba_present": bool(path["target_mamba_present"]),
            "prefix_digest_match": str(path["prefix_sha256"]) == digest,
        },
    }
    return candidate, handle, row


def validate_candidate_at_barrier(
    client: object,
    candidate: CheckpointCandidate,
    handle: RuntimeCheckpointHandle,
) -> dict[str, object]:
    """在 D1 完成后的同一 barrier 再次核对 candidate residency。"""
    response = inspect_checkpoint(
        client,
        f"{candidate.checkpoint_id}_BARRIER",
        handle.token_ids,
    )
    path = response["after"]["path"]
    consistent = bool(
        int(path["node_id"]) == handle.expected_node_id
        and int(path["prefix_tokens"]) == candidate.token_pos
        and str(path["prefix_sha256"]) == handle.expected_prefix_digest
        and bool(path["target_full_present"]) is candidate.fa_resident
        and bool(path["path_full_all_present"])
        and bool(path["target_mamba_present"])
        is candidate.recurrent_resident
    )
    return {
        "consistent": consistent,
        "node_id": int(path["node_id"]),
        "token_pos": int(path["prefix_tokens"]),
        "fa_resident": bool(
            path["target_full_present"] and path["path_full_all_present"]
        ),
        "recurrent_resident": bool(path["target_mamba_present"]),
        "prefix_digest": str(path["prefix_sha256"]),
    }


def build_pending_set(
    barrier_client: BarrierFAControlClient,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[tuple[PendingContinuation, ...], list[dict[str, object]]]:
    """只用当前 turn 2 input_ids 构造四个 pending continuation。"""
    continuations = []
    rows = []
    for label, workflow_id in WORKFLOWS.items():
        request = requests[(label, PENDING_TURN)]
        input_ids = request["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("pending input_ids 必须是列表")
        lookup = barrier_client.inspect_fa_frontier(
            input_ids,
            extra_key=None,
            limit=None,
            nonce=f"openhands-policy-input:{label}:turn-002",
        )
        if not lookup.get("state_equal"):
            raise RuntimeError(
                f"{label}2 FA frontier 查询改变状态：{lookup['changed_fields']}"
            )
        if lookup["scope_before"] != lookup["scope_after"]:
            raise RuntimeError(f"{label}2 查询前后 runtime scope 不一致")
        anchor_pos = len(input_ids)
        resident_fa_frontier = int(lookup["resident_fa_frontier"])
        continuation = PendingContinuation(
            continuation_id=f"OPENHANDS_BARRIER_{label}_TURN_002",
            workflow_id=workflow_id,
            lineage_path=lineage_path(workflow_id),
            anchor_pos=anchor_pos,
            resident_fa_frontier=resident_fa_frontier,
        )
        continuations.append(continuation)
        rows.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_label": label,
                "workflow_id": workflow_id,
                "lineage_path": list(continuation.lineage_path),
                "anchor_pos": continuation.anchor_pos,
                "resident_fa_frontier": resident_fa_frontier,
                "planning_target": continuation.planning_target,
                "input_token_digest": token_digest(input_ids),
                "effective_lookup_limit": int(
                    lookup["effective_lookup_limit"]
                ),
                "extra_key": lookup["extra_key"],
                "page_size": int(lookup["page_size"]),
                "traversed_node_ids": list(lookup["traversed_node_ids"]),
                "partial_match": bool(lookup["partial_match"]),
                "stop_reason": lookup["stop_reason"],
                "query_state_equal": True,
                "query_changed_fields": list(lookup["changed_fields"]),
            }
        )
    return tuple(continuations), rows


def build_policy_metadata(
    candidates: Sequence[CheckpointCandidate],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], tuple[CheckpointRecency, ...]]:
    """按既有 LRU 与 Marconi 冻结语义构造在线 metadata。"""
    validate_unique_checkpoint_ids(candidates)
    row_by_id = {str(row["checkpoint_id"]): row for row in candidate_rows}
    recency_records = tuple(
        CheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=int(row_by_id[candidate.checkpoint_id]["creation_order"]),
            last_access_order=int(
                row_by_id[candidate.checkpoint_id]["last_access_order"]
            ),
        )
        for candidate in candidates
    )
    incremental_spans = build_marconi_flop_saved(candidates)
    metadata_rows = []
    for candidate in candidates:
        recency = next(
            item
            for item in recency_records
            if item.checkpoint_id == candidate.checkpoint_id
        )
        span = float(incremental_spans[candidate.checkpoint_id])
        metadata_rows.append(
            {
                "checkpoint_id": candidate.checkpoint_id,
                "creation_order": recency.creation_order,
                "last_access_order": recency.last_access_order,
                "marconi_recency": float(recency.last_access_order),
                "marconi_incremental_span": span,
                "marconi_raw_flop_efficiency": (
                    span / candidate.memory_bytes
                ),
                "marconi_first_checkpoint": True,
                "marconi_parent_position": 0,
                "marconi_alpha": CONTROLLED_MARCONI_ALPHA,
            }
        )
    return metadata_rows, recency_records


def common_candidate_universe(
    candidates: Sequence[CheckpointCandidate],
) -> dict[str, object]:
    """冻结三种策略共同消费且本步骤不选择的 candidate universe。"""
    candidate_ids = tuple(
        candidate.checkpoint_id
        for candidate in sorted(candidates, key=lambda item: item.checkpoint_id)
    )
    by_policy = {
        "LRU": list(candidate_ids),
        "Marconi": list(candidate_ids),
        "FlowState": list(candidate_ids),
    }
    return {
        "candidate_ids_by_policy": by_policy,
        "all_equal": len({tuple(value) for value in by_policy.values()}) == 1,
    }


def build_summary(
    *,
    artifact: Path,
    records: Sequence[Mapping[str, object]],
    censuses: Sequence[Mapping[str, object]],
    candidates: Sequence[CheckpointCandidate],
    candidate_rows: Sequence[Mapping[str, object]],
    pending: Sequence[PendingContinuation],
    pending_rows: Sequence[Mapping[str, object]],
    metadata_rows: Sequence[Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    universe: Mapping[str, object],
    environment: Mapping[str, object] | None,
    fatal_error: str | None,
) -> dict[str, object]:
    """按十四项 gate 条件汇总共同 policy 输入快照。"""
    request_censuses = [
        census for census in censuses if int(census["request_ordinal"]) > 0
    ]
    executed = bool(
        len(records) == len(SCHEDULE)
        and all(record.get("status") == "PASS" for record in records)
    )
    native_eviction = any(
        bool(census["native_mamba_capacity_eviction_inferred"])
        for census in request_censuses
    )
    fa_cascade = any(
        bool(census["fa_kv_cascade_eviction_inferred"])
        for census in request_censuses
    )
    materialized = bool(
        len(candidate_rows) == len(WORKFLOWS)
        and all(
            row.get("materialization_proof", {}).get(
                "target_mamba_present"
            )
            for row in candidate_rows
        )
    )
    residency_consistent = bool(
        len(candidate_rows) == len(WORKFLOWS)
        and all(
            row.get("barrier_validation", {}).get("consistent") is True
            for row in candidate_rows
        )
    )
    query_state_equal = bool(
        len(pending_rows) == len(WORKFLOWS)
        and all(row.get("query_state_equal") is True for row in pending_rows)
    )
    metadata_ready = bool(
        len(metadata_rows) == len(candidate_rows) == len(WORKFLOWS)
        and all(
            row.get("marconi_parent_position") == 0
            and row.get("marconi_incremental_span") is not None
            for row in metadata_rows
        )
    )
    future_information_used = any(
        audit.get("r_plus_2_message_consumed") is not False
        or audit.get("r_plus_2_request_materialized") is not False
        or audit.get("pending_assistant_output_read") is not False
        for audit in boundary_audit
    )
    flowstate_ready = bool(
        len(candidates) == len(pending) == len(WORKFLOWS)
        and all(candidate.recurrent_resident for candidate in candidates)
        and all(
            continuation.planning_target >= 0 for continuation in pending
        )
    )
    passed = bool(
        fatal_error is None
        and executed
        and not native_eviction
        and not fa_cascade
        and materialized
        and residency_consistent
        and len(pending_rows) == len(WORKFLOWS)
        and query_state_equal
        and metadata_ready
        and flowstate_ready
        and universe.get("all_equal") is True
        and not future_information_used
    )
    return {
        "schema_version": "flowstate.openhands_common_barrier_snapshot.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "workflows": WORKFLOWS,
        "schedule": [f"{label}1" for label in WORKFLOWS],
        "pending_schedule": [f"{label}2" for label in WORKFLOWS],
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_COMMON_BARRIER,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "physical_max_mamba_cache_size": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
        "logical_k": LOGICAL_K,
        "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
        "budget_bytes": BUDGET_BYTES,
        "executed_requests": list(records),
        "all_executed_requests_passed": executed,
        "native_mamba_capacity_eviction_observed": native_eviction,
        "fa_kv_cascade_eviction_observed": fa_cascade,
        "candidate_count": len(candidate_rows),
        "candidates": list(candidate_rows),
        "all_candidates_materialized": materialized,
        "runtime_residency_consistent": residency_consistent,
        "pending_count": len(pending_rows),
        "pending": list(pending_rows),
        "fa_frontier_query_state_equality": query_state_equal,
        "lru_metadata_ready": metadata_ready,
        "marconi_metadata_ready": metadata_ready,
        "marconi_alpha": CONTROLLED_MARCONI_ALPHA,
        "marconi_first_checkpoint_semantics": (
            "复用 build_marconi_flop_saved：没有同 workflow/lineage 的更浅候选时，"
            "parent_pos=0，incremental_span=token_pos"
        ),
        "metadata": list(metadata_rows),
        "flowstate_input_ready": flowstate_ready,
        "common_candidate_universe": dict(universe),
        "online_information_boundary": list(boundary_audit),
        "future_information_used": future_information_used,
        "future_leakage": future_information_used,
        "policy_selection_executed": False,
        "flowstate_optimizer_executed": False,
        "controller_reconcile_executed": False,
        "recurrent_eviction_executed": False,
        "pending_requests_executed": False,
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """执行 A1/B1/C1/D1，并在共同 barrier 冻结 policy 输入。"""
    requests_path = artifact / "requests.jsonl"
    census_path = artifact / "census.jsonl"
    candidates_path = artifact / "candidates.jsonl"
    pending_path = artifact / "pending.jsonl"
    for path in (
        requests_path,
        census_path,
        candidates_path,
        pending_path,
    ):
        path.write_text("", encoding="utf-8")

    records: list[dict[str, object]] = []
    censuses: list[dict[str, object]] = []
    candidates: list[CheckpointCandidate] = []
    handles: list[RuntimeCheckpointHandle] = []
    candidate_rows: list[dict[str, object]] = []
    pending: tuple[PendingContinuation, ...] = ()
    pending_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    boundary_audit: list[dict[str, object]] = []
    universe: dict[str, object] = {
        "candidate_ids_by_policy": {},
        "all_equal": False,
    }
    fatal_error = None
    environment = None
    engine = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests, boundary_audit = load_barrier_requests(tokenizer)
        _write_json(
            artifact / "config.json",
            {
                "workflows": WORKFLOWS,
                "executed_schedule": [f"{label}1" for label in WORKFLOWS],
                "pending_schedule": [f"{label}2" for label in WORKFLOWS],
                "visible_request_metadata": [
                    {
                        "workflow": label,
                        "turn": turn,
                        "input_tokens": len(requests[(label, turn)]["input_ids"]),
                        "input_token_digest": token_digest(
                            requests[(label, turn)]["input_ids"]
                        ),
                    }
                    for label in WORKFLOWS
                    for turn in (EXECUTED_TURN, PENDING_TURN)
                ],
                "engine_configuration": ENGINE_CONFIGURATION_COMMON_BARRIER,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "logical_k": LOGICAL_K,
                "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
                "budget_bytes": BUDGET_BYTES,
                "online_information_boundary": boundary_audit,
                "policy_selection_executed": False,
                "pending_requests_executed": False,
                "environment": environment,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_COMMON_BARRIER
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census("openhands-policy-input:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["event"] = "baseline"
        censuses.append(baseline)
        _append_jsonl(census_path, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("Engine 初始 census 含有 Mamba checkpoint")

        previous = baseline
        for ordinal, (label, turn) in enumerate(SCHEDULE, start=1):
            request = requests[(label, turn)]
            try:
                record = execute_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(
                        f"openhands-policy-input:after:{label}{turn}"
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
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                records.append(record)
                _append_jsonl(requests_path, record)
                raise
            records.append(record)
            censuses.append(census)
            candidates.append(candidate)
            handles.append(handle)
            candidate_rows.append(row)
            _append_jsonl(requests_path, record)
            _append_jsonl(census_path, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{label}{turn} 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                raise RuntimeError("观察到原生 Mamba capacity eviction")
            if census["fa_kv_cascade_eviction_inferred"]:
                raise RuntimeError("观察到 FA-KV cascade eviction")

        validate_unique_checkpoint_ids(candidates)
        for index, (candidate, handle) in enumerate(
            zip(candidates, handles)
        ):
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                handle,
            )
            candidate_rows[index]["barrier_validation"] = validation
            if not validation["consistent"]:
                raise RuntimeError(
                    f"candidate {candidate.checkpoint_id} barrier residency 不一致"
                )

        barrier_client = BarrierFAControlClient(client)
        pending, pending_rows = build_pending_set(barrier_client, requests)
        post_query = compact_census(
            client.census("openhands-policy-input:after-pending-lookups"),
            ordinal=len(SCHEDULE),
            request=None,
            previous=previous,
        )
        post_query["event"] = "after_pending_lookups"
        censuses.append(post_query)
        _append_jsonl(census_path, post_query)
        if (
            post_query["added_mamba_node_ids"]
            or post_query["removed_mamba_node_ids"]
            or post_query["changed_existing_mamba_node_ids"]
            or post_query["removed_full_device_node_ids"]
            or post_query["removed_structure_node_ids"]
        ):
            raise RuntimeError("四次 pending 查询后 runtime census 发生变化")

        metadata_rows, _ = build_policy_metadata(
            candidates,
            candidate_rows,
        )
        metadata_by_id = {
            str(row["checkpoint_id"]): row for row in metadata_rows
        }
        for row in candidate_rows:
            row["policy_metadata"] = metadata_by_id[row["checkpoint_id"]]
            _append_jsonl(candidates_path, row)
        for row in pending_rows:
            _append_jsonl(pending_path, row)
        universe = common_candidate_universe(candidates)
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = f"关闭 Engine 失败：{error!r}"

    summary = build_summary(
        artifact=artifact,
        records=records,
        censuses=censuses,
        candidates=candidates,
        candidate_rows=candidate_rows,
        pending=pending,
        pending_rows=pending_rows,
        metadata_rows=metadata_rows,
        boundary_audit=boundary_audit,
        universe=universe,
        environment=environment,
        fatal_error=fatal_error,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次共同 barrier 输入门。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
