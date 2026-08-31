#!/usr/bin/env python3
"""执行 OpenHands 四工作流循环状态单独驱逐因果门。"""

from __future__ import annotations

from array import array
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    SchedulerRuntimeAdapter,
    inspect_checkpoint,
    wait_for_transport,
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
    build_request_inputs,
    load_workflow_messages,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from flowstate.adapters.sglang import RuntimeCheckpointHandle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
TARGET_TURNS = (1, 2, 3)
PRE_EVICTION_SCHEDULE = tuple(
    (label, turn) for turn in (1, 2) for label in WORKFLOWS
)
POST_EVICTION_SCHEDULE = tuple((label, 3) for label in WORKFLOWS)
SCHEDULE = PRE_EVICTION_SCHEDULE + POST_EVICTION_SCHEDULE
EVICTION_TARGETS = ("B", "D")
RETAINED_TARGETS = ("A", "C")
RID_TEMPLATE = "openhands-causal-gate-{label}-turn-{turn:03d}"
EARLIER_NONZERO_GAPS = {
    ("B", 1): {"h": 2_437, "e": 0, "g": 2_437},
    ("C", 1): {"h": 2_437, "e": 2_432, "g": 5},
    ("D", 1): {"h": 2_437, "e": 2_432, "g": 5},
    ("B", 2): {"h": 2_437, "e": 2_432, "g": 5},
}
ENGINE_CONFIGURATION_CAUSAL_GATE = {
    **ENGINE_CONFIGURATION_128K,
    "max_mamba_cache_size": 28,
}


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_recurrent_only_causal_gate_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """优先返回仓库内相对路径。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def exact_lcp(left: Sequence[int], right: Sequence[int]) -> int:
    """计算两个令牌序列从起点开始的精确公共前缀长度。"""
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right))


def token_digest(token_ids: Sequence[int]) -> str:
    """计算正式运行时句柄使用的令牌摘要。"""
    values = array("q", (int(value) for value in token_ids))
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_requests(
    tokenizer: object,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """按冻结顺序构造四条 workflow 的前三轮请求。"""
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    recorded_turns: dict[str, int] = {}
    for label, workflow_id in WORKFLOWS.items():
        messages, n_turns = load_workflow_messages(
            dataset_path=DATASET_PATH,
            workflow_id=workflow_id,
        )
        recorded_turns[label] = n_turns
        requests = build_request_inputs(
            tokenizer,
            messages,
            target_turns=TARGET_TURNS,
        )
        previous_ids: list[int] | None = None
        for request in requests:
            turn = int(request["turn"])
            input_ids = request["input_ids"]
            if not isinstance(input_ids, list):
                raise TypeError("input_ids 必须是列表")
            adjacent_lcp = None
            if previous_ids is not None:
                adjacent_lcp = exact_lcp(previous_ids, input_ids)
            previous_ids = input_ids
            by_key[(label, turn)] = {
                "workflow_label": label,
                "workflow_id": workflow_id,
                "turn": turn,
                "rid": RID_TEMPLATE.format(label=label.lower(), turn=turn),
                "input_ids": input_ids,
                "exact_adjacent_lcp": adjacent_lcp,
            }
    return [by_key[key] for key in SCHEDULE], recorded_turns


def audit_earlier_nonzero_gaps(
    requests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """离线检查旧 gap 是否落在跨工作流精确公共前缀内。"""
    seen: list[Mapping[str, object]] = []
    audit: dict[str, object] = {}
    for request in requests:
        key = (str(request["workflow_label"]), int(request["turn"]))
        if key in EARLIER_NONZERO_GAPS:
            input_ids = request["input_ids"]
            if not isinstance(input_ids, list):
                raise TypeError("input_ids 必须是列表")
            prior_lcps = []
            for prior in seen:
                prior_ids = prior["input_ids"]
                if not isinstance(prior_ids, list):
                    raise TypeError("input_ids 必须是列表")
                prior_lcps.append(
                    {
                        "request": (
                            f"{prior['workflow_label']}{int(prior['turn'])}"
                        ),
                        "same_workflow": (
                            prior["workflow_label"] == request["workflow_label"]
                        ),
                        "exact_lcp": exact_lcp(input_ids, prior_ids),
                    }
                )
            cross_lcps = [
                int(item["exact_lcp"])
                for item in prior_lcps
                if not item["same_workflow"]
            ]
            observed = EARLIER_NONZERO_GAPS[key]
            cross_max = max(cross_lcps, default=0)
            possible = cross_max >= int(observed["h"])
            audit[f"{key[0]}{key[1]}"] = {
                **observed,
                "prior_exact_lcps": prior_lcps,
                "max_cross_workflow_lcp": cross_max,
                "cross_workflow_prefix_possible": possible,
                "interpretation": (
                    "POSSIBLE_CROSS_WORKFLOW_COMMON_PREFIX"
                    if possible
                    else "UNRESOLVED"
                ),
                "causal_source_proven": False,
            }
        seen.append(request)
    return audit


def locate_latest_checkpoint(
    client: object,
    request: Mapping[str, object],
    census: Mapping[str, object],
) -> tuple[RuntimeCheckpointHandle, dict[str, object]]:
    """由请求后唯一新增节点定位最新持久循环状态。"""
    added_ids = [int(value) for value in census["added_mamba_node_ids"]]
    if len(added_ids) != 1:
        raise RuntimeError(
            f"{request['workflow_label']}{request['turn']} 未产生唯一新增 Mamba 节点："
            f"{added_ids}"
        )
    node_id = added_ids[0]
    resident_nodes = {
        int(item["node_id"]): item for item in census["resident_mamba_nodes"]
    }
    if node_id not in resident_nodes:
        raise RuntimeError(f"新增节点 {node_id} 不在驻留节点集合中")
    position = int(resident_nodes[node_id]["token_position"])
    input_ids = request["input_ids"]
    if not isinstance(input_ids, list):
        raise TypeError("input_ids 必须是列表")
    if position <= 0 or position > len(input_ids):
        raise RuntimeError(f"checkpoint position 非法：{position}")
    prefix_ids = tuple(int(value) for value in input_ids[:position])
    checkpoint_id = (
        f"OPENHANDS_CAUSAL_{request['workflow_label']}_"
        f"TURN_{int(request['turn']):03d}"
    )
    response = inspect_checkpoint(
        client,
        checkpoint_id=checkpoint_id,
        token_ids=prefix_ids,
    )
    path = response["after"]["path"]
    if int(path["node_id"]) != node_id:
        raise RuntimeError(
            f"census 与 inspect 节点不一致：{node_id} != {path['node_id']}"
        )
    if int(path["prefix_tokens"]) != position:
        raise RuntimeError("inspect 返回的 checkpoint position 不一致")
    if not path["target_full_present"] or not path["path_full_all_present"]:
        raise RuntimeError("checkpoint 的 FA-KV 路径不完整")
    if not path["target_mamba_present"]:
        raise RuntimeError("checkpoint 的 Mamba 状态不在设备上")
    digest = token_digest(prefix_ids)
    if str(path["prefix_sha256"]) != digest:
        raise RuntimeError("checkpoint 前缀摘要不一致")
    handle = RuntimeCheckpointHandle(
        checkpoint_id=checkpoint_id,
        token_ids=prefix_ids,
        expected_node_id=node_id,
        expected_prefix_digest=digest,
    )
    return handle, {
        "workflow": request["workflow_label"],
        "turn": int(request["turn"]),
        "checkpoint_id": checkpoint_id,
        "node_id": node_id,
        "token_position": position,
        "slots": [int(value) for value in resident_nodes[node_id]["slots"]],
        "inspection": response,
    }


def validate_eviction_response(response: Mapping[str, object]) -> bool:
    """验证正式 actuator 返回的循环状态单独驱逐证明。"""
    before = response["before"]
    after = response["after"]
    proof = response["proof"]
    if not all(isinstance(value, Mapping) for value in (before, after, proof)):
        return False
    before_path = before["path"]
    after_path = after["path"]
    required_proof = (
        "same_node",
        "fa_unchanged",
        "path_unchanged",
        "tree_unchanged",
        "only_target_mamba_changed",
        "sanity_check",
        "cascade_called",
        "fa_identity_unchanged",
    )
    return bool(
        before_path["target_mamba_present"]
        and not after_path["target_mamba_present"]
        and before_path["target_full_present"]
        and after_path["target_full_present"]
        and before_path["path_full_all_present"]
        and after_path["path_full_all_present"]
        and all(bool(proof.get(name)) for name in required_proof[:-2])
        and proof.get("cascade_called") is False
        and proof.get("fa_identity_unchanged") is True
    )


def verify_final_checkpoint_state(
    client: object,
    handle: RuntimeCheckpointHandle,
    expected_mamba_present: bool,
) -> dict[str, object]:
    """在两次干预后重新验证精确节点的 FA 与 Mamba 状态。"""
    response = inspect_checkpoint(
        client,
        checkpoint_id=f"{handle.checkpoint_id}_FINAL",
        token_ids=handle.token_ids,
    )
    path = response["after"]["path"]
    valid = bool(
        int(path["node_id"]) == handle.expected_node_id
        and str(path["prefix_sha256"]) == handle.expected_prefix_digest
        and path["target_full_present"]
        and path["path_full_all_present"]
        and bool(path["target_mamba_present"]) is expected_mamba_present
    )
    return {
        "valid": valid,
        "expected_mamba_present": expected_mamba_present,
        "node_id": int(path["node_id"]),
        "token_position": int(path["prefix_tokens"]),
        "target_full_present": bool(path["target_full_present"]),
        "path_full_all_present": bool(path["path_full_all_present"]),
        "target_mamba_present": bool(path["target_mamba_present"]),
        "response": response,
    }


def _census_unexpected_mamba_change(
    census: Mapping[str, object],
    expected_removed_node_id: int | None = None,
) -> bool:
    """判断 census 差分是否包含干预目标之外的 Mamba 删除或替换。"""
    expected = set()
    if expected_removed_node_id is not None:
        expected.add(int(expected_removed_node_id))
    removed = {int(value) for value in census["removed_mamba_node_ids"]}
    changed = {
        int(value) for value in census["changed_existing_mamba_node_ids"]
    }
    return bool((removed - expected) or changed or not expected.issubset(removed))


def _record_request(
    engine: object,
    client: object,
    request: Mapping[str, object],
    ordinal: int,
) -> dict[str, object]:
    """执行请求并补充同工作流相邻轮次的精确 LCP。"""
    record = execute_request(engine, client, request, ordinal)
    record["exact_adjacent_lcp"] = request["exact_adjacent_lcp"]
    record["phase"] = (
        "pre_eviction" if ordinal <= len(PRE_EVICTION_SCHEDULE) else "post_eviction"
    )
    return record


def build_summary(
    *,
    records: Sequence[Mapping[str, object]],
    censuses: Sequence[Mapping[str, object]],
    evictions: Sequence[Mapping[str, object]],
    checkpoint_states: Mapping[str, Mapping[str, object]],
    earlier_gap_audit: Mapping[str, object] | None,
    artifact: Path,
    recorded_turns: Mapping[str, int] | None,
    environment: Mapping[str, object] | None,
    fatal_error: str | None,
    unexpected_native_eviction: bool,
    fa_cascade: bool,
) -> dict[str, object]:
    """构建因果门的完整正确性汇总。"""
    by_key = {
        (str(item["workflow"]), int(item["turn"])): item for item in records
    }
    all_completed = len(records) == len(SCHEDULE) and all(
        item.get("request_completed") is True for item in records
    )
    token_correctness = len(records) == len(SCHEDULE) and all(
        item.get("token_count_exact") is True for item in records
    )
    runtime_valid = len(records) == len(SCHEDULE) and all(
        item.get("runtime_metrics_valid") is True for item in records
    )
    eviction_by_workflow = {
        str(item["workflow"]): item for item in evictions if "workflow" in item
    }
    eviction_success = all(
        label in eviction_by_workflow
        and eviction_by_workflow[label].get("correctness_pass") is True
        and checkpoint_states.get(label, {}).get("valid") is True
        for label in EVICTION_TARGETS
    )
    retained_valid = all(
        checkpoint_states.get(label, {}).get("valid") is True
        for label in RETAINED_TARGETS
    )
    evicted_gap = {}
    for label in EVICTION_TARGETS:
        record = by_key.get((label, 3), {})
        evicted_gap[label] = bool(
            record.get("request_completed") is True
            and record.get("h") is not None
            and record.get("e") is not None
            and record.get("g") is not None
            and int(record["h"]) > int(record["e"])
            and int(record["g"]) > 0
        )
    retained_requests_valid = {}
    for label in RETAINED_TARGETS:
        record = by_key.get((label, 3), {})
        checkpoint_position = checkpoint_states.get(label, {}).get("token_position")
        retained_requests_valid[label] = bool(
            record.get("request_completed") is True
            and record.get("runtime_metrics_valid") is True
            and checkpoint_position is not None
            and record.get("e") is not None
            and int(record["e"]) >= int(checkpoint_position)
        )
    oom = any(item.get("oom") is True for item in records)
    clipping = any(
        item.get("truncation_or_clipping") is True for item in records
    )
    passed = bool(
        fatal_error is None
        and all_completed
        and token_correctness
        and runtime_valid
        and eviction_success
        and retained_valid
        and all(evicted_gap.values())
        and all(retained_requests_valid.values())
        and not unexpected_native_eviction
        and not fa_cascade
        and not oom
        and not clipping
    )
    return {
        "schema_version": "flowstate.openhands_recurrent_only_causal_gate.v1",
        "status": "PASS" if passed else "FAIL",
        "schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
        "intervention_after": "D2",
        "eviction_targets": ["B2", "D2"],
        "retained_targets": ["A2", "C2"],
        "workflows": WORKFLOWS,
        "recorded_n_turns": dict(recorded_turns or {}),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_CAUSAL_GATE,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "dataset_path": str(DATASET_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "artifact": _display_path(artifact),
        "request_count": len(records),
        "requests": list(records),
        "census_count": len(censuses),
        "evictions": list(evictions),
        "checkpoint_states_after_intervention": dict(checkpoint_states),
        "all_requests_completed": all_completed,
        "token_correctness": token_correctness,
        "runtime_semantic_checks": "PASS" if runtime_valid else "FAIL",
        "recurrent_only_eviction_success": eviction_success,
        "fa_kv_preserved": eviction_success,
        "evicted_workflows_show_h_gt_e": evicted_gap,
        "non_evicted_workflows_remain_valid": retained_requests_valid,
        "retained_checkpoint_state_valid": retained_valid,
        "native_mamba_capacity_eviction_observed": unexpected_native_eviction,
        "fa_kv_cascade_eviction_observed": fa_cascade,
        "oom": oom,
        "truncation_or_clipping": clipping,
        "policy_executed": False,
        "logical_k_allocator_executed": False,
        "active_recurrent_only_interventions": ["B2", "D2"],
        "earlier_natural_nonzero_gap_audit": dict(earlier_gap_audit or {}),
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """执行唯一一次十二请求与两次循环状态单独驱逐实验。"""
    requests_path = artifact / "requests.jsonl"
    census_path = artifact / "census.jsonl"
    evictions_path = artifact / "evictions.jsonl"
    for path in (requests_path, census_path, evictions_path):
        path.write_text("", encoding="utf-8")
    records: list[dict[str, object]] = []
    censuses: list[dict[str, object]] = []
    evictions: list[dict[str, object]] = []
    checkpoint_states: dict[str, dict[str, object]] = {}
    request_censuses: dict[tuple[str, int], dict[str, object]] = {}
    requests_by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    fatal_error = None
    unexpected_native_eviction = False
    fa_cascade = False
    engine = None
    recorded_turns = None
    environment = None
    earlier_gap_audit = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests, recorded_turns = prepare_requests(tokenizer)
        requests_by_key = {
            (str(item["workflow_label"]), int(item["turn"])): item
            for item in requests
        }
        earlier_gap_audit = audit_earlier_nonzero_gaps(requests)
        _write_json(
            artifact / "config.json",
            {
                "workflows": WORKFLOWS,
                "schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
                "intervention": {
                    "after": "D2",
                    "evict_recurrent_only": ["B2", "D2"],
                    "retain": ["A2", "C2"],
                    "formal_policy": False,
                },
                "recorded_n_turns": recorded_turns,
                "offline_input_tokens": [
                    {
                        "workflow": request["workflow_label"],
                        "turn": request["turn"],
                        "tokens": len(request["input_ids"]),
                        "exact_adjacent_lcp": request["exact_adjacent_lcp"],
                    }
                    for request in requests
                ],
                "engine_configuration": ENGINE_CONFIGURATION_CAUSAL_GATE,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "environment": environment,
                "policy_executed": False,
                "logical_k_allocator_executed": False,
                "concurrency": 1,
                "earlier_natural_nonzero_gap_audit": earlier_gap_audit,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_CAUSAL_GATE)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census("openhands_causal:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["event"] = "baseline"
        censuses.append(baseline)
        _append_jsonl(census_path, baseline)
        if baseline["mamba_node_count"] != 0:
            raise RuntimeError("Engine 初始 census 含有非空 Mamba checkpoint")

        previous = baseline
        for ordinal, key in enumerate(PRE_EVICTION_SCHEDULE, start=1):
            request = requests_by_key[key]
            try:
                record = _record_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(f"openhands_causal:after:{ordinal:03d}"),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["event"] = f"after_{key[0]}{key[1]}"
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                record["exact_adjacent_lcp"] = request["exact_adjacent_lcp"]
                record["phase"] = "pre_eviction"
                records.append(record)
                _append_jsonl(requests_path, record)
                raise
            records.append(record)
            censuses.append(census)
            request_censuses[key] = census
            _append_jsonl(requests_path, record)
            _append_jsonl(census_path, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError("服务端 prompt_tokens 与离线长度不一致")
            if census["native_mamba_capacity_eviction_inferred"]:
                unexpected_native_eviction = True
                raise RuntimeError("驱逐前观察到原生 Mamba capacity eviction")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("驱逐前观察到 FA-KV 节点删除")

        handles: dict[str, RuntimeCheckpointHandle] = {}
        checkpoint_info: dict[str, dict[str, object]] = {}
        for label in WORKFLOWS:
            request = requests_by_key[(label, 2)]
            handle, info = locate_latest_checkpoint(
                client,
                request,
                request_censuses[(label, 2)],
            )
            handles[label] = handle
            checkpoint_info[label] = info

        pre_eviction_census = previous
        runtime_adapter = SchedulerRuntimeAdapter(client)
        for label in EVICTION_TARGETS:
            handle = handles[label]
            runtime_adapter.evict_mamba_only(handle)
            response = runtime_adapter.eviction_responses[-1]
            correctness = validate_eviction_response(response)
            eviction_record = {
                "workflow": label,
                "turn": 2,
                "checkpoint_id": handle.checkpoint_id,
                "node_id": handle.expected_node_id,
                "token_position": len(handle.token_ids),
                "pre_request_metrics": {
                    name: next(
                        item[name]
                        for item in records
                        if item["workflow"] == label and item["turn"] == 2
                    )
                    for name in ("h", "e", "g")
                },
                "correctness_pass": correctness,
                "formal_response": response,
            }
            evictions.append(eviction_record)
            _append_jsonl(evictions_path, eviction_record)
            eviction_census = compact_census(
                client.census(f"openhands_causal:after_evict:{label}2"),
                ordinal=len(PRE_EVICTION_SCHEDULE),
                request=requests_by_key[(label, 2)],
                previous=previous,
            )
            eviction_census["event"] = f"after_evict_{label}2"
            eviction_census["expected_recurrent_only_removal_node_id"] = (
                handle.expected_node_id
            )
            censuses.append(eviction_census)
            _append_jsonl(census_path, eviction_census)
            if _census_unexpected_mamba_change(
                eviction_census,
                handle.expected_node_id,
            ):
                unexpected_native_eviction = True
            if eviction_census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
            previous = eviction_census
            if not correctness or unexpected_native_eviction or fa_cascade:
                raise RuntimeError(f"{label}2 recurrent-only correctness 检查失败")

        if (
            pre_eviction_census["full_allocator"] != previous["full_allocator"]
            or pre_eviction_census["full_device_node_ids"]
            != previous["full_device_node_ids"]
            or pre_eviction_census["structure_node_ids"]
            != previous["structure_node_ids"]
        ):
            fa_cascade = True
            raise RuntimeError("两次 recurrent-only 干预后 FA-KV 或树身份发生变化")

        for label in WORKFLOWS:
            expected_present = label in RETAINED_TARGETS
            state = verify_final_checkpoint_state(
                client,
                handles[label],
                expected_present,
            )
            state.update(
                {
                    "workflow": label,
                    "turn": 2,
                    "slots_before": checkpoint_info[label]["slots"],
                }
            )
            checkpoint_states[label] = state
        if not all(item["valid"] for item in checkpoint_states.values()):
            raise RuntimeError("驱逐后 checkpoint/FA-KV 保持检查失败")

        start = len(PRE_EVICTION_SCHEDULE) + 1
        for offset, key in enumerate(POST_EVICTION_SCHEDULE):
            ordinal = start + offset
            request = requests_by_key[key]
            try:
                record = _record_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(f"openhands_causal:after:{ordinal:03d}"),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["event"] = f"after_{key[0]}{key[1]}"
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                record["exact_adjacent_lcp"] = request["exact_adjacent_lcp"]
                record["phase"] = "post_eviction"
                records.append(record)
                _append_jsonl(requests_path, record)
                raise
            records.append(record)
            censuses.append(census)
            _append_jsonl(requests_path, record)
            _append_jsonl(census_path, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError("服务端 prompt_tokens 与离线长度不一致")
            if census["native_mamba_capacity_eviction_inferred"]:
                unexpected_native_eviction = True
                raise RuntimeError("干预后观察到原生 Mamba capacity eviction")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("干预后观察到 FA-KV 节点删除")
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
        records=records,
        censuses=censuses,
        evictions=evictions,
        checkpoint_states=checkpoint_states,
        earlier_gap_audit=earlier_gap_audit,
        artifact=artifact,
        recorded_turns=recorded_turns,
        environment=environment,
        fatal_error=fatal_error,
        unexpected_native_eviction=unexpected_native_eviction,
        fa_cascade=fa_cascade,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次因果门实验。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
