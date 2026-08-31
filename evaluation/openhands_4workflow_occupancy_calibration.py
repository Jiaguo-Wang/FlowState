#!/usr/bin/env python3
"""执行四条 OpenHands workflow 的循环顺序状态占用校准。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    inspect_checkpoint,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _optional_int,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    build_request_inputs,
    load_workflow_messages,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from evaluation.sota_latency_runtime import measure_streaming_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
WORKFLOWS = {
    "A": "nebius-swe-rebench-openhands::chatcmpl-0063c3ccef5e68d790c496c97203112c",
    "B": "nebius-swe-rebench-openhands::chatcmpl-001958ec05de484109affffb7e57723c",
    "C": "nebius-swe-rebench-openhands::chatcmpl-00f5b1870a29222c1c6ab527e7726cfb",
    "D": "nebius-swe-rebench-openhands::chatcmpl-0069c298d0f1b6ce337cf5dae899deae",
}
TARGET_TURNS = tuple(range(1, 6))
SCHEDULE = tuple(
    (label, turn) for turn in TARGET_TURNS for label in WORKFLOWS
)
RID_TEMPLATE = "openhands-occupancy-{label}-turn-{turn:03d}"

PHYSICAL_MAX_MAMBA_CACHE_SIZE = 28
DEFAULT_MAX_MAMBA_CACHE_SIZE = int(
    ENGINE_CONFIGURATION_128K["max_mamba_cache_size"]
)
LINEAR_LAYER_COUNT = 24
CONV_ELEMENTS_PER_LAYER = 8_192 * 3
TEMPORAL_ELEMENTS_PER_LAYER = 32 * 128 * 128
MAMBA_SLOT_BYTES = LINEAR_LAYER_COUNT * (
    CONV_ELEMENTS_PER_LAYER * 2 + TEMPORAL_ELEMENTS_PER_LAYER * 4
)
MAMBA_POOL_BYTES = (PHYSICAL_MAX_MAMBA_CACHE_SIZE + 1) * MAMBA_SLOT_BYTES
ADDITIONAL_POOL_BYTES = (
    PHYSICAL_MAX_MAMBA_CACHE_SIZE - DEFAULT_MAX_MAMBA_CACHE_SIZE
) * MAMBA_SLOT_BYTES
EXPECTED_PERSISTENT_CHECKPOINTS = len(SCHEDULE)
RUNTIME_WORKING_SLOT_RESERVE = 4
EXPLICIT_SPARE_SLOTS = (
    PHYSICAL_MAX_MAMBA_CACHE_SIZE
    - EXPECTED_PERSISTENT_CHECKPOINTS
    - RUNTIME_WORKING_SLOT_RESERVE
)
ENGINE_CONFIGURATION_CALIBRATION = {
    **ENGINE_CONFIGURATION_128K,
    "max_mamba_cache_size": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
}


def _artifact_directory() -> Path:
    """创建不会覆盖既有产物的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_4workflow_occupancy_calibration_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """优先返回仓库内相对路径。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def prepare_requests(
    tokenizer: object,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """按冻结顺序构造四条 workflow 的前五轮请求。"""
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    recorded_turns = {}
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
        previous_length = None
        for request in requests:
            turn = int(request["turn"])
            input_ids = request["input_ids"]
            if not isinstance(input_ids, list):
                raise TypeError("input_ids 必须是列表")
            if previous_length is not None and len(input_ids) < previous_length:
                raise RuntimeError(f"workflow {label} 的输入长度发生下降")
            previous_length = len(input_ids)
            by_key[(label, turn)] = {
                "workflow_label": label,
                "workflow_id": workflow_id,
                "turn": turn,
                "rid": RID_TEMPLATE.format(label=label.lower(), turn=turn),
                "input_ids": input_ids,
            }
    return [by_key[key] for key in SCHEDULE], recorded_turns


def reconstruct_node_positions(
    structure_rows: Sequence[Sequence[object]],
) -> dict[int, int]:
    """根据父节点关系和分段长度重建每个节点的 token position。"""
    rows = {
        int(row[0]): (
            None if row[1] is None else int(row[1]),
            int(row[3]),
        )
        for row in structure_rows
    }
    positions: dict[int, int] = {}
    visiting: set[int] = set()

    def resolve(node_id: int) -> int:
        if node_id in positions:
            return positions[node_id]
        if node_id in visiting:
            raise RuntimeError("基数树结构存在父节点环")
        visiting.add(node_id)
        parent_id, segment_length = rows[node_id]
        position = segment_length
        if parent_id is not None:
            if parent_id not in rows:
                raise RuntimeError(f"节点 {node_id} 的父节点不存在")
            position += resolve(parent_id)
        visiting.remove(node_id)
        positions[node_id] = position
        return position

    for node_id in rows:
        resolve(node_id)
    return positions


def compact_census(
    response: Mapping[str, object],
    *,
    ordinal: int,
    request: Mapping[str, object] | None,
    previous: Mapping[str, object] | None,
) -> dict[str, object]:
    """把已有 census 响应整理为可审计的状态占用记录。"""
    accounting = response["accounting"]
    tree = response["tree"]
    if not isinstance(accounting, Mapping) or not isinstance(tree, Mapping):
        raise TypeError("census 响应结构异常")
    structure_rows = tree["structure_rows"]
    mamba_rows = tree["mamba_rows"]
    full_rows = tree["full_rows"]
    if not all(isinstance(value, list) for value in (structure_rows, mamba_rows, full_rows)):
        raise TypeError("census tree 字段必须是列表")

    positions = reconstruct_node_positions(structure_rows)
    resident_nodes = [
        {
            "node_id": int(row[0]),
            "slots": [int(slot) for slot in row[1]],
            "token_position": positions.get(int(row[0])),
        }
        for row in mamba_rows
    ]
    resident_map = {
        int(row["node_id"]): tuple(int(slot) for slot in row["slots"])
        for row in resident_nodes
    }
    full_device_ids = {int(row[0]) for row in full_rows if row[1] is not None}
    structure_ids = {int(row[0]) for row in structure_rows}

    previous_mamba = {}
    previous_full: set[int] = set()
    previous_structure: set[int] = set()
    if previous is not None:
        previous_mamba = {
            int(row["node_id"]): tuple(int(slot) for slot in row["slots"])
            for row in previous["resident_mamba_nodes"]
        }
        previous_full = {int(value) for value in previous["full_device_node_ids"]}
        previous_structure = {
            int(value) for value in previous["structure_node_ids"]
        }

    added = sorted(set(resident_map) - set(previous_mamba))
    removed = sorted(set(previous_mamba) - set(resident_map))
    changed = sorted(
        node_id
        for node_id in set(resident_map) & set(previous_mamba)
        if resident_map[node_id] != previous_mamba[node_id]
    )
    removed_full = sorted(previous_full - full_device_ids)
    removed_structure = sorted(previous_structure - structure_ids)
    resident_slot_count = sum(len(slots) for slots in resident_map.values())
    available_slots = int(accounting["mamba_available"])

    return {
        "request_ordinal": ordinal,
        "workflow": None if request is None else request["workflow_label"],
        "workflow_id": None if request is None else request["workflow_id"],
        "turn": None if request is None else int(request["turn"]),
        "rid": None if request is None else request["rid"],
        "input_tokens": None if request is None else len(request["input_ids"]),
        "mamba_total_slots": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
        "mamba_available_slots": available_slots,
        "mamba_schedulable_available_slots": int(
            accounting["mamba_schedulable_available"]
        ),
        "mamba_evictable_slots": int(accounting["mamba_evictable"]),
        "mamba_protected_slots": int(accounting["mamba_protected"]),
        "device_resident_mamba_slots": resident_slot_count,
        "unaccounted_or_working_slots_at_safe_point": (
            PHYSICAL_MAX_MAMBA_CACHE_SIZE - available_slots - resident_slot_count
        ),
        "mamba_node_count": int(tree["mamba_node_count"]),
        "mamba_rows_count": len(mamba_rows),
        "resident_mamba_nodes": resident_nodes,
        "added_mamba_node_ids": added,
        "removed_mamba_node_ids": removed,
        "changed_existing_mamba_node_ids": changed,
        "full_device_node_count": len(full_device_ids),
        "full_device_node_ids": sorted(full_device_ids),
        "full_evictable_tokens": int(accounting["full_evictable"]),
        "full_protected_tokens": int(accounting["full_protected"]),
        "full_allocator": accounting["full_allocator"],
        "tree_node_count": int(tree["node_count"]),
        "structure_node_ids": sorted(structure_ids),
        "removed_full_device_node_ids": removed_full,
        "removed_structure_node_ids": removed_structure,
        "native_mamba_capacity_eviction_inferred": bool(removed or changed),
        "fa_kv_cascade_eviction_inferred": bool(removed_full or removed_structure),
        "raw_accounting": accounting,
        "raw_tree": tree,
    }


def _runtime_metrics(client: object, rid: str, input_tokens: int) -> dict[str, object]:
    """读取已有的 H、E、G 指标并验证基本关系。"""
    try:
        metrics = query_runtime_metrics(client, rid)
        h_value = _optional_int(metrics.get("physical_fa_hit"))
        e_value = _optional_int(metrics.get("executable_prefix"))
        g_value = _optional_int(metrics.get("replay_gap"))
        available = all(value is not None for value in (h_value, e_value, g_value))
        valid = (
            available
            and 0 <= int(e_value) <= int(h_value) <= input_tokens
            and int(g_value) == int(h_value) - int(e_value)
        )
        return {
            "h": h_value,
            "e": e_value,
            "g": g_value,
            "runtime_metrics_available": available,
            "runtime_metrics_valid": valid,
            "runtime_metrics_error": None,
        }
    except Exception as error:
        return {
            "h": None,
            "e": None,
            "g": None,
            "runtime_metrics_available": False,
            "runtime_metrics_valid": None,
            "runtime_metrics_error": repr(error),
        }


def _path_inspection(
    client: object,
    request: Mapping[str, object],
    ordinal: int,
) -> dict[str, object]:
    """使用已验证的精确前缀接口关联 workflow 与运行时节点。"""
    try:
        response = inspect_checkpoint(
            client,
            checkpoint_id=(
                f"OPENHANDS_OCCUPANCY_{request['workflow_label']}_"
                f"TURN_{int(request['turn']):03d}"
            ),
            token_ids=tuple(int(value) for value in request["input_ids"]),
        )
        path = response["after"]["path"]
        return {
            "available": True,
            "request_ordinal": ordinal,
            "node_id": int(path["node_id"]),
            "prefix_tokens": int(path["prefix_tokens"]),
            "path_mamba_positions": path["path_mamba_positions"],
            "target_mamba_present": bool(path["target_mamba_present"]),
            "target_mamba_host_present": bool(path["target_mamba_host_present"]),
            "error": None,
        }
    except Exception as error:
        return {
            "available": False,
            "request_ordinal": ordinal,
            "node_id": None,
            "prefix_tokens": len(request["input_ids"]),
            "path_mamba_positions": [],
            "target_mamba_present": None,
            "target_mamba_host_present": None,
            "error": repr(error),
        }


def execute_request(
    engine: object,
    client: object,
    request: Mapping[str, object],
    ordinal: int,
) -> dict[str, object]:
    """顺序执行一个请求并读取服务端和 Hybrid Runtime 指标。"""
    input_ids = request["input_ids"]
    if not isinstance(input_ids, list):
        raise TypeError("input_ids 必须是列表")
    timing = measure_streaming_request(
        engine,
        request_id=str(request["rid"]),
        token_ids=input_ids,
    )
    metadata = timing["server_metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("server_metadata 必须是对象")
    offline_tokens = len(input_ids)
    server_tokens = _optional_int(metadata.get("prompt_tokens"))
    token_count_exact = server_tokens == offline_tokens
    return {
        "request_ordinal": ordinal,
        "workflow": request["workflow_label"],
        "workflow_id": request["workflow_id"],
        "turn": int(request["turn"]),
        "rid": request["rid"],
        "offline_input_tokens": offline_tokens,
        "server_prompt_tokens": server_tokens,
        "cached_tokens": _optional_int(metadata.get("cached_tokens")),
        "completion_tokens": _optional_int(metadata.get("completion_tokens")),
        "ttft_ms": float(timing["ttft_ms"]),
        "request_latency_ms": float(timing["request_latency_ms"]),
        "finish_reason": metadata.get("finish_reason"),
        "request_completed": True,
        "token_count_exact": token_count_exact,
        "truncation_or_clipping": not token_count_exact,
        "oom": False,
        **_runtime_metrics(client, str(request["rid"]), offline_tokens),
        "status": "PASS" if token_count_exact else "FAIL",
        "error": None,
    }


def _failure_record(
    request: Mapping[str, object], ordinal: int, error: Exception
) -> dict[str, object]:
    """记录首个请求异常，不重试。"""
    message = repr(error)
    lowered = message.lower()
    return {
        "request_ordinal": ordinal,
        "workflow": request["workflow_label"],
        "workflow_id": request["workflow_id"],
        "turn": int(request["turn"]),
        "rid": request["rid"],
        "offline_input_tokens": len(request["input_ids"]),
        "server_prompt_tokens": None,
        "cached_tokens": None,
        "completion_tokens": None,
        "ttft_ms": None,
        "request_latency_ms": None,
        "finish_reason": None,
        "request_completed": False,
        "token_count_exact": False,
        "truncation_or_clipping": False,
        "oom": "out of memory" in lowered or "oom" in lowered,
        "h": None,
        "e": None,
        "g": None,
        "runtime_metrics_available": False,
        "runtime_metrics_valid": None,
        "runtime_metrics_error": None,
        "status": "FAIL",
        "error": message,
    }


def _environment() -> dict[str, object]:
    """采集解释本次校准所需的最小环境信息。"""
    import torch

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "sglang_version": importlib.metadata.version("sglang"),
        "transformers_version": importlib.metadata.version("transformers"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
    }


def build_census_attribution(
    censuses: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """用逐请求新增节点与最终驻留集合建立确定性归属。"""
    if not censuses:
        return {
            "method": "逐请求 census 新增节点与最终驻留集合",
            "checkpoints": {},
            "resident_checkpoint_counts": {
                label: 0 for label in WORKFLOWS
            },
            "all_workflows_have_resident_state": False,
            "all_workflows_have_exclusive_resident_state": False,
        }
    final_resident = {
        int(item["node_id"])
        for item in censuses[-1]["resident_mamba_nodes"]
    }
    checkpoints: dict[str, list[dict[str, object]]] = {
        label: [] for label in WORKFLOWS
    }
    for census in censuses:
        label = census.get("workflow")
        if label not in WORKFLOWS:
            continue
        added_ids = {
            int(value) for value in census["added_mamba_node_ids"]
        }
        for node in census["resident_mamba_nodes"]:
            node_id = int(node["node_id"])
            if node_id in added_ids and node_id in final_resident:
                checkpoints[str(label)].append(
                    {
                        "turn": int(census["turn"]),
                        "request_ordinal": int(census["request_ordinal"]),
                        "node_id": node_id,
                        "token_position": int(node["token_position"]),
                        "slots": [
                            int(value) for value in node["slots"]
                        ],
                    }
                )
    counts = {
        label: len(checkpoints[label]) for label in WORKFLOWS
    }
    all_resident = all(counts[label] > 0 for label in WORKFLOWS)
    return {
        "method": "逐请求 census 新增节点与最终驻留集合",
        "checkpoints": checkpoints,
        "resident_checkpoint_counts": counts,
        "all_workflows_have_resident_state": all_resident,
        "all_workflows_have_exclusive_resident_state": all_resident,
    }

def build_summary(
    *,
    records: Sequence[Mapping[str, object]],
    censuses: Sequence[Mapping[str, object]],
    attribution: Mapping[str, object] | None,
    artifact: Path,
    recorded_turns: Mapping[str, int] | None,
    fatal_error: str | None,
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """构建状态占用、物理余量和安全性门禁。"""
    request_censuses = [item for item in censuses if item["request_ordinal"] > 0]
    all_completed = (
        len(records) == len(SCHEDULE)
        and all(item.get("request_completed") is True for item in records)
    )
    token_correctness = (
        len(records) == len(SCHEDULE)
        and all(item.get("token_count_exact") is True for item in records)
    )
    per_workflow_lengths: dict[str, list[int]] = defaultdict(list)
    for record in records:
        per_workflow_lengths[str(record["workflow"])].append(
            int(record["offline_input_tokens"])
        )
    growing = (
        set(per_workflow_lengths) == set(WORKFLOWS)
        and all(
            len(values) == len(TARGET_TURNS)
            and all(left <= right for left, right in zip(values, values[1:]))
            for values in per_workflow_lengths.values()
        )
    )
    native_eviction = any(
        item["native_mamba_capacity_eviction_inferred"] for item in request_censuses
    )
    fa_cascade = any(
        item["fa_kv_cascade_eviction_inferred"] for item in request_censuses
    )
    peak_resident = max(
        (int(item["device_resident_mamba_slots"]) for item in censuses), default=0
    )
    minimum_available = min(
        (int(item["mamba_available_slots"]) for item in censuses),
        default=PHYSICAL_MAX_MAMBA_CACHE_SIZE,
    )
    peak_nodes = max((int(item["mamba_node_count"]) for item in censuses), default=0)
    metrics_available = (
        len(records) == len(SCHEDULE)
        and all(item.get("runtime_metrics_available") is True for item in records)
    )
    runtime_valid = metrics_available and all(
        item.get("runtime_metrics_valid") is True for item in records
    )
    gap_values = [int(item["g"]) for item in records if item.get("g") is not None]
    fatal_lowered = (fatal_error or "").lower()
    oom = (
        any(item.get("oom") is True for item in records)
        or "out of memory" in fatal_lowered
        or "oom" in fatal_lowered
    )
    clipping = any(item.get("truncation_or_clipping") is True for item in records)
    simultaneous = bool(
        attribution and attribution.get("all_workflows_have_exclusive_resident_state")
    )
    baseline_clean = bool(
        censuses
        and censuses[0]["mamba_node_count"] == 0
        and censuses[0]["device_resident_mamba_slots"] == 0
    )
    physical_headroom = (
        not native_eviction and minimum_available >= RUNTIME_WORKING_SLOT_RESERVE
    )
    passed = (
        fatal_error is None
        and all_completed
        and token_correctness
        and growing
        and baseline_clean
        and simultaneous
        and not native_eviction
        and not fa_cascade
        and not oom
        and not clipping
        and runtime_valid
        and physical_headroom
    )
    return {
        "schema_version": "flowstate.openhands_4workflow_occupancy_calibration.v1",
        "status": "PASS" if passed else "FAIL",
        "schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
        "workflows": WORKFLOWS,
        "recorded_n_turns": dict(recorded_turns or {}),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_CALIBRATION,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "dataset_path": str(DATASET_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "artifact": _display_path(artifact),
        "selection_rationale": {
            "expected_persistent_checkpoints": EXPECTED_PERSISTENT_CHECKPOINTS,
            "runtime_working_slot_reserve": RUNTIME_WORKING_SLOT_RESERVE,
            "explicit_spare_slots": EXPLICIT_SPARE_SLOTS,
            "default_slots": DEFAULT_MAX_MAMBA_CACHE_SIZE,
            "calibration_slots": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
        },
        "physical_mamba_pool_memory": {
            "bytes_per_slot": MAMBA_SLOT_BYTES,
            "mib_per_slot": MAMBA_SLOT_BYTES / (1 << 20),
            "allocated_slots_including_dummy": PHYSICAL_MAX_MAMBA_CACHE_SIZE + 1,
            "pool_bytes": MAMBA_POOL_BYTES,
            "pool_gib": MAMBA_POOL_BYTES / (1 << 30),
            "additional_bytes_vs_24": ADDITIONAL_POOL_BYTES,
            "additional_mib_vs_24": ADDITIONAL_POOL_BYTES / (1 << 20),
        },
        "request_count": len(records),
        "requests": list(records),
        "census_count": len(censuses),
        "all_requests_completed": all_completed,
        "token_correctness": token_correctness,
        "growing_workflows": growing,
        "baseline_census_clean": baseline_clean,
        "peak_resident_mamba_slots": peak_resident,
        "minimum_available_mamba_slots": minimum_available,
        "peak_mamba_node_count": peak_nodes,
        "final_attribution": attribution,
        "multiple_workflows_simultaneously_resident": simultaneous,
        "native_mamba_capacity_eviction_observed": native_eviction,
        "native_mamba_capacity_eviction_count": sum(
            len(item["removed_mamba_node_ids"])
            + len(item["changed_existing_mamba_node_ids"])
            for item in request_censuses
        ),
        "fa_kv_cascade_eviction_observed": fa_cascade,
        "removed_full_device_node_count": sum(
            len(item["removed_full_device_node_ids"]) for item in request_censuses
        ),
        "active_flowstate_eviction_executed": False,
        "policy_executed": False,
        "logical_budget_k": None,
        "concurrency": 1,
        "runtime_metrics_available": metrics_available,
        "runtime_semantic_checks": "PASS" if runtime_valid else "FAIL",
        "h_min": min(
            (int(item["h"]) for item in records if item.get("h") is not None),
            default=None,
        ),
        "h_max": max(
            (int(item["h"]) for item in records if item.get("h") is not None),
            default=None,
        ),
        "e_min": min(
            (int(item["e"]) for item in records if item.get("e") is not None),
            default=None,
        ),
        "e_max": max(
            (int(item["e"]) for item in records if item.get("e") is not None),
            default=None,
        ),
        "g_min": min(gap_values, default=None),
        "g_max": max(gap_values, default=None),
        "all_gaps_zero": len(gap_values) == len(SCHEDULE) and max(gap_values) == 0,
        "oom": oom,
        "truncation_or_clipping": clipping,
        "physical_headroom": "SUFFICIENT" if physical_headroom else "INSUFFICIENT",
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
        "native_eviction_log_count": None,
        "native_eviction_log_evidence": [],
    }


def _scan_logs(artifact: Path, summary: dict[str, object]) -> None:
    """只提取日志中明确标记的原生状态驱逐证据。"""
    evidence = []
    for filename in ("stdout.log", "stderr.log"):
        path = artifact / filename
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            lowered = line.lower()
            if (
                "mamba" in lowered
                and "evict" in lowered
                and "radix_eviction_policy" not in lowered
                and "mamba_max_states_per_path" not in lowered
            ):
                evidence.append(
                    {"file": filename, "line": line_number, "text": line[:500]}
                )
    summary["native_eviction_log_count"] = len(evidence)
    summary["native_eviction_log_evidence"] = evidence[:20]


def _run(artifact: Path) -> dict[str, object]:
    """在一个 Engine 中执行唯一一次二十请求校准。"""
    requests_path = artifact / "requests.jsonl"
    census_path = artifact / "census.jsonl"
    requests_path.write_text("", encoding="utf-8")
    census_path.write_text("", encoding="utf-8")
    records: list[dict[str, object]] = []
    censuses: list[dict[str, object]] = []
    latest_requests: dict[str, Mapping[str, object]] = {}
    attribution = None
    fatal_error = None
    engine = None
    recorded_turns = None
    environment = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        requests, recorded_turns = prepare_requests(tokenizer)
        _write_json(
            artifact / "config.json",
            {
                "workflows": WORKFLOWS,
                "schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
                "recorded_n_turns": recorded_turns,
                "offline_input_tokens": [
                    {
                        "workflow": request["workflow_label"],
                        "turn": request["turn"],
                        "tokens": len(request["input_ids"]),
                    }
                    for request in requests
                ],
                "engine_configuration": ENGINE_CONFIGURATION_CALIBRATION,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "physical_mamba_pool_memory": {
                    "bytes_per_slot": MAMBA_SLOT_BYTES,
                    "pool_bytes": MAMBA_POOL_BYTES,
                    "additional_bytes_vs_24": ADDITIONAL_POOL_BYTES,
                },
                "selection_rationale": {
                    "expected_persistent_checkpoints": EXPECTED_PERSISTENT_CHECKPOINTS,
                    "runtime_working_slot_reserve": RUNTIME_WORKING_SLOT_RESERVE,
                    "explicit_spare_slots": EXPLICIT_SPARE_SLOTS,
                },
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "environment": environment,
                "policy_executed": False,
                "active_eviction_executed": False,
                "logical_budget_k": None,
                "concurrency": 1,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_CALIBRATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)

        baseline = compact_census(
            client.census("openhands_occupancy:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        censuses.append(baseline)
        _append_jsonl(census_path, baseline)
        if baseline["mamba_node_count"] != 0:
            raise RuntimeError("Engine 初始 census 含有非空 Mamba checkpoint")

        previous = baseline
        for ordinal, request in enumerate(requests, start=1):
            try:
                record = execute_request(engine, client, request, ordinal)
                record["path_inspection"] = _path_inspection(client, request, ordinal)
                census = compact_census(
                    client.census(f"openhands_occupancy:after:{ordinal:03d}"),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                records.append(record)
                _append_jsonl(requests_path, record)
                fatal_error = repr(error)
                break

            records.append(record)
            censuses.append(census)
            latest_requests[str(request["workflow_label"])] = request
            _append_jsonl(requests_path, record)
            _append_jsonl(census_path, census)
            previous = census

            if record["status"] != "PASS":
                fatal_error = "服务端 prompt_tokens 与离线长度不一致"
                break
            if census["native_mamba_capacity_eviction_inferred"]:
                fatal_error = "PHYSICAL_HEADROOM_INSUFFICIENT"
                break
            if census["fa_kv_cascade_eviction_inferred"]:
                fatal_error = "检测到 FA-KV 节点删除或级联驱逐"
                break

        if len(records) == len(SCHEDULE) and fatal_error is None:
            attribution = build_census_attribution(censuses)
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
        attribution=attribution,
        artifact=artifact,
        recorded_turns=recorded_turns,
        fatal_error=fatal_error,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次占用校准。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _scan_logs(artifact, summary)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
