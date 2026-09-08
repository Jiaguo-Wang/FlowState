#!/usr/bin/env python3
"""执行 RQ4 冻结快照到真实 TTFT 的统一运行时门禁。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
import sys
import time
import traceback
from typing import Mapping, Sequence

from evaluation.openhands_policy_to_actuator_mapping_gate import (
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
    inspect_candidate_states,
)
from evaluation.openhands_sequential_eviction_rematerialization_audit import (
    SequentialTraceRuntimeAdapter,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    _template_input_ids,
    normalize_message,
)
from evaluation.rq3_formal_policy_evaluation import (
    _load_allocation_snapshot,
    create_budget_variant,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    _select_policy,
    evaluate_objective,
)
from evaluation.rq3_openhands_neutral_collector import (
    GroupReplayTrace,
    SGLangGroupRuntime,
    _boundary_audit_has_leakage,
    assemble_group_snapshot,
    materialize_group_requests,
    query_gpu_compute_processes,
    query_gpu_memory_used_mib,
    replay_group_to_barrier,
    wait_gpu_stable,
)
from evaluation.rq3_openhands_population import (
    WorkflowGroup,
    load_session_messages,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_ROOT = (
    REPOSITORY_ROOT
    / "evaluation"
    / "runtime_artifacts"
    / "rq3_openhands_main_formal_20260904_001017"
)
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
SMOKE_SNAPSHOT_IDS = (
    "rq3-openhands-main-g016-round2",
    "rq3-openhands-main-g119-round5",
)
FORMAL_SNAPSHOT_IDS = (
    "rq3-openhands-main-g016-round2",
    "rq3-openhands-main-g156-round2",
    "rq3-openhands-main-g112-round2",
    "rq3-openhands-main-g176-round2",
    "rq3-openhands-main-g172-round2",
    "rq3-openhands-main-g184-round2",
    "rq3-openhands-main-g037-round3",
    "rq3-openhands-main-g157-round3",
    "rq3-openhands-main-g177-round3",
    "rq3-openhands-main-g165-round3",
    "rq3-openhands-main-g069-round3",
    "rq3-openhands-main-g005-round3",
    "rq3-openhands-main-g122-round4",
    "rq3-openhands-main-g050-round4",
    "rq3-openhands-main-g058-round4",
    "rq3-openhands-main-g078-round4",
    "rq3-openhands-main-g070-round4",
    "rq3-openhands-main-g094-round4",
    "rq3-openhands-main-g171-round5",
    "rq3-openhands-main-g199-round5",
    "rq3-openhands-main-g167-round5",
    "rq3-openhands-main-g043-round5",
    "rq3-openhands-main-g015-round5",
    "rq3-openhands-main-g119-round5",
)
POLICIES = ("LRU", "Marconi", "FlowState")
SAMPLING_SEED = 20260907
WORKER_TIMEOUT_S = 1800.0
FORMAL_REPETITIONS = 3
FORMAL_RUN_COUNT = 216


class RQ4CorrectnessError(RuntimeError):
    """表示当前 run 不满足 fail-closed correctness gate。"""


def _write_json(path: Path, value: object) -> None:
    """以稳定格式原子写出一个 JSON 文件。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    """追加一行 JSON 并立即刷新。"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()


def rq4_logical_k(candidate_count: int) -> int:
    """按冻结的 25% 预算规则计算逻辑 K。"""
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise TypeError("candidate_count 必须是整数")
    if candidate_count <= 0:
        raise ValueError("candidate_count 必须大于零")
    return max(1, math.floor(0.25 * candidate_count))


def select_frozen_policy(
    snapshot: AllocationSnapshot,
    policy: str,
) -> tuple[str, ...]:
    """调用冻结 selector，并验证选择过程确定且不修改快照。"""
    if policy not in POLICIES:
        raise ValueError(f"RQ4 不支持 policy：{policy}")
    initial_digest = snapshot.content_digest()
    selected, _ = _select_policy(policy, snapshot, evaluate_objective)
    repeated, _ = _select_policy(policy, snapshot, evaluate_objective)
    if tuple(selected) != tuple(repeated):
        raise RQ4CorrectnessError(f"{policy} selector 非确定")
    if snapshot.content_digest() != initial_digest:
        raise RQ4CorrectnessError(f"{policy} selector 修改了冻结快照")
    if len(selected) > snapshot.logical_budget_k:
        raise RQ4CorrectnessError(
            f"{policy} selected 数超过 K：{len(selected)} > "
            f"{snapshot.logical_budget_k}"
        )
    candidate_ids = {
        item.checkpoint_id for item in snapshot.eligible_candidates
    }
    if len(set(selected)) != len(selected) or not set(selected).issubset(
        candidate_ids
    ):
        raise RQ4CorrectnessError(f"{policy} selected candidate IDs 非法")
    return tuple(selected)


def _snapshot_universe(snapshot: AllocationSnapshot) -> dict[str, object]:
    """提取与当前 Engine handle 无关的冻结逻辑 universe。"""
    return {
        "allocation_epoch": snapshot.allocation_epoch,
        "snapshot_id": snapshot.snapshot_id,
        "pending_continuations": [
            asdict(item) for item in snapshot.pending_continuations
        ],
        "eligible_candidates": [
            asdict(item) for item in snapshot.eligible_candidates
        ],
        "candidate_metadata": [
            asdict(item) for item in snapshot.candidate_metadata
        ],
        "lfu_access_frequency": [
            asdict(item) for item in snapshot.lfu_access_frequency
        ],
        "frequency_observed_through_epoch": (
            snapshot.frequency_observed_through_epoch
        ),
        "marconi_alpha": snapshot.marconi_alpha,
        "recovery_model": asdict(snapshot.recovery_model),
        "online_boundary": asdict(snapshot.online_boundary),
    }


def validate_rebuilt_snapshot(
    frozen: AllocationSnapshot,
    rebuilt: AllocationSnapshot,
) -> dict[str, object]:
    """验证 replay 重建结果与 frozen logical snapshot 完全一致。"""
    frozen_universe = _snapshot_universe(frozen)
    rebuilt_universe = _snapshot_universe(rebuilt)
    logical_exact = rebuilt_universe == frozen_universe
    full_digest_exact = rebuilt.content_digest() == frozen.content_digest()
    result = {
        "logical_universe_exact": logical_exact,
        "full_snapshot_digest_exact": full_digest_exact,
        "frozen_snapshot_digest": frozen.content_digest(),
        "rebuilt_snapshot_digest": rebuilt.content_digest(),
        "candidate_universe_exact": (
            rebuilt_universe["eligible_candidates"]
            == frozen_universe["eligible_candidates"]
        ),
        "pending_universe_exact": (
            rebuilt_universe["pending_continuations"]
            == frozen_universe["pending_continuations"]
        ),
    }
    if not logical_exact:
        raise RQ4CorrectnessError("replay 与 frozen logical universe 不一致")
    if not full_digest_exact:
        raise RQ4CorrectnessError("replay 与 frozen snapshot digest 不一致")
    return result


def build_current_runtime_handles(
    trace: GroupReplayTrace,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, RuntimeCheckpointHandle]:
    """只用当前 Engine replay 事实为每个 candidate 构造唯一句柄。"""
    handles: dict[str, RuntimeCheckpointHandle] = {}
    identities: set[tuple[int, str]] = set()
    for observation in trace.checkpoints:
        if observation.checkpoint_id in handles:
            raise RQ4CorrectnessError(
                f"candidate handle 重复：{observation.checkpoint_id}"
            )
        request = requests[(observation.workflow_label, observation.turn)]
        raw_ids = request.get("input_ids")
        if not isinstance(raw_ids, list):
            raise RQ4CorrectnessError("replay request 缺少 input_ids")
        token_ids = tuple(
            int(value) for value in raw_ids[: observation.token_pos]
        )
        from evaluation.openhands_common_barrier_snapshot_gate import (
            token_digest,
        )

        prefix_digest = token_digest(token_ids)
        if prefix_digest != observation.prefix_digest:
            raise RQ4CorrectnessError(
                f"当前 Engine prefix digest 不一致：{observation.checkpoint_id}"
            )
        identity = (observation.node_id, prefix_digest)
        if identity in identities:
            raise RQ4CorrectnessError("多个 candidate 映射到同一 runtime identity")
        identities.add(identity)
        handles[observation.checkpoint_id] = RuntimeCheckpointHandle(
            checkpoint_id=observation.checkpoint_id,
            token_ids=token_ids,
            expected_node_id=observation.node_id,
            expected_prefix_digest=prefix_digest,
        )
    expected_ids = {item.checkpoint_id for item in trace.checkpoints}
    if set(handles) != expected_ids:
        raise RQ4CorrectnessError("runtime handle mapping 不完整")
    return handles


def trace_has_rematerialization(
    trace_rows: Sequence[Mapping[str, object]],
    evicted_ids: Sequence[str],
) -> bool:
    """检查 reconcile 内目标缺失后是否意外重新设备驻留。"""
    absent_seen = {checkpoint_id: False for checkpoint_id in evicted_ids}
    for row in trace_rows:
        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            raise RQ4CorrectnessError("reconcile trace 缺少 checkpoints")
        for checkpoint_id in evicted_ids:
            state = checkpoints.get(checkpoint_id)
            if not isinstance(state, Mapping):
                raise RQ4CorrectnessError(
                    f"reconcile trace 缺少 {checkpoint_id}"
                )
            present = bool(state["recurrent_present"])
            if absent_seen[checkpoint_id] and present:
                return True
            if not present:
                absent_seen[checkpoint_id] = True
    return False


def _group_from_manifest(
    manifest: Mapping[str, object],
    group_ordinal: int,
) -> WorkflowGroup:
    """从 frozen population manifest 重建指定 workflow group。"""
    rows = [
        row
        for row in manifest["groups"]
        if int(row["group_ordinal"]) == group_ordinal
    ]
    if len(rows) != 1:
        raise RQ4CorrectnessError(
            f"group {group_ordinal} 在 manifest 中不是唯一一行"
        )
    row = rows[0]
    return WorkflowGroup(
        group_ordinal=int(row["group_ordinal"]),
        population_segment=str(row["population_segment"]),
        allocation_round=int(row["allocation_round"]),
        session_ids=tuple(str(value) for value in row["session_ids"]),
    )


def _snapshot_path_for_group(
    formal_root: Path,
    group_ordinal: int,
) -> Path:
    """定位唯一的 ELIGIBLE canonical snapshot artifact。"""
    summary = json.loads(
        (formal_root / "collection_summary.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in summary["verdicts"]
        if int(row["group_ordinal"]) == group_ordinal
        and row["status"] == "ELIGIBLE"
    ]
    if len(rows) != 1:
        raise RQ4CorrectnessError(
            f"group {group_ordinal} 不是唯一 ELIGIBLE snapshot"
        )
    path = Path(rows[0]["snapshot_artifact"])
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path


def _snapshot_group_ordinal(snapshot_id: str) -> int:
    """从冻结 snapshot ID 解析 group ordinal。"""
    marker = "-g"
    if marker not in snapshot_id or "-round" not in snapshot_id:
        raise ValueError(f"snapshot ID 格式非法：{snapshot_id}")
    return int(snapshot_id.split(marker, 1)[1].split("-round", 1)[0])


def _future_leakage(audits: Sequence[Mapping[str, object]]) -> bool:
    """统一判断 pending 物化是否越过在线信息边界。"""
    return _boundary_audit_has_leakage(audits)


def _executable_checkpoint(
    snapshot: AllocationSnapshot,
    workflow_id: str,
    executable_frontier: int,
    actual_residency: Mapping[str, bool],
) -> str | None:
    """返回请求开始时能解释 E 的最深设备驻留 candidate。"""
    matches = [
        item
        for item in snapshot.eligible_candidates
        if item.workflow_id == workflow_id
        and item.token_pos <= executable_frontier
        and actual_residency.get(item.checkpoint_id) is True
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item.token_pos).checkpoint_id


def _validate_pending_record(record: Mapping[str, object]) -> None:
    """验证一个 resumed pending request 的全部硬门禁。"""
    required = (record.get("h"), record.get("e"), record.get("g"))
    if record.get("status") != "PASS" or not record.get("request_completed"):
        raise RQ4CorrectnessError("pending request 未完成")
    if record.get("runtime_metrics_valid") is not True:
        raise RQ4CorrectnessError("pending request 的 H/E/G 无效")
    if any(value is None for value in required):
        raise RQ4CorrectnessError("pending request 缺少 H/E/G")
    if int(record["g"]) != int(record["h"]) - int(record["e"]):
        raise RQ4CorrectnessError("pending request 违反 G=H-E")
    if record.get("ttft_ms") is None:
        raise RQ4CorrectnessError("pending request 缺少 TTFT")
    if record.get("oom") or record.get("truncation_or_clipping"):
        raise RQ4CorrectnessError("pending request 出现 OOM 或 truncation")
    if record.get("expected_actual_residency_exact") is not True:
        raise RQ4CorrectnessError("pending 前 expected/actual residency 不一致")


def _run_worker(
    *,
    artifact_root: Path,
    formal_root: Path,
    run_id: str,
) -> int:
    """在一个独立进程和 Engine lifecycle 中执行单个 RQ4 run。"""
    from transformers import AutoTokenizer

    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        wait_for_transport,
    )
    from evaluation.openhands_4workflow_occupancy_calibration import (
        execute_request,
    )
    from evaluation.controlled_multiworkflow_v1.scenario import (
        CHECKPOINT_SIZE_BYTES,
    )

    plan = json.loads(
        (artifact_root / "manifest.json").read_text(encoding="utf-8")
    )
    plan_rows = [row for row in plan["runs"] if row["run_id"] == run_id]
    if len(plan_rows) != 1:
        raise SystemExit(f"run plan 不唯一：{run_id}")
    run_plan = plan_rows[0]
    run_directory = artifact_root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    pending_path = run_directory / "pending_requests.jsonl"
    pending_path.touch(exist_ok=False)

    started = time.perf_counter()
    phase = "prepare"
    engine = None
    runtime = None
    shutdown_error = None
    fatal_error = None
    replay_validation = None
    mapping_validation = None
    selection_record = None
    reconcile_record = None
    pending_records: list[dict[str, object]] = []
    timings: dict[str, float | None] = {
        "engine_init_s": None,
        "replay_s": None,
        "reconcile_s": None,
        "pending_resume_s": None,
        "total_run_s": None,
    }
    boundary_audits: list[dict[str, object]] = []
    fresh_engine_empty = False
    try:
        manifest = json.loads(
            (formal_root / "population_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        group = _group_from_manifest(
            manifest, int(run_plan["group_ordinal"])
        )
        snapshot_path = _snapshot_path_for_group(
            formal_root, group.group_ordinal
        )
        frozen = _load_allocation_snapshot(snapshot_path)
        if frozen.snapshot_id != run_plan["snapshot_id"]:
            raise RQ4CorrectnessError("run plan snapshot ID 不一致")
        if frozen.content_digest() != run_plan["snapshot_digest"]:
            raise RQ4CorrectnessError("run plan snapshot digest 不一致")
        k = rq4_logical_k(len(frozen.eligible_candidates))
        if k != int(run_plan["logical_k"]):
            raise RQ4CorrectnessError("run plan K 不一致")
        snapshot_k = create_budget_variant(frozen, k)
        selected_ids = select_frozen_policy(snapshot_k, run_plan["policy"])
        if list(selected_ids) != run_plan["selected_candidate_ids"]:
            raise RQ4CorrectnessError("worker selector 与冻结 selected IDs 不一致")
        selection_record = {
            "policy": run_plan["policy"],
            "candidate_count": len(frozen.eligible_candidates),
            "logical_k": k,
            "budget_bytes": snapshot_k.budget_bytes,
            "selected_candidate_ids": list(selected_ids),
            "selected_candidate_ids_exact": True,
            "snapshot_digest_before": frozen.content_digest(),
            "snapshot_digest_after": frozen.content_digest(),
        }
        _write_json(run_directory / "selection.json", selection_record)

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        messages_by_label = {
            label: load_session_messages(session_id, DATASET_PATH)
            for label, session_id in group.session_by_label.items()
        }
        requests, audits = materialize_group_requests(
            tokenizer,
            messages_by_label,
            group=group,
            normalize_message=normalize_message,
            template_input_ids=_template_input_ids,
        )
        boundary_audits = [dict(item) for item in audits]
        if _future_leakage(boundary_audits):
            raise RQ4CorrectnessError("request materialization 出现 future leakage")

        phase = "engine_init"
        engine_started = time.perf_counter()
        from targeted_probe import ControlClient
        from sequential_eviction_trace_transport import (
            SequentialEvictionTraceGateEngine,
            requested_control_port,
        )

        engine = SequentialEvictionTraceGateEngine(
            **dict(manifest["engine_configuration"])
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        timings["engine_init_s"] = time.perf_counter() - engine_started
        runtime = SGLangGroupRuntime(engine, client)

        baseline = runtime.census(
            f"rq4:{run_id}:baseline",
            ordinal=0,
            request=None,
            previous=None,
        )
        fresh_engine_empty = int(baseline["mamba_node_count"]) == 0
        if not fresh_engine_empty:
            raise RQ4CorrectnessError("fresh Engine 初始含 recurrent state")

        phase = "replay"
        replay_started = time.perf_counter()
        trace = replay_group_to_barrier(runtime, group, requests)
        trace = replace(trace, boundary_audit=tuple(audits))
        assembly = assemble_group_snapshot(
            trace,
            checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        )
        if assembly.status != "ELIGIBLE" or assembly.snapshot is None:
            raise RQ4CorrectnessError(
                f"runtime replay 重建失败：{assembly.primary_reason}"
            )
        replay_validation = validate_rebuilt_snapshot(
            frozen, assembly.snapshot
        )
        handles = build_current_runtime_handles(trace, requests)
        candidates = snapshot_k.core_candidates()
        candidate_ids = tuple(item.checkpoint_id for item in candidates)
        if set(handles) != set(candidate_ids):
            raise RQ4CorrectnessError("handle mapping 与 frozen candidates 不一致")
        before_states, before_inspections = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"RQ4_{run_id}_MAPPING",
        )
        if not all(
            state["recurrent_resident"] and state["fa_resident"]
            for state in before_states.values()
        ):
            raise RQ4CorrectnessError("replay barrier candidate 未全部驻留")
        mapping_validation = {
            "status": "PASS",
            "candidate_count": len(candidate_ids),
            "handle_count": len(handles),
            "complete": set(handles) == set(candidate_ids),
            "unique_runtime_identities": len(
                {
                    (
                        handle.expected_node_id,
                        handle.expected_prefix_digest,
                    )
                    for handle in handles.values()
                }
            )
            == len(handles),
            "current_engine_handles_only": True,
            "inspection_count": len(before_inspections),
        }
        timings["replay_s"] = time.perf_counter() - replay_started
        _write_json(
            run_directory / "replay.json",
            {
                "validation": replay_validation,
                "boundary_audit": boundary_audits,
                "request_rows": [dict(item) for item in trace.request_rows],
                "census_rows": [dict(item) for item in trace.census_rows],
            },
        )
        _write_json(run_directory / "handle_mapping.json", mapping_validation)

        phase = "reconcile"
        reconcile_started = time.perf_counter()
        previous_census = dict(trace.census_rows[-1])
        before_census = runtime.census(
            f"rq4:{run_id}:before-reconcile",
            ordinal=4 * group.allocation_round,
            request=None,
            previous=previous_census,
        )
        before_states, _ = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"RQ4_{run_id}_BEFORE",
        )
        expected_evicted_ids = tuple(
            sorted(set(candidate_ids) - set(selected_ids))
        )
        trace_adapter = SequentialTraceRuntimeAdapter(
            client,
            handles,
            nonce_namespace=f"rq4:{run_id}",
        )
        recording_adapter = RecordingRuntimeAdapter(trace_adapter)
        controller = StateController(
            FrozenSelectedSetOptimizer(selected_ids),
            recording_adapter,
        )
        allocation = None
        try:
            allocation = controller.reconcile(
                snapshot_k.core_continuations(),
                candidates,
                handles,
                snapshot_k.budget_bytes,
            )
        finally:
            trace_adapter.finish()
        controller_report = build_controller_report(
            allocation=allocation,
            adapter=recording_adapter,
        )
        after_census = runtime.census(
            f"rq4:{run_id}:after-reconcile",
            ordinal=4 * group.allocation_round,
            request=None,
            previous=before_census,
        )
        after_states, _ = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"RQ4_{run_id}_AFTER",
        )
        invariants = evaluate_mapping_invariants(
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            expected_evicted_ids=expected_evicted_ids,
            handles=handles,
            before_states=before_states,
            after_states=after_states,
            before_census=before_census,
            after_census=after_census,
            controller_report=controller_report,
        )
        unexpected_rematerialization = trace_has_rematerialization(
            trace_adapter.trace_rows,
            expected_evicted_ids,
        )
        reconcile_record = {
            "status": (
                "PASS"
                if invariants["status"] == "PASS"
                and not unexpected_rematerialization
                else "FAIL"
            ),
            "controller_report": controller_report,
            "invariants": invariants,
            "unexpected_rematerialization": unexpected_rematerialization,
            "trace_rows": trace_adapter.trace_rows,
        }
        if reconcile_record["status"] != "PASS":
            raise RQ4CorrectnessError("recurrent reconcile correctness 失败")
        timings["reconcile_s"] = time.perf_counter() - reconcile_started
        _write_json(run_directory / "reconcile.json", reconcile_record)

        phase = "pending_resume"
        resume_started = time.perf_counter()
        previous = after_census
        pending_turn = group.allocation_round + 1
        expected_residency = {
            checkpoint_id: checkpoint_id in set(selected_ids)
            for checkpoint_id in candidate_ids
        }
        for offset, label in enumerate(group.session_by_label, start=1):
            request = requests[(label, pending_turn)]
            workflow_id = str(request["workflow_id"])
            workflow_candidates = tuple(
                item for item in candidates if item.workflow_id == workflow_id
            )
            pre_states, _ = inspect_candidate_states(
                client,
                workflow_candidates,
                handles,
                phase=f"RQ4_{run_id}_{label}_PENDING",
            )
            actual_residency = {
                checkpoint_id: bool(state["recurrent_resident"])
                for checkpoint_id, state in pre_states.items()
            }
            relevant_expected = {
                item.checkpoint_id: expected_residency[item.checkpoint_id]
                for item in workflow_candidates
            }
            residency_exact = actual_residency == relevant_expected
            if not residency_exact:
                raise RQ4CorrectnessError(
                    f"{label}{pending_turn} 前 residency 与 reconcile 结果不一致"
                )
            ordinal = 4 * group.allocation_round + offset
            record = execute_request(engine, client, request, ordinal)
            census = runtime.census(
                f"rq4:{run_id}:after:{label}{pending_turn}",
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            enriched = {
                **dict(record),
                "snapshot_id": run_plan["snapshot_id"],
                "policy": run_plan["policy"],
                "repetition": run_plan["repetition"],
                "request_id": record.get("rid"),
                "input_token_digest": next(
                    item.input_token_digest
                    for item in trace.pendings
                    if item.workflow_label == label
                ),
                "selected_checkpoint_ids_for_request": [
                    item.checkpoint_id
                    for item in workflow_candidates
                    if item.checkpoint_id in set(selected_ids)
                ],
                "expected_recurrent_residency": relevant_expected,
                "actual_recurrent_residency": actual_residency,
                "expected_actual_residency_exact": residency_exact,
                "executable_checkpoint_id": _executable_checkpoint(
                    snapshot_k,
                    workflow_id,
                    int(record.get("e") or 0),
                    actual_residency,
                ),
                "recovery_indicator": bool(int(record.get("g") or 0) > 0),
                "recovery_latency_ms": None,
                "recovery_latency_telemetry": "PARTIAL",
                "native_mamba_capacity_eviction": bool(
                    census["native_mamba_capacity_eviction_inferred"]
                ),
                "fa_kv_cascade": bool(
                    census["fa_kv_cascade_eviction_inferred"]
                ),
            }
            _validate_pending_record(enriched)
            if enriched["native_mamba_capacity_eviction"]:
                raise RQ4CorrectnessError("pending resume 发生原生 recurrent eviction")
            if enriched["fa_kv_cascade"]:
                raise RQ4CorrectnessError("pending resume 发生 FA cascade")
            pending_records.append(enriched)
            _append_jsonl(pending_path, enriched)
            previous = census
        timings["pending_resume_s"] = time.perf_counter() - resume_started
        if len(pending_records) != 4:
            raise RQ4CorrectnessError("pending request 完成数不是 4")
    except Exception as error:
        fatal_error = {
            "phase": phase,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception as error:
                shutdown_error = repr(error)
        elif engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)
        timings["total_run_s"] = time.perf_counter() - started

    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and fresh_engine_empty
        and replay_validation is not None
        and replay_validation["logical_universe_exact"]
        and mapping_validation is not None
        and mapping_validation["complete"]
        and reconcile_record is not None
        and reconcile_record["status"] == "PASS"
        and len(pending_records) == 4
        and all(item["runtime_metrics_valid"] for item in pending_records)
        and not _future_leakage(boundary_audits)
    )
    result = {
        "schema_version": "flowstate.rq4_runtime_run.v1",
        "run_id": run_id,
        "snapshot_id": run_plan["snapshot_id"],
        "snapshot_digest": run_plan["snapshot_digest"],
        "group_ordinal": run_plan["group_ordinal"],
        "allocation_round": run_plan["allocation_round"],
        "policy": run_plan["policy"],
        "repetition": run_plan["repetition"],
        "candidate_count": run_plan["candidate_count"],
        "logical_k": run_plan["logical_k"],
        "selected_candidate_ids": run_plan["selected_candidate_ids"],
        "engine_lifecycle": "independent_fresh_process",
        "fresh_engine_empty": fresh_engine_empty,
        "status": "PASS" if passed else "INVALID",
        "replay_validation": replay_validation,
        "handle_mapping": mapping_validation,
        "selection": selection_record,
        "reconcile": reconcile_record,
        "pending_requests": pending_records,
        "pending_completed": len(pending_records),
        "future_leakage": _future_leakage(boundary_audits),
        "recovery_latency_telemetry": "PARTIAL",
        "timings": timings,
        "fatal_error": fatal_error,
        "shutdown_error": shutdown_error,
    }
    _write_json(run_directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


def _policy_order(snapshot_id: str, repetition: int) -> tuple[str, ...]:
    """按冻结 seed 对三策略做 Latin-square 顺序平衡。"""
    digest = sha256(
        f"rq4-policy-order|{SAMPLING_SEED}|{snapshot_id}".encode("utf-8")
    ).digest()
    offset = (digest[0] + repetition - 1) % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def build_run_plan(
    formal_root: Path,
    snapshot_ids: Sequence[str],
    repetitions: int,
) -> list[dict[str, object]]:
    """冻结每个 isolated run 的 snapshot、K 和 selected IDs。"""
    if repetitions <= 0:
        raise ValueError("repetitions 必须大于零")
    runs = []
    for repetition in range(1, repetitions + 1):
        for snapshot_id in snapshot_ids:
            group_ordinal = _snapshot_group_ordinal(snapshot_id)
            snapshot_path = _snapshot_path_for_group(
                formal_root, group_ordinal
            )
            snapshot = _load_allocation_snapshot(snapshot_path)
            if snapshot.snapshot_id != snapshot_id:
                raise RQ4CorrectnessError("请求的 snapshot ID 与 artifact 不一致")
            candidate_count = len(snapshot.eligible_candidates)
            k = rq4_logical_k(candidate_count)
            snapshot_k = create_budget_variant(snapshot, k)
            for policy in _policy_order(snapshot_id, repetition):
                selected = select_frozen_policy(snapshot_k, policy)
                run_id = (
                    f"g{group_ordinal:03d}_round{snapshot.allocation_epoch}_"
                    f"{policy.lower()}_rep{repetition:02d}"
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "snapshot_id": snapshot_id,
                        "snapshot_digest": snapshot.content_digest(),
                        "snapshot_artifact": str(snapshot_path),
                        "group_ordinal": group_ordinal,
                        "allocation_round": snapshot.allocation_epoch,
                        "policy": policy,
                        "repetition": repetition,
                        "candidate_count": candidate_count,
                        "logical_k": k,
                        "budget_bytes": snapshot_k.budget_bytes,
                        "selected_candidate_ids": list(selected),
                    }
                )
    return runs


def _source_digest(path: Path) -> str:
    """返回一个源文件的 SHA-256。"""
    return sha256(path.read_bytes()).hexdigest()


def _frozen_source_paths() -> tuple[Path, ...]:
    """返回正式运行前后必须保持不变的实现文件。"""
    return (
        REPOSITORY_ROOT / "evaluation" / "rq3_frozen_snapshot_evaluator.py",
        REPOSITORY_ROOT / "evaluation" / "rq3_formal_policy_evaluation.py",
        REPOSITORY_ROOT / "evaluation" / "rq3_openhands_neutral_collector.py",
        REPOSITORY_ROOT
        / "evaluation"
        / "openhands_policy_to_actuator_mapping_gate.py",
        REPOSITORY_ROOT / "flowstate" / "optimizer.py",
        REPOSITORY_ROOT / "flowstate" / "recovery_model.py",
        Path(__file__).resolve(),
    )


def _source_integrity(
    paths: Sequence[Path],
    before: Mapping[str, str],
) -> dict[str, object]:
    """比较 collection 前后的冻结源码摘要。"""
    rows = []
    for path in paths:
        key = str(path)
        after = _source_digest(path)
        rows.append(
            {
                "path": key,
                "before_sha256": before[key],
                "after_sha256": after,
                "unchanged": before[key] == after,
            }
        )
    passed = all(row["unchanged"] for row in rows)
    return {
        "schema_version": "flowstate.rq4_source_integrity.v1",
        "status": "PASS" if passed else "FAIL",
        "sources": rows,
    }


def _frozen_protocol() -> dict[str, object]:
    """返回 RQ4-A 已冻结且从历史记录恢复的正式协议。"""
    return {
        "schema_version": "flowstate.rq4_formal_protocol.v1",
        "status": "FROZEN",
        "rq4_a_verdict": "RQ4_PROTOCOL_READY",
        "source_population": "rq3_openhands_main_formal_20260904_001017",
        "source_population_size": 168,
        "selection_rule": (
            "按 round 固定配额；round 内按四个 pending 的 anchor_pos 总和排序并"
            "划分 6 个等频 workload-size strata；每层选择 "
            "SHA256('flowstate-rq4-a|seed|round|stratum|snapshot_id|"
            "snapshot_digest') 最小者"
        ),
        "sampling_seed": SAMPLING_SEED,
        "policy_blind": True,
        "snapshot_ids": list(FORMAL_SNAPSHOT_IDS),
        "round_distribution": {"2": 6, "3": 6, "4": 6, "5": 6},
        "policies": list(POLICIES),
        "budget_ratio": 0.25,
        "budget_rule": "K = max(1, floor(0.25 * |C|))，且 |S| <= K",
        "candidate_count_to_k": {"8": 2, "12": 3, "16": 4, "20": 5},
        "repetitions": FORMAL_REPETITIONS,
        "planned_isolated_runs": FORMAL_RUN_COUNT,
        "run_order": (
            "repetition → RQ4-A frozen snapshot list → seeded Latin-square policy order"
        ),
        "primary_metric": "TTFT",
        "statistical_unit": "snapshot",
        "recovery_latency_telemetry": "PARTIAL",
        "infrastructure_retry_limit": 0,
        "future_trajectory_used": False,
        "b2_raw_token_lcp_compatibility_used": False,
    }


def _run_correctness(result: Mapping[str, object]) -> dict[str, object]:
    """将一个 run 的全部 correctness gate 固化为独立记录。"""

    def section(value: object) -> Mapping[str, object]:
        return value if isinstance(value, Mapping) else {}

    replay = section(result.get("replay_validation"))
    mapping = section(result.get("handle_mapping"))
    selection = section(result.get("selection"))
    reconcile = section(result.get("reconcile"))
    invariants = section(reconcile.get("invariants"))
    pending = [
        item
        for item in result.get("pending_requests", ())
        if isinstance(item, Mapping)
    ]
    checks = {
        "frozen_snapshot_digest_exact": replay.get("full_snapshot_digest_exact")
        is True,
        "candidate_universe_exact": replay.get("candidate_universe_exact") is True,
        "pending_universe_exact": replay.get("pending_universe_exact") is True,
        "logical_k_exact": selection.get("logical_k") == result.get("logical_k"),
        "selected_candidate_ids_exact": selection.get(
            "selected_candidate_ids_exact"
        )
        is True,
        "handle_mapping_complete_unique": mapping.get("complete") is True
        and mapping.get("unique_runtime_identities") is True,
        "expected_actual_recurrent_residency_exact": (
            invariants.get("selected_residency_exact") is True
            and len(pending) == 4
            and all(
                item.get("expected_actual_residency_exact") is True
                for item in pending
            )
        ),
        "fa_preserved": invariants.get("fa_residency_preserved") is True,
        "g_equals_h_minus_e": len(pending) == 4
        and all(
            item.get("g") is not None
            and int(item["g"]) == int(item["h"]) - int(item["e"])
            for item in pending
        ),
        "native_recurrent_eviction_zero": invariants.get(
            "native_mamba_capacity_eviction"
        )
        is False
        and all(
            item.get("native_mamba_capacity_eviction") is False
            for item in pending
        ),
        "unexpected_rematerialization_zero": reconcile.get(
            "unexpected_rematerialization"
        )
        is False,
        "fa_cascade_zero": invariants.get("fa_kv_cascade") is False
        and all(item.get("fa_kv_cascade") is False for item in pending),
        "truncation_zero": all(
            item.get("truncation_or_clipping") is False for item in pending
        ),
        "oom_zero": all(item.get("oom") is False for item in pending),
        "future_leakage_zero": result.get("future_leakage") is False,
        "fresh_engine_empty": result.get("fresh_engine_empty") is True,
        "pending_requests_complete": len(pending) == 4
        and all(item.get("request_completed") is True for item in pending),
        "worker_completed": result.get("worker_exit_code") == 0
        and result.get("worker_timed_out") is False,
    }
    passed = result.get("status") == "PASS" and all(checks.values())
    return {
        "schema_version": "flowstate.rq4_run_correctness.v1",
        "run_id": result.get("run_id"),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
    }


def _create_artifact_root(explicit: Path | None, mode: str) -> Path:
    """创建不会覆盖既有结果的 runtime artifact root。"""
    if explicit is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ARTIFACT_ROOT / f"rq4_runtime_{mode}_{timestamp}"
    else:
        path = explicit
    path.mkdir(parents=True, exist_ok=False)
    (path / "runs").mkdir(exist_ok=False)
    (path / "workers").mkdir(exist_ok=False)
    return path


def _worker_command(
    artifact_root: Path,
    formal_root: Path,
    run_id: str,
) -> list[str]:
    """构造单个 isolated worker 命令。"""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--artifact-root",
        str(artifact_root),
        "--formal-root",
        str(formal_root),
        "--run-id",
        run_id,
    ]


def _execute_worker(
    *,
    artifact_root: Path,
    formal_root: Path,
    run_plan: Mapping[str, object],
) -> dict[str, object]:
    """启动独立进程，且只在超时时清理本 worker 的进程组。"""
    import signal

    run_id = str(run_plan["run_id"])
    log_path = artifact_root / "workers" / f"{run_id}.log"
    command = _worker_command(artifact_root, formal_root, run_id)
    process_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    timed_out = False
    try:
        exit_code = process.wait(timeout=WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        exit_code = process.returncode
    process_wall_s = time.perf_counter() - process_started
    result_path = artifact_root / "runs" / run_id / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            **dict(run_plan),
            "status": "INVALID",
            "fatal_error": {
                "phase": "worker_process",
                "error": "worker 未生成 result.json",
            },
        }
    result["worker_exit_code"] = exit_code
    result["worker_timed_out"] = timed_out
    result["worker_process_wall_s"] = process_wall_s
    result["worker_log"] = str(log_path)
    if timed_out or exit_code != 0:
        result["status"] = "INVALID"
    if result_path.parent.exists():
        _write_json(result_path, result)
        shutil.copyfile(log_path, result_path.parent / "runtime.log")
        pending_path = result_path.parent / "pending_requests.jsonl"
        if pending_path.exists():
            shutil.copyfile(
                pending_path,
                result_path.parent / "request_telemetry.jsonl",
            )
        _write_json(
            result_path.parent / "correctness.json",
            _run_correctness(result),
        )
    return result


def _summarize_runtime(
    artifact_root: Path,
    run_plan: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
    final_gpu: Mapping[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    """汇总 smoke 或 formal collection 的 correctness 与 telemetry。"""
    passed_runs = [row for row in runs if row.get("status") == "PASS"]
    pending = [
        request
        for row in passed_runs
        for request in row.get("pending_requests", ())
    ]

    def section(
        row: Mapping[str, object],
        name: str,
    ) -> Mapping[str, object]:
        """把缺失或空的 run 子记录安全规范为空映射。"""
        value = row.get(name)
        return value if isinstance(value, Mapping) else {}

    def timing_mean(name: str) -> float | None:
        values = [
            float(row["timings"][name])
            for row in passed_runs
            if isinstance(row.get("timings"), Mapping)
            and row["timings"].get(name) is not None
        ]
        return mean(values) if values else None

    required_runs = 6 if mode == "smoke" else FORMAL_RUN_COUNT
    expected_pending = required_runs * 4
    any_invalid = any(row.get("status") != "PASS" for row in runs)
    all_pass = (
        len(run_plan) == required_runs
        and len(runs) == required_runs
        and len(passed_runs) == required_runs
        and final_gpu.get("stable") is True
    )
    if all_pass:
        status = f"RQ4_RUNTIME_{mode.upper()}_READY"
    elif mode == "formal" and not any_invalid:
        status = "RQ4_RUNTIME_FORMAL_IN_PROGRESS"
    else:
        status = f"RQ4_RUNTIME_{mode.upper()}_BLOCKED"
    policy_ttft = {}
    for policy in POLICIES:
        values = [
            float(item["ttft_ms"])
            for row in passed_runs
            if row.get("policy") == policy
            for item in row.get("pending_requests", ())
            if item.get("ttft_ms") is not None
        ]
        policy_ttft[policy] = {
            "request_count": len(values),
            "mean_ttft_ms": mean(values) if values else None,
            "median_ttft_ms": median(values) if values else None,
        }
    cross_run_contamination = not bool(final_gpu.get("stable")) or any(
        row.get("fresh_engine_empty") is not True
        or section(row, "gpu_wait_before").get("stable") is not True
        or section(row, "gpu_wait_after").get("stable") is not True
        for row in runs
    )
    summary = {
        "schema_version": f"flowstate.rq4_runtime_{mode}.v1",
        "status": status,
        "mode": mode,
        "artifact_root": str(artifact_root),
        "planned_runs": len(run_plan),
        "completed_runs": len(runs),
        "passed_runs": len(passed_runs),
        "pending_completed": len(pending),
        "handle_mapping_pass": all(
            section(row, "handle_mapping").get("status") == "PASS"
            for row in passed_runs
        )
        and all_pass,
        "residency_validation_pass": all(
            section(section(row, "reconcile"), "invariants").get("status")
            == "PASS"
            for row in passed_runs
        )
        and all_pass,
        "heg_telemetry_pass": len(pending) == expected_pending
        and all(item.get("runtime_metrics_valid") is True for item in pending),
        "g_equals_h_minus_e_count": sum(
            item.get("g") is not None
            and int(item["g"]) == int(item["h"]) - int(item["e"])
            for item in pending
        ),
        "ttft_telemetry_pass": len(pending) == expected_pending
        and all(item.get("ttft_ms") is not None for item in pending),
        "recovery_latency_telemetry": "PARTIAL",
        "fa_preserved": all(
            section(section(row, "reconcile"), "invariants").get(
                "fa_residency_preserved"
            )
            is True
            for row in passed_runs
        )
        and all_pass,
        "native_recurrent_eviction": sum(
            bool(
                section(section(row, "reconcile"), "invariants").get(
                    "native_mamba_capacity_eviction"
                )
            )
            + sum(
                bool(item.get("native_mamba_capacity_eviction"))
                for item in row.get("pending_requests", ())
            )
            for row in runs
        ),
        "unexpected_rematerialization": sum(
            bool(
                section(row, "reconcile").get(
                    "unexpected_rematerialization"
                )
            )
            for row in runs
        ),
        "fa_cascade": sum(
            bool(
                section(section(row, "reconcile"), "invariants").get(
                    "fa_kv_cascade"
                )
            )
            + sum(
                bool(item.get("fa_kv_cascade"))
                for item in row.get("pending_requests", ())
            )
            for row in runs
        ),
        "truncation": sum(
            bool(item.get("truncation_or_clipping"))
            for row in runs
            for item in row.get("pending_requests", ())
        ),
        "oom": sum(
            bool(item.get("oom"))
            for row in runs
            for item in row.get("pending_requests", ())
        ),
        "future_leakage": sum(bool(row.get("future_leakage")) for row in runs),
        "cross_run_contamination": cross_run_contamination,
        "mean_engine_init_s": timing_mean("engine_init_s"),
        "mean_replay_s": timing_mean("replay_s"),
        "mean_reconcile_s": timing_mean("reconcile_s"),
        "mean_pending_resume_s": timing_mean("pending_resume_s"),
        "mean_total_run_s": timing_mean("total_run_s"),
        "mean_worker_process_wall_s": (
            mean(float(row["worker_process_wall_s"]) for row in passed_runs)
            if passed_runs
            else None
        ),
        "reestimated_216_run_gpu_hours": (
            mean(float(row["worker_process_wall_s"]) for row in passed_runs)
            * 216
            / 3600.0
            if passed_runs
            else None
        ),
        "final_gpu": dict(final_gpu),
        "preliminary_policy_metrics": policy_ttft,
        "formal_statistics_complete": mode == "formal" and all_pass,
        "statistical_unit": "snapshot",
    }
    if mode == "smoke":
        summary["runs"] = list(runs)
    return summary


def summarize_smoke(
    artifact_root: Path,
    run_plan: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
    final_gpu: Mapping[str, object],
) -> dict[str, object]:
    """保持既有 smoke 汇总接口。"""
    return _summarize_runtime(
        artifact_root,
        run_plan,
        runs,
        final_gpu,
        mode="smoke",
    )


def summarize_formal(
    artifact_root: Path,
    run_plan: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
    final_gpu: Mapping[str, object],
) -> dict[str, object]:
    """汇总正式 216-run collection，不把每个 run 内嵌进大 JSON。"""
    return _summarize_runtime(
        artifact_root,
        run_plan,
        runs,
        final_gpu,
        mode="formal",
    )


def _runtime_correctness_summary(
    run_plan: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """汇总正式 collection 的逐 run correctness gate。"""
    rows = [_run_correctness(row) for row in runs]
    pass_count = sum(row["status"] == "PASS" for row in rows)
    failed = [row["run_id"] for row in rows if row["status"] != "PASS"]
    return {
        "schema_version": "flowstate.rq4_runtime_correctness.v1",
        "status": (
            "PASS"
            if len(run_plan) == FORMAL_RUN_COUNT
            and len(rows) == FORMAL_RUN_COUNT
            and pass_count == FORMAL_RUN_COUNT
            else "INCOMPLETE" if not failed else "FAIL"
        ),
        "planned_runs": len(run_plan),
        "completed_runs": len(rows),
        "passed_runs": pass_count,
        "invalid_runs": len(failed),
        "invalid_run_ids": failed,
        "runs": rows,
    }


def _write_final_report(
    path: Path,
    summary: Mapping[str, object],
    correctness: Mapping[str, object],
    integrity: Mapping[str, object],
) -> None:
    """写出只陈述 collection 与 correctness 的中文正式报告。"""
    metrics = summary.get("preliminary_policy_metrics", {})
    lines = [
        "# RQ4-C 正式端到端运行时采集报告",
        "",
        f"- 状态：{summary.get('status')}",
        f"- 计划 run：{summary.get('planned_runs')}",
        f"- 完成 run：{summary.get('completed_runs')}",
        f"- 有效 run：{summary.get('passed_runs')}",
        f"- 完成 pending request：{summary.get('pending_completed')}",
        f"- correctness：{correctness.get('status')}",
        f"- source integrity：{integrity.get('status')}",
        f"- GPU 最终清理：{'PASS' if summary.get('final_gpu', {}).get('stable') else 'FAIL'}",
        "- recovery latency telemetry：PARTIAL；未用 TTFT 代填。",
        "- 正式统计单位：snapshot；本文件中的 policy 均值仅为 preliminary aggregate。",
        "",
        "## Preliminary request-level TTFT",
        "",
    ]
    for policy in POLICIES:
        row = metrics.get(policy, {}) if isinstance(metrics, Mapping) else {}
        lines.append(
            f"- {policy}：requests={row.get('request_count')}，"
            f"mean={row.get('mean_ttft_ms')} ms，"
            f"median={row.get('median_ttft_ms')} ms"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_parent(args: argparse.Namespace) -> int:
    """冻结 run plan，顺序执行并在首个 INVALID 时停止。"""
    if args.mode == "formal":
        if args.snapshot_id:
            raise RQ4CorrectnessError("formal 模式禁止覆盖 RQ4-A frozen snapshot list")
        if args.repetitions != FORMAL_REPETITIONS:
            raise RQ4CorrectnessError("formal 模式 repetitions 必须等于 3")
        snapshot_ids = FORMAL_SNAPSHOT_IDS
    else:
        snapshot_ids = tuple(args.snapshot_id or SMOKE_SNAPSHOT_IDS)
    artifact_root = _create_artifact_root(args.artifact_root, args.mode)
    runs = build_run_plan(args.formal_root, snapshot_ids, args.repetitions)
    if args.mode == "smoke" and not args.snapshot_id and len(runs) != 6:
        raise RQ4CorrectnessError("正式 smoke run plan 必须恰好包含 6 个 run")
    if args.mode == "formal":
        rounds = Counter(int(row["allocation_round"]) for row in runs)
        if len(runs) != FORMAL_RUN_COUNT or rounds != {2: 54, 3: 54, 4: 54, 5: 54}:
            raise RQ4CorrectnessError("formal run plan 数量或 round 分布不一致")
    frozen_sources = _frozen_source_paths()
    source_before = {str(path): _source_digest(path) for path in frozen_sources}
    manifest = {
        "schema_version": f"flowstate.rq4_runtime_{args.mode}_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "formal_root": str(args.formal_root),
        "snapshot_ids": list(snapshot_ids),
        "policies": list(POLICIES),
        "budget_ratio": 0.25,
        "budget_rule": "K = max(1, floor(0.25 * |C|))",
        "repetitions": args.repetitions,
        "physical_gpu_index": args.physical_gpu_index,
        "runtime_visible_gpu_index": args.gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sampling_seed": SAMPLING_SEED,
        "fresh_process_and_engine_per_run": True,
        "recovery_latency_telemetry": "PARTIAL",
        "source_files": [
            {"path": str(path), "sha256": source_before[str(path)]}
            for path in frozen_sources
        ],
        "input_dataset": {
            "path": str(DATASET_PATH),
            "sha256": _source_digest(DATASET_PATH),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("sglang", "transformers", "pyarrow")
        },
        "runs": runs,
    }
    _write_json(artifact_root / "manifest.json", manifest)
    if args.mode == "formal":
        _write_json(artifact_root / "frozen_protocol.json", _frozen_protocol())
        _write_json(
            artifact_root / "run_plan.json",
            {
                "schema_version": "flowstate.rq4_formal_run_plan.v1",
                "order": _frozen_protocol()["run_order"],
                "planned_runs": len(runs),
                "runs": runs,
            },
        )
    (artifact_root / "runs.jsonl").touch(exist_ok=False)

    initial_processes = query_gpu_compute_processes(args.gpu_index)
    manifest["pre_existing_compute_processes"] = initial_processes
    manifest["initial_gpu_used_mib"] = query_gpu_memory_used_mib(args.gpu_index)
    _write_json(artifact_root / "manifest.json", manifest)
    if initial_processes:
        prefix = args.mode.upper()
        summary = {
            "status": f"RQ4_RUNTIME_{prefix}_BLOCKED",
            "artifact_root": str(artifact_root),
            "blocker": "实验前 GPU 已存在 compute process，为避免影响外部任务未启动 collection",
            "initial_compute_processes": initial_processes,
        }
        name = "collection_summary.json" if args.mode == "formal" else "summary.json"
        _write_json(artifact_root / name, summary)
        return 1

    completed: list[dict[str, object]] = []
    final_gpu: Mapping[str, object] = {
        "stable": False,
        "compute_processes": initial_processes,
    }
    try:
        for run_plan in runs:
            wait_before = wait_gpu_stable(gpu_index=args.gpu_index)
            result = _execute_worker(
                artifact_root=artifact_root,
                formal_root=args.formal_root,
                run_plan=run_plan,
            )
            try:
                wait_after = wait_gpu_stable(gpu_index=args.gpu_index)
            except Exception as error:
                wait_after = {"stable": False, "error": repr(error)}
                result["status"] = "INVALID"
            result["gpu_wait_before"] = dict(wait_before)
            result["gpu_wait_after"] = dict(wait_after)
            result_path = (
                artifact_root
                / "runs"
                / str(run_plan["run_id"])
                / "result.json"
            )
            _write_json(result_path, result)
            _write_json(
                result_path.parent / "correctness.json",
                _run_correctness(result),
            )
            completed.append(result)
            _append_jsonl(artifact_root / "runs.jsonl", result)
            if args.mode == "formal":
                progress = summarize_formal(
                    artifact_root,
                    runs,
                    completed,
                    {"stable": False, "collection_in_progress": True},
                )
                _write_json(artifact_root / "collection_summary.json", progress)
                _write_json(
                    artifact_root / "runtime_correctness.json",
                    _runtime_correctness_summary(runs, completed),
                )
                print(
                    f"正式采集进度：{len(completed)}/{len(runs)}，"
                    f"当前 run={run_plan['run_id']}，status={result['status']}",
                    flush=True,
                )
            if result["status"] != "PASS":
                break
        try:
            final_gpu = wait_gpu_stable(gpu_index=args.gpu_index)
        except Exception as error:
            final_gpu = {
                "stable": False,
                "error": repr(error),
                "used_mib": query_gpu_memory_used_mib(args.gpu_index),
                "compute_processes": query_gpu_compute_processes(
                    args.gpu_index
                ),
            }
    finally:
        summary = (
            summarize_formal(artifact_root, runs, completed, final_gpu)
            if args.mode == "formal"
            else summarize_smoke(artifact_root, runs, completed, final_gpu)
        )
        integrity = _source_integrity(frozen_sources, source_before)
        if integrity["status"] != "PASS":
            summary["status"] = f"RQ4_RUNTIME_{args.mode.upper()}_BLOCKED"
            summary["source_integrity_failure"] = True
        if args.mode == "formal":
            correctness = _runtime_correctness_summary(runs, completed)
            _write_json(artifact_root / "collection_summary.json", summary)
            _write_json(artifact_root / "runtime_correctness.json", correctness)
            _write_json(artifact_root / "source_integrity.json", integrity)
            _write_final_report(
                artifact_root / "final_report.md",
                summary,
                correctness,
                integrity,
            )
        else:
            _write_json(artifact_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    ready = f"RQ4_RUNTIME_{args.mode.upper()}_READY"
    return 0 if summary["status"] == ready else 1


def main() -> int:
    """解析统一 harness 的 parent/worker 模式。"""
    parser = argparse.ArgumentParser(description="RQ4 统一运行时 harness")
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=DEFAULT_FORMAL_ROOT,
        help="冻结 OpenHands formal population root",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="新建的 RQ4 runtime artifact root",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "formal"),
        default="smoke",
        help="执行 smoke 或冻结的正式 216-run collection",
    )
    parser.add_argument(
        "--snapshot-id",
        action="append",
        default=None,
        help="要执行的冻结 snapshot ID；省略时使用两个 smoke snapshot",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="每个 snapshot-policy 的独立重复数",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="容器内清洁门禁检查的可见 GPU 编号",
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=0,
        help="仅用于 artifact 记录的宿主机物理 GPU 编号",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.worker:
        if args.artifact_root is None or args.run_id is None:
            raise SystemExit("worker 模式必须提供 artifact root 与 run ID")
        return _run_worker(
            artifact_root=args.artifact_root,
            formal_root=args.formal_root,
            run_id=args.run_id,
        )
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
