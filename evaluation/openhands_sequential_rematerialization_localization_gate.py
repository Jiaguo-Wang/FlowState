#!/usr/bin/env python3
"""在 Barrier 2 的 LRU 连续驱逐中定位首次循环状态重驻留。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback

from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SchedulerRuntimeAdapter,
    wait_for_transport,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    _environment,
    compact_census,
    execute_request,
)
from evaluation.openhands_barrier2_second_reconcile_gate import (
    ENGINE_CONFIGURATION_SECOND_RECONCILE,
    build_second_reconcile_plan,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    locate_materialized_candidate,
    validate_candidate_at_barrier,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    FrozenSelectedSetOptimizer,
)
from evaluation.openhands_round2_dynamic_registry_selection_gate import (
    EXPECTED_BARRIER_ONE_SELECTION,
    ROUND_ONE_SCHEDULE,
    ROUND_TWO_SCHEDULE,
    build_dynamic_metadata,
    build_round_three_pending,
    build_round_three_pending_compatible,
    load_round2_visible_requests,
    refresh_registry,
    register_round_two_materialization,
    registry_candidates,
    registry_entry_from_round_one,
    run_policy_selector,
)
from evaluation.openhands_sequential_eviction_rematerialization_audit import (
    LRU_SECOND_EVICTION_ORDER,
    LRU_TRACKED_CHECKPOINTS,
    SequentialTraceRuntimeAdapter,
    find_first_rematerializations,
    validate_sequential_trace,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    TOKENIZER_PATH,
)
from flowstate.controller import StateController


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
POLICY = "LRU"


def _artifact_directory(attempt: int) -> Path:
    """创建包含尝试序号且不会覆盖既有结果的产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        "openhands_sequential_rematerialization_localization_"
        f"attempt{attempt}_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _tracked_handles(registry, eligible_handles):
    """按冻结的六检查点范围构造追踪句柄集合。"""
    result = {}
    for checkpoint_id in LRU_TRACKED_CHECKPOINTS:
        if checkpoint_id in eligible_handles:
            result[checkpoint_id] = eligible_handles[checkpoint_id]
            continue
        entry = registry.get(checkpoint_id)
        if entry is None:
            raise RuntimeError(f"追踪 registry 缺少 {checkpoint_id}")
        result[checkpoint_id] = entry.handle
    return result


def _stage_rows(trace_rows):
    """提取本步骤要求报告的 B1 与 C1 边界。"""
    wanted = {
        ("OPENHANDS_BARRIER_B_TURN_001", boundary)
        for boundary in ("S0", "S1", "S2", "S3", "S4")
    }
    wanted.update(
        {
            ("OPENHANDS_BARRIER_C_TURN_001", boundary)
            for boundary in ("S0", "S1", "S2", "S3")
        }
    )
    return [
        dict(row)
        for row in trace_rows
        if (row["target_checkpoint_id"], row["boundary"]) in wanted
    ]


def _run_attempt(artifact: Path, attempt: int) -> dict[str, object]:
    """复现一次 LRU Barrier-2 lifecycle，并只执行连续驱逐定位。"""
    engine = None
    client = None
    records = []
    registry = {}
    trace_rows = []
    first_selection = None
    second_selection = None
    second_plan = None
    rematerializations = {}
    fatal_error = None
    shutdown_error = None
    environment = _environment()
    try:
        from sequential_eviction_trace_transport import (
            SequentialEvictionTraceGateEngine,
            requested_control_port,
        )
        from targeted_probe import ControlClient

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests, boundary_audit = load_round2_visible_requests(tokenizer)
        if any(
            row.get("round_4_message_consumed")
            or row.get("round_4_request_materialized")
            or row.get("future_timing_read")
            or row.get("future_checkpoint_read")
            for row in boundary_audit
        ):
            raise RuntimeError("在线信息边界包含未来信息")

        engine = SequentialEvictionTraceGateEngine(
            **ENGINE_CONFIGURATION_SECOND_RECONCILE
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"step12h9b:{attempt}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("fresh Engine 初始含循环检查点")
        previous = baseline

        round_one_candidates = []
        round_one_handles = {}
        for ordinal, (label, turn) in enumerate(
            ROUND_ONE_SCHEDULE,
            start=1,
        ):
            request = requests[(label, turn)]
            record = execute_request(engine, client, request, ordinal)
            census = compact_census(
                client.census(
                    f"step12h9b:{attempt}:after:{label}{turn}"
                ),
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
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
            records.append(record)
            _append_jsonl(artifact / "requests.jsonl", record)
            _append_jsonl(artifact / "census.jsonl", census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{label}1 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                raise RuntimeError("Round 1 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
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
            turn=2,
            policy=POLICY,
        )
        candidates_one = registry_candidates(registry)
        metadata_one, _ = build_dynamic_metadata(registry, candidates_one)
        first_selection = run_policy_selector(
            POLICY,
            candidates_one,
            pending_two,
            metadata_one,
        )
        if set(first_selection["selected_checkpoint_ids"]) != set(
            EXPECTED_BARRIER_ONE_SELECTION[POLICY]
        ):
            raise RuntimeError("Barrier 1 LRU selected set 异常")
        first_controller = StateController(
            FrozenSelectedSetOptimizer(
                first_selection["selected_checkpoint_ids"]
            ),
            SchedulerRuntimeAdapter(client),
        )
        first_controller.reconcile(
            pending_two,
            candidates_one,
            round_one_handles,
            BUDGET_BYTES,
        )
        after_first = compact_census(
            client.census(f"step12h9b:{attempt}:after-first-reconcile"),
            ordinal=len(ROUND_ONE_SCHEDULE),
            request=None,
            previous=previous,
        )
        _append_jsonl(artifact / "census.jsonl", after_first)
        previous = after_first
        refresh_registry(client, registry, phase="STEP12H9B_POST_BARRIER1")

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
                    f"step12h9b:{attempt}:after:{label}{turn}"
                ),
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            register_round_two_materialization(
                client,
                registry,
                request,
                record,
                census,
                event_order=ordinal,
                previously_resident_ids=previously_resident,
            )
            refresh_registry(
                client,
                registry,
                phase=f"STEP12H9B_{label}{turn}",
            )
            records.append(record)
            _append_jsonl(artifact / "requests.jsonl", record)
            _append_jsonl(artifact / "census.jsonl", census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{label}2 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                raise RuntimeError("Round 2 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                raise RuntimeError("Round 2 发生 FA 级联")

        refresh_registry(client, registry, phase="STEP12H9B_BARRIER2")
        candidates_two = registry_candidates(registry)
        metadata_two, _ = build_dynamic_metadata(registry, candidates_two)
        pending_three, _ = build_round_three_pending(
            barrier_client,
            requests,
            policy=POLICY,
        )
        second_selection = run_policy_selector(
            POLICY,
            candidates_two,
            pending_three,
            metadata_two,
        )
        second_plan = build_second_reconcile_plan(
            POLICY,
            candidates_two,
            second_selection,
        )
        eligible_ids = set(second_plan["eligible_candidate_ids"])
        eligible_candidates = tuple(
            candidate
            for candidate in candidates_two
            if candidate.checkpoint_id in eligible_ids
        )
        eligible_handles = {
            candidate.checkpoint_id: registry[candidate.checkpoint_id].handle
            for candidate in eligible_candidates
        }
        for candidate in eligible_candidates:
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                eligible_handles[candidate.checkpoint_id],
            )
            if not validation["consistent"]:
                raise RuntimeError("Barrier 2 candidate residency 不一致")

        tracked_handles = _tracked_handles(registry, eligible_handles)
        trace_adapter = SequentialTraceRuntimeAdapter(
            client,
            tracked_handles,
        )
        second_controller = StateController(
            FrozenSelectedSetOptimizer(
                second_plan["selected_checkpoint_ids"]
            ),
            trace_adapter,
        )
        try:
            second_controller.reconcile(
                pending_three,
                eligible_candidates,
                eligible_handles,
                BUDGET_BYTES,
            )
        finally:
            trace_adapter.finish()
            trace_rows = list(trace_adapter.trace_rows)
            for row in trace_rows:
                _append_jsonl(artifact / "trace.jsonl", row)

        validate_sequential_trace(
            trace_rows,
            LRU_SECOND_EVICTION_ORDER,
            tuple(sorted(tracked_handles)),
        )
        rematerializations = {
            checkpoint_id: (
                None if event is None else event.row()
            )
            for checkpoint_id, event in find_first_rematerializations(
                trace_rows,
                (
                    "OPENHANDS_BARRIER_A_TURN_001",
                    "OPENHANDS_BARRIER_B_TURN_001",
                ),
            ).items()
        }
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    reproduced = bool(
        rematerializations
        and all(value is not None for value in rematerializations.values())
    )
    result = {
        "schema_version": "flowstate.openhands_rematerialization_trace.v1",
        "attempt": attempt,
        "status": (
            "CONFIRMED"
            if reproduced and fatal_error is None and shutdown_error is None
            else "NOT_REPRODUCED"
            if fatal_error is None and shutdown_error is None
            else "BLOCKED"
        ),
        "policy": POLICY,
        "engine_configuration": ENGINE_CONFIGURATION_SECOND_RECONCILE,
        "first_selection": first_selection,
        "second_selection": second_selection,
        "second_plan": second_plan,
        "request_count": len(records),
        "trace_complete": len(trace_rows) == 20,
        "stage_rows": _stage_rows(trace_rows),
        "rematerializations": rematerializations,
        "fatal_error": fatal_error,
        "shutdown_error": shutdown_error,
        "environment": environment,
    }
    _write_json(artifact / "summary.json", result)
    return result


def main() -> int:
    """最多执行三次不变配置复现，并在首次定位后立即停止。"""
    attempts = []
    for attempt in range(1, 4):
        artifact = _artifact_directory(attempt)
        for name in ("requests.jsonl", "census.jsonl", "trace.jsonl"):
            (artifact / name).touch(exist_ok=False)
        _write_json(
            artifact / "config.json",
            {
                "attempt": attempt,
                "policy": POLICY,
                "engine_configuration": ENGINE_CONFIGURATION_SECOND_RECONCILE,
                "tracked_checkpoints": list(LRU_TRACKED_CHECKPOINTS),
                "eviction_order": list(LRU_SECOND_EVICTION_ORDER),
                "repair_enabled": False,
            },
        )
        with ArtifactLogCapture(artifact):
            result = _run_attempt(artifact, attempt)
        attempts.append({**result, "artifact": str(artifact)})
        if result["status"] in {"CONFIRMED", "BLOCKED"}:
            break
    final = attempts[-1]
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if final["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
