#!/usr/bin/env python3
"""执行三轮 OpenHands 在线分配的最终运行时稳定性门禁。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SchedulerRuntimeAdapter,
    wait_for_transport,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    WORKFLOWS,
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
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
    inspect_candidate_states,
)
from evaluation.openhands_round2_dynamic_registry_selection_gate import (
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
    SequentialTraceRuntimeAdapter,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import TOKENIZER_PATH
from flowstate.controller import StateController


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
POLICY_RUN_COUNTS = (("LRU", 3), ("Marconi", 2), ("FlowState", 2))
ROUND_THREE_SCHEDULE = tuple((label, 3) for label in WORKFLOWS)


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的最终门禁产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / f"openhands_final_multi_round_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _run_directory(root: Path, policy: str, run_index: int) -> Path:
    """创建单个独立 Engine lifecycle 的产物目录。"""
    directory = root / f"{policy.lower()}_run_{run_index:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    for name in (
        "requests.jsonl",
        "census.jsonl",
        "barriers.jsonl",
        "trace.jsonl",
    ):
        (directory / name).touch(exist_ok=False)
    return directory


def _trace_has_rematerialization(
    trace_rows: Sequence[Mapping[str, object]],
    evicted_ids: Sequence[str],
) -> bool:
    """检测目标被原语驱逐后是否在同一 reconcile 内重新设备驻留。"""
    absent_seen = {checkpoint_id: False for checkpoint_id in evicted_ids}
    for row in trace_rows:
        checkpoints = row["checkpoints"]
        for checkpoint_id in evicted_ids:
            state = checkpoints[checkpoint_id]
            present = bool(state["recurrent_present"])
            if absent_seen[checkpoint_id] and present:
                return True
            if not present:
                absent_seen[checkpoint_id] = True
    return False


def _barrier_reconcile(
    *,
    policy: str,
    barrier: int,
    client: object,
    pending: Sequence[object],
    candidates: Sequence[object],
    handles: Mapping[str, object],
    selected_ids: Sequence[str],
    previous_census: Mapping[str, object],
    run_directory: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """执行一次选择结果映射，并在下一请求前验证最终驻留。"""
    candidate_ids = tuple(candidate.checkpoint_id for candidate in candidates)
    expected_evicted_ids = tuple(
        sorted(set(candidate_ids) - set(selected_ids))
    )
    before_census = compact_census(
        client.census(f"final:{policy}:barrier{barrier}:before"),
        ordinal=barrier * 4,
        request=None,
        previous=previous_census,
    )
    before_states, _ = inspect_candidate_states(
        client,
        candidates,
        handles,
        phase=f"FINAL_{policy}_BARRIER{barrier}_BEFORE",
    )
    trace_delegate = SequentialTraceRuntimeAdapter(
        client,
        handles,
        nonce_namespace=f"final:{policy}:barrier{barrier}",
    )
    recording_adapter = RecordingRuntimeAdapter(trace_delegate)
    controller = StateController(
        FrozenSelectedSetOptimizer(selected_ids),
        recording_adapter,
    )
    allocation = None
    try:
        allocation = controller.reconcile(
            pending,
            candidates,
            handles,
            BUDGET_BYTES,
        )
    finally:
        trace_delegate.finish()
        for row in trace_delegate.trace_rows:
            _append_jsonl(
                run_directory / "trace.jsonl",
                {"barrier": barrier, **row},
            )
    controller_report = build_controller_report(
        allocation=allocation,
        adapter=recording_adapter,
    )
    after_census = compact_census(
        client.census(f"final:{policy}:barrier{barrier}:after"),
        ordinal=barrier * 4,
        request=None,
        previous=before_census,
    )
    after_states, _ = inspect_candidate_states(
        client,
        candidates,
        handles,
        phase=f"FINAL_{policy}_BARRIER{barrier}_AFTER",
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
    unexpected_rematerialization = _trace_has_rematerialization(
        trace_delegate.trace_rows,
        expected_evicted_ids,
    )
    barrier_result = {
        "policy": policy,
        "barrier": barrier,
        "candidate_ids": list(candidate_ids),
        "selected_ids": list(selected_ids),
        "expected_retained_ids": sorted(selected_ids),
        "actual_retained_ids": sorted(invariants["actual_retained_ids"]),
        "retained_set_match": (
            set(invariants["actual_retained_ids"]) == set(selected_ids)
        ),
        "expected_evicted_ids": list(expected_evicted_ids),
        "unexpected_rematerialization": unexpected_rematerialization,
        "controller_report": controller_report,
        "invariants": invariants,
        "status": (
            "PASS"
            if invariants["status"] == "PASS"
            and not unexpected_rematerialization
            else "FAIL"
        ),
    }
    _append_jsonl(run_directory / "barriers.jsonl", barrier_result)
    _append_jsonl(
        run_directory / "census.jsonl",
        {"event": f"barrier_{barrier}_before", **before_census},
    )
    _append_jsonl(
        run_directory / "census.jsonl",
        {"event": f"barrier_{barrier}_after", **after_census},
    )
    return barrier_result, after_census


def _execute_round_request(
    *,
    engine: object,
    client: object,
    request: Mapping[str, object],
    ordinal: int,
    previous_census: Mapping[str, object],
    run_directory: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """执行一个冻结请求并检查请求级 runtime 正确性。"""
    record = execute_request(engine, client, request, ordinal)
    census = compact_census(
        client.census(
            f"final:after:{request['workflow_label']}{request['turn']}"
        ),
        ordinal=ordinal,
        request=request,
        previous=previous_census,
    )
    _append_jsonl(run_directory / "requests.jsonl", record)
    _append_jsonl(run_directory / "census.jsonl", census)
    if record["status"] != "PASS":
        raise RuntimeError(
            f"{request['workflow_label']}{request['turn']} 请求失败"
        )
    if record.get("runtime_metrics_valid") is not True:
        raise RuntimeError(
            f"{request['workflow_label']}{request['turn']} H/E/G 无效"
        )
    if census["native_mamba_capacity_eviction_inferred"]:
        raise RuntimeError("请求执行中发生原生 Mamba 驱逐")
    if census["fa_kv_cascade_eviction_inferred"]:
        raise RuntimeError("请求执行中发生 FA 级联")
    return record, census


def _run_policy_lifecycle(
    *,
    policy: str,
    run_index: int,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    run_directory: Path,
) -> dict[str, object]:
    """在独立 fresh Engine 中执行三个 round 与两个 barrier。"""
    engine = None
    records = []
    barriers = []
    registry = {}
    fatal_error = None
    shutdown_error = None
    try:
        from sequential_eviction_trace_transport import (
            SequentialEvictionTraceGateEngine,
            requested_control_port,
        )
        from targeted_probe import ControlClient

        engine = SequentialEvictionTraceGateEngine(
            **ENGINE_CONFIGURATION_SECOND_RECONCILE
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        previous = compact_census(
            client.census(f"final:{policy}:{run_index}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        _append_jsonl(
            run_directory / "census.jsonl",
            {"event": "baseline", **previous},
        )
        if int(previous["mamba_node_count"]) != 0:
            raise RuntimeError("fresh Engine 初始含循环检查点")

        round_one_candidates = []
        round_one_handles = {}
        for ordinal, (label, turn) in enumerate(
            ROUND_ONE_SCHEDULE,
            start=1,
        ):
            request = requests[(label, turn)]
            record, census = _execute_round_request(
                engine=engine,
                client=client,
                request=request,
                ordinal=ordinal,
                previous_census=previous,
                run_directory=run_directory,
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
            previous = census

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
            policy=policy,
        )
        candidates_one = registry_candidates(registry)
        metadata_one, _ = build_dynamic_metadata(registry, candidates_one)
        selection_one = run_policy_selector(
            policy,
            candidates_one,
            pending_two,
            metadata_one,
        )
        if selection_one["selection_valid"] is not True:
            raise RuntimeError("Barrier 1 policy selection 无效")
        barrier_one, previous = _barrier_reconcile(
            policy=policy,
            barrier=1,
            client=client,
            pending=pending_two,
            candidates=candidates_one,
            handles=round_one_handles,
            selected_ids=selection_one["selected_checkpoint_ids"],
            previous_census=previous,
            run_directory=run_directory,
        )
        barriers.append(barrier_one)
        if barrier_one["status"] != "PASS":
            raise RuntimeError(
                f"{policy} Barrier 1 最终驻留或不变量失败"
            )
        refresh_registry(
            client,
            registry,
            phase=f"FINAL_{policy}_POST_BARRIER1",
        )

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
            record, census = _execute_round_request(
                engine=engine,
                client=client,
                request=request,
                ordinal=ordinal,
                previous_census=previous,
                run_directory=run_directory,
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
                phase=f"FINAL_{policy}_{label}{turn}",
            )
            records.append(record)
            previous = census

        refresh_registry(
            client,
            registry,
            phase=f"FINAL_{policy}_BARRIER2",
        )
        candidates_two = registry_candidates(registry)
        metadata_two, _ = build_dynamic_metadata(registry, candidates_two)
        pending_three, _ = build_round_three_pending(
            barrier_client,
            requests,
            policy=policy,
        )
        selection_two = run_policy_selector(
            policy,
            candidates_two,
            pending_three,
            metadata_two,
        )
        second_plan = build_second_reconcile_plan(
            policy,
            candidates_two,
            selection_two,
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
        barrier_two, previous = _barrier_reconcile(
            policy=policy,
            barrier=2,
            client=client,
            pending=pending_three,
            candidates=eligible_candidates,
            handles=eligible_handles,
            selected_ids=second_plan["selected_checkpoint_ids"],
            previous_census=previous,
            run_directory=run_directory,
        )
        barriers.append(barrier_two)
        if barrier_two["status"] != "PASS":
            raise RuntimeError(
                f"{policy} Barrier 2 最终驻留或不变量失败"
            )

        for offset, (label, turn) in enumerate(
            ROUND_THREE_SCHEDULE,
            start=1,
        ):
            ordinal = len(ROUND_ONE_SCHEDULE) + len(ROUND_TWO_SCHEDULE) + offset
            request = requests[(label, turn)]
            record, census = _execute_round_request(
                engine=engine,
                client=client,
                request=request,
                ordinal=ordinal,
                previous_census=previous,
                run_directory=run_directory,
            )
            records.append(record)
            previous = census
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    round_summaries = []
    for round_number in (1, 2, 3):
        round_records = [
            record for record in records if record.get("turn") == round_number
        ]
        round_summaries.append(
            {
                "round": round_number,
                "request_count": len(round_records),
                "h_values": [record.get("h") for record in round_records],
                "e_values": [record.get("e") for record in round_records],
                "g_values": [record.get("g") for record in round_records],
                "ttft_ms": [
                    record.get("ttft_ms") for record in round_records
                ],
                "status": (
                    "PASS"
                    if len(round_records) == 4
                    and all(record.get("status") == "PASS" for record in round_records)
                    else "FAIL"
                ),
            }
        )
    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and len(records) == 12
        and len(barriers) == 2
        and all(item["status"] == "PASS" for item in round_summaries)
        and all(item["status"] == "PASS" for item in barriers)
    )
    result = {
        "policy": policy,
        "run": run_index,
        "fresh_engine": True,
        "status": "PASS" if passed else "FAIL",
        "barriers": barriers,
        "rounds": round_summaries,
        "requests": records,
        "fatal_error": fatal_error,
        "shutdown_error": shutdown_error,
    }
    _write_json(run_directory / "summary.json", result)
    return result


def _build_summary(
    artifact: Path,
    runs: Sequence[Mapping[str, object]],
    environment: Mapping[str, object],
) -> dict[str, object]:
    """汇总有限重复及最终 correctness 判定。"""
    expected_run_total = sum(count for _, count in POLICY_RUN_COUNTS)
    complete = len(runs) == expected_run_total
    all_pass = complete and all(run["status"] == "PASS" for run in runs)
    barriers = [barrier for run in runs for barrier in run["barriers"]]
    requests = [request for run in runs for request in run["requests"]]
    return {
        "schema_version": "flowstate.openhands_final_multi_round.v1",
        "result": (
            "MULTI_ROUND_RUNTIME_READY"
            if all_pass
            else "MULTI_ROUND_RUNTIME_NOT_READY"
        ),
        "artifact": str(artifact),
        "policy_run_counts": dict(POLICY_RUN_COUNTS),
        "completed_run_count": len(runs),
        "runs": list(runs),
        "barrier_correctness": barriers,
        "fa_preserved": bool(
            all(
                barrier["invariants"]["fa_residency_preserved"]
                for barrier in barriers
            )
        ),
        "unexpected_rematerialization": any(
            barrier["unexpected_rematerialization"] for barrier in barriers
        ),
        "native_eviction_contamination": any(
            barrier["invariants"]["native_mamba_capacity_eviction"]
            for barrier in barriers
        ),
        "fa_cascade": any(
            barrier["invariants"]["fa_kv_cascade"] for barrier in barriers
        ),
        "oom_or_truncation": any(
            request.get("oom") or request.get("truncation_or_clipping")
            for request in requests
        ),
        "environment": dict(environment),
    }


def main() -> int:
    """按 LRU、Marconi、FlowState 的冻结次数顺序执行最终门禁。"""
    artifact = _artifact_directory()
    for name in ("runs.jsonl", "summary.json"):
        (artifact / name).touch(exist_ok=False)
    environment = _environment()
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
        raise RuntimeError("最终门禁在线信息边界包含 Round 4+ 信息")
    _write_json(
        artifact / "config.json",
        {
            "policy_run_counts": dict(POLICY_RUN_COUNTS),
            "engine_configuration": ENGINE_CONFIGURATION_SECOND_RECONCILE,
            "logical_k": 2,
            "physical_mamba_pool": 28,
            "rounds": 3,
            "online_information_boundary": boundary_audit,
            "environment": environment,
        },
    )
    runs = []
    with ArtifactLogCapture(artifact):
        for policy, count in POLICY_RUN_COUNTS:
            for run_index in range(1, count + 1):
                run_directory = _run_directory(
                    artifact,
                    policy,
                    run_index,
                )
                result = _run_policy_lifecycle(
                    policy=policy,
                    run_index=run_index,
                    requests=requests,
                    run_directory=run_directory,
                )
                runs.append(result)
                _append_jsonl(artifact / "runs.jsonl", result)
                if result["status"] != "PASS":
                    break
            if runs[-1]["status"] != "PASS":
                break
    summary = _build_summary(artifact, runs, environment)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["result"] == "MULTI_ROUND_RUNTIME_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
