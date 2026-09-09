#!/usr/bin/env python3
"""在 RQ4 冻结 24 快照上测量 FlowState 运行时控制路径开销。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter_ns
import traceback
from typing import Any, Iterable, Mapping

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
from evaluation.rq3_frozen_snapshot_evaluator import evaluate_objective
from evaluation.rq3_openhands_neutral_collector import (
    SGLangGroupRuntime,
    assemble_group_snapshot,
    materialize_group_requests,
    query_gpu_compute_processes,
    query_gpu_memory_used_mib,
    replay_group_to_barrier,
    wait_gpu_stable,
)
from evaluation.rq3_openhands_population import load_session_messages
from evaluation.rq4_unified_runtime_harness import (
    DEFAULT_FORMAL_ROOT,
    FORMAL_SNAPSHOT_IDS,
    RQ4CorrectnessError,
    _future_leakage,
    _group_from_manifest,
    _snapshot_path_for_group,
    build_current_runtime_handles,
    build_run_plan,
    rq4_logical_k,
    trace_has_rematerialization,
    validate_rebuilt_snapshot,
)
from evaluation.rq6_system_overhead import (
    REPOSITORY_ROOT,
    RUNTIME_REPETITIONS,
    RUNTIME_SNAPSHOT_COUNT,
    append_records,
    reference_flowstate_selection,
    source_manifest,
    source_paths,
    timed_flowstate_selection,
    write_json,
)
from flowstate.controller import StateController


WORKER_TIMEOUT_S = 1800.0


def read_only_runtime_introspection(
    runtime: Any,
    client: Any,
    candidates: tuple[Any, ...],
    handles: Mapping[str, Any],
    *,
    label: str,
    ordinal: int,
    previous_census: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    """只执行一次 census 与一次候选状态读取，并返回阶段耗时。"""
    started = perf_counter_ns()
    census = runtime.census(
        label,
        ordinal=ordinal,
        request=None,
        previous=previous_census,
    )
    states, inspections = inspect_candidate_states(
        client,
        candidates,
        handles,
        phase=label,
    )
    elapsed = perf_counter_ns() - started
    return dict(census), dict(states), list(inspections), elapsed


def build_runtime_plan(formal_root: Path) -> list[dict[str, Any]]:
    """从 RQ4 计划中只保留 24×3 个 FlowState 独立运行。"""
    full = build_run_plan(formal_root, FORMAL_SNAPSHOT_IDS, RUNTIME_REPETITIONS)
    rows = [dict(row) for row in full if row["policy"] == "FlowState"]
    if len(rows) != RUNTIME_SNAPSHOT_COUNT * RUNTIME_REPETITIONS:
        raise RuntimeError("RQ6 runtime 计划必须恰好包含 24×3 个 FlowState run")
    return rows


def _load_plan(output_root: Path, run_id: str) -> dict[str, Any]:
    """读取唯一匹配的冻结 runtime run。"""
    rows = json.loads(
        (output_root / "provenance/runtime_run_plan.json").read_text(encoding="utf-8")
    )
    matched = [row for row in rows if row["run_id"] == run_id]
    if len(matched) != 1:
        raise RuntimeError(f"runtime run plan 不唯一：{run_id}")
    return dict(matched[0])


def _runtime_record_base(run_plan: Mapping[str, Any]) -> dict[str, Any]:
    """构造失败时仍可写出的完整原始记录骨架。"""
    return {
        "schema_version": "flowstate.rq6_per_epoch_overhead.v1",
        "record_kind": "runtime_control",
        "population": "RQ4-OpenHands",
        "run_id": run_plan["run_id"],
        "snapshot_id": run_plan["snapshot_id"],
        "snapshot_digest": run_plan["snapshot_digest"],
        "budget_ratio": 0.25,
        "k": run_plan["logical_k"],
        "pending_count": 4,
        "candidate_count": run_plan["candidate_count"],
        "timing_repetition": int(run_plan["repetition"]) - 1,
        "timings_ns": {
            "introspection": None,
            "construction": None,
            "allocation": None,
            "reconciliation": None,
            "total_control": None,
        },
        "selected_checkpoint_ids": list(run_plan["selected_candidate_ids"]),
        "reference_selected_checkpoint_ids": list(run_plan["selected_candidate_ids"]),
        "instrumentation_equivalent": False,
        "snapshot_immutable": False,
        "heg_equivalent": False,
        "future_leakage": False,
        "cross_run_contamination": True,
        "correctness": {},
        "status": "INVALID",
    }


def _run_worker(output_root: Path, formal_root: Path, run_id: str) -> int:
    """在独立 Engine 生命周期中执行一个不含 TTFT 的控制路径。"""
    from transformers import AutoTokenizer

    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
    from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES

    run_plan = _load_plan(output_root, run_id)
    run_directory = output_root / "runtime_runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    record = _runtime_record_base(run_plan)
    engine = None
    runtime = None
    fatal_error = None
    shutdown_error = None
    replay_validation: Mapping[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    reconcile_record: dict[str, Any] | None = None
    fresh_engine_empty = False
    boundary_audits: list[dict[str, Any]] = []
    snapshot_immutable = False
    gpu_cleanup = False
    phase = "准备"
    try:
        manifest = json.loads(
            (formal_root / "population_manifest.json").read_text(encoding="utf-8")
        )
        group = _group_from_manifest(manifest, int(run_plan["group_ordinal"]))
        snapshot_path = _snapshot_path_for_group(formal_root, group.group_ordinal)
        frozen = _load_allocation_snapshot(snapshot_path)
        frozen_digest = frozen.content_digest()
        if frozen_digest != run_plan["snapshot_digest"]:
            raise RQ4CorrectnessError("RQ6 run plan 的 snapshot digest 不一致")
        k = rq4_logical_k(len(frozen.eligible_candidates))
        if k != int(run_plan["logical_k"]):
            raise RQ4CorrectnessError("RQ6 run plan 的 K 不一致")
        snapshot_k = create_budget_variant(frozen, k)
        reference_ids = reference_flowstate_selection(snapshot_k)
        if list(reference_ids) != run_plan["selected_candidate_ids"]:
            raise RQ4CorrectnessError("RQ6 参考选择与 RQ4 冻结选择不一致")

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
            raise RQ4CorrectnessError("RQ6 request materialization 出现未来信息")

        phase = "Engine 初始化"
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
        runtime = SGLangGroupRuntime(engine, client)
        baseline = runtime.census(
            f"rq6:{run_id}:baseline", ordinal=0, request=None, previous=None
        )
        fresh_engine_empty = int(baseline["mamba_node_count"]) == 0
        if not fresh_engine_empty:
            raise RQ4CorrectnessError("RQ6 fresh Engine 含有跨 run recurrent state")

        phase = "冻结 barrier replay"
        trace = replay_group_to_barrier(runtime, group, requests)
        trace = replace(trace, boundary_audit=tuple(audits))
        assembly = assemble_group_snapshot(
            trace, checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES
        )
        if assembly.status != "ELIGIBLE" or assembly.snapshot is None:
            raise RQ4CorrectnessError(
                f"RQ6 runtime replay 重建失败：{assembly.primary_reason}"
            )
        replay_validation = validate_rebuilt_snapshot(frozen, assembly.snapshot)
        handles = build_current_runtime_handles(trace, requests)
        candidates = snapshot_k.core_candidates()
        candidate_ids = tuple(item.checkpoint_id for item in candidates)
        if set(handles) != set(candidate_ids):
            raise RQ4CorrectnessError("RQ6 handle mapping 与候选集合不一致")
        mapping = {
            "status": "PASS",
            "complete": set(handles) == set(candidate_ids),
            "unique_runtime_identities": len(
                {
                    (handle.expected_node_id, handle.expected_prefix_digest)
                    for handle in handles.values()
                }
            )
            == len(handles),
        }

        phase = "控制路径"
        control_start = perf_counter_ns()

        previous_census = dict(trace.census_rows[-1])
        (
            before_census,
            before_states,
            before_inspections,
            introspection_ns,
        ) = read_only_runtime_introspection(
            runtime,
            client,
            candidates,
            handles,
            label=f"RQ6_{run_id}_BEFORE",
            ordinal=4 * group.allocation_round,
            previous_census=previous_census,
        )
        if not all(
            state["recurrent_resident"] and state["fa_resident"]
            for state in before_states.values()
        ):
            raise RQ4CorrectnessError("RQ6 barrier candidate 未全部驻留")

        timed = timed_flowstate_selection(snapshot_k)
        selected_ids = timed.selected_checkpoint_ids
        if selected_ids != reference_ids:
            raise RQ4CorrectnessError("RQ6 instrumentation 改变 selected set")

        reconciliation_start = perf_counter_ns()
        expected_evicted_ids = tuple(sorted(set(candidate_ids) - set(selected_ids)))
        trace_adapter = SequentialTraceRuntimeAdapter(
            client, handles, nonce_namespace=f"rq6:{run_id}"
        )
        recording_adapter = RecordingRuntimeAdapter(trace_adapter)
        controller = StateController(
            FrozenSelectedSetOptimizer(selected_ids), recording_adapter
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
            allocation=allocation, adapter=recording_adapter
        )
        after_census = runtime.census(
            f"rq6:{run_id}:after-reconcile",
            ordinal=4 * group.allocation_round,
            request=None,
            previous=before_census,
        )
        after_states, after_inspections = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"RQ6_{run_id}_AFTER",
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
        unexpected = trace_has_rematerialization(
            trace_adapter.trace_rows, expected_evicted_ids
        )
        reconciliation_ns = perf_counter_ns() - reconciliation_start
        total_control_ns = perf_counter_ns() - control_start
        reconcile_record = {
            "invariants": invariants,
            "unexpected_rematerialization": unexpected,
            "controller_report": controller_report,
            "trace_rows": trace_adapter.trace_rows,
            "before_inspection_count": len(before_inspections),
            "after_inspection_count": len(after_inspections),
        }
        if invariants["status"] != "PASS" or unexpected:
            raise RQ4CorrectnessError("RQ6 recurrent reconcile 正确性失败")

        reference_objective = evaluate_objective(snapshot_k, reference_ids)
        timed_objective = evaluate_objective(snapshot_k, selected_ids)
        heg_reference = [
            (
                row.target_tokens,
                row.executable_frontier_tokens,
                row.recovery_gap_tokens,
            )
            for row in reference_objective.per_continuation
        ]
        heg_timed = [
            (
                row.target_tokens,
                row.executable_frontier_tokens,
                row.recovery_gap_tokens,
            )
            for row in timed_objective.per_continuation
        ]
        heg_equivalent = heg_reference == heg_timed
        snapshot_immutable = frozen.content_digest() == frozen_digest
        correctness = {
            "handle_mapping_pass": mapping["complete"]
            and mapping["unique_runtime_identities"],
            "recurrent_residency_validation_pass": invariants.get(
                "selected_residency_exact"
            )
            is True,
            "fa_preserved": invariants.get("fa_residency_preserved") is True,
            "native_recurrent_eviction_zero": invariants.get(
                "native_mamba_capacity_eviction"
            )
            is False,
            "unexpected_rematerialization_zero": not unexpected,
            "fa_cascade_zero": invariants.get("fa_kv_cascade") is False,
            "oom_zero": True,
            "truncation_zero": True,
            "future_leakage_zero": not _future_leakage(boundary_audits),
            "fresh_engine_empty": fresh_engine_empty,
            "gpu_cleanup_pass": False,
        }
        passed_before_cleanup = all(
            value for key, value in correctness.items() if key != "gpu_cleanup_pass"
        ) and snapshot_immutable and heg_equivalent
        record.update(
            {
                "fresh_engine_empty": fresh_engine_empty,
                "timings_ns": {
                    "introspection": introspection_ns,
                    "construction": timed.construction_ns,
                    "allocation": timed.allocation_ns,
                    "reconciliation": reconciliation_ns,
                    "total_control": total_control_ns,
                },
                "selected_checkpoint_ids": list(selected_ids),
                "instrumentation_equivalent": selected_ids == reference_ids,
                "snapshot_immutable": snapshot_immutable,
                "heg_equivalent": heg_equivalent,
                "future_leakage": _future_leakage(boundary_audits),
                "correctness": correctness,
                "status": "PASS" if passed_before_cleanup else "INVALID",
            }
        )
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
        cleanup = {"stable": None, "deferred_to_parent": True}

    correctness = dict(record.get("correctness", {}))
    correctness["gpu_cleanup_pass"] = False
    record["correctness"] = correctness
    record["cross_run_contamination"] = not (fresh_engine_empty and gpu_cleanup)
    record["fatal_error"] = fatal_error
    record["shutdown_error"] = shutdown_error
    record["gpu_cleanup"] = cleanup
    if (
        fatal_error is not None
        or shutdown_error is not None
        or not correctness
        or not all(
            value
            for key, value in correctness.items()
            if key != "gpu_cleanup_pass"
        )
    ):
        record["status"] = "INVALID"
    else:
        record["status"] = "PASS_PENDING_CLEANUP"
    write_json(run_directory / "record.json", record)
    if reconcile_record is not None:
        write_json(run_directory / "reconciliation.json", reconcile_record)
    if replay_validation is not None:
        write_json(run_directory / "replay_validation.json", replay_validation)
    if mapping is not None:
        write_json(run_directory / "handle_mapping.json", mapping)
    return 0 if record["status"] == "PASS_PENDING_CLEANUP" else 1


def _worker_command(
    output_root: Path, formal_root: Path, run_id: str
) -> list[str]:
    """构造一个隔离 worker 的命令。"""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output-root",
        str(output_root),
        "--formal-root",
        str(formal_root),
        "--run-id",
        run_id,
    ]


def _execute_worker(
    output_root: Path,
    formal_root: Path,
    run_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """执行一个新进程，并在超时后只终止该 worker 进程组。"""
    import signal

    run_id = str(run_plan["run_id"])
    log_path = output_root / "logs" / f"runtime_{run_id}.log"
    process = subprocess.Popen(
        _worker_command(output_root, formal_root, run_id),
        stdout=log_path.open("w", encoding="utf-8"),
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
    record_path = output_root / "runtime_runs" / run_id / "record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
    else:
        record = _runtime_record_base(run_plan)
        record["fatal_error"] = {
            "phase": "worker process",
            "error": "worker 未生成 record.json",
        }
    try:
        cleanup = wait_gpu_stable(gpu_index=0)
    except Exception as error:
        cleanup = {"stable": False, "error": repr(error)}
    cleanup_pass = cleanup.get("stable") is True
    correctness = dict(record.get("correctness", {}))
    correctness["gpu_cleanup_pass"] = cleanup_pass
    record["correctness"] = correctness
    record["gpu_cleanup"] = cleanup
    record["cross_run_contamination"] = not (
        record.get("fresh_engine_empty", True) and cleanup_pass
    )
    record["worker_exit_code"] = exit_code
    record["worker_timed_out"] = timed_out
    if (
        timed_out
        or exit_code != 0
        or record.get("status") != "PASS_PENDING_CLEANUP"
        or not cleanup_pass
        or not correctness
        or not all(correctness.values())
    ):
        record["status"] = "INVALID"
    else:
        record["status"] = "PASS"
    if record_path.parent.exists():
        write_json(record_path, record)
        shutil.copyfile(log_path, record_path.parent / "runtime.log")
    return record


def run_parent(output_root: Path, formal_root: Path) -> int:
    """顺序执行正式 72-run 计划，并在首个无效样本后停止。"""
    plan_path = output_root / "provenance/runtime_run_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = build_runtime_plan(formal_root)
        write_json(plan_path, plan)
    source_before = source_manifest(source_paths())
    write_json(output_root / "provenance/source_before_runtime.json", source_before)
    completed: list[dict[str, Any]] = []
    try:
        initial = wait_gpu_stable(gpu_index=0)
    except Exception as error:
        initial = {"stable": False, "error": repr(error)}
    write_json(output_root / "correctness/initial_gpu_state.json", initial)
    if initial.get("stable") is not True:
        raise RuntimeError("RQ6 runtime 启动前 GPU 不满足空闲门禁")
    for ordinal, row in enumerate(plan, start=1):
        record = _execute_worker(output_root, formal_root, row)
        append_records(output_root / "raw/per_epoch_overhead.jsonl", [record])
        completed.append(record)
        print(
            f"RQ6 runtime 进度：{ordinal}/{len(plan)}，"
            f"run={row['run_id']}，status={record['status']}",
            flush=True,
        )
        if record["status"] != "PASS":
            break
    source_after = source_manifest(source_paths())
    integrity = {
        "status": "PASS" if source_before == source_after else "FAIL",
        "before": source_before,
        "after": source_after,
    }
    write_json(output_root / "provenance/source_after_runtime.json", source_after)
    write_json(output_root / "correctness/runtime_source_integrity.json", integrity)
    write_json(
        output_root / "correctness/runtime_collection_status.json",
        {
            "planned_runs": len(plan),
            "completed_runs": len(completed),
            "passed_runs": sum(row["status"] == "PASS" for row in completed),
            "source_integrity": integrity["status"],
            "status": (
                "PASS"
                if len(completed) == len(plan)
                and all(row["status"] == "PASS" for row in completed)
                and integrity["status"] == "PASS"
                else "FAIL"
            ),
        },
    )
    return 0 if len(completed) == len(plan) and all(
        row["status"] == "PASS" for row in completed
    ) and integrity["status"] == "PASS" else 1


def run_preflight(output_root: Path, formal_root: Path) -> int:
    """执行一个不进入正式 JSONL 的隔离诊断 run。"""
    if output_root.exists():
        raise FileExistsError(f"preflight 输出目录已存在：{output_root}")
    for relative in ("provenance", "runtime_runs", "logs"):
        (output_root / relative).mkdir(parents=True, exist_ok=False)
    plan = build_runtime_plan(formal_root)[:1]
    write_json(output_root / "provenance/runtime_run_plan.json", plan)
    record = _execute_worker(output_root, formal_root, plan[0])
    write_json(output_root / "preflight_result.json", record)
    print(record["status"], flush=True)
    return 0 if record["status"] == "PASS" else 1


def main(argv: Iterable[str] | None = None) -> int:
    """解析 parent/worker 参数并执行 runtime overhead。"""
    parser = argparse.ArgumentParser(description="RQ6 runtime 控制面开销")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.worker:
        if not args.run_id:
            raise ValueError("worker 模式必须提供 run ID")
        return _run_worker(
            args.output_root, args.formal_root, args.run_id
        )
    if args.preflight:
        return run_preflight(args.output_root, args.formal_root)
    return run_parent(args.output_root, args.formal_root)


if __name__ == "__main__":
    raise SystemExit(main())
