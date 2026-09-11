#!/usr/bin/env python3
"""在冻结 RQ6 runtime population 上执行批量控制优化及严格等价验证。"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter_ns
import traceback
from typing import Any, Mapping, Sequence

from evaluation import rq6_runtime_overhead as base
from evaluation.openhands_4workflow_occupancy_calibration import compact_census
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    compact_checkpoint_state,
)
from evaluation.rq3_frozen_snapshot_evaluator import evaluate_objective
from evaluation.rq6_system_overhead import (
    REPOSITORY_ROOT,
    append_records,
    overhead_summary,
    reference_flowstate_selection,
    source_manifest,
    source_paths,
    summarize_values,
    timed_flowstate_selection,
    write_json,
)


DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "evaluation/rq6_batched_control_output"
FROZEN_RQ6 = REPOSITORY_ROOT / "evaluation/rq6_overhead_output/rq6_overhead_20260909_061500"
FORMAL_ROOT = base.DEFAULT_FORMAL_ROOT
WORKER_TIMEOUT_S = 1800.0
BASELINE_MS = {
    "total_control": {"mean": 8398.847, "p95": 26811.052},
    "introspection": {"mean": 1913.301},
    "reconciliation": {"mean": 6483.256},
}


def _handle_rows(
    candidates: Sequence[Any], handles: Mapping[str, Any]
) -> list[dict[str, object]]:
    """按候选顺序序列化完整运行时句柄。"""
    rows = []
    for candidate in candidates:
        handle = handles[candidate.checkpoint_id]
        rows.append(
            {
                "checkpoint_id": handle.checkpoint_id,
                "token_ids": [int(value) for value in handle.token_ids],
                "extra_key": handle.extra_key,
                "expected_node_id": handle.expected_node_id,
                "expected_prefix_digest": handle.expected_prefix_digest,
            }
        )
    return rows


def _state_from_view(
    view: Mapping[str, Any], candidate_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    """把一次批量原始视图转换为原有映射门禁使用的候选状态。"""
    return {
        checkpoint_id: compact_checkpoint_state(
            {
                "after": {
                    "path": view["paths"][checkpoint_id],
                    "tree": view["tree"],
                    "accounting": view["accounting"],
                }
            }
        )
        for checkpoint_id in candidate_ids
    }


class BatchedControlClient:
    """用每阶段一次 RPC 调用批量运行时控制动作。"""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.rpc_count = 0

    def introspect(
        self, *, nonce: str, candidates: Sequence[Any], handles: Mapping[str, Any]
    ) -> dict[str, Any]:
        """读取一个一致的全候选 FA 与循环状态视图。"""
        self.rpc_count += 1
        return self._delegate._call(
            {
                "op": "checkpoint_control",
                "nonce": nonce,
                "action": "flowstate_batch_introspection",
                "candidate_ids": [item.checkpoint_id for item in candidates],
                "handles": _handle_rows(candidates, handles),
            }
        )

    def reconcile(
        self,
        *,
        nonce: str,
        candidates: Sequence[Any],
        handles: Mapping[str, Any],
        selected_ids: Sequence[str],
        expected_view_digest: str,
    ) -> dict[str, Any]:
        """一次提交完整 selected set 并执行批量协调与统一验证。"""
        self.rpc_count += 1
        return self._delegate._call(
            {
                "op": "checkpoint_control",
                "nonce": nonce,
                "action": "flowstate_batch_reconciliation",
                "candidate_ids": [item.checkpoint_id for item in candidates],
                "selected_ids": list(selected_ids),
                "expected_view_digest": expected_view_digest,
                "handles": _handle_rows(candidates, handles),
            }
        )


def _record_base(plan: Mapping[str, Any]) -> dict[str, Any]:
    """构造失败时也能审计的正式原始记录。"""
    return {
        "schema_version": "flowstate.rq6e_batched_control.v1",
        "record_kind": "runtime_control",
        "population": "RQ4-OpenHands",
        "run_id": plan["run_id"],
        "snapshot_id": plan["snapshot_id"],
        "snapshot_digest": plan["snapshot_digest"],
        "budget_ratio": 0.25,
        "k": plan["logical_k"],
        "pending_count": 4,
        "candidate_count": plan["candidate_count"],
        "timing_repetition": int(plan["repetition"]) - 1,
        "timings_ns": {name: None for name in (
            "introspection", "construction", "allocation", "reconciliation", "total_control"
        )},
        "selected_checkpoint_ids": [],
        "reference_selected_checkpoint_ids": list(plan["selected_candidate_ids"]),
        "instrumentation_equivalent": False,
        "snapshot_immutable": False,
        "heg_equivalent": False,
        "future_leakage": False,
        "cross_run_contamination": True,
        "rpc_counts": {},
        "correctness": {},
        "status": "INVALID",
    }


def _load_plan(output: Path, run_id: str, smoke: bool) -> dict[str, Any]:
    """读取当前模式中唯一匹配的运行计划。"""
    path = output / ("smoke/plan.json" if smoke else "provenance/runtime_run_plan.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    matched = [row for row in rows if row["run_id"] == run_id]
    if len(matched) != 1:
        raise RuntimeError(f"批量运行计划不唯一：{run_id}")
    return dict(matched[0])


def _heg(snapshot: Any, selected_ids: Sequence[str]) -> list[tuple[int, int, int]]:
    """按冻结 common objective 提取 H/E/G 等价三元组。"""
    result = evaluate_objective(snapshot, selected_ids)
    return [
        (
            row.target_tokens,
            row.executable_frontier_tokens,
            row.recovery_gap_tokens,
        )
        for row in result.per_continuation
    ]


def _run_worker(output: Path, run_id: str, smoke: bool) -> int:
    """在独立 Engine 生命周期中执行一个批量控制 epoch。"""
    from transformers import AutoTokenizer
    from targeted_probe import ControlClient
    from rq6_batched_control_transport import (
        RQ6BatchedControlEngine,
        requested_control_port,
    )
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
    from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES

    plan = _load_plan(output, run_id, smoke)
    parent = output / ("smoke/runs" if smoke else "runtime_runs")
    run_directory = parent / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    record = _record_base(plan)
    engine = None
    runtime = None
    replay_validation = None
    fatal_error = None
    shutdown_error = None
    fresh_engine_empty = False
    phase = "准备"
    try:
        manifest = json.loads(
            (FORMAL_ROOT / "population_manifest.json").read_text(encoding="utf-8")
        )
        group = base._group_from_manifest(manifest, int(plan["group_ordinal"]))
        snapshot_path = base._snapshot_path_for_group(FORMAL_ROOT, group.group_ordinal)
        frozen = base._load_allocation_snapshot(snapshot_path)
        frozen_digest = frozen.content_digest()
        if frozen_digest != plan["snapshot_digest"]:
            raise RuntimeError("冻结 snapshot 摘要不一致")
        snapshot_k = base.create_budget_variant(
            frozen, base.rq4_logical_k(len(frozen.eligible_candidates))
        )
        reference_ids = reference_flowstate_selection(snapshot_k)
        if list(reference_ids) != plan["selected_candidate_ids"]:
            raise RuntimeError("冻结 selected set 不一致")

        tokenizer = AutoTokenizer.from_pretrained(
            base.TOKENIZER_PATH, local_files_only=True
        )
        messages = {
            label: base.load_session_messages(session_id, base.DATASET_PATH)
            for label, session_id in group.session_by_label.items()
        }
        requests, audits = base.materialize_group_requests(
            tokenizer,
            messages,
            group=group,
            normalize_message=base.normalize_message,
            template_input_ids=base._template_input_ids,
        )
        if base._future_leakage(audits):
            raise RuntimeError("请求物化读取了未来信息")

        phase = "Engine 初始化"
        engine = RQ6BatchedControlEngine(**dict(manifest["engine_configuration"]))
        raw_client = ControlClient(requested_control_port())
        wait_for_transport(raw_client)
        runtime = base.SGLangGroupRuntime(engine, raw_client)
        initial = runtime.census(
            f"rq6e:{run_id}:initial", ordinal=0, request=None, previous=None
        )
        fresh_engine_empty = int(initial["mamba_node_count"]) == 0
        if not fresh_engine_empty:
            raise RuntimeError("全新 Engine 含跨运行循环状态")

        phase = "冻结 barrier replay"
        trace = base.replay_group_to_barrier(runtime, group, requests)
        trace = replace(trace, boundary_audit=tuple(audits))
        assembly = base.assemble_group_snapshot(
            trace, checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES
        )
        if assembly.status != "ELIGIBLE" or assembly.snapshot is None:
            raise RuntimeError(f"运行时重建失败：{assembly.primary_reason}")
        replay_validation = base.validate_rebuilt_snapshot(frozen, assembly.snapshot)
        handles = base.build_current_runtime_handles(trace, requests)
        candidates = snapshot_k.core_candidates()
        candidate_ids = tuple(item.checkpoint_id for item in candidates)
        handle_mapping = bool(
            set(handles) == set(candidate_ids)
            and len({
                (item.expected_node_id, item.expected_prefix_digest)
                for item in handles.values()
            }) == len(handles)
        )
        if not handle_mapping:
            raise RuntimeError("运行时句柄映射不完整或不唯一")

        phase = "批量控制路径"
        client = BatchedControlClient(raw_client)
        control_start = perf_counter_ns()
        introspection_start = perf_counter_ns()
        introspection = client.introspect(
            nonce=f"rq6e:{run_id}:introspection",
            candidates=candidates,
            handles=handles,
        )
        before_states = _state_from_view(introspection["view"], candidate_ids)
        before_census = compact_census(
            {"tree": introspection["view"]["tree"],
             "accounting": introspection["view"]["accounting"]},
            ordinal=4 * group.allocation_round,
            request=None,
            previous=trace.census_rows[-1],
        )
        introspection_ns = perf_counter_ns() - introspection_start
        if introspection.get("state_mutated") is not False:
            raise RuntimeError("批量 introspection 未证明只读")
        if not all(
            row["fa_resident"] and row["recurrent_resident"]
            for row in before_states.values()
        ):
            raise RuntimeError("barrier 候选未全部驻留")

        timed = timed_flowstate_selection(snapshot_k)
        selected_ids = timed.selected_checkpoint_ids
        if selected_ids != reference_ids:
            raise RuntimeError("批量路径改变 selected set")

        reconciliation_start = perf_counter_ns()
        reconciliation = client.reconcile(
            nonce=f"rq6e:{run_id}:reconcile",
            candidates=candidates,
            handles=handles,
            selected_ids=selected_ids,
            expected_view_digest=introspection["view_digest"],
        )
        after_states = _state_from_view(reconciliation["after"], candidate_ids)
        after_census = compact_census(
            {"tree": reconciliation["after"]["tree"],
             "accounting": reconciliation["after"]["accounting"]},
            ordinal=4 * group.allocation_round,
            request=None,
            previous=before_census,
        )
        expected_evicted_ids = tuple(sorted(set(candidate_ids) - set(selected_ids)))
        proof = reconciliation["proof"]
        actual_retained = tuple(sorted(
            checkpoint_id for checkpoint_id, row in after_states.items()
            if row["recurrent_resident"]
        ))
        actual_evicted = tuple(sorted(
            checkpoint_id for checkpoint_id, row in before_states.items()
            if row["recurrent_resident"] and not after_states[checkpoint_id]["recurrent_resident"]
        ))
        unified_validation = bool(
            reconciliation["before_view_digest"] == introspection["view_digest"]
            and tuple(reconciliation["selected_ids"]) == selected_ids
            and tuple(reconciliation["evicted_ids"]) == expected_evicted_ids
            and tuple(reconciliation["completed_eviction_ids"]) == expected_evicted_ids
            and actual_retained == tuple(sorted(selected_ids))
            and actual_evicted == expected_evicted_ids
            and proof["status"] == "PASS"
            and reconciliation["post_validation_count"] == 1
        )
        reconciliation_ns = perf_counter_ns() - reconciliation_start
        total_control_ns = perf_counter_ns() - control_start
        if not unified_validation:
            raise RuntimeError("批量协调客户端统一验证失败")

        heg_reference = _heg(snapshot_k, reference_ids)
        heg_batched = _heg(snapshot_k, selected_ids)
        heg_equivalent = heg_reference == heg_batched
        snapshot_immutable = frozen.content_digest() == frozen_digest
        correctness = {
            "handle_mapping_pass": handle_mapping,
            "recurrent_residency_validation_pass": unified_validation,
            "fa_preserved": proof["fa_preserved"] is True,
            "native_recurrent_eviction_zero": proof["native_recurrent_eviction"] is False,
            "unexpected_rematerialization_zero": proof["unexpected_rematerialization"] is False,
            "fa_cascade_zero": proof["fa_cascade"] is False,
            "oom_zero": not trace.oom,
            "truncation_zero": not trace.truncation,
            "future_leakage_zero": not base._future_leakage(audits),
            "fresh_engine_empty": fresh_engine_empty,
            "gpu_cleanup_pass": False,
        }
        passed_before_cleanup = bool(
            all(value for key, value in correctness.items() if key != "gpu_cleanup_pass")
            and selected_ids == reference_ids
            and heg_equivalent
            and snapshot_immutable
            and client.rpc_count == 2
        )
        record.update(
            {
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
                "heg_reference": heg_reference,
                "heg_batched": heg_batched,
                "future_leakage": False,
                "fresh_engine_empty": fresh_engine_empty,
                "rpc_counts": {
                    "before": 2 * len(candidate_ids) + 2 * len(expected_evicted_ids) + 2,
                    "after": client.rpc_count,
                    "batched_introspection": 1,
                    "batched_reconciliation": 1,
                    "worker_internal_evictions": len(expected_evicted_ids),
                    "post_validation": 1,
                },
                "correctness": correctness,
                "status": "PASS_PENDING_CLEANUP" if passed_before_cleanup else "INVALID",
            }
        )
        write_json(run_directory / "batch_introspection.json", introspection)
        write_json(run_directory / "batch_reconciliation.json", reconciliation)
        write_json(run_directory / "before_states.json", before_states)
        write_json(run_directory / "after_states.json", after_states)
        write_json(run_directory / "before_census.json", before_census)
        write_json(run_directory / "after_census.json", after_census)
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
    record["fatal_error"] = fatal_error
    record["shutdown_error"] = shutdown_error
    record["gpu_cleanup"] = {"stable": None, "deferred_to_parent": True}
    record["cross_run_contamination"] = True
    if fatal_error is not None or shutdown_error is not None:
        record["status"] = "INVALID"
    write_json(run_directory / "record.json", record)
    if replay_validation is not None:
        write_json(run_directory / "replay_validation.json", replay_validation)
    return 0 if record["status"] == "PASS_PENDING_CLEANUP" else 1


def _worker_command(output: Path, run_id: str, smoke: bool) -> list[str]:
    """构造隔离 worker 命令。"""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output", str(output), "--run-id", run_id]
    if smoke:
        command.append("--smoke-worker")
    return command


def _execute_worker(output: Path, plan: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    """执行一个隔离 worker，并由父进程验证 GPU 清理。"""
    import signal

    run_id = str(plan["run_id"])
    log_parent = output / ("smoke/logs" if smoke else "logs")
    run_parent = output / ("smoke/runs" if smoke else "runtime_runs")
    log_path = log_parent / f"{run_id}.log"
    process = subprocess.Popen(
        _worker_command(output, run_id, smoke),
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
    record_path = run_parent / run_id / "record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else _record_base(plan)
    try:
        cleanup = base.wait_gpu_stable(gpu_index=0)
    except Exception as error:
        cleanup = {"stable": False, "error": repr(error)}
    correctness = dict(record.get("correctness") or {})
    correctness["gpu_cleanup_pass"] = cleanup.get("stable") is True
    record.update({
        "correctness": correctness,
        "gpu_cleanup": cleanup,
        "cross_run_contamination": not (
            record.get("fresh_engine_empty") is True and cleanup.get("stable") is True
        ),
        "worker_exit_code": exit_code,
        "worker_timed_out": timed_out,
    })
    if (
        timed_out or exit_code != 0
        or record.get("status") != "PASS_PENDING_CLEANUP"
        or not correctness or not all(correctness.values())
        or record["cross_run_contamination"]
    ):
        record["status"] = "INVALID"
    else:
        record["status"] = "PASS"
    if record_path.parent.exists():
        write_json(record_path, record)
        shutil.copyfile(log_path, record_path.parent / "runtime.log")
    return record


def initialize(output: Path) -> None:
    """创建新的、不会覆盖任何冻结结果的 RQ6-E 目录。"""
    if output.exists():
        raise FileExistsError(f"RQ6-E 输出目录已存在：{output}")
    for relative in (
        "smoke/runs", "smoke/logs", "provenance", "runtime_runs", "logs",
        "correctness", "analysis", "raw",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    plan = base.build_runtime_plan(FORMAL_ROOT)
    write_json(output / "provenance/runtime_run_plan.json", plan)
    write_json(output / "smoke/plan.json", [dict(plan[0])])
    write_json(output / "provenance/protocol.json", {
        "task": "RQ6-E Batched Runtime Control Optimization",
        "formal_population": {"snapshots": 24, "repetitions": 3, "runs": 72},
        "timed_rpc_per_epoch": 2,
        "introspection": "一次一致性只读视图，含全候选FA与循环状态",
        "reconciliation": "一次selected-set提交，内部调用冻结单状态原语，完成后一次统一验证",
        "frozen_reference": str(FROZEN_RQ6),
        "semantic_changes": False,
        "future_information": False,
        "kimi_started": False,
    })
    write_json(output / "provenance/source_before.json", _source_manifest())


def _source_manifest() -> dict[str, Any]:
    """同时覆盖冻结核心与本轮批量实现源码。"""
    paths = source_paths() + (
        REPOSITORY_ROOT / "evaluation/rq6_batched_control.py",
        REPOSITORY_ROOT / "tests/runtime/rq6_batched_control_transport.py",
    )
    return source_manifest(paths)


def run_smoke(output: Path) -> int:
    """执行一个小规模隔离 smoke，失败时禁止正式采集。"""
    plan = json.loads((output / "smoke/plan.json").read_text())
    initial = base.wait_gpu_stable(gpu_index=0)
    write_json(output / "smoke/initial_gpu_state.json", initial)
    if initial.get("stable") is not True:
        raise RuntimeError("smoke 前 GPU 不满足空闲门禁")
    record = _execute_worker(output, plan[0], True)
    write_json(output / "smoke/result.json", record)
    write_json(output / "smoke/status.json", {
        "status": "PASS" if record["status"] == "PASS" else "FAIL",
        "run_id": record["run_id"],
    })
    return 0 if record["status"] == "PASS" else 1


def run_formal(output: Path) -> int:
    """smoke 通过后顺序执行 72 个独立正式 run，失败即停止。"""
    smoke = json.loads((output / "smoke/status.json").read_text())
    if smoke.get("status") != "PASS":
        raise RuntimeError("smoke 未通过，禁止启动正式采集")
    plan = json.loads((output / "provenance/runtime_run_plan.json").read_text())
    source_before = json.loads((output / "provenance/source_before.json").read_text())
    completed = []
    write_json(output / "correctness/initial_gpu_state.json", base.wait_gpu_stable(gpu_index=0))
    for ordinal, row in enumerate(plan, start=1):
        record = _execute_worker(output, row, False)
        append_records(output / "raw/per_epoch_overhead.jsonl", [record])
        completed.append(record)
        print(f"RQ6-E 正式进度：{ordinal}/{len(plan)}，run={row['run_id']}，status={record['status']}", flush=True)
        if record["status"] != "PASS":
            break
    source_after = _source_manifest()
    source_ok = source_before == source_after
    write_json(output / "provenance/source_after.json", source_after)
    write_json(output / "correctness/source_integrity.json", {
        "status": "PASS" if source_ok else "FAIL",
        "before": source_before, "after": source_after,
    })
    status = bool(
        len(completed) == 72 and all(row["status"] == "PASS" for row in completed)
        and source_ok
    )
    write_json(output / "correctness/collection_status.json", {
        "planned_runs": 72, "completed_runs": len(completed),
        "passed_runs": sum(row["status"] == "PASS" for row in completed),
        "source_integrity": "PASS" if source_ok else "FAIL",
        "status": "PASS" if status else "FAIL",
    })
    return 0 if status else 1


def analyze(output: Path) -> dict[str, Any]:
    """汇总 before/after 性能、消息数与全部 fail-closed 语义门禁。"""
    records = [json.loads(line) for line in (output / "raw/per_epoch_overhead.jsonl").read_text().splitlines()]
    if len(records) != 72 or not all(row["status"] == "PASS" for row in records):
        raise RuntimeError("正式 72-run 尚未全部通过")
    if len({row["snapshot_id"] for row in records}) != 24:
        raise RuntimeError("正式 snapshot 数量不是24")
    summary = overhead_summary(records)["by_kind"]["runtime_control"]["stages_ms"]
    before_rpc = sum(int(row["rpc_counts"]["before"]) for row in records)
    after_rpc = sum(int(row["rpc_counts"]["after"]) for row in records)
    gates = {
        "formal_runs": len(records) == 72,
        "selected_set_equivalence": all(row["instrumentation_equivalent"] for row in records),
        "heg_equivalence": all(row["heg_equivalent"] for row in records),
        "snapshot_immutable": all(row["snapshot_immutable"] for row in records),
        "fa_preservation": all(row["correctness"]["fa_preserved"] for row in records),
        "recurrent_residency": all(row["correctness"]["recurrent_residency_validation_pass"] for row in records),
        "no_native_recurrent_eviction": all(row["correctness"]["native_recurrent_eviction_zero"] for row in records),
        "no_unexpected_rematerialization": all(row["correctness"]["unexpected_rematerialization_zero"] for row in records),
        "no_fa_cascade": all(row["correctness"]["fa_cascade_zero"] for row in records),
        "no_future_leakage": all(row["correctness"]["future_leakage_zero"] for row in records),
        "no_cross_run_contamination": all(not row["cross_run_contamination"] for row in records),
        "gpu_cleanup": all(row["correctness"]["gpu_cleanup_pass"] for row in records),
        "two_rpc_per_epoch": all(row["rpc_counts"]["after"] == 2 for row in records),
    }
    reduction = {
        stage: {
            "before_mean_ms": BASELINE_MS[stage]["mean"],
            "after_mean_ms": summary[stage]["mean"],
            "reduction_percent": 100.0 * (BASELINE_MS[stage]["mean"] - summary[stage]["mean"]) / BASELINE_MS[stage]["mean"],
            "speedup": BASELINE_MS[stage]["mean"] / summary[stage]["mean"],
        }
        for stage in ("total_control", "introspection", "reconciliation")
    }
    result = {
        "status": "RQ6_BATCHED_CONTROL_READY" if all(gates.values()) else "RQ6_BATCHED_CONTROL_BLOCKED",
        "snapshots": 24, "formal_runs": 72,
        "stages_ms": summary,
        "rpc": {
            "before_total": before_rpc, "after_total": after_rpc,
            "before_mean_per_epoch": before_rpc / 72,
            "after_mean_per_epoch": after_rpc / 72,
            "reduction_percent": 100.0 * (before_rpc - after_rpc) / before_rpc,
        },
        "performance": reduction, "gates": gates,
    }
    write_json(output / "analysis/aggregate_results.json", result)
    lines = [
        "# RQ6-E 批量运行时控制优化", "",
        f"状态：{result['status']}。24个冻结snapshot的72/72个独立run全部通过。", "",
        "## 性能", "",
        f"总控制路径由{reduction['total_control']['before_mean_ms']:.3f} ms降至{reduction['total_control']['after_mean_ms']:.3f} ms，降低{reduction['total_control']['reduction_percent']:.4f}%，加速{reduction['total_control']['speedup']:.3f}倍。",
        f"批量introspection均值{summary['introspection']['mean']:.3f} ms、P95 {summary['introspection']['p95']:.3f} ms；批量reconciliation均值{summary['reconciliation']['mean']:.3f} ms、P95 {summary['reconciliation']['p95']:.3f} ms。",
        f"同步RPC总数由{before_rpc}降至{after_rpc}，每epoch均值由{before_rpc / 72:.3f}降至2。", "",
        "## 语义与正确性", "",
        "selected set、H/E/G、快照摘要、循环状态驻留和FA状态全部等价；无原生循环状态驱逐、意外重驻留、FA级联、未来信息或跨运行污染。每个epoch只在批量协调后执行一次统一重型验证。", "",
    ]
    (output / "analysis/final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    """解析阶段命令，确保 smoke 与正式采集顺序不可绕过。"""
    parser = argparse.ArgumentParser(description="RQ6-E 批量运行时控制实验")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_PARENT / f"rq6e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.initialize:
        initialize(output)
        print(output)
        return 0
    if args.worker:
        if not args.run_id:
            parser.error("worker 必须提供 --run-id")
        return _run_worker(output, args.run_id, args.smoke_worker)
    if args.smoke:
        return run_smoke(output)
    if args.formal:
        return run_formal(output)
    if args.analyze:
        analyze(output)
        return 0
    parser.error("必须指定一个执行阶段")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
