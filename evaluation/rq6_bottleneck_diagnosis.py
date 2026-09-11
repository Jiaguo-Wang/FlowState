"""复用冻结运行时重放，诊断控制往返、只读查询与驱逐开销。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
import traceback

from evaluation import rq6_runtime_overhead as base
from evaluation.rq6_system_overhead import summarize_values, spearman, write_json
from evaluation.openhands_sequential_eviction_rematerialization_audit import validate_sequential_trace


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evaluation/rq6_overhead_output/rq6_overhead_20260909_061500"
FORMAL = base.DEFAULT_FORMAL_ROOT


def timing_record(request, response, start, end):
    """分离客户端往返、服务端工作与剩余传输等待，拒绝负时间。"""
    timing = response["diagnostic_timing"]
    server = int(timing["server_total_ns"])
    total = end - start
    if not 0 <= server <= total:
        raise ValueError("服务端耗时超出客户端往返边界")
    return {
        "action": request.get("action", request["op"]),
        "roundtrip_ns": total,
        "server_ns": server,
        "outside_server_ns": total - server,
        "arrival_wait_ns": int(timing["server_started_ns"]) - start,
        "return_ns": end - int(timing["server_ended_ns"]),
        "profile": timing.get("profile"),
    }


def assert_read_only(before, after):
    """以全局树、驻留槽位和分配器事实证明只读阶段没有变更。"""
    for key in ("tree", "accounting"):
        if before[key] != after[key]:
            raise RuntimeError(f"只读阶段发生状态变更：{key}")


class MeasuredClient:
    """保留正式客户端接口，仅在标记阶段附加计时元数据。"""

    def __init__(self, client):
        self.client = client
        self.phase = None
        self.rows = []

    def _call(self, request):
        if self.phase is None:
            return self.client._call(request)
        request = {**request, "rq6d_measure": True}
        start = perf_counter_ns()
        request["client_sent_ns"] = start
        response = self.client._call(request)
        end = perf_counter_ns()
        row = timing_record(request, response, start, end)
        row["phase"] = self.phase
        self.rows.append(row)
        if request.get("action") == "inspect":
            assert_read_only(response["before"], response["after"])
        return response

    def census(self, nonce):
        return self._call({"op": "census", "nonce": nonce})

    def ping(self):
        return self.client.ping()

    def checkpoint_control(self, **kwargs):
        from array import array

        tokens = [int(value) for value in kwargs["token_ids"]]
        return self._call({"op": "checkpoint_control", **kwargs, "token_ids": tokens,
            "extra_key": kwargs.get("extra_key"),
            "expected_prefix_sha256": hashlib.sha256(array("q", tokens).tobytes()).hexdigest()})


def diagnostic_plan():
    """在查看延迟前按候选数与组编号确定三个规模，每个规模两个独立条件。"""
    frozen = json.loads((FROZEN / "provenance/runtime_run_plan.json").read_text())
    result = []
    for count in (8, 12, 20):
        row = min((r for r in frozen if r["candidate_count"] == count),
                  key=lambda r: (r["group_ordinal"], r["repetition"]))
        for mode in ("one", "multiple"):
            result.append({**row, "mode": mode, "run_id": f"g{row['group_ordinal']:03d}_{mode}"})
    return result


def reconcile(runtime, client, candidates, handles, pending, selected, before, states, label, budget_bytes=None):
    """按正式控制器和连续追踪路径执行一次诊断条件及完整验证。"""
    client.phase = label
    start = perf_counter_ns()
    adapter = base.SequentialTraceRuntimeAdapter(client, handles, nonce_namespace=label)
    recording = base.RecordingRuntimeAdapter(adapter)
    controller = base.StateController(base.FrozenSelectedSetOptimizer(selected), recording)
    if budget_bytes is None:
        budget_bytes = sum(c.memory_bytes for c in candidates)
    allocation = controller.reconcile(pending, candidates, handles, budget_bytes)
    adapter.finish()
    controller_ns = perf_counter_ns() - start
    report = base.build_controller_report(allocation=allocation, adapter=recording)
    after = runtime.census(label + ":after", ordinal=0, request=None, previous=before)
    after_states, _ = base.inspect_candidate_states(client, candidates, handles, phase=label)
    ids = tuple(c.checkpoint_id for c in candidates)
    evicted = tuple(sorted(set(ids) - set(selected)))
    invariants = base.evaluate_mapping_invariants(
        candidate_ids=ids, selected_ids=selected, expected_evicted_ids=evicted,
        handles=handles, before_states=states, after_states=after_states,
        before_census=before, after_census=after, controller_report=report,
    )
    unexpected = base.trace_has_rematerialization(adapter.trace_rows, evicted)
    if evicted:
        validate_sequential_trace(adapter.trace_rows, evicted, ids)
    if invariants["status"] != "PASS" or unexpected:
        raise RuntimeError("诊断驱逐正确性验证失败")
    elapsed = perf_counter_ns() - start
    client.phase = None
    return {
        "phase": label, "eviction_count": len(evicted), "elapsed_ns": elapsed,
        "budget_bytes": budget_bytes,
        "controller_and_trace_ns": controller_ns,
        "post_validation_ns": elapsed - controller_ns,
        "invariants": invariants, "unexpected_rematerialization": unexpected,
        "trace_rows": adapter.trace_rows, "controller_report": report,
    }, after, after_states


def worker(output, run_id):
    """在全新引擎中依次测量只读控制、保留全部与一个驱逐条件。"""
    from transformers import AutoTokenizer
    from targeted_probe import ControlClient
    from rq6_bottleneck_transport import RQ6BottleneckGateEngine, requested_control_port
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
    from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES

    plan = next(r for r in json.loads((output / "plan.json").read_text()) if r["run_id"] == run_id)
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    engine = None
    client = None
    record = {"plan": plan, "status": "INVALID"}
    try:
        manifest = json.loads((FORMAL / "population_manifest.json").read_text())
        group = base._group_from_manifest(manifest, plan["group_ordinal"])
        frozen = base._load_allocation_snapshot(Path(plan["snapshot_artifact"]))
        if frozen.content_digest() != plan["snapshot_digest"]:
            raise RuntimeError("冻结快照摘要不一致")
        tokenizer = AutoTokenizer.from_pretrained(base.TOKENIZER_PATH, local_files_only=True)
        messages = {label: base.load_session_messages(sid, base.DATASET_PATH)
                    for label, sid in group.session_by_label.items()}
        requests, audits = base.materialize_group_requests(
            tokenizer, messages, group=group, normalize_message=base.normalize_message,
            template_input_ids=base._template_input_ids)
        if base._future_leakage(audits):
            raise RuntimeError("物化边界包含未来信息")
        engine = RQ6BottleneckGateEngine(**manifest["engine_configuration"])
        client = MeasuredClient(ControlClient(requested_control_port()))
        wait_for_transport(client)
        runtime = base.SGLangGroupRuntime(engine, client)
        initial = client.census(run_id + ":initial")
        if initial["tree"]["mamba_node_count"] != 0:
            raise RuntimeError("新引擎含有历史循环状态")
        trace = base.replay_group_to_barrier(runtime, group, requests)
        trace = replace(trace, boundary_audit=tuple(audits))
        assembly = base.assemble_group_snapshot(trace, checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES)
        if assembly.status != "ELIGIBLE":
            raise RuntimeError("运行时重建不符合冻结快照")
        base.validate_rebuilt_snapshot(frozen, assembly.snapshot)
        handles = base.build_current_runtime_handles(trace, requests)
        candidates = frozen.core_candidates()
        ids = tuple(c.checkpoint_id for c in candidates)
        if set(handles) != set(ids) or len({h.expected_node_id for h in handles.values()}) != len(ids):
            raise RuntimeError("句柄映射不完整或不唯一")
        readonly_before = client.census(run_id + ":readonly-before")
        client.phase = "noop"
        for index in range(20):
            client._call({"op": "checkpoint_control", "action": "flowstate_rq6d_noop",
                          "nonce": f"{run_id}:noop:{index}"})
        client.phase = "introspection"
        before, states, _, intro_ns = base.read_only_runtime_introspection(
            runtime, client, candidates, handles, label=run_id + ":introspection",
            ordinal=0, previous_census=trace.census_rows[-1])
        client.phase = None
        readonly_after = client.census(run_id + ":readonly-after")
        assert_read_only(readonly_before, readonly_after)
        if not all(s["fa_resident"] and s["recurrent_resident"] for s in states.values()):
            raise RuntimeError("候选初始驻留不完整")
        zero, before, states = reconcile(runtime, client, candidates, handles,
            frozen.core_continuations(), ids, before, states, run_id + ":zero")
        assert_read_only(readonly_after, client.census(run_id + ":zero-proof"))
        formal_selected = tuple(plan["selected_candidate_ids"])
        formal_snapshot = base.create_budget_variant(frozen, plan["logical_k"])
        if base.reference_flowstate_selection(formal_snapshot) != formal_selected:
            raise RuntimeError("正式选择与冻结计划不一致")
        target = min(set(ids) - set(formal_selected))
        selected = tuple(i for i in ids if i != target) if plan["mode"] == "one" else formal_selected
        changed, _, _ = reconcile(runtime, client, candidates, handles,
            frozen.core_continuations(), selected, before, states, run_id + ":" + plan["mode"],
            budget_bytes=plan["budget_bytes"] if plan["mode"] == "multiple" else None)
        write_json(run_dir / "zero.json", zero)
        write_json(run_dir / "changed.json", changed)
        write_json(run_dir / "readonly_proof.json", {"before": readonly_before, "after": readonly_after})
        record.update({"status": "PASS_PENDING_CLEANUP", "read_only_no_mutation": True,
            "introspection_ns": intro_ns, "zero_ns": zero["elapsed_ns"],
            "changed_ns": changed["elapsed_ns"], "eviction_count": changed["eviction_count"],
            "correctness": changed["invariants"], "future_leakage": False,
            "fresh_engine_empty": True, "snapshot_immutable": frozen.content_digest() == plan["snapshot_digest"]})
    except Exception:
        record["traceback"] = traceback.format_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                record["shutdown_error"] = traceback.format_exc()
                record["status"] = "INVALID"
        if client is not None:
            write_json(run_dir / "rpc_timings.json", client.rows)
        write_json(run_dir / "record.json", record)
    return 0 if record["status"] == "PASS_PENDING_CLEANUP" else 1


def parent(output):
    """顺序执行预定义六次诊断，并在每次运行后检查设备清理。"""
    plan = json.loads((output / "plan.json").read_text())
    write_json(output / "initial_gpu.json", base.wait_gpu_stable(gpu_index=0))
    for row in plan:
        run_id = row["run_id"]
        with (output / f"{run_id}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "evaluation.rq6_bottleneck_diagnosis",
                "--output", str(output), "--worker", run_id], stdout=log, stderr=subprocess.STDOUT,
                timeout=1800)
        cleanup = base.wait_gpu_stable(gpu_index=0)
        path = output / "runs" / run_id / "record.json"
        record = json.loads(path.read_text()) if path.exists() else {"status": "INVALID"}
        record["gpu_cleanup"] = cleanup
        record["worker_exit_code"] = result.returncode
        if result.returncode == 0 and cleanup.get("stable") and record["status"] == "PASS_PENDING_CLEANUP":
            record["status"] = "PASS"
        else:
            record["status"] = "INVALID"
        write_json(path, record)
        print(f"诊断进度：{run_id} {record['status']}", flush=True)
        if record["status"] != "PASS":
            return 1
    return 0


def main():
    """区分离线准备、隔离父进程和单次诊断执行。"""
    parser = argparse.ArgumentParser(description="RQ6-D 运行时瓶颈诊断")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", diagnostic_plan())
        return 0
    if args.worker:
        return worker(args.output, args.worker)
    return parent(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
