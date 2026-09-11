"""核对冻结记录与独立诊断的消息计数、阶段耗时和相关关系。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from evaluation.rq6_bottleneck_diagnosis import FROZEN
from evaluation.rq6_system_overhead import summarize_values, spearman, write_json


def message_counts(candidate_count, eviction_count):
    """按真实调用图推导同步控制消息及路径快照调用次数。"""
    c, n = candidate_count, eviction_count
    if c < 1 or not 0 <= n <= c:
        raise ValueError("候选数或驱逐数非法")
    result = {
        "controller_rpc": 2 * c + 2 * n + 2,
        "scheduler_commands": 2 * c + 2 * n + 2,
        "tcp_connections": 2 * c + 2 * n + 2,
        "request_and_response_messages": 2 * (2 * c + 2 * n + 2),
        "per_candidate_inspect_rpc": 2 * c,
        "per_eviction_rpc": n,
        "s4_snapshot_rpc": n,
        "census_rpc": 2,
        "scope_validation_calls": 2 * c + 2 * n + 2,
        "trace_full_candidate_snapshots": 5 * n,
        "probe_path_snapshot_calls": 4 * c + 2 * n + 5 * n * c,
        "probe_global_tree_snapshot_calls": 4 * c + 2 * n + 2,
        "probe_accounting_snapshot_calls": 4 * c + 7 * n + 2,
        "event_wait_calls": 2 * c + 2 * n + 2,
        "adapter_pre_validation_calls": n,
        "adapter_post_validation_calls": n,
        "controller_mapping_invariant_calls": 1,
    }
    return result


def exclusive_breakdown(rows, elapsed_ns):
    """构造无嵌套重复计数的阶段分解，保留所有未归属耗时。"""
    profile = Counter()
    for row in rows:
        profile.update((row.get("profile") or {}).get("inclusive_ns", {}))
    total_rpc = sum(r["roundtrip_ns"] for r in rows)
    server = sum(r["server_ns"] for r in rows)
    path = profile["_path_snapshot"]
    tree = profile["_global_maps"]
    accounting = profile["_accounting_snapshot"]
    primitive = profile["adapter_primitive"]
    validation = profile["_validate_runtime_scope"]
    attributed = path + tree + accounting + primitive + validation
    # 子调用仅作为旁注，不与父级耗时重复相加。
    result = {
        "total_ms": elapsed_ns / 1e6,
        "pre_handler_transport_and_wait_ms": sum(r["arrival_wait_ns"] for r in rows) / 1e6,
        "post_handler_transport_and_wait_ms": sum(r["return_ns"] for r in rows) / 1e6,
        "worker_path_lookup_and_snapshot_ms": path / 1e6,
        "worker_global_tree_validation_snapshot_ms": tree / 1e6,
        "worker_allocator_snapshot_ms": accounting / 1e6,
        "worker_scope_validation_ms": validation / 1e6,
        "worker_adapter_primitive_ms": primitive / 1e6,
        "worker_other_ms": (server - attributed) / 1e6,
        "controller_outside_rpc_ms": (elapsed_ns - total_rpc) / 1e6,
        "nested_details_ms": {k: v / 1e6 for k, v in profile.items()},
        "rpc_count": len(rows),
        "worker_total_ms": server / 1e6,
        "outside_worker_handler_fraction": (total_rpc - server) / elapsed_ns,
        "closure_error_ns": elapsed_ns - (elapsed_ns - total_rpc + server
            + sum(r["outside_server_ns"] for r in rows)),
    }
    if all("pre_submit_ns" in row for row in rows):
        for key in ("pre_submit_ns", "queue_wait_ns", "event_return_ns", "post_submit_ns"):
            result[key.replace("_ns", "_ms")] = sum(row[key] for row in rows) / 1e6
    return result


def analyze(output):
    """逐条验证冻结 72-run，汇总诊断并保留未可识别的因果部分。"""
    records = [json.loads(line) for line in (FROZEN / "raw/per_epoch_overhead.jsonl").read_text().splitlines()]
    records = [r for r in records if r["record_kind"] == "runtime_control"]
    if len(records) != 72 or len({r["snapshot_id"] for r in records}) != 24:
        raise ValueError("冻结 population 数量不一致")
    formal_rows = []
    for r in records:
        evidence = json.loads((FROZEN / "runtime_runs" / r["run_id"] / "reconciliation.json").read_text())
        c = r["candidate_count"]
        n = len(evidence["controller_report"]["successful_eviction_ids"])
        if n != c - len(r["selected_checkpoint_ids"]) or len(evidence["trace_rows"]) != 5 * n:
            raise ValueError("冻结驱逐数量与 selected set 或 trace 不一致")
        if evidence["invariants"]["status"] != "PASS":
            raise ValueError("冻结正确性失败")
        formal_rows.append({"run_id": r["run_id"], "candidate_count": c,
            "handle_count": c, "eviction_count": n, **message_counts(c, n),
            "timings_ms": {k: v / 1e6 for k, v in r["timings_ns"].items()}})
    correlations = {}
    for stage in ("introspection", "reconciliation", "total_control"):
        correlations[stage] = {key: spearman([r[key] for r in formal_rows],
             [r["timings_ms"][stage] for r in formal_rows])
            for key in ("candidate_count", "eviction_count", "handle_count", "controller_rpc",
                        "scope_validation_calls", "probe_path_snapshot_calls")}
    by_c = {}
    for c in (8, 12, 16, 20):
        rows = [r for r in formal_rows if r["candidate_count"] == c]
        by_c[str(c)] = {"count": len(rows), "eviction_count": rows[0]["eviction_count"],
            "message_counts": message_counts(c, rows[0]["eviction_count"]),
            "timings_ms": {k: summarize_values([r["timings_ms"][k] for r in rows])
                           for k in ("introspection", "reconciliation", "total_control")}}
    write_json(output / "formal_message_counts.json", formal_rows)
    write_json(output / "formal_scaling.json", {"by_candidate_count": by_c,
        "spearman": correlations, "说明": "规模、句柄数、驱逐数与消息数单调共变，相关不能分离因果。"})

    diagnostic = []
    all_noops = []
    all_primitive = []
    for plan in json.loads((output / "plan.json").read_text()):
        path = output / "runs" / plan["run_id"]
        record = json.loads((path / "record.json").read_text())
        if record["status"] != "PASS":
            raise ValueError("诊断运行未通过")
        rows = json.loads((path / "rpc_timings.json").read_text())
        all_noops.extend(r["roundtrip_ns"] / 1e6 for r in rows if r["phase"] == "noop")
        breakdowns = {}
        for label, key in (("introspection", "introspection_ns"),
                           (plan["run_id"] + ":zero", "zero_ns"),
                           (plan["run_id"] + ":" + plan["mode"], "changed_ns")):
            selected = [r for r in rows if r["phase"] == label]
            breakdowns[key] = exclusive_breakdown(selected, record[key])
        for row in rows:
            if row["action"] == "flowstate_trace_evict_mamba_only":
                p = row["profile"]["inclusive_ns"]
                all_primitive.append({"run_id": plan["run_id"],
                    "detach_ms": p["_evict_component_and_detach_lru"] / 1e6,
                    "free_ms": p["_free_values"] / 1e6,
                    "remove_free_ms": (p["_evict_component_and_detach_lru"] + p["_free_values"]) / 1e6,
                    "primitive_including_checks_ms": p["adapter_primitive"] / 1e6,
                    "sanity_check_ms": p["sanity_check"] / 1e6})
        diagnostic.append({"run_id": plan["run_id"], "candidate_count": plan["candidate_count"],
            "mode": plan["mode"], "eviction_count": record["eviction_count"], "breakdowns": breakdowns})
    summary = {
        "status": "RQ6_RUNTIME_BOTTLENECK_PARTIAL",
        "noop_ms": summarize_values(all_noops),
        "primitive_ms": {k: summarize_values([r[k] for r in all_primitive]) for k in
            ("detach_ms", "free_ms", "remove_free_ms", "primitive_including_checks_ms", "sanity_check_ms")},
        "diagnostic_runs": diagnostic,
        "correctness": "PASS",
        "说明": "原语计时是工作线程主机调用时间；未插入 GPU 同步，不能声称设备异步工作完成时间。",
    }
    write_json(output / "primitive_timings.json", all_primitive)
    write_json(output / "diagnostic_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ6-D 冻结记录和运行时诊断分析")
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
