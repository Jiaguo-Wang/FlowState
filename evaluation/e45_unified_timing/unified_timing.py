# -*- coding: utf-8 -*-
"""E4/E5 统一性能与 E2E 计时离线汇总（只读，不运行 GPU，不修改任何冻结产物）。

复用两套已冻结的严格 Runtime Runner 原始记录：
- E4-1B：evaluation/native_resume_performance_e4_phase1b（A/C × Greedy/Exact × 3 ＋ Cold ×2 ＋ Smoke ×4）
- E5-1 ：evaluation/exact_swap_runtime_e5_step1（A/B/C/D × Exact/Swap × 3 ＋ Cold ×4 ＋ Smoke ×8）

统一口径（全部为“单个独立 Engine 生命周期内”的客户端单调钟端点差）：
- T0 控制RPC发起；T1 控制RPC返回（内含观察、Strict认证、求解、Plan验证、协调及必要后验）
- T2 目标请求提交；T3 首个非空流式输出到达；T4 流结束
- Control Time = T1−T0（control_rpc_wall_ms，含客户端证据持久化，非纯服务端控制成本）
- Request TTFT = T3−T2；Epoch Time（Control-inclusive）= T4−T0
- 前置重型参考审计为独立区间 T0a→T1a（reference RPC），另计不并入 Control Time；
  Audit-inclusive Epoch = T4−T0a 单独成列
- CUDA Prefill / COW CUDA 取请求内既有 CUDA 事件（trace.compute 的 is_extend 行、trace.copies），
  不用客户端 Wall Time 替代；CUDA 事件区间含提交空隙，不是纯内核之和
- Cold 臂按设计没有控制窗口：T0/T1 缺席，一律标记 control_window=ABSENT，留空不推算
- 每行 = 一个独立 Engine 生命周期；任何跨行加总都不是“一个 Epoch”，只允许描述性统计

用法：
  /home/wjg/uv312/bin/python -I evaluation/e45_unified_timing/unified_timing.py [更多实验目录 ...]
  附加目录需包含 experiment_protocol.json 与同构运行子目录；标签取目录名。
输出（写入本目录）：unified_timing_rows.csv、unified_timing_stats.csv、unified_timing_verification.json
"""
import csv
import json
import statistics
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]

# 默认纳入的实验目录；后续正式 GPU 实验追加于此或经命令行传入
EXPERIMENTS = [
    ("E4-1B", ROOT / "evaluation/native_resume_performance_e4_phase1b"),
    ("E5-1", ROOT / "evaluation/exact_swap_runtime_e5_step1"),
]

# 控制 RPC 内七个互不嵌套阶段（与 reconciliation.json timing 一致；不可重复相加）
STAGES = [
    "metadata_observation",
    "strict_epoch_authentication",
    "benefit_matrix",
    "solver",
    "plan_build_validate",
    "apply_time_freshness_preflight",
    "strict_reconciliation_with_postchecks",
]

# 汇总统计只覆盖正式测量臂的五项统一指标 ＋ 请求 Wall（均 ms）
CORE_METRICS = [
    "control_time_ms",
    "cuda_prefill_ms",
    "request_ttft_ms",
    "epoch_time_control_inclusive_ms",
    "audit_inclusive_epoch_ms",
    "request_wall_ms",
]


def load(path):
    return json.loads(Path(path).read_text())


def num(x):
    """CSV 数值列统一格式；None/缺失写空串，绝不伪造成 0。"""
    return "" if x is None else x


def extract_run(exp_dir, item):
    """从单个运行目录抽取统一指标行；返回 (row, checks)。

    checks 为本脚本的独立复核结论（区别于各实验已有 finalize/review 门禁）：
    只做端点代数一致性、原始 ns 与派生 ms 对账、执行证据计数等只读检查。
    """
    d = Path(exp_dir) / item["name"]
    row = {
        "run": item["name"],
        "phase": item["phase"],
        "pair": item.get("pair", ""),
        "solver": item["solver"],
        "pending": item.get("pending", ""),
        "status": "MISSING",
        "control_window": "",
        "notes": "",
    }
    checks = []
    if (d / "failure.json").exists():
        row["status"] = "BLOCKED"
        row["notes"] = "failure.json 存在，详见该运行目录"
        _fill_blank(row)
        return row, checks
    if not (d / "summary.json").exists():
        _fill_blank(row)
        return row, checks
    s = load(d / "summary.json")
    row["status"] = s.get("status", "UNKNOWN")
    m = s.get("native_match", {})
    row["H_physical_fa_hit"] = m.get("physical_fa_hit")
    row["E_executable_prefix"] = m.get("executable_prefix")
    row["G_replay_gap"] = m.get("replay_gap")
    row["input_token_count"] = m.get("input_token_count")
    row["request_ttft_ms"] = s.get("ttft_ms")
    row["request_wall_ms"] = s.get("request_wall_ms")
    cold = s.get("arm") == "cold"
    if cold:
        # Cold 基线：同一 T2→T4 请求边界；控制窗口按设计不存在，不推算 T0/T1
        row["control_window"] = "ABSENT(cold)"
        row["control_time_ms"] = None
        row["epoch_time_control_inclusive_ms"] = None
        row["audit_inclusive_epoch_ms"] = None
        row["notes"] = "无控制窗口：Control Time/Epoch Time 不适用，非缺失故障"
    else:
        row["control_window"] = "MEASURED"
        row["control_time_ms"] = s.get("control_rpc_wall_ms")
        row["epoch_time_control_inclusive_ms"] = s.get("control_through_completion_ms")
        row["audit_inclusive_epoch_ms"] = s.get("audit_inclusive_through_completion_ms")
        row["reference_audit_rpc_client_ms"] = s.get("reference_observation_rpc_wall_ms")

    # trace：既有 CUDA 事件与外层区间（大文件只取 trace 段需要的字段）
    tr = load(d / "post_request.json")["trace"]
    comp = tr.get("compute", [])
    execn = tr.get("execution", [])
    copies = tr.get("copies", [])
    row["cuda_prefill_ms"] = sum(x["cuda_elapsed_ms"] for x in comp if x["is_extend"])
    row["decode_cuda_ms"] = sum(x["cuda_elapsed_ms"] for x in comp if not x["is_extend"])
    row["cow_cuda_ms"] = sum(x["cuda_elapsed_ms"] for x in copies) if copies else None
    first_decode = next((e for e in execn if not e["is_extend"]), None)
    if first_decode is not None:
        row["first_decode_outer_wall_ms"] = (first_decode["ns_after"] - first_decode["ns_before"]) / 1e6
        row["first_decode_inner_eager_cuda_ms"] = next(
            (c["cuda_elapsed_ms"] for c in comp if not c["is_extend"]), None
        )
    row["decode_window_ms"] = (
        row["request_wall_ms"] - row["request_ttft_ms"]
        if row.get("request_wall_ms") is not None and row.get("request_ttft_ms") is not None
        else None
    )
    # 实际 Prefill Token（独立复算：外层 extend 行计数 ＝ 输入长 − 可执行前缀）
    if row.get("input_token_count") is not None and row.get("E_executable_prefix") is not None:
        row["actual_prefill_tokens"] = sum(
            e["extend_token_count"] for e in execn if e["is_extend"]
        )
        checks.append(
            (
                "actual_prefill_consistent",
                row["actual_prefill_tokens"]
                == row["input_token_count"] - row["E_executable_prefix"],
            )
        )

    if cold:
        row.update({k: None for k in (
            "reference_audit_rpc_server_ms",
            "stage_observation_ms", "stage_epoch_authentication_ms", "stage_benefit_matrix_ms",
            "stage_solver_ms", "stage_plan_build_validate_ms", "stage_apply_preflight_ms",
            "stage_reconciliation_ms", "stage_sum_ms", "rpc_unattributed_ms",
            "control_to_request_gap_ms", "reference_to_control_gap_ms",
        )})
        checks.append(("execution_rows_complete", len(execn) == len(comp) == 8))
        return row, checks

    # ---- 以下仅控制臂：阶段拆分与端点代数 ----
    r = load(d / "reconciliation.json")
    stage = {x["stage"]: x["ms"] for x in r.get("timing", [])}
    checks.append(("seven_stages_present", all(k in stage for k in STAGES)))
    row["stage_observation_ms"] = stage.get("metadata_observation")
    row["stage_epoch_authentication_ms"] = stage.get("strict_epoch_authentication")
    row["stage_benefit_matrix_ms"] = stage.get("benefit_matrix")
    row["stage_solver_ms"] = stage.get("solver")
    row["stage_plan_build_validate_ms"] = stage.get("plan_build_validate")
    row["stage_apply_preflight_ms"] = stage.get("apply_time_freshness_preflight")
    row["stage_reconciliation_ms"] = stage.get("strict_reconciliation_with_postchecks")
    row["stage_sum_ms"] = sum(stage[k] for k in STAGES if k in stage)
    # RPC 未归因余额：端点差额，不分解臆测（详见 e4_control_timing_audit）
    row["rpc_unattributed_ms"] = row["control_time_ms"] - row["stage_sum_ms"]
    # 参考审计服务端自测区间（与客户端 RPC Wall 分列，不相减得出任何“纯”成本）
    row["reference_audit_rpc_server_ms"] = load(d / "reference_audit.json").get("wall_ms")

    # 端点代数：T4−T0 = (T1−T0) + (T2−T1) + (T4−T2)；T4−T0a = (T1a−T0a) + (T0−T1a) + (T4−T0)
    tol = 1e-3
    gap1 = row["epoch_time_control_inclusive_ms"] - row["control_time_ms"] - row["request_wall_ms"]
    row["control_to_request_gap_ms"] = gap1
    checks.append(("epoch_algebra_nonnegative_gap", gap1 >= -tol))
    gap2 = (
        row["audit_inclusive_epoch_ms"]
        - row["reference_audit_rpc_client_ms"]
        - row["epoch_time_control_inclusive_ms"]
    )
    row["reference_to_control_gap_ms"] = gap2
    checks.append(("audit_algebra_nonnegative_gap", gap2 >= -tol))

    # 原始 ns 与派生 ms 对账（stream_chunks 与 summary 同源单调钟，应严格一致）
    sc = load(d / "stream_chunks.json")
    ttft_ns = (sc["first_token_ns"] - sc["start_ns"]) / 1e6
    wall_ns = (sc["end_ns"] - sc["start_ns"]) / 1e6
    checks.append(("ttft_ns_agrees_summary", abs(ttft_ns - row["request_ttft_ms"]) < tol))
    checks.append(("wall_ns_agrees_summary", abs(wall_ns - row["request_wall_ms"]) < tol))
    # 顺序：T2 ≤ T3 ≤ T4；TTFT 不超过请求 Wall
    checks.append(("ordering_ttft_within_wall", 0 <= row["request_ttft_ms"] <= row["request_wall_ms"]))
    checks.append(("execution_rows_complete", len(execn) == len(comp) == 8))
    checks.append(("copies_all_returned", all(c.get("returned") for c in copies) and len(copies) > 0))
    return row, checks


def _fill_blank(row):
    """失败/缺失行补空列，保持 CSV 列对齐。"""
    for k in (
        "control_window", "control_time_ms", "request_ttft_ms", "request_wall_ms",
        "epoch_time_control_inclusive_ms", "audit_inclusive_epoch_ms",
        "reference_audit_rpc_client_ms", "reference_audit_rpc_server_ms",
        "cuda_prefill_ms", "decode_cuda_ms", "cow_cuda_ms", "decode_window_ms",
        "first_decode_outer_wall_ms", "first_decode_inner_eager_cuda_ms",
        "H_physical_fa_hit", "E_executable_prefix", "G_replay_gap", "input_token_count",
        "actual_prefill_tokens", "stage_observation_ms", "stage_epoch_authentication_ms",
        "stage_benefit_matrix_ms", "stage_solver_ms", "stage_plan_build_validate_ms",
        "stage_apply_preflight_ms", "stage_reconciliation_ms", "stage_sum_ms",
        "rpc_unattributed_ms", "control_to_request_gap_ms", "reference_to_control_gap_ms",
    ):
        row.setdefault(k, None)
    row["control_window"] = row["control_window"] or "UNAVAILABLE"


COLUMNS = [
    "experiment", "run", "phase", "pair", "solver", "pending", "status", "control_window",
    "control_time_ms", "request_ttft_ms", "request_wall_ms",
    "epoch_time_control_inclusive_ms", "audit_inclusive_epoch_ms",
    "reference_audit_rpc_client_ms", "reference_audit_rpc_server_ms",
    "cuda_prefill_ms", "decode_cuda_ms", "cow_cuda_ms", "decode_window_ms",
    "first_decode_outer_wall_ms", "first_decode_inner_eager_cuda_ms",
    "H_physical_fa_hit", "E_executable_prefix", "G_replay_gap",
    "input_token_count", "actual_prefill_tokens",
    "stage_observation_ms", "stage_epoch_authentication_ms", "stage_benefit_matrix_ms",
    "stage_solver_ms", "stage_plan_build_validate_ms", "stage_apply_preflight_ms",
    "stage_reconciliation_ms", "stage_sum_ms", "rpc_unattributed_ms",
    "control_to_request_gap_ms", "reference_to_control_gap_ms", "notes",
]


def collect(experiments):
    rows = []
    problems = []
    for label, exp_dir in experiments:
        protocol = load(Path(exp_dir) / "experiment_protocol.json")
        for item in protocol["order"]:
            row, checks = extract_run(exp_dir, item)
            row["experiment"] = label
            rows.append(row)
            for name, ok in checks:
                if not ok:
                    problems.append(f"{label}/{item['name']}: {name}")
    return rows, problems


def write_rows(rows):
    with (P / "unified_timing_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r.get("experiment", ""), r.get("run", ""), r.get("phase", ""),
                        r.get("pair", ""), r.get("solver", ""), r.get("pending", ""),
                        r.get("status", ""), r.get("control_window", "")]
                       + [num(r.get(k)) for k in COLUMNS[8:]])


def write_stats(rows):
    """正式测量臂的描述性统计；单生命周期重复，不做任何跨生命周期“Epoch 加总”。"""
    out = []
    measured = [r for r in rows if r["phase"] == "measurement" and r["status"] == "PASS"]
    for exp in sorted({r["experiment"] for r in measured}):
        for pending in sorted({r["pending"] for r in measured if r["experiment"] == exp}):
            for solver in sorted({r["solver"] for r in measured
                                  if r["experiment"] == exp and r["pending"] == pending}):
                grp = [r for r in measured
                       if r["experiment"] == exp and r["pending"] == pending and r["solver"] == solver]
                for metric in CORE_METRICS:
                    vals = [r[metric] for r in grp if r.get(metric) is not None]
                    if vals:
                        out.append({
                            "experiment": exp, "pending": pending, "solver": solver,
                            "metric": metric, "n_lifecycles": len(vals),
                            "mean_ms": statistics.mean(vals),
                            "median_ms": statistics.median(vals),
                            "min_ms": min(vals), "max_ms": max(vals),
                        })
    with (P / "unified_timing_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "pending", "solver", "metric",
                                          "n_lifecycles", "mean_ms", "median_ms", "min_ms", "max_ms"])
        w.writeheader()
        w.writerows(out)
    return out


def main():
    experiments = list(EXPERIMENTS)
    for extra in sys.argv[1:]:
        p = Path(extra).resolve()
        experiments.append((p.name, p))
    rows, problems = collect(experiments)
    write_rows(rows)
    stats = write_stats(rows)
    failed = [r for r in rows if r["status"] != "PASS"]
    (P / "unified_timing_verification.json").write_text(
        json.dumps({
            "gate": "PASS" if not problems else "FAIL",
            "experiments": [label for label, _ in experiments],
            "lifecycle_rows": len(rows),
            "pass_rows": sum(r["status"] == "PASS" for r in rows),
            "non_pass_rows": sorted({f'{r["experiment"]}/{r["run"]}:{r["status"]}' for r in failed}),
            "stat_groups": len(stats),
            "consistency_problems": problems,
            "说明": "每行为独立Engine生命周期；Cold臂控制窗口按设计缺席；未做任何跨生命周期Epoch加总",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"行数={len(rows)} PASS={sum(r['status'] == 'PASS' for r in rows)} "
          f"统计组={len(stats)} 一致性问题={len(problems)}")


if __name__ == "__main__":
    main()
