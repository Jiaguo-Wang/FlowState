# -*- coding: utf-8 -*-
"""unified_timing 的 CPU 离线回归：合成最小实验目录，验证抽取、代数、缺失标记与失败处理。

不启动 GPU、不读真实冻结目录；仅用临时目录与标准库。运行：
  /home/wjg/uv312/bin/python -I evaluation/e45_unified_timing/test_unified_timing.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_timing as ut  # noqa: E402


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def compute_row(is_extend, cuda_ms, ns_before, ns_after, extend_tokens=0):
    return {
        "is_extend": is_extend,
        "cuda_elapsed_ms": cuda_ms,
        "ns_before": ns_before,
        "ns_after": ns_after,
        "returned": True,
    }


def build_experiment(root):
    """构造 1 个 Cold、1 个正式测量、1 个失败运行的合成实验。"""
    write(root / "experiment_protocol.json", {
        "order": [
            {"name": "C_Cold", "solver": "Cold", "pending": "P_C", "phase": "cold", "pair": 0},
            {"name": "m1_Exact", "solver": "Exact", "pending": "P_A", "phase": "measurement", "pair": 1},
            {"name": "m2_Broken", "solver": "Exact", "pending": "P_A", "phase": "measurement", "pair": 2},
        ]
    })
    # ---- Cold 臂：无控制窗口，T2/T3/T4 可测 ----
    d = root / "C_Cold"
    write(d / "summary.json", {
        "status": "PASS", "arm": "cold",
        "native_match": {"physical_fa_hit": 0, "executable_prefix": 0, "replay_gap": 0,
                         "input_token_count": 100},
        "ttft_ms": 10.0, "request_wall_ms": 20.0,
        "control_rpc_wall_ms": None, "control_through_completion_ms": None,
        "audit_inclusive_through_completion_ms": None, "reference_observation_rpc_wall_ms": None,
    })
    comp = [compute_row(True, 5.0, 0, 5_000_000, 100)] + [
        compute_row(False, 1.0, 6_000_000, 7_000_000) for _ in range(7)]
    execn = [dict(compute_row(True, 5.0, 0, 5_000_000), extend_token_count=100, native_run_returned=True)] + [
        dict(compute_row(False, 1.0, 6_000_000, 7_000_000), extend_token_count=0, native_run_returned=True)
        for _ in range(7)]
    write(d / "post_request.json", {"trace": {"compute": comp, "execution": execn, "copies": []}})

    # ---- 正式测量臂：端点代数自洽 ----
    d = root / "m1_Exact"
    write(d / "summary.json", {
        "status": "PASS", "arm": "strict",
        "native_match": {"physical_fa_hit": 80, "executable_prefix": 30, "replay_gap": 50,
                         "input_token_count": 150},
        "ttft_ms": 50.0, "request_wall_ms": 200.0,
        "control_rpc_wall_ms": 100.0, "control_through_completion_ms": 320.0,
        "audit_inclusive_through_completion_ms": 700.0,
        "reference_observation_rpc_wall_ms": 300.0,
    })
    write(d / "reconciliation.json", {
        "ok": True,
        "timing": [
            {"stage": "metadata_observation", "ms": 10.0},
            {"stage": "strict_epoch_authentication", "ms": 20.0},
            {"stage": "benefit_matrix", "ms": 1.0},
            {"stage": "solver", "ms": 2.0},
            {"stage": "plan_build_validate", "ms": 3.0},
            {"stage": "apply_time_freshness_preflight", "ms": 4.0},
            {"stage": "strict_reconciliation_with_postchecks", "ms": 20.0},
        ],
    })
    write(d / "reference_audit.json", {"wall_ms": 5.0})
    write(d / "stream_chunks.json", {
        "start_ns": 1_000, "first_token_ns": 1_000 + 50_000_000,
        "end_ns": 1_000 + 200_000_000, "chunks": [], "arrivals": [],
    })
    comp = [compute_row(True, 5.0, 0, 5_000_000)] + [
        compute_row(False, 1.0, 6_000_000, 7_000_000) for _ in range(7)]
    execn = [dict(compute_row(True, 5.0, 0, 5_000_000), extend_token_count=120, native_run_returned=True)] + [
        dict(compute_row(False, 1.0, 6_000_000, 6_000_000 + (150_000_000 if i == 0 else 1_000_000)),
             extend_token_count=0, native_run_returned=True) for i in range(7)]
    write(d / "post_request.json", {
        "trace": {"compute": comp, "execution": execn,
                  "copies": [{"returned": True, "cuda_elapsed_ms": 2.0}]}})

    # ---- 失败运行：只留 failure.json ----
    write(root / "m2_Broken" / "failure.json", {"status": "BLOCKED", "error": "synthetic"})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "synthetic_exp"
        build_experiment(root)
        rows, problems = ut.collect([("SYN", root)])
        by_run = {r["run"]: r for r in rows}

        # 1. 行数与状态
        assert len(rows) == 3, "应有 3 行生命周期记录"
        assert by_run["C_Cold"]["status"] == "PASS"
        assert by_run["m1_Exact"]["status"] == "PASS"
        assert by_run["m2_Broken"]["status"] == "BLOCKED", "failure.json 应标记 BLOCKED"

        # 2. Cold 臂：控制窗口缺席标记，不推算
        cold = by_run["C_Cold"]
        assert cold["control_window"] == "ABSENT(cold)", "Cold 应标记无控制窗口"
        assert cold["control_time_ms"] is None and cold["epoch_time_control_inclusive_ms"] is None
        assert cold["request_ttft_ms"] == 10.0 and cold["cuda_prefill_ms"] == 5.0

        # 3. 正式臂：统一指标与端点代数
        m1 = by_run["m1_Exact"]
        assert m1["control_window"] == "MEASURED"
        assert m1["control_time_ms"] == 100.0, "Control Time = T1−T0"
        assert m1["request_ttft_ms"] == 50.0 and m1["request_wall_ms"] == 200.0
        assert m1["epoch_time_control_inclusive_ms"] == 320.0, "Epoch = T4−T0"
        assert abs(m1["control_to_request_gap_ms"] - 20.0) < 1e-9, "e−d 间隙"
        assert abs(m1["reference_to_control_gap_ms"] - 80.0) < 1e-9, "c−b 间隙"
        assert abs(m1["rpc_unattributed_ms"] - 40.0) < 1e-9, "RPC 未归因余额"
        assert m1["actual_prefill_tokens"] == 120 == 150 - 30, "实际 Prefill 独立复算"
        assert abs(m1["first_decode_outer_wall_ms"] - 150.0) < 1e-9, "首 Decode 外层 Wall"
        assert m1["cow_cuda_ms"] == 2.0 and m1["reference_audit_rpc_server_ms"] == 5.0

        # 4. 一致性检查全部通过；失败行不产生检查项
        assert problems == [], f"合成数据应无一致性问题：{problems}"

        # 5. 统计只含 PASS 正式测量臂，失败/Cold/Smoke 不进入
        stats = [s for s in rows if s["phase"] == "measurement" and s["status"] == "PASS"]
        assert {s["run"] for s in stats} == {"m1_Exact"}

        # 6. 注入矛盾数据：TTFT 大于请求 Wall 且与原始 ns 不符 → 必须被标记
        summary_path = root / "m1_Exact" / "summary.json"
        bad = json.loads(summary_path.read_text())
        bad["ttft_ms"] = 250.0  # 大于 request_wall=200，且偏离 stream ns
        summary_path.write_text(json.dumps(bad), encoding="utf-8")
        _, problems2 = ut.collect([("SYN", root)])
        names = {p.split(": ")[1] for p in problems2}
        assert "ordering_ttft_within_wall" in names, "顺序违规必须被捕获"
        assert "ttft_ns_agrees_summary" in names, "ns 对账失败必须被捕获"

    print("PASS：合成目录抽取/代数/缺失标记/失败处理/统计边界全部通过")


if __name__ == "__main__":
    main()
