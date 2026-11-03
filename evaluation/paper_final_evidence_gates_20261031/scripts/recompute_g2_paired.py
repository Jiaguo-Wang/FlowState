#!/usr/bin/env python3
"""Gate B/B4 独立复算：B5-C4/K2 Exact vs Greedy 五轮 GPU 配对数据（只读，零 GPU）。

数据源：evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv（40 臂原始行），
辅以 arm_status.json / checks.json 核对中断与重试。不读取既有统计产物
（paired_analysis.csv / paper/recompute.*），全部统计量从原始行独立重建。

口径（与论文口径分列，严禁混用）：
  A. 配对差中位数 —— 同轮同 Pending 单对差 Δ_{r,p}=Greedy−Exact，对 5 轮取中位（请求级主口径）；
  B. 两组中位数之差 —— 各自 5 轮中位相减（辅助口径，仅对照展示）；
  C. 轮级等权 —— 每轮 4 个单对差等权均值 → 5 轮取中位（集合级口径）。
输出：b4_g2_paired_recompute.json（写入本门禁目录）。
"""
import csv
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
FORMAL = BASE / "evaluation/b5c4_formal_gpu_g2/formal"
OUT_DIR = Path(__file__).resolve().parents[1]

# Pending 完整 Token 数（len(pt)，来自 prepared_inputs；用于 Prefill=len(pt)−E 合同核对）
FULL_LEN = {"A": 4862, "B": 4860, "C": 4435, "D": 4486}
# v6.5 正文引用值（用于逐位核对）
V65 = {"C_ttft_paired": 34.71, "D_ttft_paired": 36.71, "round_equal_weight": 17.74,
       "prefill_C": {"Greedy": 2003, "Exact": 915}, "prefill_D": {"Greedy": 2054, "Exact": 966},
       "J_exact": 48.23, "J_greedy": 130.80}


def main():
    rows = list(csv.DictReader((FORMAL / "raw_results.csv").open()))
    assert len(rows) == 40, f"应为 40 臂，实测 {len(rows)}"
    # pending 列为完整 continuation_id（B5C2_{标签}_TURN_003），统一提取标签
    for r in rows:
        r["pending"] = r["pending"].split("_")[1]
        r["solver"] = r["solver"].strip().lower()  # solver 列实际为 Exact/Greedy，统一小写

    # ---- 有效性自验（逐臂） ----
    valid = []
    for r in rows:
        checks = {
            "audit_entries==2": int(r["audit_entries"]) == 2,
            "请求窗口零新增编译": int(r["request_window_new_kernels"]) == 0,
            "cold_output_match": r["cold_output_match"] in ("True", "true", "1"),  # CSV 中以 1/True 编码
            "prefill合同": int(r["prefill_tokens"]) == FULL_LEN[r["pending"]] - int(r["E"]),
            "E∈合法前沿": int(r["E"]) in (3712, 3520, 2432),
        }
        valid.append(all(checks.values()) or checks)

    arm_status = json.loads((FORMAL / "arm_status.json").read_text())
    status_str = json.dumps(arm_status)
    failures = status_str.lower().count("fail")
    retries = sum(1 for r in rows if r["run"].endswith(("_r2", "_retry")))

    # ---- 结构核对：J/E/选集逐臂 ----
    struct = {}
    for r in rows:
        key = (r["solver"], r["pending"])
        expect_J = 48.228728759089606 if r["solver"] == "exact" else 130.8030838298517
        expect_E = {"A": 3712, "B": 3712, "C": 3520 if r["solver"] == "exact" else 2432,
                    "D": 3520 if r["solver"] == "exact" else 2432}[r["pending"]]
        ok = (abs(float(r["J_ms"]) - expect_J) <= 1e-9 and int(r["E"]) == expect_E)
        struct.setdefault(key, []).append(ok)
    struct_ok = all(all(v) for v in struct.values())

    # ---- 配对（先按轮次与 Pending 配对） ----
    by_key = {(r["round"], r["pending"], r["solver"]): r for r in rows}
    rounds = sorted({r["round"] for r in rows}, key=int)
    report = {"口径声明": {"A": "配对差中位数（主口径）", "B": "两组中位数之差（辅助）",
                       "C": "轮级等权五轮中位（集合级）"},
              "有效性": {"40臂": len(rows), "逐臂检查全过": all(v is True for v in valid),
                       "失败臂计数_arm_status": failures, "重试后缀臂": retries},
              "结构核对": {"J与E逐臂==冻结参照(48.2287/130.803, E 3712/3520/2432)": struct_ok}}

    # 口径 A/B：逐 Pending
    per_pending = {}
    for p in "ABCD":
        deltas, exacts, greedys, d_pref = [], [], [], []
        for rd in rounds:
            e, g = by_key[(rd, p, "exact")], by_key[(rd, p, "greedy")]
            exacts.append(float(e["ttft_ms"]))
            greedys.append(float(g["ttft_ms"]))
            deltas.append(float(g["ttft_ms"]) - float(e["ttft_ms"]))
            d_pref.append(float(g["cuda_prefill_ms"]) - float(e["cuda_prefill_ms"]))
        per_pending[p] = {
            "ΔTTFT_逐轮": [round(d, 3) for d in deltas],
            "A_配对差中位_TTFT": round(statistics.median(deltas), 3),
            "B_两组中位之差_TTFT": round(statistics.median(greedys) - statistics.median(exacts), 3),
            "Exact中位": round(statistics.median(exacts), 3), "Greedy中位": round(statistics.median(greedys), 3),
            "A_配对差中位_CUDAPrefill": round(statistics.median(d_pref), 3),
            "Exact中位_CUDAPrefill": round(statistics.median([float(by_key[(rd, p, 'exact')]['cuda_prefill_ms']) for rd in rounds]), 3),
            "Greedy中位_CUDAPrefill": round(statistics.median([float(by_key[(rd, p, 'greedy')]['cuda_prefill_ms']) for rd in rounds]), 3),
            "Exact_PrefillTokens": sorted({int(by_key[(rd, p, "exact")]["prefill_tokens"]) for rd in rounds}),
            "Greedy_PrefillTokens": sorted({int(by_key[(rd, p, "greedy")]["prefill_tokens"]) for rd in rounds}),
            "五轮TTFT全为Greedy更慢(Δ>0)": sum(d > 0 for d in deltas),
        }
    report["逐Pending"] = per_pending

    # 口径 C：轮级等权
    round_deltas, round_deltas_pref = [], []
    for rd in rounds:
        ds = [float(by_key[(rd, p, "greedy")]["ttft_ms"]) - float(by_key[(rd, p, "exact")]["ttft_ms"]) for p in "ABCD"]
        dp = [float(by_key[(rd, p, "greedy")]["cuda_prefill_ms"]) - float(by_key[(rd, p, "exact")]["cuda_prefill_ms"]) for p in "ABCD"]
        round_deltas.append(sum(ds) / 4)
        round_deltas_pref.append(sum(dp) / 4)
    report["轮级等权"] = {
        "逐轮ΔTTFT": [round(x, 3) for x in round_deltas],
        "逐轮ΔCUDAPrefill": [round(x, 3) for x in round_deltas_pref],
        "五轮中位ΔTTFT": round(statistics.median(round_deltas), 3),
        "五轮中位ΔCUDAPrefill": round(statistics.median(round_deltas_pref), 3),
        "五轮全正_TTFT": all(x > 0 for x in round_deltas),
        "五轮全正_Prefill": all(x > 0 for x in round_deltas_pref),
    }

    # ---- v6.5 数字核对 ----
    report["v65核对"] = {
        "C_34.71": abs(per_pending["C"]["A_配对差中位_TTFT"] - V65["C_ttft_paired"]) < 0.005,
        "D_36.71": abs(per_pending["D"]["A_配对差中位_TTFT"] - V65["D_ttft_paired"]) < 0.005,
        "等权17.74": abs(report["轮级等权"]["五轮中位ΔTTFT"] - V65["round_equal_weight"]) < 0.005,
        "prefill_C_2003/915": per_pending["C"]["Greedy_PrefillTokens"] == [2003] and per_pending["C"]["Exact_PrefillTokens"] == [915],
        "prefill_D_2054/966": per_pending["D"]["Greedy_PrefillTokens"] == [2054] and per_pending["D"]["Exact_PrefillTokens"] == [966],
        "说明": "J 48.23/130.80 为 6 位舍入显示，全精度见结构核对",
    }

    (OUT_DIR / "b4_g2_paired_recompute.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["有效性"], ensure_ascii=False))
    print(json.dumps(report["结构核对"], ensure_ascii=False))
    for p in "ABCD":
        v = per_pending[p]
        print(f"{p}: ΔTTFT配对中位 {v['A_配对差中位_TTFT']:+.2f} (两组口径 {v['B_两组中位之差_TTFT']:+.2f})  "
              f"PrefillTokens Exact{v['Exact_PrefillTokens']}/Greedy{v['Greedy_PrefillTokens']}")
    print(f"轮级等权中位 ΔTTFT {report['轮级等权']['五轮中位ΔTTFT']:+.2f} / ΔPrefill {report['轮级等权']['五轮中位ΔCUDAPrefill']:+.2f} "
          f"全正={report['轮级等权']['五轮全正_TTFT']}/{report['轮级等权']['五轮全正_Prefill']}")
    print(json.dumps(report["v65核对"], ensure_ascii=False))


if __name__ == "__main__":
    main()
