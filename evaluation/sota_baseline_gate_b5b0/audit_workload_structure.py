# -*- coding: utf-8 -*-
"""
Phase B5-B0 工作负载结构审计脚本（只读）。

数据源（全部为冻结产物，本脚本不修改任何文件）：
- evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g00{1..4}/snapshot.json
- evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/batch_summary.json
- evaluation/strict_compatibility_b4c/snapshot_budget_policy.csv（B4-C 冻结结果）
- evaluation/strict_compatibility_b4c/input_integrity.json（SHA256 复核基准）

计算内容：
1. 各实例候选数 / Pending 数 / 严格边（EPOCH_ADMISSIBLE / REJECTED）；
2. 按 B4-C 冻结 Phi 公式计算每条准入边的正收益（b>1e-9）与收益值；
3. 跨 Pending 正收益共享（同一候选准入到 >=2 个 Pending，即 fanout q>=2）；
4. 同一 Pending 的收益重叠（同 workflow 候选按 L 排序的嵌套覆盖对数）；
5. K_min 独立复算（覆盖全部 Pending 最大前沿所需最少候选数）；
6. 与 B4-C 冻结结果的差异对齐（逐条件列出与 FlowState 有严格差异的策略）；
7. 快照 SHA256 完整性复核。

输出：evaluation/sota_baseline_gate_b5b0/workload_structure_metrics.json
"""

import csv
import hashlib
import json
import os

# 路径均相对项目根，避免依赖当前工作目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CAPTURE = os.path.join(ROOT, "evaluation", "strict_compatibility_b4b1_output", "capture_g001_g006_run01")
B4C = os.path.join(ROOT, "evaluation", "strict_compatibility_b4c")
OUT = os.path.join(ROOT, "evaluation", "sota_baseline_gate_b5b0", "workload_structure_metrics.json")

GROUPS = ["g001", "g002", "g003", "g004"]

# B4-C 冻结的 Phi 公式（position_aware_quadratic_v1），此处仅为结构复算，不重新拟合
PHI_A = 37.828150
PHI_B = 0.345974143
PHI_C = -0.156201917


def phi(g_tokens: int, t_tokens: int) -> float:
    """冻结 Phi：g=G/1024、t=T/1024 下的二次位置感知恢复成本（modeled ms）。"""
    g = g_tokens / 1024.0
    t = t_tokens / 1024.0
    return PHI_A * g + PHI_B * g * t + PHI_C * g * g


def edge_benefit(l_tokens: int, t_tokens: int) -> float:
    """单候选独占时该边的潜在收益：E 从 0 提升到 L带来的 J 下降量。"""
    return phi(t_tokens, t_tokens) - phi(t_tokens - l_tokens, t_tokens)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_group(gid: str) -> dict:
    """对一个实例快照做只读结构审计，返回结构指标字典。"""
    snap_path = os.path.join(CAPTURE, gid, "snapshot.json")
    d = json.load(open(snap_path))

    cands = {c["checkpoint_id"]: c for c in d["candidates"]}
    pends = {p["continuation_id"]: p for p in d["pendings"]}
    # L 为候选 token 位置（已推进的 token 前缀长度），T 为 Pending 规划目标
    cand_L = {cid: c["token_position_L"] for cid, c in cands.items()}
    pend_T = {pid: p["planning_target_T"] for pid, p in pends.items()}

    adm = [e for e in d["edges"] if e["status"] == "EPOCH_ADMISSIBLE"]
    rej = [e for e in d["edges"] if e["status"] == "REJECTED"]
    unk = [e for e in d["edges"] if e["status"] not in ("EPOCH_ADMISSIBLE", "REJECTED")]

    # 每条准入边的收益与正收益判定
    edge_rows = []
    for e in adm:
        cid, pid = e["checkpoint_id"], e["continuation_id"]
        b = edge_benefit(cand_L[cid], pend_T[pid])
        edge_rows.append({
            "checkpoint_id": cid, "continuation_id": pid,
            "L": cand_L[cid], "T": pend_T[pid],
            "b_ms": b, "positive": b > 1e-9,
        })

    # 跨 Pending 共享：候选 fanout q = 准入到的 Pending 数
    fanout = {cid: 0 for cid in cands}
    for e in adm:
        fanout[e["checkpoint_id"]] += 1
    shared = {cid: q for cid, q in fanout.items() if q >= 2}

    # 每个 Pending 的准入候选数（入度）与同 workflow 嵌套覆盖对
    per_pending = {}
    for pid in pends:
        cs = [e["checkpoint_id"] for e in adm if e["continuation_id"] == pid]
        same_wf_pairs = 0
        dominated = []
        for i, c1 in enumerate(cs):
            for c2 in cs[i + 1:]:
                if cands[c1]["workflow_id"] == cands[c2]["workflow_id"]:
                    # 同 workflow 两候选对同一 Pending 都有收益：浅者被深者覆盖（E=max L）
                    shallow, deep = (c1, c2) if cand_L[c1] < cand_L[c2] else (c2, c1)
                    same_wf_pairs += 1
                    dominated.append((shallow, deep, cand_L[shallow], cand_L[deep]))
        per_pending[pid] = {
            "T": pend_T[pid],
            "admissible_candidates": len(cs),
            "max_L": max((cand_L[c] for c in cs), default=0),
            "argmax_L": [c for c in cs if cand_L[c] == max((cand_L[x] for x in cs), default=0)],
            "same_workflow_overlap_pairs": same_wf_pairs,
            "overlap_pairs_detail": dominated,
        }

    # K_min 独立复算：每个 Pending 取其 argmax-L 候选，去重后的最少候选数
    need = set()
    for pid, info in per_pending.items():
        for c in info["argmax_L"]:
            need.add(c)
    k_min_recomputed = len(need)
    k_min_argmax_sets = {pid: info["argmax_L"] for pid, info in per_pending.items()}

    return {
        "group": gid,
        "snapshot_sha256_actual": sha256_of(snap_path),
        "n_candidates": len(cands),
        "n_pendings": len(pends),
        "edges": {"admissible": len(adm), "rejected": len(rej), "unknown": len(unk)},
        "positive_edges": sum(1 for r in edge_rows if r["positive"]),
        "all_admissible_edges_positive": all(r["positive"] for r in edge_rows),
        "cross_pending_shared_candidates": {
            cid: {"fanout": q,
                  "workflow_label": cands[cid]["workflow_label"],
                  "L": cand_L[cid]} for cid, q in shared.items()
        },
        "per_pending": per_pending,
        "k_min_recomputed": k_min_recomputed,
        "k_min_cover_set": sorted(need),
        "k_min_argmax_sets": k_min_argmax_sets,
        "candidate_L_summary": {
            "min": min(cand_L.values()), "max": max(cand_L.values()),
            "distinct_workflows": len({c["workflow_id"] for c in cands.values()}),
        },
    }


def load_b4c_differences() -> dict:
    """读取 B4-C 冻结 CSV，按 (group,K) 汇总各策略相对 FlowState 的严格差异。"""
    rows = list(csv.DictReader(open(os.path.join(B4C, "snapshot_budget_policy.csv"))))
    by_cond = {}
    for r in rows:
        key = (r["group"], int(r["K"]))
        by_cond.setdefault(key, {})[r["policy"]] = {
            "J_ms": float(r["J_ms"]),
            "diff_vs_flowstate_ms": float(r["baseline_minus_flowstate_ms"]),
            "WTL": r["FlowState_WTL"],
            "same_set": r["same_set_as_flowstate"] == "True",
        }
    out = {}
    for (g, k), pol in sorted(by_cond.items()):
        fs = pol["FlowState"]["J_ms"]
        diffs = {p: v for p, v in pol.items()
                 if p != "FlowState" and abs(v["diff_vs_flowstate_ms"]) > 1e-9}
        out[f"{g}/K{k}"] = {
            "flowstate_J_ms": fs,
            "policies_differing_from_flowstate": sorted(diffs.keys()),
            "details": {p: {"diff_ms": v["diff_vs_flowstate_ms"], "dir": v["WTL"]}
                        for p, v in sorted(diffs.items())},
        }
    return out


def main():
    metrics = {"groups": {}, "b4c_differences": {}, "integrity": {}}

    # SHA256 与 B4-C input_integrity.json 逐一复核
    integrity = json.load(open(os.path.join(B4C, "input_integrity.json")))
    for gid in GROUPS:
        m = audit_group(gid)
        metrics["groups"][gid] = m
        expected = integrity["sha256"].get(os.path.join(CAPTURE, gid, "snapshot.json"))
        metrics["integrity"][gid] = {
            "expected": expected, "actual": m["snapshot_sha256_actual"],
            "match": expected == m["snapshot_sha256_actual"],
        }

    metrics["b4c_differences"] = load_b4c_differences()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    # 控制台摘要
    for gid in GROUPS:
        m = metrics["groups"][gid]
        print(f"{gid}: 候选={m['n_candidates']} Pending={m['n_pendings']} "
              f"准入边={m['edges']['admissible']} 正收益边={m['positive_edges']} "
              f"跨Pending共享={len(m['cross_pending_shared_candidates'])} "
              f"K_min复算={m['k_min_recomputed']} 完整性={'PASS' if metrics['integrity'][gid]['match'] else 'FAIL'}")
        for pid, info in m["per_pending"].items():
            print(f"   {pid}: 入度={info['admissible_candidates']} "
                  f"argmaxL={info['argmax_L']} 同workflow重叠对={info['same_workflow_overlap_pairs']}")
    n_diff = sum(1 for v in metrics["b4c_differences"].values() if v["policies_differing_from_flowstate"])
    print(f"B4-C 条件总数={len(metrics['b4c_differences'])} 存在策略严格差异的条件数={n_diff}")
    print(f"输出已写入 {OUT}")


if __name__ == "__main__":
    main()
