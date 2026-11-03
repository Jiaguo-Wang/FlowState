#!/usr/bin/env python3
"""Gate B/B3 独立复算：B5-C4 共享分支工作负载的模型化分配（只读，零 GPU）。

数据源（全部为冻结原始文件，不读取任何派生统计产物）：
  - evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json
      candidates[].token_position_L     —— 7 个候选的真实驻留位置
      edges[]（28 条，status=EPOCH_ADMISSIBLE 为准入边）
      pendings[].planning_target_T      —— 4 个 Pending 的目标位置 T
  - 冻结 Φ 公式 position_aware_quadratic_v1（硬编码独立录入，并与生产模块交叉核对）

复算内容：
  1) J(S)=Σ_p Φ(T_p−E_p(S), T_p)，E_p(S)=S 中准入候选的最大位置（无则 0）；
  2) K=1..4 全子集枚举 argmin（含空集；tie-break：项数最少→ID 字典序）；
  3) Marginal Greedy 从零重放（每步取边际收益最大者，无正收益即停）；
  4) 对照 evaluation/paper_allocation_structure_strict20.csv 的 B5-C4 冻结行。
输出：b3_allocation_recompute.json + b3_k2_subsets.csv（写入本门禁目录）。
"""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

# ---- 冻结输入（只读） ----
BASE = Path(__file__).resolve().parents[3]
SNAP = BASE / "evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json"
STRICT20 = BASE / "evaluation/paper_allocation_structure_strict20.csv"
OUT_DIR = Path(__file__).resolve().parents[1]

# 冻结 Φ 系数（position_aware_quadratic_v1；独立录入，运行时与生产模块核对）
PHI_A, PHI_B, PHI_C = 37.828150, 0.345974143, -0.156201917


def phi(gap_tokens: int, target_tokens: int) -> float:
    """Φ(G,T)=a·g+b·g·t+c·g²，g/t 单位 Ki-token（1024 token），输出 ms。"""
    g = gap_tokens / 1024.0
    t = target_tokens / 1024.0
    return PHI_A * g + PHI_B * g * t + PHI_C * g * g


def load_epoch():
    """从 b5c3 冻结快照装载候选/准入边/Pending 目标。"""
    snap = json.loads(SNAP.read_text())
    cands = {c["checkpoint_id"]: c["token_position_L"] for c in snap["candidates"]}
    labels = {p["continuation_id"]: p["label"] for p in snap["pendings"]}
    targets = {p["label"]: p["planning_target_T"] for p in snap["pendings"]}
    admissible = {cid: set() for cid in cands}
    status_count = {}
    for e in snap["edges"]:
        status_count[e["status"]] = status_count.get(e["status"], 0) + 1
        if e["status"] == "EPOCH_ADMISSIBLE":
            admissible[e["checkpoint_id"]].add(labels[e["continuation_id"]])
    return cands, targets, admissible, status_count


def frontier(selected, cands, admissible, label):
    """E_p = 选集中准入候选的最大位置；无准入候选则 0。"""
    positions = [cands[c] for c in selected if label in admissible[c]]
    return max(positions) if positions else 0


def j_of(selected, cands, targets, admissible):
    """J(S)=Σ_p Φ(T_p−E_p, T_p)。"""
    return sum(
        phi(targets[lb] - frontier(selected, cands, admissible, lb), targets[lb])
        for lb in sorted(targets)
    )


def exact_best(cands, targets, admissible, k):
    """全子集枚举（|S|≤K，含空集），返回 argmin 及全部子集明细。tie-break：项数最少→ID 字典序。"""
    ids = sorted(cands)
    subsets = []
    for size in range(0, k + 1):
        for combo in itertools.combinations(ids, size):
            subsets.append((tuple(combo), j_of(combo, cands, targets, admissible)))
    # 与生产 _enumerate_exact_subsets 同 tie-breaking：J 最小 → 项数最少 → 字典序
    best = min(subsets, key=lambda x: (round(x[1], 9), len(x[0]), x[0]))
    return best, subsets


def marginal_greedy(cands, targets, admissible, k):
    """Marginal Greedy 重放：每步取边际 J 降幅最大者；无正收益即停。"""
    selected, trace = [], []
    base = j_of((), cands, targets, admissible)
    while len(selected) < k:
        gains = [
            (j_of(selected, cands, targets, admissible)
             - j_of(tuple(sorted(selected + [c])), cands, targets, admissible), c)
            for c in sorted(cands) if c not in selected
        ]
        gain, pick = max(gains)
        if gain <= 0:
            break
        selected.append(pick)
        trace.append({"pick": pick, "marginal_gain_ms": gain, "J_after": base - sum(t["marginal_gain_ms"] for t in trace)})
    return tuple(sorted(selected)), trace


def main():
    cands, targets, admissible, status_count = load_epoch()

    # 与生产 Φ 模块交叉核对（硬编码公式 vs flowstate.recovery_model）
    sys.path.insert(0, str(BASE))
    from flowstate.recovery_model import RecoveryCostModel  # 只读导入，用于一致性交叉核对
    prod = RecoveryCostModel()
    cross = max(
        abs(phi(g, t) - prod.estimate(g, t))
        for g in (0, 46, 384, 448, 1134, 1472) for t in (3566, 3904, 4096, 4160)
    )

    # strict20 冻结参照行
    ref = {}
    with STRICT20.open() as f:
        for row in csv.DictReader(f):
            if row["case"] == "B5-C4":
                ref[int(row["K"])] = row

    report = {
        "输入": {
            "候选数": len(cands),
            "候选位置": cands,
            "准入边数": sum(len(v) for v in admissible.values()),
            "边状态统计": status_count,
            "Pending目标T": targets,
            "Φ交叉核对_最大绝对差_ms": cross,
        },
        "逐K复算": {},
        "与strict20一致": {},
    }

    for k in (1, 2, 3, 4):
        (best_set, best_j), subsets = exact_best(cands, targets, admissible, k)
        greedy_set, greedy_trace = marginal_greedy(cands, targets, admissible, k)
        greedy_j = j_of(greedy_set, cands, targets, admissible)
        n_sub = len(subsets)
        entry = {
            "子集数_含空集": n_sub,
            "理论子集数": sum(math.comb(len(cands), i) for i in range(k + 1)),
            "exact选集": list(best_set),
            "exact_J_ms": best_j,
            "greedy选集": list(greedy_set),
            "greedy_J_ms": greedy_j,
        }
        if k == 2:
            # K=2 逐 Pending E/G（两计划对照）与次优差距
            ranked = sorted(subsets, key=lambda x: x[1])
            entry["次优子集"] = [list(ranked[1][0]), ranked[1][1]]
            entry["argmin唯一性_gap_ms"] = ranked[1][1] - ranked[0][1]
            entry["逐Pending"] = {
                plan: {
                    lb: {"E": frontier(tuple(s), cands, admissible, lb),
                         "G": targets[lb] - frontier(tuple(s), cands, admissible, lb)}
                    for lb in sorted(targets)
                }
                for plan, s in (("Exact", best_set), ("Greedy", greedy_set))
            }
            with (OUT_DIR / "b3_k2_subsets.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["subset", "size", "J_ms"])
                for combo, j in sorted(subsets, key=lambda x: x[1]):
                    w.writerow(["|".join(combo), len(combo), f"{j:.9f}"])
        report["逐K复算"][f"K={k}"] = entry

        row = ref[k]
        # strict20 CSV 的 J 为 6 位小数舍入，容差取 1e-6（大于舍入半步长 5e-7）
        ok_exact = (set(best_set) == set(row["exact_set"].split(";"))
                    and abs(best_j - float(row["J_exact"])) <= 1e-6)
        ok_greedy = (set(greedy_set) == set(row["greedy_set"].split(";"))
                     and abs(greedy_j - float(row["J_greedy"])) <= 1e-6)
        report["与strict20一致"][f"K={k}"] = {
            "exact选集与J一致": ok_exact, "greedy选集与J一致": ok_greedy,
            "参照_J_exact": float(row["J_exact"]), "参照_J_greedy": float(row["J_greedy"]),
        }

    (OUT_DIR / "b3_allocation_recompute.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    # 摘要打印
    print(f"准入边 {report['输入']['准入边数']}（参照 13）；Φ 交叉核对最大差 {cross:.3e} ms")
    for k in (1, 2, 3, 4):
        e = report["逐K复算"][f"K={k}"]
        c = report["与strict20一致"][f"K={k}"]
        print(f"K={k}: 子集 {e['子集数_含空集']}  Exact {e['exact选集']} J={e['exact_J_ms']:.6f}  "
              f"Greedy {e['greedy选集']} J={e['greedy_J_ms']:.6f}  一致={c['exact选集与J一致']}/{c['greedy选集与J一致']}")


if __name__ == "__main__":
    main()
