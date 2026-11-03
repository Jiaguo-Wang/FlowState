"""分配结构与求解行为归因审计的离线提取（只读，不启动 GPU、不修改任何原始数据）。

数据来源（全部冻结产物，本脚本只读）：
  1) greedy_swap_structure_e3_step1/condition_results.csv —— 新 Strict 20 条件 × 7 策略的选集与共同目标 J；
  2) greedy_swap_structure_e3_step1/legacy_573_structure.csv —— 旧 573 条 Formal 条件的结构特征与最优证书；
  3) c_formal_perf_gpu/policy_plans.json + pending_policy_matrix.csv —— C/K3 三策略冻结保留集、逐 Pending E/G
     与正式 GPU 配对实验中位计时；
  4) 各快照的正收益支持表（w1–w4 见 independent_runtime_allocation_e2/workload_structure_analysis.md，
     B5-C4 见 multishare_allocation_gate_b5c4/candidate_structure_audit.md）——以常量形式转录并在注释中标明出处。

输出（均为派生只读 CSV）：
  paper_allocation_structure_strict20.csv —— 逐条件结构特征 + 7 策略 J + Greedy/Exact 差距与最早分歧；
  paper_allocation_structure_legacy573.csv —— 旧 573 条按数据集聚合的结构与证书统计；
  paper_allocation_structure_gpu_ck3.csv  —— C/K3 静态冗余 ↔ 真实 GPU 恢复前沿/计时对照。

口径声明：J 与差距均为冻结 Phi 下的 modeled ms，不是 GPU 延迟；GPU 列一律来自 c_formal_perf_gpu 实测中位，
两套数字从不换算或相加。
"""
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent
E3 = EVAL / 'greedy_swap_structure_e3_step1'
CFG = EVAL / 'c_formal_perf_gpu'

# 五个冻结快照的正收益支持表（候选 → 其正收益 Pending 集合）。
# 出处：independent_runtime_allocation_e2/workload_structure_analysis.md（w1–w4）；
#       multishare_allocation_gate_b5c4/candidate_structure_audit.md（B5-C4）。
SUPPORTS = {
    'w1':    {'B5C3_NODE_18': ('A', 'B'), 'B5C3_NODE_20': ('A', 'B', 'C', 'D'),
              'B5C3_NODE_21': ('C', 'D'), 'B5C3_NODE_22': ('A',), 'B5C3_NODE_24': ('B',),
              'B5C3_NODE_25': ('C',), 'B5C3_NODE_27': ('D',)},
    'w2':    {'B5C3_NODE_18': ('A', 'B'), 'B5C3_NODE_20': ('A', 'B', 'C', 'D'),
              'B5C3_NODE_21': ('C', 'D'), 'B5C3_NODE_22': ('A',), 'B5C3_NODE_24': ('B',),
              'B5C3_NODE_25': ('C',), 'B5C3_NODE_27': ('D',)},
    'w3':    {'B5C3_NODE_18': ('A', 'B'), 'B5C3_NODE_20': ('A', 'B', 'C', 'D'),
              'B5C3_NODE_21': ('C', 'D'), 'B5C3_NODE_22': ('A',), 'B5C3_NODE_24': ('A', 'B'),
              'B5C3_NODE_25': ('C',), 'B5C3_NODE_27': ('C', 'D')},
    'w4':    {'B5C3_NODE_18': ('A', 'B'), 'B5C3_NODE_20': ('A', 'B', 'C', 'D'),
              'B5C3_NODE_21': ('C', 'D'), 'B5C3_NODE_22': ('A',), 'B5C3_NODE_24': ('A', 'B'),
              'B5C3_NODE_25': ('C',), 'B5C3_NODE_27': ('D',)},
    'B5-C4': {'B5C3_NODE_18': ('A', 'B'), 'B5C3_NODE_20': ('A', 'B', 'C', 'D'),
              'B5C3_NODE_21': ('C', 'D'), 'B5C3_NODE_22': ('A',), 'B5C3_NODE_24': ('B',),
              'B5C3_NODE_25': ('C',), 'B5C3_NODE_27': ('C', 'D')},
}
K_MIN = {c: 4 for c in SUPPORTS}  # 各报告已按严格兼容关系计算：全候选前沿最少需 4 项
POLICIES = ['Exact', 'Greedy', 'Greedy+1-Swap', 'WFC', 'StandaloneTopK', 'PCB', 'LDR']


def struct_features(case):
    """由冻结支持表派生结构特征：合法正收益边数、共享候选数、同支持重叠对数等。"""
    sup = SUPPORTS[case]
    n_shared = sum(1 for v in sup.values() if len(v) >= 2)
    n_edges = sum(len(v) for v in sup.values())
    # 同一 Pending 组合上的候选重叠：支持集合完全相同的无序候选对（深层同链冗余的静态来源）
    by_support = defaultdict(list)
    for cand, pset in sup.items():
        by_support[frozenset(pset)].append(cand)
    overlap_pairs = sorted('|'.join(sorted(p)) for p in by_support.values() if len(p) > 1)
    return sup, n_edges, n_shared, overlap_pairs


def main():
    # ── 一、Strict 20 条件：7 策略 J + 结构特征 + Greedy/Exact 最早分歧 ──
    rows = list(csv.DictReader(open(E3 / 'condition_results.csv')))
    by_cond = defaultdict(dict)  # (case,K) -> policy -> row
    for r in rows:
        by_cond[(r['case'], int(r['K']))][r['policy']] = r
    assert len(by_cond) == 20 and all(len(v) == 7 for v in by_cond.values()), '应为 20 条件 × 7 策略'

    # 记录各策略与 Greedy 的选集一致数（供报告引用；仅描述性计数）
    same_sel = Counter()
    out20 = []
    exact_sets = {}  # case -> {K: set}，用于跨预算嵌套性检查
    for (case, k), pr in sorted(by_cond.items()):
        sup, n_edges, n_shared, overlap_pairs = struct_features(case)
        sel = {p: ast.literal_eval(pr[p]['selected']) for p in POLICIES}
        j = {p: float(pr[p]['J_ms']) for p in POLICIES}
        exact_sets.setdefault(case, {})[k] = set(sel['Exact'])
        # 最早分歧：Greedy 按其顺序逐项选择，首个落入 Exact 最优集之外的步序（不可回退签名）
        eset = set(sel['Exact'])
        first_div = next((i + 1 for i, c in enumerate(sel['Greedy'][:k]) if c not in eset), 0)
        greedy_pick1_in_exact = sel['Greedy'][0] in eset
        for p in POLICIES:
            if p != 'Greedy' and set(sel[p]) == set(sel['Greedy']):
                same_sel[p] += 1
        out20.append({
            'case': case, 'K': k, 'K_min': K_MIN[case], 'K_lt_Kmin': int(k < K_MIN[case]),
            'n_pending': 4, 'n_candidate': len(sup), 'positive_edges': n_edges,
            'shared_candidates': n_shared, 'same_support_overlap_pairs': ';'.join(overlap_pairs) or '0',
            'greedy_first_pick': sel['Greedy'][0], 'first_pick_in_exact': int(greedy_pick1_in_exact),
            'J_exact': round(j['Exact'], 6), 'J_greedy': round(j['Greedy'], 6),
            'gap_greedy_exact': round(j['Greedy'] - j['Exact'], 6),
            'J_1swap': round(j['Greedy+1-Swap'], 6), 'gap_1swap_exact': round(j['Greedy+1-Swap'] - j['Exact'], 6),
            'J_wfc': round(j['WFC'], 6), 'J_standalone_topk': round(j['StandaloneTopK'], 6),
            'J_pcb': round(j['PCB'], 6), 'J_ldr': round(j['LDR'], 6),
            'first_divergence_step': first_div,
            'exact_set': ';'.join(sorted(eset)), 'greedy_set': ';'.join(sel['Greedy'][:k]),
            '1swap_set': ';'.join(sel['Greedy+1-Swap'][:k]),
        })

    # 跨预算最优嵌套性（按 Exact 选集包含关系判定；与 e2 报告的"不存在嵌套最优延伸"结论互核）
    nesting = []
    for case, ks in exact_sets.items():
        for a, b in [(1, 2), (2, 3), (3, 4)]:
            nesting.append((case, a, b, int(ks[a] <= ks[b])))
    nested_pairs = ['%s K%d⊂K%d' % (c, a, b) for c, a, b, ok in nesting if ok]

    with open(EVAL / 'paper_allocation_structure_strict20.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out20[0]))
        w.writeheader()
        w.writerows(out20)

    # ── 二、旧 573 条 Formal：结构与证书聚合（只读重计数）──
    leg = list(csv.DictReader(open(E3 / 'legacy_573_structure.csv')))
    out573 = []
    for ds in sorted({r['dataset'] for r in leg}):
        g = [r for r in leg if r['dataset'] == ds]
        certs = Counter(r['optimality_certificate'] for r in g)
        out573.append({
            'dataset': ds, 'conditions': len(g), 'snapshots': len({r['snapshot_id'] for r in g}),
            'n_candidate_min': min(int(r['n']) for r in g), 'n_candidate_max': max(int(r['n']) for r in g),
            'conditions_with_shared': sum(int(r['shared_candidates']) > 0 for r in g),
            'conditions_with_overlap': sum(int(r['overlap_pairs']) > 0 for r in g),
            'conditions_K_lt_Kmin': sum(r['tight_budget'] == 'True' for r in g),
            'greedy_optimal_in_legacy_model': sum(r['greedy_optimal_in_legacy_model'] == 'True' for r in g),
            'certificates': json.dumps(dict(certs), ensure_ascii=False),
            'strict_runtime_certified': sum(r['strict_runtime_certification'] != 'NOT_CERTIFIED' for r in g),
        })
    with open(EVAL / 'paper_allocation_structure_legacy573.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out573[0]))
        w.writeheader()
        w.writerows(out573)

    # ── 三、C/K3：静态冗余 ↔ 真实 GPU 对照（保留集来自冻结 policy_plans；计时为正式实验中位）──
    plans = json.load(open(CFG / 'policy_plans.json'))
    mapping = json.load(open(CFG / 'cpu_candidate_prefix_mapping.json'))['mapping']
    gpu = {(r['pending'], r['pol']): r for r in csv.DictReader(open(CFG / 'pending_policy_matrix.csv'))}
    POLMAP = {'FlowState-Greedy': 'FS', 'StandaloneTopK': 'SA', 'WFC': 'WFC'}
    outgpu = []
    for pol, short in POLMAP.items():
        retained = plans[pol]['selected_checkpoint_ids']
        # 静态同链冗余检测：同一 prefix_owner 保留两条时，较浅一条对该前沿无增量贡献（被更深同主候选遮蔽）
        by_owner = defaultdict(list)
        for cid in retained:
            by_owner[mapping[cid]['prefix_owner']].append(cid)
        shadowed = ';'.join(sorted(c for ow, cs in by_owner.items() if len(cs) > 1
                                   for c in cs if mapping[c]['L'] < max(mapping[x]['L'] for x in cs)))
        for p in ('C_A', 'C_B', 'C_C', 'C_D'):
            eg = plans[pol]['expected_E_G'][p]
            g = gpu[(p, short)]
            outgpu.append({
                'policy': pol, 'pending': p, 'retained': ';'.join(retained), 'shadowed_same_owner': shadowed or '0',
                'E': eg['E'], 'G': eg['G'], 'T': eg['T'], 'prefill_tokens': g['prefill_tokens'],
                'cuda_prefill_ms_median': g['cuda_prefill_ms'], 'ttft_ms_median': g['ttft_ms'], 'n': g['n'],
            })
    with open(EVAL / 'paper_allocation_structure_gpu_ck3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(outgpu[0]))
        w.writeheader()
        w.writerows(outgpu)

    # ── 控制台摘要（供审计报告核对）──
    print('== Strict 20 概览 ==')
    print('Greedy 最优 %d/20；1-Swap 最优 %d/20' % (
        sum(r['gap_greedy_exact'] == 0 for r in out20), sum(r['gap_1swap_exact'] == 0 for r in out20)))
    print('与 Greedy 选集完全一致：%s' % dict(same_sel))
    print('Greedy 首选 ∈ Exact 最优集的条件：%d/20（不含者即不可回退签名）' % sum(r['first_pick_in_exact'] for r in out20))
    print('Exact 跨预算嵌套成立的对：%s' % (nested_pairs or '无'))
    for r in out20:
        print('%-6s K%d gap=%.6f first_div=step%d first_pick_in_exact=%d overlap=%s' % (
            r['case'], r['K'], r['gap_greedy_exact'], r['first_divergence_step'],
            r['first_pick_in_exact'], r['same_support_overlap_pairs']))
    print('\n== 旧 573 聚合 ==')
    for r in out573:
        print(r)
    print('\n== C/K3 GPU 对照（前 6 行示例）==')
    for r in outgpu[:6]:
        print(r)
    print('\n写出 3 个 CSV：strict20(%d 行) / legacy573(%d 行) / gpu_ck3(%d 行)' % (len(out20), len(out573), len(outgpu)))


if __name__ == '__main__':
    main()
