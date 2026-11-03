"""P0-1 收尾检查 1：Strict20 Exact 最优集合唯一性与嵌套性（CPU-only 只读复算）。

数据源（只读）：
- greedy_swap_structure_e3_step1/{condition_results.csv, selection_traces.json}

口径（与 e3/t1 完全一致）：
- 前沿矩阵 = selection_traces 的 step1（prefix 空）advances：候选单独可给各 Pending 的恢复深度；
- 目标 J(S)=Σ_p Phi(T_p−E_p(S), T_p)，E_p=max(选中候选 advances ∪ {0})；Phi 冻结系数；
- 每条件（5 快照×K1-4）对全部 |S|≤K 的 2^7 子集枚举，最优 J*（容差 1e-9）下收集全部最优集合。

三项统计（不改冻结 Exact 选集与既有 11/15 tie-break 非嵌套统计）：
1 逐条件最优集合个数（唯一性）；
2 相邻预算转移 K→K+1 是否存在嵌套最优集合对（∃ S_K∈OPT(K), S_{K+1}∈OPT(K+1), S_K⊆S_{K+1}）；
3 逐快照是否存在 K=1→4 完整嵌套最优解链（回溯搜索）。
区分口径：tie-break 终集非嵌套（既有 11/15）≠ 任何最优解都无法嵌套（结构性非嵌套，本脚本量出）。
"""
import csv
import itertools
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
OUT = ROOT / 'evaluation/paper_workload_structure_final'
EPS = 1e-9
PHI_A, PHI_B, PHI_C = Decimal('37.828150'), Decimal('0.345974143'), Decimal('0.156201917')


def phi(g, t):
    """冻结 Phi 成本模型（g/t 为 Token，内部按 1024 块换算）。"""
    g, t = Decimal(g) / 1024, Decimal(t) / 1024
    return float(PHI_A * g + PHI_B * g * t - PHI_C * g * g)


def main():
    rows = list(csv.DictReader((E3 / 'condition_results.csv').open()))
    traces = json.loads((E3 / 'selection_traces.json').read_text())
    step1 = {(t['case'], t['K']): t for t in traces if t['step'] == 1}
    # 前沿矩阵与 T
    matrix, T = {}, {}
    for (case, K), t in step1.items():
        matrix[case] = {c['checkpoint_id']: dict(c['advances']) for c in t['ranked']}
    for r in rows:
        if r['policy'] == 'Exact':
            T[(r['case'], int(r['K']))] = {p['continuation_id']: p['T']
                                           for p in json.loads(r['per_pending'])}
    cases = sorted(matrix)
    # 各快照 Pending 命名空间不同（IMW1_* / B5C2_* 等）：pids 与 T 均按 case 取
    pids = {case: list(next(iter(matrix[case].values())).keys()) for case in cases}

    def J(case, S):
        m = matrix[case]
        tt = T[(case, 1)]
        return sum(phi(tt[p] - max([m[c][p] for c in S] + [0]), tt[p]) for p in pids[case])

    # 逐条件收集全部最优集合
    opt = {}   # (case,K) -> (J*, [frozenset...])
    for case in cases:
        ids = sorted(matrix[case])
        for K in (1, 2, 3, 4):
            subsets = [frozenset(S) for k in range(K + 1) for S in itertools.combinations(ids, k)]
            js = [(J(case, S), S) for S in subsets]
            jstar = min(v for v, _ in js)
            opt[(case, K)] = (jstar, sorted([S for v, S in js if abs(v - jstar) <= EPS],
                                            key=lambda S: (len(S), sorted(S))))
    # 冻结 tie-break 终集核对（(大小,字典序) 最小规则应与 e3 Exact 选集一致）
    frozen = {(r['case'], int(r['K'])): sorted(json.loads(r['selected']))
              for r in rows if r['policy'] == 'Exact'}
    cross_ok = all(sorted(opt[k][1][0]) == frozen[k] for k in opt)

    # 相邻转移嵌套对存在性 + 完整链
    detail, transfer_rows = [], []
    for case in cases:
        chain_exists = False
        chain = None
        # 回溯找完整链 S1⊆S2⊆S3⊆S4
        for s4 in opt[(case, 4)][1]:
            for s3 in [s for s in opt[(case, 3)][1] if s <= s4]:
                for s2 in [s for s in opt[(case, 2)][1] if s <= s3]:
                    for s1 in [s for s in opt[(case, 1)][1] if s <= s2]:
                        chain_exists, chain = True, [s1, s2, s3, s4]
                        break
                    if chain_exists: break
                if chain_exists: break
            if chain_exists: break
        for K in (1, 2, 3):
            tb_k, tb_k1 = opt[(case, K)][1][0], opt[(case, K + 1)][1][0]
            nested_pair = any(a <= b for a in opt[(case, K)][1] for b in opt[(case, K + 1)][1])
            transfer_rows.append({
                'case': case, 'transfer': 'K%d->K%d' % (K, K + 1),
                'n_optima_K': len(opt[(case, K)][1]), 'n_optima_K1': len(opt[(case, K + 1)][1]),
                'tiebreak_nested': tb_k <= tb_k1,
                'exists_nested_pair': nested_pair,
                'structural_nonnested': not nested_pair})
        detail.append({'case': case, 'full_chain_exists': chain_exists,
                       'chain': [sorted(s) for s in chain] if chain else None,
                       'optima_counts': {str(K): len(opt[(case, K)][1]) for K in (1, 2, 3, 4)},
                       'K1_optima': [sorted(s) for s in opt[(case, 1)][1]],
                       'K2_optima': [sorted(s) for s in opt[(case, 2)][1]]})
    n_trans = len(transfer_rows)
    result = {
        'tiebreak_crosscheck_with_e3_frozen': cross_ok,
        'transfers_total': n_trans,
        'tiebreak_nonnested': sum(1 for r in transfer_rows if not r['tiebreak_nested']),
        'exists_nested_pair': sum(1 for r in transfer_rows if r['exists_nested_pair']),
        'structural_nonnested': sum(1 for r in transfer_rows if r['structural_nonnested']),
        'full_chain_snapshots': sum(1 for d in detail if d['full_chain_exists']),
        'unique_optima_conditions': sum(1 for k in opt if len(opt[k][1]) == 1),
        'detail': detail, 'transfers': transfer_rows}
    (OUT / 'strict20_optima_nesting.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    with (OUT / 'strict20_optima_nesting.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(transfer_rows[0].keys()))
        w.writeheader()
        w.writerows(transfer_rows)
    print('tie-break 终集==e3 冻结 Exact 选集：', cross_ok)
    print('唯一最优条件数： %d/20' % result['unique_optima_conditions'])
    print('转移 %d：tie-break 非嵌套 %d | 存在嵌套最优对 %d | 结构性非嵌套（无任何嵌套对）%d' % (
        n_trans, result['tiebreak_nonnested'], result['exists_nested_pair'], result['structural_nonnested']))
    print('完整嵌套链快照数： %d/5' % result['full_chain_snapshots'])
    for d in detail:
        print('%s: 各K最优数 %s 完整链 %s' % (d['case'], d['optima_counts'], d['full_chain_exists']))


if __name__ == '__main__':
    main()
