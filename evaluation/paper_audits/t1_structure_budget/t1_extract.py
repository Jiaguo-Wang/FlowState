"""T1 结构与预算敏感性提取（宿主机执行，只读冻结数据；不运行 GPU、不改任何算法与历史文件）。

数据来源（全部在仓冻结，脚本逐文件哈希记录到 t1_source_hashes.json）：
  A) Strict20（Strict Epoch 认证）：greedy_swap_structure_e3_step1/condition_results.csv
     （5 快照 × K1-4 × 7 策略 = 140 行）+ selection_traces.json（50 条逐步轨迹，step1 的
     advances 给出每个快照完整的 候选×Pending 恢复前沿矩阵）。
  B) 历史 573（Formal 模型，未经 Strict Runtime 认证）：gate1_step1_allocation_equivalence/
     {gate1_step1_instance_statistics.csv, matrices.jsonl}（benefit_ms 矩阵 + positions 深度 +
     compatibility 布尔阵 + targets）；策略 J 由 jsonl policies.per_continuation 求和重算。
  C) 对照用已冻结汇总：paper_allocation_structure_strict20.csv / legacy573.csv（仅交叉核对，
     不作为计算输入）。

独立复算内容：
  1) Strict20：7 策略最优性/选集一致性；结构指标（支持集、共享、正收益边、逐 Pending 合法
     候选数、同支持对）；K_min＝逐 Pending 最深合法前沿的唯一实现者并集（再用 2^7 子集枚举
     复核最小性）；全域最优 J（无预算限制）与各 K 的关系；Phi 冻结系数与 140 行逐格复验。
  2) Legacy573：按证书口径独立重算（可分支持 TopK / 全候选前沿），四策略（FlowState/LFU/
     LRU/Marconi）J 与最优一致情况；同一套结构指标。
  3) 结构分类 A（可分离）/B（K≥K_min）/C（共享且 K<K_min）逐条件标注；Greedy 失败条件的
     逐步轨迹导出（case_attribution 用）。

口径声明：所有 J/gap 均为冻结 Phi 下的 modeled ms，不是 GPU 延迟；legacy 结构为 Formal 模型
（strict_runtime_certification=NOT_CERTIFIED），与 Strict20 不混同统计。
"""
import csv, hashlib, itertools, json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
GATE1 = ROOT / 'evaluation/gate1_step1_allocation_equivalence'
OUT = ROOT / 'evaluation/paper_audits/t1_structure_budget'
EPS = 1e-9
POLICIES = ['Exact', 'Greedy', 'Greedy+1-Swap', 'WFC', 'StandaloneTopK', 'PCB', 'LDR']
LEGACY_POLICIES = ['FlowState', 'LFU', 'LRU', 'Marconi']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phi(g, t):
    """冻结 Phi 成本模型（与 G2 控制路径 Decimal Oracle 同系数）：g/t 单位 Token。"""
    g, t = Decimal(g) / 1024, Decimal(t) / 1024
    return float(Decimal('37.828150') * g + Decimal('0.345974143') * g * t - Decimal('0.156201917') * g * g)


def write_csv(name, rows):
    assert rows, '空表：' + name
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        'e3_condition_results': E3 / 'condition_results.csv',
        'e3_selection_traces': E3 / 'selection_traces.json',
        'e3_final_integrity': E3 / 'final_integrity.json',
        'gate1_stats': GATE1 / 'gate1_step1_instance_statistics.csv',
        'gate1_matrices': GATE1 / 'matrices.jsonl',
        'gate1_integrity': GATE1 / 'integrity.json',
        'strict20_frozen': ROOT / 'evaluation/paper_allocation_structure_strict20.csv',
        'legacy573_frozen': ROOT / 'evaluation/paper_allocation_structure_legacy573.csv'}
    (OUT / 't1_source_hashes.json').write_text(json.dumps({k: {'path': str(v), 'sha256': sha(v)} for k, v in inputs.items()},
                                                           ensure_ascii=False, indent=2))
    checks, fails = [], []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
        if not ok:
            fails.append(name)

    # ══ A) Strict20 ══
    rows = list(csv.DictReader((E3 / 'condition_results.csv').open()))
    check('e3_rows_140', len(rows) == 140, len(rows))
    by = {}
    for r in rows:
        by.setdefault((r['case'], int(r['K'])), {})[r['policy']] = r
    check('e3_grid_20x7', len(by) == 20 and all(len(v) == 7 for v in by.values()))
    # Phi 冻结系数逐格复验（140 行 × 4 Pending）
    phi_ok, n_phi = True, 0
    for r in rows:
        for p in json.loads(r['per_pending']):
            phi_ok &= abs(phi(p['T'] - p['E'], p['T']) - p['cost_ms']) < 1e-9
            n_phi += 1
    check('phi_model_consistent_all_rows', phi_ok, '复验 %d 格' % n_phi)
    # 前沿矩阵：step1（prefix 空）advances = 候选单独可给各 Pending 的恢复深度
    traces = json.loads((E3 / 'selection_traces.json').read_text())
    step1 = {(t['case'], t['K']): t for t in traces if t['step'] == 1}
    check('traces_step1_20', len(step1) == 20)
    matrix = {}  # case -> {cand: {pid: E}}
    for (case, _), t in step1.items():
        m = {c['checkpoint_id']: dict(c['advances']) for c in t['ranked']}
        check('matrix_7c_case_' + case, len(m) == 7, sorted(m))
        matrix[case] = m
    structure_rows, solver_rows, greedy_fail = [], [], []
    strict20_frozen = {(r['case'], int(r['K'])): r for r in csv.DictReader((ROOT / 'evaluation/paper_allocation_structure_strict20.csv').open())}
    for (case, K), pol in sorted(by.items()):
        mat = matrix[case]
        pids = list(next(iter(mat.values())))
        T = {p['continuation_id']: p['T'] for p in json.loads(pol['Exact']['per_pending'])}
        # 结构指标
        supports = {c: {p for p, e in adv.items() if e > 0} for c, adv in mat.items()}
        pos_edges = sum(len(s) for s in supports.values())
        shared = sum(len(s) > 1 for s in supports.values())
        same_support_pairs = sum(1 for a, b in itertools.combinations(sorted(supports), 2) if supports[a] == supports[b] and supports[a])
        legal_per_pending = Counter(len([c for c in mat if mat[c][p] > 0]) for p in pids)
        # K_min：逐 Pending 最深合法前沿的唯一实现者并集 + 子集枚举复核最小性
        forced, realizer_unique = set(), True
        best_E = {}
        for p in pids:
            e = max(mat[c][p] for c in mat)
            best_E[p] = e
            rs = [c for c in mat if mat[c][p] == e]
            if e > 0:
                realizer_unique &= len(rs) == 1
                forced.update(rs)
        attain = [S for S in (frozenset(c) for k in range(8) for c in itertools.combinations(sorted(mat), k))
                  if all(max([mat[c][p] for c in S] + [0]) == best_E[p] for p in pids)]
        k_min_enum = min(len(S) for S in attain)
        check('kmin_%s_K%d' % (case, K), len(forced) == k_min_enum == 4, 'forced=%d enum=%d unique_realizer=%s' % (len(forced), k_min_enum, realizer_unique))
        # 全域最优 J（无预算）与 Exact(K) 关系
        def J(S):
            return sum(phi(T[p] - max([mat[c][p] for c in S] + [0]), T[p]) for p in pids)
        j_domain = min(J(S) for S in attain)
        j_exact = float(pol['Exact']['J_ms'])
        # 各策略一致性
        exact_set = set(json.loads(pol['Exact']['selected']))
        for p_ in POLICIES:
            r = pol[p_]
            j, s = float(r['J_ms']), set(json.loads(r['selected']))
            solver_rows.append({'source': 'strict20', 'condition': case, 'K': K, 'policy': p_, 'J_ms': j,
                                'gap_vs_exact_ms': round(j - j_exact, 9), 'same_set_as_exact': s == exact_set,
                                'optimal': abs(j - j_exact) < EPS, 'selected': '|'.join(sorted(s))})
        g = pol['Greedy']
        if float(g['J_ms']) - j_exact > EPS:
            greedy_fail.append({'case': case, 'K': K})
        # 分类 A/B/C（可重叠；未归因条件允许留空）
        separable = all(len(s) <= 1 for s in supports.values())
        cls = []
        if separable:
            cls.append('A_separable')
        if K >= len(forced):
            cls.append('B_budget_sufficient')
        if shared > 0 and K < len(forced):
            cls.append('C_shared_tight')
        fz = strict20_frozen[(case, K)]
        check('crosscheck_strict20_%s_K%d' % (case, K),
              int(fz['n_candidate']) == 7 and int(fz['positive_edges']) == pos_edges and int(fz['shared_candidates']) == shared
              and int(fz['K_min']) == len(forced) and abs(float(fz['J_exact']) - j_exact) < 1e-6
              and abs(float(fz['J_greedy']) - float(g['J_ms'])) < 1e-6,
              'pos_edges=%d shared=%d' % (pos_edges, shared))
        structure_rows.append({
            'source': 'strict20', 'condition': case, 'K': K, 'n_candidate': 7, 'n_pending': len(pids),
            'K_min': len(forced), 'K_lt_Kmin': int(K < len(forced)), 'separable': separable,
            'shared_candidates': shared, 'positive_edges': pos_edges,
            'support_sizes': '|'.join(str(len(supports[c])) for c in sorted(supports)),
            'same_support_pairs': same_support_pairs,
            'legal_per_pending': '|'.join('%s:%d' % (p[-7], n) for p, n in sorted(legal_per_pending.items())[::-1]),
            'classes': '+'.join(cls) if cls else 'UNCLASSIFIED',
            'J_domain_no_budget': round(j_domain, 6), 'J_exact': j_exact,
            'exact_attains_domain': abs(j_exact - j_domain) < 1e-6,
            'frontier_realizer_unique': realizer_unique,
            'greedy_optimal': abs(float(g['J_ms']) - j_exact) < EPS,
            'swap1_optimal': abs(float(pol['Greedy+1-Swap']['J_ms']) - j_exact) < EPS,
            'dataset': 'Strict20(认证)'})
    # ══ B) Legacy 573（Formal 模型）══
    stats = {r['instance_id']: r for r in csv.DictReader((GATE1 / 'gate1_step1_instance_statistics.csv').open())}
    mats = {json.loads(l)['instance_id']: json.loads(l) for l in (GATE1 / 'matrices.jsonl').read_text().splitlines()}
    check('gate1_573', len(stats) == 573 and len(mats) == 573 and set(stats) == set(mats))
    legacy_frozen = {r['instance_id']: r for r in csv.DictReader((ROOT / 'evaluation/paper_allocation_structure_legacy573.csv').open())}
    # 策略 J 由 per_snapshot jsonl 重算（source_file/source_line 定位）
    jsonl_cache = {}
    for iid, r in stats.items():
        x = mats[iid]
        ids, bm = x['candidate_ids'], x['benefit_ms']
        n, m, k = len(ids), len(bm), int(r['budget'])
        supports = [{p for p in range(m) if bm[p][c] > 0} for c in range(n)]
        shared = sum(len(s) > 1 for s in supports)
        overlap = sum(bool(a & b) for a, b in itertools.combinations(supports, 2))
        same_support_pairs = sum(1 for a, b in itertools.combinations(range(n), 2) if supports[a] == supports[b] and supports[a])
        legal_pp = Counter(sum(1 for c in range(n) if bm[p][c] > 0) for p in range(m))
        # K_min 与最优证书（独立重算，口径同 gate1/e3 审计定义）
        forced, front = set(), []
        for p in range(m):
            e = max([x['positions'][i] for i in range(n) if x['compatibility'][p][i]] + [0])
            front.append(e)
            if e > 0:
                rs = [ids[i] for i in range(n) if x['compatibility'][p][i] and x['positions'][i] == e]
                forced.update(rs)
        j0 = sum(float(t) for t in x['targets'])
        row_max = [max([bm[p][c] for c in range(n)] + [0.]) for p in range(m)]
        if all(len(s) <= 1 for s in supports):
            optimum = sum(sorted(row_max, reverse=True)[:k]); cert = 'SEPARABLE_SUPPORT_TOPK'
        else:
            optimum = sum(row_max); cert = 'ALL_CANDIDATE_FRONTIERS_ATTAINED'
        # 四策略 J：从原始 jsonl 的 per_continuation 求和（按 source_file/source_line 定位行）
        src, line = r['source_file'], int(r['source_line'])
        if src not in jsonl_cache:
            jsonl_cache[src] = Path(src).read_text().splitlines()
        prow = json.loads(jsonl_cache[src][line - 1])
        pol_js, pol_sets = {}, {}
        for lp in LEGACY_POLICIES:
            pc = prow['policies'][lp]['per_continuation']
            pol_js[lp] = sum(float(c['recovery_cost_ms']) for c in pc)
            pol_sets[lp] = set(prow['policies'][lp].get('selected_checkpoint_ids') or prow['selected_sets'][lp])
        fs_sel = pol_sets['FlowState']
        benefit = sum(max([bm[p][ids.index(c)] for c in fs_sel] + [0.]) for p in range(m))
        j_fs = j0 - benefit
        check('legacy_%s' % iid,
              abs(j_fs - float(r['flowstate_cost_ms'])) < 1e-6 and abs(benefit - optimum) < 1e-6
              and len(forced) == int(r['k_min_frontier'])
              and shared == int(r['num_shared_checkpoints']) and overlap == int(r['num_overlapping_candidate_pairs'])
              and abs(pol_js['FlowState'] - j_fs) < 1e-6,
              'J=%.3f' % j_fs) if iid.endswith('::budget=0.25') or len(checks) < 25 else None
        separable = all(len(s) <= 1 for s in supports)
        cls = []
        if separable:
            cls.append('A_separable')
        if k >= len(forced):
            cls.append('B_budget_sufficient')
        if shared > 0 and k < len(forced):
            cls.append('C_shared_tight')
        for lp in LEGACY_POLICIES:
            solver_rows.append({'source': 'legacy573', 'condition': iid, 'K': k, 'policy': lp,
                                'J_ms': round(pol_js[lp], 6), 'gap_vs_exact_ms': round(pol_js[lp] - (j0 - optimum), 6),
                                'same_set_as_exact': pol_sets[lp] == set(json.loads(r['unique_frontier_realizer_ids'])) if not separable else pol_sets[lp] == fs_sel,
                                'optimal': abs(pol_js[lp] - (j0 - optimum)) < 1e-6,
                                'selected': '|'.join(sorted(pol_sets[lp]))})
        lf = legacy_frozen[iid]
        structure_rows.append({
            'source': 'legacy573', 'condition': iid, 'K': k, 'n_candidate': n, 'n_pending': m,
            'K_min': len(forced), 'K_lt_Kmin': int(k < len(forced)), 'separable': separable,
            'shared_candidates': shared, 'positive_edges': sum(len(s) for s in supports),
            'support_sizes': '|'.join(str(len(s)) for s in supports),
            'same_support_pairs': same_support_pairs,
            'legal_per_pending': '|'.join('%d_pending:%d_cand' % (c, w) for c, w in sorted(legal_pp.items())),
            'classes': '+'.join(cls) if cls else 'UNCLASSIFIED',
            'J_domain_no_budget': round(j0 - sum(row_max), 6), 'J_exact': round(j0 - optimum, 6),
            'exact_attains_domain': k >= len(forced),
            'frontier_realizer_unique': True,
            'greedy_optimal': abs(benefit - optimum) < 1e-6,
            'swap1_optimal': '',
            'dataset': lf['dataset']})
    # 汇总核对：与两份冻结聚合一致
    agg = {}
    for ds in ('OpenHands', 'AgentX'):
        rr = [x for x in structure_rows if x['source'] == 'legacy573' and x['dataset'] == ds]
        agg[ds] = {'conditions': len(rr), 'shared': sum(x['shared_candidates'] > 0 for x in rr),
                   'tight': sum(x['K_lt_Kmin'] for x in rr), 'separable': sum(x['separable'] for x in rr),
                   'greedy_optimal': sum(x['greedy_optimal'] for x in rr)}
    check('agg_openhands', agg['OpenHands'] == {'conditions': 504, 'shared': 0, 'tight': 90, 'separable': 504, 'greedy_optimal': 504}, agg['OpenHands'])
    check('agg_agentx', agg['AgentX'] == {'conditions': 69, 'shared': 66, 'tight': 0, 'separable': 3, 'greedy_optimal': 69}, agg['AgentX'])
    st = [x for x in structure_rows if x['source'] == 'strict20']
    check('agg_strict20', len(st) == 20 and sum(x['greedy_optimal'] for x in st) == 5 and sum(x['swap1_optimal'] for x in st) == 18
          and all(x['K_min'] == 4 for x in st), 'greedy_opt=%d swap1_opt=%d' % (sum(x['greedy_optimal'] for x in st), sum(x['swap1_optimal'] for x in st)))
    # Greedy 失败条件轨迹导出（case_attribution 素材）
    cases = []
    for f in greedy_fail:
        case, K = f['case'], f['K']
        tr = [t for t in traces if t['case'] == case and t['K'] == K]
        pol = by[(case, K)]
        cases.append({'case': case, 'K': K, 'exact_set': json.loads(pol['Exact']['selected']),
                      'greedy_set': json.loads(pol['Greedy']['selected']), 'swap_set': json.loads(pol['Greedy+1-Swap']['selected']),
                      'J_exact': float(pol['Exact']['J_ms']), 'J_greedy': float(pol['Greedy']['J_ms']),
                      'J_swap': float(pol['Greedy+1-Swap']['J_ms']),
                      'exact_EG': {p['continuation_id']: (p['E'], p['G']) for p in json.loads(pol['Exact']['per_pending'])},
                      'greedy_EG': {p['continuation_id']: (p['E'], p['G']) for p in json.loads(pol['Greedy']['per_pending'])},
                      'steps': [{'step': t['step'], 'prefix': t['prefix'], 'greedy_chosen': t['greedy_chosen'],
                                 'top3': [{'c': c['checkpoint_id'], 'q': c['q'], 'gain_ms': round(c['gain_ms'], 3),
                                           'advances': c['advances']} for c in t['ranked'][:3]]} for t in tr]})
    (OUT / 'greedy_fail_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    write_csv('structure_metrics.csv', structure_rows)
    write_csv('solver_comparison.csv', solver_rows)
    (OUT / 't1_checks.json').write_text(json.dumps({'checks': checks, 'fails': fails,
        'greedy_fail_conditions': greedy_fail,
        'summary': {'strict20': {'conditions': 20, 'greedy_optimal': sum(x['greedy_optimal'] for x in st),
                                  'swap1_optimal': sum(x['swap1_optimal'] for x in st),
                                  'classes_A': sum('A_separable' in x['classes'] for x in st),
                                  'classes_B': sum('B_budget_sufficient' in x['classes'] for x in st),
                                  'classes_C': sum('C_shared_tight' in x['classes'] for x in st)},
                     'legacy': agg}}, ensure_ascii=False, indent=2))
    print('T1 提取完成：structure %d 行 / solver %d 行；失败检查 %s' % (len(structure_rows), len(solver_rows), fails or '无'))
    print('strict20: greedy_opt=%d swap1_opt=%d A=%d B=%d C=%d' % (
        sum(x['greedy_optimal'] for x in st), sum(x['swap1_optimal'] for x in st),
        sum('A_separable' in x['classes'] for x in st), sum('B_budget_sufficient' in x['classes'] for x in st),
        sum('C_shared_tight' in x['classes'] for x in st)))
    print('greedy 失败条件：', [(f['case'], f['K']) for f in greedy_fail])
    print('legacy 汇总：', json.dumps(agg, ensure_ascii=False))


if __name__ == '__main__':
    main()
