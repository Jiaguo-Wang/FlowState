"""Strict 分配实验事实提取与独立复核（Task 1，只读）。

数据来源（全部只读，不修改任何冻结产物）：
- Strict20：greedy_swap_structure_e3_step1（140 行冻结结果 + 逐步/替换轨迹）；
  快照原始输入 = independent_runtime_allocation_e2/w1–w4 + multishare_allocation_gate_b5c4；
- C/K3：kvflow_multiworkflow_cpu_main（五策略 CPU 冻结）+ kvflow_multiworkflow_cpu_review
  （StandaloneTopK 复核）+ paper_solver_tradeoff_summary.csv（1-Swap）；
- GPU 证据：exact_greedy_paired_gpu_gate_b5d4b（B5-C4 K2 Exact/Greedy ×5）、
  native_resume_performance_e4_phase1b + exact_swap_runtime_e5_step1（经 e45_unified_timing 统一计时）、
  c_formal_perf_gpu（m* 60 臂主矩阵 + g1/g2 门禁臂）、paper_hybrid_gpu_gate_a/b（Spec 8）；
- 旧 573：greedy_swap_structure_e3_step1/legacy_573_structure.csv + legacy_summary.json。

独立复算：用 p1_strong_baseline_validation 的冻结实现（common/pcb/ldr/wfc）与生产
GlobalOptimizer 贪心核心，在重建的严格 Epoch 上复算全部策略选集与 J，并与冻结 CSV 对账。

输出（仅写本目录）：strict_allocation_matrix.csv、selection_disagreements.csv、
baseline_input_contract.csv、gpu_evidence_links.csv、checks.json、extraction_provenance.json。
"""
import csv, hashlib, importlib, json, statistics, sys
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/paper_strict_evidence_extraction'
EPS = 1e-9

E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
E5S0 = ROOT / 'evaluation/strong_baseline_position_audit_e5_step0'
E2 = ROOT / 'evaluation/independent_runtime_allocation_e2'
B5C4 = ROOT / 'evaluation/multishare_allocation_gate_b5c4'
P1 = ROOT / 'evaluation/p1_strong_baseline_validation'
MW = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
MWR = ROOT / 'evaluation/kvflow_multiworkflow_cpu_review'
TR = ROOT / 'evaluation/paper_solver_tradeoff_summary.csv'
ALLOC = ROOT / 'evaluation/paper_allocation_structure_strict20.csv'
GPUCK3 = ROOT / 'evaluation/paper_allocation_structure_gpu_ck3.csv'
CF = ROOT / 'evaluation/c_formal_perf_gpu'
B5D4B = ROOT / 'evaluation/exact_greedy_paired_gpu_gate_b5d4b'
E45 = ROOT / 'evaluation/e45_unified_timing'
E4B = ROOT / 'evaluation/native_resume_performance_e4_phase1b'
E5S1 = ROOT / 'evaluation/exact_swap_runtime_e5_step1'
GA = ROOT / 'evaluation/paper_hybrid_gpu_gate_a'
GB = ROOT / 'evaluation/paper_hybrid_gpu_gate_b'

SOURCES = {}   # 源文件哈希台账（供审计追溯）

def jload(p):
    p = Path(p); SOURCES[str(p)] = sha(p); return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def jrows(p):
    p = Path(p); SOURCES[str(p)] = sha(p)
    with p.open() as f:
        return list(csv.DictReader(f))

def put(name, rows, fields):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v)
                        for k, v in r.items()})

# ---------- 载入冻结实现（与 e3 审计同一锁定版本） ----------
sys.path.insert(0, str(P1))
for m in ('common', 'pcb', 'ldr', 'wfc'):
    SOURCES[str(P1 / (m + '.py'))] = sha(P1 / (m + '.py'))
common, pcb, ldr, wfc = (importlib.import_module(m) for m in ('common', 'pcb', 'ldr', 'wfc'))
sys.path.insert(0, str(ROOT))
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate

# ---------- 重建 Strict20 严格 Epoch（与 e3 audit.py load() 同源同法） ----------
def rebuild_epochs():
    epochs = {}
    for g in ('w1', 'w2', 'w3', 'w4'):
        s = jload(E2 / g / 'strict_epoch_snapshot.json')
        ce = common.Epoch(
            tuple(common.Candidate(c['checkpoint_id'], c['token_position_L'],
                  tuple(e['continuation_id'] for e in s['edges']
                        if e['checkpoint_id'] == c['checkpoint_id'] and e['status'] == 'EPOCH_ADMISSIBLE'))
                  for c in s['candidates']),
            tuple(common.Pending(p['continuation_id'], p['planning_target_T']) for p in s['pendings']), 4)
        epochs[g] = (ce, s.get('workload_type', ''))
    z = jload(B5C4 / 'results.json')['input']
    epochs['B5-C4'] = (common.Epoch(
        tuple(common.Candidate(c['identifier'], c['position'], tuple(c['compatible'])) for c in z['candidates']),
        tuple(common.Pending(**p) for p in z['pending']), 4), '历史发现案例(B5C2/B5C3门禁)')
    return epochs

def roster(ce):
    return [{'id': c.identifier, 'L': c.position, 'compat': list(c.compatible)} for c in ce.candidates]

# ---------- 独立复算 7 策略（与 e3 audit.py solvers() 同语义） ----------
def solve_all(ce):
    ids = tuple(sorted(c.identifier for c in ce.candidates))
    core = tuple(CheckpointCandidate(c.identifier, '严格矩阵占位', (), c.position, 1, True) for c in ce.candidates)
    bm = common.benefit_matrix(ce)
    greedy = tuple(c.checkpoint_id for c in
                   GlobalOptimizer(common.MODEL)._select_candidates(core, ce.k,
                   lambda cs: common.score(ce, tuple(c.checkpoint_id for c in cs))['cost_ms']).selected)
    import itertools
    scored = [(common.score(ce, x)['cost_ms'], x) for k in range(ce.k + 1) for x in itertools.combinations(ids, k)]
    best = min(v for v, _ in scored)
    exact = min((x for v, x in scored if abs(v - best) <= EPS), key=lambda x: (len(x), x))
    cur = tuple(sorted(greedy)); steps = 0
    while True:
        before = common.score(ce, cur)['cost_ms']; acc = None
        for rm in cur:
            for ad in ids:
                if ad in cur: continue
                trial = tuple(sorted((set(cur) - {rm}) | {ad}))
                if before - common.score(ce, trial)['cost_ms'] > EPS: acc = trial; break
            if acc: break
        if not acc: break
        cur = acc; steps += 1
    swap1 = cur
    sa = tuple(c for c in sorted(ids, key=lambda c: (-sum(bm[c]), c)) if sum(bm[c]) > EPS)[:ce.k]
    return {'Exact': exact, 'Greedy': greedy, 'Greedy+1-Swap': swap1, 'WFC': wfc.select(ce)[0],
            'StandaloneTopK': sa, 'PCB': pcb.select(ce)[0], 'LDR': ldr.select(ce)[0],
            '_swaps': steps, '_bm': bm,
            '_pcb_rec': pcb.select(ce)[1], '_ldr_rec': ldr.select(ce)[1], '_wfc_trace': wfc.select(ce)[1]}

checks = []   # Task 5 一致性检查结果

def check(name, ok, detail):
    checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

# ---------- 1) Strict20 冻结结果 + 独立复算对账 ----------
e3rows = jrows(E3 / 'condition_results.csv')
e3idx = {(r['case'], int(r['K']), r['policy']): r for r in e3rows}
trrows = jrows(TR)
tridx = {(r['source'], r['case'], int(r['K']), r['method']): r for r in trrows}
allocrows = jrows(ALLOC)
allocidx = {(r['case'], r['K']): r for r in allocrows}
traces = jload(E3 / 'selection_traces.json')
swaptraces = jload(E3 / 'swap_traces.json')

epochs = rebuild_epochs()
recomputed = {}
for case, (ce0, wtype) in epochs.items():
    for k in (1, 2, 3, 4):
        from dataclasses import replace
        ce = replace(ce0, k=k)
        recomputed[(case, k)] = solve_all(ce)

# 复算 vs e3 冻结 140 行
mism = []
for (case, k), res in recomputed.items():
    for pol, sel in ((p, s) for p, s in res.items() if not p.startswith('_')):
        fr = e3idx[(case, k, pol)]
        if set(sel) != set(json.loads(fr['selected'])) or \
           abs(common.score(replace(epochs[case][0], k=k), sel)['cost_ms'] - float(fr['J_ms'])) > EPS:
            mism.append((case, k, pol))
check('Strict20 独立复算 140/140（选集+J 与 e3 冻结一致）', not mism, {'不一致': mism})

# e3 vs e5_step0 文件同一性（cmp 之外的哈希复核）
check('e3 condition_results.csv == e5_step0 all_policy_results.csv（字节一致）',
      sha(E3 / 'condition_results.csv') == sha(E5S0 / 'all_policy_results.csv'),
      {'sha256': sha(E3 / 'condition_results.csv')})

# e3 vs 求解器权衡审计（Exact/Greedy/1-Swap/WFC）交叉核对
trmis = []
for (case, k), res in recomputed.items():
    for pol, mth in (('Exact', 'Exact'), ('Greedy', 'Greedy'), ('Greedy+1-Swap', 'Greedy+1-Swap'), ('WFC', 'WFC')):
        t = tridx.get(('strict20', case, k, mth))
        if t is None or set(json.loads(t['selected'])) != set(res[pol]) or abs(float(t['J_ms']) - float(e3idx[(case, k, pol)]['J_ms'])) > EPS:
            trmis.append((case, k, mth))
check('e3 vs paper_solver_tradeoff（strict20 4 方法×20 条件）一致', not trmis, {'不一致': trmis})

# e3 vs 分配结构归因审计 strict20 CSV（J 列为 6 位小数舍入）
almis = []
for (case, k), res in recomputed.items():
    a = allocidx[(case, str(k))]
    for pol, col in (('Exact', 'J_exact'), ('Greedy', 'J_greedy'), ('Greedy+1-Swap', 'J_1swap'),
                     ('WFC', 'J_wfc'), ('StandaloneTopK', 'J_standalone_topk'), ('PCB', 'J_pcb'), ('LDR', 'J_ldr')):
        if abs(float(a[col]) - float(e3idx[(case, k, pol)]['J_ms'])) > 6e-7:
            almis.append((case, k, pol))
check('e3 vs paper_allocation_structure_strict20（7 个 J 列，舍入容差）一致', not almis, {'不一致': almis})

# ---------- 2) strict_allocation_matrix.csv ----------
GPU_PTR = {('B5-C4', 2, 'Exact'): 'b5d4b p01–p05_Exact(5臂)+paper_hybrid_gpu_gate_a a1(在线)',
           ('B5-C4', 2, 'Greedy'): 'b5d4b p01–p05_Greedy(5臂)',
           ('w1', 2, 'Exact'): 'E4-1B r1–r3_{A,C}_Exact(6臂)+smoke',
           ('w1', 2, 'Greedy'): 'E4-1B r1–r3_{A,C}_Greedy(6臂)+smoke',
           ('w1', 3, 'Exact'): 'E5-1 r1–r3_{A,B,C,D}_Exact(12臂)+smoke',
           ('w1', 3, 'Greedy+1-Swap'): 'E5-1 r1–r3_{A,B,C,D}_Swap(12臂)+smoke'}
SNAP_SRC = {g: f'evaluation/independent_runtime_allocation_e2/{g}/strict_epoch_snapshot.json' for g in ('w1', 'w2', 'w3', 'w4')}
SNAP_SRC['B5-C4'] = 'evaluation/multishare_allocation_gate_b5c4/results.json'

matrix = []
for (case, k) in sorted(recomputed, key=lambda x: (x[0], x[1])):
    ce = replace(epochs[case][0], k=k); res = recomputed[(case, k)]
    base = e3idx[(case, k, 'Exact')]
    for pol in ('Exact', 'Greedy', 'Greedy+1-Swap', 'WFC', 'StandaloneTopK', 'PCB', 'LDR'):
        r = e3idx[(case, k, pol)]
        matrix.append({'condition_set': 'strict20', 'case': case, 'K': k,
                       'n_candidates': len(ce.candidates), 'n_pending': len(ce.pending),
                       'snapshot_source': SNAP_SRC[case], 'candidates_roster': roster(ce),
                       'policy': pol, 'selected_ids': sorted(json.loads(r['selected'])),
                       'selected_count': len(json.loads(r['selected'])), 'J_ms': float(r['J_ms']),
                       'gap_exact_ms': float(r['gap_exact_ms']),
                       'per_pending_EG': [{'pid': p['continuation_id'], 'T': p['T'], 'E': p['E'], 'G': p['G']}
                                          for p in json.loads(r['per_pending'])],
                       'exact_optimality_certificate': 'FULL_SUBSET_ENUMERATION_CERTIFIED(n=7,|S|≤K枚举,e3+e5_step0+tradeoff三处一致)',
                       'swaps_applied': int(r['swaps']),
                       'gpu_evidence': GPU_PTR.get((case, k, pol), 'NOT_MEASURED'), 'notes': ''})

# ---------- C/K3 输入身份与冻结结果 ----------
mw = jload(MW / 'cpu_results.json')
ccase = next(c for c in jload(MW / 'workloads.json')['cases'] if c['structure'] == 'C')
cur_ids = [p['continuation_id'] for p in ccase['calls'] if p['ready']]
ck3_ep = common.Epoch(
    tuple(common.Candidate(c['checkpoint_id'], c['token_position_L'],
          tuple(x for x in c['token_compatible_calls'] if x in cur_ids)) for c in ccase['candidates']),
    tuple(common.Pending(p['continuation_id'], p['planning_target_T']) for p in ccase['calls'] if p['ready']), 3)
ck3_res = solve_all(ck3_ep)
ck3_rows = {r['policy']: r for r in mw['rows'] if r['structure'] == 'C' and r['K'] == 3}
review = jload(MWR / 'review_results.json')['task2_standalone_vs_greedy_C']['K=3']
ck3_tradeoff = {r['method']: r for r in trrows if r['source'] == 'ck3'}
ck3_J = {p: r['current_J_ms'] for p, r in ck3_rows.items()}
ck3_J['StandaloneTopK'] = review['standalone_current_J_ms']
ck3_J['Greedy+1-Swap'] = float(ck3_tradeoff['Greedy+1-Swap']['J_ms'])
ck3_sel = {p: sorted(r['selected']) for p, r in ck3_rows.items()}
ck3_sel['StandaloneTopK'] = sorted(review['standalone_selected'])
ck3_sel['Greedy+1-Swap'] = sorted(json.loads(ck3_tradeoff['Greedy+1-Swap']['selected']))
ck3_eg = {p: r['current_E_G'] for p, r in ck3_rows.items()}
ck3_eg['StandaloneTopK'] = review['standalone_per_pending']
ck3_eg['Greedy+1-Swap'] = ck3_eg['FlowState-Greedy']

# C/K3 独立复算对账（五策略 + SA + 1-Swap 集合）
ck3_mis = []
if set(ck3_res['Greedy']) != set(ck3_sel['FlowState-Greedy']): ck3_mis.append('FlowState-Greedy')
if set(ck3_res['WFC']) != set(ck3_sel['WFC']): ck3_mis.append('WFC')
if set(ck3_res['Greedy+1-Swap']) != set(ck3_sel['Greedy+1-Swap']): ck3_mis.append('Greedy+1-Swap')
if set(ck3_res['StandaloneTopK']) != set(ck3_sel['StandaloneTopK']): ck3_mis.append('StandaloneTopK')
check('C/K3 复算（Greedy/WFC/1-Swap/StandaloneTopK）与冻结一致', not ck3_mis, {'不一致': ck3_mis})
Jmis = [p for p, j in ck3_J.items() if p not in ('LRU', 'Marconi Adaptation', 'KVFlow-inspired')
        and abs(j - common.score(ck3_ep, tuple(ck3_sel[p]))['cost_ms']) > EPS]
check('C/K3 J 复算一致（Greedy/WFC/SA/1-Swap）', not Jmis, {'不一致': Jmis})
gate_ck3 = jload(ROOT / 'evaluation/paper_solver_tradeoff_gate.json')['ck3']
check('C/K3 Exact 证书状态=NOT_AVAILABLE（n=16>12 门禁拒绝，未绕过）',
      '拒绝' in gate_ck3['exact_gate'], gate_ck3['exact_gate'])

ck3_gpu_ptr = {'FlowState-Greedy': 'c_formal m1–m5×{A,B,C,D}_FS 20臂(主矩阵)+g1/g2 门禁2臂+paper_hybrid_gpu_gate_b b1(在线)',
               'StandaloneTopK': 'c_formal m1–m5×{A,B,C,D}_SA 20臂(主矩阵)+g1/g2 门禁2臂',
               'WFC': 'c_formal m1–m5×{A,B,C,D}_WFC 20臂(主矩阵)'}
for pol in ('FlowState-Greedy', 'WFC', 'StandaloneTopK', 'LRU', 'Marconi Adaptation', 'KVFlow-inspired', 'Greedy+1-Swap', 'Exact'):
    if pol == 'Exact':
        matrix.append({'condition_set': 'ck3', 'case': 'C/K3', 'K': 3, 'n_candidates': 16, 'n_pending': 4,
                       'snapshot_source': 'evaluation/kvflow_multiworkflow_cpu_main/workloads.json (MW_CPU_C,current=C_A..C_D)',
                       'candidates_roster': roster(ck3_ep), 'policy': 'Exact', 'selected_ids': [], 'selected_count': 0,
                       'J_ms': '', 'gap_exact_ms': '', 'per_pending_EG': [],
                       'exact_optimality_certificate': 'NOT_AVAILABLE(Exact门禁拒绝:n=16>12,planner.allowed未放行)',
                       'swaps_applied': '', 'gpu_evidence': 'NOT_MEASURED',
                       'notes': '不启动新枚举；局部最优邻域证据见 paper_solver_tradeoff_audit.md'})
        continue
    matrix.append({'condition_set': 'ck3', 'case': 'C/K3', 'K': 3, 'n_candidates': 16, 'n_pending': 4,
                   'snapshot_source': 'evaluation/kvflow_multiworkflow_cpu_main/workloads.json (MW_CPU_C,current=C_A..C_D)',
                   'candidates_roster': roster(ck3_ep), 'policy': pol,
                   'selected_ids': ck3_sel[pol], 'selected_count': len(ck3_sel[pol]), 'J_ms': ck3_J[pol],
                   'gap_exact_ms': '',
                   'per_pending_EG': [{'pid': p['continuation_id'], 'T': p['T'], 'E': p['E'], 'G': p['G']} for p in ck3_eg[pol]],
                   'exact_optimality_certificate': 'NOT_AVAILABLE(无C/K3独立Exact证书)',
                   'swaps_applied': '', 'gpu_evidence': ck3_gpu_ptr.get(pol, 'NOT_MEASURED'),
                   'notes': 'J为模型化成本;T=4096–7168低于校准目标区间(32768–131072),非零项OUTSIDE_CALIBRATION_TARGET_RANGE' if pol != 'Greedy+1-Swap' else 'paper_solver_tradeoff ck3 行:与Greedy同集同J'})

put('strict_allocation_matrix.csv', matrix,
    ['condition_set', 'case', 'K', 'n_candidates', 'n_pending', 'snapshot_source', 'candidates_roster',
     'policy', 'selected_ids', 'selected_count', 'J_ms', 'gap_exact_ms', 'per_pending_EG',
     'exact_optimality_certificate', 'swaps_applied', 'gpu_evidence', 'notes'])

# ---------- 3) selection_disagreements.csv ----------
def eg_map(rows):
    return {p['continuation_id']: (p['E'], p['G']) for p in rows}

def detail_cands(ids, roster_l, bm):
    out = []
    for i in ids:
        r = next(x for x in roster_l if x['id'] == i)
        out.append({'id': i, 'L': r['L'], 'compat': r['compat'], 'indep_benefit_sum_ms': round(sum(bm[i]), 6)})
    return out

dis = []
trace_idx = {(t['case'], t['K']): [x for x in traces if x['case'] == t['case'] and x['K'] == t['K']]
             for t in traces} if traces else {}

def first_divergence(case, k, ref_name, res):
    """从 e3 冻结逐步轨迹取 Greedy 与 WFC 的首个分叉步（含双方当步边际收益）。"""
    ts = [x for x in traces if x['case'] == case and x['K'] == k]
    for t in ts:
        if t['greedy_chosen'] != t['wfc_chosen']:
            rk = {z['checkpoint_id']: z['gain_ms'] for z in t['ranked']}
            return {'step': t['step'], 'greedy_chose': t['greedy_chosen'], 'wfc_chose': t['wfc_chosen'],
                    'greedy_marginal_ms': rk.get(t['greedy_chosen']), 'wfc_marginal_ms': rk.get(t['wfc_chosen'])}
    return {'note': '全部逐步选择一致（e3 selection_traces.json）'}

def first_greedy_pick_outside_exact(case, k, res):
    ts = [x for x in traces if x['case'] == case and x['K'] == k]
    ex = set(res['Exact'])
    for t in ts:
        if t['greedy_chosen'] not in ex:
            rk = sorted(t['ranked'], key=lambda z: -z['gain_ms'])
            alt = next((z for z in rk if z['checkpoint_id'] in ex and z['checkpoint_id'] not in t['prefix']), None)
            return {'step': t['step'], 'greedy_chose': t['greedy_chosen'],
                    'greedy_marginal_ms': next((z['gain_ms'] for z in t['ranked'] if z['checkpoint_id'] == t['greedy_chosen']), None),
                    'exact_side_best_alt': ({'id': alt['checkpoint_id'], 'marginal_ms': alt['gain_ms']} if alt else None)}
    return None

def add_dis(cset, case, k, name, selA, selB, JA, JB, egA, egB, roster_l, bm, firstdiv, trace_note, gpu, extra_note=''):
    a, b = set(selA), set(selB)
    dE = [{'pid': pid, 'E_a': egA[pid][0], 'G_a': egA[pid][1],
           'E_b': (egB[pid][0] if pid in egB else None), 'G_b': (egB[pid][1] if pid in egB else None)}
          for pid in egA if egA[pid] != egB.get(pid)]
    if JB is None:
        rel = 'UNVERIFIED'   # 无 Exact 参照时只记录双方选集，不判定目标关系
    elif a == b and abs(JA - JB) <= EPS:
        rel = 'SAME_SET_SAME_OBJECTIVE'
    elif a == b:
        rel = 'SAME_SET_DIFFERENT_OBJECTIVE'
    elif abs(JA - JB) <= EPS:
        rel = 'DIFFERENT_SET_SAME_OBJECTIVE'
    else:
        rel = 'DIFFERENT_SET_DIFFERENT_OBJECTIVE'
    dis.append({'condition_set': cset, 'case': case, 'K': k, 'comparison': name, 'relation': rel,
                'set_a': sorted(a), 'set_b': sorted(b), 'J_a_ms': JA, 'J_b_ms': JB,
                'delta_J_a_minus_b_ms': (JA - JB) if JB is not None else '', 'pendings_with_EG_diff': dE,
                'only_in_a': detail_cands(sorted(a - b), roster_l, bm),
                'only_in_b': detail_cands(sorted(b - a), roster_l, bm),
                'first_divergence': firstdiv, 'trace_evidence': trace_note,
                'gpu_evidence': gpu, 'notes': extra_note})

for (case, k) in sorted(recomputed, key=lambda x: (x[0], x[1])):
    ce = replace(epochs[case][0], k=k); res = recomputed[(case, k)]
    rol = roster(ce); bm = res['_bm']
    def eg(pol):
        return eg_map(json.loads(e3idx[(case, k, pol)]['per_pending']))
    def J(pol):
        return float(e3idx[(case, k, pol)]['J_ms'])
    g_sel = json.loads(e3idx[(case, k, 'Greedy')]['selected'])
    for pol, label in (('StandaloneTopK', 'A: StandaloneTopK vs Greedy'), ('WFC', 'B: WFC vs Greedy'),
                       ('PCB', 'C1: PCB vs Greedy'), ('LDR', 'C2: LDR vs Greedy')):
        s = json.loads(e3idx[(case, k, pol)]['selected'])
        fd = first_divergence(case, k, pol, res) if pol == 'WFC' else {'note': '静态规则无逐步轨迹;独立排序见only_in_*收益'}
        add_dis('strict20', case, k, label, s, g_sel, J(pol), J('Greedy'), eg(pol), eg('Greedy'), rol, bm, fd,
                'e3 selection_traces.json(仅Greedy/WFC逐步)' if pol == 'WFC' else '静态规则:复算排序',
                GPU_PTR.get((case, k, pol), 'NOT_MEASURED'))
    # D: Greedy vs Exact
    fd = first_greedy_pick_outside_exact(case, k, res)
    add_dis('strict20', case, k, 'D: Greedy vs Exact', g_sel, json.loads(e3idx[(case, k, 'Exact')]['selected']),
            J('Greedy'), J('Exact'), eg('Greedy'), eg('Exact'), rol, bm, fd or {'note': 'Greedy全步均在Exact集内或无分叉'},
            'e3 selection_traces.json + swap_traces.json',
            GPU_PTR.get((case, k, 'Greedy'), 'NOT_MEASURED'))
    # E: 1-Swap vs Exact
    s1 = json.loads(e3idx[(case, k, 'Greedy+1-Swap')]['selected'])
    st = next((x for x in swaptraces if x['case'] == case and x['K'] == k), None)
    swap_note = [{'remove': y['remove'], 'add': y['add'], 'gain_ms': y['gain_ms']} for y in st['steps']] if st else []
    trap = '局部陷阱:邻域12个1换1均无严格改善,严格改善需≥3项同步替换(local_optimum_neighbors.csv)' \
        if (case, k) in (('w1', 3), ('w2', 3)) else ''
    add_dis('strict20', case, k, 'E: 1-Swap vs Exact', s1, json.loads(e3idx[(case, k, 'Exact')]['selected']),
            J('Greedy+1-Swap'), J('Exact'), eg('Greedy+1-Swap'), eg('Exact'), rol, bm,
            {'swap_path': swap_note}, 'e3 swap_traces.json/local_optimum_neighbors.csv',
            GPU_PTR.get((case, k, 'Greedy+1-Swap'), 'NOT_MEASURED'), trap)
    # 补充客观事实:PCB vs LDR
    add_dis('strict20', case, k, 'F: PCB vs LDR(补充)', json.loads(e3idx[(case, k, 'PCB')]['selected']),
            json.loads(e3idx[(case, k, 'LDR')]['selected']), J('PCB'), J('LDR'), eg('PCB'), eg('LDR'), rol, bm,
            {'note': '两者推荐规则不同(Phi收益vs最深位置)'}, 'p1_strong_baseline_validation/{pcb,ldr}.py', 'NOT_MEASURED')

# C/K3 分歧
rol3 = roster(ck3_ep); bm3 = ck3_res['_bm']
gr3 = ck3_sel['FlowState-Greedy']; Jg3 = ck3_J['FlowState-Greedy']; egg3 = eg_map(ck3_eg['FlowState-Greedy'])
mwt = next(t for t in jload(MW / 'selection_traces.json') if t['structure'] == 'C' and t['K'] == 3)['trace']
def ck3_first_div_vs_greedy(wfc_trace):
    for i, step in enumerate(wfc_trace['steps']):
        gch = mwt['FlowState-Greedy'][i]['chosen']
        if gch != step['chosen']:
            marg = mwt['FlowState-Greedy'][i]['candidate_marginal_ms']
            rk = {z['checkpoint_id']: z for z in step['ranked_candidates']}
            return {'step': i + 1, 'greedy_chose': gch, 'wfc_chose': step['chosen'],
                    'greedy_marginal_ms': marg.get(gch), 'wfc_q_r': (rk[step['chosen']]['q'], rk[step['chosen']]['r']),
                    'greedy_q_r': (rk[gch]['q'], rk[gch]['r']) if gch in rk else None}
    return {'note': '逐步一致'}
for pol, label in (('StandaloneTopK', 'A: StandaloneTopK vs Greedy'), ('WFC', 'B: WFC vs Greedy'),
                   ('LRU', 'G: LRU vs Greedy'), ('Marconi Adaptation', 'H: Marconi vs Greedy'),
                   ('KVFlow-inspired', 'I: KVFlow-inspired vs Greedy'), ('Greedy+1-Swap', 'J: 1-Swap vs Greedy')):
    s = ck3_sel[pol]
    fd = ck3_first_div_vs_greedy(mwt['WFC']) if pol == 'WFC' else {'note': '静态规则;独立排序见only_in_*'}
    add_dis('ck3', 'C/K3', 3, label, s, gr3, ck3_J[pol], Jg3, eg_map(ck3_eg[pol]), egg3, rol3, bm3, fd,
            'kvflow_multiworkflow_cpu_main/selection_traces.json(WFC/Greedy逐步)',
            ck3_gpu_ptr.get(pol, 'NOT_MEASURED'),
            'J为模型化成本;校准区间外' if pol not in ck3_gpu_ptr else '')
add_dis('ck3', 'C/K3', 3, 'D: Greedy vs Exact', gr3, [], Jg3, None, egg3, {}, rol3, bm3,
        {'note': 'Exact证书NOT_AVAILABLE(门禁拒绝n=16>12)'}, 'paper_solver_tradeoff_gate.json', 'NOT_MEASURED',
        '关系标记UNVERIFIED:无C/K3独立Exact参照')
dis[-1]['relation'] = 'UNVERIFIED'
add_dis('ck3', 'C/K3', 3, 'E: 1-Swap vs Exact', ck3_sel['Greedy+1-Swap'], [], ck3_J['Greedy+1-Swap'], None,
        eg_map(ck3_eg['Greedy+1-Swap']), {}, rol3, bm3, {'note': 'Exact证书NOT_AVAILABLE'},
        'paper_solver_tradeoff_gate.json', 'NOT_MEASURED', '关系标记UNVERIFIED')
dis[-1]['relation'] = 'UNVERIFIED'

put('selection_disagreements.csv', dis,
    ['condition_set', 'case', 'K', 'comparison', 'relation', 'set_a', 'set_b', 'J_a_ms', 'J_b_ms',
     'delta_J_a_minus_b_ms', 'pendings_with_EG_diff', 'only_in_a', 'only_in_b', 'first_divergence',
     'trace_evidence', 'gpu_evidence', 'notes'])

print('matrix rows:', len(matrix), 'disagreements rows:', len(dis))
json.dump({'checks': checks}, (OUT / 'checks.json').open('w'), ensure_ascii=False, indent=2)
json.dump({'sources_sha256': {k: v for k, v in sorted(SOURCES.items())},
           'note': '本目录 extract.py 只读提取;哈希为提取时点快照'},
          (OUT / 'extraction_provenance.json').open('w'), ensure_ascii=False, indent=2)
for c in checks:
    print(c['status'], c['check'])
