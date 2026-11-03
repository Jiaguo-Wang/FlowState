"""只读复核冻结结果；不调用任何策略求解器，仅补充结构统计与局部邻域证书。"""
import csv
import hashlib
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EPS = 1e-9
sources = {}
checks = []

def read(path):
    p = ROOT / path
    b = p.read_bytes()
    sources[str(p.relative_to(ROOT))] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return b.decode()

def js(path):
    return json.loads(read(path))

def ck(label, condition):
    checks.append({'check': label, 'pass': bool(condition)})
    if not condition:
        raise ValueError('核验失败：' + label)

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def csvout(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})

def phi(g, t):
    g, t = g / 1024, t / 1024
    return 37.82815 * g + .345974143 * g * t - .156201917 * g * g

def matrices(inp):
    cs, ps = inp['candidates'], inp['pending']
    es = {c['identifier']: [c['position'] if p['identifier'] in c['compatible'] and c['position'] <= p['target'] else 0 for p in ps] for c in cs}
    bs = {c: [phi(p['target'], p['target']) - phi(p['target'] - e, p['target']) for p, e in zip(ps, ee)] for c, ee in es.items()}
    return es, bs

def evaluate(inp, es, selected):
    e = [max([es[c][i] for c in selected] + [0]) for i in range(len(inp['pending']))]
    g = [p['target'] - ee for p, ee in zip(inp['pending'], e)]
    return e, g, sum(phi(gg, p['target']) for gg, p in zip(g, inp['pending']))

def metrics(inp, es, bs):
    ids = list(es)
    m = len(inp['pending'])
    ss = {c: {i for i, b in enumerate(bs[c]) if b > EPS} for c in ids}
    best = [max([es[c][i] for c in ids] + [0]) for i in range(m)]
    masks = {c: sum(1 << i for i in range(m) if best[i] > 0 and es[c][i] == best[i]) for c in ids}
    goal = sum(1 << i for i in range(m) if best[i] > 0)
    # 最深前沿覆盖的动态规划，仅有至多八个需求位，不枚举大候选域的分配集合。
    dp = {0: []}
    for c, mask in masks.items():
        for s, witness in list(dp.items()):
            nxt = s | mask
            if nxt not in dp or len(witness) + 1 < len(dp[nxt]):
                dp[nxt] = witness + [c]
    kmin = len(dp[goal])
    # 独立下界证书：这些冻结实例的非空最大覆盖掩码两两不交，任何候选最多覆盖一块。
    unique = set(masks.values()) - {0}
    maximal = [a for a in unique if not any(a != b and a & b == a for b in unique)]
    independent = all(a & b == 0 for a, b in itertools.combinations(maximal, 2)) and len(maximal) == kmin
    ck('最深前沿覆盖独立证书', independent)
    pairs = list(itertools.combinations(ids, 2))
    return {'n_candidate': len(ids), 'n_pending': m, 'K_min': kmin, 'K_min_witness': dp[goal],
            'full_frontiers': best, 'J_full_ms': evaluate(inp, es, ids)[2],
            'positive_supports': {c: [inp['pending'][i]['identifier'] for i in sorted(s)] for c, s in ss.items()},
            'support_size_distribution': dict(Counter(map(len, ss.values()))),
            'shared_candidates': sum(len(s) > 1 for s in ss.values()),
            'shared_fraction': sum(len(s) > 1 for s in ss.values()) / len(ids),
            'positive_edges': sum(map(len, ss.values())),
            'legal_candidates_per_pending': {p['identifier']: sum(p['identifier'] in c['compatible'] and c['position'] <= p['target'] for c in inp['candidates']) for p in inp['pending']},
            'overlap_pairs': sum(bool(ss[a] & ss[b]) for a, b in pairs),
            'equal_nonempty_support_pairs': sum(bool(ss[a]) and ss[a] == ss[b] for a, b in pairs),
            'crossing_pairs': sum(bool(ss[a] & ss[b]) and not ss[a] <= ss[b] and not ss[b] <= ss[a] for a, b in pairs),
            'separable': all(len(s) <= 1 for s in ss.values())}

E3 = 'evaluation/greedy_swap_structure_e3_step1/'
P1 = 'evaluation/p1_strong_baseline_validation/'
manifest_checks = []
for path in ['evaluation/independent_runtime_allocation_e2/evidence_manifest.json', E3 + 'evidence_manifest.json', P1 + 'artifact_manifest.json', 'evaluation/multishare_allocation_gate_b5c4/evidence_manifest.json']:
    manifest = js(path)
    for name, expected in manifest['files'].items():
        p = Path(path).parent / name
        actual = hashlib.sha256((ROOT / p).read_bytes()).hexdigest() if (ROOT / p).exists() else 'MISSING'
        manifest_checks.append({'manifest': path, 'file': str(p), 'expected': expected, 'actual': actual, 'match': actual == expected})

legacy = [json.loads(l) for l in read(P1 + 'per_case_results.jsonl').splitlines()]
gate = [json.loads(l) for l in read('evaluation/gate1_step1_allocation_equivalence/matrices.jsonl').splitlines()]
gate = {(r['source_file'], r['source_line']): r for r in gate}
originals = {}
structures, solvers, detail = [], [], {}
legacy_equal = Counter()
for line, r in enumerate(legacy, 1):
    inp, k = r['input'], r['k']
    case = r['snapshot_id'] + '::budget=' + str(r['budget_ratio'])
    es, bs = matrices(inp)
    ck(case + '收益矩阵', all(abs(a - b) <= EPS for c in bs for a, b in zip(bs[c], r['benefit_matrix_ms'][c])))
    old = gate[(r['source_file'], r['source_line'])]
    ck(case + 'Gate1矩阵', all(abs(bs[c][i] - old['benefit_ms'][i][j]) <= EPS for j, c in enumerate(old['candidate_ids']) for i in range(len(inp['pending']))))
    if r['source_file'] not in originals:
        originals[r['source_file']] = [json.loads(l) for l in read(r['source_file']).splitlines()]
    original = originals[r['source_file']][r['source_line'] - 1]
    ck(case + '原始身份', original['snapshot_id'] == r['snapshot_id'] and original['k'] == k and original['snapshot_digest'] == r['snapshot_digest'])
    mm = metrics(inp, es, bs)
    ck(case + '等大小预算', len(set(old['checkpoint_bytes'])) == 1 and old['budget_bytes'] // old['checkpoint_bytes'][0] == k)
    base = sum(phi(p['target'], p['target']) for p in inp['pending'])
    if mm['separable']:
        optimum = base - sum(sorted([max(bs[c][i] for c in bs) for i in range(len(inp['pending']))], reverse=True)[:k])
        cert = 'SEPARABLE_TOPK_PROOF'
    else:
        ck(case + '共享条件预算足够', k >= mm['K_min'])
        optimum, cert = mm['J_full_ms'], 'FULL_FRONTIER_LOWER_BOUND'
    pols = r['policies']
    reference = pols['FlowState']
    for policy in ['FlowState', 'PCB', 'LDR', 'WFC']:
        p = pols[policy]
        e, g, j = evaluate(inp, es, p['selected'])
        ck(case + policy + '目标', abs(j - p['cost_ms']) <= EPS and abs(j - optimum) <= EPS)
        ck(case + policy + '逐请求', e == [x['E'] for x in p['per_pending']] and g == [x['G'] for x in p['per_pending']])
        # 强基线来自独立 P1 结果；原始历史文件只核对原有 FlowState。
        if policy == 'FlowState':
            op = original['policies']['FlowState']
            ck(case + '原始选集与成本', set(op['selected_checkpoint_ids']) == set(p['selected']) and abs(op['total_recovery_cost_ms'] - j) <= EPS)
        same = set(p['selected']) == set(reference['selected'])
        equal = abs(p['cost_ms'] - reference['cost_ms']) <= EPS
        category = 'SAME_SET_SAME_COST' if same and equal else 'DIFFERENT_SET_SAME_COST' if equal else 'DIFFERENT_COST'
        legacy_equal[policy + ':' + category] += 1
        solvers.append({'dataset': r['population'] + '_FORMAL', 'case': case, 'K': k, 'policy': policy, 'selected': p['selected'], 'E': e, 'G': g, 'J_ms': j, 'optimum_ms': optimum, 'gap_ms': j - optimum, 'optimal': abs(j - optimum) <= EPS, 'same_set_as_exact': 'UNKNOWN_NO_ENUMERATED_REPRESENTATIVE', 'relation_to_reference': category, 'optimum_evidence': cert, 'source_row': str(line), 'source': P1 + 'per_case_results.jsonl'})
    structures.append({'dataset': r['population'] + '_FORMAL', 'case': case, 'K': k, **mm, 'budget_relation': '<' if k < mm['K_min'] else '=' if k == mm['K_min'] else '>', 'strict_runtime_certification': 'UNKNOWN_NOT_CERTIFIED', 'classes': ('A;' if mm['separable'] else '') + ('B;' if k >= mm['K_min'] else '') + ('C;' if mm['shared_candidates'] and k < mm['K_min'] else ''), 'greedy_gap_ms': reference['cost_ms'] - optimum, 'swap_gap_ms': 'UNKNOWN_NOT_RUN', 'source': P1 + 'per_case_results.jsonl', 'source_row': line})

rows = list(csv.DictReader(read(E3 + 'condition_results.csv').splitlines()))
traces = js(E3 + 'selection_traces.json')
swaps = js(E3 + 'swap_traces.json')
traps = js(E3 + 'local_trap_diagnostics.json')
frozen_c4 = js('evaluation/multishare_allocation_gate_b5c4/results.json')
ck('冻结Phi', frozen_c4['cost_model']['coefficient_a'] == 37.82815 and frozen_c4['cost_model']['coefficient_b'] == .345974143 and frozen_c4['cost_model']['coefficient_c'] == -.156201917)
by = {}
for number, r in enumerate(rows, 2):
    r['csv_line'] = number
    by.setdefault((r['case'], int(r['K'])), {})[r['policy']] = r
for case in ['w1', 'w2', 'w3', 'w4', 'B5-C4']:
    path = 'evaluation/independent_runtime_allocation_e2/' + case + '/strict_epoch_snapshot.json' if case != 'B5-C4' else 'evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json'
    epoch = js(path)
    candidates = epoch['candidates']
    pendings = epoch['pendings']
    edges = {(e['checkpoint_id'], e['continuation_id']): e for e in epoch['edges']}
    ck(case + '完整严格关系', len(edges) == len(candidates) * len(pendings) and all(e['status'] in ['EPOCH_ADMISSIBLE', 'REJECTED'] for e in edges.values()))
    inp = {'candidates': [{'identifier': c['checkpoint_id'], 'position': c['token_position_L'], 'compatible': [p['continuation_id'] for p in pendings if edges[(c['checkpoint_id'], p['continuation_id'])]['status'] == 'EPOCH_ADMISSIBLE']} for c in candidates], 'pending': [{'identifier': p['continuation_id'], 'target': p['planning_target_T']} for p in pendings]}
    for c in candidates:
        for p in pendings:
            edge = edges[(c['checkpoint_id'], p['continuation_id'])]
            if edge['status'] == 'EPOCH_ADMISSIBLE':
                ck(case + '准入身份', all(edge['conditions'].values()) and c['barrier_id'] == p['barrier_id'] == epoch['epoch_id'] and c['configuration_digest'] == p['configuration_digest'] == epoch['configuration_digest'])
    es, bs = matrices(inp)
    for c in candidates:
        for p in pendings:
            if edges[(c['checkpoint_id'], p['continuation_id'])]['status'] == 'EPOCH_ADMISSIBLE':
                ck(case + '原始Token前缀', c['token_ids'] == p['token_ids'][:c['token_position_L']] and len(c['token_ids']) == c['token_position_L'])
    mm = metrics(inp, es, bs)
    detail[case] = {'epoch_id': epoch['epoch_id'], 'input': inp, 'E_matrix': es, 'benefit_matrix_ms': bs, 'metrics': mm, 'conditions': []}
    for k in range(1, 5):
        pp = by[case, k]
        exact = pp['Exact']
        optimum = float(exact['J_ms'])
        exact_set = set(json.loads(exact['selected']))
        for policy, r in pp.items():
            selected = json.loads(r['selected'])
            e, g, j = evaluate(inp, es, selected)
            ck(case + str(k) + policy + '独立目标复核', len(selected) <= k and len(set(selected)) == len(selected) and abs(j - float(r['J_ms'])) <= EPS)
            per = json.loads(r['per_pending'])
            ck(case + str(k) + policy + '前沿复核', e == [x['E'] for x in per] and g == [x['G'] for x in per])
            same = set(selected) == exact_set
            equal = abs(j - optimum) <= EPS
            solvers.append({'dataset': 'Strict20', 'case': case, 'K': k, 'policy': policy, 'selected': selected, 'E': e, 'G': g, 'J_ms': j, 'optimum_ms': optimum, 'gap_ms': j - optimum, 'optimal': equal, 'same_set_as_exact': same, 'relation_to_reference': 'SAME_SET_SAME_COST' if same and equal else 'DIFFERENT_SET_SAME_COST' if equal else 'DIFFERENT_COST', 'optimum_evidence': 'FROZEN_EXACT', 'source_row': str(r['csv_line']), 'source': E3 + 'condition_results.csv'})
        tt = [t for t in traces if t['case'] == case and t['K'] == k]
        for t in tt:
            before_e, _, before_j = evaluate(inp, es, t['prefix'])
            for a in t['ranked']:
                after_e, _, after_j = evaluate(inp, es, t['prefix'] + [a['checkpoint_id']])
                ck(case + str(k) + '冻结边际轨迹', abs(before_j - after_j - a['gain_ms']) <= EPS and list(a['advances'].values()) == [x-y for x,y in zip(after_e,before_e)])
            ck(case + str(k) + 'WFC赢家一致', t['wfc_chosen'] == t['greedy_chosen'] and next((a['q'], a['r']) for a in t['ranked'] if a['checkpoint_id'] == t['wfc_chosen']) == max((a['q'], a['r']) for a in t['ranked']))
            ck(case + str(k) + '冻结Greedy步', abs(next(a['gain_ms'] for a in t['ranked'] if a['checkpoint_id'] == t['greedy_chosen']) - max(a['gain_ms'] for a in t['ranked'])) <= EPS)
        cond = {'K': k, 'policies': pp, 'trace': tt, 'swap_trace': [s for s in swaps if s['case'] == case and s['K'] == k]}
        if float(pp['Greedy+1-Swap']['J_ms']) - optimum > EPS:
            s = set(json.loads(pp['Greedy+1-Swap']['selected']))
            neighbor_min = {}
            for count in range(1, k + 1):
                vals = [evaluate(inp, es, s - set(remove) | set(add))[2] for remove in itertools.combinations(sorted(s), count) for add in itertools.combinations(sorted(set(es) - s), count)]
                neighbor_min[str(count)] = min(vals)
            cond['fixed_local_set_neighborhood_min_J'] = neighbor_min
            ck(case + '局部停点', neighbor_min['1'] >= float(pp['Greedy+1-Swap']['J_ms']) - EPS)
        detail[case]['conditions'].append(cond)
        structures.append({'dataset': 'Strict20', 'case': case, 'K': k, **mm, 'budget_relation': '<' if k < mm['K_min'] else '=' if k == mm['K_min'] else '>', 'strict_runtime_certification': 'EPOCH_ADMISSIBLE_NOT_ALL_RESUME_VALIDATED', 'classes': ('A;' if mm['separable'] else '') + ('B;' if k >= mm['K_min'] else '') + ('C;' if mm['shared_candidates'] and k < mm['K_min'] else ''), 'greedy_gap_ms': float(pp['Greedy']['J_ms']) - optimum, 'swap_gap_ms': float(pp['Greedy+1-Swap']['J_ms']) - optimum, 'source': path, 'source_row': '完整Epoch；结果CSV见solver_comparison'})

raw = list(csv.DictReader(read('evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv').splitlines()))
g2 = {}
for policy in ['Exact', 'Greedy']:
    for label in 'ABCD':
        rr = [r for r in raw if r['solver'] == policy and '_' + label + '_' in r['pending']]
        ref = by['B5-C4', 2][policy]
        expected = json.loads(ref['per_pending'])['ABCD'.index(label)]
        ck('G2冻结结构对应' + policy + label, len(rr) == 5 and all(int(r['E']) == expected['E'] and int(r['G_model']) == expected['G'] and set(r['selected'].split('|')) == set(json.loads(ref['selected'])) and r['cold_output_match'] == '1' for r in rr))
        g2[policy + '_' + label] = {'n': len(rr), 'E': expected['E'], 'G_model': expected['G'], 'prefill_tokens': sorted(set(int(r['prefill_tokens']) for r in rr)), 'cuda_median_ms': statistics.median(float(r['cuda_prefill_ms']) for r in rr), 'ttft_median_ms': statistics.median(float(r['ttft_ms']) for r in rr)}
read('evaluation/b5c4_formal_gpu_g2/final_report.md')
for path in [E3 + 'protocol.json', E3 + 'greedy_wfc_rule_audit.md', E3 + 'audit.py', E3 + 'diagnostics.py', E3 + 'local_optimum_neighbors.csv', 'evaluation/independent_runtime_allocation_e2/measurement_and_validity.md', 'evaluation/gate1_step3_compatibility_retention_contract_audit/summary.json', 'evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md']:
    read(path)
ck('条件计数', len(structures) == 593 and len(rows) == 140 and len(legacy) == 573)
summary = {'legacy_relations': dict(legacy_equal), 'datasets': {}, 'strict_optimal': {}, 'manifest_mismatches': [r for r in manifest_checks if not r['match']], 'checks': len(checks)}
for dataset in sorted(set(r['dataset'] for r in structures)):
    rr = [r for r in structures if r['dataset'] == dataset]
    summary['datasets'][dataset] = {'conditions': len(rr), 'shared': sum(r['shared_candidates'] > 0 for r in rr), 'separable': sum(r['separable'] for r in rr), 'tight': sum(r['K'] < r['K_min'] for r in rr), 'saturated': sum(r['K'] >= r['K_min'] for r in rr), 'crossing': sum(r['crossing_pairs'] > 0 for r in rr), 'n_range': [min(r['n_candidate'] for r in rr), max(r['n_candidate'] for r in rr)], 'm_range': [min(r['n_pending'] for r in rr), max(r['n_pending'] for r in rr)]}
for policy in by['w1', 1]:
    rr = [r for r in solvers if r['dataset'] == 'Strict20' and r['policy'] == policy]
    summary['strict_optimal'][policy] = {'optimal': sum(r['optimal'] for r in rr), 'same_set_as_exact': sum(r['same_set_as_exact'] for r in rr), 'ties_different_set': sum(r['relation_to_reference'] == 'DIFFERENT_SET_SAME_COST' for r in rr)}
for p, identity in list(sources.items()):
    ck('源文件未修改：' + p, hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == identity['sha256'])
csvout('structure_metrics.csv', structures)
csvout('solver_comparison.csv', solvers)
dump('evidence_details.json', detail)
dump('verification.json', {'summary': summary, 'checks': checks, 'manifests': manifest_checks, 'g2': g2})
dump('source_hashes.json', sources)
print(json.dumps(summary, ensure_ascii=False, indent=2))
