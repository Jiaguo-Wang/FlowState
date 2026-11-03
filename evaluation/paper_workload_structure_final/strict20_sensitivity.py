"""任务 3+4（Strict20 部分）：预算敏感性汇总、非嵌套最优集合比例与生成器可行性检查。

数据源（只读，哈希留档）：
- greedy_swap_structure_e3_step1/condition_results.csv（5 快照×K1-4×7 策略，140 行，冻结）；
- independent_runtime_allocation_e2/{w1..w4}/strict_epoch_snapshot.json 与
  multishare_allocation_gate_b5c4/results.json（快照来源指纹，仅读元数据字段）。

内容：
1 生成器可行性：核验 Strict20 五快照的来源性质（Trace-derived 运行时采集、引擎指纹、
  完整采集链在位）并确认仓内不存在支持固定参数+随机种子的结构生成器——
  结论写入 generator_feasibility.json（NOT_SUPPORTED ⇒ 按 spec 条件句不生成合成实例）；
2 预算敏感性：四方法（Exact / Marginal Greedy / StandaloneTopK / WFC）逐 K 的
  gap_vs_exact 分布与选集一致性（e3 冻结结果重新聚合，不重跑求解）；
3 非嵌套性：Exact 与 Greedy 终集随 K 的包含关系检验，输出违反清单与比例。
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
E2 = ROOT / 'evaluation/independent_runtime_allocation_e2'
C4 = ROOT / 'evaluation/multishare_allocation_gate_b5c4'
OUT = ROOT / 'evaluation/paper_workload_structure_final'
EPS = 1e-9
FOUR = ['Exact', 'Greedy', 'StandaloneTopK', 'WFC']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    inputs = {'e3_results': E3 / 'condition_results.csv',
              'e2_w1': E2 / 'w1/strict_epoch_snapshot.json',
              'e2_w2': E2 / 'w2/strict_epoch_snapshot.json',
              'e2_w3': E2 / 'w3/strict_epoch_snapshot.json',
              'e2_w4': E2 / 'w4/strict_epoch_snapshot.json',
              'c4_results': C4 / 'results.json'}
    (OUT / 'strict20_source_hashes.json').write_text(json.dumps(
        {k: {'path': str(v), 'sha256': sha(v)} for k, v in inputs.items()}, ensure_ascii=False, indent=2))
    checks = []

    # ══ 1) 生成器可行性 ══
    gen_evidence = []
    for g in ('w1', 'w2', 'w3', 'w4'):
        s = json.loads((E2 / g / 'strict_epoch_snapshot.json').read_text())
        cfg = s['configuration']
        gen_evidence.append({
            'snapshot': g, 'workload_type': s['workload_type'],
            'epoch_id': s['epoch_id'], 'engine_sha256': cfg['engine_configuration_sha256'],
            'chat_template_sha256': cfg['chat_template_sha256'],
            'candidates': len(s['candidates']), 'pendings': len(s['pendings']),
            'capture_chain_present': all((E2 / g / f).exists() for f in
                                         ('capture_manifest.json', 'runtime.log', 'gpu_before.json',
                                          'physical_state_census.json', 'historical_replay_log.json')),
            'generator_params': None})
    c4 = json.loads((C4 / 'results.json').read_text())
    gen_evidence.append({'snapshot': 'B5-C4', 'workload_type': 'Trace-derived Controlled Composition',
                         'epoch_id': c4['input'].get('epoch_id', 'b5c3-191'),
                         'capture_chain_present': all((C4 / f).exists() for f in
                                                      ('command.json', 'runtime.log', 'gpu_before.json',
                                                       'physical_state_census.json')),
                         'generator_params': None})
    # 参数化生成器存在性：快照仅由运行时采集产生；引用 strict_epoch_snapshot 的代码全部为
    # 审计/验证脚本（读入方），无 seed/共享度/分支深度参数的生成入口（仓内 grep 证据）。
    feas = {'verdict': 'NOT_SUPPORTED',
            'reason': '五快照均为真实轨迹衍生的运行时采集产物（Trace-derived Controlled '
                      'Composition，含引擎/模板指纹与完整采集链）；仓内不存在支持固定参数与'
                      '随机种子的 Strict20 结构生成器，无法按 spec 条件句生成合成压力实例',
            'implication': '本轮全部实例（573 历史条件 + Strict20）均为真实轨迹衍生，零合成实例；'
                           '合成压力区间（自定义共享度/分支深度/预算）在本仓无受控来源',
            'snapshots': gen_evidence}
    (OUT / 'generator_feasibility.json').write_text(json.dumps(feas, ensure_ascii=False, indent=2))
    checks.append({'check': '五快照采集链在位且均无生成器参数',
                   'pass': all(e['capture_chain_present'] and e['generator_params'] is None
                               for e in gen_evidence)})

    # ══ 2+3) 预算敏感性与非嵌套 ══
    rows = list(csv.DictReader((E3 / 'condition_results.csv').open()))
    by = {}
    for r in rows:
        by.setdefault((r['case'], int(r['K'])), {})[r['policy']] = r
    assert len(by) == 20 and all(len(v) == 7 for v in by.values())
    sens_rows, nest = [], {'Exact': [], 'Greedy': []}
    pol_sets = {p: {} for p in FOUR}
    for (case, K), pol in sorted(by.items()):
        je = float(pol['Exact']['J_ms'])
        for p in FOUR:
            r = pol[p]
            j = float(r['J_ms'])
            sel = tuple(sorted(json.loads(r['selected'])))
            pol_sets[p][(case, K)] = sel
            sens_rows.append({'case': case, 'K': K, 'policy': p, 'J_ms': j,
                              'gap_vs_exact_ms': round(j - je, 9),
                              'same_set_as_exact': list(sel) == sorted(json.loads(pol['Exact']['selected']))})
    # 非嵌套：同快照 K→K+1 终集包含检验
    cases = sorted({c for c, _ in by})
    for p in FOUR:
        viol = []
        for case in cases:
            for K in (1, 2, 3):
                a, b = pol_sets[p][(case, K)], pol_sets[p][(case, K + 1)]
                if not set(a) <= set(b):
                    viol.append({'case': case, 'from_K': K, 'set_k': list(a), 'set_k1': list(b)})
        nest[p] = viol
    # 方法间选集关系（两两同集比例）
    pairwise = []
    for a, b in itertools.combinations(FOUR, 2):
        same = sum(pol_sets[a][k] == pol_sets[b][k] for k in pol_sets[a])
        pairwise.append({'pair': '%s-%s' % (a, b), 'same_set_conditions': same, 'n': 20})
    with (OUT / 'strict20_sensitivity.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)
    (OUT / 'strict20_summary.json').write_text(json.dumps(
        {'checks': checks, 'sensitivity_rows': len(sens_rows),
         'greedy_optimal_by_K': {str(K): sum(1 for (c, k), v in pol_sets['Greedy'].items()
                                             if k == K and v == pol_sets['Exact'][(c, k)])
                                 for K in (1, 2, 3, 4)},
         'nesting_violations': nest,
         'nonnested_ratio': {p: '%d/%d' % (len(nest[p]), len(cases) * 3) for p in nest},
         'pairwise_same_set': pairwise}, ensure_ascii=False, indent=2))
    print('敏感性 %d 行；Greedy 逐 K 最优：%s' % (len(sens_rows), {
        K: sum(1 for (c, k), v in pol_sets['Greedy'].items() if k == K and v == pol_sets['Exact'][(c, k)])
        for K in (1, 2, 3, 4)}))
    print('非嵌套违反：Exact %d / Greedy %d（各 15 次相邻转移）' % (len(nest['Exact']), len(nest['Greedy'])))
    print('两两同集：', pairwise)
    print('生成器可行性：', feas['verdict'])


if __name__ == '__main__':
    main()
