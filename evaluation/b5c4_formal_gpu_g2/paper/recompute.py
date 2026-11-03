"""P2 复算脚本（宿主机执行，只读原始证据）：从 raw_results.csv 与逐臂 summary.json 独立重算
G2 正式统计，并与 paired_analysis.csv / checks.json / final_report.md 的既有数值交叉核对。

统计口径（严格区分，不混用）：
- 配对差：同轮同 Pending 的 (Greedy−Exact) 单对差 Δ_r,p，共 20 对；
- 每 Pending「配对差中位数」＝median(Δ_r,p, r=1..5)（协议主口径）；
- 每 Pending「两组中位数之差」＝median(Greedy 五轮)−median(Exact 五轮)（辅助口径，
  即 paired_analysis.csv 的 delta_*_median 列——本脚本核对其一致性但与主口径分列）；
- 轮级等权：每轮四 Pending 等权均值（＝该轮 4 个 Δ_r,p 的均值）→ 五轮中位（集合级口径）；
- 明确计算并标注「合并 20 样本池化中位数之差」为非协议口径，仅作警示对照，不入结果表。
逐项输出到 recompute_report.json，任一交叉核对不符即该检查 FAIL。
"""
import csv, json, statistics
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
G2 = ROOT / 'evaluation/b5c4_formal_gpu_g2'
FORMAL = G2 / 'formal'
OUT = G2 / 'paper'
LETTERS = 'ABCD'
SOLVERS = ('Exact', 'Greedy')


def read_json(p):
    return json.loads(Path(p).read_text())


def f(x):
    return float(x)


def main():
    raw = list(csv.DictReader((FORMAL / 'raw_results.csv').open()))
    checks, fails = [], []

    def check(name, ok, detail=None):
        row = {'check': name, 'pass': bool(ok)}
        if detail is not None:
            row['detail'] = detail
        checks.append(row)
        if not ok:
            fails.append(name)
        return ok

    # ── 1. 完整性与有效性（40 臂、20 对）──
    check('rows_40', len(raw) == 40, len(raw))
    keys = {(int(r['round']), r['solver'], r['pending']) for r in raw}
    check('grid_complete', keys == {(r, s, 'B5C2_%s_TURN_003' % l) for r in range(1, 6) for s in SOLVERS for l in LETTERS})
    by = {(int(r['round']), r['solver'], r['pending']): r for r in raw}
    validity_ok, val_detail = True, []
    for (r, s, p), row in sorted(by.items()):
        s_json = read_json(FORMAL / row['run'] / 'summary.json')
        ok = (s_json['status'] == 'PASS' and s_json['solver'] == s and s_json['audit_entries_total'] == 2
              and s_json['triton_new_kernels'][-1]['new_entries'] == 0 and s_json['cold_output_match'] is True
              and s_json['E_executable_prefix'] == int(row['E']) and abs(s_json['ttft_ms'] - f(row['ttft_ms'])) < 1e-9
              and s_json['prefill_token_count'] == int(row['prefill_tokens']))
        validity_ok &= ok
        if not ok:
            val_detail.append(row['run'])
    check('arm_validity_from_summaries', validity_ok, val_detail or '40/40：status/solver/audit==2/零编译/Cold一致/字段一致')

    # ── 2. 配对差（20 对）与两种中位口径 ──
    pairs = []
    for r in range(1, 6):
        for l in LETTERS:
            p = 'B5C2_%s_TURN_003' % l
            e, g = by[(r, 'Exact', p)], by[(r, 'Greedy', p)]
            pairs.append({'round': r, 'letter': l,
                          'd_ttft': f(g['ttft_ms']) - f(e['ttft_ms']),
                          'd_prefill': f(g['cuda_prefill_ms']) - f(e['cuda_prefill_ms'])})
    per_pending = {}
    for l in LETTERS:
        ps = [x for x in pairs if x['letter'] == l]
        p = 'B5C2_%s_TURN_003' % l
        med = lambda sol, m: statistics.median([f(by[(r, sol, p)][m]) for r in range(1, 6)])
        per_pending[l] = {
            'd_ttft_rounds': [round(x['d_ttft'], 2) for x in ps],
            'd_prefill_rounds': [round(x['d_prefill'], 2) for x in ps],
            'median_of_paired_d_ttft': round(statistics.median([x['d_ttft'] for x in ps]), 2),
            'median_of_paired_d_prefill': round(statistics.median([x['d_prefill'] for x in ps]), 2),
            'exact_ttft_median': round(med('Exact', 'ttft_ms'), 2), 'greedy_ttft_median': round(med('Greedy', 'ttft_ms'), 2),
            'exact_prefill_median': round(med('Exact', 'cuda_prefill_ms'), 2), 'greedy_prefill_median': round(med('Greedy', 'cuda_prefill_ms'), 2),
            'diff_of_medians_ttft': round(med('Greedy', 'ttft_ms') - med('Exact', 'ttft_ms'), 2),
            'diff_of_medians_prefill': round(med('Greedy', 'cuda_prefill_ms') - med('Exact', 'cuda_prefill_ms'), 2)}
    # 与 paired_analysis.csv 交叉核对（其 delta_*_median 列＝两组中位数之差）
    text = (FORMAL / 'paired_analysis.csv').read_text().splitlines()
    req_lines = [ln for ln in text[:text.index(next(x for x in text if x.startswith('[轮级')))] if ln.strip()]
    pp_csv = {r['pending']: r for r in csv.DictReader(req_lines)}
    cross = all(abs(per_pending[l]['diff_of_medians_ttft'] - float(pp_csv['B5C2_%s_TURN_003' % l]['delta_ttft_median'])) < 0.02
                and abs(per_pending[l]['diff_of_medians_prefill'] - float(pp_csv['B5C2_%s_TURN_003' % l]['delta_prefill_median'])) < 0.02
                and abs(per_pending[l]['exact_ttft_median'] - float(pp_csv['B5C2_%s_TURN_003' % l]['Exact_ttft_median'])) < 0.02
                for l in LETTERS)
    check('crosscheck_paired_analysis_csv', cross, 'delta_*_median 列确为两组中位数之差（辅助口径），数值一致')

    # ── 3. 轮级等权 → 五轮中位（协议集合级口径）──
    round_level = []
    for r in range(1, 6):
        ps = [x for x in pairs if x['round'] == r]
        round_level.append({'round': r,
                            'd_ttft_equal_mean': statistics.mean([x['d_ttft'] for x in ps]),
                            'd_prefill_equal_mean': statistics.mean([x['d_prefill'] for x in ps]),
                            'exact_ttft_equal_mean': statistics.mean([f(by[(r, 'Exact', 'B5C2_%s_TURN_003' % l)]['ttft_ms']) for l in LETTERS]),
                            'greedy_ttft_equal_mean': statistics.mean([f(by[(r, 'Greedy', 'B5C2_%s_TURN_003' % l)]['ttft_ms']) for l in LETTERS]),
                            'exact_prefill_equal_mean': statistics.mean([f(by[(r, 'Exact', 'B5C2_%s_TURN_003' % l)]['cuda_prefill_ms']) for l in LETTERS]),
                            'greedy_prefill_equal_mean': statistics.mean([f(by[(r, 'Greedy', 'B5C2_%s_TURN_003' % l)]['cuda_prefill_ms']) for l in LETTERS])})
    summary = {'d_ttft_median_of_round_means': round(statistics.median([x['d_ttft_equal_mean'] for x in round_level]), 2),
               'd_prefill_median_of_round_means': round(statistics.median([x['d_prefill_equal_mean'] for x in round_level]), 2),
               'exact_ttft_median_of_round_means': round(statistics.median([x['exact_ttft_equal_mean'] for x in round_level]), 2),
               'greedy_ttft_median_of_round_means': round(statistics.median([x['greedy_ttft_equal_mean'] for x in round_level]), 2)}
    ck = read_json(FORMAL / 'checks.json')['layer3_measured']['round_level_delta_median']
    check('crosscheck_checks_json_round_level',
          abs(summary['d_ttft_median_of_round_means'] - ck['delta_ttft_median']) < 0.02
          and abs(summary['d_prefill_median_of_round_means'] - ck['delta_prefill_median']) < 0.02,
          {'recompute': summary, 'checks.json': {k: ck[k] for k in ('delta_ttft_median', 'delta_prefill_median')}})
    check('all_rounds_d_ttft_positive', all(x['d_ttft_equal_mean'] > 0 for x in round_level),
          [round(x['d_ttft_equal_mean'], 2) for x in round_level])
    # 非协议口径警示对照（不入结果表）：20 样本池化中位数之差
    pooled = {'pooled_median_greedy_ttft_minus_exact': round(statistics.median([f(r['ttft_ms']) for r in raw if r['solver'] == 'Greedy'])
                                                             - statistics.median([f(r['ttft_ms']) for r in raw if r['solver'] == 'Exact']), 2),
              'note': '合并 20 样本池化中位数之差＝非实验协议口径（Pending 结构权重失真），仅警示对照，不得引用'}

    # ── 4. Exact 更慢的单臂轮次（负向观测全保留）──
    slower = sorted([(x['round'], x['letter'], round(f(by[(x['round'], 'Exact', 'B5C2_%s_TURN_003' % x['letter'])]['ttft_ms']), 1),
                      round(-x['d_ttft'], 2)) for x in pairs if x['d_ttft'] < 0])
    check('exact_slower_count_5', len(slower) == 5, slower)
    fr = (G2 / 'final_report.md').read_text()
    check('final_report_lists_slower', all(('r%d_exact_%s' % (r, l)).lower() in fr for r, l, _, _ in slower))

    # ── 5. 模型层/结构层/实测层对应（J、前沿、Native Gap、Prefill Token）──
    ref = read_json(G2 / 'preflight/g2_reference.json')
    J = {s: {f(r['J_ms']) for r in raw if r['solver'] == s} for s in SOLVERS}
    check('J_constant_and_matches_e3', J['Exact'] == {ref['Exact']['J_ms']} and J['Greedy'] == {ref['Marginal Greedy']['J_ms']},
          {'Exact': list(J['Exact']), 'Greedy': list(J['Greedy'])})
    sel = {s: {tuple(sorted(r['selected'].split('|'))) for r in raw if r['solver'] == s} for s in SOLVERS}
    check('selection_matches_e3', sel['Exact'] == {tuple(sorted(ref['Exact']['selected']))}
          and sel['Greedy'] == {tuple(sorted(ref['Marginal Greedy']['selected']))})
    struct_ok, struct = True, {}
    for l in LETTERS:
        p = 'B5C2_%s_TURN_003' % l
        row = {}
        for s in SOLVERS:
            r0 = by[(1, s, p)]
            es = {int(by[(r, s, p)]['E']) for r in range(1, 6)}
            gs = {int(by[(r, s, p)]['G_model']) for r in range(1, 6)}
            pts = {int(by[(r, s, p)]['prefill_tokens']) for r in range(1, 6)}
            T = int(r0['E']) + int(r0['G_model'])
            e3 = ref['Exact' if s == 'Exact' else 'Marginal Greedy']['per_pending'][p]
            len_pt = int(r0['H']) + int(r0['tail_beyond_frontier'])
            struct_ok &= (es == {int(r0['E'])} and gs == {int(r0['G_model'])} and pts == {int(r0['prefill_tokens'])}
                          and T == e3['T'] and int(r0['E']) == e3['E'] and int(r0['G_model']) == e3['G']
                          and int(r0['prefill_tokens']) == len_pt - int(r0['E']) and int(r0['batch_prefix']) == int(r0['E']))
            row[s] = {'E': int(r0['E']), 'T': T, 'G_model': int(r0['G_model']),
                      'prefill_tokens': int(r0['prefill_tokens']), 'H': int(r0['H']), 'len_pt': len_pt}
        struct[p] = row
    check('structure_layer_consistent', struct_ok, struct)

    verdict = 'PASS' if not fails else 'PARTIAL'
    OUT.mkdir(exist_ok=True)
    out = OUT / 'recompute_report.json'
    assert not out.exists(), '拒绝覆盖 recompute_report.json'
    out.write_text(json.dumps({'verdict': verdict, 'fails': fails, 'checks': checks,
                               'pairs': [{'round': x['round'], 'pending': x['letter'], 'd_ttft_ms': round(x['d_ttft'], 2),
                                          'd_prefill_ms': round(x['d_prefill'], 2)} for x in pairs],
                               'per_pending': per_pending, 'round_level': [{'round': x['round'],
                                   'd_ttft_equal_mean': round(x['d_ttft_equal_mean'], 2), 'd_prefill_equal_mean': round(x['d_prefill_equal_mean'], 2),
                                   'exact_ttft_equal_mean': round(x['exact_ttft_equal_mean'], 2), 'greedy_ttft_equal_mean': round(x['greedy_ttft_equal_mean'], 2),
                                   'exact_prefill_equal_mean': round(x['exact_prefill_equal_mean'], 2), 'greedy_prefill_equal_mean': round(x['greedy_prefill_equal_mean'], 2)} for x in round_level],
                               'summary_protocol': summary, 'pooled_nonprotocol_warning': pooled,
                               'exact_slower_rounds': [{'round': r, 'pending': l, 'exact_ttft_ms': t, 'greedy_faster_by_ms': d} for r, l, t, d in slower],
                               'structure': struct}, ensure_ascii=False, indent=2))
    print('复算完成：', verdict, '；失败项：', fails or '无')
    print('每Pending配对差中位 ΔTTFT：', {l: per_pending[l]['median_of_paired_d_ttft'] for l in LETTERS})
    print('轮级等权五轮中位：', summary)
    print('Exact更慢轮次：', slower)


if __name__ == '__main__':
    main()
