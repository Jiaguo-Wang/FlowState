"""G2-B 正式分析（宿主机执行，只读证据）：40 臂原始结果、配对差与三层结论核对。

统计口径（按规格，不做显著性宣称、不筛选轮次）：
- 请求级：每 Pending 五轮 TTFT / CUDA Prefill 中位数（Exact vs Greedy，附全部五轮值）；
- 集合级：每轮四 Pending 等权均值 → 五轮中位；
- 配对：同轮同 Pending 配对差 ΔTTFT_r / ΔCUDA Prefill_r（Greedy−Exact，正值=Exact 更快）；
  及轮级等权配对差的中位数；
- 三层结论区分并逐层核对：(1) 模型化 J 更低（在线 J vs e3 参照）；(2) Native Gap 减少
  （G_model=T−E 与 Prefill Token 数）；(3) CUDA Prefill / TTFT 实际改善（测量值）。
  (1)(2) 是模型/结构层，(3) 是实测层，不得由前者直接推导后者——分析只报告一致性，不外推。
产物：raw_results.csv / paired_analysis.csv / checks.json / arm_status.json（均在 formal/）。
"""
import csv, json, statistics
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
G2 = ROOT / 'evaluation/b5c4_formal_gpu_g2'
FORMAL = G2 / 'formal'
REF = json.loads((G2 / 'preflight/g2_reference.json').read_text())
SCHEDULE = json.loads((FORMAL / 'schedule.json').read_text())['schedule']
LETTERS = 'ABCD'


def load_arm(base):
    """取该基础 RUN 的最新 PASS 臂（含重试）；返回 (run, summary) 或 None。"""
    for candidate in [base] + [base + '_r%d' % i for i in range(1, 20)][::-1]:
        d = FORMAL / candidate
        if (d / 'summary.json').exists():
            s = json.loads((d / 'summary.json').read_text())
            if s.get('status') == 'PASS':
                return candidate, s
    return None


def attempts_of(base):
    """该基础 RUN 的全部尝试（成功与失败）留档。"""
    rows = []
    for candidate in [base] + [base + '_r%d' % i for i in range(1, 20)]:
        d = FORMAL / candidate
        if (d / 'summary.json').exists():
            rows.append({'run': candidate, 'outcome': 'PASS'})
        elif (d / 'failure.json').exists():
            f = json.loads((d / 'failure.json').read_text())
            rows.append({'run': candidate, 'outcome': 'FAIL', 'error': f.get('error', '')[:160]})
    return rows


def main():
    raw_rows, arm_status, failures = [], {}, []
    arms = {}
    for a in SCHEDULE:
        base = a['run']
        got = load_arm(base)
        arm_status[base] = {'round': a['round'], 'solver': a['solver'], 'pending': a['pending'],
                            'attempts': attempts_of(base),
                            'final': got[0] if got else None, 'status': 'PASS' if got else 'FAIL'}
        if not got:
            failures.append(base)
            continue
        run, s = got
        arms[(a['round'], a['solver'], a['letter'])] = s
        d = s['timing_decomposition']
        raw_rows.append({'run': run, 'round': a['round'], 'solver': a['solver'], 'pending': a['pending'],
                         'selected': '|'.join(s['selected_ids']), 'J_ms': s['plan_recovery_cost_ms'],
                         'E': s['E_executable_prefix'], 'H': s['H_physical_fa_hit'],
                         'G_model': s['G_model_replay_gap'], 'tail_beyond_frontier': s['tail_beyond_frontier'],
                         'prefill_tokens': s['prefill_token_count'], 'batch_prefix': s['batch_prefix_length'],
                         'cuda_prefill_ms': s['cuda_prefill_ms'], 'cuda_decode_ms': s['cuda_decode_ms'],
                         'ttft_ms': s['ttft_ms'], 'request_wall_ms': s['request_wall_ms'],
                         'queue_ms': d['queue_ms'], 'execwin_ms': d['execwin_ms'], 'compute_cuda_ms': d['compute_cuda_ms'],
                         'postgap_ms': d['postgap_ms'], 'delivery_ms': d['delivery_ms'],
                         'control_rpc_wall_ms': s['control_rpc_wall_ms'], 'cow_copy_cuda_ms': s['cow_copy_cuda_ms'],
                         'online_solve_ms': next(t['ms'] for t in s['control_timing_stages'] if t['stage'] == 'online_solve'),
                         'cold_output_match': int(s['cold_output_match']), 'audit_entries': s['audit_entries_total'],
                         'request_window_new_kernels': s['triton_new_kernels'][-1]['new_entries']})
    with (FORMAL / 'raw_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)

    # ── 请求级：每 Pending 五轮中位 ──
    per_pending = {}
    for letter in LETTERS:
        row = {'pending': 'B5C2_%s_TURN_003' % letter}
        for solver in ('Exact', 'Greedy'):
            vals_t = [arms[(r, solver, letter)]['ttft_ms'] for r in range(1, 6) if (r, solver, letter) in arms]
            vals_p = [arms[(r, solver, letter)]['cuda_prefill_ms'] for r in range(1, 6) if (r, solver, letter) in arms]
            row[solver + '_ttft_rounds'] = '|'.join('%.2f' % v for v in vals_t)
            row[solver + '_ttft_median'] = round(statistics.median(vals_t), 2) if vals_t else None
            row[solver + '_prefill_rounds'] = '|'.join('%.2f' % v for v in vals_p)
            row[solver + '_prefill_median'] = round(statistics.median(vals_p), 2) if vals_p else None
        if row.get('Exact_ttft_median') is not None and row.get('Greedy_ttft_median') is not None:
            row['delta_ttft_median'] = round(row['Greedy_ttft_median'] - row['Exact_ttft_median'], 2)
            row['delta_prefill_median'] = round(row['Greedy_prefill_median'] - row['Exact_prefill_median'], 2)
        per_pending[letter] = row

    # ── 轮级：四 Pending 等权均值 → 五轮中位；同轮配对差 ──
    round_rows, paired_round = [], []
    for r in range(1, 6):
        row = {'round': r}
        means = {}
        for solver in ('Exact', 'Greedy'):
            ts = [arms[(r, solver, l)]['ttft_ms'] for l in LETTERS if (r, solver, l) in arms]
            ps = [arms[(r, solver, l)]['cuda_prefill_ms'] for l in LETTERS if (r, solver, l) in arms]
            if len(ts) == 4:
                means[solver] = {'ttft': statistics.mean(ts), 'prefill': statistics.mean(ps)}
                row[solver + '_ttft_equal_mean'] = round(means[solver]['ttft'], 2)
                row[solver + '_prefill_equal_mean'] = round(means[solver]['prefill'], 2)
        if len(means) == 2:
            row['delta_ttft_r'] = round(means['Greedy']['ttft'] - means['Exact']['ttft'], 2)
            row['delta_prefill_r'] = round(means['Greedy']['prefill'] - means['Exact']['prefill'], 2)
            paired_round.append(row)
        round_rows.append(row)
    summary = {'round_level_median': {
        'Exact_ttft': round(statistics.median([x['Exact_ttft_equal_mean'] for x in paired_round]), 2) if paired_round else None,
        'Greedy_ttft': round(statistics.median([x['Greedy_ttft_equal_mean'] for x in paired_round]), 2) if paired_round else None,
        'Exact_prefill': round(statistics.median([x['Exact_prefill_equal_mean'] for x in paired_round]), 2) if paired_round else None,
        'Greedy_prefill': round(statistics.median([x['Greedy_prefill_equal_mean'] for x in paired_round]), 2) if paired_round else None,
        'delta_ttft_median': round(statistics.median([x['delta_ttft_r'] for x in paired_round]), 2) if paired_round else None,
        'delta_prefill_median': round(statistics.median([x['delta_prefill_r'] for x in paired_round]), 2) if paired_round else None},
        'paired_rounds_count': len(paired_round)}

    # 请求级与轮级分两段写同一个 csv（表头不同）：请求级表 + 空行 + 轮级表
    with (FORMAL / 'paired_analysis.csv').open('w') as f:
        pp = list(per_pending.values())
        w = csv.DictWriter(f, fieldnames=list(pp[0]))
        w.writeheader()
        w.writerows(pp)
        f.write('\n[轮级：四Pending等权均值与同轮配对差]\n')
        w = csv.DictWriter(f, fieldnames=list(round_rows[0]))
        w.writeheader()
        w.writerows(round_rows)
        f.write('\n[五轮中位汇总]\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    # ── 三层结论核对（只核对一致性，不做外推）──
    j_exact = [arms[(r, 'Exact', l)]['plan_recovery_cost_ms'] for r in range(1, 6) for l in LETTERS if (r, 'Exact', l) in arms]
    j_greedy = [arms[(r, 'Greedy', l)]['plan_recovery_cost_ms'] for r in range(1, 6) for l in LETTERS if (r, 'Greedy', l) in arms]
    layer1 = {'claim': '模型化 J 更低（Exact）', 'exact_J_all': j_exact[:1] and round(j_exact[0], 6), 'greedy_J_all': j_greedy[:1] and round(j_greedy[0], 6),
              'consistent': all(abs(x - REF['Exact']['J_ms']) <= 1e-9 for x in j_exact) and all(abs(x - REF['Marginal Greedy']['J_ms']) <= 1e-9 for x in j_greedy)}
    g_exact_cd = [arms[(r, 'Exact', l)]['G_model_replay_gap'] for r in range(1, 6) for l in 'CD' if (r, 'Exact', l) in arms]
    g_greedy_cd = [arms[(r, 'Greedy', l)]['G_model_replay_gap'] for r in range(1, 6) for l in 'CD' if (r, 'Greedy', l) in arms]
    layer2 = {'claim': 'Native Gap 减少（Exact，C/D 上 E 3520 vs 2432）',
              'exact_G_CD': sorted(set(g_exact_cd)), 'greedy_G_CD': sorted(set(g_greedy_cd)),
              'prefill_tokens_exact_CD': sorted({arms[(r, 'Exact', l)]['prefill_token_count'] for r in range(1, 6) for l in 'CD' if (r, 'Exact', l) in arms}),
              'prefill_tokens_greedy_CD': sorted({arms[(r, 'Greedy', l)]['prefill_token_count'] for r in range(1, 6) for l in 'CD' if (r, 'Greedy', l) in arms})}
    layer3 = {'claim': 'CUDA Prefill / TTFT 实际改善（Exact，实测层）', 'per_pending_delta_median': {l: per_pending[l].get('delta_ttft_median') for l in LETTERS},
              'round_level_delta_median': summary['round_level_median']}
    validity = {'arms_pass': len(arms), 'arms_fail': failures, 'cold_match_all': all(a['cold_output_match'] == 1 for a in arms.values()),
                'audit_all_two': all(a['audit_entries_total'] == 2 for a in arms.values()),
                'request_window_zero_compile_all': all(a['triton_new_kernels'][-1]['new_entries'] == 0 for a in arms.values()),
                'note': '保留全部有效结果（含 Exact 更慢的请求/轮次，如存在）；五轮样本不宣称统计显著性'}
    checks = {'validity': validity, 'layer1_model_J': layer1, 'layer2_native_gap': layer2, 'layer3_measured': layer3,
              'interpretation_rule': '三层依次为模型层/结构层/实测层；(1)(2) 不得直接推导 (3)，仅报告实测方向与 (1)(2) 的一致性'}
    (FORMAL / 'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    (FORMAL / 'arm_status.json').write_text(json.dumps(arm_status, ensure_ascii=False, indent=2))
    print('分析完成：PASS 臂 %d/40，失败 %s' % (len(arms), failures or '无'))
    print('轮级五轮中位：', json.dumps(summary['round_level_median'], ensure_ascii=False))
    for l in LETTERS:
        print(' ', per_pending[l])


if __name__ == '__main__':
    main()
