"""T2-G2 配对统计与复算脚本（CPU-only）：只读 60 主臂原始 summary，门禁臂不进统计。

口径（与任务规格逐条对应，三层严格分列不得混用）：
- 模型层：CPU 冻结总恢复成本 J（t2g1_plans.J_ms_current_scope）——只作结构参照，不与实测混算；
- 结构层：实际恢复前沿 E、恢复缺口 G、Prefill Tokens（每策略×Pending 恒定，验证后列表）；
- 实测层：请求 TTFT（六段分解闭合已由 verify 复核）、CUDA Prefill/Decode、COW、控制开销。

统计：
1 各 Pending 五轮配对差（同轮同 Pending 相邻执行；FS−MC 与 FS−KV 两组比较器）；
2 每轮四 Pending 等权聚合 TTFT（策略内先等权平均）→ 轮级配对差；
3 上述差值集合的中位数与极差（五轮样本不作显著性表述）；
4 胜/负/平计数与符号全保留（负向/平局如实报告）；
5 机器双峰模式标记（cuda_decode 阈值 160ms，c_formal 已知混入）与对内模式错配计数；
6 控制开销分解（apply RPC 墙钟与分段，三策略对比；生产核心 vs 研究仪表单列说明）。
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
OUT = T2G1 / 'formal_gpu1'
POLICIES = ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired']
COMPARATORS = [('FS-MC', 'FlowState-Greedy', 'Marconi-Evict'), ('FS-KV', 'FlowState-Greedy', 'KVFlow-inspired')]
PENDINGS = ['C_A', 'C_B', 'C_C', 'C_D']
DECODE_SLOW_MS = 160.0  # 双峰模式分界（c_formal 实测 ~203 vs ~121ms）


def read(p):
    return json.loads(Path(p).read_text())


def med(vals):
    return round(statistics.median(vals), 3)


def rng(vals):
    return round(min(vals), 3), round(max(vals), 3)


def main():
    protocol = read(OUT / 'protocol.json')
    plans = read(T2G1 / 't2g1_plans.json')['plans']
    cell = {}  # (rep, pending, policy) -> 关键实测字段
    for arm in protocol['main_arms']:
        s = read(OUT / arm['name'] / 'run' / 'summary.json')
        d = s['timing_decomposition']
        cell[(arm['rep'], arm['pending'], arm['policy'])] = {
            'ttft': s['ttft_ms'], 'wall': s['request_wall_ms'],
            'prefill_cuda': s['cuda_prefill_ms'], 'decode_cuda': s['cuda_decode_ms'],
            'compute': s['cuda_prefill_ms'] + s['cuda_decode_ms'],
            'cow_cuda': s['cow_copy_cuda_ms'], 'cow_host': s['cow_copy_host_ms'],
            'match': s['match_native_ms'] or 0.0, 'control': s['control_rpc_wall_ms'],
            'stages': {t['stage']: t['ms'] for t in s['control_timing_stages']},
            'mode': 'slow' if s['cuda_decode_ms'] > DECODE_SLOW_MS else 'fast',
            'E': s['E_executable_prefix'], 'G': s['G_replay_gap'], 'prefill_tok': s['prefill_token_count'],
        }
    assert len(cell) == 60, '主臂数据不全：%d' % len(cell)

    # 1) 请求级配对差（同轮同 Pending）
    pair_rows = []
    diffs = defaultdict(list)  # (comparator, metric) -> [差值]
    mismatch = defaultdict(int)
    for rep in range(1, 6):
        for pend in PENDINGS:
            base = {(p): cell[(rep, pend, p)] for p in POLICIES}
            for cname, pa, pb in COMPARATORS:
                a, b = base[pa], base[pb]
                if a['mode'] != b['mode']:
                    mismatch[cname] += 1
                row = {'rep': rep, 'pending': pend, 'comparator': cname,
                       'ttft_diff_ms': round(a['ttft'] - b['ttft'], 3),
                       'compute_diff_ms': round(a['compute'] - b['compute'], 3),
                       'prefill_cuda_diff_ms': round(a['prefill_cuda'] - b['prefill_cuda'], 3),
                       'decode_cuda_diff_ms': round(a['decode_cuda'] - b['decode_cuda'], 3),
                       'cow_diff_ms': round((a['cow_cuda'] + a['cow_host']) - (b['cow_cuda'] + b['cow_host']), 3),
                       'control_diff_ms': round(a['control'] - b['control'], 2),
                       'mode_a': a['mode'], 'mode_b': b['mode']}
                pair_rows.append(row)
                for m in ['ttft_diff_ms', 'compute_diff_ms', 'prefill_cuda_diff_ms', 'control_diff_ms']:
                    diffs[(cname, m)].append(row[m])

    # 2) 轮级四 Pending 等权聚合
    round_rows = []
    round_ttft = defaultdict(dict)  # rep -> {policy: 等权均值}
    for rep in range(1, 6):
        for p in POLICIES:
            round_ttft[rep][p] = statistics.mean(cell[(rep, pend, p)]['ttft'] for pend in PENDINGS)
        for cname, pa, pb in COMPARATORS:
            d = round_ttft[rep][pa] - round_ttft[rep][pb]
            round_rows.append({'rep': rep, 'comparator': cname, 'policy_a': pa, 'policy_b': pb,
                               'ttft_ew_a_ms': round(round_ttft[rep][pa], 3),
                               'ttft_ew_b_ms': round(round_ttft[rep][pb], 3),
                               'diff_ms': round(d, 3)})
            diffs[(cname, 'round_equal_weight')].append(d)

    # 3) 每 Pending 分组统计 + 全局
    def summarize(vals):
        lo, hi = rng(vals)
        neg = sum(1 for v in vals if v < 0)
        pos = sum(1 for v in vals if v > 0)
        zero = sum(1 for v in vals if v == 0)
        return {'median_ms': med(vals), 'min_ms': lo, 'max_ms': hi, 'neg': neg, 'zero': zero, 'pos': pos, 'n': len(vals)}

    per_pending = {}
    for pend in PENDINGS:
        per_pending[pend] = {}
        for cname, _, _ in COMPARATORS:
            for m in ['ttft_diff_ms', 'compute_diff_ms']:
                vals = [r[m] for r in pair_rows if r['pending'] == pend and r['comparator'] == cname]
                per_pending[pend]['%s_%s' % (cname, m)] = summarize(vals)

    # 4) 策略×Pending 矩阵（实测中位 + 结构层 + 模型层并列，分列不混算）
    matrix = []
    for pend in PENDINGS:
        for p in POLICIES:
            rows = [cell[(rep, pend, p)] for rep in range(1, 6)]
            matrix.append({'pending': pend, 'policy': p,
                           'E': rows[0]['E'], 'G': rows[0]['G'], 'prefill_tokens': rows[0]['prefill_tok'],
                           'J_modeled_ms': plans[p].get('J_ms_current_scope'),
                           'ttft_median_ms': med([r['ttft'] for r in rows]),
                           'ttft_min_ms': rng([r['ttft'] for r in rows])[0],
                           'ttft_max_ms': rng([r['ttft'] for r in rows])[1],
                           'compute_median_ms': med([r['compute'] for r in rows]),
                           'prefill_cuda_median_ms': med([r['prefill_cuda'] for r in rows]),
                           'decode_cuda_median_ms': med([r['decode_cuda'] for r in rows]),
                           'control_rpc_median_ms': med([r['control'] for r in rows]),
                           'mode_counts': {m: sum(1 for r in rows if r['mode'] == m) for m in ('fast', 'slow')}})

    # 5) 控制开销分解（三策略对比；分段取五轮×四Pending 均值）
    control = {}
    for p in POLICIES:
        rows = [cell[(rep, pend, p)] for rep in range(1, 6) for pend in PENDINGS]
        stages = defaultdict(list)
        for r in rows:
            for k, v in r['stages'].items():
                stages[k].append(v)
        control[p] = {'apply_rpc_wall_median_ms': med([r['control'] for r in rows]),
                      'stages_mean_ms': {k: round(statistics.mean(v), 2) for k, v in sorted(stages.items())}}

    summary = {
        'request_level': {'%s_%s' % (c, m): summarize(v) for (c, m), v in diffs.items()
                          if m != 'round_equal_weight'},
        'round_equal_weight': {'%s' % c: summarize(v) for (c, m), v in diffs.items()
                               if m == 'round_equal_weight'},
        'per_pending': per_pending,
        'mode_mismatch_pairs': dict(mismatch),
        'decode_slow_arms': sum(1 for v in cell.values() if v['mode'] == 'slow'),
        'control_decomposition': control,
        'structure_layer': [{'pending': r['pending'], 'policy': r['policy'], 'E': r['E'], 'G': r['G'],
                             'prefill_tokens': r['prefill_tokens'], 'J_modeled_ms': r['J_modeled_ms']}
                            for r in matrix],
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (OUT / 'pair_analysis.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    with (OUT / 'pending_policy_matrix.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
        w.writeheader()
        w.writerows(matrix)
    with (OUT / 'round_equal_weight.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(round_rows[0].keys()))
        w.writeheader()
        w.writerows(round_rows)
    print('配对统计完成：20 对×2 比较器；轮级等权 5×2；矩阵 12 格')
    for c, _, _ in COMPARATORS:
        r = summary['request_level']['%s_ttft_diff_ms' % c]
        ew = summary['round_equal_weight'][c]
        print('%s 请求级TTFT中位 %.1fms [%s, %s] 负%d/平%d/正%d | 轮级等权中位 %.1fms 负%d/正%d' % (
            c, r['median_ms'], r['min_ms'], r['max_ms'], r['neg'], r['zero'], r['pos'],
            ew['median_ms'], ew['neg'], ew['pos']))


if __name__ == '__main__':
    main()
