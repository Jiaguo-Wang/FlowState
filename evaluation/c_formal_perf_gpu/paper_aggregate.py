"""四 Pending 整体性能归因审计的离线重聚合（不启动 GPU、不改任何原始数据）。

口径：每轮 r、每策略 π 的四 Pending 等权请求级指标 x̄_{r,π} = (1/4)Σ_p x_{r,π,p}；
     TTFT 与 CUDA Prefill 同法。配对差为同轮等权均值之差（等价于各 Pending 配对差的等权平均）。
慢模式规则（沿用 final_report §5.1 的量化口径）：cuda_decode_ms > 150ms 记为慢模式臂；
     主结果使用全部 60 臂不剔除；快模式仅作次要敏感性分析并注明每格样本数。
输出：paper_pending_aggregate.csv + 控制台全部数字（供审计报告引用）。
"""
import csv
import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
POL = {'FS': 'FlowState-Greedy', 'SA': 'StandaloneTopK', 'WFC': 'WFC'}
PENDINGS = ['C_A', 'C_B', 'C_C', 'C_D']


def qmed(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def stats(vals):
    return {'mean': round(st.mean(vals), 1), 'median': round(qmed(vals), 1),
            'min': round(min(vals), 1), 'max': round(max(vals)), 'n': len(vals)}


def main():
    rows = [r for r in csv.DictReader(open(OUT / 'lifecycle_results.csv'))
            if r.get('status') == 'PASS' and r['name'].startswith('m')]
    assert len(rows) == 60, '主矩阵应为 60 臂'
    # 索引：(轮, 策略, Pending) → 行
    cell = {}
    for r in rows:
        key = (int(r['rep']), r['pol'], r['pending'])
        assert key not in cell, '重复生命周期：' + str(key)
        cell[key] = {'ttft': float(r['ttft_ms']), 'prefill': float(r['cuda_prefill_ms']),
                     'decode': float(r['cuda_decode_ms']), 'queue': float(r['queue_ms']),
                     'pregap': float(r['pregap_ms']), 'ctrl': float(r['control_rpc_wall_ms']),
                     'auth': float(r['ctrl_auth_plan_ms'] or 0), 'recon': float(r['ctrl_reconcile_ms'] or 0)}
    slow = {k: v['decode'] > 150 for k, v in cell.items()}

    # ── §三：每轮每策略四 Pending 等权指标 ──
    eq = {}  # (rep, pol) -> {'ttft','prefill','slow_n'}
    for rep in range(1, 6):
        for pol in POL:
            tt = [cell[(rep, pol, p)]['ttft'] for p in PENDINGS]
            pf = [cell[(rep, pol, p)]['prefill'] for p in PENDINGS]
            sn = sum(slow[(rep, pol, p)] for p in PENDINGS)
            eq[(rep, pol)] = {'ttft': st.mean(tt), 'prefill': st.mean(pf), 'slow_n': sn}

    out_rows = [['section', 'rep', 'policy_or_pair', 'pending', 'metric', 'value', 'n', 'note']]
    print('== §三 每轮每策略 四Pending等权 TTFT / CUDA Prefill (ms) 与慢模式臂数 ==')
    print('%-4s %-4s %10s %10s %s' % ('rep', 'pol', 'eqTTFT', 'eqPrefill', '慢臂/4'))
    for rep in range(1, 6):
        for pol in POL:
            e = eq[(rep, pol)]
            print('%-4d %-4s %10.1f %10.1f %d' % (rep, pol, e['ttft'], e['prefill'], e['slow_n']))
            out_rows.append(['per_rep', rep, pol, '', 'eq_ttft_ms', round(e['ttft'], 1), 4, ''])
            out_rows.append(['per_rep', rep, pol, '', 'eq_prefill_ms', round(e['prefill'], 1), 4, ''])
            out_rows.append(['per_rep', rep, pol, '', 'slow_arms_of_4', e['slow_n'], 4, '规则:decode>150ms'])

    print('\n== §三 五轮统计（等权指标跨轮）==')
    for metric in ('ttft', 'prefill'):
        for pol in POL:
            s = stats([eq[(rep, pol)][metric] for rep in range(1, 6)])
            print('%s %-4s %s' % (metric, pol, s))
            out_rows.append(['five_rep_stats', '', pol, '', 'eq_' + metric + '_ms', json.dumps(s), 5, ''])

    # ── §三：每轮配对差 + 五轮统计 ──
    print('\n== §三 每轮配对差（等权 TTFT / Prefill）与五轮统计 ==')
    for pair in ('FS-SA', 'FS-WFC'):
        a, b = pair.split('-')
        for metric in ('ttft', 'prefill'):
            ds = [eq[(rep, a)][metric] - eq[(rep, b)][metric] for rep in range(1, 6)]
            s = stats(ds)
            print('%s Δ%s 每轮: %s → %s' % (pair, metric, [round(d, 1) for d in ds], s))
            for rep, d in enumerate(ds, 1):
                out_rows.append(['paired_delta', rep, pair, '', 'eq_' + metric + '_ms', round(d, 1), 4, ''])
            out_rows.append(['paired_delta_stats', '', pair, '', 'eq_' + metric + '_ms', json.dumps(s), 5, ''])

    # ── §三.5 / §五：单 Pending 正负贡献（中位配对差；对等权均值贡献 = Δp/4）──
    print('\n== §五 单 Pending 配对差中位（TTFT/Prefill, ms）与对等权均值贡献 ==')
    for pair in ('FS-SA', 'FS-WFC'):
        a, b = pair.split('-')
        for p in PENDINGS:
            dt = qmed([cell[(rep, a, p)]['ttft'] - cell[(rep, b, p)]['ttft'] for rep in range(1, 6)])
            dp = qmed([cell[(rep, a, p)]['prefill'] - cell[(rep, b, p)]['prefill'] for rep in range(1, 6)])
            print('%s %-4s Δttft 中位 %+7.1f (对均值贡献 %+6.1f) | Δprefill 中位 %+7.1f' % (pair, p, dt, dt / 4, dp))
            out_rows.append(['pending_contribution', '', pair, p, 'd_ttft_ms_median', round(dt, 1), 5,
                             '对等权均值贡献=%.1f' % (dt / 4)])
            out_rows.append(['pending_contribution', '', pair, p, 'd_prefill_ms_median', round(dp, 1), 5, ''])

    # ── §四：慢模式分布与影响 ──
    print('\n== §四 每轮每策略慢模式臂数（规则 decode>150ms；总 17/60）==')
    tab = {}
    for rep in range(1, 6):
        tab[rep] = {pol: eq[(rep, pol)]['slow_n'] for pol in POL}
        print('rep%d: %s (合计 %d)' % (rep, tab[rep], sum(tab[rep].values())))
    total_by_pol = {pol: sum(tab[rep][pol] for rep in range(1, 6)) for pol in POL}
    print('按策略合计:', total_by_pol)
    out_rows.append(['slow_mode', '', 'FS|SA|WFC', '', 'slow_arms_total',
                     json.dumps(total_by_pol), 60, '分布见 per_rep slow_arms_of_4'])

    # 同轮配对的模式匹配情况（双慢=同中招；错配=仅一方慢，模式直接驱动差值符号）
    print('\n== §四 FS−SA / FS−WFC 20 个同轮配对的模式构成 ==')
    for pair in ('FS-SA', 'FS-WFC'):
        a, b = pair.split('-')
        both_fast = both_slow = mm = 0
        for rep in range(1, 6):
            for p in PENDINGS:
                sa, sb = slow[(rep, a, p)], slow[(rep, b, p)]
                if sa and sb:
                    both_slow += 1
                elif sa or sb:
                    mm += 1
                else:
                    both_fast += 1
        print('%s: 双快 %d / 双慢 %d / 错配 %d（错配对中差值含 ±15–20ms 模式分量）' % (pair, both_fast, both_slow, mm))
        out_rows.append(['slow_mode_pairmix', '', pair, '', 'fast_fast|slow_slow|mismatch',
                         '%d|%d|%d' % (both_fast, both_slow, mm), 20, ''])

    # 次要敏感性分析：快模式格中位再等权（规则 cuda_decode≤150；每格 n=2–4 如实标注）
    print('\n== §四 敏感性：快模式格中位 → 等权（次要分析，非主结果）==')
    for pol in POL:
        meds, ns = [], []
        for p in PENDINGS:
            fast = [cell[(rep, pol, p)]['ttft'] for rep in range(1, 6) if not slow[(rep, pol, p)]]
            meds.append(qmed(fast))
            ns.append(len(fast))
        print('%s 快模式等权 TTFT %.1f（各格 n=%s，格中位 %s）' % (pol, st.mean(meds), ns, [round(m, 1) for m in meds]))
        out_rows.append(['fast_only_sensitivity', '', pol, '', 'eq_ttft_ms', round(st.mean(meds), 1),
                         sum(ns), '各格n=' + ','.join(map(str, ns))])
    for pair in ('FS-SA', 'FS-WFC'):
        a, b = pair.split('-')
        fa = st.mean([qmed([cell[(rep, a, p)]['ttft'] for rep in range(1, 6) if not slow[(rep, a, p)]]) for p in PENDINGS])
        fb = st.mean([qmed([cell[(rep, b, p)]['ttft'] for rep in range(1, 6) if not slow[(rep, b, p)]]) for p in PENDINGS])
        print('快模式等权 %s Δ = %.1f' % (pair, fa - fb))
        out_rows.append(['fast_only_sensitivity', '', pair, '', 'eq_ttft_delta_ms', round(fa - fb, 1), 0,
                         '基于各格快模式中位'])

    # ── §六：控制分段（实测）与增量 ──
    print('\n== §六 控制分段中位（ms/apply，三策略对比→增量控制成本）==')
    for pol in POL:
        wall = qmed([cell[(rep, pol, p)]['ctrl'] for rep in range(1, 6) for p in PENDINGS])
        auth = qmed([cell[(rep, pol, p)]['auth'] for rep in range(1, 6) for p in PENDINGS])
        recon = qmed([cell[(rep, pol, p)]['recon'] for rep in range(1, 6) for p in PENDINGS])
        print('%s: RPC墙钟 %.0f | 认证+计划 %.0f | 协调+后检 %.0f | 生产核心合计 %.0f' % (pol, wall, auth, recon, auth + recon))
        out_rows.append(['control_segments', '', pol, '', 'rpc_wall|auth_plan|reconcile|core_sum',
                         '%.0f|%.0f|%.0f|%.0f' % (wall, auth, recon, auth + recon), 20, '中位'])

    # 队列/pregap（未解释宿主计时的量化上界：窗口内非计算段）
    print('\n== 窗口内非计算段（queue+pregap 中位, ms）==')
    for pol in POL:
        q = qmed([cell[(rep, pol, p)]['queue'] + cell[(rep, pol, p)]['pregap'] for rep in range(1, 6) for p in PENDINGS])
        print('%s: %.1f' % (pol, q))

    with (OUT / 'paper_pending_aggregate.csv').open('w', newline='') as f:
        csv.writer(f).writerows(out_rows)
    print('\n写出 paper_pending_aggregate.csv（%d 行）' % len(out_rows))


if __name__ == '__main__':
    main()
