"""P0-4 五轮矩阵·统计提取(第五阶段;前置条件:15/15 臂全 PASS,否则拒绝运行)。

口径(任务书第四阶段纪律):
- 逐轮配对差(FS−MC、FS−KV)后再取五轮中位数——禁止"总中位数相减"式配对;
- 双口径计时(control_cycle_ms / cycle_ms_full_incl_warm)全程分开输出,严禁混用;
- r1 Pilot 不进样本;n=5 不做显著性声明,仅报告逐轮值/中位数/极值;
- 逐 Pending TTFT 配对差同样逐轮计算(按 pending 顺序对齐,冻结输入恒同)。

产物(一次性写入,拒绝覆盖):
- arms/five_round_per_round.csv   逐轮×策略明细(双口径+ΣTTFT+Σprefill+J+选集)
- arms/five_round_paired.csv      逐轮配对差(ΣTTFT/Σprefill/逐Pending TTFT)+五轮中位行
- arms/five_round_stats.json      上述全量+全指标中位/极值+PASS 计数断言结果
"""
import csv
import json
import statistics
from pathlib import Path

from . import launch, launch_round
from .prepare_round import ROUND_ORDER, arm_name

ARMS = launch.ARMS
POLICIES = ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired']
DUAL = ['control_cycle_ms', 'cycle_ms_full_incl_warm']


def load_ledger_rows():
    """读总账并断言:15 行全 PASS、轮×序位==拉丁方、身份哈希==冻结值。"""
    rows = [json.loads(x) for x in (ARMS / 'five_round_ledger.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows) == 15, '总账须为 15 行(当前 %d),拒绝统计' % len(rows)
    seen = set()
    for r in rows:
        assert r['status'] == 'PASS', '存在非 PASS 臂(轮%s %s),按纪律拒绝五轮统计' % (r['round'], r['policy'])
        assert r['runner_sha256'] == launch_round.FROZEN_IDENTITY['runner.py'], 'runner 身份漂移'
        assert r['observation_sha256'] == launch_round.FROZEN_IDENTITY['observation.py'], 'observation 身份漂移'
        seen.add((r['round'], r['policy']))
    assert seen == {(rnd, p) for rnd in ROUND_ORDER for p in POLICIES}, '总账轮×策略覆盖不全'
    return rows


def arm_metrics(round_no, policy):
    """从臂 summary 提取指标(与总账行同源同口径,另取逐 Pending 明细)。"""
    arm = arm_name(policy, round_no)
    s = json.loads((ARMS / arm / 'run' / 'summary.json').read_text())
    rows, cyc, sel = s['pending_rows'], s['cycle'], s['selection']
    return {
        'arm': arm, 'E': [r['E'] for r in rows], 'T': [r['T'] for r in rows],
        'G': [r['G'] for r in rows],
        'ttft_ms': [round(r['ttft_ms'], 3) for r in rows],
        'cuda_prefill_ms': [round(r['cuda_prefill_ms'], 3) for r in rows],
        'control_cycle_ms': cyc['control_cycle_ms'],
        'cycle_ms_full_incl_warm': cyc['cycle_ms_full_incl_warm'],
        'request_exec_total_ms': cyc['request_exec_total_ms'],
        'sum_ttft_ms': round(sum(r['ttft_ms'] for r in rows), 3),
        'sum_cuda_prefill_ms': round(sum(r['cuda_prefill_ms'] for r in rows), 3),
        'J_ms': sel['plan_recovery_cost_ms'],
        'selection_ids': sel['selected_cpu'],
    }


def main():
    rows = load_ledger_rows()
    data = {rnd: {p: arm_metrics(rnd, p) for p in POLICIES} for rnd in ROUND_ORDER}

    # ---- 逐轮明细 CSV(双口径分列,严禁混用)----
    per_round_path = ARMS / 'five_round_per_round.csv'
    assert not per_round_path.exists(), '拒绝覆盖:' + str(per_round_path)
    with per_round_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['round', 'policy', 'arm', 'control_cycle_ms', 'cycle_ms_full_incl_warm',
                    'request_exec_total_ms', 'sum_ttft_ms', 'sum_cuda_prefill_ms', 'J_ms',
                    'ttft_ms_per_pending', 'E_per_pending', 'selection_ids'])
        for rnd in sorted(ROUND_ORDER):
            for p in POLICIES:
                m = data[rnd][p]
                w.writerow([rnd, p, m['arm'], m['control_cycle_ms'], m['cycle_ms_full_incl_warm'],
                            m['request_exec_total_ms'], m['sum_ttft_ms'], m['sum_cuda_prefill_ms'],
                            m['J_ms'], '|'.join(str(x) for x in m['ttft_ms']),
                            '|'.join(str(x) for x in m['E']), '|'.join(m['selection_ids'])])

    # ---- 逐轮配对差(先逐轮、后五轮中位)----
    paired = {}
    for base in ('Marconi-Evict', 'KVFlow-inspired'):
        key = 'FS-' + ('MC' if base.startswith('Marconi') else 'KV')
        paired[key] = {
            'sum_ttft_ms': [round(data[r]['FlowState-Greedy']['sum_ttft_ms'] - data[r][base]['sum_ttft_ms'], 3)
                            for r in sorted(ROUND_ORDER)],
            'sum_cuda_prefill_ms': [round(data[r]['FlowState-Greedy']['sum_cuda_prefill_ms'] - data[r][base]['sum_cuda_prefill_ms'], 3)
                                    for r in sorted(ROUND_ORDER)],
            'ttft_per_pending': [[round(f - b, 3) for f, b in zip(data[r]['FlowState-Greedy']['ttft_ms'], data[r][base]['ttft_ms'])]
                                 for r in sorted(ROUND_ORDER)],
        }
    paired_path = ARMS / 'five_round_paired.csv'
    assert not paired_path.exists(), '拒绝覆盖:' + str(paired_path)
    with paired_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['comparison', 'metric', 'R1', 'R2', 'R3', 'R4', 'R5', 'median_of_5'])
        for key, vals in paired.items():
            for metric in ('sum_ttft_ms', 'sum_cuda_prefill_ms'):
                v = vals[metric]
                w.writerow([key, metric] + v + [round(statistics.median(v), 3)])
            per = vals['ttft_per_pending']           # 5 轮×4 pending → 逐 pending 中位
            for i in range(4):
                v = [per[r][i] for r in range(5)]
                w.writerow([key, 'ttft_pending_%d' % (i + 1)] + v + [round(statistics.median(v), 3)])

    # ---- 全指标中位/极值 + J 一致性 ----
    summary_stats = {}
    for p in POLICIES:
        summary_stats[p] = {}
        for field in ('sum_ttft_ms', 'sum_cuda_prefill_ms', 'J_ms') + tuple(DUAL):
            v = [data[r][p][field] for r in sorted(ROUND_ORDER)]
            summary_stats[p][field] = {'median': round(statistics.median(v), 3),
                                       'min': round(min(v), 3), 'max': round(max(v), 3)}
    # J(模型化恢复成本)与实测 Σcuda_prefill 逐轮方向一致性:J 更低者实测 prefill 是否更低
    j_check = []
    for r in sorted(ROUND_ORDER):
        fs, mc, kv = (data[r][x] for x in POLICIES)
        j_check.append({
            'round': r,
            'J_order': sorted(POLICIES, key=lambda p: data[r][p]['J_ms']),
            'prefill_order': sorted(POLICIES, key=lambda p: data[r][p]['sum_cuda_prefill_ms']),
            'J_FS_minus_MC': round(fs['J_ms'] - mc['J_ms'], 3),
            'prefill_FS_minus_MC': round(fs['sum_cuda_prefill_ms'] - mc['sum_cuda_prefill_ms'], 3),
            'J_FS_minus_KV': round(fs['J_ms'] - kv['J_ms'], 3),
            'prefill_FS_minus_KV': round(fs['sum_cuda_prefill_ms'] - kv['sum_cuda_prefill_ms'], 3),
        })
    # 确定性核验:选集(集合语义)/J/E 向量五轮应逐位相同
    for p in POLICIES:
        sels = {tuple(sorted(data[r][p]['selection_ids'])) for r in ROUND_ORDER}
        js = {data[r][p]['J_ms'] for r in ROUND_ORDER}
        es = {tuple(data[r][p]['E']) for r in ROUND_ORDER}
        assert len(sels) == 1 and len(js) == 1 and len(es) == 1, '确定性破坏:' + p

    out = {
        'n_rounds': 5, 'n_arms': 15, 'all_pass': True,
        'per_round': data, 'paired': paired, 'summary_stats': summary_stats,
        'J_vs_realized_prefill': j_check,
        'note': '口径:逐轮配对差后取五轮中位;双口径分列禁混用;r1 Pilot 不进样本;n=5 不做显著性声明',
    }
    stats_path = ARMS / 'five_round_stats.json'
    assert not stats_path.exists(), '拒绝覆盖:' + str(stats_path)
    stats_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # ---- 控制台摘要 ----
    print('15/15 PASS;双口径中位数(五轮):')
    for p in POLICIES:
        st = summary_stats[p]
        print('  %-18s ctrl=%8.1f full=%8.1f ΣTTFT=%6.1f Σprefill=%6.1f J=%7.3f' % (
            p, st['control_cycle_ms']['median'], st['cycle_ms_full_incl_warm']['median'],
            st['sum_ttft_ms']['median'], st['sum_cuda_prefill_ms']['median'], st['J_ms']['median']))
    for key, vals in paired.items():
        v = vals['sum_ttft_ms']
        print('  配对 %s ΣTTFT 逐轮=%s 中位=%.1f' % (key, v, statistics.median(v)))
    print('产物:', per_round_path.name, paired_path.name, stats_path.name)


if __name__ == '__main__':
    main()
