"""P0-2：Runtime 控制成本与实际收益归因最终审计（CPU-only 只读复算）。

数据源（全部只读冻结产物，不改任何实验数据）：
- T2-G2 三策略 60 臂：evaluation/t2g1_three_policy_gpu/formal_gpu1b/
  （analysis_summary.json 的 control_decomposition / pending_policy_matrix.csv / pair_analysis.csv /
   round_equal_weight.csv）
- C/K3 三策略 60 臂：evaluation/c_formal_perf_gpu/paper_pending_aggregate.csv 的 control_segments
- RQ6 控制开销基线：evaluation/rq6_overhead_output/rq6_overhead_20260909_061500/analysis/
- RQ6-E 批量控制优化：evaluation/rq6_batched_control_output/rq6e_20260911_085820/analysis/
- Transport 优化 + Control-inclusive Resume-Epoch：
  evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json

复核内容：
1 T2-G2 控制成本：三策略 Apply RPC 分布、七段分解（生产核心 vs A 类研究仪表）、
  同轮配对控制差、四 Pending 等权 TTFT 收益复核；
2 摊销两口径：含研究仪表的 RPC 墙钟配对差口径 vs 生产核心（认证+计划+协调）口径，
  不同 Harness（T2-G2/C-K3/RQ6/control_path）测量互不拼接；
3 RQ6→RQ6-E 与 transport 前后对照的数字、门禁与语义等价独立核验；
4 Control-inclusive 完整 Epoch 配对差与 95% CI（不得解释为端到端加速）。
"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2 = ROOT / 'evaluation/t2g1_three_policy_gpu/formal_gpu1b'
CK = ROOT / 'evaluation/c_formal_perf_gpu/paper_pending_aggregate.csv'
RQ6 = ROOT / 'evaluation/rq6_overhead_output/rq6_overhead_20260909_061500/analysis/overhead_summary.json'
RQ6E = ROOT / 'evaluation/rq6_batched_control_output/rq6e_20260911_085820/analysis/aggregate_results.json'
CP = ROOT / 'evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json'
OUT = ROOT / 'evaluation/paper_control_cost_final_audit'


def main():
    checks, parts = [], {}

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})

    # ══ 1) T2-G2 控制成本与收益复核 ══
    summ = json.loads((T2 / 'analysis_summary.json').read_text())
    decomp = summ['control_decomposition']
    pols = {'FlowState-Greedy': 'FS', 'Marconi-Evict': 'MC', 'KVFlow-inspired': 'KV'}
    # 生产核心（C 类状态管理）= 认证+计划 + 协调及合同后检；其余分段为 A 类研究仪表
    # （分类依据 c_formal_perf_gpu/audit_classification.md：preflight/fa 审计/重型观察/census 属 A 类）
    core, audit, rpc_med = {}, {}, {}
    for pol, short in pols.items():
        d = decomp[pol]
        s = d['stages_mean_ms']
        core[short] = round(s['strict_epoch_authentication_and_plan'] + s['strict_reconciliation_with_postchecks'], 1)
        audit[short] = round(s['apply_time_freshness_preflight'] + s['fa_content_audit_before']
                             + s['fa_content_postcheck_and_resume_baseline'] + s['final_heavy_observation'], 1)
        rpc_med[short] = d['apply_rpc_wall_median_ms']
    ppm = list(csv.DictReader((T2 / 'pending_policy_matrix.csv').open()))
    rpc_by_pol = {}
    for pol, short in pols.items():
        rpc_by_pol[short] = [float(r['control_rpc_median_ms']) for r in ppm if r['policy'] == pol]
    # 同轮配对控制差与等权 TTFT 收益
    pairs = list(csv.DictReader((T2 / 'pair_analysis.csv').open()))
    ctrl_diff = {c: [float(r['control_diff_ms']) for r in pairs if r['comparator'] == c]
                 for c in ('FS-MC', 'FS-KV')}
    rew = list(csv.DictReader((T2 / 'round_equal_weight.csv').open()))
    ttft_diff = {c: st.median([float(r['diff_ms']) for r in rew if r['comparator'] == c])
                 for c in ('FS-MC', 'FS-KV')}
    # 摊销两口径（同一 T2-G2 协议内；每生命周期一次 apply，四 Pending 每轮各一请求）
    amort = {}
    for c in ('FS-MC', 'FS-KV'):
        d_rpc = st.median(ctrl_diff[c])                       # 含研究仪表的 RPC 墙钟配对差
        d_core = (core['FS'] - core[c.split('-')[1]])         # 生产核心均值差
        saved = -ttft_diff[c]                                 # 每请求等权平均节省（正数）
        amort[c] = {
            'ctrl_rpc_paired_diff_median_ms': round(d_rpc, 1),
            'production_core_diff_ms': round(d_core, 1),
            'eq_ttft_saved_per_request_ms': round(saved, 1),
            'break_even_rpc_inclusive_requests': -(-d_rpc // saved) if d_rpc > 0 else None,
            'break_even_production_core_requests': -(-d_core // saved) if d_core > 0 else 0.0}
    parts['T2G2_control'] = {
        'apply_rpc_wall_median_ms': rpc_med,
        'production_core_ms': core, 'audit_stages_ms': audit,
        'production_core_share_percent': {k: round(core[k] / rpc_med[k] * 100, 1) for k in core},
        'paired_control_diff_ms': {c: {'median': round(st.median(v), 1),
                                       'min': round(min(v), 1), 'max': round(max(v), 1),
                                       'n': len(v)} for c, v in ctrl_diff.items()},
        'eq_ttft_paired_diff_median_ms': {c: round(v, 3) for c, v in ttft_diff.items()},
        'amortization': amort}
    check('T2G2: spec 给定等权配对差复核（FS-MC −114.127 / FS-KV −105.728）',
          abs(ttft_diff['FS-MC'] + 114.127) < 0.01 and abs(ttft_diff['FS-KV'] + 105.728) < 0.01,
          ttft_diff)
    check('T2G2: 生产核心占 RPC 墙钟比例 <15%（仪表主导墙钟）',
          all(v < 15 for v in parts['T2G2_control']['production_core_share_percent'].values()),
          parts['T2G2_control']['production_core_share_percent'])

    # ══ 2) C/K3 控制分段复核（另一 Harness，不与 T2-G2 拼接） ══
    seg = {}
    for r in csv.DictReader(CK.open()):
        if r['section'] == 'control_segments':
            seg[r['policy_or_pair']] = dict(zip(['rpc_wall', 'auth_plan', 'reconcile', 'core_sum'],
                                                (float(x) for x in r['value'].split('|'))))
    ck_amort = {}
    for base, saved_ms in (('SA', 34.1), ('WFC', 46.9)):  # C/K3 等权配对 TTFT 中位（冻结）
        d_core = seg['FS']['core_sum'] - seg[base]['core_sum']
        ck_amort['FS-%s' % base] = {
            'production_core_diff_ms': round(d_core, 1),
            'eq_ttft_saved_per_request_ms': saved_ms,
            'break_even_production_core_requests': -(-d_core // saved_ms) if d_core > 0 else 0.0,
            'fs_self_core_amortize_deep_requests': -(-seg['FS']['core_sum'] // 163.2)}  # C_B 型单请求 163.2ms
    parts['CK3_control'] = {'segments': seg, 'amortization': ck_amort}
    check('CK3: 生产核心 FS 与基线差 <20ms（协议内可比）',
          all(abs(ck_amort[k]['production_core_diff_ms']) < 20 for k in ck_amort), ck_amort)

    # ══ 3) RQ6 → RQ6-E 批量控制优化独立核验 ══
    rq6 = json.loads(RQ6.read_text())['by_kind']['runtime_control']['stages_ms']
    rq6e = json.loads(RQ6E.read_text())
    perf = rq6e['performance']
    # 前后为同一冻结快照集（24 快照 72 正式 run）、同四阶段计时口径
    same_baseline = abs(rq6['total_control']['mean'] - perf['total_control']['before_mean_ms']) < 1
    parts['RQ6_optimization'] = {
        'harness': 'RQ6 冻结 24 快照 × 3 = 72 正式 run（前后同基线、同 perf_counter_ns 四阶段口径）',
        'before_total_ms': rq6['total_control']['mean'],
        'after_total_ms': perf['total_control']['after_mean_ms'],
        'reduction_percent': round(perf['total_control']['reduction_percent'], 3),
        'stages': {k: {'before': rq6[k]['mean'],
                       'after': perf[k]['after_mean_ms']} for k in ('introspection', 'reconciliation')},
        'rpc': {'before_per_epoch': rq6e['rpc']['before_mean_per_epoch'],
                'after_per_epoch': rq6e['rpc']['after_mean_per_epoch']},
        'semantic_equivalence_gates': rq6e['gates']}
    check('RQ6E: before 均值与 RQ6 基线一致（同基线可比）', same_baseline,
          {'rq6': rq6['total_control']['mean'], 'rq6e_before': perf['total_control']['before_mean_ms']})
    check('RQ6E: 8398.847→629.827 复核（−92.50%）',
          abs(perf['total_control']['before_mean_ms'] - 8398.847) < 0.01
          and abs(perf['total_control']['after_mean_ms'] - 629.827) < 0.01
          and abs(perf['total_control']['reduction_percent'] - 92.501) < 0.01)
    check('RQ6E: 语义等价门禁全过（selected-set/H-E-G/FA 等）', all(rq6e['gates'].values()))

    # ══ 4) Transport 优化 + Control-inclusive 完整 Epoch ══
    cp = json.loads(CP.read_text())
    old_fs = cp['old_means']['FlowState']
    new_fs = cp['new_means']['FlowState']
    paired_ci = {p: cp['new_paired'][p]['control_inclusive_epoch_latency_ms'] for p in ('LRU', 'Marconi')}
    parts['control_path_and_epoch'] = {
        'transport_optimization': {
            'harness': 'control_path resume-epoch harness（前后各自完整采集，时段不同、不拼接）',
            'flowstate_total_control_before_ms': old_fs['total_control_latency_ms'],
            'flowstate_total_control_after_ms': new_fs['total_control_latency_ms'],
            'reduction_percent': round((1 - new_fs['total_control_latency_ms']
                                        / old_fs['total_control_latency_ms']) * 100, 1),
            'semantic_equivalence': {'identical_plan': cp['identical_plan'],
                                     'same_selected_sets': cp['same_selected_sets'],
                                     'same_request_heg': cp['same_request_heg']}},
        'request_execution_ms': {p: {'before': cp['old_means'][p]['mean_request_execution_ms'],
                                     'after': cp['new_means'][p]['mean_request_execution_ms']}
                                 for p in ('LRU', 'Marconi', 'FlowState')},
        'control_inclusive_epoch_paired': {
            p: {'mean_diff_ms': round(v['mean_difference_ms'], 1),
                'median_diff_ms': round(v['median_difference_ms'], 1),
                'bootstrap_95_ci_ms': [round(x, 1) for x in v['bootstrap_95_ci_ms']],
                'win_loss': '%d/%d' % (v['win'], v['loss'])} for p, v in paired_ci.items()},
        'statistically_stable_net_benefit_vs_both': cp['statistically_stable_net_benefit_vs_both']}
    check('control_path: 656.771→131.904 复核（≈−79.9%）',
          abs(old_fs['total_control_latency_ms'] - 656.771) < 0.01
          and abs(new_fs['total_control_latency_ms'] - 131.904) < 0.01)
    check('control_path: 语义保持（identical_plan 且选集/HEG 等价）',
          cp['identical_plan'] and cp['same_selected_sets'] == 216 and cp['same_request_heg'] == 864)
    check('完整 Epoch: 两组配对差 95%CI 均含零（未证明稳定净收益）',
          all(v['bootstrap_95_ci_ms'][0] < 0 < v['bootstrap_95_ci_ms'][1] for v in paired_ci.values()))
    check('完整 Epoch: FlowState 平均 Request Execution 最低（前后均成立）',
          min(cp['old_means'][p]['mean_request_execution_ms'] for p in ('LRU', 'Marconi', 'FlowState'))
          == old_fs['mean_request_execution_ms']
          and min(cp['new_means'][p]['mean_request_execution_ms'] for p in ('LRU', 'Marconi', 'FlowState'))
          == new_fs['mean_request_execution_ms'])

    parts['checks'] = checks
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_data.json').write_text(json.dumps(parts, ensure_ascii=False, indent=1))
    print('== T2-G2 ==')
    print(' RPC 墙钟中位:', rpc_med, '生产核心:', core, '（占比 %s%%）' % parts['T2G2_control']['production_core_share_percent'])
    print(' 配对控制差:', parts['T2G2_control']['paired_control_diff_ms'])
    print(' 等权 TTFT 差:', parts['T2G2_control']['eq_ttft_paired_diff_median_ms'])
    print(' 摊销:', json.dumps(amort, ensure_ascii=False))
    print('== C/K3 ==', json.dumps(parts['CK3_control'], ensure_ascii=False))
    print('== RQ6E == 8398.847→629.827（%.3f%%），门禁 %d/%d' % (
        perf['total_control']['reduction_percent'], sum(rq6e['gates'].values()), len(rq6e['gates'])))
    print('== control_path == 656.771→131.904；完整 Epoch CI:',
          json.dumps(parts['control_path_and_epoch']['control_inclusive_epoch_paired'], ensure_ascii=False))
    print('检查 %d 项，失败：%s' % (len(checks), [c['check'] for c in checks if not c['pass']] or '无'))


if __name__ == '__main__':
    main()
