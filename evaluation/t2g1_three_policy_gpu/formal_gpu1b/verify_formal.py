"""T2-G2 正式 GPU 对照收尾验证：按冻结合同逐臂核对（CPU-only，读原始证据，不重跑任何 GPU）。

用法：python3 -B verify_formal.py gate|main|all
- gate：3 门禁臂（结果只作正确性/有效性证据，不进性能统计）；
- main：60 主臂（性能统计的数据基座，全部合同必须 PASS）；
- all：两者并验。

核对项（每臂，与 protocol.gate_criteria / c_formal 测量合同逐条对应）：
1 summary.status==PASS 且 policy/pending 与 protocol 冻结调度一致；
2 census 16/16/0；selected_cpu == t2g1_plans 冻结选集；
3 E == 计划 E、G == T−E、H == 冻结 attention_frontier、prefill == len(pt)−E、batch_prefix == E；
4 测量有效性：measure_mode=True、窗口审计项==2、形状预热行 E 与 FA 命中均恰为计划 E、
   请求窗口 Triton 新增编译==0；
5 输出与独立 Cold 参照逐 Token 一致（独立重比，不信 summary 内的 cold_output_match 标志）；
6 驱逐 13 项全部 returned（读 reconciliation.json 原始记录）；
7 TTFT 六段分解闭合残差 <5ms（runner 内已断言，此处复核留档）。
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
OUT = T2G1 / 'formal_gpu1b'
POLICIES = ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired']


def read(p):
    return json.loads(Path(p).read_text())


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    protocol = read(OUT / 'protocol.json')
    plans = read(T2G1 / 't2g1_plans.json')['plans']
    cold = {pid: read(OUT / ('cold_reference_%s.json' % pid))['output_ids'] for pid in ['C_A', 'C_B', 'C_C', 'C_D']}
    ptlen = {p['continuation_id']: len(p['token_ids']) for p in read(OUT / 'prepared_inputs.json')['pendings']}
    frontier = {p['continuation_id']: p['attention_frontier'] for p in read(OUT / 'prepared_inputs.json')['pendings']}

    arms = (protocol['gate_arms'] if phase in ('gate', 'all') else []) + \
           (protocol['main_arms'] if phase in ('main', 'all') else [])
    checks, rows, failures = [], [], []

    def check(n, name, ok, detail=''):
        checks.append({'arm': n, 'check': name, 'pass': bool(ok), 'detail': str(detail)[:200]})
        if not ok:
            failures.append((n, name, str(detail)[:200]))

    for arm in arms:
        case = OUT / arm['name']
        s = read(case / 'run' / 'summary.json')
        exp = plans[arm['policy']]['expected_E_G'][arm['pending']]
        n = arm['name']
        check(n, 'status', s['status'] == 'PASS' and s['policy'] == arm['policy'] and s['pending'] == arm['pending'])
        check(n, 'census16', s['census_counts'] == {'physical': 16, 'mapped': 16, 'unmapped_extras': 0})
        check(n, 'selected', s['selected_cpu'] == plans[arm['policy']]['selected_checkpoint_ids'])
        check(n, 'E_G_H', s['E_executable_prefix'] == exp['E'] and s['G_replay_gap'] == exp['G'] and
              s['H_physical_fa_hit'] == frontier[arm['pending']], 'E=%s G=%s H=%s 预期E=%s G=%s H=%s' % (
                  s['E_executable_prefix'], s['G_replay_gap'], s['H_physical_fa_hit'], exp['E'], exp['G'],
                  frontier[arm['pending']]))
        check(n, 'prefill', s['prefill_token_count'] == ptlen[arm['pending']] - exp['E'] and
              s['batch_prefix_length'] == exp['E'], 'prefill=%s batch_prefix=%s' % (
                  s['prefill_token_count'], s['batch_prefix_length']))
        check(n, 'measure_mode', s.get('measure_mode') is True and s.get('audit_entries_total') == 2,
              'audit=%s' % s.get('audit_entries_total'))
        sw = s.get('shape_warm_b', {})
        check(n, 'shape_warm', sw.get('E') == exp['E'] and sw.get('fa_hit') == exp['E'],
              'swE=%s swFA=%s 计划E=%s' % (sw.get('E'), sw.get('fa_hit'), exp['E']))
        tw = s.get('triton_new_kernels', [])
        check(n, 'zero_compile', bool(tw) and tw[-1]['new_entries'] == 0, str(tw[-1] if tw else None))
        check(n, 'cold_tokens', s['output_ids'] == cold[arm['pending']],
              '输出=%s Cold=%s' % (s['output_ids'][:10], cold[arm['pending']][:10]))
        recon = read(case / 'run' / 'reconciliation.json')
        ops = recon['operations']
        check(n, 'evictions13', len(ops) == 13 and all(x['returned'] for x in ops), '驱逐数=%d' % len(ops))
        d = s.get('timing_decomposition', {})
        check(n, 'ttft_closure', 'ttft_ms_check' in d and abs(
            (d['queue_ms'] + d['execwin_ms'] + d['delivery_ms']) - d['ttft_ms_check']) < 5.0)
        rows.append({'arm': n, 'policy': arm['policy'], 'pending': arm['pending'],
                     'rep': arm.get('rep', 0), 'phase': 'gate' if n.startswith('g_') else 'main',
                     'E': s['E_executable_prefix'], 'G': s['G_replay_gap'],
                     'prefill_tokens': s['prefill_token_count'],
                     'ttft_ms': round(s['ttft_ms'], 3), 'request_wall_ms': round(s['request_wall_ms'], 3),
                     'cuda_prefill_ms': round(s['cuda_prefill_ms'], 3),
                     'cuda_decode_ms': round(s['cuda_decode_ms'], 3),
                     'cow_copy_cuda_ms': round(s['cow_copy_cuda_ms'], 4),
                     'cow_copy_host_ms': round(s['cow_copy_host_ms'], 4),
                     'match_native_ms': s['match_native_ms'],
                     'control_rpc_wall_ms': round(s['control_rpc_wall_ms'], 2),
                     'compute_cuda_in_window_ms': round(d.get('compute_cuda_ms', 0), 3),
                     'shape_warm_E': sw.get('E'), 'audit': s.get('audit_entries_total'),
                     'J_plan_ms': plans[arm['policy']].get('J_ms_current_scope')})

    verdict = 'PASS' if not failures else 'FAIL'
    (OUT / ('formal_verification_%s.json' % phase)).write_text(json.dumps(
        {'verdict': verdict, 'phase': phase, 'arms_checked': len(arms), 'checks_total': len(checks),
         'failures': failures, 'checks': checks}, ensure_ascii=False, indent=2))
    cols = list(rows[0].keys())
    with (OUT / ('formal_arms_%s.csv' % phase)).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print('验证 %s：%d 臂 %d 检查 → %s' % (phase, len(arms), len(checks), verdict))
    for name, chk, det in failures:
        print('FAIL', name, chk, det)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
