"""T2-G1 Smoke 收尾验证：三臂结果与 T2-G1 计划/独立 Cold 参照逐项核对，产出 smoke_results.json。

核对项（每臂）：
- summary.status==PASS、policy/pending 正确、census 物理 16；
- E == t2g1_plans.expected_E_G[C_B].E；prefill_token_count == len(pending tokens) − E；batch_prefix == E；
- FA 命中 == 冻结 attention_frontier 且未被策略改变（runner 内已断言，此处复核记录）；
- 输出 Token 与 pilot 冻结 cold_C_B（独立空缓存生命周期）逐 Token 一致；
- 控制分段计时与驱逐落实记录在案（计时为指示性，非正式测量口径）。
"""
import csv
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
SMOKE = ROOT / 'evaluation/t2g1_three_policy_gpu/smoke'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
ARMS = ['strict_FlowState-Greedy_C_B', 'strict_Marconi-Evict_C_B', 'strict_KVFlow-inspired_C_B']


def read(p):
    return json.loads(Path(p).read_text())


def main():
    plans = read(ROOT / 'evaluation/t2g1_three_policy_gpu/t2g1_plans.json')['plans']
    cold = read(PILOT / 'cold_C_B/run/summary.json')
    pending = read(ROOT / 'evaluation/c_shared_gpu_pilot_k3/prepared_inputs.json')['pendings']
    pt = next(p['token_ids'] for p in pending if p['continuation_id'] == 'C_B')
    rows, checks = [], []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
        if not ok:
            raise AssertionError('smoke 验证失败：%s %s' % (name, detail))

    for arm in ARMS:
        policy = arm.split('_', 2)[1]
        s = read(SMOKE / arm / 'run/summary.json')
        exp = plans[policy]['expected_E_G']['C_B']
        check('%s_status' % policy, s['status'] == 'PASS' and s['policy'] == policy and s['pending'] == 'C_B')
        check('%s_census16' % policy, s['census_counts']['physical'] == 16 and s['census_counts']['mapped'] == 16
              and s['census_counts']['unmapped_extras'] == 0)
        check('%s_E' % policy, s['E_executable_prefix'] == exp['E'],
              'E实际=%s 预期=%s' % (s['E_executable_prefix'], exp['E']))
        check('%s_prefill' % policy, s['prefill_token_count'] == len(pt) - exp['E'] and s['batch_prefix_length'] == exp['E'],
              'prefill=%s batch_prefix=%s 预期E=%s' % (s['prefill_token_count'], s['batch_prefix_length'], exp['E']))
        check('%s_fa' % policy, s['H_physical_fa_hit'] == 5120 and s['G_replay_gap'] == 5120 - exp['E'])
        match_cold = s['output_ids'] == cold['output_ids']
        check('%s_output_vs_cold' % policy, match_cold, '输出与独立Cold参照逐Token一致' if match_cold else
              '输出与独立Cold参照不一致：实际=%s 预期=%s' % (s['output_ids'], cold['output_ids']))
        check('%s_selected' % policy, s['selected_cpu'] == plans[policy]['selected_checkpoint_ids'])
        recon = read(SMOKE / arm / 'run/reconciliation.json')
        evicted = [x['checkpoint_id'] for x in recon['operations']]
        check('%s_evictions' % policy, len(evicted) == 13 and all(x['returned'] for x in recon['operations']),
              '驱逐数=%d' % len(evicted))
        rows.append({'policy': policy, 'pending': 'C_B', 'selected_cpu': s['selected_cpu'],
                     'E': s['E_executable_prefix'], 'G': s['G_replay_gap'], 'prefill_tokens': s['prefill_token_count'],
                     'ttft_ms': round(s['ttft_ms'], 2), 'cuda_prefill_ms': round(s['cuda_prefill_ms'], 2),
                     'cuda_decode_ms': round(s['cuda_decode_ms'], 2), 'cow_copy_cuda_ms': round(s['cow_copy_cuda_ms'], 2),
                     'control_rpc_wall_ms': round(s['control_rpc_wall_ms'], 2),
                     'control_stages_ms': {t['stage']: round(t['ms'], 2) for t in s['control_timing_stages']},
                     'evicted_count': len(evicted), 'output_match_cold': match_cold,
                     'J_plan_ms': plans[policy]['J_ms_current_scope']})
    (SMOKE / 'smoke_results.json').write_text(json.dumps(
        {'verdict': 'PASS', 'checks': checks, 'arms': rows,
         'note': '计时为单臂指示性观测（无形状预热/测量模式），正式测量见 formal_plan.md 60 臂协议；'
                 '输出一致性对照=pilot 冻结 cold_C_B（独立空缓存生命周期）',
         'cold_reference': {'source': 'c_shared_gpu_pilot_k3/cold_C_B', 'output_ids': cold['output_ids'],
                            'prefill_full': cold['prefill_token_count']}}, ensure_ascii=False, indent=2))
    with (SMOKE / 'smoke_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'control_stages_ms'])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != 'control_stages_ms'})
    print('smoke 验证 PASS：%d 检查' % len(checks))
    for r in rows:
        print('%-16s E=%d prefill=%d ttft=%.1f cuda_prefill=%.1f 控制RPC=%.0f' % (
            r['policy'], r['E'], r['prefill_tokens'], r['ttft_ms'], r['cuda_prefill_ms'], r['control_rpc_wall_ms']))


if __name__ == '__main__':
    main()
