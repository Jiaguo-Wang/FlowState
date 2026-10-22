"""离线汇总：只读各臂产物，产出 7 项交付数据（CSV/JSON），不重跑、不改臂内文件。

- arm_results.csv：逐臂 请求级性能(H/E/G、Prefill Token、CUDA Prefill、TTFT) + 控制开销(RPC/分段)；
- pending_policy_matrix.csv：Pending×Policy 恢复前沿与性能对照（问题 A/B/C 数据基础）；
- control_overhead.csv：控制分段墙钟 vs 请求执行收益（问题 D 数据基础）；
- cold_comparison.json：逐 Pending Cold 输出一致性（交付 4）；
- finalize_results.json：全部汇总 + census 账本 + 结论数据（READY/PARTIAL/BLOCKED 依据）。
"""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))


def read(p):
    return json.loads(Path(p).read_text())


def main():
    assert not (OUT / 'finalize_results.json').exists(), '汇总结果已存在，禁止覆盖'
    protocol = read(OUT / 'protocol.json')
    arms = protocol['arms']
    rows = []
    matrix = {}
    overhead = []
    census_ledger = None
    cold_outputs = {}
    evictions = {}
    failures = {}
    for arm in arms:
        case = OUT / arm
        mode = arm.split('_')[0]
        summary_path = case / 'run' / 'summary.json'
        if not summary_path.exists():
            rows.append({'arm': arm, 'status': 'NOT_RUN'})
            continue
        s = read(summary_path)
        if s.get('status') != 'PASS':
            rows.append({'arm': arm, 'status': s.get('status', 'UNKNOWN')})
            continue
        if mode == 'census':
            census_ledger = {'physical_domain_size': s['physical_domain_size'],
                             'cpu_domain_size': s['cpu_domain_size'], 'mapped_count': s['mapped_count'],
                             'unmapped_extras': s['unmapped_extras'], 'missing_physical': s['missing_physical'],
                             'mapping': s['mapping'], 'pendings_live_vs_declared': s['pendings_live_vs_declared'],
                             'slots_used': s['slots_used'], 'free_slots': s['free_slots'],
                             'pool_capacity': s['pool_capacity'], 'warmup_wall_ms': s['warmup_wall_ms']}
            rows.append({'arm': arm, 'status': 'PASS', 'mode': 'census',
                         'physical_states': s['physical_domain_size'], 'mapped': s['mapped_count'],
                         'unmapped_extras': len(s['unmapped_extras']), 'missing_physical': len(s['missing_physical']),
                         'slots_used': s['slots_used'], 'free_slots': s['free_slots']})
            continue
        rec = read(case / 'run' / 'reconciliation.json') if mode == 'strict' else None
        row = {'arm': arm, 'status': 'PASS', 'mode': mode, 'policy': s.get('policy'),
               'pending': s['pending'], 'H_physical_fa_hit': s['H_physical_fa_hit'],
               'E_executable_prefix': s['E_executable_prefix'], 'G_replay_gap': s['G_replay_gap'],
               'prefill_token_count': s['prefill_token_count'], 'batch_prefix_length': s['batch_prefix_length'],
               'cuda_prefill_ms': s['cuda_prefill_ms'], 'cuda_decode_ms': s['cuda_decode_ms'],
               'cow_copy_count': s['cow_copy_count'], 'cow_copy_cuda_ms': s['cow_copy_cuda_ms'],
               'ttft_ms': s['ttft_ms'], 'request_wall_ms': s['request_wall_ms'],
               'control_rpc_wall_ms': s['control_rpc_wall_ms'], 'warmup_wall_ms': s['warmup_wall_ms'],
               'epoch_wall_ms': s['epoch_wall_ms']}
        # 数据一致性：Prefill Token 必须等于 输入长−E（strict 臂 H=T 时等价于 G+32；cold 臂为全量）
        req_len = len(read(case / 'run' / 'request.json')['pending']['token_ids'])
        assert row['prefill_token_count'] == req_len - row['E_executable_prefix'], \
            'Prefill Token 与 输入长−E 不符：' + arm
        if mode == 'strict':
            row['selected_cpu'] = ';'.join(rec['policy_selected_cpu_ids'])
            row['plan_recovery_cost_ms'] = rec['plan']['recovery_cost_ms']
            cpu_of = {m['physical_checkpoint_id']: m['cpu_checkpoint_id'] for m in rec['census']['mapped']}
            evictions[arm] = {'selected_cpu': rec['policy_selected_cpu_ids'],
                              'evicted_cpu': sorted(cpu_of[i] for i in rec['plan']['logical_eviction_ids']),
                              'unmapped_extras_evicted': [e['physical_checkpoint_id'] for e in rec['census']['unmapped_extras']],
                              'operations_all_returned': all(o['returned'] for o in rec['operations'])}
            stages = {t['stage']: round(t['ms'], 3) for t in rec['timing']}
            overhead.append({'arm': arm, 'policy': s['policy'], 'pending': s['pending'],
                             'control_rpc_wall_ms': s['control_rpc_wall_ms'], **stages})
            matrix.setdefault(s['pending'], {})[s['policy']] = {
                'E': s['E_executable_prefix'], 'G': s['G_replay_gap'], 'H': s['H_physical_fa_hit'],
                'prefill_token_count': s['prefill_token_count'], 'cuda_prefill_ms': s['cuda_prefill_ms'],
                'ttft_ms': s['ttft_ms'], 'request_wall_ms': s['request_wall_ms'],
                'selected_cpu': rec['policy_selected_cpu_ids']}
        else:
            cold_outputs[s['pending']] = read(case / 'run' / 'output.json')
        rows.append(row)
        f = case / 'run' / 'failure.json'
        if f.exists():
            failures[arm] = read(f)['error']
    # Cold 输出一致性（交付4）：同 Pending 每个 strict 臂与 cold 逐Token比较，不允许放宽
    from evaluation.lightweight_runtime_e1.protocol import compare_outputs
    cold_comparison = {}
    for arm_dir in sorted(p for p in OUT.glob('strict_*') if (p / 'run' / 'output.json').exists()):
        # 臂名 = strict_<policy>_<pending>，pending 如 C_B 自带下划线，须 maxsplit=2 解析
        pending = arm_dir.name.split('_', 2)[2]
        if pending in cold_outputs:
            cold_comparison[arm_dir.name] = compare_outputs(read(arm_dir / 'run' / 'output.json'), cold_outputs[pending])
    complete = len([r for r in rows if r.get('status') == 'PASS'])
    conclusion = 'READY' if complete == len(arms) and not failures else ('PARTIAL' if complete > 1 else 'BLOCKED')
    result = {'protocol_version': protocol['version'], 'arms_total': len(arms), 'arms_passed': complete,
              'conclusion': conclusion, 'census_ledger': census_ledger, 'evictions': evictions,
              'pending_policy_matrix': matrix, 'control_overhead': overhead,
              'cold_comparison': cold_comparison, 'failures': failures,
              'pending_tasks': [r['arm'] for r in rows if r.get('status') == 'NOT_RUN']}
    with (OUT / 'finalize_results.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (OUT / 'arm_results.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'pending_policy_matrix.csv').open('x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pending', 'policy', 'E', 'G', 'H', 'prefill_token_count', 'cuda_prefill_ms', 'ttft_ms', 'request_wall_ms', 'selected_cpu'])
        for pending, policies in sorted(matrix.items()):
            for policy, v in sorted(policies.items()):
                w.writerow([pending, policy, v['E'], v['G'], v['H'], v['prefill_token_count'],
                            v['cuda_prefill_ms'], v['ttft_ms'], v['request_wall_ms'], ';'.join(v['selected_cpu'])])
    with (OUT / 'control_overhead.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in overhead for k in r}))
        w.writeheader()
        w.writerows(overhead)
    print('汇总完成：', conclusion, '通过臂', complete, '/', len(arms),
          '矩阵Pending数', len(matrix), 'Cold对照', len(cold_comparison))


if __name__ == '__main__':
    main()
