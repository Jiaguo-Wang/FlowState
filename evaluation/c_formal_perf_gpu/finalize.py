"""离线汇总：62 生命周期 → 三口径分层 CSV + 配对统计 + 口径表 + 结论数据（只读各生命周期产物）。

- lifecycle_results.csv：逐次全指标（口径A 计算事件 / 口径B 请求六段 / 口径C 控制分段）；
- pending_policy_matrix.csv：4 Pending × 3 策略 12 格聚合（均值/中位/最小/最大）；
- pair_analysis.csv：同轮同 Pending 三策略两两差（FS−SA、FS−WFC、SA−WFC）；C_C/C_D 的 FS−SA 为
  相同恢复前沿（E=6144/7168）的经验零假设带宽；
- timing_scopes.md：口径表（含本轮中位数）；finalize_results.json：配对统计与结论要素。
不把 5 次重复解释为充分统计显著性；全部负面结果如实输出。
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
POL_SHORT = {'FlowState-Greedy': 'FS', 'StandaloneTopK': 'SA', 'WFC': 'WFC'}
METRICS = ['ttft_ms', 'compute_cuda_ms', 'execwin_ms', 'queue_ms', 'pregap_ms', 'delivery_ms',
           'request_wall_ms', 'cuda_prefill_ms', 'cuda_decode_ms', 'cow_copy_cuda_ms']


def read(p):
    return json.loads(Path(p).read_text())


def qmed(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def agg(vals):
    if not vals:
        return None
    return {'mean': round(sum(vals) / len(vals), 1), 'median': round(qmed(vals), 1),
            'min': round(min(vals), 1), 'max': round(max(vals), 1), 'n': len(vals)}


def main():
    protocol = read(OUT / 'protocol.json')
    arms = protocol['gate_arms'] + protocol['main_arms']
    rows = []
    by_cell = {}
    for arm in arms:
        sp = OUT / arm['name'] / 'run' / 'summary.json'
        if not sp.exists():
            rows.append({'name': arm['name'], 'status': 'NOT_RUN'})
            continue
        s = read(sp)
        if s.get('status') != 'PASS':
            rows.append({'name': arm['name'], 'status': s.get('status', 'UNKNOWN')})
            continue
        d = s['timing_decomposition']
        tw = {w['window']: w['new_entries'] for w in s['triton_new_kernels']}
        stages = {t['stage']: round(t['ms'], 1) for t in s['control_timing_stages']}
        row = {'name': arm['name'], 'status': 'PASS', 'rep': arm.get('rep') or arm.get('pair'),
               'policy': s['policy'], 'pol': POL_SHORT[s['policy']], 'pending': s['pending'],
               'E': s['E_executable_prefix'], 'G': s['G_replay_gap'], 'H': s['H_physical_fa_hit'],
               'prefill_token_count': s['prefill_token_count'],
               'ttft_ms': round(s['ttft_ms'], 1), 'request_wall_ms': round(s['request_wall_ms'], 1),
               'stream_tpot_ms': round(s['stream_tpot_ms'], 1),
               'queue_ms': round(d['queue_ms'], 1), 'pregap_ms': round(d['pregap_ms'], 1),
               'execwin_ms': round(d['execwin_ms'], 1), 'compute_cuda_ms': round(d['compute_cuda_ms'], 1),
               'postgap_ms': round(d['postgap_ms'], 1), 'delivery_ms': round(d['delivery_ms'], 1),
               'first_decode_cuda_ms': round(d['first_decode_cuda_ms'], 1) if d['first_decode_cuda_ms'] else None,
               'cuda_prefill_ms': round(s['cuda_prefill_ms'], 1), 'cuda_decode_ms': round(s['cuda_decode_ms'], 1),
               'cow_copy_count': s['cow_copy_count'], 'cow_copy_cuda_ms': round(s['cow_copy_cuda_ms'], 2),
               'cow_copy_host_ms': round(s['cow_copy_host_ms'], 2), 'match_native_ms': round(s['match_native_ms'], 2),
               'queue_pregap_ms': round(d['queue_ms'] + d['pregap_ms'], 1),
               'control_rpc_wall_ms': round(s['control_rpc_wall_ms'], 1),
               'ctrl_auth_plan_ms': stages.get('strict_epoch_authentication_and_plan'),
               'ctrl_reconcile_ms': stages.get('strict_reconciliation_with_postchecks'),
               'ctrl_heavy_obs_ms': stages.get('final_heavy_observation'),
               'ctrl_fa_audit_ms': round((stages.get('fa_content_audit_before', 0) +
                                          stages.get('fa_content_postcheck_and_resume_baseline', 0)), 1),
               'ctrl_preflight_ms': stages.get('apply_time_freshness_preflight'),
               'shape_warm_b_ms': round(s['shape_warm_b']['wall_ms'], 1),
               'triton_new_shape_warm': tw.get('post_control→post_shape_warm'),
               'triton_new_request': tw.get('post_shape_warm→post_request'),
               'audit_entries_total': s['audit_entries_total'],
               'engine_boot_ms': round(s['engine_boot_ms'], 1), 'warmup1_ms': round(s['warmup1_wall_ms'], 1),
               'warmup2_ms': round(s['warmup2_wall_ms'], 1), 'epoch_wall_ms': round(s['epoch_wall_ms'], 1),
               'cold_output_match': s['cold_output_match']}
        rows.append(row)
        by_cell.setdefault((s['pending'], POL_SHORT[s['policy']]), []).append(row)

    # 派生汇总产物：可重复生成（'w'），与测量产物的 write-once 纪律区分
    with (OUT / 'lifecycle_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    # 12 格聚合
    matrix = []
    for (pending, pol), rs in sorted(by_cell.items()):
        rec = {'pending': pending, 'pol': pol, 'n': len(rs),
               'E': rs[0]['E'], 'G': rs[0]['G'], 'prefill_tokens': rs[0]['prefill_token_count']}
        for k in ['ttft_ms', 'compute_cuda_ms', 'queue_ms', 'pregap_ms', 'queue_pregap_ms',
                  'request_wall_ms', 'cuda_prefill_ms', 'control_rpc_wall_ms']:
            a = agg([r[k] for r in rs])
            rec[k] = a['median']
            rec[k + '_min'] = a['min']
            rec[k + '_max'] = a['max']
        matrix.append(rec)
    if matrix:
        with (OUT / 'pending_policy_matrix.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
            w.writeheader()
            w.writerows(matrix)

    # 两两配对差（同轮同 Pending，仅主矩阵 m* 生命周期；门禁臂不进配对）
    pairs = []
    deltas = {}
    by_pr = {}
    for r in rows:
        if r.get('status') == 'PASS' and str(r.get('name', '')).startswith('m'):
            by_pr.setdefault((r['pending'], r['rep']), {})[r['pol']] = r
    for (pending, rep), pols in sorted(by_pr.items()):
        if len(pols) != 3:
            continue
        for a, b in [('FS', 'SA'), ('FS', 'WFC'), ('SA', 'WFC')]:
            rec = {'pending': pending, 'rep': rep, 'pair': f'{a}-{b}'}
            for k in METRICS:
                rec['d_' + k] = round(pols[a][k] - pols[b][k], 1)
                deltas.setdefault((pending, f'{a}-{b}'), {}).setdefault(k, []).append(pols[a][k] - pols[b][k])
            pairs.append(rec)
    if pairs:
        with (OUT / 'pair_analysis.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)

    # 配对统计（逐 Pending×pair 逐指标：中位/最小/最大/n）
    stats = {}
    for (pending, pair), ks in sorted(deltas.items()):
        stats[f'{pending}|{pair}'] = {k: agg(v) for k, v in ks.items()}

    passed = [r for r in rows if r.get('status') == 'PASS']
    result = {'protocol_version': protocol['version'],
              'lifecycles_total': len(arms), 'lifecycles_passed': len(passed),
              'pair_delta_stats': stats,
              'null_bands_same_frontier': {'C_C|FS-SA': stats.get('C_C|FS-SA', {}).get('ttft_ms'),
                                           'C_D|FS-SA': stats.get('C_D|FS-SA', {}).get('ttft_ms')},
              'conclusion': 'READY' if len(passed) == len(arms) else 'PARTIAL'}
    with (OUT / 'finalize_results.json').open('w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 口径表
    med = lambda k: qmed([r[k] for r in passed]) if any(k in r for r in passed) else None
    scopes = [
        ('口径A cuda_prefill_ms（compute_cuda_ms 同值）', '首 extend 批 CUDA 事件窗（含 host 提交空隙；形状预热后无编译/autotune）'),
        ('口径A cow_copy_cuda_ms / cow_copy_host_ms', 'COW 设备拷贝事件 / 原生宿主调用（B 类保留项）'),
        ('口径B ttft_ms', 'client 发送 → 首个含 Token 流式 chunk；测量模式：窗口内零 A 类审计零落盘'),
        ('口径B queue/pregap/execwin/postgap/delivery', '六段分解（闭合断言 5ms）；pregap 对 E>0 含原生 mamba 绑定（B 类）'),
        ('口径B request_wall_ms / stream_tpot_ms', '整请求墙钟与逐 Token 间隔（含 8 Token 解码）'),
        ('口径C control_rpc_wall_ms', 'b5d4_apply RPC 墙钟（审计增强研究仪表：重型观察+FA 内容审计+apply 基线快照）'),
        ('口径C ctrl_auth_plan_ms / ctrl_reconcile_ms', '生产控制核心近似（认证+计划 / 真实驱逐+合同后检），单列供净收益核算'),
    ]
    with (OUT / 'timing_scopes.md').open('w') as f:
        f.write('# 计时口径表（三口径分层）\n\n| 指标 | 覆盖事件 | 本轮中位数(ms) |\n|---|---|---|\n')
        keys = ['cuda_prefill_ms', None, 'ttft_ms', None, 'request_wall_ms', 'control_rpc_wall_ms', None]
        for (title, scope), k in zip(scopes, keys):
            v = med(k) if k else None
            f.write(f'| {title} | {scope} | {round(v, 1) if isinstance(v, float) else "—"} |\n')

    print('汇总完成：', result['conclusion'], '生命周期', len(passed), '/', len(arms))
    for p in ['C_A', 'C_B', 'C_C', 'C_D']:
        for pair in ['FS-SA', 'FS-WFC', 'SA-WFC']:
            t = stats.get(f'{p}|{pair}', {}).get('ttft_ms')
            c = stats.get(f'{p}|{pair}', {}).get('compute_cuda_ms')
            if t and c:
                print(f"  {p} {pair}: Δttft 中位 {t['median']} [{t['min']},{t['max']}]"
                      f"  Δcompute 中位 {c['median']} [{c['min']},{c['max']}]")


if __name__ == '__main__':
    main()
