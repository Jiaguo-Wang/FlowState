"""读取本轮原始记录，分开报告原时钟与审计完成时钟；逐轮配对不拼接历史。"""
import csv
import json
from pathlib import Path
import statistics

PKG = Path(__file__).resolve().parent
METRICS = ['control_cycle_ms', 'full_cycle_ms', 'audit_inclusive_ms', 'request_execution_ms',
           'sum_ttft_ms', 'sum_cuda_prefill_ms', 'control_rpc_wall_ms', 'rpc_tail_ms',
           'observation_ms', 'selection_call_ms', 'selection_stage_ms', 'freshness_ms', 'reconciliation_ms']


def csvwrite(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (PKG/name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'values': values}


def main():
    lock = json.loads((PKG/'execution_lock_gpu1.json').read_text())
    rows, pending, attempts = [], [], []
    byarm = {}
    for spec in lock['schedule']:
        arm = PKG/'arms'/spec['arm']
        if not arm.exists():
            continue
        path = arm/'run/summary.json'
        attempt = {**spec, 'summary_exists': path.exists(), 'source': str(arm)}
        for name in ['failure.json', 'failure_after_summary.json']:
            if (arm/'run'/name).exists():
                attempt['failure'] = (arm/'run'/name).read_text()
        attempts.append(attempt)
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        r = json.loads((arm/'run/reconciliation.json').read_text())
        c, a = s['cycle'], s['separation']
        stages = {x['stage']: x['ms'] for x in r['timing']}
        row = {**spec, 'status': s['status'], 'source': str(path),
               't0_ns': c['t0_ns'], 't3_ns': c['t3_ns'], 't4_ns': c['t4_ns'], 't5_ns': a['t5_ns'],
               'control_cycle_ms': c['control_cycle_ms'], 'full_cycle_ms': c['cycle_ms_full_incl_warm'],
               'audit_inclusive_ms': a['audit_inclusive_ms'], 'post_t4_audit_ms': a['post_t4_audit_ms'],
               'request_execution_ms': c['request_exec_total_ms'],
               'sum_ttft_ms': sum(x['ttft_ms'] for x in s['pending_rows']),
               'sum_cuda_prefill_ms': sum(x['cuda_prefill_ms'] for x in s['pending_rows']),
               'control_rpc_wall_ms': c['control_rpc_wall_ms'], 'rpc_tail_ms': c['rpc_tail_ms'],
               'observation_ms': stages['final_heavy_observation'],
               'selection_call_ms': s['selection']['selection_call_ms'],
               'selection_stage_ms': stages['policy_online_select'],
               'freshness_ms': stages['apply_time_freshness_preflight'],
               'reconciliation_ms': stages['strict_reconciliation_with_postchecks'],
               'epoch_authentication_ms': stages['strict_epoch_authentication'],
               'fa_before_ms': stages['fa_content_audit_before'],
               'fa_post_ms': stages['fa_content_postcheck_and_resume_baseline'],
               'J_ms': s['selection']['plan_recovery_cost_ms'],
               'selected_cpu': '|'.join(s['selection']['selected_cpu']),
               'E': '|'.join(str(x['E']) for x in s['pending_rows']),
               'G': '|'.join(str(x['G']) for x in s['pending_rows']),
               'prefill_tokens': '|'.join(str(x['prefill_token_count']) for x in s['pending_rows']),
               'shape_warm_ms': sum(w['wall_ms'] for w in c['shape_warm_block_ms']),
               'stage_gaps_ms': sum(c['apply_stage_gaps_ms']),
               'window_gaps_ms': sum(c['window_gaps_ms']),
               'journal_serialize_ms': s['journal_io']['in_window_serialize_ms'],
               'journal_after_serialize_ms': s['journal_io']['after_window_serialize_ms'],
               'journal_flush_ms': s['journal_io']['after_window_flush_ms'],
               'report_wall_ms': a['report_timing']['wall_ms'],
               'report_cpu_ms': a['report_timing']['cpu_ms'],
               'report_start_ns': a['report_timing']['start_ns'],
               'report_end_ns': a['report_timing']['end_ns']}
        transfers = a['transfers']
        if transfers:
            t = transfers[0]
            row.update(wire_raw_bytes=t['raw_bytes'], wire_compressed_bytes=t['compressed_bytes'],
                       encode_wall_ms=t['encoding']['wall_ms'], encode_cpu_ms=t['encoding']['cpu_ms'],
                       compression_ms=t['encoding']['compression_ms'],
                       decode_wall_ms=t['wall_ms'], decode_cpu_ms=t['cpu_ms'], digest_verified=t['digest_verified'])
        else:
            row.update(encode_wall_ms='UNKNOWN', encode_cpu_ms='UNKNOWN', decode_wall_ms='UNKNOWN',
                       decode_cpu_ms='UNKNOWN', digest_verified='原线协议不封装')
        for rpc in (arm/'run').glob('rpc_*'):
            req = rpc/'request.json'
            if req.exists() and json.loads(req.read_text()).get('action') == 'b5d4_apply':
                row['wire_response_raw_bytes'] = (rpc/'response.raw').stat().st_size
                row['durable_response_json_bytes'] = (rpc/'response.json').stat().st_size
                row['raw_response_path'] = str(rpc/'response.raw')
        rows.append(row)
        byarm[spec['arm']] = row
        for x in s['pending_rows']:
            pending.append({**{k: spec[k] for k in ['arm', 'phase', 'round', 'mode', 'policy']}, **x})
    csvwrite('performance_results.csv', rows)
    csvwrite('pending_results.csv', pending)
    csvwrite('attempts.csv', attempts)
    pairs = []
    for r in range(1, 6):
        comparisons = [('serving-original', f'p{r}_serving_fs', f'p{r}_original_fs'),
                       ('serving_FS-MC', f't{r}_fs', f't{r}_mc'),
                       ('serving_FS-KV', f't{r}_fs', f't{r}_kv')]
        for label, a, b in comparisons:
            if a in byarm and b in byarm:
                pairs.append({'comparison': label, 'round': r, 'left': a, 'right': b,
                              **{m: byarm[a][m]-byarm[b][m] for m in METRICS}})
    csvwrite('paired_comparison.csv', pairs)
    stats = {'groups': {}, 'paired': {}, 'n_completed': len(rows), 'n_attempted': len(attempts)}
    for phase, mode, policy in dict.fromkeys((r['phase'], r['mode'], r['policy']) for r in rows if r['phase'] != 'smoke'):
        key = '/'.join([phase, mode, policy])
        group = [r for r in rows if (r['phase'], r['mode'], r['policy']) == (phase, mode, policy)]
        stats['groups'][key] = {m: summarize([r[m] for r in group]) for m in METRICS}
    for label in dict.fromkeys(p['comparison'] for p in pairs):
        group = [p for p in pairs if p['comparison'] == label]
        stats['paired'][label] = {}
        for metric in METRICS:
            values = [p[metric] for p in group]
            stats['paired'][label][metric] = {**summarize(values), 'wins': sum(v < 0 for v in values),
                                             'ties': sum(v == 0 for v in values), 'losses': sum(v > 0 for v in values)}
    with (PKG/'statistics.json').open('x') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {m: v[m] for m in ['control_cycle_ms', 'full_cycle_ms', 'audit_inclusive_ms']}
                      for k, v in stats['paired'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
