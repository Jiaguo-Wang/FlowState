"""只读取新批次原始结果，按轮次配对，保留全部方向和计时口径。"""
import csv
import json
from pathlib import Path
import statistics

PKG = Path(__file__).resolve().parent


def main():
    lock = json.loads((PKG/'execution_lock.json').read_text())
    rows = []
    lookup = {}
    pending = []
    for spec in lock['schedule']:
        base = PKG/'arms'/spec['arm']/'run'
        if not (base/'summary.json').exists():
            continue
        s = json.loads((base/'summary.json').read_text())
        rec = json.loads((base/'reconciliation.json').read_text())
        row = {**spec, 'status': s['status'], **{k: s['cycle'][k] for k in
               ['control_cycle_ms', 'cycle_ms_full_incl_warm', 'request_exec_total_ms', 'control_rpc_wall_ms', 'rpc_tail_ms']},
               'sum_ttft_ms': sum(p['ttft_ms'] for p in s['pending_rows']),
               'sum_cuda_prefill_ms': sum(p['cuda_prefill_ms'] for p in s['pending_rows']),
               'J_ms': s['selection']['plan_recovery_cost_ms'],
               'selected': '|'.join(s['selection']['selected_cpu']),
               'source': str(base/'summary.json')}
        for stage in rec['timing']:
            row[stage['stage']] = stage['ms']
        row['journal_serialize_ms'] = (s.get('journal_io') or {}).get('in_window_serialize_ms', '')
        rows.append(row)
        lookup[spec['arm']] = row
        for p in s['pending_rows']:
            pending.append({'arm': spec['arm'], 'phase': spec['phase'], 'policy': spec['policy'],
                            'mode': spec['mode'], 'round': spec['round'], **p})
    def csvwrite(name, records):
        if not records:
            return
        keys = list(dict.fromkeys(k for row in records for k in row))
        with (PKG/name).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)
    csvwrite('performance_results.csv', rows)
    csvwrite('pending_results.csv', pending)
    metrics = ['control_cycle_ms', 'cycle_ms_full_incl_warm', 'sum_ttft_ms', 'sum_cuda_prefill_ms',
               'control_rpc_wall_ms', 'rpc_tail_ms', 'request_exec_total_ms']
    pairs = []
    for r in range(1, 6):
        for kind, a, b in [('optimized-original', f'p{r}_batched_fs', f'p{r}_original_fs'),
                           ('FS-MC', f't{r}_fs', f't{r}_mc'), ('FS-KV', f't{r}_fs', f't{r}_kv')]:
            if a not in lookup or b not in lookup:
                continue
            pairs.append({'comparison': kind, 'round': r, 'left': a, 'right': b,
                          **{m: lookup[a][m]-lookup[b][m] for m in metrics}})
    csvwrite('paired_comparison.csv', pairs)
    summary = {}
    for comparison in dict.fromkeys(p['comparison'] for p in pairs):
        matched = [p for p in pairs if p['comparison'] == comparison]
        summary[comparison] = {'pairs': len(matched)}
        for metric in metrics:
            values = [p[metric] for p in matched]
            summary[comparison][metric] = {'median': statistics.median(values), 'min': min(values), 'max': max(values),
                                           'wins': sum(v < 0 for v in values), 'ties': sum(v == 0 for v in values),
                                           'losses': sum(v > 0 for v in values), 'values': values}
    (PKG/'performance_statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
