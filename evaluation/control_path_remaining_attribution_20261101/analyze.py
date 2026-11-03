"""只读复算最新三策略五轮，按真实区间构建可加总分解与来源清单。"""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'evaluation/control_path_separation_20261101'
OUT = Path(__file__).resolve().parent
POLICIES = {'fs': 'FlowState-Greedy', 'mc': 'Marconi-Evict Adaptation', 'kv': 'KVFlow-inspired'}


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    manifest = {}
    def read(path):
        raw = path.read_bytes()
        manifest[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    frozen = {str(Path(k).resolve()): v for k,v in read(SRC/'final_source_manifest.json')['files'].items()}
    source_csv = list(csv.DictReader((SRC/'performance_results.csv').open()))
    subset = [r for r in source_csv if r['phase'] == 'three_policy' and r['mode'] == 'serving']
    assert len(subset) == 15 and len({r['arm'] for r in subset}) == 15
    records = []
    for rnd in range(1, 6):
        for tag, policy in POLICIES.items():
            arm = f't{rnd}_{tag}'
            base = SRC/'arms'/arm/'run'
            s = read(base/'summary.json')
            r = read(base/'reconciliation.json')
            post = read(base/'post_request.json')
            assert s['status'] == 'PASS' and s['oracle_match'] and s['cycle']['identity_exact']
            assert hashlib.sha256((base/'summary.json').read_bytes()).hexdigest() == next(
                x['source_hashes'][str(base/'summary.json')] for x in read(SRC/'verified_evidence_inventory.json')['records'] if x['spec']['arm']==arm)
            c, a, j = s['cycle'], s['separation'], s['journal_io']
            t = {x['stage']: x['ms'] for x in r['timing']}
            assert len(t) == 8
            row = {'arm': arm, 'round': rnd, 'tag': tag, 'policy': policy,
                   'control_ms': c['control_cycle_ms'], 'full_ms': c['cycle_ms_full_incl_warm'],
                   'audit_ms': a['audit_inclusive_ms'], 'post_ms': a['post_t4_audit_ms'],
                   'rpc_tail_ms': c['rpc_tail_ms'], 'control_rpc_wall_ms': c['control_rpc_wall_ms'],
                   'warm_ms': sum(x['wall_ms'] for x in c['shape_warm_block_ms']),
                   'requests_ms': sum((x['end_ns']-x['start_ns'])/1e6 for x in s['pending_rows']),
                   'window_gaps_ms': sum(c['window_gaps_ms']),
                   'request_execution_ms': c['request_exec_total_ms'],
                   'sum_ttft_ms': sum(x['ttft_ms'] for x in s['pending_rows']),
                   'sum_cuda_ms': sum(x['cuda_prefill_ms'] for x in s['pending_rows']),
                   'observation_ms': t['final_heavy_observation'],
                   'mapping_ms': t['cpu_physical_mapping'],
                   'epoch_ms': t['strict_epoch_authentication'],
                   'selection_stage_ms': t['policy_online_select'],
                   'selection_call_ms': s['selection']['selection_call_ms'],
                   'freshness_ms': t['apply_time_freshness_preflight'],
                   'fa_before_ms': t['fa_content_audit_before'],
                   'reconciliation_ms': t['strict_reconciliation_with_postchecks'],
                   'fa_post_ms': t['fa_content_postcheck_and_resume_baseline'],
                   'journal_in_ms': j['in_window_serialize_ms'],
                   'journal_after_ms': j['after_window_serialize_ms'],
                   'journal_flush_ms': j['after_window_flush_ms'],
                   'report_ms': a['report_timing']['wall_ms'],
                   'report_cpu_ms': a['report_timing']['cpu_ms'], 'oracle_ms': s['oracle_check_ms']}
            row['adapter_calls_ms'] = sum((o['end_ns']-o['start_ns'])/1e6 for o in r['operations'])
            row['reconciliation_rest_ms'] = row['reconciliation_ms']-row['adapter_calls_ms']
            row['control_unknown_ms'] = row['control_ms']-sum(t.values())-row['journal_in_ms']
            baseline = r['post_apply_baseline_ns']
            row['baseline_ms'] = (baseline['end_ns']-baseline['start_ns'])/1e6
            rpc_end = c['t3_ns'] + round(c['rpc_tail_ms']*1e6)
            transfer = a['transfers'][0]
            encode = transfer['encoding']
            assert c['t3_ns'] <= baseline['start_ns'] <= baseline['end_ns'] <= encode['start_ns']
            assert encode['end_ns'] <= transfer['start_ns'] <= transfer['end_ns'] <= rpc_end
            row['encode_ms'] = encode['wall_ms']
            row['encode_cpu_ms'] = encode['cpu_ms']
            row['encode_json_ms'] = encode['json_ms']
            row['compress_ms'] = encode['compression_ms']
            row['decode_ms'] = transfer['wall_ms']
            row['decode_cpu_ms'] = transfer['cpu_ms']
            row['rpc_unknown_ms'] = row['rpc_tail_ms']-sum(row[k] for k in ['journal_after_ms','journal_flush_ms','baseline_ms','encode_ms','decode_ms'])
            audits = post['trace']['audit']
            assert len(audits) == 2 and all(x['kind']=='cache_state_full_hash' for x in audits)
            row['post_state_audit_ms'] = audits[1]['ns']/1e6
            assert c['t4_ns'] <= a['report_timing']['start_ns'] <= a['report_timing']['end_ns'] <= a['t5_ns']
            row['post_unknown_ms'] = row['post_ms']-row['post_state_audit_ms']-row['report_ms']-row['oracle_ms']
            row['protected_indices'] = len(r['fa_protected_indices'])
            row['protected_bytes'] = sum(x['bytes'] for x in r['fa_protected_payload'])
            row['post_fa_entries'] = len(post['state']['fa_map'])
            row['post_node_count'] = len(post['state']['nodes'])
            row['journal_files_bytes'] = sum(f.stat().st_size for f in (base/'control_evidence').glob('*.json'))
            row['post_json_bytes'] = (base/'post_request.json').stat().st_size
            row['report_json_bytes'] = (base/'reconciliation.json').stat().st_size
            row['E'] = '|'.join(str(x['E']) for x in s['pending_rows'])
            row['G'] = '|'.join(str(x['G']) for x in s['pending_rows'])
            row['source_summary'] = str(base/'summary.json')
            row['source_control'] = str(base/'reconciliation.json')
            row['source_post'] = str(base/'post_request.json')
            assert all(row[k] >= -0.0001 for k in ['control_unknown_ms','rpc_unknown_ms','post_unknown_ms'])
            parts = ['control_ms','rpc_tail_ms','warm_ms','requests_ms','window_gaps_ms','post_ms']
            assert abs(row['audit_ms']-sum(row[k] for k in parts)) < 0.0001
            assert abs(row['full_ms']-sum(row[k] for k in parts[:-1])) < 0.0001
            src = next(v for v in subset if v['arm'] == arm)
            for k,n in [('control_ms','control_cycle_ms'),('full_ms','full_cycle_ms'),('audit_ms','audit_inclusive_ms')]:
                assert abs(row[k]-float(src[n])) < 1e-8
            records.append(row)
    write_csv('per_arm_attribution.csv', records)
    numeric = [k for k,v in records[0].items() if type(v) in (int,float) and k != 'round']
    pairs = []
    for rnd in range(1,6):
        fs = next(r for r in records if r['round']==rnd and r['tag']=='fs')
        for tag in ['mc','kv']:
            base = next(r for r in records if r['round']==rnd and r['tag']==tag)
            pairs.append({'round':rnd,'comparison':'FS-'+tag.upper(),'left':fs['arm'],'right':base['arm'],
                          **{k:fs[k]-base[k] for k in numeric}})
    write_csv('paired_attribution.csv', pairs)
    summaries = []
    for comparison in ['FS-MC','FS-KV']:
        group = [r for r in pairs if r['comparison']==comparison]
        for k in numeric:
            values = [r[k] for r in group]
            summaries.append({'comparison':comparison,'metric':k,'mean':statistics.mean(values),
                              'median':statistics.median(values),'min':min(values),'max':max(values),
                              'wins':sum(v<0 for v in values),'ties':sum(v==0 for v in values),'losses':sum(v>0 for v in values)})
    write_csv('paired_summary.csv', summaries)
    for path in [SRC/'performance_results.csv',SRC/'paired_comparison.csv',SRC/'final_report.md',SRC/'performance_analysis.md',SRC/'measurement_boundaries.md',SRC/'runner.py',SRC/'control.py',SRC/'bridge.py',SRC/'codec.py']:
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in frozen:
            assert manifest[str(path)] == frozen[str(path)]
    inventory = read(SRC/'verified_evidence_inventory.json')
    used_arms = {r['arm'] for r in subset}
    for entry in inventory['records']:
        if entry['spec']['arm'] not in used_arms:
            continue
        for name, expected in entry['source_hashes'].items():
            path=Path(name);actual=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual==expected, name
            manifest[name]=actual
    with (OUT/'source_manifest.json').open('w') as f:json.dump({'input_scope':'仅three_policy/serving的15臂，5轮；无旧批量/Legacy数据','files':manifest,'raw_summary_csv_reconciled':True,'per_arm_additive_identity_pass':True},f,ensure_ascii=False,indent=2)
    for comp in ['FS-MC','FS-KV']:
        print('\n',comp)
        for row in summaries:
            if row['comparison']==comp and row['metric'] in ['control_ms','full_ms','audit_ms','post_ms','baseline_ms','post_state_audit_ms','fa_post_ms','control_unknown_ms','rpc_unknown_ms','post_unknown_ms','warm_ms','requests_ms','report_ms','encode_ms','decode_ms']:
                print(row['metric'],'mean',round(row['mean'],3),'median',round(row['median'],3),'range',round(row['min'],3),round(row['max'],3))
    print('工作量',[(tag,[(r['protected_indices'],r['protected_bytes']/1024**2,r['post_fa_entries']) for r in records if r['tag']==tag]) for tag in POLICIES])


if __name__ == '__main__':
    main()
