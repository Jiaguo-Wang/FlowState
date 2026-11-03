"""只读两臂真实证据，整理正确性、原始计时及不能独立拆分的区间。"""
import csv
import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent
ARMS = ['A_shadow_validation','B_serving']


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def csv_write(path, rows):
    with path.open('x',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def logical(r):
    mapping = {x['physical_checkpoint_id']:x['cpu_checkpoint_id'] for x in r['census']['mapped']}
    value = {'configuration_digest':r['epoch']['configuration_digest'], 'K':r['epoch']['K'],
             'candidates':sorted((mapping[x['candidate']['checkpoint_id']],x['candidate']['token_pos'],x['token_digest'])
                                 for x in r['epoch']['candidates']),
             'pendings':sorted((x['pending_id'],x['target'],x['token_digest']) for x in r['epoch']['pendings']),
             'edges':sorted((mapping[x['candidate_id']],x['pending_id'],x['status'],x['conditions']) for x in r['epoch']['edges'])}
    return value


def main():
    assert load(PKG/'execution_result.json')['status']=='GPU_CORRECTNESS_PASS'
    details, stages, pending, bytes_rows, records = [], [], [], [], {}
    evidence = {}
    for arm in ARMS:
        directory = PKG/'arms'/arm
        base = directory/'run'
        r,s,p = [load(base/f) for f in ('reconciliation.json','summary.json','post_request.json')]
        records[arm] = (r,s,p)
        receipt = load(directory/'independent_validation.json')
        assert receipt['status']=='PASS' and s['status']=='PASS'
        assert r['before']['attention_digest']==r['after']['attention_digest']
        assert r['before']['non_recurrent_digest']==r['after']['non_recurrent_digest']
        assert r['selected_payload_before']==r['selected_payload_after']==p['state']['selected_payload']
        selected = set(r['plan']['selected_ids'])
        by_token = {tuple(c['token_ids']):c['candidate']['checkpoint_id'] for c in r['epoch']['candidates']}
        assert {by_token[tuple(n['token_ids'])] for n in r['after']['nodes'] if n['slots']}==selected
        assert {o['checkpoint_id'] for o in r['operations']}==set(r['plan']['logical_eviction_ids'])
        assert len(r['reconciliation']['completed_evictions'])==13
        assert all(o['returned'] for o in r['operations'])
        metric = r['admission_interface_metrics']
        transfer = s['separation']['transfers'][0]
        rpc = base/Path(transfer['call_saved_interval']['directory']).name
        packet = load(rpc/'response.json')
        br = {'arm':arm,'mode':metric['mode'], 'pre_codec_json_bytes':packet['raw_bytes'],
              'compressed_bytes':packet['encoding']['compressed_bytes'],
              'payload_base64_bytes':len(packet['payload_b64'].encode()),
              'wire_response_raw_bytes':(rpc/'response.raw').stat().st_size,
              'saved_response_json_bytes':(rpc/'response.json').stat().st_size,
              'returned_baseline_json_bytes':len(json.dumps(r['admission_baseline'],ensure_ascii=False,separators=(',',':')).encode()),
              'sidecar_json_bytes':len(json.dumps(r['admission_baseline_serving'],ensure_ascii=False,separators=(',',':')).encode()) if 'admission_baseline_serving' in r else 0}
        bytes_rows.append(br)
        row = {'arm':arm,'mode':metric['mode'],
               'baseline_full_builder_ms':metric['full_baseline']['wall_ms'],
               'projection_ms':metric['projection']['wall_ms'],
               'server_validation_ms':metric['server_validation']['wall_ms'],
               'client_validation_ms':transfer['client_validation_ms'],
               'baseline_parent_ms':(r['post_apply_baseline_ns']['end_ns']-r['post_apply_baseline_ns']['start_ns'])/1e6,
               'response_encoding_ms':transfer['encoding']['wall_ms'],
               'json_encode_ms':transfer['encoding']['json_ms'],
               'compression_ms':transfer['encoding']['compression_ms'],
               'client_decode_ms':transfer['wall_ms'],
               'call_saved_parent_ms':transfer['call_saved_interval']['wall_ms'],
               'encode_end_to_saved_end_ms':(transfer['call_saved_interval']['end_ns']-transfer['encoding']['end_ns'])/1e6,
               'pure_network_ms':'UNKNOWN',
               'control_T0_T3_ms':s['cycle']['control_cycle_ms'],
               'control_rpc_wall_ms':s['cycle']['control_rpc_wall_ms'],
               'rpc_tail_ms':s['cycle']['rpc_tail_ms'],
               'full_cycle_T0_T4_ms':s['cycle']['cycle_ms_full_incl_warm'],
               'audit_inclusive_T0_T5_ms':s['separation']['audit_inclusive_ms'],
               'audit_T4_T5_ms':s['separation']['post_t4_audit_ms']}
        stages.append(row)
        assert metric['full_baseline']['end_ns']==metric['projection']['start_ns']
        assert metric['projection']['end_ns']==metric['server_validation']['start_ns']
        assert transfer['encoding']['end_ns']<=transfer['call_saved_interval']['end_ns']
        binding = r['reconciliation']['binding']
        detail = {'arm':arm,'status':'PASS','mode':metric['mode'],
                  'epoch_id':r['epoch']['epoch_id'],'epoch_digest':r['epoch']['content_digest'],
                  'engine_id':r['after']['engine_id'],'barrier_id':r['after']['barrier_id'],
                  'binding_identity':{k:binding[k] for k in ('engine_id','barrier_id','lease_id','epoch_digest','plan_digest')},
                  'selected_cpu':s['selection']['selected_cpu'],'selected_physical':r['plan']['selected_ids'],
                  'J_model_ms':r['plan']['recovery_cost_ms'],
                  'candidates':len(r['epoch']['candidates']),'pendings':len(r['epoch']['pendings']),
                  'edge_counts':{k:sum(x['status']==k for x in r['epoch']['edges']) for k in ('EPOCH_ADMISSIBLE','REJECTED','UNKNOWN')},
                  'selected_binding':[x for x in binding['candidates'] if x['candidate_id'] in selected],
                  'completed_evictions':len(r['reconciliation']['completed_evictions']),
                  'free_before':len(r['before']['free_slots']),'free_after':len(r['after']['free_slots']),
                  'resident_before':sum(bool(x['slots']) for x in r['before']['nodes']),
                  'resident_after':sum(bool(x['slots']) for x in r['after']['nodes']),
                  'selected_content_before_after_post_equal':True,'fa_structural_reconcile_equal':True,
                  'fa_content_guard_enforced_by_original_control':True,
                  'protected_fa_post_equal':p['state']['fa_protected_payload']==r['fa_protected_payload'],
                  'two_real_full_audits':len(p['trace']['audit'])==2,'provenance':'PROVENANCE_UNVERIFIED',
                  'pending_evidence':[]}
        for pr in s['pending_rows']:
            pid = pr['pending_id']
            matches = [x for x in p['trace']['matches'] if x['rid']==pid]
            assert len(matches)==1
            native = matches[0]['native']
            assert (native['executable_prefix'],native['replay_gap'])==(pr['E'],pr['G'])
            cold = load(directory/f'cold_reference_{pid}.json')
            assert pr['output_ids']==cold['output_ids'] and len(pr['output_ids'])==8
            copies = [x for x in p['trace']['copies'] if set(x['destination_slots']) & set(native['working_slots'])]
            if pr['E']>0:
                assert native['cow_source_slots'] and native['working_slots'] and copies
                assert all(x['returned'] and not set(x['source_slots']) & set(x['destination_slots']) for x in copies)
            pending.append({'arm':arm,'mode':metric['mode'],'pending_id':pid,'H':pr['physical_fa_hit'],
                            'T':pr['T'],'E':pr['E'],'G':pr['G'],'prefill_tokens':pr['prefill_token_count'],
                            'cuda_prefill_ms':pr['cuda_prefill_ms'],'ttft_ms':pr['ttft_ms'],
                            'request_wall_ms':pr['request_wall_ms'],'cow_completed_count':len(copies),
                            'cold_exact':True,'output_ids':json.dumps(pr['output_ids'])})
            detail['pending_evidence'].append({'pending_id':pid,'source_checkpoint_expected':r['targets'][pid]['source_checkpoint'],
                'native':native,'cow_completed':copies,'cold_output_ids':cold['output_ids'],'actual_output_ids':pr['output_ids']})
        details.append(detail)
        for path in [base/x for x in ('summary.json','reconciliation.json','post_request.json','engine_cleanup.json')]+[
            directory/'runtime.log',directory/'gpu_cleanup.json',directory/'preflight.json',directory/'command.json',
            directory/'independent_validation.json',rpc/'response.raw',rpc/'response.json']:
            evidence[str(path.relative_to(PKG))] = sha(path)
    a,b = records[ARMS[0]], records[ARMS[1]]
    assert logical(a[0])==logical(b[0]), '逻辑候选、Token、T或严格边不同'
    assert a[1]['selection']['selected_cpu']==b[1]['selection']['selected_cpu']
    assert a[0]['plan']['recovery_cost_ms']==b[0]['plan']['recovery_cost_ms']
    # 容器内PID可以重复；独立性由宿主执行区间、容器命令和清理证据共同认证。
    ad, bd = [PKG/'arms'/x for x in ARMS]
    ah, bh = [load(d/'host_exit.json') for d in (ad,bd)]
    assert ah['end_ns'] < bh['start_ns'] and ah['returncode']==bh['returncode']==0
    ac, bc = [load(d/'command.json') for d in (ad,bd)]
    assert ac[ac.index('--name')+1] != bc[bc.index('--name')+1]
    assert a[0]['after']['barrier_id'] != b[0]['after']['barrier_id']
    for d in (ad,bd):
        initial = load(d/'run/initial.json')
        assert not any(n['slots'] for n in initial['before']['nodes'])
        cleanup = load(d/'gpu_cleanup.json')
        assert 'flowstate-serving-gate-' not in cleanup['containers']
        assert not any(line.split(',')[0].strip()=='GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'
                       for line in cleanup['compute_processes'].splitlines())
    cross = {'logical_epoch_equal':True,'selected_cpu_equal':True,'J_equal':True,
             'E_G_equal':[x['E'] for x in a[1]['pending_rows']]==[x['E'] for x in b[1]['pending_rows']] and
                         [x['G'] for x in a[1]['pending_rows']]==[x['G'] for x in b[1]['pending_rows']],
             'independent_host_lifecycles':True,'container_local_engine_id_repeated':a[0]['after']['engine_id']==b[0]['after']['engine_id'],'independent_barriers':True,'physical_pointer_comparison':'不适用，分别验证当前绑定',
             'performance_claim':'不作两臂差值或生产性能主张'}
    write(PKG/'correctness_evidence.json',{'status':'GPU_CORRECTNESS_PASS','arms':details,'cross_lifecycle':cross})
    write(PKG/'response_sizes.json',bytes_rows)
    write(PKG/'evidence_manifest.json',{'sha256':evidence})
    csv_write(PKG/'stage_timings.csv',stages)
    csv_write(PKG/'pending_timings.csv',pending)
    print('两臂真实证据与逻辑对照通过；没有统计性能推断。')


if __name__=='__main__':
    main()
