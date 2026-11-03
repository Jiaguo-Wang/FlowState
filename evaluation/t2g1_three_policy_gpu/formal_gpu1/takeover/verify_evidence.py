"""接管时只读复核原始合同；仅在原检查器及原始证据均通过时生成门禁凭据。"""
import hashlib
import json
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
checks=[]
source_hashes={}
def read(p):
    p=Path(p);b=p.read_bytes();source_hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def ck(name,ok):
    checks.append({'check':name,'pass':bool(ok)})
    if not ok:raise ValueError('合同失败：'+name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main(phase):
    protocol=read(OUT/'protocol.json');pre=read(OUT/'formal_precheck.json')
    for name,h in pre['shared_files'].items():ck('共享输入 '+name,sha(OUT/name)==h)
    for name,key in [('protocol.json','protocol'),('policy_plans.json','policy_plans')]:ck('冻结 '+name,sha(OUT/name)==pre[key])
    ck('上游计划',sha(OUT.parent/'t2g1_plans.json')==pre['source_t2g1_plans'])
    ck('原执行方案',sha(OUT.parent/'formal_plan.md')==pre['source_formal_plan'])
    lock=read(OUT/('implementation_lock_'+phase+'.json'))
    for p,h in lock['files'].items():ck('实现锁 '+p,sha(p)==h)
    base=read(OUT/('formal_verification_'+phase+'.json'))
    arms=protocol[phase+'_arms'];ck('原检查器',base['verdict']=='PASS' and base['arms_checked']==len(arms) and all(c['pass'] for c in base['checks']))
    plans=read(OUT.parent/'t2g1_plans.json')['plans'];inputs=read(OUT/'prepared_inputs.json')
    ps={p['continuation_id']:p for p in inputs['pendings']}
    lifecycle=[]
    for a in arms:
        n=a['name'];p=OUT/n;d=p/'run';expected=plans[a['policy']]
        for name in pre['shared_files']:ck(n+'逐臂输入 '+name,sha(p/name)==sha(OUT/name))
        ck(n+'逐臂选集文件',sha(p/'policy_plans.json')==sha(OUT/'policy_plans.json'))
        s=read(d/'summary.json');r=read(d/'reconciliation.json');post=read(d/'post_request.json');output=read(d/'output.json');request=read(d/'request.json')
        cold=read(OUT/('cold_reference_'+a['pending']+'.json'))
        ck(n+'原始Cold',output['output_ids']==s['output_ids']==cold['output_ids'] and len(output['output_ids'])==8 and output['meta_info']['completion_tokens']==8 and int(output['meta_info'].get('num_retractions',0) or 0)==0)
        ck(n+'输入与生成参数',request['pending']['token_ids']==ps[a['pending']]['token_ids'] and request['sampling']=={'temperature':0,'max_new_tokens':8,'ignore_eos':True})
        ck(n+'计划身份',r['ok'] and r['policy']==a['policy'] and r['budget_K']==3 and r['policy_selected_cpu_ids']==expected['selected_checkpoint_ids'])
        target=r['target'];e=expected['expected_E_G'][a['pending']]
        ck(n+'T/E/G',all(target['expected'][k]==e[k] for k in ['T','E','G']) and e['T']-e['E']==e['G'])
        ck(n+'连续窗口',r['apply_validation']['status']=='PASS' and r['apply_validation']['same_scheduler_call'] and r['before']['barrier_id']==r['after']['barrier_id']==r['plan']['epoch_id'])
        ledger=r['reconciliation'];ck(n+'协调完成',ledger['status']=='PASS' and ledger['phase']=='complete' and not ledger['cleanup_required'] and ledger['error'] is None and ledger['failed_checkpoint_id'] is None)
        ck(n+'准确驱逐补集',[x['checkpoint_id'] for x in r['operations']]==r['plan']['logical_eviction_ids'] and len(r['operations'])==13 and all(x['returned'] for x in r['operations']))
        selected=set(r['plan']['selected_ids']);bound={b['candidate_id']:b['node'] for b in ledger['binding']['candidates']};after={x['node_id']:x for x in r['after']['nodes']}
        for cid,node in bound.items():
            actual=after[node['node_id']]
            ck(n+'驻留 '+cid,(actual['conv_complete'] and actual['temporal_complete'] and actual['slots']==node['slots'] and actual['component_ids']==node['component_ids']) if cid in selected else (not actual['slots'] and not actual['component_ids']))
        ck(n+'槽位记账',len(r['before']['free_slots'])==12 and len(r['after']['free_slots'])==25 and r['before']['pool_capacity']==r['after']['pool_capacity']==28)
        ck(n+'FA树索引',r['before']['attention_digest']==r['after']['attention_digest'] and r['audit_before']['state']['tree']['full_tree_sha256']==r['audit_after']['state']['tree']['full_tree_sha256'] and r['audit_before']['state']['tree']['structure_sha256']==r['audit_after']['state']['tree']['structure_sha256'])
        ck(n+'保留内容与请求保护',r['selected_payload_before']==r['selected_payload_after']==post['state']['selected_payload'] and post['state']['selected_components']==r['admission_baseline']['selected_components'])
        ck(n+'受保护FA内容',post['state']['fa_protected_payload']==r['fa_protected_payload'] and all(post['state']['fa_map'].get(k)==v for k,v in r['fa_map_before'].items()))
        tr=post['trace'];matches=[m['native'] for m in tr['matches'] if m.get('native',{}).get('request_id')==a['pending']];ck(n+'Native匹配存在',bool(matches));m=matches[-1]
        ck(n+'NativeToken和前沿',m['input_token_digest']==target['pending']['token_prefix_sha256'] and m['executable_prefix']==e['E'] and m['physical_fa_hit']==ps[a['pending']]['attention_frontier'])
        ck(n+'实际执行',bool(tr['execution']) and all(x['native_run_returned'] for x in tr['execution']) and len(tr['execution'])==len(tr['compute']) and tr['compute'][0]['is_extend'] and all(x['returned'] for x in tr['compute']))
        ck(n+'实际Prefill',tr['execution'][0]['batch_prefix_length']==e['E'] and tr['execution'][0]['extend_token_count']==len(ps[a['pending']]['token_ids'])-e['E'])
        if e['E']>0:
            source=bound[target['source_checkpoint']];ck(n+'COW来源',m['best_match_node']==source['node_id'] and m['cow_source_slots']==source['slots'])
            ck(n+'COW完成',any(c['returned'] and c['source_slots']==source['slots'] and c['destination_slots']==m['working_slots'] for c in tr['copies']))
            ck(n+'工作槽隔离',not set(m['working_slots']) & {slot for cid,node in bound.items() if cid in selected for slot in node['slots']})
        sw=[m for m in tr['shape_warm_matches'] if m['request_id']=='SHAPE_WARM_B_POST_APPLY'][-1]
        ck(n+'形状预热',sw['executable_prefix']==sw['physical_fa_hit']==e['E'] and sw['input_token_count']==len(ps[a['pending']]['token_ids']))
        ck(n+'编译和审计窗口',s['triton_new_kernels'][-1]['window']=='post_shape_warm→post_request' and s['triton_new_kernels'][-1]['new_entries']==0 and len(tr['audit'])==2 and all(x['kind']=='cache_state_full_hash' for x in tr['audit']))
        device=read(p/'device_binding.json');ck(n+'GPU1唯一绑定',device['status']=='PASS' and device['host_gpu_uuid']=='GPU-051c9abc-864a-9f31-c4f7-a737aed997ff' and device['cuda_device_count']==1 and device['logical_device']==0)
        ck(n+'进程退出',read(OUT/(n+'_process.json'))['returncode']==0)
        clean=read(OUT/(n+'_cleanup.json'));ck(n+'清理',clean['status']=='PASS' and not clean['own_container'] and len(clean['observations'])==3 and all(x['memory_mib']<100 and len(x['processes'].strip().splitlines())==1 for x in clean['observations']))
        ec=read(d/'engine_cleanup.json');ck(n+'Engine关闭',ec['engine_created'] and ec['shutdown_returned'])
        lifecycle.append(r['before']['barrier_id'])
    ck('独立生命周期',len(set(lifecycle))==len(arms))
    report={'status':'PASS','phase':phase,'checks':checks,'sha256':source_hashes,'notes':['协调完整FA内容检查由哈希锁定control.py在返回ok前执行；原始独立全量FA内容哈希未额外落盘。本复核直接比较落盘树/索引/受保护内容，不伪造额外内容摘要。','Gate计时不得进入正式主矩阵性能统计。']}
    rp=OUT/'takeover'/('evidence_'+phase+'.json');write(rp,report)
    if phase=='gate':
        write(OUT/'gate_passed.json',{'passed':True,'method':'原verify_formal.py加原始证据合同CPU复核','verification_sha256':sha(OUT/'formal_verification_gate.json'),'evidence_sha256':sha(rp),'protocol_sha256':sha(OUT/'protocol.json'),'checker_sha256':sha(__file__),'arms':[a['name'] for a in arms]})
    print(phase,len(arms),len(checks),'PASS')

if __name__=='__main__':
    phase=sys.argv[1]
    try:main(phase)
    except Exception as exc:
        write(OUT/'takeover'/('failure_'+phase+'.json'),{'error':repr(exc),'checks':checks})
        raise
