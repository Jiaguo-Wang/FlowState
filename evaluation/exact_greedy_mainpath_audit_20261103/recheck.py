"""复用既有CPU门禁，经主库入口复算，并只读核查已有GPU证据。"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
WATCH={}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    p=Path(p)
    WATCH[str(p)]=sha(p)
    return json.loads(p.read_text())

def save(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    # 只改评估程序输出位置，旧程序和冻结产物均不改动。
    gate=ROOT/'evaluation/paper_hybrid_solver_integration_gate.py'
    old=ROOT/'evaluation/paper_hybrid_solver_integration_gate.json'
    WATCH[str(gate)]=sha(gate);WATCH[str(old)]=sha(old)
    spec=importlib.util.spec_from_file_location('existing_cpu_gate',gate)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.EVAL=OUT
    m.main()
    result=read(OUT/'paper_hybrid_solver_integration_gate.json')
    refs={}
    source=ROOT/'evaluation/greedy_swap_structure_e3_step1/condition_results.csv'
    WATCH[str(source)]=sha(source)
    with source.open() as f:
        for r in csv.DictReader(f):refs[r['case'],int(r['K']),r['policy']]=r
    rows=[]
    for r in result['results']['A_strict20']:
        ref=refs[r['case'],r['K'],'Exact'];selected=json.loads(ref['selected'])
        rows.append({'condition':r['case'],'n':r['gate']['candidates'],'pending_count':r['gate']['pendings'],
                     'K':r['K'],'enumerable_subsets':r['gate']['enumerable_subsets'],'actual_solver':r['solver'],
                     'selected':json.dumps(r['selected']),'J_ms':r['J_ms'],'reference_type':'冻结Exact Oracle',
                     'reference_selected':json.dumps(selected),'reference_J_ms':float(ref['J_ms']),
                     'same_set':set(r['selected'])==set(selected),'delta_J_ms':r['J_ms']-float(ref['J_ms']),
                     'validate_plan':'PASS','independent_oracle_match':r['matches_b5d1_oracle']})
    r=result['results']['B_ck3'];j=154.34895561599998
    rows.append({'condition':'C/K3','n':16,'pending_count':4,'K':3,'enumerable_subsets':r['gate']['enumerable_subsets'],
                 'actual_solver':r['solver'],'selected':json.dumps(r['selected']),'J_ms':r['J_ms'],
                 'reference_type':'冻结Greedy参考，非Exact最优证书','reference_selected':json.dumps(r['selected']),
                 'reference_J_ms':j,'same_set':r['matches_frozen_plan'],'delta_J_ms':r['J_ms']-j,
                 'validate_plan':'PASS','independent_oracle_match':'不适用：超Exact准入域'})
    with (ROOT/'solver_dispatch_cpu_recheck.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    base=ROOT/'evaluation/hybrid_solver_gpu_gate_g1'
    summaries=[];details=[]
    arms={'exact':['e1_w1k2_exact_A','e2_w1k2_exact_B','e3_w1k2_exact_C','e4_w1k2_exact_D'],
          'greedy':['g1_ck3_greedy_A_r3','g2_ck3_greedy_B_r3','g3_ck3_greedy_C_r3','g4_ck3_greedy_D_r3']}
    for branch,names in arms.items():
        identity=read(base/branch/'source_identity.json')
        for p,h in identity['hybrid_entry_sources'].items():
            WATCH[p]=sha(p);assert WATCH[p]==h,p
        for run in names:
            s=read(base/branch/run/'summary.json');proc=read(base/branch/f'{run}_process.json');cl=read(base/branch/f'{run}_cleanup.json')
            cold_id=s['native_match']['request_id']
            cold=read(base/branch/f'cold_reference_{cold_id}.json')
            lock=read(base/branch/f'execution_lock_{run}.json')
            for p,h in identity['hybrid_entry_sources'].items():assert lock['files'][p]==h
            assert s['status']=='PASS' and proc['returncode']==0 and cl['status']=='PASS'
            assert s['selection_source']=='GlobalOptimizer.plan_strict(在线)'
            assert s['output_ids']==cold['output_ids'] and len(s['output_ids'])==8
            assert len(s['evictions'])==(5 if branch=='exact' else 13) and all(v for _,v in s['evictions'])
            summaries.append({'branch':branch,'run':run,'solver':s['strict_decision']['solver'],
                              'gate':s['strict_decision']['gate'],'selected':s['strict_decision']['selected_ids'],
                              'J_ms':s['strict_decision']['objective_ms'],'H':s['native_match']['physical_fa_hit'],
                              'E':s['native_match']['executable_prefix'],'G':s['native_match']['replay_gap'],
                              'evictions':len(s['evictions']),'cold_exact_match':True,'cleanup':'PASS'})
        # 代表臂下钻原始协调、Native执行和实际复制；不运行任何历史验证器的落盘入口。
        run=names[1];p=base/branch/run
        r=read(p/'reconciliation.json');post=read(p/'post_request.json');s=read(p/'summary.json')
        from flowstate.strict_epoch import StrictAllocationEpoch,StrictAllocationPlan,PendingPlan,StrictCandidate,StrictPending,StrictEdge
        from flowstate.state_catalog import CheckpointCandidate
        from flowstate.strict_control import validate_plan
        from dataclasses import asdict
        raw=json.dumps(r['raw_epoch'],sort_keys=True,ensure_ascii=False).encode()
        # GPU记录使用已认证的规范Epoch；该原始capture不是旧版B4 Snapshot格式。
        ed=dict(r['epoch']);assert hashlib.sha256(raw).hexdigest()==ed['snapshot_sha256']
        cs=[]
        for item in ed['candidates']:
            item=dict(item);core=dict(item['candidate']);core['lineage_path']=tuple(core['lineage_path'])
            item['candidate']=CheckpointCandidate(**core);item['token_ids']=tuple(item['token_ids'])
            cs.append(StrictCandidate(**item))
        ed['candidates']=tuple(cs)
        ed['pendings']=tuple(StrictPending(**dict(item,token_ids=tuple(item['token_ids']))) for item in ed['pendings'])
        ed['edges']=tuple(StrictEdge(**dict(item,conditions=tuple(tuple(x) for x in item['conditions']))) for item in ed['edges'])
        epoch=StrictAllocationEpoch(**ed)
        plan_data=dict(r['plan']);plan_data['pending']=tuple(PendingPlan(**x) for x in plan_data['pending'])
        for key in ('selected_ids','logical_eviction_ids'):plan_data[key]=tuple(plan_data[key])
        plan=StrictAllocationPlan(**plan_data);validate_plan(epoch,plan)
        assert asdict(epoch)==r['epoch'] or json.loads(json.dumps(asdict(epoch)))==r['epoch']
        assert r['strict_decision']==s['strict_decision']
        assert list(plan.selected_ids)==s['strict_decision']['selected_ids']
        ledger=r['reconciliation'];binding=ledger['binding']
        assert ledger['status']=='PASS' and ledger['phase']=='complete' and ledger['error'] is None
        assert ledger['completed_evictions']==list(plan.logical_eviction_ids)
        assert binding['selected_ids']==list(plan.selected_ids)
        assert binding['evicted_ids']==list(plan.logical_eviction_ids)
        assert r['apply_validation']['same_scheduler_call'] and r['apply_validation']['status']=='PASS'
        assert r['selected_payload_before']==r['selected_payload_after']
        assert r['before']['attention_digest']==r['after']['attention_digest']
        if branch=='exact':assert r['fa_content_before']==r['fa_content_after'] and r['fa_content_postcheck']=='PASS'
        assert post['ok']
        copy=post['trace']['copies'];assert len(copy)==1
        copy=copy[0];assert copy['returned']
        assert copy['source_payload_before']==copy['source_payload_after']==copy['destination_payload_after']
        assert not set(copy['source_slots'])&set(copy['destination_slots'])
        execute=[x for x in post['trace']['execution'] if x.get('is_extend')]
        assert execute and all(x['native_run_returned'] for x in execute)
        assert execute[0]['extend_start']==s['native_match']['executable_prefix']
        assert copy['source_slots']==s['native_match']['cow_source_slots']
        assert copy['destination_slots']==s['native_match']['working_slots']
        bound=[b for b in binding['candidates'] if b['node']['node_id']==s['native_match']['best_match_node']]
        assert len(bound)==1 and bound[0]['candidate_id'] in plan.selected_ids
        assert bound[0]['node']['slots']==copy['source_slots']
        details.append({'branch':branch,'run':run,'epoch_id':epoch.epoch_id,'snapshot_sha256':epoch.snapshot_sha256,
                        'epoch_digest':epoch.content_digest,'solver':s['strict_decision'],'plan':r['plan'],
                        'source_checkpoint':bound[0]['candidate_id'],'source_node':bound[0]['node']['node_id'],
                        'source_slots':copy['source_slots'],'working_slots':copy['destination_slots'],
                        'reconciliation_status':ledger['status'],'completed_evictions':ledger['completed_evictions'],
                        'same_scheduler_call':True,'selected_payload_unchanged':True,'attention_digest_unchanged':True,
                        'cow_completed_content_equal_and_isolated':True,'native_extend':execute[0],
                        'summary_cold_output_match':True,'说明':'原始GPU证据只读核查，未执行新的GPU生命周期。'})
    save('gpu_recheck.json',{'verdict':'PASS','summaries':summaries,'representative_chains':details,
                           '注意':'历史失败尝试仍保留，见原greedy/verify.json；不作性能结论。'})
    for p,h in WATCH.items():assert sha(p)==h,p
    save('source_integrity.json',{'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                'optimizer_worktree_modified':bool(subprocess.check_output(['git','diff','--','flowstate/optimizer.py'],cwd=ROOT)),
                                'files':WATCH,'unchanged':True})
    print('当前入口CPU21条件与已有GPU8臂证据复核PASS。')

if __name__=='__main__':main()
