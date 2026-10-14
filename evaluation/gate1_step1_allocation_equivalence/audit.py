"""只读核验冻结选择并汇总兼容、收益重叠与有效容量压力，不执行新策略。"""
import csv,json,sys,hashlib,itertools
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'evaluation/p1_strong_baseline_validation'
sys.path[:0]=[str(SRC),str(ROOT)]
import common
from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
paths=set(Path(p) for p in common.input_hashes())
paths.update(SRC.glob('*.*'))
paths.update([ROOT/'evaluation/controlled_multiworkflow_v1/scenario.py',ROOT/'evaluation/rq3_agentx_formal_policy_evaluation.py',ROOT/'evaluation/rq3_formal_policy_evaluation.py',ROOT/'evaluation/p4_complete_experimental_evidence/complete_experiment_handbook.md'])
before={str(p):sha(p) for p in paths if p.is_file()}
assert all(sha(SRC/n)==h for n,h in read(SRC/'implementation_lock.json').items())
loaded=common.load_cases()
old=lines(SRC/'per_case_results.jsonl')
p0={(r['population'],r['snapshot_id'],r['k']):r for r in lines(ROOT/'evaluation/p0_allocation_mechanism_audit/all_cases.jsonl')}
assert len(loaded)==len(old)==573
result=[];matrices=[];prefixes=[]
original={f:lines(Path(f)) for f in {r['source_file'] for r in old}}
for line,(case,r) in enumerate(zip(loaded,old),1):
    assert (case['population'],case['snapshot_id'],case['k'])==(r['population'],r['snapshot_id'],r['k'])
    epoch=case['epoch'];assert common.fingerprint(epoch)==r['input_digest']
    assert case['benefit_matrix_ms']==r['benefit_matrix_ms']
    ids=[c.identifier for c in epoch.candidates];bm=r['benefit_matrix_ms'];n=len(epoch.pending);k=epoch.k
    support={c:{i for i,v in enumerate(bm[c]) if v>0} for c in ids}
    overlap=sum(bool(support[a]&support[b]) for a,b in itertools.combinations(ids,2))
    laminar=all(not(support[a]&support[b]) or support[a]<=support[b] or support[b]<=support[a] for a,b in itertools.combinations(ids,2))
    fs=r['policies']['FlowState'];assert set(fs['selected'])==set(case['references']['FlowState']['selected'])
    sizes={CHECKPOINT_SIZE_BYTES};size_source='RECONSTRUCTED_FROM_CODE'
    if r['population']=='OpenHands':
        path=Path(p0[(r['population'],r['snapshot_id'],k)]['metadata']['source'])
        before.setdefault(str(path),sha(path))
        sn=json.loads(read(path)['canonical_snapshot'])
        sizes={c['memory_bytes'] for c in sn['eligible_candidates']};size_source='RECORDED'
        assert all(c['recurrent_resident'] for c in sn['eligible_candidates'])
    assert len(sizes)==1
    size=next(iter(sizes));budget=k*size
    empty=sum(common.MODEL.estimate(p.target,p.target) for p in epoch.pending)
    item=dict(instance_id=f"{r['snapshot_id']}::budget={r['budget_ratio']}",workload=r['population'],snapshot_id=r['snapshot_id'],budget_ratio=r['budget_ratio'],num_continuations=n,num_candidates=len(ids),budget=k,checkpoint_size_bytes=size,budget_bytes=budget,byte_evidence=size_source,num_shared_checkpoints=sum(len(v)>1 for v in support.values()),num_overlapping_candidate_pairs=overlap,num_positive_candidates=sum(bool(v) for v in support.values()),candidate_compatibility_degrees=json.dumps([len(c.compatible) for c in epoch.candidates]),continuation_compatibility_degrees=json.dumps([sum(p.identifier in c.compatible for c in epoch.candidates) for p in epoch.pending]),positive_support_laminar=laminar,k_min_frontier=r['budget_structure']['minimum_checkpoints_for_all_frontiers'],budget_pressure=k<r['budget_structure']['minimum_checkpoints_for_all_frontiers'],source_file=r['source_file'],source_line=r['source_line'],p1b_line=line,input_digest=r['input_digest'])
    raw=original[r['source_file']][r['source_line']-1]
    item['trace_ids']=json.dumps([p['workflow_id'] for p in p0[(r['population'],r['snapshot_id'],k)]['metadata']['pending']]) if r['population']=='OpenHands' else raw['trace_id']
    forced=set()
    for p,e in zip(epoch.pending,r['budget_structure']['all_candidate_frontiers']):
        if e>0:
            maximal=[c.identifier for c in epoch.candidates if p.identifier in c.compatible and c.position==e]
            assert len(maximal)==1
            forced.update(maximal)
    assert len(forced)==item['k_min_frontier']
    item['unique_frontier_realizer_ids']=json.dumps(sorted(forced))
    if not item['budget_pressure']:assert forced==set(fs['selected'])
    gaps=[]
    for name,key in [('FlowState','flowstate'),('PCB','pcb'),('LDR','ldr'),('WFC','wfc')]:
        v=r['policies'][name];score=common.score(epoch,v['selected']);common.close(score['cost_ms'],v['cost_ms'])
        for actual,saved in zip(score['per_pending'],v['per_pending']):
            assert all(actual[t]==saved[t] for t in ('continuation_id','T','E','G'))
            common.close(actual['cost_ms'],saved['cost_ms'])
        assert len(v['selected'])==len(set(v['selected'])) and set(v['selected'])<=set(ids)
        assert len(v['selected'])*size<=budget
        item[key+'_selected_ids']=json.dumps(sorted(v['selected']))
        item[key+'_used_bytes']=len(v['selected'])*size
        item[key+'_cost_ms']=score['cost_ms'];item[key+'_benefit_ms']=empty-score['cost_ms']
        if name!='FlowState':
            item['flowstate_'+key+'_same']=set(v['selected'])==set(fs['selected'])
            item[key+'_objective_gap']=score['cost_ms']-item['flowstate_cost_ms'];gaps.append(item[key+'_objective_gap'])
    item['objective_gap']=max(abs(v) for v in gaps)
    u=[0.]*n;singleton={c:sum(bm[c]) for c in ids};selected=[];changed=False;inversion=False;ties=0
    for step,chosen in enumerate(r['flowstate_order']):
        remaining=[c for c in ids if c not in selected]
        marginal={c:sum(max(0.,v-u[i]) for i,v in enumerate(bm[c])) for c in remaining}
        changed|=any(abs(singleton[c]-marginal[c])>1e-9 for c in remaining)
        inv=any((singleton[a]-singleton[b])*(marginal[a]-marginal[b])<0 and abs(singleton[a]-singleton[b])>1e-9 and abs(marginal[a]-marginal[b])>1e-9 for a,b in itertools.combinations(remaining,2));inversion|=inv
        best=max(marginal.values());tied=[c for c in remaining if abs(marginal[c]-best)<=1e-9];ties+=len(tied)>1
        assert best-marginal[chosen]<1e-8
        prefixes.append(dict(instance_id=item['instance_id'],step=step+1,prefix=selected.copy(),chosen=chosen,singleton_ms={c:singleton[c] for c in remaining},marginal_ms=marginal,top_tied_ids=tied,strict_pair_order_reversal=inv))
        selected.append(chosen);u=[max(x,y) for x,y in zip(u,bm[chosen])]
    remaining_positive=sum(sum(max(0.,v-u[i]) for i,v in enumerate(bm[c]))>1e-9 for c in ids if c not in selected)
    item.update(remaining_positive_marginal_candidates=remaining_positive,budget_underfilled=len(selected)<k,positive_candidates_less_than_k=item['num_positive_candidates']<k,singleton_marginal_changed=changed,strict_pair_order_reversal=inversion,flowstate_top_tie_steps=ties)
    assert (remaining_positive>0)==item['budget_pressure']
    reasons=[]
    if not item['num_shared_checkpoints']:reasons.append('A：无共享正收益')
    if item['num_shared_checkpoints'] and overlap==0:reasons.append('B：共享但无正收益覆盖重叠')
    if overlap and not item['budget_pressure']:reasons.append('C：存在重叠但足以达到全候选前沿')
    if item['budget_pressure'] and all(not v['order_different'] for v in r['comparisons'].values()):reasons.append('D：紧预算下实际顺序一致')
    if not item['num_positive_candidates']:reasons.append('退化：无正收益候选')
    item['equivalence_explanation']='；'.join(reasons)
    result.append(item)
    matrices.append(dict(instance_id=item['instance_id'],candidate_ids=ids,pending_ids=[p.identifier for p in epoch.pending],targets=[p.target for p in epoch.pending],positions=[c.position for c in epoch.candidates],compatibility=[[int(p.identifier in c.compatible) for c in epoch.candidates] for p in epoch.pending],benefit_ms=[[bm[c][i] for c in ids] for i in range(n)],budget_bytes=budget,checkpoint_bytes=[size]*len(ids),source_file=r['source_file'],source_line=r['source_line']))
with (OUT/'gate1_step1_instance_statistics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
for name,values in [('matrices.jsonl',matrices),('prefix_diagnostics.jsonl',prefixes)]:
    (OUT/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
summary={}
for pop in ['OpenHands','AgentX']:
    a=[r for r in result if r['workload']==pop]
    summary[pop]=dict(cases=len(a),snapshots=len({r['snapshot_id'] for r in a}),num_continuations=dict(Counter(r['num_continuations'] for r in a)),candidate_range=[min(r['num_candidates'] for r in a),max(r['num_candidates'] for r in a)],budget_range=[min(r['budget'] for r in a),max(r['budget'] for r in a)])
    for field in ['budget_pressure','budget_underfilled','positive_candidates_less_than_k','singleton_marginal_changed','strict_pair_order_reversal','positive_support_laminar']:
        summary[pop][field]=sum(r[field] for r in a)
    for field in ['num_shared_checkpoints','num_overlapping_candidate_pairs','flowstate_top_tie_steps']:
        summary[pop][field+'_nonzero_cases']=sum(r[field]>0 for r in a)
    summary[pop]['classification']={letter:sum(letter+'：' in r['equivalence_explanation'] for r in a) for letter in 'ABCD'}
    summary[pop]['comparisons']={n:dict(same_set=sum(r['flowstate_'+n+'_same'] for r in a),different_set_same_objective=sum(not r['flowstate_'+n+'_same'] and abs(r[n+'_objective_gap'])<=1e-9 for r in a),different_set_different_objective=sum(not r['flowstate_'+n+'_same'] and abs(r[n+'_objective_gap'])>1e-9 for r in a),unknown=0) for n in ['pcb','ldr','wfc']}
save('summary.json',summary)
assert all(sha(Path(p))==h for p,h in before.items())
save('integrity.json',dict(source_sha256=before,unchanged=True,implementation_lock_pass=True,formal_input_checks=573,common_scores_checked=2292,gpu_started=False))
print(json.dumps(summary,ensure_ascii=False,indent=2))
