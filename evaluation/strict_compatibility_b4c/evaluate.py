"""只评估四个正式入选严格快照，不启动Runtime或改写原始证据。"""
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
BASE=ROOT/'evaluation/p1_strong_baseline_validation'
sys.path.insert(0,str(BASE))
import common,pcb,ldr,wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from evaluation.sota_policies import MarconiStylePolicy,_build_marconi_metrics
MODEL=RecoveryCostModel();EPS=1e-9;GROUPS=(1,2,3,4)

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def subsets(ids,k):return [s for n in range(k+1) for s in itertools.combinations(sorted(ids),n)]

class MatrixOptimizer(GlobalOptimizer):
    """只替换成本接口，保留生产Greedy循环、停止规则与同分规则。"""
    def __init__(self,epoch):super().__init__(MODEL);self.epoch=epoch
    def _recovery_cost(self,continuations,selected):return common.score(self.epoch,tuple(c.checkpoint_id for c in selected))['cost_ms']

def load(group):
    assert group in GROUPS
    s=read(RAW/f'g{group:03d}'/'snapshot.json');summary=read(RAW/f'g{group:03d}'/'summary.json')
    assert sha(RAW/f'g{group:03d}'/'snapshot.json')==summary['snapshot_sha256']
    assert summary['status']=='CAPTURED' and summary['edge_counts']['UNKNOWN']==0
    cs=s['candidates'];ps=s['pendings'];edges=s['edges']
    assert len({c['checkpoint_id'] for c in cs})==len(cs)
    assert len(edges)==len(cs)*len(ps)==len({(e['checkpoint_id'],e['continuation_id']) for e in edges})
    for e in edges:
        assert e['status'] in ('EPOCH_ADMISSIBLE','REJECTED')
        if e['status']=='EPOCH_ADMISSIBLE':assert all(v is True for v in e['conditions'].values())
    candidates=tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(e['continuation_id'] for e in edges if e['checkpoint_id']==c['checkpoint_id'] and e['status']=='EPOCH_ADMISSIBLE')) for c in cs)
    pending=tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in ps)
    epoch=common.Epoch(candidates,pending,len(candidates))
    core=tuple(CheckpointCandidate(c['checkpoint_id'],c['workflow_id'],tuple(c['metadata']['lineage_path']),c['token_position_L'],c['metadata']['memory_bytes'],c['state_identity']['target_mamba_present']) for c in cs)
    assert len({c.memory_bytes for c in core})==1 and all(c.recurrent_resident for c in core)
    return s,epoch,core

def evaluate(group):
    s,epoch,core=load(group);ids=tuple(c.identifier for c in epoch.candidates);bm=common.benefit_matrix(epoch)
    def f(chosen):return sum(max([bm[c][i] for c in chosen]+[0.]) for i in range(len(epoch.pending)))
    j0=common.score(epoch,())['cost_ms'];assert f(())==0
    for e in s['edges']:
        i=next(i for i,p in enumerate(epoch.pending) if p.identifier==e['continuation_id'])
        value=bm[e['checkpoint_id']][i];assert value>=0
        if e['status']=='REJECTED':assert value==0
    allfront=common.all_frontier(epoch)
    kmin=next(k for k in range(len(ids)+1) if any(common.frontier_vector(common.score(epoch,x))==allfront for x in itertools.combinations(ids,k)))
    md={c['checkpoint_id']:c['metadata'] for c in s['candidates']}
    recency={c:float(v['last_access_order']) for c,v in md.items()};flop={c:v['marconi_flop_saved'] for c,v in md.items()}
    assert {v['marconi_alpha'] for v in md.values()}=={1.0}
    marconi_metrics=_build_marconi_metrics(core,recency,flop,1.0)
    all_sets=subsets(ids,4);scores={x:common.score(epoch,x) for x in all_sets}
    for x,score in scores.items():assert abs(score['cost_ms']-(j0-f(x)))<EPS
    rows=[];traces={}
    for k in range(1,5):
        current=replace(epoch,k=k);available=[x for x in all_sets if len(x)<=k]
        optimum=min(scores[x]['cost_ms'] for x in available)
        ties=[x for x in available if abs(scores[x]['cost_ms']-optimum)<=EPS]
        exact=min(ties,key=lambda x:(len(x),x))
        greedy=tuple(c.checkpoint_id for c in MatrixOptimizer(epoch).select((),core,k*core[0].memory_bytes).selected)
        selections={'FlowState':greedy,'Exact OPT':exact}
        for name,module in [('PCB',pcb),('LDR',ldr),('WFC',wfc)]:
            selections[name],traces[f'{k}/{name}']=module.select(current)
        ranked=sorted(ids,key=lambda c:(-sum(bm[c]),c))
        selections['StandaloneTopK']=tuple(sorted([c for c in ranked if sum(bm[c])>EPS][:k]))
        selections['Marconi Adaptation']=MarconiStylePolicy().select(core,k,recency,flop,1.0).selected_checkpoint_ids
        prefix=[];steps=[]
        for chosen in greedy:
            gains={c:f(prefix+[c])-f(prefix) for c in ids if c not in prefix}
            best=max(gains.values());assert best-gains[chosen]<=EPS
            assert chosen==min(c for c in gains if abs(gains[c]-best)<=EPS)
            steps.append({'prefix':list(prefix),'chosen':chosen,'marginals_ms':gains});prefix.append(chosen)
        traces[f'{k}/FlowState']=steps
        for name,chosen in selections.items():
            score=common.score(current,chosen);delta=score['cost_ms']-common.score(epoch,greedy)['cost_ms']
            rows.append({'group':f'g{group:03d}','K':k,'K_min':kmin,'policy':name,'selected':list(chosen),'F_ms':f(chosen),'J_ms':score['cost_ms'],'gap_to_exact_ms':score['cost_ms']-optimum,'baseline_minus_flowstate_ms':delta,'FlowState_WTL':'W' if delta>EPS else 'L' if delta< -EPS else 'T','same_set_as_flowstate':set(chosen)==set(greedy),'different_set_same_objective':set(chosen)!=set(greedy) and abs(delta)<=EPS,'per_pending':score['per_pending'],'exact_opt_sets':[list(x) for x in ties] if name=='Exact OPT' else None})
    return {'group':f'g{group:03d}','structure':s['structure'],'K_min':kmin,'all_candidate_frontiers':allfront,'J_empty_ms':j0,'benefit_matrix_ms':bm,'input':asdict(epoch),'raw_metadata':md,'marconi_utilities':marconi_metrics,'rows':rows,'traces':traces,'exact_subsets_checked':len(all_sets)}

def main():
    lock=read(BASE/'implementation_lock.json')
    for p,h in lock.items():assert sha(BASE/p)==h
    batch=read(RAW/'batch_summary.json')
    assert [x['group'] for x in batch['groups'] if x['decision']=='入选']==list(GROUPS)
    sources=[RAW/f'g{i:03d}'/'snapshot.json' for i in GROUPS]+[RAW/'batch_summary.json',ROOT/'flowstate/optimizer.py',ROOT/'flowstate/recovery_model.py',ROOT/'evaluation/sota_policies.py',ROOT/'evaluation/rq3_sanity_structure_audit.py',ROOT/'evaluation/rq3_frozen_snapshot_evaluator.py']+[BASE/p for p in lock]
    before={str(p):sha(p) for p in sources}
    results=[evaluate(i) for i in GROUPS];rows=[r for x in results for r in x['rows']]
    write('results.json',{'status':'PASS','scope':'四个正式入选Snapshot；16预算条件；112策略结果；仅CPU模型化ms','cost_model':asdict(MODEL.metadata),'snapshots':results})
    fields=['group','K','K_min','policy','selected','F_ms','J_ms','gap_to_exact_ms','baseline_minus_flowstate_ms','FlowState_WTL','same_set_as_flowstate','different_set_same_objective','per_pending']
    with (OUT/'snapshot_budget_policy.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
    assert before=={str(p):sha(p) for p in sources};write('input_integrity.json',{'status':'PASS','sha256':before})
    for x in results:
        print(x['group'],'Kmin',x['K_min'])
        for k in range(1,5):print(k,[(r['policy'],round(r['J_ms'],6)) for r in x['rows'] if r['K']==k])
if __name__=='__main__':main()
