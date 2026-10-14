"""只读汇总冻结轨迹，并以冻结选择器验证隔离的数学反例。"""
import json
import sys
import hashlib
import itertools
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'evaluation/p1_strong_baseline_validation'
sys.path[:0]=[str(SRC),str(ROOT)]
import common,pcb,ldr,wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.workflow import PendingContinuation
from flowstate.state_catalog import CheckpointCandidate,is_compatible

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
paths=list(SRC.glob('*.py'))+list(SRC.glob('*.json'))+[SRC/'per_case_results.jsonl']+list((ROOT/'flowstate').glob('*.py'))
paths += [ROOT/'evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md',ROOT/'evaluation/p0_allocation_mechanism_audit/final_report.md',ROOT/'evaluation/p4_evaluation_restructuring/final_report.md']
paths += [ROOT/'evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl',ROOT/'rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl']
before={str(p):digest(p) for p in paths}
lock=json.loads((SRC/'implementation_lock.json').read_text())
assert all(digest(SRC/p)==v for p,v in lock.items())
historical=json.loads((SRC/'source_integrity_before.json').read_text())
for name in ('optimizer.py','recovery_model.py','state_catalog.py','workflow.py','executable_state.py'):
    p=ROOT/'flowstate'/name
    assert digest(p)==historical[str(p)]
rows=[json.loads(s) for s in (SRC/'per_case_results.jsonl').read_text().splitlines()]
assert len(rows)==573
original={str(p):[json.loads(s) for s in p.read_text().splitlines()] for p in paths[-2:]}
summary={};certificates=[];differences=[]
for lineno,r in enumerate(rows,1):
    fs=r['policies']['FlowState']
    raw=original[r['source_file']][r['source_line']-1]
    assert raw['snapshot_id']==r['snapshot_id'] and raw['snapshot_digest']==r['snapshot_digest']
    assert raw['k']==r['k'] and raw['budget_ratio']==r['budget_ratio']
    assert set(raw['policies']['FlowState']['selected_checkpoint_ids'])==set(fs['selected'])
    assert abs(raw['policies']['FlowState']['total_recovery_cost_ms']-fs['cost_ms'])<1e-9
    for name in ('PCB','LDR','WFC'):
        other=r['policies'][name]
        assert set(fs['selected'])==set(other['selected'])
        assert fs['cost_ms']==other['cost_ms']
        assert fs['per_pending']==other['per_pending']
        assert (r['flowstate_order']!=other['selected'])==r['comparisons'][name]['order_different']
    forced=set();maximizers={}
    for p,e in zip(r['input']['pending'],r['budget_structure']['all_candidate_frontiers']):
        if e<=0:continue
        ids=[c['identifier'] for c in r['input']['candidates'] if p['identifier'] in c['compatible'] and c['position']==e]
        assert len(ids)==1
        maximizers[p['identifier']]=ids[0];forced.update(ids)
    assert len(forced)==r['budget_structure']['minimum_checkpoints_for_all_frontiers']
    if r['k']>=len(forced):assert set(fs['selected'])==forced
    cert=dict(snapshot_id=r['snapshot_id'],budget_ratio=r['budget_ratio'],k=r['k'],source_line=lineno,source_file=r['source_file'],original_source_line=r['source_line'],forced_ids=sorted(forced),maximizers=maximizers,binding=r['k']<len(forced))
    certificates.append(cert)
    if any(v['order_different'] for v in r['comparisons'].values()):
        differences.append(dict(**cert,input=r['input'],policies={n:r['policies'][n] for n in ('FlowState','PCB','LDR','WFC')},flowstate_order=r['flowstate_order'],traces=r['traces'],comparisons=r['comparisons']))
for pop in ('OpenHands','AgentX'):
    a=[r for r in rows if r['population']==pop];u=list({r['snapshot_id']:r for r in a}.values())
    summary[pop]=dict(cases=len(a),snapshots=len(u),positive_degree=dict(Counter(sum(v>0 for v in vs) for r in u for vs in r['benefit_matrix_ms'].values())),compatibility_degree=dict(Counter(len(c['compatible']) for r in u for c in r['input']['candidates'])),binding=sum(r['budget_structure']['K_binds_best_frontiers'] for r in a),k_min=dict(Counter(r['budget_structure']['minimum_checkpoints_for_all_frontiers'] for r in a)),order_differences={n:sum(r['comparisons'][n]['order_different'] for r in a) for n in ('PCB','LDR','WFC')},shared_snapshots=sum(any(sum(v>0 for v in vs)>1 for vs in r['benefit_matrix_ms'].values()) for r in u))
dump('structure_summary.json',summary);dump('frontier_certificates.json',certificates);dump('order_difference_evidence.json',differences)
model=RecoveryCostModel();examples=[]

def example(name,k,root_l=None):
    """建立合法谱系；只运行小型解析输入，不执行正式总体上的选择器。"""
    if root_l is None:
        pending=[PendingContinuation('p1','w1',('r1',),8192,8192),PendingContinuation('p2','w2',('r2',),131072,131072)]
        candidates=[CheckpointCandidate('a','w1',('r1',),4096,1),CheckpointCandidate('b','w2',('r2',),4096,1),CheckpointCandidate('c','w1',('r1',),2048,1)]
    else:
        pending=[PendingContinuation('p1','w',('r','x'),8192,8192),PendingContinuation('p2','w',('r','y'),8192,8192)]
        candidates=[CheckpointCandidate('a','w',('r',),root_l,1),CheckpointCandidate('b','w',('r','x'),6144,1),CheckpointCandidate('c','w',('r','y'),6144,1)]
    epoch=common.Epoch(tuple(common.Candidate(c.checkpoint_id,c.token_pos,tuple(p.continuation_id for p in pending if is_compatible(c,p))) for c in candidates),tuple(common.Pending(p.continuation_id,p.planning_target) for p in pending),k)
    fs=GlobalOptimizer(model).select(pending,candidates,k)
    orders={'FlowState':[c.checkpoint_id for c in fs.selected]};traces={}
    for n,m in [('PCB',pcb),('LDR',ldr),('WFC',wfc)]:orders[n],traces[n]=m.select(epoch)
    results={n:common.score(epoch,s) for n,s in orders.items()}
    assert abs(results['FlowState']['cost_ms']-fs.recovery_cost_after_ms)<1e-9
    marginal=[];prefix=[]
    for cid in orders['FlowState']:
        current=common.score(epoch,prefix)['cost_ms']
        marginal.append(dict(prefix=prefix.copy(),gains={c.identifier:current-common.score(epoch,prefix+[c.identifier])['cost_ms'] for c in epoch.candidates if c.identifier not in prefix},chosen=cid))
        prefix.append(cid)
    exact=min((common.score(epoch,s)['cost_ms'],s) for j in range(k+1) for s in itertools.combinations([c.identifier for c in epoch.candidates],j))
    from dataclasses import asdict
    examples.append(dict(case_id=name,k=k,input=asdict(epoch),lineage=[asdict(c) for c in candidates],pending_lineage=[asdict(p) for p in pending],benefit_matrix_ms=common.benefit_matrix(epoch),results=results,traces=traces,flowstate_steps=marginal,exact_cost_ms=exact[0],exact_selected=exact[1]))
example('数学甲',1)
example('数学乙一',1,4096)
example('数学乙二',2,4096)
example('数学丙',1,1024)
dump('analytical_examples.json',examples)
assert examples[0]['results']['FlowState']['cost_ms']<examples[0]['results']['LDR']['cost_ms']
assert examples[1]['results']['FlowState']['cost_ms']<examples[1]['results']['PCB']['cost_ms']
assert examples[2]['results']['FlowState']['cost_ms']>examples[2]['results']['PCB']['cost_ms']
assert examples[3]['results']['FlowState']['cost_ms']<examples[3]['results']['WFC']['cost_ms']
assert all(digest(Path(p))==v for p,v in before.items())
dump('integrity.json',dict(before=before,after={p:digest(Path(p)) for p in before},implementation_lock_pass=True,unchanged=True,formal_rows=573,synthetic_cases=4))
print(json.dumps(summary,ensure_ascii=False,indent=2))
for e in examples:print(e['case_id'],{n:(r['selected'],r['cost_ms']) for n,r in e['results'].items()})
