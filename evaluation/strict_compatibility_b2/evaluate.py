"""在冻结准入矩阵上执行独立 CPU 分配比较，不调用 Runtime。"""
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'evaluation/p1_strong_baseline_validation'
sys.path.insert(0,str(BASE))
import common
import pcb
import ldr
import wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
OUT = Path(__file__).resolve().parent
B1 = ROOT/'evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json'
B1C = ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2'
EPS = 1e-9
MODEL = RecoveryCostModel()

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    lock=read(BASE/'implementation_lock.json')
    for name,digest in lock.items():
        assert sha(BASE/name)==digest
    frozen=read(B1); s=frozen['positive_arm']
    assert s['edge_status_counts']=={'EPOCH_ADMISSIBLE':11,'REJECTED':21,'UNKNOWN':0}
    cs={c['checkpoint_id']:c for c in s['checkpoints']};ps={p['continuation_id']:p for p in s['pendings']}
    assert len(cs)==len(s['checkpoints'])==8 and len(ps)==4
    assert len(s['edges'])==32 and len({(e['checkpoint_id'],e['continuation_id']) for e in s['edges']})==32
    edges=[]
    for e in s['edges']:
        c,p=cs[e['checkpoint_id']],ps[e['continuation_id']]
        assert len(c['token_ids'])==c['token_position_L']==c['state_identity']['prefix_tokens']
        assert p['known_before_dispatch'] and p['planning_target_T']<=len(p['token_ids'])
        if e['status']=='EPOCH_ADMISSIBLE':
            assert all(e['conditions'].values())
            assert c['token_position_L']<=p['planning_target_T']
            assert c['token_ids']==p['token_ids'][:c['token_position_L']]
        validated=c['checkpoint_id']=='RQ3_G000_B_TURN_001' and p['continuation_id']=='RQ3_G000_A_PENDING_TURN_003'
        edges.append({**e,'B1C_evidence_level':'RESUME_VALIDATED' if validated else '未由B1-C逐边恢复认证'})
    for a in ('retained','removed','reference'):
        d=read(B1C/a/'evidence.json');assert d['status']=='PASS' and d['b1_sha256']==sha(B1)
    for a in frozen['future_boundary_audit']:
        assert not any(a[k] for k in ('future_checkpoint_read','future_timing_read','pending_turn_output_read','r_plus_2_message_consumed','r_plus_2_request_materialized'))
    candidates=tuple(common.Candidate(cid,c['token_position_L'],tuple(e['continuation_id'] for e in edges if e['checkpoint_id']==cid and e['status']=='EPOCH_ADMISSIBLE')) for cid,c in sorted(cs.items()))
    pending=tuple(common.Pending(pid,p['planning_target_T']) for pid,p in sorted(ps.items()))
    return common.Epoch(candidates,pending,8),edges,cs

EPOCH,EDGES,RAW=load()
BM=common.benefit_matrix(EPOCH)
IDS=tuple(c.identifier for c in EPOCH.candidates)
J0=common.score(EPOCH,())['cost_ms']

def benefit(selected):
    return sum(max([BM[c][i] for c in selected]+[0.0]) for i in range(4))

class MatrixOptimizer(GlobalOptimizer):
    """复用生产贪心循环，仅将成本接口绑定到冻结准入矩阵。"""
    def _recovery_cost(self, continuations, selected):
        return common.score(EPOCH,tuple(c.checkpoint_id for c in selected))['cost_ms']

def greedy(k):
    candidates=[CheckpointCandidate(c.identifier,RAW[c.identifier]['workflow_id'],(),c.position,1) for c in EPOCH.candidates]
    r=MatrixOptimizer(MODEL).select((),candidates,k)
    return tuple(c.checkpoint_id for c in r.selected)

def subsets(k):
    return [s for n in range(k+1) for s in itertools.combinations(IDS,n)]

def exact(k):
    allsets=subsets(k);costs=[common.score(EPOCH,s)['cost_ms'] for s in allsets];minimum=min(costs)
    ties=[s for s,c in zip(allsets,costs) if abs(c-minimum)<=EPS]
    return min(ties,key=lambda s:(len(s),s)),ties

def run():
    records=[];traces={}
    for k in range(1,5):
        epoch=replace(EPOCH,k=k);opt,ties=exact(k);fs=greedy(k)
        choices={'FlowState':fs,'Exact OPT':opt}
        for name,module in [('PCB',pcb),('LDR',ldr),('WFC',wfc)]:
            choices[name],trace=module.select(epoch);traces[f'{k}/{name}']=trace
        ranking=sorted(IDS,key=lambda c:(-sum(BM[c]),c))
        choices['StandaloneTopK']=tuple(sorted([c for c in ranking if sum(BM[c])>EPS][:k]))
        prefix=[];steps=[]
        for cid in fs:
            gains={c:benefit(prefix+[c])-benefit(prefix) for c in IDS if c not in prefix}
            steps.append({'prefix':list(prefix),'marginal_ms':gains,'chosen':cid,'zero_marginal_ids':[c for c,v in gains.items() if abs(v)<=EPS]});prefix.append(cid)
        traces[f'{k}/FlowState']=steps
        for name,s in choices.items():
            result=common.score(epoch,s);f=benefit(s)
            records.append({'K':k,'strategy':name,'selected':list(s),'F_ms':f,'J_ms':result['cost_ms'],'gap_vs_opt_ms':result['cost_ms']-common.score(epoch,opt)['cost_ms'],'same_set_as_flowstate':set(s)==set(fs),'tie_with_different_set':set(s)!=set(fs) and abs(result['cost_ms']-common.score(epoch,fs)['cost_ms'])<=EPS,'per_pending':result['per_pending'],'exact_opt_sets':[list(t) for t in ties] if name=='Exact OPT' else None})
    return records,traces

def main():
    sources=[B1,BASE/'implementation_lock.json',ROOT/'flowstate/optimizer.py',ROOT/'flowstate/recovery_model.py',ROOT/'evaluation/FORMAL_RECOVERY_MODEL.md',ROOT/'evaluation/sota_policies.py',ROOT/'evaluation/sota_metadata.py',ROOT/'evaluation/rq3_sanity_structure_audit.py']
    sources += [BASE/n for n in ('common.py','pcb.py','ldr.py','wfc.py')]
    sources += [p for p in B1C.rglob('*') if p.is_file() and 'triton_cache' not in p.parts]
    before={str(p):sha(p) for p in sources}
    records,traces=run()
    for r in records:
        assert abs(r['J_ms']-(J0-r['F_ms']))<EPS
    allfront=common.all_frontier(EPOCH)
    kmin=min(len(s) for s in subsets(8) if common.frontier_vector(common.score(EPOCH,s))==allfront)
    result={'records':records,'traces':traces,'K_min_frontier':kmin,'J_empty_ms':J0,'cost_model':asdict(MODEL.metadata),'marconi_status':'NOT_COMPARABLE','marconi_reason':'B1未保存逐候选last_access_order或等价的冻结Marconi时序输入；census的物化顺序不能替代访问顺序。不得混入B1-C其他生命周期元数据。','scope':'单Snapshot、四个逻辑预算、CPU模型化评价，不是GPU性能或物理容量竞争'}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'matrix.json').write_text(json.dumps({'input':asdict(EPOCH),'edges':EDGES,'benefit_matrix_ms':BM},ensure_ascii=False,indent=2)+'\n')
    with (OUT/'allocation_results.csv').open('w') as f:
        fields=['K','strategy','selected','F_ms','J_ms','gap_vs_opt_ms','same_set_as_flowstate','tie_with_different_set','per_pending']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
        for k in range(1,5):w.writerow({'K':k,'strategy':'Marconi Adaptation','selected':'NOT_COMPARABLE'})
    with (OUT/'compatibility_benefit.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['checkpoint_id','pending_id','status','B1C_evidence_level','L','T','benefit_ms'])
        for e in EDGES:
            cid,pid=e['checkpoint_id'],e['continuation_id'];i=next(i for i,p in enumerate(EPOCH.pending) if p.identifier==pid)
            w.writerow([cid,pid,e['status'],e['B1C_evidence_level'],RAW[cid]['token_position_L'],EPOCH.pending[i].target,BM[cid][i]])
    assert before=={str(p):sha(p) for p in sources}
    (OUT/'input_integrity.json').write_text(json.dumps({'status':'PASS','sha256':before},ensure_ascii=False,indent=2)+'\n')
    for r in records:print(r['K'],r['strategy'],[c.replace('RQ3_G000_','') for c in r['selected']],round(r['J_ms'],9),round(r['gap_vs_opt_ms'],9))

if __name__=='__main__':main()
