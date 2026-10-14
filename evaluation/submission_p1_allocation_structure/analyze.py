"""按冻结协议进行显式 token 矩阵的 CPU 情景分析，不调用 Runtime。"""
import csv, hashlib, itertools, json, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'evaluation/p1_strong_baseline_validation'),str(ROOT)]
from common import Candidate, Pending, Epoch, score, benefit_matrix
import pcb,ldr,wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate,is_compatible
from flowstate.workflow import PendingContinuation
MODEL=RecoveryCostModel();EPS=1e-9
FLAGS=dict(planning_target_source='ASSUMED_FULL_FA',physical_fa_evidence='NOT_VALIDATED',pending_event_evidence='UNKNOWN')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def greedy(e):
    """保持正式优化器的成本差、容差、ID 平局及非正停止规则。"""
    selected=[];trace=[];current=score(e,selected)['cost_ms']
    for _ in range(min(e.k,len(e.candidates))):
        best=None;ranking=[]
        for c in sorted(e.candidates,key=lambda c:c.identifier):
            if c.identifier in selected:continue
            after=score(e,selected+[c.identifier])['cost_ms'];gain=current-after
            if gain < -EPS:raise ValueError('负边际收益')
            gain=max(gain,0.)
            ranking.append(dict(checkpoint_id=c.identifier,gain_ms=gain))
            if best is None or gain>best[1]+EPS or (abs(gain-best[1])<=EPS and c.identifier<best[0]):best=(c.identifier,gain,after)
        if best is None or best[1]<=0:break
        trace.append(dict(prefix=list(selected),ranking=ranking,chosen=best[0]));selected.append(best[0]);current=best[2]
    return tuple(selected),trace

def check_adapter():
    """手工合法血缘例仅用于代码一致性，不为 LMCache 编造身份。"""
    checks=[]
    for shared in [False,True]:
        pp=[PendingContinuation('P1','test',('r','a'),2048,2048),PendingContinuation('P2','test',('r','b'),3072,3072)]
        cc=[CheckpointCandidate('C0','test',('r',) if shared else ('r','a'),512,1),CheckpointCandidate('C1','test',('r','a'),1024,1),CheckpointCandidate('C2','test',('r','b'),1024,1)]
        e=Epoch(tuple(Candidate(c.checkpoint_id,c.token_pos,tuple(p.continuation_id for p in pp if is_compatible(c,p))) for c in cc),tuple(Pending(p.continuation_id,p.planning_target) for p in pp),0)
        for k in range(4):
            selected,_=greedy(replace(e,k=k));native=GlobalOptimizer(MODEL).select(pp,cc,k)
            assert selected==tuple(c.checkpoint_id for c in native.selected)
            assert abs(score(replace(e,k=k),selected)['cost_ms']-native.recovery_cost_after_ms)<=EPS
            checks.append(dict(shared=shared,k=k,status='PASS',selected=selected))
    return checks

def run():
    lock=read(OUT/'protocol_lock.json');mp=Path(lock['input_manifest']);assert sha(mp)==lock['input_manifest_sha256']
    before=read(OUT/'source_integrity_before.json')
    for p,h in before.items():assert sha(p)==h,p
    adapter=check_adapter();records=[json.loads(s) for s in mp.read_text().splitlines()]
    assert {r['case_id'] for r in records}=={f'lmcache_{i:02d}' for i in range(1,17)}
    cms={};bms={};candidates=[];budgets=[];policies=[];details=[];cases=[]
    for r in sorted(records,key=lambda r:(r['case_id']!='lmcache_10',r['case_id'])):
        cid=r['case_id'];group='完整消息共享组' if cid in ['lmcache_10','lmcache_12','lmcache_13','lmcache_16'] else '部分消息共享组'
        assert [sha(p) for p in r['input_artifact_paths']]==r['input_sha256']
        data=[read(p) for p in r['input_artifact_paths']];ids=[x['input_ids'] for x in data];n=list(map(len,ids));a,b=ids
        lcp=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(n));assert lcp==r['prefix_tokens']
        c0=lcp//256*256
        if r['checkpoint_candidate_positions'] != [c0]:
            cases.append(dict(case_id=cid,status='PROTOCOL_CONFLICT',**FLAGS));continue
        positions=[c0,(lcp+n[0])//512*256,(lcp+n[1])//512*256];source_indices=[0,0,1]
        cc=[];cm={}
        for j,(pos,source) in enumerate(zip(positions,source_indices)):
            name=f'C{j}';valid=pos>0 if j==0 else lcp<pos<n[source]
            row=dict(case_id=cid,checkpoint_id=name,position=pos,source_input=r['input_artifact_paths'][source],source_sha256=r['input_sha256'][source],status='HISTORIC_A5_C0_CERTIFIED' if cid=='lmcache_10' and j==0 else 'PROPOSED',included=valid,reason='满足固定位置规则' if valid else '固定中点对齐后不在分歧与输入末端之间，不替补',GPU_CONTEXT_FEASIBLE=max(n)<=45056,**FLAGS)
            if valid:
                prefix=ids[source][:pos];edges={}
                for i,t in enumerate(n):
                    yes=pos<=t and prefix==ids[i][:pos]
                    edges[f'P{i+1}']=dict(token_compatible=yes,position=pos,T=t,same_target_identity=True,source_traceable=True,input_source=r['input_artifact_paths'][i],evidence='TOKEN_COMPATIBLE' if yes else 'TOKEN_PREFIX_MISMATCH',**FLAGS)
                cm[name]=edges;cc.append(Candidate(name,pos,tuple(p for p,v in edges.items() if v['token_compatible'])))
                row['prefix_sha256']=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
            candidates.append(row)
        e=Epoch(tuple(cc),tuple(Pending(f'P{i+1}',t) for i,t in enumerate(n)),len(cc));cms[cid]=cm
        subsets=[s for k in range(len(cc)+1) for s in itertools.combinations([c.identifier for c in cc],k)]
        def frontier(s):return tuple(max([c.position for c in cc if c.identifier in s and p.identifier in c.compatible]+[0]) for p in e.pending)
        full=frontier(tuple(c.identifier for c in cc));kmin=min(len(s) for s in subsets if frontier(s)==full)
        shared=[c.identifier for c in cc if len(c.compatible)>1]
        overlaps=[(x.identifier,y.identifier,sorted(set(x.compatible)&set(y.compatible))) for x,y in itertools.combinations(cc,2) if set(x.compatible)&set(y.compatible)]
        try:
            bm=benefit_matrix(e)
            for c in cc:
                for p,v in zip(e.pending,bm[c.identifier]):
                    assert v>=0
                    if p.identifier in c.compatible:assert MODEL.derivative_ms_per_ki_token(p.target-c.position,p.target)>0 and v>0
        except (ValueError,AssertionError) as exc:
            cases.append(dict(case_id=cid,status='COST_MODEL_UNAVAILABLE',reason=str(exc),**FLAGS));continue
        bms[cid]={c.identifier:{p.identifier:dict(benefit_ms=v,T=p.target,L=c.position,G=p.target-c.position if p.identifier in c.compatible else p.target,compatible=p.identifier in c.compatible,empty_cost_ms=MODEL.estimate(p.target,p.target),effective_cost_ms=MODEL.estimate(p.target-c.position if p.identifier in c.compatible else p.target,p.target),**FLAGS) for p,v in zip(e.pending,bm[c.identifier])} for c in cc}
        cases.append(dict(case_id=cid,status='PASS',group=group,N1=n[0],N2=n[1],LCP=lcp,common_messages=r['shared_complete_messages'],candidate_count=len(cc),positive_edges=sum(v>0 for vs in bm.values() for v in vs),shared=shared,overlaps=overlaps,K_min=kmin,tight_budget_exists=kmin>1,GPU_CONTEXT_FEASIBLE=max(n)<=45056,**FLAGS))
        for k in range(len(cc)+1):
            ek=replace(e,k=k);legal=[s for s in subsets if len(s)<=k];values={s:score(ek,s)['cost_ms'] for s in legal};opt=min(values.values());opts=sorted(s for s,v in values.items() if abs(v-opt)<=EPS)
            fs,trace=greedy(ek)
            outputs={'FlowState':(fs,trace),'PCB':pcb.select(ek),'LDR':ldr.select(ek),'WFC':wfc.select(ek)}
            ranked=sorted([(c.identifier,sum(bm[c.identifier])) for c in cc if sum(bm[c.identifier])>EPS],key=lambda x:(-x[1],x[0]))
            outputs['StandaloneTopK']=(tuple(sorted(c for c,v in ranked[:k])),dict(ranking=ranked));outputs['Exact OPT']=(opts[0],dict(all_optimal_sets=opts))
            budgets.append(dict(case_id=cid,group=group,K=k,candidate_count=len(cc),K_min=kmin,tight=1<=k<kmin,all_candidate_frontiers=json.dumps(full),irreducible_residual=json.dumps([t-f for t,f in zip(n,full)]),GPU_CONTEXT_FEASIBLE=max(n)<=45056,**FLAGS))
            for name,(selected,trace) in outputs.items():
                result=score(ek,selected);cost=result['cost_ms'];empty=score(ek,())['cost_ms']
                per=[dict(x,benefit_ms=MODEL.estimate(x['T'],x['T'])-x['cost_ms']) for x in result['per_pending']]
                policies.append(dict(case_id=cid,group=group,K=k,policy=name,selected_set=json.dumps(sorted(selected)),selection_order=json.dumps(selected),cost_ms=cost,benefit_ms=empty-cost,gap_to_opt_ms=cost-opt,per_pending=json.dumps(per,ensure_ascii=False),GPU_CONTEXT_FEASIBLE=max(n)<=45056,**FLAGS))
                details.append(dict(case_id=cid,K=k,policy=name,trace=trace,**FLAGS))
    save('compatibility_matrix.json',cms);save('benefit_matrix.json',bms);save('case_structure.json',cases);save('selection_traces.json',details)
    with (OUT/'candidate_manifest.jsonl').open('w') as f:
        for r in candidates:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    csvsave('budget_structure_analysis.csv',budgets);csvsave('policy_comparison.csv',policies)
    stats={}
    for group in ['全部','完整消息共享组','部分消息共享组']:
        subset=[r for r in policies if group=='全部' or r['group']==group];keys=sorted({(r['case_id'],r['K']) for r in subset});lookup={(r['case_id'],r['K'],r['policy']):r for r in subset};s={}
        for rival in ['PCB','LDR','WFC','StandaloneTopK','Exact OPT']:
            wins=[];ties=[];losses=[]
            for cid,k in keys:
                d=lookup[cid,k,rival]['cost_ms']-lookup[cid,k,'FlowState']['cost_ms']
                (wins if d>EPS else losses if d < -EPS else ties).append(dict(case_id=cid,K=k,difference_ms=d))
            s[rival]=dict(W=len(wins),T=len(ties),L=len(losses),wins=wins,losses=losses,ties=ties)
        stats[group]=s
    save('paired_summary.json',stats)
    assert all(sha(p)==h for p,h in before.items())
    save('validation.json',dict(status='PASS',adapter_tests=adapter,source_files_unchanged=len(before),case_count=len(cases),case_failures=[c for c in cases if c['status']!='PASS'],policy_rows=len(policies),gpu_executed=False,protocol_sha256=sha(OUT/'protocol_lock.json')))
    print(json.dumps({g:{p:{k:v[k] for k in ['W','T','L']} for p,v in s.items()} for g,s in stats.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
