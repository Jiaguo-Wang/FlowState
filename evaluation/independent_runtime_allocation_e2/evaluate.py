"""固定严格输入的独立CPU比较；不具备Runtime控制能力。"""
import csv,hashlib,itertools,json,math,sys
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evaluation/p1_strong_baseline_validation'))
import common,pcb,ldr,wfc
from evaluation.exact_planner_contract_gate_b5d1 import planner as d
EPS=1e-9
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def evaluate(raw,origins=None):
 s=read(raw/'strict_epoch_snapshot.json');physical=read(raw/'physical_state_census.json');inputs=read(raw/'prepared_inputs.json')
 if origins is None:
  origins=[]
  histories=[read(p) for p in sorted(raw.glob('history_*.json'))]
  for c in s['candidates']:
   matches=[h for h in histories if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots']==c['state_identity']['target_mamba_slots'] for n in h['state']['before']['nodes'])]
   assert matches,'缺少候选首次观察记录'
   origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':matches[0]['request']['request_id'],'provenance':'PROVENANCE_UNVERIFIED'})
 ev=d.FrozenEvidence((raw/'strict_epoch_snapshot.json').read_bytes(),sha(raw/'strict_epoch_snapshot.json'),physical_snapshot=(raw/'physical_state_census.json').read_bytes(),physical_sha256=sha(raw/'physical_state_census.json'))
 n=len(s['candidates']);m=len(s['pendings']);rows=[];exact=[]
 for k in range(1,min(4,n)+1):
  ep=d.from_frozen(ev,k,(physical,origins,inputs));pr=d.prepare(ep,ev);bm=d.benefits(pr)
  ce=common.Epoch(tuple(common.Candidate(c.candidate.checkpoint_id,c.candidate.token_pos,tuple(p.pending_id for j,p in enumerate(ep.pendings) if pr.compatible[i][j])) for i,c in enumerate(ep.candidates)),tuple(common.Pending(p.pending_id,p.target) for p in ep.pendings),k)
  ids=tuple(sorted(pr.ids));count=sum(math.comb(n,i) for i in range(k+1));scored=[]
  if n<=24 and m<=4 and count<=20000:
   for z in range(k+1):
    for subset in itertools.combinations(ids,z):
     v=common.score(ce,subset)['cost_ms']; f=sum(max([bm[pr.ids.index(c)][j] for c in subset]+[0.]) for j in range(m))
     assert abs(sum(pr.costs)-f-v)<EPS
     assert abs(v-d.score(pr,subset))<EPS
     scored.append((v,subset))
  choices={'FlowState-Greedy':d.solve(pr,bm,'Greedy')};traces={}
  if scored:
   best=min(v for v,_ in scored);ties=[x for v,x in scored if abs(v-best)<=EPS];chosen=min(ties,key=lambda x:(len(x),x))
   if d.allowed(ep):assert d.solve(pr,bm,'Exact')==chosen
   choices['FlowState-Exact']=chosen
   exact.append({'K':k,'status':'ONLINE_DOMAIN_EXACT' if d.allowed(ep) else 'OFFLINE_EXACT','J_ms':best,'optimal_sets':ties,'enumerated':count})
  else:best=None;exact.append({'K':k,'status':'NOT_TRACTABLE','enumerated':0,'required':count})
  for name,mod in [('PCB',pcb),('LDR',ldr),('WFC',wfc)]:choices[name],traces[name]=mod.select(ce)
  rank=sorted(ids,key=lambda c:(-sum(bm[pr.ids.index(c)]),c));choices['StandaloneTopK']=tuple(c for c in rank if sum(bm[pr.ids.index(c)])>EPS)[:k]
  for name,selected in choices.items():
   plan=d.finish(pr,selected);score=common.score(ce,selected);assert abs(plan.recovery_cost_ms-score['cost_ms'])<EPS
   steps=[];prefix=[]
   for c in selected:
    steps.append({'checkpoint':c,'marginal_ms':d.score(pr,prefix)-d.score(pr,prefix+[c])});prefix.append(c)
   rows.append({'K':k,'policy':name,'selected':list(selected),'J_ms':score['cost_ms'],'gap_exact_ms':None if best is None else score['cost_ms']-best,'per_pending':score['per_pending'],'plan':asdict(plan),'steps':steps,'native_trace':traces.get(name)})
 supports={c:[ep.pendings[j].pending_id for j,v in enumerate(bm[i]) if v>0] for i,c in enumerate(pr.ids)}
 full=ep.frontiers(pr.ids);kmin=None
 for z in range(min(n,m)+1):
  if sum(math.comb(n,i) for i in range(z+1))>20000:break
  if any(ep.frontiers(x)==full for x in itertools.combinations(ids,z)):kmin=z;break
 return {'n':n,'m':m,'K_min':kmin,'supports':supports,'origins':origins,'benefits':dict(zip(pr.ids,bm)),'laminar':all(not(set(a)&set(b)) or set(a)<=set(b) or set(b)<=set(a) for a,b in itertools.combinations(supports.values(),2)),'rows':rows,'exact':exact}
def regression():
 raw=ROOT/'evaluation/multishare_runtime_gate_b5c3';orig=read(raw/'candidate_provenance.json')
 result=evaluate(raw,orig)
 old=read(ROOT/'evaluation/multishare_allocation_gate_b5c4/results.json')
 names={'FlowState':'FlowState-Greedy','Exact OPT':'FlowState-Exact'}
 for row in old['rows']:
  now=next(r for r in result['rows'] if r['K']==row['K'] and r['policy']==names.get(row['policy'],row['policy']))
  assert set(now['selected'])==set(row['selected']) and abs(now['J_ms']-row['J_ms'])<EPS
 save(OUT/'cpu_regression.json',{'status':'PASS','frozen_conditions':24,'source_sha256':sha(ROOT/'evaluation/multishare_allocation_gate_b5c4/results.json')})
if __name__=='__main__':regression()
