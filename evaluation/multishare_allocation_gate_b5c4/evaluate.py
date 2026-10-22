"""读取冻结多共享快照，以历史锁定策略和独立穷举执行CPU审计。"""
import csv,hashlib,itertools,json,subprocess,sys,types
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RAW=ROOT/'evaluation/multishare_runtime_gate_b5c3';B4=ROOT/'evaluation/strict_compatibility_b4c';BASE=ROOT/'evaluation/p1_strong_baseline_validation'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import common,pcb,ldr,wfc
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
EPS=1e-9
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 sources[str(p)]=sha(p)
 return json.loads(p.read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 for base in (RAW,B4):
  manifest=read(base/'evidence_manifest.json')
  for name,h in manifest['files'].items():
   p=Path(name) if Path(name).is_absolute() else base/name
   assert sha(p)==h,str(p)
   sources[str(p)]=h
 lock=read(B4/'input_integrity.json')['sha256']
 for p in [BASE/n for n in ('common.py','pcb.py','ldr.py','wfc.py')]+[ROOT/'flowstate/recovery_model.py']:
  assert sha(p)==lock[str(p)],str(p)
  sources[str(p)]=sha(p)
 legacy=subprocess.check_output(['git','show','42e494026b0d39398ea0d2629d324690ae65b383:flowstate/optimizer.py'],cwd=ROOT)
 assert hashlib.sha256(legacy).hexdigest()==lock[str(ROOT/'flowstate/optimizer.py')]
 (OUT/'frozen_optimizer.py').write_bytes(legacy)
 module=types.ModuleType('flowstate.b5c4_frozen_optimizer');module.__package__='flowstate';sys.modules[module.__name__]=module
 exec(compile(legacy,str(OUT/'frozen_optimizer.py'),'exec'),module.__dict__)
 model=RecoveryCostModel();assert asdict(model.metadata)==read(B4/'results.json')['cost_model']
 s=read(RAW/'strict_epoch_snapshot.json');physical=read(RAW/'physical_state_census.json');prior=read(RAW/'sharing_support_matrix.json')
 cs=s['candidates'];ps=s['pendings'];assert len(cs)==7 and len(ps)==4
 assert len({c['checkpoint_id'] for c in cs})==7 and len({p['continuation_id'] for p in ps})==4
 edges={(e['checkpoint_id'],e['continuation_id']):e for e in s['edges']};assert len(edges)==len(s['edges'])==28
 counts={v:sum(e['status']==v for e in edges.values()) for v in ('EPOCH_ADMISSIBLE','REJECTED','UNKNOWN')}
 assert counts=={'EPOCH_ADMISSIBLE':13,'REJECTED':15,'UNKNOWN':0}
 sizes={};allslots=[]
 for c in cs:
  assert c['barrier_id']==s['epoch_id'] and c['configuration_digest']==s['configuration_digest']
  assert len(c['token_ids'])==c['token_position_L']
  node=next(n for n in physical['before']['nodes'] if n['node_id']==c['state_identity']['node_id'])
  assert node['conv_complete'] and node['temporal_complete'] and len(node['slots'])==1
  allslots+=node['slots'];assert len(node['component_ids'])==2
  sizes[c['checkpoint_id']]=sum(size for cid in node['component_ids'] for _,size in physical['audit_before']['component_slices'][cid]['byte_ranges'])
  for p in ps:
   assert p['barrier_id']==s['epoch_id'] and p['configuration_digest']==s['configuration_digest']
   e=edges[c['checkpoint_id'],p['continuation_id']]
   assert e['conditions']['exact_token_prefix']==(c['token_ids']==p['token_ids'][:c['token_position_L']])
   assert e['conditions']['L_le_T']==(c['token_position_L']<=p['planning_target_T'])
   if e['status']=='EPOCH_ADMISSIBLE':assert all(v is True for v in e['conditions'].values())
 assert len(set(sizes.values()))==1 and len(set(allslots))==7
 size=next(iter(sizes.values()));assert size>0
 epoch=common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(p['continuation_id'] for p in ps if edges[c['checkpoint_id'],p['continuation_id']]['status']=='EPOCH_ADMISSIBLE')) for c in cs),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in ps),7)
 ids=tuple(sorted(c.identifier for c in epoch.candidates));bm=common.benefit_matrix(epoch)
 for row in prior['rows']:
  for i,p in enumerate(ps):assert abs(bm[row['checkpoint_id']][i]-row['standalone_positive_evidence_ms'][p['label']])<=EPS
 assert all(v>=0 for values in bm.values() for v in values)
 def f(chosen):return sum(max([bm[c][i] for c in chosen]+[0.]) for i in range(4))
 j0=common.score(epoch,())['cost_ms']
 subsets=[x for k in range(8) for x in itertools.combinations(ids,k)]
 assert len(subsets)==128
 scores={x:common.score(epoch,x) for x in subsets}
 for x,score in scores.items():assert abs(j0-f(x)-score['cost_ms'])<=EPS
 frontier=common.all_frontier(epoch)
 kmin=min(len(x) for x in subsets if common.frontier_vector(scores[x])==frontier)
 core=tuple(CheckpointCandidate(c.identifier,'仅作模型对象占位，不用于兼容判断',(),c.position,size,True) for c in epoch.candidates)
 class MatrixOptimizer(module.GlobalOptimizer):
  """沿用B4-C成本接口适配，严格边直接决定评分，不调用旧兼容谓词。"""
  def _recovery_cost(self,continuations,selected):return common.score(epoch,tuple(c.checkpoint_id for c in selected))['cost_ms']
 rows=[];traces={};exact=[]
 for k in range(1,5):
  current=replace(epoch,k=k)
  best=min(scores[x]['cost_ms'] for x in subsets if len(x)<=k)
  ties=[x for x in subsets if len(x)<=k and abs(scores[x]['cost_ms']-best)<=EPS]
  selections={'FlowState':tuple(c.checkpoint_id for c in MatrixOptimizer(model).select((),core,k*size).selected),'Exact OPT':min(ties,key=lambda x:(len(x),x))}
  for name,mod in [('PCB',pcb),('LDR',ldr),('WFC',wfc)]:selections[name],traces[f'{k}/{name}/native']=mod.select(current)
  ranked=sorted(ids,key=lambda c:(-sum(bm[c]),c))
  selections['StandaloneTopK']=tuple(sorted([c for c in ranked if sum(bm[c])>EPS][:k]))
  for name,selected in selections.items():
   assert len(set(selected))==len(selected)<=k
   order=selected if name!='StandaloneTopK' else tuple(c for c in ranked if c in selected)
   steps=[];prefix=[]
   for c in order:
    gains={x:common.score(epoch,prefix)['cost_ms']-common.score(epoch,prefix+[x])['cost_ms'] for x in ids if x not in prefix}
    if name=='FlowState':
     assert max(gains.values())-gains[c]<=EPS
     assert c==min(x for x,v in gains.items() if abs(v-max(gains.values()))<=EPS)
    steps.append({'prefix':prefix[:],'chosen':c,'marginals_ms':gains,'selected_marginal_ms':gains[c]});prefix.append(c)
   traces[f'{k}/{name}/evaluation']=steps
   score=common.score(current,selected)
   delta=score['cost_ms']-common.score(epoch,selections['FlowState'])['cost_ms']
   removal={c:common.score(epoch,tuple(x for x in selected if x!=c))['cost_ms']-score['cost_ms'] for c in selected}
   swaps={c:{x:score['cost_ms']-common.score(epoch,tuple(v for v in selected if v!=c)+(x,))['cost_ms'] for x in ids if x not in selected} for c in selected}
   rows.append({'K':k,'policy':name,'selected':list(selected),'selection_order':list(order),'J_ms':score['cost_ms'],'F_ms':f(selected),'gap_to_exact_ms':score['cost_ms']-best,'per_pending':score['per_pending'],'baseline_minus_flowstate_ms':delta,'FlowState_WTL':'W' if delta>EPS else 'L' if delta< -EPS else 'T','removal_contribution_ms':removal,'replacement_diagnostics_ms':swaps})
  exact.append({'K':k,'J_opt_ms':best,'all_optimal_sets':ties,'subsets_within_budget':sum(len(x)<=k for x in subsets)})
 supports={c:tuple(ps[i]['label'] for i,v in enumerate(bm[c]) if v>0) for c in ids}
 dominance=[{'dominator':a,'dominated':b} for a in ids for b in ids if a!=b and all(x>=y-EPS for x,y in zip(bm[a],bm[b])) and any(x>y+EPS for x,y in zip(bm[a],bm[b]))]
 equal=[(a,b) for a,b in itertools.combinations(ids,2) if all(abs(x-y)<=EPS for x,y in zip(bm[a],bm[b]))]
 laminar=all(not(set(a)&set(b)) or set(a)<=set(b) or set(b)<=set(a) for a,b in itertools.combinations(supports.values(),2))
 save('results.json',{'gate':'READY','cost_model':asdict(model.metadata),'input':asdict(epoch),'size_bytes':sizes,'edge_counts':counts,'benefit_matrix_ms':bm,'K_min':kmin,'all_frontiers':frontier,'supports':supports,'dominance':dominance,'equal_benefits':equal,'laminar':laminar,'rows':rows,'traces':traces})
 save('exact_verification.json',{'status':'PASS','subsets_enumerated':128,'J_F_equivalence_checked':128,'K_min':kmin,'all_frontiers':frontier,'exact':exact,'all_subsets':[{'selected':x,'J_ms':scores[x]['cost_ms'],'F_ms':f(x),'per_pending':scores[x]['per_pending']} for x in subsets]})
 with (OUT/'policy_comparison.csv').open('w') as fcsv:
  w=csv.DictWriter(fcsv,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
 assert all(sha(Path(p))==h for p,h in sources.items())
 save('input_integrity.json',{'status':'PASS','sources_sha256':sources,'historical_optimizer_commit':'42e494026b0d39398ea0d2629d324690ae65b383','historical_optimizer_sha256':hashlib.sha256(legacy).hexdigest(),'equal_checkpoint_bytes':size,'source_unchanged':True})
 for k in range(1,5):print(k,[(r['policy'],r['selected'],round(r['J_ms'],9),round(r['gap_to_exact_ms'],9)) for r in rows if r['K']==k])
 print('K_min',kmin,'dominance',dominance,'equal',equal,'bytes',size)
if __name__=='__main__':main()
