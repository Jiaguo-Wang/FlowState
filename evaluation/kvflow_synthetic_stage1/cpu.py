"""合成规划域CPU对照；候选是假设物化的PROPOSED状态，不构造严格认证。"""
import json,csv,sys,hashlib
from pathlib import Path
from evaluation.kvflow_inspired_retention_cpu.policy import select,compute_ste
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate
from evaluation.sota_policies import MarconiStylePolicy
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'evaluation/p1_strong_baseline_validation'))
import common,wfc

def main():
 data=json.loads((OUT/'workload.json').read_text());rows=[];priorities=[]
 for cycle in data['cycles']:
  cs=cycle['candidates'];ps=cycle['pendings'];cd=cycle['configuration_digest'];eid=cycle['epoch_id']
  assert list(compute_ste(cycle['graph'],eid).values())==list(range(10))
  for k in data['protocol']['budgets']:
   ep=common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(p['continuation_id'] for p in ps if c['token_ids']==p['token_ids'][:c['token_position_L']])) for c in cs),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in ps),k)
   selected,pr=select(cs,cycle['graph'],cycle['declarations'],eid,cd,k)
   for x in pr:priorities.append({'epoch':eid,'K':k,**x})
   core=tuple(CheckpointCandidate(c['checkpoint_id'],c['checkpoint_id'],(),c['token_position_L'],1,True) for c in cs)
   greedy=GlobalOptimizer(common.MODEL)._select_candidates(core,k,lambda chosen:common.score(ep,tuple(c.checkpoint_id for c in chosen))['cost_ms']).selected
   recency={c['checkpoint_id']:c['last_access_order'] for c in cs}
   marconi=MarconiStylePolicy().select(core,k,recency,{c['checkpoint_id']:c['token_position_L'] for c in cs},1.0).selected_checkpoint_ids
   choices={'KVFlow-inspired':selected,'FlowState-Greedy':tuple(c.checkpoint_id for c in greedy),'WFC':wfc.select(ep)[0],'LRU':tuple(sorted(recency,key=lambda c:(-recency[c],c))[:k]),'Marconi Adaptation':marconi}
   ref=common.score(ep,selected)
   for name,s in choices.items():
    out=common.score(ep,s);delta=out['cost_ms']-ref['cost_ms']
    rows.append({'epoch':eid,'K':k,'policy':name,'selected':s,'J_ms':out['cost_ms'],'per_pending':out['per_pending'],'set_vs_kvflow':'SAME' if set(s)==set(selected) else 'DIFFERENT','WTL_vs_kvflow':'WIN' if delta< -1e-9 else 'LOSS' if delta>1e-9 else 'TIE','status':'CPU_PROPOSED','recency_source':'声明的两轮固定提示预热顺序；非GPU访问观测'})
   assert all(abs(r['J_ms']-ref['cost_ms'])<1e-9 for r in rows if r['epoch']==eid and r['K']==k)
 def table(n,rs):
  with (OUT/n).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader()
   for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
 table('cpu_comparison.csv',rows);table('candidate_priorities.csv',priorities)
 (OUT/'cpu_results.json').write_text(json.dumps({'rows':rows,'priorities':priorities,'说明':'等长互不共享固定前缀的条件性结果；未来顺序需求等权评价，不是同时Ready或真实GPU收益'},ensure_ascii=False,indent=2)+'\n')
 print([(r['K'],r['policy'],r['selected'],round(r['J_ms'],6)) for r in rows[:10]])
if __name__=='__main__':main()
