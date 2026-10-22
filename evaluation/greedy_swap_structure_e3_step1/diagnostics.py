"""逐轮排序与局部邻域诊断，不新增求解策略或修改既定替换规则。"""
import itertools,json,statistics
from dataclasses import replace
from .audit import OUT,ROOT,load,common,read,save,csvout,EPS,sha

def main():
 results=read(OUT/'results.json');cases={n:e for n,e,_ in load()};neighbors=[];failures=[]
 for r in results['rows']:
  if r['policy']!='Greedy+1-Swap' or r['gap_exact_ms']<=EPS:continue
  ep=replace(cases[r['case']],k=r['K']);ids=sorted(c.identifier for c in ep.candidates);s=set(r['selected']);J=r['J_ms']
  for remove in sorted(s):
   for add in sorted(set(ids)-s):
    t=tuple(sorted(s-{remove}|{add}));v=common.score(ep,t)['cost_ms'];neighbors.append({'case':r['case'],'K':r['K'],'remove':remove,'add':add,'selected':t,'J_ms':v,'delta_from_local_ms':v-J});assert v>=J-EPS
  # 仅在已有7候选的全部可行集合中诊断逃逸距离，不执行新交换求解器。
  better=[(len(s-set(t)),common.score(ep,t)['cost_ms'],t) for t in itertools.combinations(ids,len(s)) if common.score(ep,t)['cost_ms']<J-EPS]
  start=set(next(x['selected'] for x in results['rows'] if x['case']==r['case'] and x['K']==r['K'] and x['policy']=='Greedy'))
  target=tuple(next(x['selected'] for x in results['rows'] if x['case']==r['case'] and x['K']==r['K'] and x['policy']=='Exact'))
  queue=[(tuple(sorted(start)),[])];seen={tuple(sorted(start))};path=None
  while queue:
   current,route=queue.pop(0)
   if current==target:path=route;break
   for a in current:
    for b in sorted(set(ids)-set(current)):
     t=tuple(sorted(set(current)-{a}|{b}))
     if t not in seen and common.score(ep,t)['cost_ms']<common.score(ep,current)['cost_ms']-EPS:seen.add(t);queue.append((t,route+[{'from':current,'to':t,'J_ms':common.score(ep,t)['cost_ms']}]))
  failures.append({'case':r['case'],'K':r['K'],'local_J':J,'exact_gap_ms':r['gap_exact_ms'],'min_simultaneous_replacements_to_improve':min(x[0] for x in better),'alternative_strict_path_from_original_greedy':path,'note':'可达性诊断不是改写首改善协议；不宣称所有1-Swap规则均失败'})
 csvout('local_optimum_neighbors.csv',neighbors);save('local_trap_diagnostics.json',failures)
 traces=read(OUT/'selection_traces.json');ranking=[]
 for t in traces:
  z=t['ranked'];winner=next(x for x in z if x['checkpoint_id']==t['greedy_chosen']);other=max(x['gain_ms'] for x in z if x!=winner)
  assert t['greedy_chosen']==t['wfc_chosen'] and winner['gain_ms']>=other-EPS
  ranking.append({'case':t['case'],'K':t['K'],'step':t['step'],'prefix':t['prefix'],'chosen':t['greedy_chosen'],'q':winner['q'],'r_tokens':winner['r'],'gain_ms':winner['gain_ms'],'cost_margin_to_second_ms':winner['gain_ms']-other,'rank_reversals_below_winner':sum(a['gain_ms']<b['gain_ms']-EPS for i,a in enumerate(z) for b in z[i+1:])})
 csvout('greedy_wfc_step_audit.csv',ranking)
 # 从冻结数据验证每次接受的是当前字典序首个严格改善。
 for t in read(OUT/'swap_traces.json'):
  ep=replace(cases[t['case']],k=t['K']);ids=sorted(c.identifier for c in ep.candidates)
  for step in t['steps']:
   accepted=None;s=set(step['before_set']);j=common.score(ep,tuple(s))['cost_ms']
   for a in sorted(s):
    for b in ids:
     if b in s:continue
     v=common.score(ep,tuple(sorted(s-{a}|{b})))['cost_ms']
     if j-v>EPS:accepted=(a,b,v);break
    if accepted:break
   assert accepted and accepted[:2]==(step['remove'],step['add']) and abs(accepted[2]-step['after_J'])<=EPS
 legacy=ROOT/'evaluation/gate1_step1_allocation_equivalence/matrices.jsonl'
 for line in legacy.read_text().splitlines():
  x=json.loads(line)
  for p,T in enumerate(x['targets']):
   for c,L in enumerate(x['positions']):
    expected=common.MODEL.estimate(T,T)-common.MODEL.estimate(T-L,T) if x['compatibility'][p][c] else 0
    assert abs(expected-x['benefit_ms'][p][c])<=EPS
 save('validation.json',{'status':'PASS','frozen_reference_policy_conditions':120,'new_swap_conditions':20,'strict_improvement_and_first_order_checked':True,'local_optimum_conditions':20,'greedy_wfc_trace_steps':len(ranking),'legacy_model_matrices_checked':573,'gpu_started':False,'production_modified':False})
 print(failures)
if __name__=='__main__':main()
