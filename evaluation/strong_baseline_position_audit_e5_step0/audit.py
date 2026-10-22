"""只读冻结模型与快照，在独立CPU审计层比较决策，不重新拟合。"""
import csv,json,itertools,statistics,hashlib
from pathlib import Path
from dataclasses import replace
from evaluation.greedy_swap_structure_e3_step1 import audit as prior
ROOT=prior.ROOT;OUT=Path(__file__).resolve().parent;EPS=1e-9
SOURCES={}
def read(p):
 SOURCES[str(p)]=prior.sha(p);return json.loads(Path(p).read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(n,rows):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(tuple,list,dict)) else v for k,v in r.items()})
def relation(a,b):
 if set(a['selected'])==set(b['selected']):
  assert [p['E'] for p in a['per_pending']]==[p['E'] for p in b['per_pending']]
  return '同集同前沿'
 return '异集同前沿' if [p['E'] for p in a['per_pending']]==[p['E'] for p in b['per_pending']] else '异集异前沿'

def main():
 prior.manifest(ROOT/'evaluation/greedy_swap_structure_e3_step1');cases=prior.load()
 frozen=read(ROOT/'evaluation/greedy_swap_structure_e3_step1/results.json')['rows']
 base=ROOT/'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020'
 models=read(base/'candidate_models.json');read(base/'config.json')
 points=list(csv.DictReader((base/'calibration_points.csv').open()));SOURCES[str(base/'calibration_points.csv')]=prior.sha(base/'calibration_points.csv');assert len(points)==20
 knots=[(round(v['gap_ki']*1024),v['cost_ms']) for v in models['M0']['parameters']['knots']]
 for gap,cost in knots:
  if gap==0:assert cost==0;continue
  vals=[float(p['measured_phi_ms']) for p in points if int(p['gap_tokens'])==gap]
  assert abs(statistics.mean(vals)-cost)<EPS
 for p in points:
  path=ROOT/p['source'];SOURCES[str(path)]=prior.sha(path);raw=list(csv.DictReader(path.open()))
  match=[r for r in raw if int(r['gap_tokens'])==int(p['gap_tokens']) and (p['subset']=='long_gap' or int(r['target_position'])==int(p['target_tokens']))]
  assert len(match)==1 and abs(float(match[0]['measured_phi_ms'])-float(p['measured_phi_ms']))<EPS
 class GapModel:
  """执行冻结M0插值，禁止外推或根据分配结果调参。"""
  def estimate(self,g,t):
   assert 0<=g<=t<=131072
   for (a,x),(b,y) in zip(knots,knots[1:]):
    if a<=g<=b:return x+(y-x)*(g-a)/(b-a)
   raise ValueError('超出冻结插值域')
 model_pa=prior.common.MODEL;model_gap=GapModel();pairs=[];comparisons=[];outputs=[];ties=[]
 for name,base_ep,_ in cases:
  for k in range(1,5):
   ep=replace(base_ep,k=k);old={r['policy']:r for r in frozen if r['case']==name and r['K']==k}
   for r in old.values():
    sc=prior.common.score(ep,r['selected']);assert abs(sc['cost_ms']-r['J_ms'])<=EPS and sc['per_pending']==r['per_pending']
   for a,b in itertools.combinations(prior.POLICIES,2):pairs.append({'case':name,'K':k,'policy_a':a,'policy_b':b,'relation':relation(old[a],old[b]),'delta_J_a_minus_b':old[a]['J_ms']-old[b]['J_ms']})
   ids=tuple(sorted(c.identifier for c in ep.candidates))
   prior.common.MODEL=model_gap
   gap_funcs=prior.solvers(ep)
   gap_sets={p:gap_funcs[p]() for p in ('Exact','Greedy','Greedy+1-Swap','StandaloneTopK')}
   all_gap=[(prior.common.score(ep,s)['cost_ms'],s) for j in range(k+1) for s in itertools.combinations(ids,j)]
   min_gap=min(v for v,_ in all_gap);optimal=[s for v,s in all_gap if abs(v-min_gap)<=EPS]
   prior.common.MODEL=model_pa
   for policy,selected in gap_sets.items():
    pa=old[policy];gap_eval_pa=prior.common.score(ep,selected)
    prior.common.MODEL=model_gap;gap_eval=prior.common.score(ep,selected);pa_eval_gap=prior.common.score(ep,pa['selected']);prior.common.MODEL=model_pa
    gr={'selected':selected,'per_pending':gap_eval['per_pending']}
    row={'case':name,'K':k,'solver':policy,'relation':relation(pa,gr),'pa_selected':pa['selected'],'gap_selected':selected,'pa_frontiers':pa['per_pending'],'gap_frontiers':gap_eval['per_pending'],'pa_choice_J_pa':pa['J_ms'],'gap_choice_J_pa':gap_eval_pa['cost_ms'],'pa_choice_J_gap':pa_eval_gap['cost_ms'],'gap_choice_J_gap':gap_eval['cost_ms'],'selected_positions':{c.identifier:c.position for c in ep.candidates if c.identifier in set(pa['selected'])|set(selected)}}
    comparisons.append(row)
   ties.append({'case':name,'K':k,'gap_exact_ties':len(optimal),'pa_exact_also_gap_optimal':abs(min_gap-next(v for v,s in all_gap if set(s)==set(old['Exact']['selected'])))<=EPS,'gap_optimal_sets':optimal})
   outputs.extend(old.values())
 save('model_contract.json',{'source':str(base),'M0':models['M0'],'points_verified':20,'refitted':False,'solver_scope':['Exact','Greedy','Greedy+1-Swap','StandaloneTopK'],'说明':'只改变独立审计层成本接口，PCB/LDR/WFC沿用冻结结果；不改变生产文件'})
 table('policy_pair_matrix.csv',pairs);table('position_vs_gap.csv',comparisons);table('all_policy_results.csv',outputs);save('gap_exact_ties.json',ties)
 save('results.json',{'rows':outputs,'pair_matrix':pairs,'model_comparisons':comparisons,'ties':ties})
 SOURCES.update(prior.SOURCES)
 for p,h in SOURCES.items():assert prior.sha(p)==h,p
 save('source_integrity.json',{'status':'PASS','sha256':SOURCES,'unchanged':True})
 print('配对分类', {v:sum(x['relation']==v for x in pairs) for v in ['同集同前沿','异集同前沿','异集异前沿']})
 for p in ['Exact','Greedy','Greedy+1-Swap','StandaloneTopK']:
  print(p,[(x['case'],x['K'],x['relation']) for x in comparisons if x['solver']==p and x['relation']!='同集同前沿'])
 print('Exact同分',[(t['case'],t['K'],t['gap_exact_ties'],t['pa_exact_also_gap_optimal']) for t in ties])
if __name__=='__main__':main()
