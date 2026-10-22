"""复核旧573条模型结构，不枚举大规模候选子集，不升级执行认证。"""
import csv,json,itertools
from .audit import ROOT,OUT,sha,save,csvout,EPS,common

def main():
 base=ROOT/'evaluation/gate1_step1_allocation_equivalence';paths=[base/n for n in ('gate1_step1_instance_statistics.csv','matrices.jsonl','summary.json','integrity.json','audit.py')];before={str(p):sha(p) for p in paths}
 stats=list(csv.DictReader(paths[0].open()));mat={r['instance_id']:r for r in map(json.loads,paths[1].read_text().splitlines())};assert len(stats)==len(mat)==573
 out=[]
 for r in stats:
  x=mat[r['instance_id']];ids=x['candidate_ids'];bm=x['benefit_ms'];n=len(ids);m=len(bm);k=int(r['budget']);supports=[{p for p in range(m) if bm[p][c]>0} for c in range(n)]
  assert sum(len(s)>1 for s in supports)==int(r['num_shared_checkpoints'])
  overlap=sum(bool(a&b) for a,b in itertools.combinations(supports,2));assert overlap==int(r['num_overlapping_candidate_pairs'])
  forced=set();front=[]
  for p in range(m):
   e=max([x['positions'][i] for i in range(n) if x['compatibility'][p][i]]+[0]);front.append(e)
   if e>0:
    real=[ids[i] for i in range(n) if x['compatibility'][p][i] and x['positions'][i]==e];assert len(real)==1;forced.update(real)
  assert len(forced)==int(r['k_min_frontier']) and forced==set(json.loads(r['unique_frontier_realizer_ids']))
  selected=json.loads(r['flowstate_selected_ids']);j0=sum(common.MODEL.estimate(t,t) for t in x['targets'])
  benefit=sum(max([bm[p][ids.index(c)] for c in selected]+[0.]) for p in range(m));j=j0-benefit
  assert abs(j-float(r['flowstate_cost_ms']))<EPS
  for key in ('pcb','ldr','wfc'):assert set(selected)==set(json.loads(r[key+'_selected_ids'])) and abs(j-float(r[key+'_cost_ms']))<EPS
  if all(len(s)<=1 for s in supports):
   optimum=sum(sorted((max(row,default=0) for row in bm),reverse=True)[:k]);certificate='SEPARABLE_SUPPORT_TOPK'
  else:
   assert k>=len(forced) and forced==set(selected);optimum=sum(max(row,default=0) for row in bm);certificate='ALL_CANDIDATE_FRONTIERS_ATTAINED'
  assert abs(benefit-optimum)<EPS
  out.append({'instance_id':r['instance_id'],'dataset':r['workload'],'snapshot_id':r['snapshot_id'],'n':n,'m':m,'K':k,'K_min':len(forced),'shared_candidates':sum(len(s)>1 for s in supports),'overlap_pairs':overlap,'tight_budget':k<len(forced),'selected':selected,'J_ms':j,'optimality_certificate':certificate,'greedy_optimal_in_legacy_model':True,'strict_runtime_certification':'NOT_CERTIFIED','source_file':r['source_file'],'source_line':r['source_line']})
 csvout('legacy_573_structure.csv',out)
 summary={}
 for pop in ('OpenHands','AgentX'):
  rr=[r for r in out if r['dataset']==pop];summary[pop]={'conditions':len(rr),'snapshots':len({r['snapshot_id'] for r in rr}),'tight_budget':sum(r['tight_budget'] for r in rr),'shared_conditions':sum(r['shared_candidates']>0 for r in rr),'overlap_conditions':sum(r['overlap_pairs']>0 for r in rr),'model_optimal':len(rr),'n_range':[min(r['n'] for r in rr),max(r['n'] for r in rr)]}
 assert all(sha(p)==h for p,h in before.items());save('legacy_summary.json',{'summary':summary,'source_sha256':before,'method':'复用结构并逐条件复核；可分支持TopK或达到全候选前沿给出最优证书；没有运行新Exact或旧策略','runtime_boundary':'旧Formal矩阵不是Strict Runtime认证，AgentX既有Formal-only问题未修复'})
 print(summary)
if __name__=='__main__':main()
