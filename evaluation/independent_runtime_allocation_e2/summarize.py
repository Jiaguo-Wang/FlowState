"""汇总全部预注册组，失败不补位；保留逐条件及逐边证据。"""
import csv,itertools,json
from .evaluate import OUT,ROOT,read,save,sha,evaluate,EPS

def table(name,rows):
 if not rows:return
 with (OUT/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def main():
 material=[];edges=[];alloc=[];results={};checks={}
 for group in ('w1','w2','w3','w4'):
  raw=OUT/group
  manifest={str(p.relative_to(raw)):sha(p) for p in raw.rglob('*') if p.is_file() and 'triton_cache' not in p.parts}
  save(raw/'capture_manifest.json',manifest)
  if not (raw/'strict_epoch_snapshot.json').exists():
   material.append(dict(group=group,status='FAILED_OR_NOT_EXECUTED',candidates=None,pending=None,admissible=None,rejected=None,unknown=None,shared_candidates=None,K_min=None,cleanup=read(raw/'gpu_cleanup.json')['status'] if (raw/'gpu_cleanup.json').exists() else 'UNKNOWN'));continue
  s=read(raw/'strict_epoch_snapshot.json');p=read(raw/'physical_state_census.json')
  counts={v:sum(e['status']==v for e in s['edges']) for v in ('EPOCH_ADMISSIBLE','REJECTED','UNKNOWN')}
  for e in s['edges']:edges.append({'group':group,'epoch_id':s['epoch_id'],**e})
  checks[group]={'physical_unchanged':p['unchanged'],'pending_dispatches':p['pending_dispatches'],'destructive_calls':p['destructive_calls'],'cleanup':read(raw/'gpu_cleanup.json'),'raw_status':p['status'],'provenance':'PROVENANCE_UNVERIFIED','pending_resume_validated':False,'resident_outside_candidate_set':p['resident_outside_candidate_set'],'native_eviction_events':p['native_eviction_events'],'pool_accounting':p['audit_before']['state']['accounting'],'candidate_mapping':[{ 'checkpoint_id':c['checkpoint_id'],'position':c['token_position_L'],'node':c['state_identity']['node_id'],'slots':c['state_identity']['target_mamba_slots'],'components':c['component_ids']} for c in s['candidates']],'planning_targets':{x['continuation_id']:x['planning_target_T'] for x in s['pendings']},'fa_observations':p['fa_observations']}
  try:
   assert p['unchanged'] and counts['UNKNOWN']==0 and p['ok']
   result=evaluate(raw);save(raw/'allocation_results.json',result);results[group]=result
   status='READY'
  except Exception as exc:
   checks[group]['allocation_failure']=repr(exc);result=None;status='BLOCKED'
  material.append(dict(group=group,status=status,candidates=len(s['candidates']),pending=len(s['pendings']),admissible=counts['EPOCH_ADMISSIBLE'],rejected=counts['REJECTED'],unknown=counts['UNKNOWN'],shared_candidates=sum(len(v)>1 for v in result['supports'].values()) if result else None,K_min=result['K_min'] if result else None,cleanup=checks[group]['cleanup']['status']))
  if not result:continue
  for r in result['rows']:
   g=next(x for x in result['rows'] if x['K']==r['K'] and x['policy']=='FlowState-Greedy')
   alloc.append({'group':group,'K':r['K'],'policy':r['policy'],'selected':r['selected'],'J_ms':r['J_ms'],'gap_exact_ms':r['gap_exact_ms'],'per_pending':r['per_pending'],'same_set_as_greedy':set(r['selected'])==set(g['selected']),'same_objective_as_greedy':abs(r['J_ms']-g['J_ms'])<=EPS,'same_EG_as_greedy':[(p['E'],p['G']) for p in r['per_pending']]==[(p['E'],p['G']) for p in g['per_pending']],'policy_vs_greedy':'WIN' if r['J_ms']<g['J_ms']-EPS else 'LOSS' if r['J_ms']>g['J_ms']+EPS else 'TIE','selection_steps':r['steps']})
  assert all(sha(raw/n)==h for n,h in manifest.items())
 table('runtime_materialization_summary.csv',material);table('strict_compatibility_matrix.csv',edges);table('allocation_comparison.csv',alloc);save(OUT/'runtime_correctness.json',checks)
 save(OUT/'aggregate_results.json',{'materialization':material,'results':results})
 print(json.dumps(material,ensure_ascii=False,indent=2))
 for group,r in results.items():
  print(group,'K_min',r['K_min'],'supports',r['supports'])
  for k in range(1,5):print(k,[(x['policy'],round(x['J_ms'],6),x['selected']) for x in r['rows'] if x['K']==k])
if __name__=='__main__':main()
