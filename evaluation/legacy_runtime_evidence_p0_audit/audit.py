"""核对旧版216个生命周期与原始TTFT，不启动Runtime或改写历史。"""
import json,csv,hashlib,statistics,itertools
from pathlib import Path
from collections import defaultdict
from evaluation import rq4_snapshot_level_analysis as original
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652'
HASH={}
def read(p):
 HASH[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def lines(p):
 HASH[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(n,rows):
 with (OUT/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 plan=read(BASE/'run_plan.json')['runs'];assert len(plan)==216
 assert set(x['run_id'] for x in plan)==set(p.name for p in (BASE/'runs').iterdir() if p.is_dir())
 ledger=[];groups=defaultdict(list);unmapped=[];meta={};missing=[]
 indexed={r['run_id']:r for r in lines(BASE/'runs.jsonl')};assert len(indexed)==216
 for item in plan:
  d=BASE/'runs'/item['run_id']
  required=['correctness.json','selection.json','request_telemetry.jsonl','replay.json','runtime.log','result.json','pending_requests.jsonl','handle_mapping.json','reconcile.json']
  miss=[n for n in required if not (d/n).is_file() or (d/n).stat().st_size==0];missing.extend((item['run_id'],n) for n in miss)
  result=read(d/'result.json');corr=read(d/'correctness.json');selection=read(d/'selection.json');replay=read(d/'replay.json');rec=read(d/'reconcile.json');requests=lines(d/'pending_requests.jsonl');telemetry=lines(d/'request_telemetry.jsonl')
  assert result==indexed[item['run_id']]
  assert corr['status']=='PASS' and all(corr['checks'].values()) and result['status']=='PASS' and len(requests)==4
  assert requests==result['pending_requests'] and len(telemetry)==4
  assert selection['selected_candidate_ids']==item['selected_candidate_ids'] and result['logical_k']==item['logical_k']==max(1,int(.25*item['candidate_count']))
  assert selection['snapshot_digest_before']==selection['snapshot_digest_after']==item['snapshot_digest']
  assert result['gpu_wait_before']['stable'] and result['gpu_wait_after']['stable'] and not result['shutdown_error'] and result['worker_exit_code']==0
  worker=BASE/'workers'/Path(result['worker_log']).name;assert worker.name==item['run_id']+'.log' and worker.is_file() and worker.stat().st_size
  frozen=read(Path(item['snapshot_artifact']));snap=json.loads(frozen['canonical_snapshot']);assert hashlib.sha256(frozen['canonical_snapshot'].encode()).hexdigest()==item['snapshot_digest']
  meta[item['snapshot_id']]=snap
  if item['policy']=='LRU':
   md={m['checkpoint_id']:m for m in snap['candidate_metadata']}
   expected=sorted(snap['eligible_candidates'],key=lambda c:(-md[c['checkpoint_id']]['last_access_order'],-md[c['checkpoint_id']]['creation_order'],c['checkpoint_id']))[:item['logical_k']]
   assert [c['checkpoint_id'] for c in expected]==selection['selected_candidate_ids']
  if item['policy']=='Marconi':
   cs=snap['eligible_candidates'];md={m['checkpoint_id']:m for m in snap['candidate_metadata']};alpha=snap['marconi_alpha']
   recency={c['checkpoint_id']:md[c['checkpoint_id']]['last_access_order'] for c in cs};eff={c['checkpoint_id']:md[c['checkpoint_id']]['marconi_flop_saved']/c['memory_bytes'] for c in cs}
   norm=lambda v,k:0 if max(v.values())==min(v.values()) else (v[k]-min(v.values()))/(max(v.values())-min(v.values()))
   expected=sorted(recency,key=lambda c:(-(norm(recency,c)+alpha*norm(eff,c)),c))[:item['logical_k']]
   assert expected==selection['selected_candidate_ids']
   for c in cs:
    parents=[z['token_pos'] for z in cs if z['workflow_id']==c['workflow_id'] and z['token_pos']<c['token_pos'] and c['lineage_path'][:len(z['lineage_path'])]==z['lineage_path']]
    assert md[c['checkpoint_id']]['marconi_flop_saved']==c['token_pos']-max(parents,default=0)
  mean=statistics.mean(r['ttft_ms'] for r in requests);groups[(item['snapshot_id'],item['policy'])].append(mean)
  for r in requests:
   if r['e']>0 and r['executable_checkpoint_id'] is None:unmapped.append({'run':item['run_id'],'request':r['request_id'],'H':r['h'],'E':r['e'],'G':r['g']})
  ledger.append({'run_id':item['run_id'],'snapshot_id':item['snapshot_id'],'policy':item['policy'],'repetition':item['repetition'],'candidate_count':item['candidate_count'],'K':item['logical_k'],'mean_ttft_ms':mean,'raw_requests':len(requests),'correctness':'PASS','cleanup':'PASS','missing_files':';'.join(miss)})
 rows=[]
 for sid in sorted(meta):
  means={p:statistics.mean(groups[sid,p]) for p in ['FlowState','LRU','Marconi']};assert all(len(groups[sid,p])==3 for p in means)
  rows.append({'snapshot_id':sid,**means,**{p+'_relative':(means[p]-means['FlowState'])/means[p] for p in ['LRU','Marconi']}})
 summary={}
 for p in ['LRU','Marconi']:
  differences=[r[p]-r['FlowState'] for r in rows]
  summary[p]={'mean_relative_percent':100*statistics.mean(r[p+'_relative'] for r in rows),'mean_absolute_ms':statistics.mean(differences),'win_tie_loss':[sum(v>1e-9 for v in differences),sum(abs(v)<=1e-9 for v in differences),sum(v< -1e-9 for v in differences)],'losses':[r for r in rows if r[p]<r['FlowState']]}
 runs,snapshot_ids=original.load_runs(BASE);snapshots=original.aggregate_snapshots(runs,snapshot_ids);paired=original.paired_comparison(snapshots,'mean_ttft_ms')
 bootstrap=original.bootstrap_results(paired,original.bootstrap_samples(snapshots))
 sources=read(BASE/'source_integrity.json');current=[]
 for row in sources['sources']:
  f=Path(row['path']);current.append({**row,'current_exists':f.exists(),'current_matches_historical':f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==row['after_sha256']})
 save('summary.json',{'classification':'QUALIFIED','runs':len(ledger),'requests':sum(r['raw_requests'] for r in ledger),'missing':missing,'mean_ttft_ms':{p:statistics.mean(r[p] for r in rows) for p in ['LRU','Marconi','FlowState']},'comparisons':summary,'positive_E_no_legacy_source':unmapped,'marconi_reproduced_runs':72,'alphas':sorted(set(s['marconi_alpha'] for s in meta.values())),'source_identity':current,'snapshot_ids_verified':snapshot_ids})
 save('bootstrap_reproduction.json',bootstrap);table('lifecycle_inventory.csv',ledger);table('snapshot_ttft.csv',rows)
 for f,h in HASH.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 save('read_integrity.json',{'unchanged':True,'sha256':HASH})
 print(json.dumps(summary,ensure_ascii=False));print('missing',missing,'unmapped',len(unmapped),'current_source',[r['path'] for r in current if not r['current_matches_historical']])
if __name__=='__main__':main()
