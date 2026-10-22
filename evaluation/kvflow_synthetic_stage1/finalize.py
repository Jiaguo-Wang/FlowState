"""汇总既有CPU与一次Runtime尝试，不重跑、不放宽门禁。"""
import json,csv,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent

def main():
 work=json.loads((OUT/'workload.json').read_text());state=json.loads((OUT/'strict/history_20.json').read_text())['state'];rows=[]
 for n in state['before']['nodes']:
  if not n['slots']:continue
  owners=[i for i,d in enumerate(work['cycles'][0]['declarations']) if n['token_ids']==d['fixed_token_ids'][:len(n['token_ids'])]]
  rows.append({'node_id':n['node_id'],'L':len(n['token_ids']),'slots':n['slots'],'fixed_prefix_agents':owners,'fixed_endpoint':len(n['token_ids'])==4096,'conv_complete':n['conv_complete'],'temporal_complete':n['temporal_complete'],'references':n['references'],'physical_mapping':'本次已结束生命周期，禁止用于未来控制'})
 with (OUT/'runtime_materialization.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 cleanup=json.loads((OUT/'gpu_cleanup.json').read_text())
 lock=json.loads((OUT/'source_identity.json').read_text())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in lock['sha256'].items())
 result={'gate':'PARTIAL','cpu':'PASS','runtime_control':'BLOCKED','history_requests_completed':20,'native_resident_candidates':len(rows),'fixed_endpoint_candidates':sum(r['fixed_endpoint'] for r in rows),'extra_ancestor_candidates':sum(not r['fixed_endpoint'] for r in rows),'actual_evictions':0,'pending_dispatched':0,'cold_executed':False,'resume_validated':False,'gpu_cleanup':cleanup,'source_files_unchanged':True,'reason':'实验入口要求所有候选恰为固定末端；额外4032祖先状态导致驱逐前拒绝。不是无法物化，也不是策略或恢复计算错误。'}
 (OUT/'result_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
