"""只读提取冻结预热日志中的物理占用及匹配时刻。"""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'evaluation/kvflow_synthetic_stage1'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 inputs=read(OLD/'prepared_inputs.json')
 rows=[]
 final=read(OLD/'strict/history_20.json')['state']
 for n in final['before']['nodes']:
  if not n['slots']:continue
  matching=[h['request_id'] for h in inputs['history'] if n['token_ids']==h['token_ids'][:len(n['token_ids'])]]
  rows.append({k:n[k] for k in ['node_id','slots','component_ids','conv_complete','temporal_complete','references','protected','attention_present']}|{'L':len(n['token_ids']),'matching_history':matching})
 slots=[s for n in rows for s in n['slots']];components=[s for n in rows for s in n['component_ids']]
 assert len(rows)==20 and len(set(slots))==20 and len(set(components))==len(components)
 assert sorted(x['L'] for x in rows)==[4032]*10+[4096]*10
 events=[]
 for i in [1,11]:
  p=OLD/f'strict/history_{i:02d}.json';d=read(p);s=d['state'];rid=d['request']['request_id']
  events.append({'path':str(p),'sha256':digest(p),'request_id':rid,'input_length':len(d['request']['token_ids']),'native_matches':s['matches'].get(rid), 'resident_after':[{'node':n['node_id'],'L':len(n['token_ids']),'slots':n['slots']} for n in s['before']['nodes'] if n['slots'] and n['token_ids']==d['request']['token_ids'][:len(n['token_ids'])]]})
 result={'status':'PASS','source':str(OLD/'strict/history_20.json'),'source_sha256':digest(OLD/'strict/history_20.json'),'resident_slot_count':len(slots),'free_slots':final['before']['free_slots'],'pool_capacity':final['before']['pool_capacity'],'candidate_rows':rows,'agent00_events':events,'native_eviction_events':final['native_eviction_events'],'说明':'不同槽和组件切片不能合并计为一个；首次观察时间不构成张量跨Workflow血缘证明'}
 (OUT/'historical_mapping.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('冻结物理占用：20独立槽；Agent00匹配时序已保存')
if __name__=='__main__':main()
