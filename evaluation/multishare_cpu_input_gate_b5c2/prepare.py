"""按既定来源顺序锁定四会话；只保存第三次请求之前的消息。"""
import json,hashlib
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'evaluation/workload_structure_feasibility_b5c1'
DATA=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
old=json.loads((OLD/'source_manifest.json').read_text())
assert sha(DATA)==old[str(DATA)]['sha256']
manifest=json.loads((OLD/'artifact_manifest.json').read_text())
for name,digest in manifest['files'].items():assert sha(OLD/name)==digest,name
summary=json.loads((OLD/'audit_summary.json').read_text())
chosen=[(task,sid) for task,ids in sorted(summary['openhands']['multi_session_task_groups'].items())[:2] for sid in sorted(ids)[:2]]
assert len(chosen)==4
rows=pq.read_table(DATA).to_pylist()
records=[]
for label,(task,sid) in zip('ABCD',chosen):
 matches=[(i,r) for i,r in enumerate(rows) if r['session_id']==sid]
 assert len(matches)==1
 index,row=matches[0]
 assert json.loads(row['ground_truth_meta_json'])['instance_id']==task
 raw=json.loads(row['messages_json']);cut=None;turns=[]
 for i,m in enumerate(raw):
  if m.get('role')=='assistant':
   turns.append(i)
   if len(turns)==3:cut=i;break
 assert cut is not None,'固定来源不足三次请求，禁止补位'
 history=raw[:cut]
 save(f'{label}_source_history.json',history)
 records.append({'label':label,'instance_id':task,'session_id':sid,'source_id':row['source_id'],'source_dataset':row['source_dataset'],'parquet_row_index_zero_based':index,'assistant_message_indices_zero_based':turns,'pending_cut_index':cut,'source_history_file':f'{label}_source_history.json','source_history_sha256':sha(OUT/f'{label}_source_history.json'),'recorded_model':row['recorded_model'],'third_assistant_content_read':False,'source_wall_clock':'UNKNOWN'})
save('source_selection_lock.json',{'workload_type':'Trace-derived Controlled Composition','rule':'沿B5-C1既定instance_id、session_id字典序，前两个任务各前两个Session；失败不补位','source_path':str(DATA),'source_sha256':sha(DATA),'records':records,'prior_protocol_sha256':sha(OLD/'candidate_workload_plan.md'),'prior_inventory_sha256':sha(OLD/'openhands_session_inventory.json'),'original_row_order_is_selection_order':False,'说明':'原始行号只作溯源；选取顺序是已预定字典序，不是原始时间顺序；只向分词脚本提供截断历史'})
print('四会话来源锁定完成。')
