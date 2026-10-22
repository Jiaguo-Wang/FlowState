"""仅离线提取失败RPC及历史Census，不重跑策略或Engine。"""
import ast,json,hashlib,csv
from pathlib import Path
P=Path(__file__).resolve().parent
failure=json.loads((P/'p01_Greedy/failure.json').read_text())
line=next(x for x in failure['traceback'].splitlines() if x.startswith('RuntimeError: probe operation failed: '))
response=ast.literal_eval(line.split('RuntimeError: probe operation failed: ',1)[1])
(P/'failed_control_response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
inputs=json.loads((P/'prepared_inputs.json').read_text());pending=next(p for p in inputs['pendings'] if p['continuation_id']=='B5C2_C_TURN_003')
rows=[]
for f in sorted((P/'p01_Greedy').glob('history_*.json')):
 x=json.loads(f.read_text())
 for n in x['state']['before']['nodes']:
  if not n['slots']:continue
  if any(z['node_id']==n['node_id'] for z in rows):continue
  rows.append({'node_id':n['node_id'],'position':len(n['token_ids']),'slots':n['slots'],
   'first_observed_after':x['request']['request_id'],'mapped_workflow_id':x['request']['workflow_id'],
   'target_workflow_id':pending['workflow_id'],'token_prefix_of_C':pending['token_ids'][:len(n['token_ids'])]==n['token_ids'],
   'gate_workflow_not_equal':x['request']['workflow_id']!=pending['workflow_id'],'evidence_path':str(f.relative_to(P))})
assert response['operations']==[] and response['observations']==[]
(P/'failure_source_audit.json').write_text(json.dumps({'status':'CONFIRMED_INSTRUMENTATION_GATE_ERROR','provenance_rows':rows,'failed_control_response':'failed_control_response.json','native_evictions':0,'formal_pending_executed':0,'warning':'首次观察边界不是精确insert事件或源张量生成者认证'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
