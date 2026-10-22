"""在冻结镜像中只运行CPU分词；不导入SGLang或创建引擎。"""
import ast
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence, Callable
from transformers import AutoTokenizer

ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/multishare_cpu_input_gate_b5c2'
MODEL=Path('/home/wjg/models/qwen3.5-9b')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def token_sha(tokens):return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
def save(name,x):
 x=json.loads(json.dumps(x,ensure_ascii=False))
 if (OUT/name).exists():
  assert json.loads((OUT/name).read_text())==x,'已有本轮文件内容冲突：'+name
  return
 with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def lcp(*xs):
 for i,vs in enumerate(zip(*xs)):
  if len(set(vs))>1:return i
 return min(map(len,xs))
def load_functions(path,names,env):
 tree=ast.parse(path.read_text())
 body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(body)==len(names)+1
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),env)
 return {n:env[n] for n in names}

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not list(Path('/dev').glob('nvidia*')),'不应挂载GPU设备'
source=json.loads((OUT/'source_selection_lock.json').read_text())
b4=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
identity=json.loads((b4/'g001/configuration.json').read_text())
versions={n:importlib.metadata.version(n) for n in ('transformers','tokenizers','sglang','jinja2')}
assert versions['transformers']==identity['transformers_version']
assert versions['sglang']==identity['sglang_version']=='0.5.17'
for item in identity['tokenizer_files']+identity['model_files']:
 if 'sha256' in item:assert sha(Path(item['path']))==item['sha256'],item['path']
 else:assert not Path(item['path']).exists()
locked=json.loads((b4/'execution_lock.json').read_text())['sha256']
paths=[ROOT/'evaluation/openhands_single_workflow_smoke.py',ROOT/'evaluation/rq3_openhands_neutral_collector.py']
for p in paths:assert sha(p)==locked[str(p)],str(p)
env={'json':json,'Mapping':Mapping,'Sequence':Sequence,'Callable':Callable}
load_functions(paths[0],['_parse_tool_calls','normalize_message','_template_input_ids'],env)
load_functions(paths[1],['request_id_for','materialize_group_requests','CollectionAbort'],env)
tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
assert hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()==identity['chat_template_sha256']
config={'workload_type':source['workload_type'],'model_path':str(MODEL),'versions':versions,'image_id':'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926','b4_identity':identity,'files':{str(p):sha(p) for p in paths+list(MODEL.glob('*token*json'))+[MODEL/'chat_template.jinja',MODEL/'config.json']},'loaded_chat_template_sha256':identity['chat_template_sha256'],'tokenizer_class':type(tokenizer).__name__,'apply_chat_template':{'tokenize':True,'add_generation_prompt':True,'tools':'未传入，与B4一致','enable_thinking':'未传入，使用冻结模板默认值','truncation':False},'normalization':'原样AST加载冻结normalize_message、_parse_tool_calls与_template_input_ids；保留role/content/tool_call_id，工具参数JSON转换为对象','tools_definition_source':'模板不额外注入tools列表；保留原System消息内的工具说明和已有tool_calls，与B4完全一致','engine_started':False,'gpu_devices':[],'sglang_imported':False,'model_weights_loaded':False,'difference_from_B4':'仅四来源组合及CPU执行范围不同；未执行历史回放，不构造候选、FA或实际Planning Target'}
save('input_configuration_lock.json',config)
histories={}
causal=[]
normalized={}
for row in source['records']:
 label=row['label'];p=OUT/row['source_history_file'];assert sha(p)==row['source_history_sha256']
 raw=json.loads(p.read_text());assert sum(m.get('role')=='assistant' for m in raw)==2
 seen_calls={};tool_evidence=[]
 for i,m in enumerate(raw):
  calls=m.get('tool_calls_json')
  if calls:
   for c in env['_parse_tool_calls'](calls):
    if c.get('id'):seen_calls[c['id']]=i
  if m.get('role')=='tool':
   call=m.get('tool_call_id');assert call in seen_calls and seen_calls[call]<i,'工具结果无法对应先前调用'
   tool_evidence.append({'message_index':i,'tool_call_id':call,'call_message_index':seen_calls[call],'before_pending_cut':i<row['pending_cut_index'],'wall_clock':'UNKNOWN'})
 histories[label]=raw+[{'role':'assistant'}]
 normalized[label]=[env['normalize_message'](m) for m in raw]
 causal.append({'label':label,'cut_index':row['pending_cut_index'],'tool_results':tool_evidence,'historical_assistant_outputs':2,'third_output_available_to_builder':False,'later_messages_available_to_builder':False,'ordering_evidence':'原始消息索引及tool_call_id；无原始墙钟时间','target_native_outputs':False})
group=SimpleNamespace(group_ordinal=0,allocation_round=2,session_by_label={r['label']:r['session_id'] for r in source['records']})
requests,audits=env['materialize_group_requests'](tokenizer,histories,group=group,normalize_message=env['normalize_message'],template_input_ids=env['_template_input_ids'])
assert len(requests)==12
pending={};boundaries={};rendered={};request_rows=[]
for (label,turn),r in requests.items():
 tokens=r['input_ids'];assert tokens and all(isinstance(t,int) for t in tokens)
 assert len(tokens)<identity['engine_configuration']['context_length']
 rid=f'B5C2_{label}_TURN_{turn:03d}'
 name=f'{rid}_tokens.json';save(name,tokens)
 request_rows.append({'label':label,'turn':turn,'request_id':rid,'token_file':name,'token_count':len(tokens),'canonical_token_sha256':token_sha(tokens),'file_sha256':sha(OUT/name),'legacy_builder_request_id':r['rid']})
 if turn==3:pending[label]=tokens
for label,msgs in normalized.items():
 text=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
 rendered[label]=text
 (OUT/f'{label}_pending_rendered.txt').write_text(text)
 encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 assert encoded['input_ids']==pending[label]
 save(f'{label}_pending_offsets.json',encoded['offset_mapping'])
 parts=[]
 for count in range(1,len(msgs)+1):
  if count==1:
   assert msgs[0]['role']=='system' and msgs[1]['role']=='user'
   prefix=text[:text.index('<|im_start|>user\n')]
  else:
   prefix=tokenizer.apply_chat_template(msgs[:count],tokenize=False,add_generation_prompt=False)
  is_prefix=text.startswith(prefix)
  pt=tokenizer(prefix,add_special_tokens=False)['input_ids']
  parts.append({'after_message_count':count,'last_role':msgs[count-1]['role'],'rendered_character_count':len(prefix),'rendered_is_prefix':is_prefix,'tokenized_prefix_length':len(pt),'token_lcp_with_full_input':lcp(pt,pending[label]),'exact_token_prefix':pending[label][:len(pt)]==pt})
 boundaries[label]=parts
save('requests.json',request_rows)
save('causal_checks.json',{'source_rows':causal,'collector_audits':audits,'boundary_mapping':boundaries,'replay_order':[f'B5C2_{label}_TURN_{t:03d}' for t in (1,2) for label in 'ABCD'],'pending_order':list('ABCD')})
matrix={a:{b:lcp(pending[a],pending[b]) for b in 'ABCD'} for a in 'ABCD'}
common=lcp(*pending.values());ab=matrix['A']['B'];cd=matrix['C']['D']
positions=[]
for name,origin,pos in [('common','A',common),('task1','A',ab),('task2','C',cd)]:
 prefix=pending[origin][:pos]
 supports=[label for label,ids in pending.items() if len(ids)>=pos and ids[:pos]==prefix]
 divergence={label:{'next_token_id':ids[pos] if pos<len(ids) else None,'next_token_text':tokenizer.decode(ids[pos:pos+1]),'character_offset':json.loads((OUT/f'{label}_pending_offsets.json').read_text())[pos] if pos<len(ids) else None} for label,ids in pending.items() if label in supports}
 positions.append({'name':name,'position':pos,'origin':origin,'support':supports,'prefix_token_sha256':token_sha(prefix),'divergence':divergence,'runtime_materialization':'UNKNOWN','planning_target':'UNKNOWN','position_within_input':all(pos<=len(pending[x]) for x in supports),'required_future_target':f'T_p >= {pos}；本轮不假设FA或T','checkpoint_created':False})
system_tokens={label:row[0]['token_lcp_with_full_input'] for label,row in boundaries.items()}
structure=common>0 and ab>common and cd>common and positions[1]['support']==['A','B'] and positions[2]['support']==['C','D'] and all(matrix[x][y]==common for x in 'AB' for y in 'CD')
assert all(boundaries[label][0]['rendered_is_prefix'] for label in 'ABCD')
result={'gate':'READY' if structure else 'BLOCKED','counts':{k:len(v) for k,v in pending.items()},'pairwise_lcp':matrix,'four_way_lcp':common,'task_pair_lcp':{'AB':ab,'CD':cd},'system_prefix_token_lengths':system_tokens,'positions':positions,'structure_pass':structure,'scope':'真实目标Tokenizer构造的CPU内容输入；非原始Runtime已接收Token；无Checkpoint物化或严格边认证'}
save('token_prefix_matrix.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
