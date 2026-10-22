"""生成事先声明的十Agent顺序合成工作流；不根据策略结果调整输入。"""
import json,random,hashlib,array
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def tokensha(x):return hashlib.sha256(array.array('q',x).tobytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=json.loads((OUT/'protocol.json').read_text());model=Path('/home/wjg/models/qwen3.5-9b');tok=json.loads((model/'tokenizer.json').read_text());tc=json.loads((model/'tokenizer_config.json').read_text())
 ids=sorted((set(tok['model']['vocab'].values())|{a['id'] for a in tok['added_tokens']})-{a['id'] for a in tok['added_tokens'] if a['special']});rng=random.Random(p['seed'])
 config=json.loads((ROOT/'evaluation/multishare_runtime_gate_b5c3/prepared_inputs.json').read_text())['configuration']
 assert hashlib.sha256(tc['chat_template'].encode()).hexdigest()==config['chat_template_sha256']
 fixed=[[rng.choice(ids) for _ in range(p['fixed_tokens'])] for _ in range(p['agents'])]
 suffixes=[[[rng.choice(ids) for _ in range(p['dynamic_tokens'])] for _ in fixed] for _ in range(p['workflow_cycles'])]
 cd=digest(config);history=[];cycles=[]
 for cycle in range(p['fixed_warmup_cycles']):
  for i,f in enumerate(fixed):history.append({'request_id':f'WARM_{cycle}_{i:02d}','workflow_id':f'AGENT_{i:02d}','token_ids':f,'token_prefix_sha256':tokensha(f),'label':str(i),'turn':cycle})
 for cycle in range(p['workflow_cycles']):
  eid=f'SYNTH_CYCLE_{cycle}';graph=[];declarations=[];pendings=[];candidates=[]
  for i,f in enumerate(fixed):
   inv=f'INVOKE_{cycle}_{i:02d}';full=f+suffixes[cycle][i]
   graph.append({'invocation_id':inv,'ready':i==0,'predecessors':[] if i==0 else [f'INVOKE_{cycle}_{i-1:02d}'],'aggregation':'AND','epoch_id':eid,'known_at_epoch':True,'evidence_path':str(OUT/'protocol.json')})
   declarations.append({'invocation_id':inv,'epoch_id':eid,'known_at_epoch':True,'configuration_digest':cd,'evidence_path':str(OUT/'protocol.json'),'fixed_token_ids':f,'input_token_ids':full,'dynamic_suffix_confirmed':True})
   pendings.append({'continuation_id':inv,'request_id':inv,'workflow_id':f'AGENT_{i:02d}','label':str(i),'token_ids':full,'token_prefix_sha256':tokensha(full),'planning_target_T':len(f),'ready':i==0,'controlled_input_ready':True})
   candidates.append({'checkpoint_id':f'CP_{i:02d}','token_ids':f,'token_position_L':len(f),'token_prefix_sha256':tokensha(f),'barrier_id':eid,'configuration_digest':cd,'state_status':'PROPOSED','logical_cost':1,'last_access_order':10+i,'creation_order':i})
  cycles.append({'epoch_id':eid,'configuration_digest':cd,'graph':graph,'declarations':declarations,'pendings':pendings,'candidates':candidates})
 save('workload.json',{'protocol':p,'configuration':config,'chat_template_applied':False,'input_mode':'直接input_ids；模板仅记录配置身份','cycles':cycles,'history':history})
 save('prepared_inputs.json',{'configuration':config,'sampling':p['warmup_sampling'],'history':history,'pendings':cycles[0]['pendings'],'workload_type':'reconstructed_synthetic_sequential_workflow'})
 paths=[OUT/'protocol.json',OUT/'workload.json',OUT/'prepared_inputs.json',model/'tokenizer.json',model/'tokenizer_config.json',model/'config.json',ROOT/'evaluation/kvflow_inspired_retention_cpu/policy.py',ROOT/'flowstate/recovery_model.py',ROOT/'flowstate/optimizer.py']
 save('input_lock.json',{'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},'eligible_vocabulary_size':len(ids),'tokenizer_class':'冻结tokenizer.json词表直接采样；未做文本编码'})
 print('生成完成：10 Agent、两轮已声明输入、20次固定提示预热计划；尚无Runtime执行')
if __name__=='__main__':main()
