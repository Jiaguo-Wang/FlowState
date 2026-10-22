"""冻结的三类合成输入生成器；只生成数据，不调用任何策略。"""
import json,random,hashlib,time
from pathlib import Path
from evaluation.kvflow_synthetic_stage1.generate import digest,tokensha
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'evaluation/kvflow_synthetic_stage1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 assert not (OUT/'workloads.json').exists(),'不得覆盖已生成工作负载'
 p=read(OUT/'protocol.json');old=read(OLD/'workload.json');model=Path('/home/wjg/models/qwen3.5-9b')
 tok=read(model/'tokenizer.json');ids=sorted((set(tok['model']['vocab'].values())|{a['id'] for a in tok['added_tokens']})-{a['id'] for a in tok['added_tokens'] if a['special']})
 rng=random.Random(p['seed']);rand=lambda n:[rng.choice(ids) for _ in range(n)]
 config=old['configuration'];cd=digest(config)
 assert hashlib.sha256(read(model/'tokenizer_config.json')['chat_template'].encode()).hexdigest()==config['chat_template_sha256']
 cases=[]
 def build(name,prefixes,calls,order,description):
  eid='MW_CPU_'+name;graph=[];decl=[];ps=[];cs=[];events=[];warmups=[]
  for call in calls:
   inv=call['id'];fixed=prefixes[call['prefix']];full=call.get('full',fixed+rand(p['dynamic_tokens']))
   graph.append({'invocation_id':inv,'ready':call['ready'],'predecessors':call['parents'],'aggregation':'AND','epoch_id':eid,'known_at_epoch':True,'evidence_path':str(OUT/'protocol.json'),'workflow_id':call['workflow'],'stage_label':call.get('stage',0)})
   decl.append({'invocation_id':inv,'epoch_id':eid,'known_at_epoch':True,'configuration_digest':cd,'evidence_path':str(OUT/'protocol.json'),'fixed_token_ids':fixed,'input_token_ids':full,'dynamic_suffix_confirmed':True})
   ps.append({'continuation_id':inv,'workflow_id':call['workflow'],'token_ids':full,'token_prefix_sha256':tokensha(full),'planning_target_T':len(fixed),'attention_frontier_assumed':len(fixed),'ready':call['ready'],'controlled_input_ready':True,'runtime_received':False})
  for owner,fixed in prefixes.items():
   for length in [len(fixed)-p['ancestor_distance'],len(fixed)]:
    tokens=fixed[:length];cid=f'{name}_{owner}_P{length:05d}'
    cs.append({'checkpoint_id':cid,'prefix_owner':owner,'token_ids':tokens,'token_position_L':length,'token_prefix_sha256':tokensha(tokens),'barrier_id':eid,'configuration_digest':cd,'state_status':'PROPOSED','runtime_materialized':False,'logical_cost':1,'expected_independent_slot_cost':1,'runtime_handle':None,'control_domain':'POLICY_MANAGED','creation_order':len(cs)})
  assert len({tuple(c['token_ids']) for c in cs})==len(cs),'重复Token状态不能按两个物理候选计数'
  for cycle in range(2):
   for owner,fixed in prefixes.items():
    rid=f'{name}_WARM_{cycle}_{owner}';warmups.append({'request_id':rid,'token_ids':fixed,'token_prefix_sha256':tokensha(fixed),'status':'PLANNED_NOT_EXECUTED'})
    for c in cs:
     if c['prefix_owner']==owner:
      e={'event_order':len(events),'checkpoint_id':c['checkpoint_id'],'request_id':rid,'event_type':'DECLARED_SYNTHETIC_ACCESS','runtime_observed':False};events.append(e);c['last_access_order']=e['event_order']
  for c in cs:
   parents=[x for x in cs if x['token_position_L']<c['token_position_L'] and x['token_ids']==c['token_ids'][:x['token_position_L']]]
   c['parent_candidate']=max(parents,key=lambda x:x['token_position_L'])['checkpoint_id'] if parents else None
   c['token_compatible_calls']=[x['continuation_id'] for x in ps if c['token_position_L']<=x['planning_target_T'] and c['token_ids']==x['token_ids'][:c['token_position_L']]]
  assert set(order)=={x['id'] for x in calls}
  cases.append({'epoch_id':eid,'structure':name,'description':description,'configuration_digest':cd,'graph':graph,'declarations':decl,'calls':ps,'candidates':cs,'predeclared_evaluation_order':order,'planned_warmup':warmups,'synthetic_access_events':events,'common_retained_state_ids':[],'runtime_actual_candidates':[]})
 cyc=old['cycles'][0];prefixes={f'AGENT{i:02d}':d['fixed_token_ids'] for i,d in enumerate(cyc['declarations'])};calls=[]
 for i,(g,d,q) in enumerate(zip(cyc['graph'],cyc['declarations'],cyc['pendings'])):
  calls.append({'id':g['invocation_id'],'prefix':f'AGENT{i:02d}','ready':g['ready'],'parents':g['predecessors'],'workflow':'A_SEQUENTIAL','stage':i,'full':d['input_token_ids']})
 build('A',prefixes,calls,[c['id'] for c in calls],'原顺序Token输入不变；统一账本新增每个末端的64 Token前祖先，当前目标仅首调用')
 prefixes={};calls=[]
 for w,lengths in enumerate(p['multi_workflow']['fixed_lengths']):
  for step,length in enumerate(lengths):
   name=f'W{w}_S{step}';inv='B_'+name;prefixes[name]=rand(length)
   calls.append({'id':inv,'prefix':name,'ready':step==0,'parents':[] if step==0 else [f'B_W{w}_S0'],'workflow':f'B_W{w}','stage':p['multi_workflow']['stage_labels'][w]+step})
 build('B',prefixes,calls,p['multi_workflow']['evaluation_order'],'三条独立合成Workflow，各有一个Ready及一个已知后继；执行阶段是预设标签，无未来输出反馈')
 spec=p['shared_branched'];root=rand(spec['root_length']);ab=root+rand(spec['branch_lengths'][0]-len(root));cdp=root+rand(spec['branch_lengths'][1]-len(root))
 prefixes={'ROOT':root,'AB':ab,'CD':cdp}
 for i,name in enumerate(['A','B','C','D']):
  parent=ab if i<2 else cdp;prefixes[name]=parent+rand(spec['leaf_lengths'][i]-len(parent))
 prefixes['JOIN']=root+rand(spec['join_length']-len(root));calls=[]
 for name in ['A','B','C','D','JOIN']:
  calls.append({'id':'C_'+name,'prefix':name,'ready':name!='JOIN','parents':[] if name!='JOIN' else spec['join_predecessors'],'workflow':'C_SHARED','stage':1 if name!='JOIN' else 2})
 build('C',prefixes,calls,spec['evaluation_order'],'事先声明公共模板、AB/CD分支模板与四个Ready分支；AND Join等待四者，输入不读取未来执行结果。预热公共模板是显式合成准备请求')
 write(OUT/'workloads.json',{'configuration':config,'chat_template_applied':False,'input_mode':'复用Stage1直接非特殊Token ID采样；不是文本编码或自然Agent轨迹','protocol_sha256':sha(OUT/'protocol.json'),'cases':cases})
 sources=[OLD/'workload.json',OLD/'protocol.json',OLD/'generate.py',OLD/'cpu.py',OUT/'protocol.json',OUT/'generate.py',OUT/'workloads.json',model/'tokenizer.json',model/'tokenizer_config.json',model/'config.json',ROOT/'flowstate/optimizer.py',ROOT/'flowstate/recovery_model.py',ROOT/'flowstate/state_catalog.py',ROOT/'evaluation/kvflow_inspired_retention_cpu/policy.py',ROOT/'evaluation/sota_policies.py',ROOT/'evaluation/p1_strong_baseline_validation/common.py',ROOT/'evaluation/p1_strong_baseline_validation/wfc.py',ROOT/'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/calibration_points.csv']
 write(OUT/'input_lock.json',{'frozen_before_policy_execution_unix_ns':time.time_ns(),'sha256':{str(x):sha(x) for x in sources}})
 print([(c['structure'],len(c['candidates']),sum(q['ready'] for q in c['calls']),len(c['calls'])) for c in cases])
if __name__=='__main__':main()
