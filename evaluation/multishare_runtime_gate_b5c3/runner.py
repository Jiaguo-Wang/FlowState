"""单Engine固定八请求回放及一次最终静止观察，不派发Pending。"""
import hashlib,importlib.metadata,json,os,time,traceback
from pathlib import Path
from .bridge import scheduler_process
OUT=Path('/b5out')
def write(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 engine=None;history=[];states=[]
 try:
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());config=inputs['configuration']
  source=json.loads((OUT/'source_identity.json').read_text());assert source['status']=='PASS'
  for p,h in source['runtime_sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
  versions={n:importlib.metadata.version(n) for n in ('sglang','transformers')}
  for n in versions:assert versions[n]==config[n+'_version'],n
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True)
  assert hashlib.sha256(str(tok.chat_template).encode()).hexdigest()==config['chat_template_sha256']
  write('runtime_identity.json',{'status':'PASS','versions':versions,'configuration':config,'runtime_sources':source['runtime_sources'],'image_id':source['image_id'],'pending_receive':'本轮不派发；CPU预备输入','runtime_source_checks':'PASS'})
  from sglang.srt.entrypoints.engine import Engine
  from targeted_probe import ControlClient
  from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
  from evaluation.strict_compatibility_b1c.protocol import token_digest
  class ObservedEngine(Engine):
   run_scheduler_process_func=staticmethod(scheduler_process)
  engine=ObservedEngine(**config['engine_configuration'])
  client=ControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']));wait_for_transport(client)
  def observe(final=False):
   result=client._call({'op':'checkpoint_control','action':'b5c3_observe','final':final,'nonce':str(time.time_ns())})
   assert result.get('ok') is True,result.get('error',str(result)[:300])
   return result
  initial=observe();assert not any(n['slots'] for n in initial['before']['nodes'])
  write('initial_state.json',initial);states.append(initial)
  for request in inputs['history']:
   start=time.monotonic_ns()
   output=engine.generate(input_ids=request['token_ids'],sampling_params=inputs['sampling'],rid=request['request_id'])
   state=observe();states.append(state)
   matches=state['matches'].get(request['request_id'],[])
   assert matches and all(m['input_token_count']==len(request['token_ids']) and m['input_token_digest']==request['token_prefix_sha256'] for m in matches),'Runtime接收身份冲突'
   meta=output['meta_info'];assert meta['prompt_tokens']==len(request['token_ids']) and meta['completion_tokens']==1
   assert len(output.get('output_ids',[]))==1 and int(meta.get('num_retractions',0) or 0)==0
   before={n['node_id']:n for n in states[-2]['before']['nodes'] if n['slots']}
   after={n['node_id']:n for n in state['before']['nodes'] if n['slots']}
   row={'ordinal':len(history)+1,'request_id':request['request_id'],'submitted_token_digest':token_digest(request['token_ids']),'received_matches':matches,'output':output,'start_ns':start,'end_ns':time.monotonic_ns(),'completed':True,'resident_node_ids_after':list(after),'removed_recurrent_node_ids':sorted(set(before)-set(after)),'new_recurrent_node_ids':sorted(set(after)-set(before)),'accounting_after':state['audit_before']['state']['accounting'],'native_eviction_events':state['native_eviction_events'],'说明':'输出属于原生正常副作用，不加入任何Pending输入'}
   history.append(row)
   write(f'history_{len(history):02d}.json',{'request':row,'state':state})
   print('历史请求完成',request['request_id'],'驻留节点',len(after),flush=True)
  write('historical_replay_log.json',history)
  final=observe(True)
  assert set(final['matches'])=={r['request_id'] for r in inputs['history']},'发现非历史请求'
  write('physical_state_census.json',final)
  write('strict_epoch_snapshot.json',final['strict_epoch'])
  write('sharing_support_matrix.json',{'status':final['status'],'rows':final['support_rows'],'epoch_id':final['window'],'native_eviction_events':final['native_eviction_events'],'positive_support_only':True,'policy_evaluation':False})
  write('result.json',{'status':final['status'],'history_completed':len(history),'pending_dispatched':0,'unchanged':final['unchanged']})
  print('最终结构门禁',final['status'],flush=True)
 except BaseException as exc:
  write('failure.json',{'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc(),'completed_history':history})
  print(traceback.format_exc(),flush=True);raise
 finally:
  if engine is not None:engine.shutdown()
  write('engine_cleanup.json',{'engine_created':engine is not None,'shutdown_returned':True,'unix_ns':time.time_ns()})
if __name__=='__main__':main()
