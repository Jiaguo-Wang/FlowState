"""严格控制与Cold分别独立生命周期，每臂只执行一个正式Pending。"""
import hashlib,importlib.metadata,json,os,time,traceback,copy
from pathlib import Path
from .bridge import scheduler_process
from .protocol import require
OUT=Path('/b5out');ARM=os.environ['B5D4_ARM'];RUN=os.environ['B5D4_RUN']
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 out=OUT/RUN;out.mkdir(exist_ok=False);engine=None;result=None
 try:
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());config=inputs['configuration'];source=json.loads((OUT/'source_identity.json').read_text())
  for p,h in source['runtime_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Runtime源码变化')
  for p,h in json.loads((OUT/'cpu_gate.json').read_text())['runtime_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'CPU门禁对应Runtime源码变化')
  for n in ['sglang','transformers']:require(importlib.metadata.version(n)==config[n+'_version'],'版本不符')
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True)
  require(hashlib.sha256(str(tok.chat_template).encode()).hexdigest()==config['chat_template_sha256'],'模板身份不符')
  from .observation import real_api_preflight
  require(real_api_preflight()['status']=='PASS','真实Req接口不符')
  from sglang.srt.entrypoints.engine import Engine
  from targeted_probe import ControlClient
  from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
  class ObservedEngine(Engine):run_scheduler_process_func=staticmethod(scheduler_process)
  engine=ObservedEngine(**config['engine_configuration']);client=ControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']));wait_for_transport(client)
  def call(action,**kw):
   r=client._call({'op':'checkpoint_control','action':action,'nonce':str(time.time_ns()),**kw});require(r.get('ok'),r.get('error','Runtime调用失败'));return r
  initial=call('b5c3_observe');require(not any(n['slots'] for n in initial['before']['nodes']),'新Engine非空');write(out/'initial.json',initial)
  if ARM=='strict':
   for i,request in enumerate(inputs['history'],1):
    output=engine.generate(input_ids=request['token_ids'],sampling_params=inputs['sampling'],rid=request['request_id']);state=call('b5c3_observe')
    matches=state['matches'].get(request['request_id'],[])
    require(matches and all(x['input_token_digest']==request['token_prefix_sha256'] and x['input_token_count']==len(request['token_ids']) for x in matches),'历史实际输入不符')
    require(len(output.get('output_ids',[]))==1 and int(output['meta_info'].get('num_retractions',0) or 0)==0,'历史异常')
    write(out/f'history_{i:02d}.json',{'request':request,'output':output,'state':state});print('历史完成',i,flush=True)
   control_start=time.monotonic_ns();result=client._call({'op':'checkpoint_control','action':'b5d4_apply','nonce':str(time.time_ns())});control_end=time.monotonic_ns()
   write(out/'reconciliation.json',result);require(result.get('ok'),result.get('error','协调失败'))
   pending=result['target']['pending']
  else:
   pending=next(p for p in inputs['pendings'] if p['continuation_id']=='B5C2_C_TURN_003');state=call('b5d4_state');write(out/'cold_initial.json',state)
   require(not any(n['slots'] for n in state['state']['nodes']),'Cold缓存污染')
  sampling={'temperature':0,'max_new_tokens':8,'ignore_eos':True}
  write(out/'request.json',{'pending':pending,'sampling':sampling})
  incremental=bool(engine.tokenizer_manager.incremental_streaming_output)
  start=time.monotonic_ns();chunks=[];arrivals=[];first=None
  for chunk in engine.generate(input_ids=pending['token_ids'],sampling_params=sampling,rid=pending['continuation_id'],stream=True):
   now=time.monotonic_ns();chunk=copy.deepcopy(chunk);chunks.append(chunk);arrivals.append({'ns':now,'output_ids':chunk.get('output_ids'),'completion_tokens':chunk.get('meta_info',{}).get('completion_tokens')})
   if first is None and (chunk.get('output_ids') or chunk.get('meta_info',{}).get('completion_tokens',0)>0):first=now
  end=time.monotonic_ns();require(chunks and first is not None,'缺少真实流式首Token');output=copy.deepcopy(chunks[-1])
  if incremental:output['output_ids']=[t for c in chunks for t in c.get('output_ids',[])]
  require(len(output['output_ids'])==output['meta_info']['completion_tokens'],'流式Token拼接与真实完成计数不一致')
  write(out/'stream_chunks.json',{'chunks':chunks,'arrivals':arrivals,'start_ns':start,'end_ns':end,'first_token_ns':first,'incremental_streaming_output':incremental})
  write(out/'output.json',output);post=call('b5d4_state');write(out/'post_request.json',post)
  require(len(output.get('output_ids',[]))==8 and output['meta_info']['completion_tokens']==8 and int(output['meta_info'].get('num_retractions',0) or 0)==0,'正式请求输出或回撤异常')
  matches=[m['native'] for m in post['trace']['matches'] if m.get('native',{}).get('cow_requested')];require(matches,'缺少真实匹配证据');actual=matches[-1]
  require(actual['input_token_digest']==pending['token_prefix_sha256'],'Pending输入不符')
  require(post['trace']['execution'] and all(x.get('native_run_returned') for x in post['trace']['execution']),'Prefill/Decode未完成')
  if ARM=='cold':require(actual['physical_fa_hit']==actual['executable_prefix']==0,'Cold命中旧状态')
  else:
   target=result['target'];bound=next(x['node'] for x in result['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
   require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'恢复来源错误')
   require(actual['executable_prefix']==target['expected']['E'] and actual['physical_fa_hit']==pending['attention_frontier'],'实际前沿错误')
   require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in post['trace']['copies']),'COW未完成或内容错误')
   kept={s for c in result['reconciliation']['binding']['candidates'] if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
   require(not kept.intersection(actual['working_slots']),'工作状态与保留状态别名')
   require(post['state']['selected_payload']==result['selected_payload_after'] and post['state']['selected_components']==result['admission_baseline']['selected_components'],'历史Recurrent状态受损')
   require(post['state']['fa_protected_payload']==result['fa_protected_payload'],'非重算FA内容变化')
   require(all(post['state']['fa_map'].get(k)==v for k,v in result['fa_map_before'].items()),'历史FA映射丢失或改变')
  write(out/'summary.json',{'status':'PASS','arm':ARM,'native_match':actual,'output_ids':output['output_ids'],'request_wall_ms':(end-start)/1e6,'ttft_ms':(first-start)/1e6,'stream_tpot_ms':(arrivals[-1]['ns']-first)/7e6,'control_rpc_wall_ms':(control_end-control_start)/1e6 if ARM=='strict' else None,'control_through_completion_ms':(end-control_start)/1e6 if ARM=='strict' else None,'说明':'受控审计化执行，完整时延含审计；CUDA事件是设备时间线区间而非纯内核之和'})
  print('完成',ARM,'E',actual['executable_prefix'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
 finally:
  if engine is not None:engine.shutdown()
  write(out/'engine_cleanup.json',{'shutdown_returned':True,'engine_created':engine is not None})
if __name__=='__main__':main()
