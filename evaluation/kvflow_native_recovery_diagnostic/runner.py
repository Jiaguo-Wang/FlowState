"""严格控制与Cold分别独立生命周期，每臂只执行一个正式Pending。"""
import hashlib,importlib.metadata,json,os,time,traceback
from pathlib import Path
from .bridge import scheduler_process
from evaluation.exact_reconciliation_correctness_gate_b5d3.protocol import require
OUT=Path('/b5out');ARM=os.environ['B5D3_ARM']
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 out=OUT/ARM;out.mkdir(exist_ok=False);engine=None;result=None
 try:
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());config=inputs['configuration'];source=json.loads((OUT/'source_identity.json').read_text())
  for p,h in source['runtime_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Runtime源码变化')
  for p,h in json.loads((OUT/'cpu_gate.json').read_text())['runtime_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'CPU门禁对应Runtime源码变化')
  for n in ['sglang','transformers']:require(importlib.metadata.version(n)==config[n+'_version'],'版本不符')
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True)
  require(hashlib.sha256(str(tok.chat_template).encode()).hexdigest()==config['chat_template_sha256'],'模板身份不符')
  from evaluation.exact_reconciliation_correctness_gate_b5d3.observation import real_api_preflight
  require(real_api_preflight()['status']=='PASS','真实Req接口不符')
  source_path=Path('/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py')
  (out/'schedule_batch_source.py').write_text(source_path.read_text())
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
   result=client._call({'op':'checkpoint_control','action':'b5d3_apply','nonce':str(time.time_ns())})
   write(out/'reconciliation.json',result);require(result.get('ok'),result.get('error','协调失败'))
   pending=result['target']['pending'];write(OUT/'target_request.json',pending)
  else:
   pending=json.loads((OUT/'target_request.json').read_text());state=call('b5d3_state');write(out/'cold_initial.json',state)
   require(not any(n['slots'] for n in state['state']['nodes']),'Cold缓存污染')
  sampling={'temperature':0,'max_new_tokens':8,'ignore_eos':True}
  write(out/'request.json',{'pending':pending,'sampling':sampling})
  start=time.monotonic_ns();output=engine.generate(input_ids=pending['token_ids'],sampling_params=sampling,rid=pending['continuation_id']);end=time.monotonic_ns()
  write(out/'output.json',output);post=call('b5d3_state');write(out/'post_request.json',post)
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
  write(out/'summary.json',{'status':'PASS','arm':ARM,'native_match':actual,'output_ids':output['output_ids'],'request_wall_ms':(end-start)/1e6,'说明':'含重型观测，不作为性能比较；实际run_batch调用区间另见trace'})
  print('完成',ARM,'E',actual['executable_prefix'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
 finally:
  if engine is not None:engine.shutdown()
  write(out/'engine_cleanup.json',{'shutdown_returned':True,'engine_created':engine is not None})
if __name__=='__main__':main()
