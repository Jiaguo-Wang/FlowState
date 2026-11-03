"""场景A单臂正确性运行器：B5C3 冻结环境重放→在线 plan_strict（Exact分支）→生产协调→正式Pending→Cold对照。

与 evaluation/exact_greedy_paired_gpu_fix_b5d4a/runner.py 的差别：
- 源码核验追加 hybrid_entry_sources（被测生产入口 flowstate/*.py 的当前哈希，运行时逐项核验）；
- 求解由控制路径内的新生产入口完成（不设 B5D4_SOLVER 环境变量）；
- 正式请求输出与 b5d4b 冻结 Cold 参照（cold/output.json，10 臂逐Token一致）在运行内直接对照。
单臂单生命周期，非性能实验：计时字段仅为执行记录，不构成任何性能口径。
"""
import hashlib,importlib.metadata,json,os,time,traceback,copy
from pathlib import Path
from .bridge import scheduler_process
from .protocol import require,compare_outputs
from .persistence import call_saved
from .source_contract import validate_native_source
OUT=Path('/b5out');ARM=os.environ['B5D4_ARM'];RUN=os.environ['B5D4_RUN']
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 out=OUT/RUN;out.mkdir(exist_ok=False);engine=None;result=None
 try:
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());config=inputs['configuration'];source=json.loads((OUT/'source_identity.json').read_text())
  for p,h in source['runtime_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Runtime源码变化')
  for p,h in source['hybrid_entry_sources'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'被测生产入口源码变化')
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
  class AuditedControlClient(ControlClient):
   def _call(self,request):
    return call_saved(self.port,self.timeout_s,request,out/('rpc_'+str(time.monotonic_ns())))
  engine=ObservedEngine(**config['engine_configuration']);client=AuditedControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']));wait_for_transport(client)
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
   require(result['strict_decision']['solver']=='Exact' and result['strict_decision']['gate_allowed'],'在线决策非门内Exact')
   pending=result['target']['pending']
  else:
   raise ValueError('本轮只执行 strict 臂（Cold 参照复用 b5d4b 冻结证据）')
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
  target=result['target'];bound=next(x['node'] for x in result['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
  native_validation=validate_native_source(target,bound,actual)
  write(out/'native_source_validation.json',native_validation)
  require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in post['trace']['copies']),'COW未完成或内容错误')
  kept={s for c in result['reconciliation']['binding']['candidates'] if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
  require(not kept.intersection(actual['working_slots']),'工作状态与保留状态别名')
  require(post['state']['selected_payload']==result['selected_payload_after'] and post['state']['selected_components']==result['admission_baseline']['selected_components'],'历史Recurrent状态受损')
  require(post['state']['fa_protected_payload']==result['fa_protected_payload'],'非重算FA内容变化')
  require(all(post['state']['fa_map'].get(k)==v for k,v in result['fa_map_before'].items()),'历史FA映射丢失或改变')
  # 与 b5d4b 冻结 Cold 参照逐Token对照（该参照已经10个严格臂逐Token一致验证）
  cold=json.loads((OUT/'frozen_cold_reference.json').read_text())
  cold_check=compare_outputs(output,cold);write(out/'cold_reference_check.json',cold_check)
  write(out/'summary.json',{'status':'PASS','arm':ARM,'scenario':'A_b5c4_k2_exact_branch','solver':result['solver'],'strict_decision':result['strict_decision'],'selection_source':'GlobalOptimizer.plan_strict(在线)','plan_selected_ids':result['plan']['selected_ids'],'plan_recovery_cost_ms':result['plan']['recovery_cost_ms'],'oracle':result['oracle'],'provenance_status':native_validation['provenance']['status'],'cross_workflow_resume_certified':False,'native_match':actual,'output_ids':output['output_ids'],'cold_reference_match':True,'evictions':[(o['checkpoint_id'],o['returned']) for o in result['operations']],'request_wall_ms':(end-start)/1e6,'ttft_ms':(first-start)/1e6,'stream_tpot_ms':(arrivals[-1]['ns']-first)/7e6,'control_rpc_wall_ms':(control_end-control_start)/1e6,'control_through_completion_ms':(end-control_start)/1e6,'control_timing_stages':result['timing'],'说明':'单臂GPU正确性验证，非性能实验；计时字段仅为执行记录；CUDA事件是设备时间线区间而非纯内核之和'})
  print('完成',ARM,'solver',result['solver'],'E',actual['executable_prefix'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
 finally:
  if engine is not None:engine.shutdown()
  write(out/'engine_cleanup.json',{'shutdown_returned':True,'engine_created':engine is not None})
if __name__=='__main__':main()
