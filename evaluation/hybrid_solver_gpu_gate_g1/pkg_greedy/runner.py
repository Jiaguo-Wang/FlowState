"""场景B单臂正确性运行器：C/K3 双轮预热重放构造 16 候选 Epoch→在线 plan_strict（Greedy分支）→
生产协调→唯一正式 Pending→与冻结 Cold 参照逐 Token 对照。

生命周期序与 c_formal_perf_gpu/runner.py 的正确性部分相同（16 预热重放→flush→空缓存断言→
二次重放→唯一 b5d4_apply→正式流式请求→请求后全量观察），但删除全部性能测量机制
（形状预热、Triton 快照、TTFT 分解、测量模式断言）：本轮是 GPU 正确性与执行路径验证，
不是性能实验；计时字段仅为执行记录。
"""
import copy,hashlib,importlib.metadata,json,os,time,traceback
from pathlib import Path
from .bridge import scheduler_process
from .protocol import require
from .persistence import call_saved
OUT=Path('/b5out');ARM=os.environ['B5D4_ARM'];RUN=os.environ['B5D4_RUN']
require(ARM=='strict','本轮只用 strict 臂（单臂正确性验证）')
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def replay_warmup(engine,call,inputs,out,prefix):
 """16 条冻结预热真实重放；首轮写 history_pre_XX.json（留证），二次（flush 后）写 history_XX.json
 （控制溯源输入——node_id 属 flush 后重建世代）。逐请求 digest/输入长度/输出数合同验证。"""
 for i,request in enumerate(inputs['history'],1):
  output=engine.generate(input_ids=request['token_ids'],sampling_params=inputs['sampling'],rid=request['request_id'])
  state=call('b5c3_observe')
  matches=state['matches'].get(request['request_id'],[])
  require(matches and all(x['input_token_digest']==request['token_prefix_sha256'] and x['input_token_count']==len(request['token_ids']) for x in matches),'历史实际输入不符')
  require(len(output.get('output_ids',[]))==1 and int(output['meta_info'].get('num_retractions',0) or 0)==0,'历史异常')
  write(out/f'{prefix}_{i:02d}.json',{'request':request,'output':output,'state':state});print('预热完成',prefix,i,flush=True)
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
  # ── 第一轮冻结预热重放（首轮写 history_pre_XX.json 留证，不作控制溯源）──
  replay_warmup(engine,call,inputs,out,'history_pre')
  # ── flush → 空缓存断言 → 二次预热重放（重建 16 候选；写 history_XX.json 供控制溯源）──
  engine.flush_cache()
  state=call('b5c3_observe');require(not any(n['slots'] for n in state['before']['nodes']),'flush后仍有驻留mamba状态');write(out/'flush_state.json',state)
  replay_warmup(engine,call,inputs,out,'history')
  # ── 唯一一次真实协调：控制路径内由新生产入口 plan_strict 在线求解 ──
  control_start=time.monotonic_ns();result=client._call({'op':'checkpoint_control','action':'b5d4_apply','nonce':str(time.time_ns())});control_end=time.monotonic_ns()
  write(out/'reconciliation.json',result);require(result.get('ok'),result.get('error','协调失败'))
  require(result['census']['physical_domain_size']==16 and len(result['census']['mapped'])==16 and
          not result['census']['unmapped_extras'] and not result['census']['missing_physical'],
          'census 域异常：physical=%d mapped=%d extras=%d missing=%d'%(
           result['census']['physical_domain_size'],len(result['census']['mapped']),
           len(result['census']['unmapped_extras']),len(result['census']['missing_physical'])))
  decision=result['strict_decision']
  require(decision['solver']=='Marginal Greedy' and not decision['gate_allowed'] and decision['gate']['candidates']==16,
          '在线决策非超门禁Greedy：'+repr(decision))
  pending=result['target']['pending']
  write(out/'target_request.json',pending)  # 留证写入本臂run目录（无读取方；多臂复用时根目录'x'写会跨臂冲突）
  # ── 正式流式请求（8 Token；与冻结 Cold 参照逐 Token 对照）──
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
  require(actual['physical_fa_hit']==pending['attention_frontier'],'FA命中与冻结边界不一致')
  expected_E=int(result['target']['expected']['E'])
  require(actual['executable_prefix']==expected_E,'实际可执行前缀与计划E不符')
  execution=post['trace']['execution']
  require(execution and all(x.get('native_run_returned') for x in execution),'Prefill/Decode未完成')
  prefill_row=execution[0]
  require(prefill_row['batch_prefix_length']==expected_E and prefill_row['extend_token_count']==len(pending['token_ids'])-expected_E,'Prefill区间与E/G合同不符')
  # E=0 目标（该计划下无兼容保留状态）：无恢复源/COW/别名断言，仅有FA命中与E/G合同（对齐c_formal生产语义）
  target=result['target']
  if expected_E>0:
   bound=next(x['node'] for x in result['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
   require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'恢复来源错误')
   require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in post['trace']['copies']),'COW未完成或内容错误')
   kept={s for c in result['reconciliation']['binding']['candidates'] if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
   require(not kept.intersection(actual['working_slots']),'工作状态与保留状态别名')
  else:
   require(target['source_checkpoint'] is None and actual['cow_source_slots']==[] and actual['working_slots']==[] and not post['trace']['copies'],'E=0却出现恢复源/COW痕迹')
  require(post['state']['selected_payload']==result['selected_payload_after'] and post['state']['selected_components']==result['admission_baseline']['selected_components'],'历史Recurrent状态受损')
  require(post['state']['fa_protected_payload']==result['fa_protected_payload'],'非重算FA内容变化')
  require(all(post['state']['fa_map'].get(k)==v for k,v in result['fa_map_before'].items()),'历史FA映射丢失或改变')
  cold=json.loads((OUT/('cold_reference_'+pending['continuation_id']+'.json')).read_text())
  require(output['output_ids']==cold['output_ids'],'与Cold参照输出不一致')
  write(out/'summary.json',{'status':'PASS','arm':ARM,'scenario':'B_ck3_greedy_branch','pending':pending['continuation_id'],
   'strict_decision':decision,'selection_source':result['selection_source'],'selected_physical_ids':result['plan']['selected_ids'],
   'selected_cpu_mapped':result['oracle_selected_cpu'],'plan_recovery_cost_ms':result['plan']['recovery_cost_ms'],
   'oracle_policy':result['oracle_policy'],'oracle_match':True,'census_counts':{'physical':result['census']['physical_domain_size'],'mapped':len(result['census']['mapped']),'unmapped_extras':len(result['census']['unmapped_extras'])},
   'H_physical_fa_hit':actual['physical_fa_hit'],'E_executable_prefix':actual['executable_prefix'],'G_replay_gap':actual['physical_fa_hit']-actual['executable_prefix'],
   'prefill_token_count':prefill_row['extend_token_count'],'batch_prefix_length':prefill_row['batch_prefix_length'],
   'evictions':[(o['checkpoint_id'],o['returned']) for o in result['operations']],
   'output_ids':output['output_ids'],'cold_output_match':True,'native_match':actual,
   'request_wall_ms':(end-start)/1e6,'ttft_ms':(first-start)/1e6,'control_rpc_wall_ms':(control_end-control_start)/1e6,
   'control_timing_stages':result['timing'],
   '说明':'单臂GPU正确性验证，非性能实验；计时字段仅为执行记录，不构成任何性能口径'})
  print('完成 场景B',pending['continuation_id'],'solver',decision['solver'],'E',actual['executable_prefix'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
 finally:
  if engine is not None:engine.shutdown()
  write(out/'engine_cleanup.json',{'shutdown_returned':True,'engine_created':engine is not None})
if __name__=='__main__':main()
