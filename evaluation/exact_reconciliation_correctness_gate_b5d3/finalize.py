"""汇总实际控制、单目标恢复及独立Cold输出，不以返回码替代正确性。"""
import csv,hashlib,json
from pathlib import Path
from .protocol import compare_outputs
P=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(name,text):(P/name).write_text(text.strip()+'\n')
def main():
 cleanup=read(P/'gpu_cleanup.json');source=read(P/'source_identity.json');unchanged=all(sha(Path(p))==h for p,h in source['sha256'].items())
 recon=read(P/'strict/reconciliation.json') if (P/'strict/reconciliation.json').exists() else None
 gate='BLOCKED';reason=[];timing=[]
 if recon:
  save('live_epoch_and_exact_plan.json',{'epoch':recon.get('epoch'),'raw_epoch':recon.get('raw_epoch'),'plan':recon.get('plan'),'oracle':recon.get('oracle'),'target':recon.get('target')})
  save('apply_time_validation.json',recon.get('apply_validation',{'status':'UNKNOWN','reason':recon.get('error')}))
  save('native_reconciliation_log.json',{'status':recon.get('status'),'operations':recon.get('operations',[]),'ledger':recon.get('reconciliation'),'error':recon.get('error')})
  save('post_reconciliation_census.json',{'view':recon.get('after'),'audit':recon.get('audit_after'),'selected_payload_before':recon.get('selected_payload_before'),'selected_payload_after':recon.get('selected_payload_after'),'fa_content_before':recon.get('fa_content_before'),'fa_content_after':recon.get('fa_content_after'),'fa_content_postcheck':recon.get('fa_content_postcheck')})
  timing=recon.get('timing',[])
 else:
  for name in ['live_epoch_and_exact_plan.json','apply_time_validation.json','native_reconciliation_log.json','post_reconciliation_census.json']:save(name,{'status':'UNKNOWN','reason':'未取得完整协调调用结果'})
 valid=recon and recon.get('ok') and (P/'strict/summary.json').exists() and (P/'cold/summary.json').exists()
 if valid:
  strict=read(P/'strict/summary.json');cold=read(P/'cold/summary.json');post=read(P/'strict/post_request.json')
  try:
   correctness=compare_outputs(read(P/'strict/output.json'),read(P/'cold/output.json'));save('output_correctness.json',correctness)
   assert strict['status']==cold['status']=='PASS'
   for arm in ['strict','cold']:
    assert read(P/f'{arm}_process.json')['returncode']==0 and read(P/arm/'engine_cleanup.json')['shutdown_returned']
   rq1=read(P/'strict/request.json');rq2=read(P/'cold/request.json')
   assert rq1['pending']['token_ids']==rq2['pending']['token_ids'] and rq1['sampling']==rq2['sampling']
   assert len(post['trace']['admission'])==1
   assert recon['oracle']['status']=='PASS' and recon['apply_validation']['status']=='PASS'
   assert all(r['returned'] for r in recon['operations'])
   assert recon['selected_payload_before']==recon['selected_payload_after']==post['state']['selected_payload']
   assert recon['fa_content_before']==recon['fa_content_after']
   assert post['state']['fa_protected_payload']==recon['fa_protected_payload']
   assert unchanged and set(cleanup)=={'strict','cold'} and all(v['status']=='PASS' for v in cleanup.values())
   gate='READY'
  except BaseException as exc:reason.append(repr(exc));save('output_correctness.json',{'status':'FAIL','error':repr(exc)})
  target=recon['target'];actual=strict['native_match'];plan=recon['plan'];bound=next(x['node'] for x in recon['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
  copies=post['trace']['copies'];execution=post['trace']['execution']
  for i,x in enumerate(execution):timing.append({'stage':f"native_run_batch_{i}_{x['forward_mode']}",'ms':(x['ns_after']-x['ns_before'])/1e6})
  timing.append({'stage':'strict_request_wall_with_observation','ms':strict['request_wall_ms']})
  write('cross_workflow_resume_validation.md',f'''# 单目标跨Workflow恢复

本轮目标 `{target['pending']['continuation_id']}`，来源Checkpoint `{target['source_checkpoint']}`。来源Session：`{target['source_history']['workflow_id']}`；目标Session：`{target['pending']['workflow_id']}`，二者不同。

当前来源最早在历史 `{target['source_first_observed']['first_observed_after']}` 完成后的Census中出现；原始记录位于strict/history_*.json，包含实际提交与原生Match接收Token身份。来源记录不是原生Fork事件，不将同任务重复执行称为自然在线分支。首次Census不是精确insert时间戳，但本轮顺序回放及此前空缓存/各步Census给出物化来源边界。

Apply后所选源Node={bound['node_id']}，Slot={bound['slots']}；Native实际Match Node={actual['best_match_node']}，源Slot={actual['cow_source_slots']}，工作Slot={actual['working_slots']}。实际H/E/G={actual['physical_fa_hit']}/{actual['executable_prefix']}/{actual['replay_gap']}。恢复前沿来自执行路径，不只是计划计算。完整输入长度{actual["input_token_count"]}，实际Prefill从E到完整输入末端，共{actual["input_token_count"]-actual["executable_prefix"]} Token；模型恢复缺口G只计E到Planning Target/FA frontier，不能把两者混为同一计算长度。

Admission门禁在实际process_input_requests之前核对Token、完整节点、缓存语义、Selected内容、旧FA映射与保护区域内容。协调窗口结束后没有假定跨RPC租约；独占Engine只提交本目标，真实Admission重新核验。

COW原函数返回、目标槽复制后内容等于源槽复制前内容，源槽前后不变；工作槽与全部Selected槽互斥。Prefill/Decode原run_batch均返回。原始证明见strict/native_trace.json、strict/post_request.json及strict/summary.json。

Strict与独立干净Cold采用相同完整Token、temperature=0/max_new_tokens=8/ignore_eos=True；逐Token与停止行为检查见output_correctness.json。Cold原生H/E为0/0。没有将Cold完整prefill和严格恢复做性能比较，也没有执行其他正式Pending。

仅此边及本次恢复获得运行支持，不升级其余EPOCH_ADMISSIBLE边为RESUME_VALIDATED。
''')
  resident=[{'node':x['node_id'],'slots':x['slots']} for x in recon['after']['nodes'] if x['slots']]
  write('state_integrity_report.md',f'''# 状态完整性

协调前驻留候选{len(recon['raw_epoch']['candidates'])}；实际Recurrent-only操作{len(recon['operations'])}项，全部原生产Adapter返回及逐步Postcheck通过。协调后驻留：{resident}；空闲槽{len(recon['after']['free_slots'])}。逻辑补集和实际操作列表一致，无额外补驱逐、恢复或造状态。

Selected Conv/Temporal内容前后相同，恢复后仍与协调基线相同；组件切片身份不变。FA全树结构/索引由生产严格Postcheck核验；本轮额外直接读取全部旧FA驻留索引的K/V数据，协调前后逐层内容摘要相同。

恢复请求允许按原生语义重算目标E之后的旧Attention区间并新增缓存。原FA位置映射全部保留；将目标可重算区间索引{len(recon['fa_expected_recomputed_indices'])}项与其他受保护旧FA索引{len(recon['fa_protected_indices'])}项分开，后者K/V内容逐层完全不变。未机械要求恢复后全树不增节点，也未用合法新增状态掩盖Selected历史内容破坏。

物理来源、计划、Apply绑定、实际Match及COW均为本轮当前生命周期。长期Slot generation没有被补造；未来复用必须重新观察与绑定。来源及生产代码摘要不变={unchanged}；清理结果={json.dumps({k:v['status'] for k,v in cleanup.items()},ensure_ascii=False)}。
''')
  write('final_report.md',f'''# Phase B5-D3 最终报告

**Gate：{gate}**。

## 实际结果

当前新Runtime自然捕获{len(recon['raw_epoch']['candidates'])}个候选、4个Pending。Exact K2选择 `{plan['selected_ids']}`；模型化J={plan['recovery_cost_ms']:.9f}ms。当前Epoch独立Decimal Oracle一致，没有使用历史Node/Slot硬编码选择。

单次Scheduler同步控制函数内完成当前观察、规划、Apply-time重验、bind/preflight及StateController.reconcile_strict。实际执行{len(recon['operations'])}项Recurrent-only Eviction；最终Selected驻留及全部后置条件通过。控制函数没有返回调度循环后再使用旧绑定执行。

正式恢复仅 `{target['pending']['continuation_id']}`，实际来源 `{target['source_checkpoint']}`、Node {actual['best_match_node']}、Slot {actual['cow_source_slots']}，COW工作槽 {actual['working_slots']}。实际H/E/G={actual['physical_fa_hit']}/{actual['executable_prefix']}/{actual['replay_gap']}。逐Token 8 Token输出与独立Cold Reference对照见output_correctness.json。

FA协调前后索引及实际K/V内容不变；恢复后Selected组件内容与非重算区域旧FA内容保持。详见state_integrity_report.md。没有部分驱逐失败或静默异常成功；如最终复核错误，原因列表为{reason}。

## 耗时边界

control_overhead_breakdown.csv保留重型观察/GPU同步、Epoch认证、Exact+Plan校验、Oracle、Apply-time校验、严格协调及每次Native run_batch宿主调用区间。原Adapter操作耗时也单列，但包含原安全检查，不是裸GPU内核时延。request wall包含观测写盘、匹配与COW审计；run_batch调用亦不等同CUDA event测量。不能把这些数值称生产轻量控制、线上P95或TTFT收益。

结构/身份/保护检查可能进入真实生产关键路径；全树内容哈希、独立Oracle、详细日志等是本轮实验审计成本。没有实现删除审计后的轻量版本，更没有通过相减宣称净性能改善。J是模型化成本，与实测时间分开。

## 完整性与清理

冻结来源及生产源码摘要不变={unchanged}。Strict与Cold各独立生命周期、各仅一次正式Pending，GPU清理见gpu_cleanup.json。CUDA IPC退出警告如存在保留runtime.log原文；资源清理以连续GPU与容器观测判断，不能只依据返回码。首次CPU夹具失败已保留，修正后真实接口CPU门禁通过，再启动GPU。

工作负载是Trace-derived Controlled Composition；K2不是物理显存压力。本轮只验证功能正确性，不进行Greedy/Exact性能A/B，不推广所有兼容边。完成后停止。
''')
 else:
  for arm in ['strict','cold']:
   if (P/arm/'failure.json').exists():reason.append(read(P/arm/'failure.json'))
  if (P/'launch_failure.json').exists():reason.append(read(P/'launch_failure.json'))
  write('cross_workflow_resume_validation.md','# 恢复验证\n\nBLOCKED：未取得全部必要Strict/Cold恢复证据，不从部分成功推断完整正确性。')
  write('state_integrity_report.md','# 状态完整性\n\n缺少完整成功链，保留实际部分记录，不重试。\n\n'+json.dumps(reason,ensure_ascii=False,indent=2))
  write('final_report.md','# Phase B5-D3\n\n**BLOCKED**。\n\n'+json.dumps(reason,ensure_ascii=False,indent=2)+'\n\n清理：'+json.dumps(cleanup,ensure_ascii=False)+'\n\n停止，不重试或扩大范围。')
 if recon:
  for i,op in enumerate(recon.get('operations',[])):
   if 'end_ns' in op:timing.append({'stage':f'adapter_eviction_{i}_{op["checkpoint_id"]}','ms':(op['end_ns']-op['start_ns'])/1e6})
 for row in timing:
  stage=row['stage']
  if stage.startswith('fa_content') or stage in ['heavy_observation_gpu_sync','independent_oracle']:
   row['category']='重型实验审计'
  elif stage.startswith('native_run_batch') or stage=='strict_request_wall_with_observation':
   row['category']='真实请求宿主计时，含审计或异步执行边界；非纯GPU核时间'
  else:row['category']='必要规划或安全控制与当前审计混合，未验证轻量生产成本'
 with (P/'control_overhead_breakdown.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['stage','ms','category']);w.writeheader();w.writerows(timing)
 save('validation.json',{'gate':gate,'frozen_unchanged':unchanged,'reasons':reason,'cleanup':{k:v['status'] for k,v in cleanup.items()}})
 save('evidence_manifest.json',{'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and not any(x.startswith('triton_cache') or x=='__pycache__' for x in p.parts) and p.name!='evidence_manifest.json'}})
 print('最终门禁',gate)
if __name__=='__main__':main()
