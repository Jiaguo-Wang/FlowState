"""同一Scheduler窗口下把当前固定前缀映射接入既有严格协调，未通过即停止。"""
from evaluation.exact_reconciliation_correctness_gate_b5d3.control import *
from evaluation.kvflow_inspired_retention_cpu.policy import select

def apply(scheduler,request):
 result={'ok':False,'status':'BLOCKED','operations':[],'timing':[],'observations':[]}
 try:
  initial=capture.observe_control(scheduler,{'final':True});require(initial.get('ok'),'当前观察失败')
  scheduler._r2c0_owner=threading.get_ident();window=initial['window'];prior.scope(scheduler)
  raw=initial['strict_epoch'];data=encode(raw);physical=encode(initial)
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());workload=json.loads((OUT/'workload.json').read_text());cycle=workload['cycles'][0]
  origins=[]
  for c in raw['candidates']:
   matches=[h for h in inputs['history'] if c['token_ids']==h['token_ids']]
   require(bool(matches),'当前候选不是声明的固定末端；禁止补造或删去自然候选')
   origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':matches[0]['request_id']})
  require(len(raw['candidates'])==10,'原生固定前缀Checkpoint域不完整；停止，不人工创建')
  graph=[dict(x,epoch_id=window) for x in cycle['graph']]
  declarations=[dict(x,epoch_id=window,configuration_digest=raw['configuration_digest']) for x in cycle['declarations']]
  selected,priority=select(raw['candidates'],graph,declarations,window,raw['configuration_digest'],3)
  evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
  epoch=planner.from_frozen(evidence,3,(initial,origins,inputs));prepared=planner.prepare(epoch,evidence);plan=planner.finish(prepared,selected)
  result['priority']=priority
  target_pending=raw['pendings'][0];expected=next(x for x in plan.pending if x.pending_id==target_pending['continuation_id'])
  candidates=[c for c in raw['candidates'] if c['checkpoint_id'] in selected and c['token_ids']==target_pending['token_ids'][:c['token_position_L']] and c['token_position_L']==expected.E]
  require(len(candidates)==1 and expected.E>0,'目标没有唯一合法恢复源')
  target={'pending':target_pending,'source_checkpoint':candidates[0]['checkpoint_id'],'expected':asdict(expected)}
  t=time.perf_counter_ns();before,audit=prior.observe(scheduler,epoch,window)
  require(asdict(before)==initial['before'] and audit['semantic']==initial['audit_before']['semantic'] and audit['component_slices']==initial['audit_before']['component_slices'],'Apply前状态变化')
  require(capture.payload(scheduler.tree_cache,before)==initial['payload_before'],'Apply前组件内容变化')
  for p in raw['pendings']:
   from evaluation.barrier_fa_frontier_control import inspect_resident_fa_frontier
   require(inspect_resident_fa_frontier(scheduler.tree_cache,p['token_ids'])['resident_fa_frontier']==p['attention_frontier'],'Apply前FA前沿变化')
  binding=bind(epoch,plan,before,window);preflight(epoch,plan,binding,before,window)
  result['apply_validation']={'status':'PASS','binding':asdict(binding),'same_scheduler_call':True,'window':window}
  result['timing'].append({'stage':'apply_time_freshness_preflight','ms':(time.perf_counter_ns()-t)/1e6})
  t=time.perf_counter_ns();old_fa=fa_map(scheduler.tree_cache);indices=set(old_fa.values());full_hash=fa_payload(scheduler.tree_cache,indices)
  result['timing'].append({'stage':'fa_content_audit_before','ms':(time.perf_counter_ns()-t)/1e6})
  result.update(before=asdict(before),audit_before=audit,selected_payload_before=selected_payload(scheduler.tree_cache,before,plan,epoch))
  @contextmanager
  def held():
   prior.scope(scheduler)
   yield window
   prior.scope(scheduler)
  def observe():
   view,a=prior.observe(scheduler,epoch,window);require(a['state']['tree']['structure_sha256']==audit['state']['tree']['structure_sha256'],'协调过程FA树结构变化');result['observations'].append(asdict(view));return view
  def invalidate(ledger):scheduler._r2c0_failed=True
  t=time.perf_counter_ns();runtime=StrictRuntimeBridge(RecordedAdapter(scheduler.tree_cache,result['operations']),held,observe,invalidate)
  ledger=StateController.reconcile_strict(epoch,plan,runtime)
  result['timing'].append({'stage':'strict_reconciliation_with_postchecks','ms':(time.perf_counter_ns()-t)/1e6})
  after,last=prior.observe(scheduler,epoch,window);source_after=selected_payload(scheduler.tree_cache,after,plan,epoch)
  require(source_after==result['selected_payload_before'],'所选组件内容变化')
  require([x['checkpoint_id'] for x in result['operations']]==list(plan.logical_eviction_ids) and all(x['returned'] for x in result['operations']),'驱逐未完整落实')
  t=time.perf_counter_ns();require(fa_map(scheduler.tree_cache)==old_fa and fa_payload(scheduler.tree_cache,indices)==full_hash,'协调改变FA索引或内容')
  # 请求可能正常重算E之后的目标FA区间，仅保护其他原始FA内容。
  digest=hashlib.sha256();recomputed=set()
  for pos,token in enumerate(target['pending']['token_ids']):
   digest.update(int(token).to_bytes(8,'little'))
   if pos>=target['expected']['E'] and digest.hexdigest() in old_fa:recomputed.add(old_fa[digest.hexdigest()])
  protected=sorted(indices-recomputed);protected_hash=fa_payload(scheduler.tree_cache,protected)
  result['timing'].append({'stage':'fa_content_postcheck_and_resume_baseline','ms':(time.perf_counter_ns()-t)/1e6})
  result.update(ok=True,status='READY',raw_epoch=raw,epoch=asdict(epoch),plan=asdict(plan),target=target,origins=origins,reconciliation=ledger,after=asdict(after),audit_after=last,selected_payload_after=source_after,fa_indices_before=sorted(indices),fa_content_before=full_hash,fa_content_after=full_hash,fa_content_postcheck='PASS',fa_protected_indices=protected,fa_protected_payload=protected_hash,fa_expected_recomputed_indices=sorted(recomputed),fa_map_before=old_fa)
  prior.scope(scheduler)
  return result,epoch,plan
 except BaseException as exc:
  if isinstance(exc,StrictReconciliationError):result['reconciliation']=exc.ledger
  scheduler._r2c0_failed=True;result.update(error=repr(exc),traceback=traceback.format_exc(),status='BLOCKED');return result,None,None
 finally:scheduler._r2c0_owner=None
