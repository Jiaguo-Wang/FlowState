"""在单次Scheduler调用内构造当前Exact计划、重验身份并调用既有严格协调。"""
import hashlib,itertools,json,threading,time,traceback
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from evaluation.multishare_runtime_gate_b5c3 import bridge as capture
from evaluation.runtime_control_closure_r2c0 import bridge as prior
from evaluation.runtime_control_closure_r2c1a.bridge import RecordedAdapter,selected_payload
from evaluation.exact_planner_contract_gate_b5d1 import planner as planner
from flowstate.controller import StateController
from flowstate.strict_control import StrictRuntimeBridge,bind,preflight,StrictReconciliationError
OUT=Path('/b5out');require=prior.require

def encode(x):return json.dumps(x,sort_keys=True,ensure_ascii=False).encode()
def fa_map(cache):
 """读取实际FA索引，以完整Token前缀摘要建立位置映射，不调用match。"""
 import targeted_probe as probe
 from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
 from evaluation.strict_compatibility_b1c.runner import tensor_ids
 result={}
 for _,node in probe._tree_items(cache.tree_core):
  if node.parent is None:continue
  value=node.component_data[CT.FULL].value
  if value is None:continue
  parts=[];cur=node
  while cur.parent is not None:parts.append(list(cur.key));cur=cur.parent
  tokens=[t for part in reversed(parts) for t in part];ids=tensor_ids(value)
  require(len(ids)==len(node.key),'FA索引与节点Token长度不一致')
  digest=hashlib.sha256()
  for i,t in enumerate(tokens):
   digest.update(int(t).to_bytes(8,'little'))
   if i>=len(tokens)-len(ids):
    key=digest.hexdigest();idx=ids[i-(len(tokens)-len(ids))]
    require(key not in result or result[key]==idx,'FA前缀索引冲突');result[key]=idx
 return result

def fa_payload(cache,indices):
 """对指定实际驻留FA槽的K/V内容只读取摘要。"""
 import torch
 pool=cache.token_to_kv_pool_allocator.get_kvcache()
 require(type(pool).__name__=='HybridLinearKVPool','FA池类型超出已审计范围')
 require(not pool.full_kv_pool.is_quantized_kv_cache,'未审计量化KV读取')
 require(pool.layer_transfer_counter is None and pool.full_kv_pool.layer_transfer_counter is None,'不支持异步FA传输')
 rows=[]
 for layer in sorted(pool.full_attention_layer_id_mapping):
  for kind,get in [('K',pool.get_key_buffer),('V',pool.get_value_buffer)]:
   tensor=get(layer);ix=torch.tensor(sorted(indices),device=tensor.device,dtype=torch.long)
   data=tensor.index_select(0,ix).detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
   rows.append({'layer':layer,'component':kind,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
 return rows

def apply(scheduler,request):
 result={'ok':False,'status':'BLOCKED','operations':[],'timing':[],'observations':[]}
 try:
  t=time.perf_counter_ns();initial=capture.observe_control(scheduler,{'final':True});require(initial.get('ok'),'只读采集失败')
  result['timing'].append({'stage':'heavy_observation_gpu_sync','ms':(time.perf_counter_ns()-t)/1e6})
  scheduler._r2c0_owner=threading.get_ident();window=initial['window'];prior.scope(scheduler)
  t=time.perf_counter_ns();raw=initial['strict_epoch'];data=encode(raw);physical=encode(initial)
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());origins=[]
  for c in raw['candidates']:
   seen=[]
   for i,h in enumerate(inputs['history'],1):
    state=json.loads((OUT/'strict'/f'history_{i:02d}.json').read_text())['state']
    if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots'] for n in state['before']['nodes']):seen.append(h['request_id'])
   require(bool(seen),'缺少本生命周期候选来源');origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':seen[0]})
  evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
  epoch=planner.from_frozen(evidence,2,(initial,origins,inputs));prepared=planner.prepare(epoch,evidence);matrix=planner.benefits(prepared)
  require(planner.allowed(epoch),'当前规模不允许Exact，不进行驱逐')
  result['timing'].append({'stage':'strict_epoch_authentication','ms':(time.perf_counter_ns()-t)/1e6})
  t=time.perf_counter_ns();selected=planner.solve(prepared,matrix,'Exact');plan=planner.finish(prepared,selected)
  result['timing'].append({'stage':'exact_solve_plan_validation','ms':(time.perf_counter_ns()-t)/1e6})
  # 独立Oracle只使用本次原始边、位置和冻结系数。
  t=time.perf_counter_ns();cs={c['checkpoint_id']:c for c in raw['candidates']};edges={(e['checkpoint_id'],e['continuation_id']) for e in raw['edges'] if e['status']=='EPOCH_ADMISSIBLE'}
  def phi(g,t):
   g=Decimal(g)/1024;t=Decimal(t)/1024
   return Decimal('37.828150')*g+Decimal('0.345974143')*g*t-Decimal('0.156201917')*g*g
  scores=[]
  for k in range(3):
   for chosen in itertools.combinations(sorted(cs),k):
    j=sum(phi(p['planning_target_T']-max([cs[c]['token_position_L'] for c in chosen if (c,p['continuation_id']) in edges]+[0]),p['planning_target_T']) for p in raw['pendings']);scores.append((j,chosen))
  best=min(v for v,_ in scores);winner=min((s for v,s in scores if abs(v-best)<=Decimal('1e-9')),key=lambda s:(len(s),s))
  require(winner==plan.selected_ids and abs(float(best)-plan.recovery_cost_ms)<1e-9,'当前Oracle不一致')
  result['oracle']={'status':'PASS','subsets':len(scores),'selected':winner,'J_ms':float(best)}
  result['timing'].append({'stage':'independent_oracle','ms':(time.perf_counter_ns()-t)/1e6})
  choices=[]
  for p in sorted(raw['pendings'],key=lambda p:p['label']):
   expected=next(x for x in plan.pending if x.pending_id==p['continuation_id'])
   for cid in sorted(plan.selected_ids):
    origin=next(o for o in origins if o['checkpoint_id']==cid);h=next(h for h in inputs['history'] if h['request_id']==origin['first_observed_after'])
    if h['workflow_id']!=p['workflow_id'] and (cid,p['continuation_id']) in edges and cs[cid]['token_position_L']==expected.E and expected.E>0 and sum(c==cid for c,_ in edges)>1:
     choices.append({'pending':p,'source_checkpoint':cid,'source_history':h,'source_first_observed':origin,'expected':asdict(expected)} )
  require(bool(choices),'无可证明来源不同的共享恢复目标');target=choices[0]
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
