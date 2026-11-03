"""G2 控制路径：B5-C4/K2 双求解臂在同一真实Scheduler进程内在线求解并严格协调。

与 paper_hybrid_gpu_fix_a/control.py（B5-C4 Exact 分支，gate_a 已验收；其结构沿自 b5d4a）的
差别仅在求解入口按 G2_SOLVER 分派：
- Exact 臂：生产 GlobalOptimizer.plan_strict 在线求解，规模门禁内应自动选择 Exact，不读冻结选集；
- Greedy 臂：B5-C4 满足 Exact 门禁，生产入口本会自动选 Exact——不篡改候选数量/门禁/预算强制
  Greedy，而是经 .greedy_adapter 复用 plan_strict 超域分支的同一生产 Marginal Greedy 内部路径
  （同一 frontiers 评分闭包与 _select_candidates 循环）生成可验证 Strict Plan，再走相同
  validate_plan / Bind / Preflight / reconcile_strict 路径。
其余（from_frozen/prepare Epoch 认证、独立 Decimal Oracle、来源合同 choose_source、Recurrent-only
协调、FA 审计）与 b5d4b 已验证路径逐项相同；任何不一致即失败停止，不回退、不修改实验输入。
Epoch 形成方式：复用 B5C3 冻结 8 条历史重放 + 实时观察准入，非全在线 Epoch 构建。
"""
import hashlib,itertools,json,threading,time,traceback,os
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from evaluation.multishare_runtime_gate_b5c3 import bridge as capture
from evaluation.runtime_control_closure_r2c0 import bridge as prior
from evaluation.runtime_control_closure_r2c1a.bridge import RecordedAdapter,selected_payload
from evaluation.exact_planner_contract_gate_b5d1 import planner as planner
from flowstate.controller import StateController
from flowstate.optimizer import GlobalOptimizer,_exact_admission
from flowstate.recovery_model import RecoveryCostModel
from flowstate.strict_control import StrictRuntimeBridge,bind,preflight,StrictReconciliationError,validate_plan
from .greedy_adapter import plan_greedy_strict
from .source_contract import choose_source
from .persistence import Journal
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
 result={'ok':False,'status':'BLOCKED','operations':[],'timing':[],'observations':[],'phase':'start'}
 journal=None
 try:
  journal=Journal(OUT/os.environ['B5D4_RUN']/'control_evidence')
  result['phase']='observe'
  t=time.perf_counter_ns();initial=capture.observe_control(scheduler,{'final':True});require(initial.get('ok'),'只读采集失败')
  result['timing'].append({'stage':'heavy_observation_gpu_sync','ms':(time.perf_counter_ns()-t)/1e6})
  scheduler._r2c0_owner=threading.get_ident();window=initial['window'];prior.scope(scheduler)
  t=time.perf_counter_ns();raw=initial['strict_epoch'];data=encode(raw);physical=encode(initial)
  result.update(raw_epoch=raw,physical_observation=initial);journal.record('observed_epoch',result)
  inputs=json.loads((OUT/'prepared_inputs.json').read_text());origins=[]
  result['phase']='epoch_authentication'
  require(len(raw['candidates'])==7 and len(raw['pendings'])==4,'候选域或Pending数与B5-C4冻结结构不符')
  for c in raw['candidates']:
   seen=[]
   for i,h in enumerate(inputs['history'],1):
    state=json.loads((OUT/os.environ['B5D4_RUN']/f'history_{i:02d}.json').read_text())['state']
    if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots'] for n in state['before']['nodes']):seen.append(h['request_id'])
   require(bool(seen),'缺少本生命周期候选来源');origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':seen[0]})
  evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
  epoch=planner.from_frozen(evidence,2,(initial,origins,inputs));result.update(epoch=asdict(epoch),origins=origins);journal.record('strict_epoch',result);prepared=planner.prepare(epoch,evidence)
  require(planner.allowed(epoch),'当前规模不允许Exact，不进行驱逐')
  result['timing'].append({'stage':'strict_epoch_authentication','ms':(time.perf_counter_ns()-t)/1e6})
  # ── 在线求解（G2 双臂分派；不读任何冻结选集）──
  result['phase']='planning'
  solver=os.environ['G2_SOLVER'];require(solver in ('Exact','Greedy'),'未授权求解臂：'+solver)
  t=time.perf_counter_ns();optimizer=GlobalOptimizer(RecoveryCostModel())
  if solver=='Exact':
   plan=optimizer.plan_strict(epoch);decision=dict(optimizer.last_strict_decision)
   require(decision['solver']=='Exact' and decision['gate_allowed'],'Exact臂预期门禁内自动Exact，实际：'+repr(decision['gate']))
   selection_source='GlobalOptimizer.plan_strict(在线,自动Exact)'
  else:
   plan=plan_greedy_strict(optimizer,epoch)
   allowed,gate=_exact_admission(len(epoch.candidates),len(epoch.pendings),epoch.K)
   require(allowed,'Greedy对照臂要求域满足Exact门禁（证明未通过篡改规模进入Greedy）')
   decision={'solver':'Marginal Greedy','gate_allowed':allowed,'gate':gate,'selected_ids':list(plan.selected_ids),
             'objective_ms':plan.recovery_cost_ms,'pure_solve_ms':None,
             'note':'G2 Greedy对照臂：域满足门禁但按实验设计直接调用生产Greedy路径（greedy_adapter复用plan_strict超域分支），未篡改门禁/候选/预算'}
   selection_source='生产Marginal Greedy(在线,Greedy对照臂)'
  result['timing'].append({'stage':'online_solve','ms':(time.perf_counter_ns()-t)/1e6})
  validate_plan(epoch,plan)
  result['solver']=solver;result['strict_decision']=decision;result['selection_source']=selection_source
  result['plan']=asdict(plan);journal.record('plan',result)
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
  if solver=='Exact':require(winner==plan.selected_ids and abs(float(best)-plan.recovery_cost_ms)<1e-9,'当前Oracle不一致')
  require(plan.recovery_cost_ms>=float(best)-1e-9,'计划不可能优于Oracle')
  result['oracle']={'status':'PASS','subsets':len(scores),'selected':winner,'J_ms':float(best)}
  result['timing'].append({'stage':'independent_oracle','ms':(time.perf_counter_ns()-t)/1e6})
  result['phase']='source_selection'
  target=choose_source(raw,epoch,plan,origins,inputs,solver)
  result['target']=target;journal.record('source_expectation',result)
  result['phase']='apply_preflight'
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
  result['phase']='reconciliation';journal.record('before_reconciliation',result)
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
  result['phase']='complete';journal.record('complete',result)
  return result,epoch,plan
 except BaseException as exc:
  if isinstance(exc,StrictReconciliationError):result['reconciliation']=exc.ledger
  scheduler._r2c0_failed=True;result.update(error=repr(exc),traceback=traceback.format_exc(),status='BLOCKED')
  if journal is not None:journal.fail(result,exc)
  return result,None,None
 finally:scheduler._r2c0_owner=None
