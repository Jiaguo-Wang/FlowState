"""独立冻结Epoch纯规划适配器；不持有Runtime执行能力。"""
from dataclasses import dataclass,asdict
import hashlib,itertools,json,math
from flowstate.strict_epoch import StrictAllocationEpoch,StrictAllocationPlan,StrictCandidate,StrictPending,StrictEdge,PendingPlan,_digest
from flowstate.state_catalog import CheckpointCandidate
from flowstate.recovery_model import RecoveryCostModel
from flowstate.optimizer import GlobalOptimizer
from flowstate.strict_control import validate_plan
EPS=1e-9
MODEL=RecoveryCostModel()
class LimitError(ValueError):
 """仅合法输入的规模拒绝，可由调用方显式选择Fallback。"""
class Cancelled(ValueError):
 """协作取消不返回部分计划，也不宣称强制中断。"""
@dataclass(frozen=True)
class FrozenEvidence:
 """原始字节及可信摘要只认证冻结边界，不认证当前GPU驻留。"""
 snapshot:bytes
 expected_sha256:str
 scope:str='frozen_snapshot'
 stale:bool=False
 physical_snapshot:bytes|None=None
 physical_sha256:str|None=None
@dataclass(frozen=True)
class Prepared:
 """已经通过共同合同检查的评分输入。"""
 epoch:StrictAllocationEpoch
 ids:tuple
 costs:tuple
 compatible:tuple
@dataclass(frozen=True)
class Outcome:
 """求解器来源与最优性不塞入既有生产Plan字段。"""
 plan:StrictAllocationPlan
 solver:str
 optimality_certified:bool
 fallback_reason:str|None
 evidence_scope:str='冻结快照纯规划；无当前物理绑定'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def read_unique(data):
 def pairs(items):
  out={}
  for k,v in items:
   require(k not in out,'JSON键重复');out[k]=v
  return out
 return json.loads(data,object_pairs_hook=pairs)
def from_frozen(evidence,K,provenance=None):
 """读取B4原格式或B5物理格式，均保留原字节SHA及严格配对。"""
 require(hashlib.sha256(evidence.snapshot).hexdigest()==evidence.expected_sha256,'来源摘要不一致')
 s=read_unique(evidence.snapshot)
 if 'barrier_id' in s:return StrictAllocationEpoch.from_snapshot_bytes(evidence.snapshot,expected_sha256=evidence.expected_sha256,K=K)
 require(provenance is not None,'B5缺少来源标识及组件切片证据')
 physical,origins,history=provenance
 cfg=_digest(s['configuration']);eid=s['epoch_id'];cs=[]
 for c in s['candidates']:
  cid=c['checkpoint_id'];node=next(n for n in physical['before']['nodes'] if n['node_id']==c['state_identity']['node_id'])
  size=sum(size for identity in node['component_ids'] for _,size in physical['audit_before']['component_slices'][identity]['byte_ranges'])
  origin=next(o for o in origins if o['checkpoint_id']==cid)
  source=next(h for h in history['history'] if h['request_id']==origin['first_observed_after'])
  core=CheckpointCandidate(cid,source['workflow_id'],(),c['token_position_L'],size,c['state_identity']['target_mamba_present'],c['state_identity']['target_full_present'] and c['state_identity']['path_full_all_present'])
  cs.append(StrictCandidate(core,tuple(c['token_ids']),c['token_prefix_sha256'],c['barrier_id'],c['configuration_digest']))
 ps=tuple(StrictPending(p['continuation_id'],p['workflow_id'],p['planning_target_T'],tuple(p['token_ids']),p['token_prefix_sha256'],p['barrier_id'],p['configuration_digest']) for p in s['pendings'])
 es=tuple(StrictEdge(e['checkpoint_id'],e['continuation_id'],e['status'],eid,tuple(sorted(e['conditions'].items()))) for e in s['edges'])
 payload=dict(epoch_id=eid,snapshot_sha256=evidence.expected_sha256,configuration_digest=cfg,candidates=tuple(cs),pendings=ps,edges=es,K=K)
 canonical={k:tuple(asdict(x) for x in v) if k in ('candidates','pendings','edges') else v for k,v in payload.items()}
 return StrictAllocationEpoch(**payload,content_digest=_digest(canonical))
def prepare(epoch,evidence):
 """先验证严格输入及冻结状态证据；任何错误均不能通过Fallback绕过。"""
 require(type(epoch) is StrictAllocationEpoch and type(evidence) is FrozenEvidence,'输入类型不合法')
 epoch.validate()
 require(evidence.scope=='frozen_snapshot' and evidence.stale is False,'陈旧或非冻结状态证据')
 require(hashlib.sha256(evidence.snapshot).hexdigest()==evidence.expected_sha256==epoch.snapshot_sha256,'来源或Epoch摘要不一致')
 s=read_unique(evidence.snapshot);eid=s.get('barrier_id',s.get('epoch_id'))
 physical=None
 if 'epoch_id' in s:
  require(type(evidence.physical_snapshot) is bytes and hashlib.sha256(evidence.physical_snapshot).hexdigest()==evidence.physical_sha256,'缺少可信物理切片证据')
  physical=read_unique(evidence.physical_snapshot)
  require(physical['unchanged'] is True and physical['window']==eid,'物理观察边界或不变性不一致')
 require(eid==epoch.epoch_id and _digest(s['configuration'])==epoch.configuration_digest,'边界或配置不一致')
 raw={c['checkpoint_id']:c for c in s['candidates']};require(len(raw)==len(s['candidates'])==len(epoch.candidates),'候选域不一致')
 nodes=[];slots=[];components=[]
 for c in epoch.candidates:
  x=raw[c.candidate.checkpoint_id];identity=x['state_identity'];nodes.append(identity['node_id'])
  require(tuple(x['token_ids'])==c.token_ids and x['token_position_L']==c.candidate.token_pos,'来源候选内容不一致')
  require(identity['target_mamba_present'] is True and identity['target_full_present'] is True and identity['path_full_all_present'] is True,'物理组件不完整')
  require(identity.get('references') and all(v==0 for v in identity['references'].values()),'引用缺失或非零')
  require(len(identity['target_mamba_slots'])==1,'未验证多槽候选成本')
  slots+=identity['target_mamba_slots'];components+=x.get('component_ids',[])
  if physical is not None:
   node=next(n for n in physical['before']['nodes'] if n['node_id']==identity['node_id'])
   require(node['slots']==identity['target_mamba_slots'] and node['conv_complete'] and node['temporal_complete'],'组件映射不一致')
   require(node['token_ids']==x['token_ids'] and node['component_ids']==x['component_ids'],'物理Token或组件身份不一致')
   size=sum(size for cid in node['component_ids'] for _,size in physical['audit_before']['component_slices'][cid]['byte_ranges'])
   require(size==c.candidate.memory_bytes,'物理切片成本不一致')
  if 'metadata' in x:require(c.candidate.memory_bytes==x['metadata']['memory_bytes'],'来源预算成本不一致')
 require(len(set(nodes))==len(nodes) and len(set(slots))==len(slots) and len(set(components))==len(components),'物理节点、槽或组件别名')
 rp={p['continuation_id']:p for p in s['pendings']}
 require(len(rp)==len(s['pendings'])==len(epoch.pendings),'Pending域不一致')
 for p in epoch.pendings:
  x=rp[p.pending_id];require(x['controlled_input_ready'] is True and tuple(x['token_ids'])==p.token_ids and x['planning_target_T']==p.target,'Pending输入或目标不一致')
 re={(e['checkpoint_id'],e['continuation_id']):(e['status'],tuple(sorted(e['conditions'].items()))) for e in s['edges']}
 require(len(re)==len(s['edges'])==len(epoch.edges),'关系域不一致')
 require(all(re[e.candidate_id,e.pending_id]==(e.status,e.conditions) for e in epoch.edges),'冻结关系被修改')
 ids=tuple(c.candidate.checkpoint_id for c in epoch.candidates)
 compat=tuple(tuple(e.status=='EPOCH_ADMISSIBLE' for p in epoch.pendings for e in epoch.edges if e.candidate_id==cid and e.pending_id==p.pending_id) for cid in ids)
 costs=tuple(MODEL.estimate(p.target,p.target) for p in epoch.pendings)
 require(all(math.isfinite(v) and v>=0 for v in costs),'基准成本非有限或负数')
 return Prepared(epoch,ids,costs,compat)
def benefits(prepared):
 """共同矩阵构建，强制有限、非负及拒绝边零收益。"""
 ep=prepared.epoch;rows=[]
 for i,c in enumerate(ep.candidates):
  row=tuple(prepared.costs[j]-MODEL.estimate(p.target-c.candidate.token_pos,p.target) if prepared.compatible[i][j] else 0. for j,p in enumerate(ep.pendings))
  require(all(math.isfinite(v) and v>=0 for v in row),'收益非有限或负数');rows.append(row)
 return tuple(rows)
def score(prepared,selected):
 ep=prepared.epoch;positions={cid:i for i,cid in enumerate(prepared.ids)}
 frontiers=tuple(max([ep.candidates[positions[c]].candidate.token_pos for c in selected if prepared.compatible[positions[c]][j]]+[0]) for j in range(len(ep.pendings)))
 return sum(MODEL.estimate(p.target-e,p.target) for p,e in zip(ep.pendings,frontiers))
def allowed(ep):
 n=len(ep.candidates);return n<=12 and len(ep.pendings)<=4 and ep.K<=4 and sum(math.comb(n,i) for i in range(min(n,ep.K)+1))<=794
def choose_exact(scored):
 """遵循B4-C相对真实最小值的容差、较少项和字典序同分规则。"""
 best=min(v for v,_ in scored)
 return min((s for v,s in scored if abs(v-best)<=EPS),key=lambda s:(len(s),s))
def solve(prepared,matrix,solver,cancel=None):
 """共同评分合同；取消只在显式检查点协作响应。"""
 require(matrix==benefits(prepared),'收益矩阵与冻结模型不一致')
 ep=prepared.epoch
 if cancel is not None and cancel():raise Cancelled('已请求取消')
 if solver=='Exact':
  if not allowed(ep):raise LimitError('超出小规模Exact准入域')
  scored=[]
  for k in range(min(ep.K,len(prepared.ids))+1):
   for subset in itertools.combinations(sorted(prepared.ids),k):
    if cancel is not None and cancel():raise Cancelled('协作取消，未返回部分最优计划')
    scored.append((score(prepared,subset),subset))
  return choose_exact(scored)
 require(solver=='Greedy','未知求解器')
 size=ep.candidates[0].candidate.memory_bytes if ep.candidates else 1
 return tuple(c.checkpoint_id for c in GlobalOptimizer(MODEL)._select_candidates(tuple(c.candidate for c in ep.candidates),ep.K*size,lambda cs:score(prepared,tuple(c.checkpoint_id for c in cs))).selected)
def finish(prepared,selected):
 """构建既有Plan并调用生产只读validate_plan；无物理绑定。"""
 ep=prepared.epoch;es=ep.frontiers(selected)
 rows=tuple(PendingPlan(p.pending_id,p.target,e,p.target-e,MODEL.estimate(p.target-e,p.target)) for p,e in zip(ep.pendings,es))
 j=sum(p.cost_ms for p in rows)
 require(all(math.isfinite(p.cost_ms) and p.cost_ms>=0 for p in rows),'计划成本非法')
 plan=StrictAllocationPlan(ep.epoch_id,ep.snapshot_sha256,ep.content_digest,ep.K,tuple(selected),len(selected),rows,j,sum(prepared.costs)-j,tuple(sorted(set(prepared.ids)-set(selected))))
 validate_plan(ep,plan);return plan
def plan(epoch,evidence,allow_fallback=False,cancel=None):
 """安全合同先于规模分支，不对输入错误做宽泛异常降级。"""
 prepared=prepare(epoch,evidence);matrix=benefits(prepared)
 solver='Exact';reason=None
 if not allowed(epoch):
  if not allow_fallback:raise LimitError('超出准入域且未授权Fallback')
  solver='Greedy';reason='完整合法Epoch超过Exact规模上限'
 selected=solve(prepared,matrix,solver,cancel)
 return Outcome(finish(prepared,selected),solver,solver=='Exact',reason)
