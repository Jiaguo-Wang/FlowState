"""严格逻辑计划的物理合同；真实观测和连续隔离由运行时桥接提供。"""
from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Protocol

from .adapters.sglang import RuntimeCheckpointHandle
from .recovery_model import RecoveryCostModel
from .strict_epoch import StrictAllocationEpoch, StrictAllocationPlan, PendingPlan


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(asdict(value),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class PhysicalNode:
    """组件标识必须指向每个实际槽位的存储切片，而非整个共享池。"""
    node_id: int
    token_ids: tuple[int, ...]
    slots: tuple[int, ...]
    component_ids: tuple[str, ...]
    conv_complete: bool
    temporal_complete: bool
    attention_present: bool
    references: int
    protected: bool
    host_copy: bool


@dataclass(frozen=True)
class PhysicalView:
    """nodes 为全部非哨兵缓存节点，根与 Attention 全树由摘要覆盖。"""
    engine_id: str
    barrier_id: str
    lease_id: str
    configuration_digest: str
    idle: bool
    admission_closed: bool
    transfers_pending: bool
    nodes: tuple[PhysicalNode, ...]
    pool_capacity: int
    free_slots: tuple[int, ...]
    attention_digest: str
    non_recurrent_digest: str


@dataclass(frozen=True)
class BoundCandidate:
    """同一逻辑候选在本控制区间内的当前物理绑定。"""
    candidate_id: str
    node: PhysicalNode


@dataclass(frozen=True)
class PhysicalBinding:
    """绑定逻辑计划、当前生命周期及预期物理视图。"""
    epoch_id: str
    epoch_digest: str
    plan_digest: str
    engine_id: str
    barrier_id: str
    lease_id: str
    view_digest: str
    K: int
    selected_ids: tuple[str, ...]
    evicted_ids: tuple[str, ...]
    candidates: tuple[BoundCandidate, ...]


class StrictRuntime(Protocol):
    """桥接须连续禁止 Admission 和外部状态变更；异常退出保持隔离且不得吞异常。"""
    def control_epoch(self) -> AbstractContextManager[str]: ...
    def observe(self) -> PhysicalView: ...
    def evict_mamba_only(self, handle: RuntimeCheckpointHandle) -> None: ...
    def invalidate(self, ledger: dict) -> None: ...


class StrictReconciliationError(RuntimeError):
    """失败账本包含已完成操作；失败操作本身也可能已部分改变状态。"""
    def __init__(self, ledger):
        self.ledger=ledger
        super().__init__('严格协调失败，必须终止或清理生命周期：'+ledger['error'])


def validate_plan(epoch, plan):
    """只核验给定计划，不重新选择，不调用 Legacy 兼容谓词。"""
    require(type(epoch) is StrictAllocationEpoch and type(plan) is StrictAllocationPlan,'严格输入类型错误')
    epoch.validate()
    require((plan.epoch_id,plan.snapshot_sha256,plan.epoch_digest,plan.K)==(epoch.epoch_id,epoch.snapshot_sha256,epoch.content_digest,epoch.K),'计划与 Epoch 身份或预算不一致')
    ids={c.candidate.checkpoint_id for c in epoch.candidates}
    selected=plan.selected_ids;evicted=plan.logical_eviction_ids
    require(type(selected) is tuple and type(evicted) is tuple,'计划集合必须不可变')
    require(len(set(selected))==len(selected) and set(selected)<=ids,'选中集合重复或越界')
    require(plan.selected_count==len(selected) and len(selected)<=epoch.K,'逻辑预算违反')
    require(tuple(sorted(ids-set(selected)))==evicted and not set(selected)&set(evicted),'驱逐补集错误')
    model=RecoveryCostModel();es=epoch.frontiers(selected)
    expected=tuple(PendingPlan(p.pending_id,p.target,e,p.target-e,model.estimate(p.target-e,p.target)) for p,e in zip(epoch.pendings,es))
    require(plan.pending==expected,'计划前沿或逐请求成本不一致')
    j=sum(p.cost_ms for p in expected);j0=sum(model.estimate(p.target,p.target) for p in epoch.pendings)
    require(abs(plan.recovery_cost_ms-j)<=1e-9 and abs(plan.benefit_ms-(j0-j))<=1e-9,'计划成本不一致')


def validate_view(view, epoch, lease_id):
    """严格限定单池、完整 census、无工作状态的连续静止控制区间。"""
    require(type(view) is PhysicalView,'缺少当前完整物理视图')
    require(all(isinstance(v,str) and v for v in (view.engine_id,view.barrier_id,view.lease_id,view.attention_digest,view.non_recurrent_digest)),'物理身份缺失')
    require(view.lease_id==lease_id and view.configuration_digest==epoch.configuration_digest,'控制区间或配置不一致')
    require(view.idle is True and view.admission_closed is True and view.transfers_pending is False,'存在活动请求、Admission 或传输')
    require(type(view.nodes) is tuple and type(view.free_slots) is tuple,'物理视图必须不可变')
    require(type(view.pool_capacity) is int and view.pool_capacity>=0,'物理池容量错误')
    node_ids=[];slots=[];components=[]
    for n in view.nodes:
        require(type(n) is PhysicalNode and type(n.node_id) is int and n.node_id>=0,'节点身份错误')
        require(type(n.token_ids) is tuple and all(type(t) is int and t>=0 for t in n.token_ids),'节点 Token 非法')
        require(type(n.slots) is tuple and type(n.component_ids) is tuple,'槽位或组件标识必须不可变')
        require(type(n.references) is int and n.references==0 and n.protected is False and n.host_copy is False,'保护引用、主机副本或保护标记未清除')
        require(n.attention_present is True,'Attention 状态不完整')
        if n.slots:
            require(n.conv_complete is True and n.temporal_complete is True and len(n.component_ids)>=2,'Recurrent 组件不完整')
            require(all(isinstance(x,str) and x for x in n.component_ids),'组件身份缺失')
        else:
            require(not n.component_ids and n.conv_complete is False and n.temporal_complete is False,'空槽位仍有可恢复组件')
        node_ids.append(n.node_id);slots.extend(n.slots);components.extend(n.component_ids)
    require(len(node_ids)==len(set(node_ids)),'Node 别名')
    require(len(slots)==len(set(slots)),'Slot 别名')
    require(len(components)==len(set(components)),'共享物理组件')
    require(len(view.free_slots)==len(set(view.free_slots)),'空闲槽位重复')
    require(all(type(v) is int and 1<=v<=view.pool_capacity for v in slots+list(view.free_slots)),'槽位越界')
    require(not set(slots)&set(view.free_slots) and set(slots)|set(view.free_slots)==set(range(1,view.pool_capacity+1)),'槽位记账不闭合')


def bind(epoch, plan, view, lease_id):
    """按当前完整 Token 精确绑定，不读取历史 Node/Slot。"""
    validate_plan(epoch,plan);validate_view(view,epoch,lease_id)
    bound=[]
    for c in epoch.candidates:
        found=[n for n in view.nodes if n.token_ids==c.token_ids]
        require(len(found)==1 and bool(found[0].slots),'候选缺少唯一完整物理映射')
        bound.append(BoundCandidate(c.candidate.checkpoint_id,found[0]))
    require(len({b.node.node_id for b in bound})==len(bound),'多个候选映射同一 Node')
    resident={n.node_id for n in view.nodes if n.slots}
    require(resident=={b.node.node_id for b in bound},'存在候选域外 Recurrent State')
    return PhysicalBinding(epoch.epoch_id,epoch.content_digest,digest(plan),view.engine_id,view.barrier_id,lease_id,digest(view),plan.K,plan.selected_ids,plan.logical_eviction_ids,tuple(bound))


def preflight(epoch,plan,binding,current,lease_id):
    """一次性重新验证全部 Keep/Evict 映射，任何失败发生在首个驱逐之前。"""
    require(bind(epoch,plan,current,lease_id)==binding,'当前 View 或绑定已经失效')
    keep=[b.node for b in binding.candidates if b.candidate_id in binding.selected_ids]
    remove=[b.node for b in binding.candidates if b.candidate_id in binding.evicted_ids]
    for field in ('node_id','slots','component_ids'):
        def values(nodes):
            return {v for n in nodes for v in ((n.node_id,) if field=='node_id' else getattr(n,field))}
        require(not values(keep)&values(remove),'Keep/Evict 物理重叠')


def postcheck(epoch,plan,binding,before,after,completed,lease_id):
    """每次驱逐及最终均验证实际变化；不把后验检查当作预防原语失效。"""
    validate_view(after,epoch,lease_id)
    require((after.engine_id,after.barrier_id,after.lease_id)==(binding.engine_id,binding.barrier_id,binding.lease_id),'协调期间生命周期身份变化')
    require(after.attention_digest==before.attention_digest and after.non_recurrent_digest==before.non_recurrent_digest,'FA 或非目标状态改变')
    require(after.pool_capacity==before.pool_capacity,'池容量变化')
    old={n.node_id:n for n in before.nodes};new={n.node_id:n for n in after.nodes}
    require(old.keys()==new.keys(),'物理节点集合变化')
    target={b.node.node_id for b in binding.candidates if b.candidate_id in completed}
    released=set()
    for node_id,n in old.items():
        current=new[node_id]
        if node_id not in target:
            require(current==n,'Selected 或非目标物理状态变化')
        else:
            require(current.token_ids==n.token_ids and current.attention_present==n.attention_present,'驱逐改变 Token 或 Attention')
            require(not current.slots and not current.component_ids,'未选中 Recurrent 状态仍驻留')
            released.update(n.slots)
    require(set(after.free_slots)==set(before.free_slots)|released,'释放槽位记账错误')
    resident={b.candidate_id for b in binding.candidates if new[b.node.node_id].slots}
    require(resident=={b.candidate_id for b in binding.candidates}-set(completed),'驻留集合与已完成操作不符')
    if tuple(completed)==binding.evicted_ids:
        require(resident==set(plan.selected_ids) and len(resident)<=plan.K,'最终驻留或预算错误')


class StrictReconciler:
    """执行一次性严格计划；失败后不重试、不回滚、不继续恢复请求。"""
    def reconcile_strict(self,epoch,plan,runtime: StrictRuntime):
        ledger={'status':'RUNNING','phase':'enter','completed_evictions':[],'failed_checkpoint_id':None,'epoch_id':getattr(epoch,'epoch_id',None),'binding':None,'last_view':None,'error':None,'cleanup_required':False}
        def fail(error):
            ledger.update(status='FAIL',error=str(error),cleanup_required=True)
            try:ledger['failure_view']=asdict(runtime.observe())
            except Exception as observation_error:ledger['failure_view_error']=str(observation_error)
            try:runtime.invalidate(ledger)
            except Exception as invalidation_error:ledger['invalidation_error']=str(invalidation_error)
            raise StrictReconciliationError(ledger) from error

        try:
            with runtime.control_epoch() as lease_id:
                try:
                    ledger['phase']='binding';before=runtime.observe();ledger['last_view']=asdict(before)
                    binding=bind(epoch,plan,before,lease_id);ledger['binding']=asdict(binding)
                    ledger['phase']='preflight';preflight(epoch,plan,binding,runtime.observe(),lease_id)
                    by_id={b.candidate_id:b.node for b in binding.candidates}
                    for cid in binding.evicted_ids:
                        ledger['phase']='before_operation';ledger['failed_checkpoint_id']=cid
                        postcheck(epoch,plan,binding,before,runtime.observe(),ledger['completed_evictions'],lease_id)
                        node=by_id[cid]
                        prefix_digest=next(c.token_digest for c in epoch.candidates if c.candidate.checkpoint_id==cid)
                        handle=RuntimeCheckpointHandle(cid,node.token_ids,expected_node_id=node.node_id,expected_prefix_digest=prefix_digest)
                        ledger['phase']='eviction';runtime.evict_mamba_only(handle)
                        ledger['completed_evictions'].append(cid)
                        ledger['phase']='postcheck';current=runtime.observe();ledger['last_view']=asdict(current)
                        postcheck(epoch,plan,binding,before,current,ledger['completed_evictions'],lease_id)
                    ledger['phase']='final_postcheck';current=runtime.observe();ledger['last_view']=asdict(current)
                    postcheck(epoch,plan,binding,before,current,ledger['completed_evictions'],lease_id)
                    ledger.update(status='PASS',failed_checkpoint_id=None,phase='complete')
                    return ledger
                except Exception as error:
                    fail(error)
        except StrictReconciliationError:
            raise
        except Exception as error:
            fail(error)


class StrictRuntimeBridge:
    """把已隔离的观测设施接到现有 Adapter；不自行启动 Engine 或伪造隔离。"""
    def __init__(self, adapter, control_epoch, observe, invalidate):
        self.adapter=adapter
        self._control_epoch=control_epoch
        self._observe=observe
        self._invalidate=invalidate
        self._used=False
        self._invalid=False
        self._held=False

    @contextmanager
    def control_epoch(self):
        require(not self._used and not self._invalid, '桥接控制区间已经消费或失效')
        self._used=True
        with self._control_epoch() as lease_id:
            self._held=True
            try:
                yield lease_id
            finally:
                self._held=False

    def observe(self):
        return self._observe()

    def evict_mamba_only(self, handle):
        require(self._held and not self._invalid, '驱逐必须处于有效连续控制区间')
        return self.adapter.evict_mamba_only(handle)

    def invalidate(self, ledger):
        self._invalid=True
        return self._invalidate(ledger)
