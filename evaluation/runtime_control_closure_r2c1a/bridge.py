"""单次Scheduler同步窗口内执行正式严格协调，禁止Pending恢复。"""
import hashlib
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from evaluation.runtime_control_closure_r2c0 import bridge as prior
from flowstate.adapters.sglang import SGLangAdapter
from flowstate.controller import StateController
from flowstate.strict_control import StrictRuntimeBridge,StrictReconciliationError
ROOT,RAW,load,require=prior.ROOT,prior.RAW,prior.load,prior.require

def selected_payload(cache,view,plan,epoch):
    """只读核对选中状态数值，不清空或改写设备张量。"""
    import torch
    tensors=cache.req_to_token_pool.mamba_pool.mamba_cache
    result={}
    for c in epoch.candidates:
        cid=c.candidate.checkpoint_id
        if cid not in plan.selected_ids:continue
        node=next(n for n in view.nodes if n.token_ids==c.token_ids)
        hashes=[]
        for slot in node.slots:
            for t in list(tensors.conv)+[tensors.temporal]:
                data=t[:,slot].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                hashes.append(hashlib.sha256(data).hexdigest())
        result[cid]=hashes
    return result

class RecordedAdapter(SGLangAdapter):
    """保留生产Adapter检查，只在原调用前后记录实际返回或异常。"""
    def __init__(self,cache,operations):
        super().__init__(cache);self.operations=operations
    def evict_mamba_only(self,handle):
        row={'checkpoint_id':handle.checkpoint_id,'node_id':handle.expected_node_id,'prefix_digest':handle.expected_prefix_digest,'start_ns':time.monotonic_ns(),'returned':False}
        self.operations.append(row)
        try:
            result=super().evict_mamba_only(handle);row['returned']=True;return result
        except BaseException as exc:
            row['error']=repr(exc);raise
        finally:row['end_ns']=time.monotonic_ns()

def control(scheduler,request):
    """复用只读门禁后不返回Scheduler主循环，直接进入生产协调。"""
    result={'ok':False,'status':'BLOCKED','operations':[],'observations':[],'pending_submitted':0}
    try:
        initial=readonly_control(scheduler,request)
        result['initial_preflight']=initial
        require(initial.get('ok') is True,'当前新生命周期的只读门禁失败')
        # 上述普通函数返回不退出本次drain_one，Scheduler仍不能接收新请求。
        scheduler._r2c0_owner=threading.get_ident()
        window=initial['continuous_window']
        raw,epoch,plan=load()
        require(asdict(plan)==initial['plan'],'同窗口内正式计划发生变化')
        prior.scope(scheduler)
        before,audit_before=prior.observe(scheduler,epoch,window)
        result['before']=asdict(before);result['audit_before']=audit_before
        result['selected_payload_before']=selected_payload(scheduler.tree_cache,before,plan,epoch)
        @contextmanager
        def held():
            prior.scope(scheduler)
            result['window_enter_ns']=time.monotonic_ns()
            try:yield window
            finally:result['window_exit_ns']=time.monotonic_ns()
        def observe():
            view,audit=prior.observe(scheduler,epoch,window)
            require(audit['state']['tree']['structure_sha256']==audit_before['state']['tree']['structure_sha256'],'全树结构发生变化')
            result['observations'].append({'ns':time.monotonic_ns(),'view':asdict(view),'semantic':audit['semantic']})
            return view
        def invalidate(ledger):
            scheduler._r2c0_failed=True
            result['invalidated_while_in_scheduler_call']=True
        adapter=RecordedAdapter(scheduler.tree_cache,result['operations'])
        runtime=StrictRuntimeBridge(adapter,held,observe,invalidate)
        ledger=StateController.reconcile_strict(epoch,plan,runtime)
        result['reconciliation']=ledger
        after,audit_after=prior.observe(scheduler,epoch,window)
        require(audit_after['state']['tree']['structure_sha256']==audit_before['state']['tree']['structure_sha256'],'最终全树结构发生变化')
        result['after']=asdict(after);result['audit_after']=audit_after
        result['selected_payload_after']=selected_payload(scheduler.tree_cache,after,plan,epoch)
        require(result['selected_payload_before']==result['selected_payload_after'],'Selected组件数值被改变')
        require([x['checkpoint_id'] for x in result['operations']]==list(plan.logical_eviction_ids),'实际驱逐目标与补集不一致')
        require(all(x['returned'] for x in result['operations']),'存在未返回成功的驱逐')
        prior.scope(scheduler)
        result.update(ok=True,status='READY',plan=asdict(plan),continuous_window=window)
    except BaseException as exc:
        if isinstance(exc,StrictReconciliationError):result['reconciliation']=exc.ledger
        scheduler._r2c0_failed=True
        result.update(error=repr(exc),traceback=traceback.format_exc(),cleanup_required=True)
    finally:scheduler._r2c0_owner=None
    return result


readonly_control=prior.control

def scheduler_process(*args,**kwargs):
    """只在当前子进程安装同步只读入口，不修改SGLang磁盘源码。"""
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    from evaluation.strict_compatibility_b1c.runner import install_observation,MATCHES,COPIES
    cls=transport._scheduler_module.Scheduler
    idle=cls.on_idle
    def guarded_idle(self):
        if getattr(self,'_r2c0_failed',False):
            print('[R2C1a] 失败隔离已保持，等待生命周期终止',flush=True)
            threading.Event().wait()
            raise RuntimeError('失败隔离不得恢复调度')
        return idle(self)
    cls.on_idle=guarded_idle
    old=transport._run_checkpoint_control
    def dispatch(scheduler,request):
        action=request.get('action')
        if action=='r2c1a_reconcile':return control(scheduler,request)
        require(not getattr(scheduler,'_r2c0_failed',False),'生命周期失效')
        if action=='flowstate_runtime_metrics':
            probe._validate_runtime_scope(scheduler,scheduler.tree_cache)
            rows=MATCHES.get(str(request['request_id']),[]);require(bool(rows),'缺少实际输入接收记录')
            return {'ok':True,'metrics':rows[-1],'records':rows}
        require(action in ('inspect_fa_frontier','inspect'),'只读入口拒绝动作：'+str(action))
        return old(scheduler,request)
    transport._run_checkpoint_control=dispatch
    transport._install_match_instrumentation=install_observation
    return transport._wrapped_run_scheduler_process(*args,**kwargs)
