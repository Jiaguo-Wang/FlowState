"""记录单个Pending的真实Admission、匹配、复制与执行，不改变原生决策。"""
import json
import os
import time
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from evaluation.runtime_control_closure_r2c1a import bridge as reconcile
from evaluation.runtime_control_closure_r2c0 import bridge as view_bridge
from evaluation.strict_compatibility_b1c import runner as native
require=reconcile.require
CURRENT=None
TARGET=None
def load():
    require(CURRENT is not None,'缺少当前实时Epoch')
    return CURRENT
TRACE={'admission':[],'matches':[],'copies':[],'execution':[],'housekeeping':[],'compute':[]}
BASELINE=None
ARM=os.environ.get('B5D4_ARM','strict')
RUN=os.environ.get('B5D4_RUN','cpu')
if ARM=='cold':TARGET='B5C2_C_TURN_003'

def save():
    path=Path('/b5out')/RUN/'native_trace.json'
    start=time.monotonic_ns();path.write_text(json.dumps(TRACE,ensure_ascii=False,indent=2));TRACE['audit_write_ns']=TRACE.get('audit_write_ns',0)+time.monotonic_ns()-start

def _cache_state(cache):
    """直接读树和张量身份，可在实际匹配前调用，不要求把活动请求伪装为空闲。"""
    import targeted_probe as probe
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from evaluation.barrier_fa_frontier_control import semantic_cache_snapshot
    nodes=[]
    for nid,node in probe._tree_items(cache.tree_core):
        parts=[];cur=node
        while cur.parent is not None:
            require(not cur.key.is_bigram,'不支持bigram');parts.append(list(cur.key));cur=cur.parent
        tokens=tuple(t for part in reversed(parts) for t in part)
        cd=node.component_data[CT.MAMBA];full=node.component_data[CT.FULL]
        nodes.append({'node_id':nid,'token_ids':list(tokens),'slots':native.tensor_ids(cd.value),'full_present':full.value is not None,'references':[int(cd.lock_ref),int(cd.host_lock_ref),int(cd.session_ref)]})
    tree=probe._global_maps(cache);accounting=probe._accounting_snapshot(cache)
    semantic=semantic_cache_snapshot(cache,component_type=CT,tree_snapshot=tree,accounting_snapshot=accounting)
    result={'nodes':nodes,'semantic':semantic,'ns':time.monotonic_ns()}
    if ARM=='strict' and BASELINE is not None:
        _,epoch,plan=load()
        view=SimpleNamespace(nodes=[SimpleNamespace(token_ids=tuple(n['token_ids']),slots=tuple(n['slots'])) for n in nodes])
        result['selected_payload']=reconcile.selected_payload(cache,view,plan,epoch)
        result['selected_components']={}
        tensors=cache.req_to_token_pool.mamba_pool.mamba_cache
        for c in epoch.candidates:
            cid=c.candidate.checkpoint_id
            if cid not in plan.selected_ids:continue
            n=next(n for n in view.nodes if n.token_ids==c.token_ids)
            result['selected_components'][cid]=[view_bridge.slice_identity(t,s)[0] for s in n.slots for t in list(tensors.conv)+[tensors.temporal]]
    if ARM=='strict' and BASELINE is not None:
        from .control import fa_map,fa_payload
        result['fa_map']=fa_map(cache)
        result['fa_protected_payload']=fa_payload(cache,BASELINE['fa_protected_indices'])
    return result

def _slot_payload(pool,slots):
    import torch
    state=pool.mamba_cache
    return [[hashlib.sha256(t[:,s].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for t in list(state.conv)+[state.temporal]] for s in slots]

def cache_state(cache):
    start=time.monotonic_ns();result=_cache_state(cache)
    TRACE.setdefault('audit',[]).append({'kind':'cache_state_full_hash','ns':time.monotonic_ns()-start})
    return result

def slot_payload(pool,slots):
    start=time.monotonic_ns();result=_slot_payload(pool,slots)
    TRACE.setdefault('audit',[]).append({'kind':'cow_content_hash','ns':time.monotonic_ns()-start})
    return result

def install_observation(cache):
    native.install_observation(cache)
    original_match=cache.match_prefix;pool=cache.req_to_token_pool.mamba_pool;original_copy=pool.copy_from
    def match(params):
        req=params.req;target=req is not None and str(req.rid)==TARGET
        record=None
        if target:
            record={'ns_before':time.monotonic_ns(),'cow_requested':bool(params.cow_mamba),'before':cache_state(cache)}
            TRACE['matches'].append(record);save()
        call_start=time.monotonic_ns();result=original_match(params);call_end=time.monotonic_ns()
        if target:
            record.update(ns_after=time.monotonic_ns(),native_call_ns=call_end-call_start,native=native.MATCHES[TARGET][-1]);save()
        return result
    def copy(src,dst):
        ss=native.tensor_ids(src);ds=native.tensor_ids(dst)
        target=any(native.COPY_OWNERS.get((s,d))==TARGET for s,d in zip(ss,ds))
        row=None
        if target:
            row={'source_slots':ss,'destination_slots':ds,'ns_before':time.monotonic_ns(),'source_payload_before':slot_payload(pool,ss)}
            TRACE['copies'].append(row);save()
        if target:
            import torch
            torch.cuda.synchronize();event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True);event_start.record()
        call_start=time.monotonic_ns();value=original_copy(src,dst);call_end=time.monotonic_ns()
        if target:event_end.record();event_end.synchronize()
        if target:
            row.update(ns_after=time.monotonic_ns(),native_host_ns=call_end-call_start,cuda_elapsed_ms=event_start.elapsed_time(event_end),returned=True,destination_payload_after=slot_payload(pool,ds),source_payload_after=slot_payload(pool,ss));save()
        return value
    cache.match_prefix=match;pool.copy_from=copy

def scheduler_process(*args,**kwargs):
    """专用进程只允许实时协调一次及只读观察；失败保持隔离。"""
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    from . import observation
    cls=transport._scheduler_module.Scheduler
    old_idle=cls.on_idle;old_input=cls.process_input_requests;old_run=cls.run_batch
    def idle(self):
        if getattr(self,'_r2c0_failed',False):
            print('[B5D4] 失败隔离，等待本轮清理',flush=True);threading.Event().wait()
        return old_idle(self)
    def inputs(self,requests):
        targets=[r for r in requests if str(getattr(r,'rid',''))==TARGET]
        if targets:
            try:
                require(len(targets)==1 and not TRACE['admission'],'目标重复Admission')
                probe._validate_runtime_scope(self,self.tree_cache)
                current=cache_state(self.tree_cache)
                received=list(targets[0].input_ids)
                TRACE['admission'].append({'state':current,'ns':time.monotonic_ns(),'input_ids':received})
                if ARM=='strict':
                    require(BASELINE is not None,'未完成协调')
                    from .protocol import admission_gate
                    admission_gate(BASELINE,current,received,BASELINE['target']['pending']['token_ids'])
                    require(current['fa_map']==BASELINE['fa_map_before'],'Admission前FA映射变化')
                    require(current['fa_protected_payload']==BASELINE['fa_protected_payload'],'Admission前受保护FA内容变化')
                save()
            except BaseException as exc:
                TRACE['admission_error']=repr(exc);save();self._r2c0_failed=True;raise
        return old_input(self,requests)
    observation.TARGET=TARGET;observation.COMPUTE=TRACE['compute']
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    EagerRunner.execute=observation.wrap_eager(EagerRunner.execute,save)
    cls.on_idle=idle;cls.process_input_requests=inputs;cls.run_batch=observation.wrap_run_batch(old_run,TRACE['execution'],save)
    def dispatch(scheduler,request):
        global BASELINE,CURRENT,TARGET
        require(not getattr(scheduler,'_r2c0_failed',False),'生命周期已失效')
        action=request.get('action')
        if action=='b5d4_apply':
            require(ARM=='strict' and BASELINE is None,'禁止重复协调')
            from .control import apply
            result,epoch,plan=apply(scheduler,request)
            if result.get('ok'):
                CURRENT=(result['raw_epoch'],epoch,plan);TARGET=result['target']['pending']['continuation_id'];observation.TARGET=TARGET
                result['observer_api_preflight']=observation.real_api_preflight()
                BASELINE=result;result['admission_baseline']=cache_state(scheduler.tree_cache)
            return result
        if action=='b5d4_state':
            probe._validate_runtime_scope(scheduler,scheduler.tree_cache)
            return {'ok':True,'state':cache_state(scheduler.tree_cache),'trace':TRACE,'observer_api_preflight':observation.real_api_preflight()}
        if action=='b5c3_observe':
            require(BASELINE is None,'协调后不重新构造分配快照')
            from evaluation.multishare_runtime_gate_b5c3.bridge import observe_control
            return observe_control(scheduler,request)
        raise ValueError('未授权动作：'+str(action))
    transport._run_checkpoint_control=dispatch;transport._install_match_instrumentation=install_observation
    return transport._wrapped_run_scheduler_process(*args,**kwargs)
