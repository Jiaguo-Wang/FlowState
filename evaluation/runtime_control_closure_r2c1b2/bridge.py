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
ROOT,RAW,load,require=reconcile.ROOT,reconcile.RAW,reconcile.load,reconcile.require
TARGET='RQ3_G001_A_PENDING_TURN_004'
SOURCE='RQ3_G001_B_TURN_001'
TRACE={'admission':[],'matches':[],'copies':[],'execution':[],'housekeeping':[]}
BASELINE=None
ARM=os.environ.get('R2_ARM','strict')

def save():
    path=Path('/r2out')/ARM/'native_trace.json'
    path.write_text(json.dumps(TRACE,ensure_ascii=False,indent=2))

def cache_state(cache):
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
    return result

def slot_payload(pool,slots):
    import torch
    state=pool.mamba_cache
    return [[hashlib.sha256(t[:,s].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for t in list(state.conv)+[state.temporal]] for s in slots]

def install_observation(cache):
    native.install_observation(cache)
    original_match=cache.match_prefix;pool=cache.req_to_token_pool.mamba_pool;original_copy=pool.copy_from
    def match(params):
        req=params.req;target=req is not None and str(req.rid)==TARGET
        record=None
        if target:
            record={'ns_before':time.monotonic_ns(),'cow_requested':bool(params.cow_mamba),'before':cache_state(cache)}
            TRACE['matches'].append(record);save()
        result=original_match(params)
        if target:
            record.update(ns_after=time.monotonic_ns(),native=native.MATCHES[TARGET][-1]);save()
        return result
    def copy(src,dst):
        ss=native.tensor_ids(src);ds=native.tensor_ids(dst)
        target=any(native.COPY_OWNERS.get((s,d))==TARGET for s,d in zip(ss,ds))
        row=None
        if target:
            row={'source_slots':ss,'destination_slots':ds,'ns_before':time.monotonic_ns(),'source_payload_before':slot_payload(pool,ss)}
            TRACE['copies'].append(row);save()
        value=original_copy(src,dst)
        if target:
            row.update(ns_after=time.monotonic_ns(),returned=True,destination_payload_after=slot_payload(pool,ds),source_payload_after=slot_payload(pool,ss));save()
        return value
    cache.match_prefix=match;pool.copy_from=copy

def scheduler_process(*args,**kwargs):
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    cls=transport._scheduler_module.Scheduler
    old_idle=cls.on_idle;old_input=cls.process_input_requests;old_run=cls.run_batch
    def idle(self):
        if getattr(self,'_r2c0_failed',False):
            print('[R2C1b2] 失败隔离，等待生命周期清理',flush=True);threading.Event().wait()
        active=BASELINE is not None and not TRACE['admission'] and not TRACE['housekeeping']
        before=cache_state(self.tree_cache) if active else None
        value=old_idle(self)
        if active:
            after=cache_state(self.tree_cache)
            event={'before_ns':before['ns'],'after_ns':after['ns'],'semantic_unchanged':before['semantic']==after['semantic'],'selected_unchanged':before.get('selected_payload')==after.get('selected_payload')}
            if not TRACE['housekeeping'] or not all((event['semantic_unchanged'],event['selected_unchanged'])):TRACE['housekeeping'].append(event);save()
        return value
    def inputs(self,requests):
        targets=[r for r in requests if str(getattr(r,'rid',''))==TARGET]
        if targets:
            try:
                require(len(targets)==1 and not TRACE['admission'],'目标请求重复Admission')
                probe._validate_runtime_scope(self,self.tree_cache)
                current=cache_state(self.tree_cache)
                TRACE['admission'].append({'state':current,'ns':time.monotonic_ns(),'input_ids':list(targets[0].input_ids)})
                if ARM=='strict':
                    require(BASELINE is not None,'未完成严格协调')
                    from .protocol import admission_gate
                    frozen,_,_=load()
                    expected=next(p['token_ids'] for p in frozen['pendings'] if p['continuation_id']==TARGET)
                    admission_gate(BASELINE,current,list(targets[0].input_ids),expected)
                save()
            except BaseException as exc:
                TRACE['admission_error']=repr(exc);save();self._r2c0_failed=True;raise
        return old_input(self,requests)
    from .observation import wrap_run_batch
    run=wrap_run_batch(old_run,TRACE['execution'],save)
    cls.on_idle=idle;cls.process_input_requests=inputs;cls.run_batch=run
    old=transport._run_checkpoint_control
    def dispatch(scheduler,request):
        global BASELINE
        action=request.get('action')
        if action=='r2c1b2_reconcile':
            require(BASELINE is None,'禁止重复协调')
            from .observation import real_api_preflight
            api=real_api_preflight()
            result=reconcile.control(scheduler,request)
            result['observer_api_preflight']=api
            if result.get('ok'):
                BASELINE=result
                result['admission_baseline']=cache_state(scheduler.tree_cache)
            return result
        if action=='r2c1b2_state':
            probe._validate_runtime_scope(scheduler,scheduler.tree_cache)
            from .observation import real_api_preflight
            api=real_api_preflight()
            return {'observer_api_preflight':api,'ok':True,'state':cache_state(scheduler.tree_cache),'trace':TRACE}
        require(not getattr(scheduler,'_r2c0_failed',False),'生命周期失效')
        if action=='flowstate_runtime_metrics':
            rows=native.MATCHES.get(str(request['request_id']),[]);require(bool(rows),'缺少原生匹配记录')
            return {'ok':True,'metrics':rows[-1],'records':rows}
        require(action in ('inspect','inspect_fa_frontier'),'不允许此控制动作')
        return old(scheduler,request)
    transport._run_checkpoint_control=dispatch;transport._install_match_instrumentation=install_observation
    return transport._wrapped_run_scheduler_process(*args,**kwargs)
