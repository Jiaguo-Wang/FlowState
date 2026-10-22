"""仅在单次调度器同步调用内观察、绑定与预检查，不提供释放入口。"""
import hashlib
import json
import os
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from flowstate.strict_control import PhysicalNode,PhysicalView,bind,preflight
from flowstate.strict_epoch import StrictAllocationEpoch
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001/snapshot.json'

def require(v,message):
    if not v:raise RuntimeError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load():
    expected=json.loads((ROOT/'evaluation/strict_compatibility_b4c/input_integrity.json').read_text())['sha256'][str(RAW)]
    epoch=StrictAllocationEpoch.from_snapshot_bytes(RAW.read_bytes(),expected_sha256=expected,K=2)
    plan=GlobalOptimizer(RecoveryCostModel()).plan_strict(epoch)
    require(set(plan.selected_ids)=={'RQ3_G001_B_TURN_001','RQ3_G001_B_TURN_003'},'冻结计划选集不符')
    return json.loads(RAW.read_text()),epoch,plan

def slice_identity(t,slot):
    """限制为标准连续层槽布局；逐层字节区间共同标识真正的存储切片。"""
    require(t.is_contiguous() and t.ndim>=3 and 0<slot<t.shape[1],'不支持当前组件布局')
    width=t.stride(1)*t.element_size()
    storage=t.untyped_storage();base=storage.data_ptr()
    ranges=[(base+(t.storage_offset()+layer*t.stride(0)+slot*t.stride(1))*t.element_size(),width) for layer in range(t.shape[0])]
    require(all(start>=base and start+size<=base+storage.nbytes() for start,size in ranges),'组件切片越界')
    info={'storage_base':base,'storage_bytes':storage.nbytes(),'slot':slot,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':str(t.device),'byte_ranges':ranges}
    return digest(info),info

def assert_nonoverlap(components):
    ranges=sorted((start,start+size,cid) for cid,info in components.items() for start,size in info['byte_ranges'])
    for a,b in zip(ranges,ranges[1:]):require(a[1]<=b[0],'组件实际字节区间重叠')

def scope(scheduler):
    """线程独占由同步调用栈提供；标志仅核对当前调用归属。"""
    import targeted_probe as probe
    require(getattr(scheduler,'_r2c0_owner',None)==threading.get_ident(),'不在连续控制调用内')
    require(not getattr(scheduler,'_r2c0_failed',False),'控制生命周期已失效')
    value=probe._validate_runtime_scope(scheduler,scheduler.tree_cache)
    cache=scheduler.tree_cache
    for name in ('ongoing_write_through','ongoing_load_back','ongoing_prefetch','ongoing_backup'):
        require(hasattr(cache,name) and not getattr(cache,name),'传输状态缺失或未完成：'+name)
    for component in cache.components.values():
        require(hasattr(component,'is_evict_device_ongoing') and component.is_evict_device_ongoing is False,'驱逐状态缺失或未完成')
    for name in ('enable_hisparse','enable_hierarchical_cache','enable_unified_memory'):
        require(hasattr(scheduler,name) and getattr(scheduler,name) is False,'异步或未知功能：'+name)
    require(hasattr(scheduler,'external_corpus_manager') and scheduler.external_corpus_manager is None,'外部语料管理未禁用')
    require(str(scheduler.disaggregation_mode).lower() in ('null','disaggregationmode.null'),'分离式传输未禁用')
    require(scheduler.tree_cache.req_to_token_pool.mamba_pool.enable_linear_replayssm is False,'额外回放状态未禁用')
    require(scheduler.tree_cache.req_to_token_pool.mamba_pool.enable_linear_replayssm_spec is False,'额外推测状态未禁用')
    return value

def observe(scheduler,epoch,window):
    import targeted_probe as probe
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from evaluation.strict_compatibility_b1c.runner import state_snapshot
    from evaluation.barrier_fa_frontier_control import semantic_cache_snapshot
    s=scope(scheduler);cache=scheduler.tree_cache;core=cache.tree_core
    state=state_snapshot(scheduler)
    sem=semantic_cache_snapshot(cache,component_type=CT,tree_snapshot=probe._global_maps(cache),accounting_snapshot=probe._accounting_snapshot(cache))
    tensors=cache.req_to_token_pool.mamba_pool.mamba_cache
    nodes=[];components={};rows={r['node_id']:r for r in state['nodes']};detail=[]
    for node_id,node in probe._tree_items(core):
        if node.parent is None:continue
        r=rows[node_id];ids=[]
        full,mamba=node.component_data[CT.FULL],node.component_data[CT.MAMBA]
        require(node.key.extra_key is None and not node.key.is_bigram,'不支持的Cache Key配置')
        refs=sum(int(getattr(cd,k)) for cd in (full,mamba) for k in ('lock_ref','host_lock_ref','session_ref'))
        slots=tuple(r['mamba_slots'])
        for slot in slots:
            for t in list(tensors.conv)+[tensors.temporal]:
                cid,info=slice_identity(t,slot);require(cid not in components,'重复组件切片')
                components[cid]=info;ids.append(cid)
        protected=refs!=0
        detail.append({'node_id':node_id,'references':refs,'full_host':full.host_value is not None,'mamba_host':mamba.host_value is not None,'mamba_lru':core.lru_lists[CT.MAMBA].in_list(node),'priority':node.priority})
        nodes.append(PhysicalNode(node_id,tuple(r['token_ids']),slots,tuple(ids),bool(slots) and bool(tensors.conv),bool(slots),r['full_present'] and r['full_count']==len(node.key),refs,protected,full.host_value is not None or mamba.host_value is not None))
    assert_nonoverlap(components)
    accounting=state['accounting'];pool=cache.req_to_token_pool.mamba_pool
    require(pool.size==28,'Recurrent Pool容量变化')
    # 非Recurrent摘要不含预计会在未来协调中改变的Mamba LRU与计数。
    nonrec={'full_allocator':accounting['full_allocator'],'full_protected':accounting['full_protected'],'full_evictable':accounting['full_evictable'],'recency':sem['recency_rows'],'full_leaves':sem['full_evictable_leaf_ids'],'full_references':[r[:4] for r in sem['reference_rows']]}
    view=PhysicalView(f'r2c0-pid{os.getpid()}',window,window,epoch.configuration_digest,True,True,False,tuple(sorted(nodes,key=lambda n:n.node_id)),pool.size,tuple(accounting['mamba_free_slots']),state['tree']['full_tree_sha256'],digest(nonrec))
    return view,{'scope':s,'state':state,'semantic':sem,'component_slices':components,'node_protection':detail,'field_sources':{'tokens':'沿当前Radix父链读取key','slots':'当前MAMBA.value','components':'当前MambaPool的Conv和Temporal逐层逐槽字节区间','references':'FULL/MAMBA的lock/host_lock/session_ref','admission':'普通Scheduler循环内单次同步控制调用阻止下一轮recv/process','transfers':'禁用HiCache/HiSparse/分离式传输/额外ReplaySSM并同步GPU','generation':'UNKNOWN；仅在同一连续调用内有效'}}

def control(scheduler,request):
    """本次单RPC区间覆盖所有安全读取；异常后由on_idle保持停机隔离。"""
    import torch
    from evaluation.barrier_fa_frontier_control import inspect_resident_fa_frontier
    from evaluation.strict_compatibility_b4b0.contracts import build_edges
    from evaluation.strict_compatibility_b1c.runner import MATCHES
    import copy
    result={'ok':False,'status':'BLOCKED','destructive_calls':0,'events':[]}
    scheduler._r2c0_owner=threading.get_ident()
    window=f'r2c0-{os.getpid()}-{time.monotonic_ns()}'
    try:
        raw,epoch,plan=load();scope(scheduler)
        effective={k:getattr(scheduler.server_args,k) for k in raw['configuration']['engine_configuration']}
        require(effective==raw['configuration']['engine_configuration'],'当前实际Engine配置与冻结配置不一致')
        result['effective_configuration']=effective
        torch.cuda.synchronize()
        result['events'].append({'event':'进入单次Scheduler同步控制区间','ns':time.monotonic_ns(),'thread':threading.get_ident()})
        before,audit=observe(scheduler,epoch,window);result['runtime_view_audit']={'before':asdict(before),'audit_before':audit}
        candidates=copy.deepcopy(raw['candidates']);pendings=copy.deepcopy(raw['pendings'])
        mapping=[]
        for c in candidates:
            matches=[n for n in before.nodes if n.token_ids==tuple(c['token_ids'])]
            require(len(matches)==1 and matches[0].slots,'当前候选不能唯一定位')
            n=matches[0];c['barrier_id']=window
            observed=next(r for r in audit['state']['nodes'] if r['node_id']==n.node_id)
            ref_names={'target_full_lock_ref':'full_lock','target_mamba_lock_ref':'mamba_lock','target_mamba_host_lock_ref':'mamba_host_lock','target_mamba_session_ref':'mamba_session'}
            c['state_identity']={'node_id':n.node_id,'target_mamba_slots':list(n.slots),'target_mamba_present':bool(n.slots),'target_full_present':n.attention_present,'path_full_all_present':all(x.attention_present for x in before.nodes if n.token_ids[:len(x.token_ids)]==x.token_ids),'references':{k:observed['references'][v] for k,v in ref_names.items()}}
            require(n.references==0 and not n.protected and not n.host_copy,'候选保护条件不成立')
            mapping.append({'checkpoint_id':c['checkpoint_id'],'current_node':n.node_id,'current_slots':n.slots,'historical_node':next(x for x in raw['candidates'] if x['checkpoint_id']==c['checkpoint_id'])['state_identity']['node_id']})
        frontiers=[]
        for p in pendings:
            p['barrier_id']=window
            fa=inspect_resident_fa_frontier(scheduler.tree_cache,p['token_ids'])
            require(fa['resident_fa_frontier']==p['attention_frontier'],'当前Attention前沿与冻结输入不一致')
            require(p['planning_target_T']<=fa['resident_fa_frontier'] and p['planning_target_T']<=len(p['token_ids']),'规划坐标不一致')
            frontiers.append({'pending_id':p['continuation_id'],'T':p['planning_target_T'],'FA':fa,'说明':'冻结已知输入，本轮不提交Pending'})
        edges=build_edges(candidates,pendings,audit['state'],epoch.configuration_digest)
        key=lambda e:(e['checkpoint_id'],e['continuation_id'])
        require({key(e):(e['status'],e['conditions']) for e in edges}=={key(e):(e['status'],e['conditions']) for e in raw['edges']},'当前严格准入矩阵与冻结条件不一致')
        result.update({'plan':asdict(plan),'current_mapping':mapping,'pending_frontiers':frontiers,'edges':edges})
        bound=bind(epoch,plan,before,window)
        current,middle=observe(scheduler,epoch,window);preflight(epoch,plan,bound,current,window)
        after,last=observe(scheduler,epoch,window)
        require(before==current==after and audit['semantic']==middle['semantic']==last['semantic'],'只读操作改变完整物理视图')
        require(audit['component_slices']==last['component_slices'],'组件存储切片变化')
        scope(scheduler)
        result['runtime_view_audit'].update({'after':asdict(after),'audit_after':last})
        result.update({'ok':True,'status':'READY','binding':asdict(bound),'unchanged':True,'continuous_window':window,'binding_pass':True,'preflight_pass':True})
        result['events'].append({'event':'同一同步区间内全部检查通过','ns':time.monotonic_ns(),'thread':threading.get_ident()})
    except BaseException as exc:
        scheduler._r2c0_failed=True
        result.update({'error':repr(exc),'traceback':traceback.format_exc(),'failure_isolation':'返回结果后on_idle停驻，等待本轮父进程终止生命周期'})
    finally:
        scheduler._r2c0_owner=None
    return result

def scheduler_process(*args,**kwargs):
    """只在当前子进程安装同步只读入口，不修改SGLang磁盘源码。"""
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    from evaluation.strict_compatibility_b1c.runner import install_observation,MATCHES,COPIES
    cls=transport._scheduler_module.Scheduler
    idle=cls.on_idle
    def guarded_idle(self):
        if getattr(self,'_r2c0_failed',False):
            print('[R2C0] 失败隔离已保持，等待生命周期终止',flush=True)
            threading.Event().wait()
            raise RuntimeError('失败隔离不得恢复调度')
        return idle(self)
    cls.on_idle=guarded_idle
    old=transport._run_checkpoint_control
    def dispatch(scheduler,request):
        action=request.get('action')
        if action=='r2c0_readonly':return control(scheduler,request)
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
