"""保留原验证及驱逐原语，通过批量只读查询和epoch句柄复用减少控制往返。"""
from time import perf_counter_ns
import uuid
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
import rq6_batched_control_transport as base
import wp3b_end_to_end_transport as frontier
import targeted_probe as probe
from evaluation.control_path_codec import unpack

_ORIGINAL_DISPATCH=base._run_checkpoint_control
_ORIGINAL_PARSE=base._parse_handles


class EpochHandles:
    """只持有当前引擎当前epoch内已完整验证的不可变句柄。"""
    def __init__(self,handles,view_digest):
        self.token=uuid.uuid4().hex
        self.handles=dict(handles)
        self.view_digest=view_digest
        self.consumed=False

    def take(self,token,candidate_ids,view_digest):
        """拒绝过期、重复消费、错序、错集合或非本决策视图的引用。"""
        if self.consumed or token!=self.token:
            raise RuntimeError('epoch句柄引用过期或身份错误')
        if list(candidate_ids)!=list(self.handles) or view_digest!=self.view_digest:
            raise RuntimeError('epoch候选映射或决策视图不一致')
        self.consumed=True
        return self.handles


def semantic_state(cache):
    """读取原有全部语义状态，保留树、allocator、引用、LRU及recency。"""
    return frontier.semantic_cache_snapshot(cache,component_type=ComponentType,
           tree_snapshot=probe._global_maps(cache),accounting_snapshot=probe._accounting_snapshot(cache))


def frontier_batch(scheduler,request):
    """在同一安全点执行四个查询，并对每个查询分别验证非干扰。"""
    queries=unpack(request['packed_queries'])
    if not isinstance(queries,list) or len(queries)!=4:
        raise RuntimeError('frontier批量必须包含四个pending请求')
    cache=scheduler.tree_cache
    initial_scope=probe._validate_runtime_scope(scheduler,cache)
    before=semantic_state(cache)
    initial=before
    observations=[]
    for query in queries:
        scope_before=probe._validate_runtime_scope(scheduler,cache)
        result=frontier.inspect_resident_fa_frontier(cache,query['token_ids'],extra_key=query.get('extra_key'),limit=query.get('limit'))
        after=semantic_state(cache)
        scope_after=probe._validate_runtime_scope(scheduler,cache)
        changed=frontier.semantic_snapshot_differences(before,after)
        if changed or scope_before!=scope_after or scope_before!=initial_scope:
            raise RuntimeError('frontier查询改变语义状态或运行模式：'+str(changed))
        observations.append({**result,'scope_before':scope_before,'scope_after':scope_after,
                             'state_equal':True,'changed_fields':[],'query_id':query['query_id']})
        before=after
    return {'ok':True,'nonce':request.get('nonce'),'observations':observations,'semantic_state':initial,
            'non_interference_checks':4,'semantic_snapshot_count':5}


def introspection(scheduler,request):
    """无损解码后调用原句柄验证与完整view读取，仅缓存不可变句柄。"""
    started=perf_counter_ns()
    handles=_ORIGINAL_PARSE({**request,'handles':unpack(request['packed_handles'])})
    old=getattr(scheduler,'_control_epoch_handles',None)
    if old is not None and not old.consumed:
        raise RuntimeError('前一epoch尚未完成，禁止覆盖句柄')
    result=base._batch_view(scheduler,handles)
    context=EpochHandles(handles,result['view_digest'])
    scheduler._control_epoch_handles=context
    return {'ok':True,'nonce':request.get('nonce'),'candidate_ids':list(handles),**result,
            'epoch_token':context.token,'worker_ns':perf_counter_ns()-started,'state_mutated':False}


def reconciliation(scheduler,request):
    """复用已验证句柄，原协调函数的当前view及全部动态检查保持不变。"""
    context=getattr(scheduler,'_control_epoch_handles',None)
    if context is None:
        raise RuntimeError('当前引擎没有本epoch的句柄')
    handles=context.take(request.get('epoch_token'),request.get('candidate_ids',()),request.get('expected_view_digest'))
    def parsed(supplied):
        if supplied is not request:
            raise RuntimeError('不允许跨控制请求复用句柄')
        return handles
    original=base._parse_handles
    try:
        base._parse_handles=parsed
        return base._batch_reconciliation(scheduler,request)
    finally:
        base._parse_handles=original
        scheduler._control_epoch_handles=None


def dispatch(scheduler,request):
    """仅替换本轮三个控制入口，其余请求沿用冻结实现。"""
    action=request.get('action')
    if action=='control_epoch_introspection':return introspection(scheduler,request)
    if action=='control_epoch_frontiers':return frontier_batch(scheduler,request)
    if action=='control_epoch_reconciliation':return reconciliation(scheduler,request)
    return _ORIGINAL_DISPATCH(scheduler,request)


def scheduler_entry(*args,**kwargs):
    """沿用冻结调度器安全点、队列和服务器，不修改调度逻辑。"""
    base._run_checkpoint_control=dispatch
    return base._wrapped_run_scheduler_process(*args,**kwargs)


class OptimizedControlEngine(Engine):
    """安装语义保持的独立控制入口。"""
    run_scheduler_process_func=staticmethod(scheduler_entry)


requested_control_port=base.requested_control_port
