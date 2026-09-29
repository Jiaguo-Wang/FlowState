"""仅增加计时观测的控制路径探针，所有状态操作仍调用冻结实现。"""
from contextlib import contextmanager
import inspect
from time import perf_counter_ns

from sglang.srt.entrypoints.engine import Engine
import rq6_batched_control_transport as batch
import targeted_probe as probe
import wp3b_end_to_end_transport as frontier
from flowstate.adapters.sglang import SGLangAdapter


class CallProfile:
    """累计嵌套操作的包含时间和排除子调用后的独占时间。"""
    def __init__(self):
        self.rows = {}
        self.stack = []

    def wrap(self, name, fn):
        """不改变参数、返回值或异常，仅记录原函数执行。"""
        def measured(*args, **kwargs):
            frame = [name, perf_counter_ns(), 0]
            path = '/'.join(x[0] for x in self.stack+[frame])
            self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = perf_counter_ns()-frame[1]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][2] += elapsed
                row = self.rows.setdefault(path, {'count':0,'inclusive_ns':0,'exclusive_ns':0})
                row['count'] += 1
                row['inclusive_ns'] += elapsed
                row['exclusive_ns'] += elapsed-frame[2]
        return measured


@contextmanager
def instrument(scheduler, profile):
    """临时包装原函数，退出时逐项恢复；不缓存或省略任何检查。"""
    targets = [(batch,n) for n in ('_parse_handles','_batch_view','_canonical_digest','_validate_identity_and_residency','_proof')]
    targets += [(probe,n) for n in ('_validate_runtime_scope','_path_snapshot','_global_maps','_accounting_snapshot','_find_exact_node','_tensor_sha256','_tensor_ids')]
    targets += [(frontier,n) for n in ('semantic_cache_snapshot','inspect_resident_fa_frontier','semantic_snapshot_differences')]
    targets += [(SGLangAdapter,n) for n in ('evict_mamba_only','validate_runtime_capabilities','_validate_runtime_version','_read_sglang_version','_find_exact_node','_validate_target','_capture_target_snapshot','_evict_mamba_component_only','_validate_postconditions')]
    targets += [(type(scheduler.tree_cache),'sanity_check')]
    originals = []
    try:
        for owner,name in targets:
            original = inspect.getattr_static(owner,name)
            fn = original.__func__ if isinstance(original,staticmethod) else original
            measured = profile.wrap(owner.__name__+'.'+name,fn)
            originals.append((owner,name,original))
            setattr(owner,name,staticmethod(measured) if isinstance(original,staticmethod) else measured)
        yield
    finally:
        for owner,name,original in reversed(originals):
            setattr(owner,name,original)


def profiled_dispatch(original):
    """在原控制处理入口外记录服务端时间及函数分解。"""
    def dispatch(scheduler, request):
        if not request.get('_control_profile'):
            return original(scheduler,request)
        start=perf_counter_ns()
        profile=CallProfile()
        with instrument(scheduler,profile):
            result=original(scheduler,request)
        end=perf_counter_ns()
        result['_control_profile']={'server_start_ns':start,'server_end_ns':end,'operations':profile.rows}
        return result
    return dispatch


def scheduler_entry(*args, **kwargs):
    """沿用冻结调度器和队列，仅附加队列提交边界与函数计时。"""
    original_submit=probe.ProbeState.submit
    def submit(self,request,*a,**kw):
        start=perf_counter_ns()
        result=original_submit(self,request,*a,**kw)
        end=perf_counter_ns()
        if request.get('_control_profile') and '_control_profile' in result:
            result['_control_profile'].update(submit_start_ns=start,submit_end_ns=end)
        return result
    probe.ProbeState.submit=submit
    batch._run_checkpoint_control=profiled_dispatch(batch._run_checkpoint_control)
    probe._run_census=profiled_dispatch(probe._run_census)
    return batch._wrapped_run_scheduler_process(*args,**kwargs)


class ProfileEngine(Engine):
    """以相同引擎配置运行冻结控制路径，仅添加计时探针。"""
    run_scheduler_process_func=staticmethod(scheduler_entry)
