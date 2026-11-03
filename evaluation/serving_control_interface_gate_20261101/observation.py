"""只读观察冻结版本的真实请求与批次（四 Pending 版：目标 rid 集合化、行级归因）。

与 G1 pkg_greedy/observation.py 的差别只在两处：
- 单值 TARGET 改为集合 TARGETS（四 Pending 串行，同一时刻仍只有一个活跃请求批次）；
- execution/compute 行内记录 rid，供请求后按 Pending 归因验收（批次独占性断言保持不变）。
"""
import time
from array import array

TARGETS = frozenset()   # 由 bridge 在协调完成后写入（四 Pending continuation_id）
ACTIVE = False
ACTIVE_RID = None   # run() 进入独占目标批次时暂存 rid；execute 处的 batch 是 ForwardBatch（无 reqs）
COMPUTE = []


def require(value, message):
    if not value:
        raise ValueError(message)


def observe_batch(batch):
    """仅Extend模式解释extend_range；Decode不把残留区间当成当前计算范围。"""
    rows = []
    for i, req in enumerate(batch.reqs):
        rid = str(req.rid)
        if rid not in TARGETS:
            continue
        row = {'rid': rid, 'ns_before': time.monotonic_ns(), 'forward_mode': str(batch.forward_mode),
               'original_input_length': len(req.origin_input_ids),
               'full_untruncated_length': len(req.full_untruncated_fill_ids),
               'prefix_indices_length': len(req.prefix_indices),
               'output_length_before': len(req.output_ids),
               'optional_extend_batch_idx': getattr(req, 'extend_batch_idx', 'UNKNOWN')}
        if batch.forward_mode.is_extend():
            fill = req.get_fill_ids(); start = len(req.prefix_indices); end = req.extend_range.end
            require(req.extend_range.start == start, 'Extend起点与前缀长度不同')
            require(len(fill) == end and len(fill[start:]) == req.extend_range.length, '真实输入区间不一致')
            require(batch.prefix_lens[i] == start and batch.extend_lens[i] == end - start, 'Batch与Req区间不一致')
            row.update(extend_start=start, extend_end=end, extend_token_count=end - start,
                       batch_prefix_length=batch.prefix_lens[i], batch_extend_length=batch.extend_lens[i])
        else:
            row.update(extend_start='不适用：非Prefill', extend_end='不适用：非Prefill', extend_token_count='不适用：非Prefill')
        rows.append(row)
    return rows


def wrap_run_batch(original, records, persist):
    """非重叠调度下测量当前CUDA流区间，包含主机提交空隙，原函数仅调用一次。"""
    def run(scheduler, batch, *args, **kwargs):
        global ACTIVE, ACTIVE_RID
        rows = observe_batch(batch)
        if not rows:
            return original(scheduler, batch, *args, **kwargs)
        require(len(rows) == 1 and len(batch.reqs) == 1, '目标必须为独占单请求批次')
        require(not scheduler.enable_overlap, '计时合同不接受重叠调度')
        import torch
        records.extend(rows); persist()
        sync_start = time.monotonic_ns(); torch.cuda.synchronize(); sync_end = time.monotonic_ns()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(); start.record(stream)
        host_start = time.monotonic_ns()
        ACTIVE = True
        ACTIVE_RID = rows[0]['rid']
        try:
            result = original(scheduler, batch, *args, **kwargs)
        finally:
            ACTIVE = False
            ACTIVE_RID = None
        host_end = time.monotonic_ns(); end.record(stream); end.synchronize(); done = time.monotonic_ns()
        for row in rows:
            row.update(native_run_returned=True, ns_after=done, native_host_ns=host_end - host_start,
                       sync_before_ns=sync_end - sync_start, event_wait_ns=done - host_end,
                       cuda_elapsed_ms=start.elapsed_time(end),
                       cuda_stream=int(stream.cuda_stream), is_extend=bool(batch.forward_mode.is_extend()),
                       timing_definition='同一CUDA流事件区间，含主机提交或首次编译空隙，非纯内核和')
        persist()
        return result
    return run


def real_api_preflight():
    """使用真实Req与ScheduleBatch，提交目标之前验证所需接口，无Engine或GPU分配。"""
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    probe_rid = 'P04_PREFLIGHT_PROBE'
    req = Req(probe_rid, '', array('q', [1, 2, 3, 4]), SamplingParams(temperature=0, max_new_tokens=8, ignore_eos=True))
    req._refresh_fill_ids(); req.set_extend_range(2, 4); req.prefix_indices = array('q', [10, 11])
    batch = ScheduleBatch(reqs=[req], device='cpu', forward_mode=ForwardMode.EXTEND, prefix_lens=[2], extend_lens=[2])
    before = (list(req.origin_input_ids), list(req.full_untruncated_fill_ids), req.extend_range,
              list(req.prefix_indices), list(batch.prefix_lens), list(batch.extend_lens))
    global TARGETS
    saved = TARGETS
    TARGETS = frozenset({probe_rid})   # 只在预检瞬间把探针 rid 纳入观测域，随后恢复
    try:
        row = observe_batch(batch)[0]
    finally:
        TARGETS = saved
    after = (list(req.origin_input_ids), list(req.full_untruncated_fill_ids), req.extend_range,
             list(req.prefix_indices), list(batch.prefix_lens), list(batch.extend_lens))
    require(before == after and row['extend_token_count'] == 2 and row['rid'] == probe_rid, '真实API只读检查失败')
    return {'status': 'PASS', 'req_class': Req.__module__ + '.' + Req.__name__,
            'batch_class': ScheduleBatch.__module__ + '.' + ScheduleBatch.__name__, 'device': 'cpu'}


def wrap_eager(original, persist):
    """只计目标的实际Eager模型区间；调用点位于延迟COW之后，不改变返回值。"""
    def execute(runner, batch, *args, **kwargs):
        if not ACTIVE:
            return original(runner, batch, *args, **kwargs)
        import torch
        rid = ACTIVE_RID   # ForwardBatch 无 reqs 字段，取 run() 在同一调用栈内暂存的独占目标 rid
        require(rid is not None, 'ACTIVE 状态下暂存 rid 缺失')
        before = time.monotonic_ns(); torch.cuda.synchronize(); sync = time.monotonic_ns()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(); start.record(stream); host_start = time.monotonic_ns()
        result = original(runner, batch, *args, **kwargs)
        host_end = time.monotonic_ns(); end.record(stream); end.synchronize()
        COMPUTE.append({'rid': rid, 'is_extend': bool(batch.forward_mode.is_extend()),
                        'forward_mode': str(batch.forward_mode),
                        'cuda_elapsed_ms': start.elapsed_time(end), 'native_host_ns': host_end - host_start,
                        'sync_before_ns': sync - before,
                        'end_wait_ns': time.monotonic_ns() - host_end, 'ns_before': before,
                        'ns_after': time.monotonic_ns(),
                        'cuda_stream': int(stream.cuda_stream), 'returned': True})
        persist()
        return result
    return execute
