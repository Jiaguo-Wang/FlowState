"""在双 rank 实际恢复请求上记录匹配前沿和调度器内部恢复时延。"""
from __future__ import annotations

from time import perf_counter_ns

import torch
import rq5b_kimi_phase1_transport as phase1
import rq5b_kimi_phase2_transport as phase2


def install_timing(scheduler):
    """只包装当前实验实例，计时覆盖首次匹配至单令牌请求完成。"""
    scheduler._rq5b3_target = None
    scheduler._rq5b3_timing = None
    cache = scheduler.tree_cache
    original_match = cache.match_prefix

    def match(params):
        req = getattr(params, 'req', None)
        measured = req is not None and req.rid == scheduler._rq5b3_target
        if measured and scheduler._rq5b3_timing is None:
            scheduler._rq5b3_timing = {'start_ns': perf_counter_ns(), 'matches': [], 'batches': []}
        result = original_match(params)
        if measured:
            scheduler._rq5b3_timing['matches'].append({
                'H': int(result.full_kv_hit_length), 'E': len(result.device_indices),
                'G': int(result.full_kv_hit_length) - len(result.device_indices),
                'cow_mamba': bool(getattr(params, 'cow_mamba', False)),
                'mamba_branching_seqlen': result.mamba_branching_seqlen})
        return result

    cache.match_prefix = match
    original_run = scheduler.run_batch

    def run_batch(batch, *args, **kwargs):
        measured = any(req.rid == scheduler._rq5b3_target for req in batch.reqs)
        if measured:
            if len(batch.reqs) != 1 or scheduler._rq5b3_timing is None:
                raise RuntimeError('恢复计时要求已记录前缀匹配且独占调度批次')
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            count = int(batch.extend_num_tokens)
        result = original_run(batch, *args, **kwargs)
        if measured:
            ended.record()
            scheduler._rq5b3_timing['batches'].append({'start_event': started,
                'end_event': ended, 'extend_tokens': count})
        return result

    scheduler.run_batch = run_batch
    original_process = scheduler.process_batch_result

    def process(batch, result):
        reqs = [req for req in batch.reqs if req.rid == scheduler._rq5b3_target]
        value = original_process(batch, result)
        if reqs and reqs[0].finished():
            scheduler._rq5b3_timing['finish_ns'] = perf_counter_ns()
            scheduler._rq5b3_timing['output_ids'] = list(reqs[0].output_ids)
        return value

    scheduler.process_batch_result = process


def control(scheduler, request):
    """复用阶段二的状态控制，只增加计时启动和结果读取。"""
    action = request['action']
    if action == 'runtime_gate':
        result = phase2.control(scheduler, request)
        install_timing(scheduler)
        return result
    if action == 'arm_resume':
        phase1.runtime_scope(scheduler, scheduler.tree_cache)
        if scheduler._rq5b3_target is not None:
            raise RuntimeError('单生命周期仅允许一个正式恢复请求')
        torch.cuda.synchronize()
        scheduler._rq5b3_target = request['request_id']
        return {'ok': True, 'request_id': scheduler._rq5b3_target}
    if action == 'resume_metrics':
        phase1.runtime_scope(scheduler, scheduler.tree_cache)
        timing = scheduler._rq5b3_timing
        if not timing or 'finish_ns' not in timing or not timing['matches'] or not timing['batches']:
            raise RuntimeError('恢复请求没有完整计时证据')
        torch.cuda.synchronize()
        batches = [{'extend_tokens': row['extend_tokens'],
                    'cuda_ms': row['start_event'].elapsed_time(row['end_event'])}
                   for row in timing['batches']]
        return {'ok': True, 'tp_rank': int(scheduler.ps.tp_rank),
            'request_id': scheduler._rq5b3_target,
            'resume_latency_ms': (timing['finish_ns'] - timing['start_ns']) / 1e6,
            'forward_cuda_ms': sum(row['cuda_ms'] for row in batches),
            'matches': timing['matches'], 'batches': batches,
            'output_ids': timing['output_ids'],
            'eviction_events': scheduler._rq5b2_events,
            '说明': '主指标为调度器首次实际前缀匹配至单令牌请求完成的墙钟时延，含恢复、固定后缀前向及结果处理；不含引擎启动、历史物化、状态观测或驱逐。CUDA 累计前向时间仅作辅助。'}
    return phase2.control(scheduler, request)


def run_scheduler(*args, **kwargs):
    """在原双 rank 安全时点入口安装实验包装，不修改运行时源码。"""
    phase1.control = control
    return phase1.wrapped_run_scheduler_process(*args, **kwargs)


class KimiPhase3Engine(phase1.SGLangEngine):
    """每个条件独立建立运行时，避免配对中的缓存污染。"""
    run_scheduler_process_func = staticmethod(run_scheduler)
