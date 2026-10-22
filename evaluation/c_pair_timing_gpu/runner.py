"""C 配对计时验证运行器：每生命周期独立 Engine，只执行一个正式 Pending，计时仪表与研究审计分离。

- strict：与 c_shared_gpu_pilot_k3 的 strict 臂完全同序（16 预热重放 → b5d4_apply 冻结协调 → 流式正式请求），
  正确性检查逐条保留，另增计时仪表：TTFT 六段分解、Triton 编译缓存逐点快照、首 decode 事件窗；
- strict_warm（形状预热修正生命周期，phase2）：预热重放 → throwaway（目标 Pending 原 Token+1 新 Token，
  在全量驻留集合下以与目标相同的 extend 形状触发内核编译，不进入任何正式计时）→ flush_cache →
  空缓存断言 → 预热二次重放（逐请求 digest 验证）→ b5d4_apply（唯一一次协调）→ 正式流式请求。
计时合同：请求级性能（TTFT 六段）、控制开销（RPC 墙钟+分段）、CUDA 事件（compute/copies）分开记录；
client 与 engine 同用 CLOCK_MONOTONIC，跨进程时间戳可直接衔接。
"""
import copy
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path
from .bridge import scheduler_process
from .control import physical_cpu_census
from evaluation.lightweight_runtime_e1.protocol import require
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.persistence import call_saved

OUT = Path('/b5out')
MODE = os.environ['PILOT_MODE']
ARM = os.environ['B5D4_ARM']
RUN = os.environ['B5D4_RUN']


def write(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def triton_snapshot(label):
    """TRITON_CACHE_DIR 逐点快照：条目数 + 内核目录名，差分定位编译发生在哪个窗口。"""
    root = Path(os.environ.get('TRITON_CACHE_DIR', '/b5out/triton_cache'))
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return {'label': label, 'entries': len(names), 'names': names}


def triton_delta(before, after):
    """窗口内新增编译条目：after 有而 before 无的内核目录名。"""
    old = set(before['names'])
    return {'window': before['label'] + '→' + after['label'], 'new_entries': len(set(after['names']) - old),
            'new_names': sorted(set(after['names']) - old)}


def decompose(stream, trace):
    """TTFT 六段分解（client/engine 同 CLOCK_MONOTONIC，可直接衔接）：
    queue=send→首batch开始；pregap=首batch开始→首kernel事件；execwin=首batch整窗（事件elapsed含host编译间隙）；
    compute=紧凑kernel事件窗；postgap=末kernel→batch结束；delivery=batch结束→首chunk到达。"""
    e0, c0 = trace['execution'][0], trace['compute'][0]
    d0 = next((c for c in trace['compute'] if not c['is_extend']), None)
    parts = {
        'queue_ms': (e0['ns_before'] - stream['start_ns']) / 1e6,
        'pregap_ms': (c0['ns_before'] - e0['ns_before']) / 1e6,
        'execwin_ms': (e0['ns_after'] - e0['ns_before']) / 1e6,
        'compute_cuda_ms': c0['cuda_elapsed_ms'],
        'postgap_ms': (e0['ns_after'] - c0['ns_after']) / 1e6,
        'delivery_ms': (stream['first_token_ns'] - e0['ns_after']) / 1e6,
        'ttft_ms_check': (stream['first_token_ns'] - stream['start_ns']) / 1e6,
        'first_decode_cuda_ms': d0['cuda_elapsed_ms'] if d0 else None,
        'e0_event_wait_ns': e0.get('event_wait_ns'), 'c0_end_wait_ns': c0.get('end_wait_ns'),
        'c0_sync_before_ns': c0.get('sync_before_ns'), 'e0_native_host_ns': e0.get('native_host_ns'),
    }
    # 分解闭合校验：三段之和应等于 TTFT（容差 5ms 覆盖时间戳取整）
    rebuilt = parts['queue_ms'] + parts['execwin_ms'] + parts['delivery_ms']
    require(abs(rebuilt - parts['ttft_ms_check']) < 5.0, 'TTFT 分解不闭合：%.1f vs %.1f' % (rebuilt, parts['ttft_ms_check']))
    return parts


def replay_warmup(engine, call, inputs, out, prefix, observe_wall_acc):
    """16 条冻结预热真实重放；prefix='history' 首轮写控制溯源证据，prefix='history2' 二轮（flush 后）另存留证。
    返回 (预热墙钟ms, observe RPC 墙钟ms)。两轮均逐请求做 digest/输入长度/输出数合同验证。"""
    start = time.monotonic_ns()
    for i, request in enumerate(inputs['history'], 1):
        output = engine.generate(input_ids=request['token_ids'], sampling_params=inputs['sampling'], rid=request['request_id'])
        t = time.monotonic_ns()
        state = call('b5c3_observe')
        observe_wall_acc.append((time.monotonic_ns() - t) / 1e6)
        matches = state['matches'].get(request['request_id'], [])
        require(matches and all(x['input_token_digest'] == request['token_prefix_sha256'] and
                                x['input_token_count'] == len(request['token_ids']) for x in matches), '历史实际输入不符')
        require(len(output.get('output_ids', [])) == 1 and int(output['meta_info'].get('num_retractions', 0) or 0) == 0, '历史异常')
        write(out / f'{prefix}_{i:02d}.json', {'request': request, 'output': output, 'state': state})
    return (time.monotonic_ns() - start) / 1e6


def main():
    out = OUT / RUN
    out.mkdir(exist_ok=False)
    engine = None
    result = None
    try:
        inputs = json.loads((OUT / 'prepared_inputs.json').read_text())
        config = inputs['configuration']
        source = json.loads((OUT / 'source_identity.json').read_text())
        for p, h in source['runtime_sources'].items():
            require(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, 'Runtime源码变化')
        for p, h in json.loads((OUT / 'cpu_gate.json').read_text())['runtime_sources'].items():
            require(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, 'CPU门禁对应Runtime源码变化')
        for n in ['sglang', 'transformers']:
            require(importlib.metadata.version(n) == config[n + '_version'], '版本不符')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config['tokenizer_path'], local_files_only=True)
        require(hashlib.sha256(str(tok.chat_template).encode()).hexdigest() == config['chat_template_sha256'], '模板身份不符')
        from evaluation.lightweight_runtime_e1.observation import real_api_preflight
        require(real_api_preflight()['status'] == 'PASS', '真实Req接口不符')
        from sglang.srt.entrypoints.engine import Engine
        from targeted_probe import ControlClient
        from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport

        class ObservedEngine(Engine):
            run_scheduler_process_func = staticmethod(scheduler_process)

        class AuditedControlClient(ControlClient):
            """每次 RPC 落盘留证并计时，控制开销与请求性能分开（与 pilot 相同）。"""

            def _call(self, request):
                return call_saved(self.port, self.timeout_s, request, out / ('rpc_' + str(time.monotonic_ns())))

        t_boot = time.monotonic_ns()
        engine = ObservedEngine(**config['engine_configuration'])
        client = AuditedControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']))
        wait_for_transport(client)
        boot_ms = (time.monotonic_ns() - t_boot) / 1e6
        snapshots = [triton_snapshot('engine_ready')]

        def call(action, **kw):
            r = client._call({'op': 'checkpoint_control', 'action': action, 'nonce': str(time.time_ns()), **kw})
            require(r.get('ok'), r.get('error', 'Runtime调用失败'))
            return r

        epoch_start = time.monotonic_ns()
        initial = call('b5c3_observe')
        require(not any(n['slots'] for n in initial['before']['nodes']), '新Engine非空')
        write(out / 'initial.json', initial)

        # ── 第一轮冻结预热重放（两模式共用）──
        # strict 模式写 history_XX.json（控制溯源输入，与 pilot 一致）；
        # strict_warm 模式首轮只做形状基座，改写 history_pre_XX.json 留证——
        # node_id 是进程生命周期计数器、跨 flush 重建不稳定（attempt1 实测：同内容节点 18→51），
        # 控制溯源必须读真正物化当前候选的二次预热，故二轮写 history_XX.json。
        observe_wall = []
        warmup1_ms = replay_warmup(engine, call, inputs, out, 'history_pre' if MODE == 'strict_warm' else 'history',
                                   observe_wall)
        snapshots.append(triton_snapshot('post_warmup'))

        # ── strict_warm：throwaway 形状预热 → flush → 二次预热重放（apply 之前，BASELINE 仍为 None）──
        warmup2_ms = None
        if MODE == 'strict_warm':
            # 先取目标 Pending（与 pilot 相同来自冻结输入声明），throwaway 用其原 Token + 1 新 Token
            pending0 = next(x for x in inputs['pendings'] if x['continuation_id'] == os.environ['PILOT_PENDING'])
            t0 = time.monotonic_ns()
            throw = engine.generate(input_ids=pending0['token_ids'],
                                    sampling_params={'temperature': 0, 'max_new_tokens': 1, 'ignore_eos': True},
                                    rid='SHAPE_WARM_THROWAWAY')
            throw_wall_ms = (time.monotonic_ns() - t0) / 1e6
            require(len(throw.get('output_ids', [])) == 1, 'throwaway 输出异常')
            write(out / 'throwaway.json', {'pending': pending0['continuation_id'], 'wall_ms': throw_wall_ms,
                                           'output_ids': throw['output_ids'], 'note': '形状预热请求，不进入任何正式计时'})
            snapshots.append(triton_snapshot('post_throwaway'))
            # flush：清空全部驻留状态（radix/FA/mamba 池），Triton 编译缓存进程内保留
            t1 = time.monotonic_ns()
            engine.flush_cache()
            flush_wall_ms = (time.monotonic_ns() - t1) / 1e6
            state = call('b5c3_observe')
            require(not any(n['slots'] for n in state['before']['nodes']), 'flush后仍有驻留mamba状态')
            write(out / 'flush_state.json', {'wall_ms': flush_wall_ms, 'state': state})
            snapshots.append(triton_snapshot('post_flush'))
            # 二次预热重放：重建 16 候选（digest 逐请求验证；写 history_XX.json 作为控制溯源输入，
            # 因当前候选的 node_id 属于 flush 后的重建世代，只有二轮观察能提供匹配的来源线索）
            observe_wall2 = []
            warmup2_ms = replay_warmup(engine, call, inputs, out, 'history', observe_wall2)
            snapshots.append(triton_snapshot('post_warmup2'))
            write(out / 'warmup2_wall.json', {'warmup1_ms': warmup1_ms, 'warmup2_ms': warmup2_ms,
                                              'observe_rpc_ms': observe_wall2, 'throwaway_wall_ms': throw_wall_ms,
                                              'flush_wall_ms': flush_wall_ms})

        # ── 冻结 CPU 策略选择 → 真实协调（唯一一次 apply，完整 pilot 合同）──
        control_start = time.monotonic_ns()
        result = client._call({'op': 'checkpoint_control', 'action': 'b5d4_apply', 'nonce': str(time.time_ns())})
        control_end = time.monotonic_ns()
        write(out / 'reconciliation.json', result)
        require(result.get('ok'), result.get('error', '协调失败'))
        require(result['policy'] == os.environ['PILOT_POLICY'], '策略标识不符')
        # 加严断言：物理域必须恰为 CPU 16 候选（pilot 由 launch census 门禁把守，本轮入 runner）
        require(result['census']['physical_domain_size'] == 16 and len(result['census']['mapped']) == 16 and
                not result['census']['unmapped_extras'] and not result['census']['missing_physical'],
                'census 域异常：physical=%d mapped=%d extras=%d missing=%d' % (
                    result['census']['physical_domain_size'], len(result['census']['mapped']),
                    len(result['census']['unmapped_extras']), len(result['census']['missing_physical'])))
        pending = result['target']['pending']
        write(OUT / 'target_request.json', pending)
        control_rpc_wall_ms = (control_end - control_start) / 1e6
        snapshots.append(triton_snapshot('post_control'))

        # ── 正式流式请求（与 pilot 逐字节同检查）──
        sampling = {'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True}
        write(out / 'request.json', {'pending': pending, 'sampling': sampling})
        incremental = bool(engine.tokenizer_manager.incremental_streaming_output)
        start = time.monotonic_ns()
        chunks = []
        arrivals = []
        first = None
        for chunk in engine.generate(input_ids=pending['token_ids'], sampling_params=sampling, rid=pending['continuation_id'], stream=True):
            now = time.monotonic_ns()
            chunk = copy.deepcopy(chunk)
            chunks.append(chunk)
            arrivals.append({'ns': now, 'output_ids': chunk.get('output_ids'),
                             'completion_tokens': chunk.get('meta_info', {}).get('completion_tokens')})
            if first is None and (chunk.get('output_ids') or chunk.get('meta_info', {}).get('completion_tokens', 0) > 0):
                first = now
        end = time.monotonic_ns()
        require(chunks and first is not None, '缺少真实流式首Token')
        output = copy.deepcopy(chunks[-1])
        if incremental:
            output['output_ids'] = [t for c in chunks for t in c.get('output_ids', [])]
        require(len(output['output_ids']) == output['meta_info']['completion_tokens'], '流式Token拼接与真实完成计数不一致')
        write(out / 'stream_chunks.json', {'chunks': chunks, 'arrivals': arrivals, 'start_ns': start, 'end_ns': end,
                                           'first_token_ns': first, 'incremental_streaming_output': incremental})
        write(out / 'output.json', output)
        post = call('b5d4_state')
        write(out / 'post_request.json', post)
        require(len(output.get('output_ids', [])) == 8 and output['meta_info']['completion_tokens'] == 8 and
                int(output['meta_info'].get('num_retractions', 0) or 0) == 0, '正式请求输出或回撤异常')
        # Cold 输出一致性（加严）：与 pilot cold 臂逐 Token 相同
        cold = json.loads((OUT / ('cold_reference_' + pending['continuation_id'] + '.json')).read_text())
        require(output['output_ids'] == cold['output_ids'], '与 Cold 参照输出不一致')

        trace = post['trace']
        all_matches = [m['native'] for m in trace['matches'] if m.get('native', {}).get('request_id') == pending['continuation_id']]
        require(all_matches, '缺少真实匹配证据')
        execution = trace['execution']
        compute = trace['compute']
        require(execution and all(x.get('native_run_returned') for x in execution), 'Prefill/Decode未完成')
        require(len(compute) == len(execution), 'compute与execution行数不一致')
        prefill_row = execution[0]
        require(compute and compute[0]['is_extend'] and isinstance(prefill_row['extend_token_count'], int), '首执行不是Prefill')
        cuda_prefill_ms = sum(c['cuda_elapsed_ms'] for c in compute if c['is_extend'])
        cuda_decode_ms = sum(c['cuda_elapsed_ms'] for c in compute if not c['is_extend'])
        cow_cuda_ms = sum(c.get('cuda_elapsed_ms', 0.0) for c in trace['copies'])

        target = result['target']
        expected_E = target['expected']['E']
        actual = all_matches[-1]
        require(actual['input_token_digest'] == pending['token_prefix_sha256'], 'Pending输入不符')
        require(actual['physical_fa_hit'] == pending['attention_frontier'], 'FA命中与冻结边界不一致')
        require(actual['executable_prefix'] == expected_E, '实际可执行前缀与计划E不符')
        require(prefill_row['batch_prefix_length'] == expected_E and
                prefill_row['extend_token_count'] == len(pending['token_ids']) - expected_E, 'Prefill区间与E/G合同不符')
        if expected_E > 0:
            bound = next(x['node'] for x in result['reconciliation']['binding']['candidates']
                         if x['candidate_id'] == target['source_checkpoint'])
            require(actual['best_match_node'] == bound['node_id'] and actual['cow_source_slots'] == bound['slots'], '恢复来源错误')
            require(any(c.get('returned') and c['source_slots'] == bound['slots'] and
                        c['destination_slots'] == actual['working_slots'] and
                        c['source_payload_before'] == c['source_payload_after'] == c['destination_payload_after']
                        for c in trace['copies']), 'COW未完成或内容错误')
            kept = {s for c in result['reconciliation']['binding']['candidates']
                    if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
            require(not kept.intersection(actual['working_slots']), '工作状态与保留状态别名')
        require(post['state']['selected_payload'] == result['selected_payload_after'] and
                post['state']['selected_components'] == result['admission_baseline']['selected_components'], '历史Recurrent状态受损')
        require(post['state']['fa_protected_payload'] == result['fa_protected_payload'], '非重算FA内容变化')
        require(all(post['state']['fa_map'].get(k) == v for k, v in result['fa_map_before'].items()), '历史FA映射丢失或改变')

        snapshots.append(triton_snapshot('post_request'))
        # 编译窗口差分：定位 Triton 新增编译发生在 预热/throwaway/二次预热/控制/正式请求 哪个窗口
        base = {'post_warmup': 'engine_ready', 'post_control': 'post_warmup2' if warmup2_ms is not None else 'post_warmup',
                'post_request': 'post_control'}
        triton_windows = [triton_delta(next(s for s in snapshots if s['label'] == base[k]), next(s for s in snapshots if s['label'] == k))
                          for k in ['post_warmup', 'post_control', 'post_request']]
        if MODE == 'strict_warm':
            triton_windows.insert(0, triton_delta(next(s for s in snapshots if s['label'] == 'post_warmup'),
                                                  next(s for s in snapshots if s['label'] == 'post_throwaway')))
            triton_windows.insert(1, triton_delta(next(s for s in snapshots if s['label'] == 'post_flush'),
                                                  next(s for s in snapshots if s['label'] == 'post_warmup2')))

        parts = decompose(json.loads((out / 'stream_chunks.json').read_text()), trace)
        base_summary = {'status': 'PASS', 'mode': MODE, 'policy': os.environ['PILOT_POLICY'],
                        'pending': pending['continuation_id'],
                        'H_physical_fa_hit': actual['physical_fa_hit'],
                        'E_executable_prefix': actual['executable_prefix'],
                        'G_replay_gap': actual['physical_fa_hit'] - actual['executable_prefix'],
                        'prefill_token_count': prefill_row['extend_token_count'],
                        'batch_prefix_length': prefill_row['batch_prefix_length'],
                        'cuda_prefill_ms': cuda_prefill_ms, 'cuda_decode_ms': cuda_decode_ms,
                        'cow_copy_count': len(trace['copies']), 'cow_copy_cuda_ms': cow_cuda_ms,
                        'ttft_ms': (first - start) / 1e6, 'request_wall_ms': (end - start) / 1e6,
                        'stream_tpot_ms': (arrivals[-1]['ns'] - first) / 7e6,
                        'control_rpc_wall_ms': control_rpc_wall_ms,
                        'warmup_wall_ms': warmup1_ms + (warmup2_ms or 0),
                        'warmup1_wall_ms': warmup1_ms, 'warmup2_wall_ms': warmup2_ms,
                        'warmup_observe_rpc_wall_ms': sum(observe_wall) / len(observe_wall),
                        'engine_boot_ms': boot_ms,
                        'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6,
                        'timing_decomposition': parts,
                        'triton_snapshots': [{'label': s['label'], 'entries': s['entries']} for s in snapshots],
                        'triton_new_kernels': triton_windows,
                        'cold_output_match': True,
                        'output_ids': output['output_ids'], 'native_matches': all_matches,
                        'control_timing_stages': result['timing'],
                        'census_counts': {'physical': result['census']['physical_domain_size'],
                                          'mapped': len(result['census']['mapped']),
                                          'unmapped_extras': len(result['census']['unmapped_extras'])},
                        'selected_cpu': result['policy_selected_cpu_ids'], 'plan_recovery_cost_ms': result['plan']['recovery_cost_ms'],
                        '说明': '受控审计化执行；CUDA事件是设备时间线区间而非纯内核之和；控制开销与请求性能分开计时'}
        write(out / 'summary.json', base_summary)
        print('完成', MODE, os.environ['PILOT_POLICY'], pending['continuation_id'],
              'E', base_summary['E_executable_prefix'], 'H', base_summary['H_physical_fa_hit'],
              'TTFT', round(base_summary['ttft_ms']), 'ms', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', {'status': 'BLOCKED', 'error': repr(exc), 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        write(out / 'engine_cleanup.json', {'shutdown_returned': True, 'engine_created': engine is not None})


if __name__ == '__main__':
    main()
