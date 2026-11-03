"""正式 GPU 性能配对实验运行器（测量模式）：每生命周期独立 Engine，三口径分层计时。

生命周期序（全部在同一受控进程内，Triton 编译缓存进程内保留）：
16 预热重放(首轮 history_pre) → flush_cache → 空缓存断言 → 16 预热二次重放(history,控制溯源)
→ 唯一 b5d4_apply(完整 pilot 冻结合同，含全量审计——控制口径) → 形状预热 B(apply 后：目标前 T Token
原样+末 32 Token 改写，max_new_tokens=2，原生准入=计划 E，编译目标 extend/decode 形状，不进正式计时)
→ 正式流式请求(测量模式：窗口内零 A 类审计) → 请求后 b5d4_state 全量观察。

三口径（§4）：A 计算口径=CUDA prefill/decode 与 COW 事件；B 请求口径=TTFT 六段分解+request_wall；
C 控制口径=apply RPC 墙钟+分段。client 与 engine 同用 CLOCK_MONOTONIC，跨进程时间戳直接衔接。
"""
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import time
import traceback
from pathlib import Path
from .bridge import scheduler_process
from evaluation.lightweight_runtime_e1.protocol import require
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.persistence import call_saved

OUT = Path('/b5out')
MODE = os.environ['PILOT_MODE']
ARM = os.environ['B5D4_ARM']
RUN = os.environ['B5D4_RUN']
require(MODE == 'strict_warm' and ARM == 'strict', '正式实验固定 strict_warm + strict 臂语义')
require(os.environ.get('FS_MEASURE_MODE') == '1', '缺少测量模式环境变量')


def write(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def triton_snapshot(label):
    """TRITON_CACHE_DIR 逐点快照：条目数 + 内核目录名，差分定位编译窗口。"""
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
    queue=send→首batch开始；pregap=首batch开始→首kernel事件；execwin=首batch整窗（含host提交空隙）；
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
    }
    rebuilt = parts['queue_ms'] + parts['execwin_ms'] + parts['delivery_ms']
    require(abs(rebuilt - parts['ttft_ms_check']) < 5.0, 'TTFT 分解不闭合：%.1f vs %.1f' % (rebuilt, parts['ttft_ms_check']))
    return parts


def replay_warmup(engine, call, inputs, out, prefix, observe_wall_acc):
    """16 条冻结预热真实重放；首轮写 history_pre_XX.json（留证），二次（flush 后）写 history_XX.json
    （控制溯源输入——node_id 属 flush 后重建世代）。两轮均逐请求 digest/输入长度/输出数合同验证。"""
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
            """每次控制 RPC 落盘留证并计时（控制口径与请求口径分离；不进请求计时窗口）。"""

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

        # ── 第一轮冻结预热重放（编译基座；首轮写 history_pre_XX.json 留证，不作控制溯源）──
        observe_wall = []
        warmup1_ms = replay_warmup(engine, call, inputs, out, 'history_pre', observe_wall)
        snapshots.append(triton_snapshot('post_warmup'))

        # ── flush → 空缓存断言 → 二次预热重放（重建 16 候选；写 history_XX.json 供控制溯源）──
        t1 = time.monotonic_ns()
        engine.flush_cache()
        flush_wall_ms = (time.monotonic_ns() - t1) / 1e6
        state = call('b5c3_observe')
        require(not any(n['slots'] for n in state['before']['nodes']), 'flush后仍有驻留mamba状态')
        write(out / 'flush_state.json', {'wall_ms': flush_wall_ms, 'state': state})
        snapshots.append(triton_snapshot('post_flush'))
        observe_wall2 = []
        warmup2_ms = replay_warmup(engine, call, inputs, out, 'history', observe_wall2)
        snapshots.append(triton_snapshot('post_warmup2'))

        # ── 冻结 CPU 策略选择 → 唯一一次真实协调（完整 pilot 合同，含全量审计；控制口径）──
        control_start = time.monotonic_ns()
        result = client._call({'op': 'checkpoint_control', 'action': 'b5d4_apply', 'nonce': str(time.time_ns())})
        control_end = time.monotonic_ns()
        write(out / 'reconciliation.json', result)
        require(result.get('ok'), result.get('error', '协调失败'))
        require(result['policy'] == os.environ['PILOT_POLICY'], '策略标识不符')
        require(result['census']['physical_domain_size'] == 16 and len(result['census']['mapped']) == 16 and
                not result['census']['unmapped_extras'] and not result['census']['missing_physical'],
                'census 域异常：physical=%d mapped=%d extras=%d missing=%d' % (
                    result['census']['physical_domain_size'], len(result['census']['mapped']),
                    len(result['census']['unmapped_extras']), len(result['census']['missing_physical'])))
        pending = result['target']['pending']
        write(OUT / 'target_request.json', pending)
        control_rpc_wall_ms = (control_end - control_start) / 1e6
        snapshots.append(triton_snapshot('post_control'))

        # ── 形状预热 B（apply 后、正式请求前）：计划保留集下原生准入恰为计划 E ──
        # v1.1 修正（attempt1 证据 g1_CB_SA：执行 pt[E:] 会在 Pending 前缀上重新物化已被计划驱逐的
        # mamba 状态，正式请求准入被抬高到 5120≠计划0）：输入改为 pt[:E] + 确定性随机 L=T+32−E 个
        # Token，在 E 处立即分叉——只在随机分支物化状态，正式请求匹配深度不超计划 E；
        # 内核形状等价：正式请求为“在位置 E 准入、extend L 个 Token、decode 起 kv=T+32”，
        # 预热请求逐项同形（E=0 即全 5152 新算，与正式 SA 同形）。
        # max_new_tokens=2 → 1 个 decode 步，把目标形状的 extend 与 decode 内核全部编译完。
        T = pending['attention_frontier']
        pt = list(pending['token_ids'])
        require(len(pt) == T + 32, '输入全长与 T+32 不符')
        expected_E = int(result['target']['expected']['E'])
        L = len(pt) - expected_E
        vocab = tok.vocab_size
        rng = random.Random('c_formal_v1_1/%s/%s' % (os.environ['PILOT_POLICY'], pending['continuation_id']))
        tail = [rng.randrange(vocab) for _ in range(L)]
        while tail and tail[0] == pt[expected_E]:
            tail[0] = rng.randrange(vocab)  # 确保在 E 处首 Token 即分叉（FA 命中恰为 E）
        sw_ids = pt[:expected_E] + tail
        require(len(sw_ids) == len(pt) and sw_ids[expected_E:] != pt[expected_E:], '形状预热输入构造异常')
        t0 = time.monotonic_ns()
        throw = engine.generate(input_ids=sw_ids,
                                sampling_params={'temperature': 0, 'max_new_tokens': 2, 'ignore_eos': True},
                                rid='SHAPE_WARM_B_POST_APPLY')
        shape_warm_wall_ms = (time.monotonic_ns() - t0) / 1e6
        require(len(throw.get('output_ids', [])) == 2 and
                int(throw['meta_info'].get('num_retractions', 0) or 0) == 0, '形状预热B输出异常')
        write(out / 'shape_warm_b.json', {'pending': pending['continuation_id'], 'wall_ms': shape_warm_wall_ms,
                                          'input_len': len(sw_ids), 'planned_E': expected_E, 'diverge_from': expected_E,
                                          'output_ids': throw['output_ids'],
                                          'note': 'apply后形状预热(v1.1)：pt[:E]+确定性随机尾，计划E准入+目标extend/decode形状编译，不进正式计时'})
        snapshots.append(triton_snapshot('post_shape_warm'))

        # ── 正式流式请求（测量模式：窗口内零 A 类审计；检查合同与 pilot 逐条对应）──
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
        cold = json.loads((OUT / ('cold_reference_' + pending['continuation_id'] + '.json')).read_text())
        require(output['output_ids'] == cold['output_ids'], '与 Cold 参照输出不一致')

        # ── 请求级合同（与 pilot/c_pair 逐条对应；COW 内容验证=Cold 逐Token一致+请求后全量观察）──
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
        cow_host_ms = sum(c.get('native_host_ns', 0) for c in trace['copies']) / 1e6
        match_native_ms = (trace['matches'][0]['native_call_ns']) / 1e6 if trace['matches'] else None

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
                        c['destination_slots'] == actual['working_slots'] for c in trace['copies']),
                    'COW未完成（结构性验证；内容验证=请求后全量观察+Cold逐Token一致）')
            kept = {s for c in result['reconciliation']['binding']['candidates']
                    if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
            require(not kept.intersection(actual['working_slots']), '工作状态与保留状态别名')
        require(post['state']['selected_payload'] == result['selected_payload_after'] and
                post['state']['selected_components'] == result['admission_baseline']['selected_components'], '历史Recurrent状态受损')
        require(post['state']['fa_protected_payload'] == result['fa_protected_payload'], '非重算FA内容变化')
        require(all(post['state']['fa_map'].get(k) == v for k, v in result['fa_map_before'].items()), '历史FA映射丢失或改变')

        # ── 测量模式有效性断言：窗口内零审计 + 请求窗口零新增编译 ──
        audit_entries = trace.get('audit', [])
        require(len(audit_entries) == 2 and all(a['kind'] == 'cache_state_full_hash' for a in audit_entries),
                '测量模式失效：审计项应为 apply基线+请求后各一次，实测 %d 次' % len(audit_entries))
        sw_rows = [m for m in trace.get('shape_warm_matches', []) if m.get('request_id') == 'SHAPE_WARM_B_POST_APPLY']
        require(sw_rows, '缺少形状预热B匹配记录')
        sw = sw_rows[-1]
        # v1.1：预热输入在 E 处分叉 ⇒ 其 FA 命中应恰为 E（=T 时与 v1.0 语义一致）；
        # 正式请求自身的 FA 命中仍须为 T（由下方 actual 断言把守）
        require(sw['executable_prefix'] == expected_E and sw['physical_fa_hit'] == expected_E and
                sw['input_token_count'] == len(pt), '形状预热B准入与计划E/FA前沿不一致')

        snapshots.append(triton_snapshot('post_request'))
        base = {'post_warmup': 'engine_ready', 'post_warmup2': 'post_flush', 'post_control': 'post_warmup2',
                'post_shape_warm': 'post_control', 'post_request': 'post_shape_warm'}
        triton_windows = [triton_delta(next(s for s in snapshots if s['label'] == base[k]),
                                       next(s for s in snapshots if s['label'] == k))
                          for k in ['post_warmup', 'post_warmup2', 'post_control', 'post_shape_warm', 'post_request']]
        new_request = triton_windows[-1]['new_entries']
        require(new_request == 0, '请求窗口存在未预热编译（%d 项：%s）——形状预热失效' %
                (new_request, triton_windows[-1]['new_names'][:5]))

        parts = decompose(json.loads((out / 'stream_chunks.json').read_text()), trace)
        summary = {'status': 'PASS', 'mode': 'measure_strict_warm', 'policy': os.environ['PILOT_POLICY'],
                   'pending': pending['continuation_id'],
                   'H_physical_fa_hit': actual['physical_fa_hit'],
                   'E_executable_prefix': actual['executable_prefix'],
                   'G_replay_gap': actual['physical_fa_hit'] - actual['executable_prefix'],
                   'prefill_token_count': prefill_row['extend_token_count'],
                   'batch_prefix_length': prefill_row['batch_prefix_length'],
                   # 口径A：计算
                   'cuda_prefill_ms': cuda_prefill_ms, 'cuda_decode_ms': cuda_decode_ms,
                   'cow_copy_count': len(trace['copies']), 'cow_copy_cuda_ms': cow_cuda_ms,
                   'cow_copy_host_ms': cow_host_ms, 'match_native_ms': match_native_ms,
                   # 口径B：请求
                   'ttft_ms': (first - start) / 1e6, 'request_wall_ms': (end - start) / 1e6,
                   'stream_tpot_ms': (arrivals[-1]['ns'] - first) / 7e6,
                   'timing_decomposition': parts,
                   # 口径C：控制（审计增强；分段单列）
                   'control_rpc_wall_ms': control_rpc_wall_ms,
                   'control_timing_stages': result['timing'],
                   'census_counts': {'physical': result['census']['physical_domain_size'],
                                     'mapped': len(result['census']['mapped']),
                                     'unmapped_extras': len(result['census']['unmapped_extras'])},
                   'selected_cpu': result['policy_selected_cpu_ids'], 'plan_recovery_cost_ms': result['plan']['recovery_cost_ms'],
                   # 测量模式与形状预热有效性
                   'measure_mode': True, 'audit_entries_total': len(audit_entries),
                   'shape_warm_b': {'wall_ms': shape_warm_wall_ms, 'E': sw['executable_prefix'],
                                    'fa_hit': sw['physical_fa_hit'], 'output_len': len(throw['output_ids'])},
                   'warmup1_wall_ms': warmup1_ms, 'warmup2_wall_ms': warmup2_ms,
                   'warmup_observe_rpc_wall_ms': sum(observe_wall2) / len(observe_wall2),
                   'flush_wall_ms': flush_wall_ms, 'engine_boot_ms': boot_ms,
                   'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6,
                   'triton_snapshots': [{'label': s['label'], 'entries': s['entries']} for s in snapshots],
                   'triton_new_kernels': triton_windows,
                   'cold_output_match': True, 'output_ids': output['output_ids'], 'native_matches': all_matches,
                   '说明': '测量模式：请求窗口（发送→首Token）零A类审计零落盘；B/C类原生操作全保留；'
                          'CUDA事件是设备时间线区间而非纯内核之和'}
        write(out / 'summary.json', summary)
        print('完成', os.environ['PILOT_POLICY'], pending['continuation_id'],
              'E', summary['E_executable_prefix'], 'H', summary['H_physical_fa_hit'],
              'TTFT', round(summary['ttft_ms']), 'ms compute', round(parts['compute_cuda_ms'], 1),
              '请求窗编译', new_request, flush=True)
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
