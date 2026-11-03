"""G2 运行器：B5-C4/K=2 Exact 与 Marginal Greedy 双臂 GPU 正确性与性能测量（测量模式）。

生命周期序（strict 臂，全部在同一受控进程内，Triton 编译缓存进程内保留）：
8 条冻结历史重放（首轮即控制溯源轮——B5-C4 单轮重放即物化 7 候选，与 b5d4b 相同）
→ 唯一 b5d4_apply（G2 控制路径：同一真实 Epoch 上按 G2_SOLVER 在线求解——Exact 臂走生产
plan_strict 自动 Exact；Greedy 臂走生产 Marginal Greedy 适配——再 Bind/Preflight/reconcile_strict）
→ 事后对照 Oracle（g2_reference.json 的 e3 冻结行，仅在求解完成后读入）
→ 形状预热 B（apply 后：pt[:E]+确定性随机尾，在 E 处分叉，max_new_tokens=2，不进正式计时）
→ 正式流式请求（测量模式：窗口内零 A 类审计）→ 请求后 b5d4_state 全量观察。
cold 臂：全新 Engine 空缓存 → 直接目标请求（Cold 参照补采，A/B/D 三臂；C 用 b5d4b 冻结 Cold）。

三口径：A 计算口径=CUDA prefill/decode 与 COW 事件；B 请求口径=TTFT 六段分解+request_wall；
C 控制口径=apply RPC 墙钟+分段。client 与 engine 同用 CLOCK_MONOTONIC。
B5-C4 语义：Prefill Token 数按「原生恢复起点 E 与完整输入长度」校验（extend=len(pt)−E），
G_model=T−E 为模型口径缺口，两者分列（输入含前沿之外的追加 Token，不照搬 C/K3 的 G+32）。
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
from .persistence import call_saved

OUT = Path('/b5out')
ARM = os.environ['B5D4_ARM']
RUN = os.environ['B5D4_RUN']
require(ARM in ('strict', 'cold'), 'G2 只使用 strict/cold 臂')
SOLVER = os.environ.get('G2_SOLVER')
PENDING = os.environ['G2_PENDING']
require(os.environ.get('FS_MEASURE_MODE') == '1', '缺少测量模式环境变量')
if ARM == 'strict':
    require(SOLVER in ('Exact', 'Greedy'), 'strict 臂要求 G2_SOLVER=Exact|Greedy')
    PHASE = os.environ['G2_PHASE']; require(PHASE in ('gate', 'formal'), 'G2_PHASE=gate|formal')
    ROUND = os.environ.get('G2_ROUND', '0')


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
    """TTFT 六段分解（client/engine 同 CLOCK_MONOTONIC，可直接衔接）。"""
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

        if ARM == 'cold':
            # ── Cold 参照补采臂：空缓存 → 设置钩子目标 → 直接请求（与 b5d4b cold 协议一致）──
            call('g2_set_cold_target', target=PENDING)
            pending = next(p for p in inputs['pendings'] if p['continuation_id'] == PENDING)
            sampling = {'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True}
            write(out / 'request.json', {'pending': pending, 'sampling': sampling})
            incremental = bool(engine.tokenizer_manager.incremental_streaming_output)
            start = time.monotonic_ns(); chunks = []; arrivals = []; first = None
            for chunk in engine.generate(input_ids=pending['token_ids'], sampling_params=sampling, rid=pending['continuation_id'], stream=True):
                now = time.monotonic_ns(); chunk = copy.deepcopy(chunk); chunks.append(chunk)
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
                    int(output['meta_info'].get('num_retractions', 0) or 0) == 0, 'Cold臂输出或回撤异常')
            trace = post['trace']
            matches = [m for m in trace['matches'] if m.get('native', {}).get('request_id') == PENDING]
            require(matches, 'Cold臂缺少真实匹配证据')
            actual = matches[-1]['native']
            require(actual['input_token_digest'] == pending['token_prefix_sha256'], 'Cold臂Pending输入不符')
            require(actual['physical_fa_hit'] == actual['executable_prefix'] == 0, 'Cold命中旧状态')
            write(out / 'summary.json', {'status': 'PASS', 'arm': 'cold', 'phase': 'cold_reference',
                                         'pending': PENDING, 'fa_hit': 0, 'executable_prefix': 0,
                                         'input_token_count': len(pending['token_ids']),
                                         'output_ids': output['output_ids'],
                                         'request_wall_ms': (end - start) / 1e6, 'ttft_ms': (first - start) / 1e6,
                                         'engine_boot_ms': boot_ms,
                                         '说明': 'Cold参照补采臂：全新Engine空缓存直接请求；计时仅为执行记录'})
            print('完成 cold', PENDING, '8 Token', flush=True)
            return

        # ── strict 臂：8 条冻结历史单轮重放（B5-C4 单轮即物化 7 候选；首轮即控制溯源轮）──
        observe_wall = []
        warmup_start = time.monotonic_ns()
        for i, request in enumerate(inputs['history'], 1):
            output_w = engine.generate(input_ids=request['token_ids'], sampling_params=inputs['sampling'], rid=request['request_id'])
            t = time.monotonic_ns()
            state = call('b5c3_observe')
            observe_wall.append((time.monotonic_ns() - t) / 1e6)
            matches = state['matches'].get(request['request_id'], [])
            require(matches and all(x['input_token_digest'] == request['token_prefix_sha256'] and
                                    x['input_token_count'] == len(request['token_ids']) for x in matches), '历史实际输入不符')
            require(len(output_w.get('output_ids', [])) == 1 and int(output_w['meta_info'].get('num_retractions', 0) or 0) == 0, '历史异常')
            write(out / f'history_{i:02d}.json', {'request': request, 'output': output_w, 'state': state})
        warmup_ms = (time.monotonic_ns() - warmup_start) / 1e6
        snapshots.append(triton_snapshot('post_warmup'))

        # ── 唯一一次真实协调：控制路径内在线求解（G2 双臂）──
        control_start = time.monotonic_ns()
        result = client._call({'op': 'checkpoint_control', 'action': 'b5d4_apply', 'nonce': str(time.time_ns())})
        control_end = time.monotonic_ns()
        write(out / 'reconciliation.json', result)
        require(result.get('ok'), result.get('error', '协调失败'))
        require(result['solver'] == SOLVER, '求解臂标识不符')
        decision = result['strict_decision']
        require(result['target']['pending']['continuation_id'] == PENDING, '目标Pending不符')
        pending = result['target']['pending']
        write(out / 'target_request.json', pending)
        control_rpc_wall_ms = (control_end - control_start) / 1e6
        snapshots.append(triton_snapshot('post_control'))

        # ── 事后对照 Oracle（e3 冻结行；仅在求解完成后读入，不参与选择）──
        ref = json.loads((OUT / 'g2_reference.json').read_text())
        rrow = ref['Exact' if SOLVER == 'Exact' else 'Marginal Greedy']
        require(set(decision['selected_ids']) == set(rrow['selected']), '在线选集与e3冻结参照不一致：' + repr(decision['selected_ids']))
        require(abs(result['plan']['recovery_cost_ms'] - rrow['J_ms']) <= 1e-9, 'J与e3冻结参照不一致')
        for prow in result['plan']['pending']:
            e = rrow['per_pending'][prow['pending_id']]
            require((prow['E'], prow['T'], prow['G']) == (e['E'], e['T'], e['G']), 'E-T-G与e3冻结参照不符：' + prow['pending_id'])

        # ── 形状预热 B（apply 后、正式请求前）：计划保留集下原生准入恰为计划 E ──
        # 输入 pt[:E] + 确定性随机尾 L=len(pt)−E，在 E 处立即分叉——只在随机分支物化状态，
        # 正式请求匹配深度不超计划 E；内核形状等价：在位置 E 准入、extend L 个 Token、decode 起步。
        pt = list(pending['token_ids'])
        expected_E = int(result['target']['expected']['E'])
        L = len(pt) - expected_E
        vocab = tok.vocab_size
        rng = random.Random('g2_v1/%s/%s/%s/%s' % (SOLVER, PHASE, ROUND, pending['continuation_id']))
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
                                          'note': 'apply后形状预热：pt[:E]+确定性随机尾，计划E准入+目标extend/decode形状编译，不进正式计时'})
        snapshots.append(triton_snapshot('post_shape_warm'))

        # ── 正式流式请求（测量模式：窗口内零 A 类审计）──
        sampling = {'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True}
        write(out / 'request.json', {'pending': pending, 'sampling': sampling})
        incremental = bool(engine.tokenizer_manager.incremental_streaming_output)
        start = time.monotonic_ns(); chunks = []; arrivals = []; first = None
        for chunk in engine.generate(input_ids=pending['token_ids'], sampling_params=sampling, rid=pending['continuation_id'], stream=True):
            now = time.monotonic_ns(); chunk = copy.deepcopy(chunk); chunks.append(chunk)
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
        require(output['output_ids'] == cold['output_ids'], '与Cold参照输出不一致')

        # ── 请求级合同（B5-C4 语义：Prefill 按原生恢复起点与完整输入长度校验）──
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
        require(actual['physical_fa_hit'] == pending['attention_frontier'], 'FA命中与观察前沿不一致')
        require(actual['executable_prefix'] == expected_E, '实际可执行前缀与计划E不符')
        require(prefill_row['batch_prefix_length'] == expected_E and
                prefill_row['extend_token_count'] == len(pending['token_ids']) - expected_E,
                'Prefill区间与E/完整输入长度合同不符')
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
        else:
            require(target['source_checkpoint'] is None and actual['cow_source_slots'] == [] and
                    actual['working_slots'] == [] and not trace['copies'], 'E=0却出现恢复源/COW痕迹')
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
        require(sw['executable_prefix'] == expected_E and sw['physical_fa_hit'] == expected_E and
                sw['input_token_count'] == len(pt), '形状预热B准入与计划E/FA前沿不一致')

        snapshots.append(triton_snapshot('post_request'))
        base = {'post_warmup': 'engine_ready', 'post_control': 'post_warmup', 'post_shape_warm': 'post_control',
                'post_request': 'post_shape_warm'}
        triton_windows = [triton_delta(next(s for s in snapshots if s['label'] == base[k]),
                                       next(s for s in snapshots if s['label'] == k))
                          for k in ['post_warmup', 'post_control', 'post_shape_warm', 'post_request']]
        new_request = triton_windows[-1]['new_entries']
        require(new_request == 0, '请求窗口存在未预热编译（%d 项：%s）——形状预热失效' %
                (new_request, triton_windows[-1]['new_names'][:5]))

        parts = decompose(json.loads((out / 'stream_chunks.json').read_text()), trace)
        summary = {'status': 'PASS', 'arm': 'strict', 'phase': PHASE, 'round': ROUND, 'solver': SOLVER,
                   'pending': pending['continuation_id'],
                   'selection_source': result['selection_source'], 'strict_decision': decision,
                   'selected_ids': list(result['plan']['selected_ids']),
                   'plan_recovery_cost_ms': result['plan']['recovery_cost_ms'],
                   'oracle_e3': {'selected': rrow['selected'], 'J_ms': rrow['J_ms'], 'match': True},
                   'independent_oracle': result['oracle'],
                   'H_physical_fa_hit': actual['physical_fa_hit'],
                   'E_executable_prefix': actual['executable_prefix'],
                   'G_model_replay_gap': target['expected']['G'],
                   'tail_beyond_frontier': len(pending['token_ids']) - pending['attention_frontier'],
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
                   # 口径C：控制
                   'control_rpc_wall_ms': control_rpc_wall_ms,
                   'control_timing_stages': result['timing'],
                   'evictions': [(o['checkpoint_id'], o['returned']) for o in result['operations']],
                   # 测量模式与形状预热有效性
                   'measure_mode': True, 'audit_entries_total': len(audit_entries),
                   'shape_warm_b': {'wall_ms': shape_warm_wall_ms, 'E': sw['executable_prefix'],
                                    'fa_hit': sw['physical_fa_hit'], 'output_len': len(throw['output_ids'])},
                   'warmup_wall_ms': warmup_ms, 'warmup_observe_rpc_wall_ms': sum(observe_wall) / len(observe_wall),
                   'engine_boot_ms': boot_ms, 'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6,
                   'triton_snapshots': [{'label': s['label'], 'entries': s['entries']} for s in snapshots],
                   'triton_new_kernels': triton_windows,
                   'cold_output_match': True, 'output_ids': output['output_ids'], 'native_matches': all_matches,
                   '说明': 'G2测量模式：请求窗口（发送→首Token）零A类审计零落盘；B/C类原生操作全保留；'
                          'CUDA事件是设备时间线区间而非纯内核之和；B5-C4语义Prefill=len(pt)−E、G_model=T−E分列'}
        write(out / 'summary.json', summary)
        print('完成 G2', PHASE, SOLVER, pending['continuation_id'],
              'E', summary['E_executable_prefix'], 'TTFT', round(summary['ttft_ms']),
              'ms cuda_prefill', round(cuda_prefill_ms, 1), '请求窗编译', new_request, flush=True)
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
