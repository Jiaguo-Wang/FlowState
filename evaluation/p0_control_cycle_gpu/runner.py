"""四 Pending 完整控制周期运行器（第一轮 GPU 验证：1 轮 × 3 策略各一生命周期）。

生命周期序（与 c_formal_perf_gpu/runner.py 的正确性部分同构，四 Pending 化）：
16 预热重放(首轮 history_pre) → flush_cache → 空缓存断言 → 16 预热二次重放(history,控制溯源)
→ 唯一 b5d4_apply（Scheduler 进程内三策略真实内核在线 Select + Recurrent-only Reconciliation）
→ [T3 之后、请求前] 形状预热块：四 Pending 各一次 v1.1（pt[:E]+确定性随机尾，种子与 c_formal
   逐字相同；编译目标 extend/decode 形状；属测量基础设施，窗口外单列，不进控制周期）
→ 四 Pending 串行正式流式请求（8 Token，与冻结 Cold 参照逐 Token 对照）
→ T4（第 4 请求执行终点）→ 请求后 b5d4_state 全量观察 → 全部正确性门禁
→ 窗口外取证（t2g1 冻结 oracle 对照/三策略 E-G 合同；磁盘与 GPU 由宿主 launch 记录）。

计时合同（双口径显式并列，禁止混用与事后扣除——论文证据审计 §5.2 风险二修复）：
- control_cycle_ms = T3−T0：控制周期（观察→在线 Select→协调→后验），不含形状预热与请求；
- cycle_ms_full_incl_warm = T4−T0：完整周期，含形状预热块与全部间隙（RPC 尾段/预热间/请求间）。
T0 = apply 内 final_heavy_observation 开始（绝对单调 ns，控制路径返回）；
T3 = 协调+必要后验完成；T4 = 第 4 请求执行终点。整数纳秒恒等式
(T4−T0) == Σ(apply内段) + Σ(段间隙) + (RPC尾段) + Σ(形状预热) + Σ(请求段) + Σ(请求间隙)
精确断言；client 与 engine 同用 CLOCK_MONOTONIC，跨进程时间戳直接衔接。
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
POLICY = os.environ['PILOT_POLICY']
require(MODE == 'strict_warm' and ARM == 'strict', '本轮固定 strict_warm + strict 臂语义')
require(os.environ.get('FS_MEASURE_MODE') == '1', '缺少测量模式环境变量')
require(POLICY in ('FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired'), '未知策略：' + POLICY)


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


def decompose(stream, exec_rows, compute_rows):
    """TTFT 六段分解（单请求过滤后的 execution/compute 行；client/engine 同 CLOCK_MONOTONIC）。"""
    e0, c0 = exec_rows[0], compute_rows[0]
    d0 = next((c for c in compute_rows if not c['is_extend']), None)
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
    require(abs(rebuilt - parts['ttft_ms_check']) < 5.0,
            'TTFT 分解不闭合：%.1f vs %.1f' % (rebuilt, parts['ttft_ms_check']))
    return parts


def replay_warmup(engine, call, inputs, out, prefix, observe_wall_acc):
    """16 条冻结预热真实重放；首轮写 history_pre_XX.json（留证），二次（flush 后）写 history_XX.json
    （控制溯源输入——node_id 属 flush 后重建世代）。两轮均逐请求 digest/输入长度/输出数合同验证。"""
    start = time.monotonic_ns()
    for i, request in enumerate(inputs['history'], 1):
        output = engine.generate(input_ids=request['token_ids'], sampling_params=inputs['sampling'],
                                 rid=request['request_id'])
        t = time.monotonic_ns()
        state = call('b5c3_observe')
        observe_wall_acc.append((time.monotonic_ns() - t) / 1e6)
        matches = state['matches'].get(request['request_id'], [])
        require(matches and all(x['input_token_digest'] == request['token_prefix_sha256'] and
                                x['input_token_count'] == len(request['token_ids']) for x in matches), '历史实际输入不符')
        require(len(output.get('output_ids', [])) == 1 and
                int(output['meta_info'].get('num_retractions', 0) or 0) == 0, '历史异常')
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
        require(hashlib.sha256(str(tok.chat_template).encode()).hexdigest() == config['chat_template_sha256'],
                '模板身份不符')
        from .observation import real_api_preflight
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

        # ── 唯一一次真实协调：Scheduler 进程内三策略真实内核在线 Select（T0–T3 在其中）──
        rpc_start = time.monotonic_ns()
        result = client._call({'op': 'checkpoint_control', 'action': 'b5d4_apply', 'nonce': str(time.time_ns())})
        rpc_end = time.monotonic_ns()
        write(out / 'reconciliation.json', result)
        require(result.get('ok'), result.get('error', '协调失败'))
        require(result['policy'] == POLICY, '策略标识不符')
        require(result['census']['physical_domain_size'] == 16 and len(result['census']['mapped']) == 16 and
                not result['census']['unmapped_extras'] and not result['census']['missing_physical'],
                'census 域异常：physical=%d mapped=%d extras=%d missing=%d' % (
                    result['census']['physical_domain_size'], len(result['census']['mapped']),
                    len(result['census']['unmapped_extras']), len(result['census']['missing_physical'])))
        control_rpc_wall_ms = (rpc_end - rpc_start) / 1e6
        snapshots.append(triton_snapshot('post_control'))

        # 四 Pending 声明顺序执行（三策略同一顺序，公平口径；串行安全性已由 CPU 侧 LCP×E 判定）
        pendings = [trow['pending'] for trow in result['targets'].values()]
        pendings.sort(key=lambda p: [x['continuation_id'] for x in inputs['pendings']].index(p['continuation_id']))
        order = [p['continuation_id'] for p in pendings]
        expected = {pid: result['targets'][pid]['expected'] for pid in order}
        write(out / 'request_plan.json', {'policy': POLICY, 'order': order,
                                          'expected_E_T_G': {pid: {k: expected[pid][k] for k in ('E', 'T', 'G')}
                                                             for pid in order}})

        # ── T0–T3 时钟复核：apply 内分段（绝对 ns）与恒等式（整数算术）──
        clock = result['clock']
        t0_ns, t3_ns = clock['t0_ns'], clock['t3_ns']
        stages = clock['stages']
        require(stages and stages[0]['start_ns'] == t0_ns and stages[-1]['end_ns'] == t3_ns, '控制时钟边界异常')
        seg_total = sum(s['end_ns'] - s['start_ns'] for s in stages)
        stage_gaps = [stages[i + 1]['start_ns'] - stages[i]['end_ns'] for i in range(len(stages) - 1)]
        require(all(g >= 0 for g in stage_gaps), 'apply内分段间隙为负')
        require((t3_ns - t0_ns) == seg_total + sum(stage_gaps), 'apply内恒等式失败（整数）')
        gap_rpc_tail_ns = rpc_end - t3_ns   # admission_baseline（A类基线）+ 序列化传输，如实计为间隙
        require(gap_rpc_tail_ns >= 0, 'RPC尾段为负')

        # ── 形状预热块（T3 后、请求前；窗口外单列）：四 Pending 各一次 v1.1 ──
        # v1.1（c_formal 同款）：pt[:E]+确定性随机尾在 E 处立即分叉——只在随机分支物化状态，
        # 正式请求匹配深度不超计划 E；内核形状等价（E 准入、extend L、decode 起 kv=T+32）；
        # 种子逐字沿用 c_formal_v1.1，三策略各自 E 向量分别预热。
        vocab = tok.vocab_size
        warm_rows = []
        warm_ns = []
        for p in pendings:
            pid = p['continuation_id']
            T = p['attention_frontier']
            pt = list(p['token_ids'])
            require(len(pt) == T + 32, '输入全长与 T+32 不符：%s' % pid)
            e = int(expected[pid]['E'])
            L = len(pt) - e
            rng = random.Random('c_formal_v1.1/%s/%s' % (POLICY, pid))
            tail = [rng.randrange(vocab) for _ in range(L)]
            while tail and tail[0] == pt[e]:
                tail[0] = rng.randrange(vocab)   # 确保在 E 处首 Token 即分叉（FA 命中恰为 E）
            sw_ids = pt[:e] + tail
            require(len(sw_ids) == len(pt) and sw_ids[e:] != pt[e:], '形状预热输入构造异常：%s' % pid)
            ws = time.monotonic_ns()
            throw = engine.generate(input_ids=sw_ids,
                                    sampling_params={'temperature': 0, 'max_new_tokens': 2, 'ignore_eos': True},
                                    rid='SHAPE_WARM_%s' % pid)
            we = time.monotonic_ns()
            require(len(throw.get('output_ids', [])) == 2 and
                    int(throw['meta_info'].get('num_retractions', 0) or 0) == 0, '形状预热输出异常：%s' % pid)
            warm_rows.append({'pending': pid, 'start_ns': ws, 'end_ns': we, 'wall_ms': (we - ws) / 1e6,
                              'input_len': len(sw_ids), 'planned_E': e, 'diverge_from': e,
                              'output_ids': throw['output_ids']})
            warm_ns.append((ws, we))
        snapshots.append(triton_snapshot('post_warm_block'))
        write(out / 'shape_warm_block.json',
              {'policy': POLICY, 'rows': warm_rows,
               'note': 'apply后形状预热块(v1.1,四Pending各一次)：计划E准入+目标extend/decode形状编译。'
                       '口径归属：不进 control_cycle_ms(T3−T0)；计入 cycle_ms_full_incl_warm(T4−T0)，'
                       '恒等式如实含 warm 段与其前后间隙；两口径并列报告，禁止事后扣除 warm 伪造连续生产窗口'})

        # ── 四 Pending 串行正式流式请求（8 Token；测量模式：窗口内零 A 类审计）──
        sampling = {'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True}
        incremental = bool(engine.tokenizer_manager.incremental_streaming_output)
        streams = {}
        for p in pendings:
            pid = p['continuation_id']
            start = time.monotonic_ns()
            chunks = []
            arrivals = []
            first = None
            for chunk in engine.generate(input_ids=p['token_ids'], sampling_params=sampling, rid=pid, stream=True):
                now = time.monotonic_ns()
                chunk = copy.deepcopy(chunk)
                chunks.append(chunk)
                arrivals.append({'ns': now, 'output_ids': chunk.get('output_ids'),
                                 'completion_tokens': chunk.get('meta_info', {}).get('completion_tokens')})
                if first is None and (chunk.get('output_ids') or
                                      chunk.get('meta_info', {}).get('completion_tokens', 0) > 0):
                    first = now
            end = time.monotonic_ns()
            require(chunks and first is not None, '缺少真实流式首Token：%s' % pid)
            output = copy.deepcopy(chunks[-1])
            if incremental:
                output['output_ids'] = [t for c in chunks for t in c.get('output_ids', [])]
            require(len(output['output_ids']) == output['meta_info']['completion_tokens'],
                    '流式Token拼接与真实完成计数不一致：%s' % pid)
            streams[pid] = {'chunks': chunks, 'arrivals': arrivals, 'start_ns': start, 'end_ns': end,
                            'first_token_ns': first, 'output': output,
                            'stream_path': out / ('stream_chunks_%s.json' % pid)}
            write(streams[pid]['stream_path'],
                  {'chunks': chunks, 'arrivals': arrivals, 'start_ns': start, 'end_ns': end,
                   'first_token_ns': first, 'incremental_streaming_output': incremental})
            write(out / ('output_%s.json' % pid), output)
        t4_ns = streams[order[-1]]['end_ns']   # T4 = 第 4 请求执行终点
        require(t4_ns - t0_ns > 0, 'T4 不晚于 T0')

        # ── T4 之后：请求后全量观察 + 全部正确性门禁（窗口外）──
        post = call('b5d4_state')
        write(out / 'post_request.json', post)
        trace = post['trace']

        # 门槛 1：Admission 四次、声明顺序、逐 rid 一次
        adm = [a for a in trace['admission']]
        require([a['rid'] for a in adm] == order, 'Admission顺序与声明不一致：%r' % [a['rid'] for a in adm])

        pending_rows = []
        for p in pendings:
            pid = p['continuation_id']
            st = streams[pid]
            output = st['output']
            e = expected[pid]
            require(len(output.get('output_ids', [])) == 8 and output['meta_info']['completion_tokens'] == 8 and
                    int(output['meta_info'].get('num_retractions', 0) or 0) == 0,
                    '正式请求输出或回撤异常：%s' % pid)
            cold = json.loads((OUT / ('cold_reference_' + pid + '.json')).read_text())
            require(output['output_ids'] == cold['output_ids'], '与 Cold 参照输出不一致：%s' % pid)

            all_matches = [m['native'] for m in trace['matches']
                           if m.get('native', {}).get('request_id') == pid]
            require(all_matches, '缺少真实匹配证据：%s' % pid)
            actual = all_matches[-1]
            require(actual['input_token_digest'] == p['token_prefix_sha256'], 'Pending输入不符：%s' % pid)
            require(actual['physical_fa_hit'] == p['attention_frontier'], 'FA命中与冻结边界不一致：%s' % pid)
            require(actual['executable_prefix'] == e['E'], '实际可执行前缀与计划E不符：%s' % pid)
            exec_rows = [r for r in trace['execution'] if r.get('rid') == pid]
            compute_rows = [r for r in trace['compute'] if r.get('rid') == pid]
            require(exec_rows and all(x.get('native_run_returned') for x in exec_rows), 'Prefill/Decode未完成：%s' % pid)
            require(len(compute_rows) == len(exec_rows), 'compute与execution行数不一致：%s' % pid)
            prefill_row = exec_rows[0]
            require(compute_rows and compute_rows[0]['is_extend'] and
                    isinstance(prefill_row['extend_token_count'], int), '首执行不是Prefill：%s' % pid)
            require(prefill_row['batch_prefix_length'] == e['E'] and
                    prefill_row['extend_token_count'] == len(p['token_ids']) - e['E'],
                    'Prefill区间与E/G合同不符：%s' % pid)
            bound_slots = []
            if e['E'] > 0:
                bound = next(x['node'] for x in result['reconciliation']['binding']['candidates']
                             if x['candidate_id'] == result['targets'][pid]['source_checkpoint'])
                bound_slots = bound['slots']
                require(actual['best_match_node'] == bound['node_id'] and
                        actual['cow_source_slots'] == bound['slots'], '恢复来源错误：%s' % pid)
                require(any(c.get('returned') and c['source_slots'] == bound['slots'] and
                            c['destination_slots'] == actual['working_slots'] for c in trace['copies']),
                        'COW未完成（结构性验证；内容验证=请求后全量观察+Cold逐Token一致）：%s' % pid)
                kept = {s for c in result['reconciliation']['binding']['candidates']
                        if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
                require(not kept.intersection(actual['working_slots']), '工作状态与保留状态别名：%s' % pid)
            cow_rows = [c for c in trace['copies'] if bound_slots and c['source_slots'] == bound_slots]
            parts = decompose(st, exec_rows, compute_rows)
            pending_rows.append({
                'pending_id': pid, 'E': e['E'], 'T': e['T'], 'G': e['G'],
                'physical_fa_hit': actual['physical_fa_hit'], 'executable_prefix': actual['executable_prefix'],
                'prefill_token_count': prefill_row['extend_token_count'],
                'batch_prefix_length': prefill_row['batch_prefix_length'],
                'cold_output_match': True,
                'ttft_ms': (st['first_token_ns'] - st['start_ns']) / 1e6,
                'request_wall_ms': (st['end_ns'] - st['start_ns']) / 1e6,
                'stream_tpot_ms': (st['arrivals'][-1]['ns'] - st['first_token_ns']) / 7e6,
                'timing_decomposition': parts,
                'cuda_prefill_ms': sum(c['cuda_elapsed_ms'] for c in compute_rows if c['is_extend']),
                'cuda_decode_ms': sum(c['cuda_elapsed_ms'] for c in compute_rows if not c['is_extend']),
                'cow_copy_count': len(cow_rows),
                'cow_copy_cuda_ms': sum(c.get('cuda_elapsed_ms', 0.0) for c in cow_rows),
                'start_ns': st['start_ns'], 'end_ns': st['end_ns'], 'first_token_ns': st['first_token_ns'],
                'output_ids': output['output_ids'],
            })

        # 门槛 2：状态完整性（终态一次断言；保护集=四请求重算并集口径）
        require(post['state']['selected_payload'] == result['selected_payload_after'] and
                post['state']['selected_components'] == result['admission_baseline']['selected_components'],
                '历史Recurrent状态受损')
        require(post['state']['fa_protected_payload'] == result['fa_protected_payload'], '非重算FA内容变化')
        require(all(post['state']['fa_map'].get(k) == v for k, v in result['fa_map_before'].items()),
                '历史FA映射丢失或改变')

        # 门槛 3：测量模式有效性（窗口内零审计）+ 形状预热 v1.1 合同（四行）
        audit_entries = trace.get('audit', [])
        require(len(audit_entries) == 2 and all(a['kind'] == 'cache_state_full_hash' for a in audit_entries),
                '测量模式失效：审计项应为 apply基线+请求后各一次，实测 %d 次' % len(audit_entries))
        sw_by_pid = {}
        for m in trace.get('shape_warm_matches', []):
            if str(m.get('request_id', '')).startswith('SHAPE_WARM_'):
                sw_by_pid[str(m['request_id'])] = m
        require(sorted(sw_by_pid) == sorted('SHAPE_WARM_' + pid for pid in order), '形状预热匹配记录不全')
        sw_rows = []
        for pid in order:
            sw = sw_by_pid['SHAPE_WARM_' + pid]
            require(sw['executable_prefix'] == expected[pid]['E'] and
                    sw['physical_fa_hit'] == expected[pid]['E'] and
                    sw['input_token_count'] == len(result['targets'][pid]['pending']['token_ids']),
                    '形状预热B准入与计划E/FA前沿不一致：%s' % pid)
            sw_rows.append({'pending': pid, 'executable_prefix': sw['executable_prefix'],
                            'physical_fa_hit': sw['physical_fa_hit']})

        # 门槛 4：请求窗口零新增编译（形状预热块后→请求后）
        snapshots.append(triton_snapshot('post_request'))
        base = {'post_warmup': 'engine_ready', 'post_warmup2': 'post_flush', 'post_control': 'post_warmup2',
                'post_warm_block': 'post_control', 'post_request': 'post_warm_block'}
        triton_windows = [triton_delta(next(s for s in snapshots if s['label'] == base[k]),
                                       next(s for s in snapshots if s['label'] == k))
                          for k in ['post_warmup', 'post_warmup2', 'post_control', 'post_warm_block', 'post_request']]
        new_request = triton_windows[-1]['new_entries']
        require(new_request == 0, '请求窗口存在未预热编译（%d 项：%s）——形状预热失效' %
                (new_request, triton_windows[-1]['new_names'][:5]))

        # ── T0–T4 整数恒等式（含 RPC 尾段、形状预热块与请求间隙；全部绝对 ns）──
        req_segs = [(streams[pid]['start_ns'], streams[pid]['end_ns']) for pid in order]
        # 段间间隙（rpc_end→预热1、预热段间、预热末→请求1、请求段间）；段长由 warm/req_total 单列，
        # 间隙只取相邻段边界差——若把段内端点也放进相邻差再另加段长，即段长双计（attempt2 失败根因）
        segments = list(warm_ns) + list(req_segs)
        gaps = [segments[0][0] - rpc_end] + [segments[i + 1][0] - segments[i][1]
                                             for i in range(len(segments) - 1)]
        require(all(g >= 0 for g in gaps), '窗口分段间隙为负')
        warm_total = sum(we - ws for ws, we in warm_ns)
        req_total = sum(we - ws for ws, we in req_segs)
        require((t4_ns - t0_ns) == (t3_ns - t0_ns) + gap_rpc_tail_ns + warm_total + req_total + sum(gaps),
                'T0–T4 恒等式失败：存在重复累加或遗漏')
        residual_rows = []
        for i, pid in enumerate(order):
            gap_after = (req_segs[i + 1][0] - req_segs[i][1]) if i + 1 < len(order) else (t4_ns - req_segs[i][1])
            residual_rows.append({'pending_id': pid, 'gap_after_request_ns': gap_after,
                                  'note': '请求间隙残余（不含请求自身执行）；末项=0（T4=第4请求终点）'})

        # ── 窗口外取证：t2g1 冻结 oracle 对照（选集/J/E-G 逐位；不参与任何在线决策）──
        t0o = time.monotonic_ns()
        t2g1 = json.loads((OUT / 't2g1_plans.json').read_text())['plans']
        frozen = t2g1[POLICY]
        require(sorted(result['policy_selected_cpu_ids']) == sorted(frozen['selected_checkpoint_ids']),
                '在线选集与冻结 oracle 不一致：%r vs %r' % (result['policy_selected_cpu_ids'],
                                                        frozen['selected_checkpoint_ids']))
        require(abs(result['plan']['recovery_cost_ms'] - frozen['J_ms_current_scope']) <= 1e-9,
                'J与冻结oracle不一致：%r vs %r' % (result['plan']['recovery_cost_ms'], frozen['J_ms_current_scope']))
        for pid in order:
            f = frozen['expected_E_G'][pid]
            require((expected[pid]['E'], expected[pid]['T'], expected[pid]['G']) == (f['E'], f['T'], f['G']),
                    'GPU实测E-G与冻结预期不符：' + pid)
        oracle_check_ms = (time.monotonic_ns() - t0o) / 1e6

        # ── 汇总落盘 ──
        summary = {'status': 'PASS', 'mode': 'four_pending_control_cycle', 'policy': POLICY,
                   'order': order,
                   'selection': {'kernel_entry': result['strict_decision']['kernel_entry'],
                                 'solver': result['strict_decision']['solver'],
                                 'selection_call_ms': result['strict_decision']['selection_call_ms'],
                                 'selected_cpu': result['policy_selected_cpu_ids'],
                                 'plan_recovery_cost_ms': result['plan']['recovery_cost_ms']},
                   'cycle': {'t0_ns': t0_ns, 't3_ns': t3_ns, 't4_ns': t4_ns,
                             'control_cycle_ms': (t3_ns - t0_ns) / 1e6,
                             'cycle_ms_full_incl_warm': (t4_ns - t0_ns) / 1e6,
                             'request_exec_total_ms': (t4_ns - streams[order[0]]['start_ns']) / 1e6,
                             'apply_stages_ms': [{'stage': s['stage'],
                                                  'ms': (s['end_ns'] - s['start_ns']) / 1e6} for s in stages],
                             'apply_stage_gaps_ms': [g / 1e6 for g in stage_gaps],
                             'rpc_tail_ms': gap_rpc_tail_ns / 1e6,
                             'control_rpc_wall_ms': control_rpc_wall_ms,
                             'shape_warm_block_ms': [{'pending': w['pending'], 'wall_ms': w['wall_ms']}
                                                     for w in warm_rows],
                             'window_gaps_ms': [g / 1e6 for g in gaps],
                             'identity_exact': True,
                             'definition_note': 'control_cycle_ms=T3−T0（不含形状预热/请求）；'
                                                'cycle_ms_full_incl_warm=T4−T0（含预热块与全部间隙）；'
                                                'request_exec_total_ms=首请求发送→T4（不含预热）；'
                                                '三口径并列，禁止混用或事后扣除'},
                   'pending_rows': pending_rows, 'residuals': residual_rows,
                   'shape_warm': sw_rows,
                   'census_counts': {'physical': result['census']['physical_domain_size'],
                                     'mapped': len(result['census']['mapped']),
                                     'unmapped_extras': len(result['census']['unmapped_extras'])},
                   'measure_mode': True, 'audit_entries_total': len(audit_entries),
                   'triton_new_kernels': triton_windows,
                   'oracle_match': True, 'oracle_check_ms': oracle_check_ms,
                   'journal_io': result.get('journal_io'),
                   'warmup1_wall_ms': warmup1_ms, 'warmup2_wall_ms': warmup2_ms,
                   'warmup_observe_rpc_wall_ms': sum(observe_wall2) / len(observe_wall2),
                   'flush_wall_ms': flush_wall_ms, 'engine_boot_ms': boot_ms,
                   'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6,
                   '说明': '双口径：control_cycle_ms=T3−T0（不含形状预热与请求）；'
                          'cycle_ms_full_incl_warm=T4−T0（含预热块与全部间隙），禁止混用/事后扣除；'
                          '恒等式(T4−T0)=apply段+间隙+RPC尾+形状预热块+请求段+请求间隙 整数精确；'
                          'T0–T3 窗口内零取证磁盘 I/O（Journal 序列化在窗内如实计入段间隙，'
                          '写盘+fsync 全部出窗）；oracle 对照与全量观察均在 T4 之后窗口外执行'}
        write(out / 'summary.json', summary)
        print('完成', POLICY, order,
              '控制周期', round(summary['cycle']['control_cycle_ms'], 1), 'ms',
              '全周期', round(summary['cycle']['cycle_ms_full_incl_warm'], 1), 'ms',
              'TTFT', [round(r['ttft_ms'], 1) for r in pending_rows], flush=True)
    except BaseException as exc:
        # 结论文件保护：summary 已落盘（全部门禁已过）时，后续异常只记旁证文件，
        # 不得以 failure.json 与结论并存污染证据链（attempt4 打印行 KeyError 教训）
        after_summary = (out / 'summary.json').exists()
        name = 'failure_after_summary.json' if after_summary else 'failure.json'
        write(out / name, {'status': 'BLOCKED_AFTER_SUMMARY' if after_summary else 'BLOCKED',
                           'error': repr(exc), 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        write(out / 'engine_cleanup.json', {'shutdown_returned': True, 'engine_created': engine is not None})


if __name__ == '__main__':
    main()
