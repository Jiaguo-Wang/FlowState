"""C Shared/Branched K=3 GPU Pilot 运行器：每臂独立 Engine 生命周期，只执行一个正式 Pending。

- census：16 条冻结预热真实重放 + 终态观察，产出物化状态 Census（不执行策略、不驱逐、不发请求）；
- strict：预热重放 → b5d4_apply（冻结 CPU 策略选择映射到物理状态并严格协调）→ 流式执行目标 Pending；
- cold：空缓存独立生命周期执行同一 Pending，作为输出一致性与全量 Prefill 对照。
计时合同：请求级性能（TTFT/流式）、控制开销（RPC 墙钟+分段）、CUDA 事件（compute/copies）分开记录。
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
            """每次 RPC 落盘留证并计时，控制开销与请求性能分开。"""

            def _call(self, request):
                return call_saved(self.port, self.timeout_s, request, out / ('rpc_' + str(time.monotonic_ns())))

        engine = ObservedEngine(**config['engine_configuration'])
        client = AuditedControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']))
        wait_for_transport(client)

        def call(action, **kw):
            r = client._call({'op': 'checkpoint_control', 'action': action, 'nonce': str(time.time_ns()), **kw})
            require(r.get('ok'), r.get('error', 'Runtime调用失败'))
            return r

        epoch_start = time.monotonic_ns()
        initial = call('b5c3_observe')
        require(not any(n['slots'] for n in initial['before']['nodes']), '新Engine非空')
        write(out / 'initial.json', initial)

        # ── 冻结预热真实重放（census 与 strict 共用；cold 不重放）──
        warmup_start = time.monotonic_ns()
        observe_wall_ns = 0
        if MODE != 'cold':
            for i, request in enumerate(inputs['history'], 1):
                output = engine.generate(input_ids=request['token_ids'], sampling_params=inputs['sampling'], rid=request['request_id'])
                t = time.monotonic_ns()
                state = call('b5c3_observe')
                observe_wall_ns += time.monotonic_ns() - t
                matches = state['matches'].get(request['request_id'], [])
                require(matches and all(x['input_token_digest'] == request['token_prefix_sha256'] and
                                        x['input_token_count'] == len(request['token_ids']) for x in matches), '历史实际输入不符')
                require(len(output.get('output_ids', [])) == 1 and int(output['meta_info'].get('num_retractions', 0) or 0) == 0, '历史异常')
                write(out / f'history_{i:02d}.json', {'request': request, 'output': output, 'state': state})
                print('历史完成', i, flush=True)
        warmup_wall_ms = (time.monotonic_ns() - warmup_start) / 1e6

        if MODE == 'census':
            # ── Census：终态真实观察 + CPU 候选域对账，仅记录事实，不做结构硬断言（由 launch 门禁把关）──
            final = call('b5c3_observe', final=True)
            write(out / 'census.json', final)
            raw = final['strict_epoch']
            mapping = json.loads((OUT / 'cpu_candidate_prefix_mapping.json').read_text())['mapping']
            rows, unmapped = physical_cpu_census(raw, inputs, mapping)
            cpu_ids = {r['cpu_checkpoint_id'] for r in rows}
            pend = [{'continuation_id': p['continuation_id'],
                     'live_planning_target_T': p['planning_target_T'],
                     'live_attention_frontier': p['attention_frontier'],
                     'declared_T': d['planning_target_T'], 'declared_frontier': d['attention_frontier'],
                     'match': p['planning_target_T'] == d['planning_target_T'] and p['attention_frontier'] == d['attention_frontier']}
                    for p, d in zip(raw['pendings'], inputs['pendings'])]
            slots_used = [s for n in final['before']['nodes'] for s in n['slots']]
            census_summary = {'status': 'PASS', 'mode': 'census',
                              'physical_domain_size': len(raw['candidates']),
                              'cpu_domain_size': len(mapping), 'mapped_count': len(rows),
                              'unmapped_extras': unmapped,
                              'missing_physical': sorted(set(mapping) - cpu_ids),
                              'mapping': rows, 'pendings_live_vs_declared': pend,
                              'slots_used': len(slots_used), 'free_slots': len(final['before']['free_slots']),
                              'pool_capacity': final['before']['pool_capacity'],
                              'warmup_wall_ms': warmup_wall_ms,
                              'warmup_observe_rpc_wall_ms': observe_wall_ns / 1e6,
                              'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6}
            write(out / 'summary.json', census_summary)
            print('Census完成：物理', len(raw['candidates']), '映射', len(rows), '额外', len(unmapped), flush=True)
            return

        if MODE == 'strict':
            control_start = time.monotonic_ns()
            result = client._call({'op': 'checkpoint_control', 'action': 'b5d4_apply', 'nonce': str(time.time_ns())})
            control_end = time.monotonic_ns()
            write(out / 'reconciliation.json', result)
            require(result.get('ok'), result.get('error', '协调失败'))
            require(result['policy'] == os.environ['PILOT_POLICY'], '策略标识不符')
            pending = result['target']['pending']
            write(OUT / 'target_request.json', pending)
            control_rpc_wall_ms = (control_end - control_start) / 1e6
        else:
            pending = json.loads((OUT / 'target_request.json').read_text())
            state = call('b5d4_state')
            write(out / 'cold_initial.json', state)
            require(not any(n['slots'] for n in state['state']['nodes']), 'Cold缓存污染')
            control_rpc_wall_ms = None

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
        base = {'status': 'PASS', 'mode': MODE, 'policy': os.environ.get('PILOT_POLICY'),
                'pending': pending['continuation_id'],
                'H_physical_fa_hit': None, 'E_executable_prefix': None, 'G_replay_gap': None,
                'prefill_token_count': prefill_row['extend_token_count'],
                'batch_prefix_length': prefill_row['batch_prefix_length'],
                'cuda_prefill_ms': cuda_prefill_ms, 'cuda_decode_ms': cuda_decode_ms,
                'cow_copy_count': len(trace['copies']), 'cow_copy_cuda_ms': cow_cuda_ms,
                'ttft_ms': (first - start) / 1e6, 'request_wall_ms': (end - start) / 1e6,
                'stream_tpot_ms': (arrivals[-1]['ns'] - first) / 7e6,
                'control_rpc_wall_ms': control_rpc_wall_ms,
                'warmup_wall_ms': warmup_wall_ms,
                'epoch_wall_ms': (time.monotonic_ns() - epoch_start) / 1e6,
                'output_ids': output['output_ids'], 'native_matches': all_matches,
                '说明': '受控审计化执行；CUDA事件是设备时间线区间而非纯内核之和；控制开销与请求性能分开计时'}
        if MODE == 'cold':
            actual = all_matches[-1]
            require(actual['physical_fa_hit'] == actual['executable_prefix'] == 0, 'Cold命中旧状态')
            require(prefill_row['batch_prefix_length'] == 0 and prefill_row['extend_token_count'] == len(pending['token_ids']), 'Cold Prefill区间不符')
            base.update(H_physical_fa_hit=actual['physical_fa_hit'], E_executable_prefix=actual['executable_prefix'],
                        G_replay_gap=actual['physical_fa_hit'] - actual['executable_prefix'])
        else:
            target = result['target']
            expected_E = target['expected']['E']
            actual = all_matches[-1]
            require(actual['input_token_digest'] == pending['token_prefix_sha256'], 'Pending输入不符')
            require(actual['physical_fa_hit'] == pending['attention_frontier'], 'FA命中与冻结边界不一致')
            require(actual['executable_prefix'] == expected_E, '实际可执行前缀与计划E不符')
            require(prefill_row['batch_prefix_length'] == expected_E and
                    prefill_row['extend_token_count'] == len(pending['token_ids']) - expected_E, 'Prefill区间与E/G合同不符')
            base.update(H_physical_fa_hit=actual['physical_fa_hit'], E_executable_prefix=actual['executable_prefix'],
                        G_replay_gap=actual['physical_fa_hit'] - actual['executable_prefix'])
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
            else:
                base['cow_absent_by_contract'] = True
            require(post['state']['selected_payload'] == result['selected_payload_after'] and
                    post['state']['selected_components'] == result['admission_baseline']['selected_components'], '历史Recurrent状态受损')
            require(post['state']['fa_protected_payload'] == result['fa_protected_payload'], '非重算FA内容变化')
            require(all(post['state']['fa_map'].get(k) == v for k, v in result['fa_map_before'].items()), '历史FA映射丢失或改变')
            base.update(control_timing_stages=result['timing'],
                        census_counts={'physical': result['census']['physical_domain_size'],
                                       'mapped': len(result['census']['mapped']),
                                       'unmapped_extras': len(result['census']['unmapped_extras'])},
                        selected_cpu=result['policy_selected_cpu_ids'], plan_recovery_cost_ms=result['plan']['recovery_cost_ms'])
        write(out / 'summary.json', base)
        print('完成', MODE, os.environ.get('PILOT_POLICY'), pending['continuation_id'],
              'E', base['E_executable_prefix'], 'H', base['H_physical_fa_hit'], flush=True)
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
