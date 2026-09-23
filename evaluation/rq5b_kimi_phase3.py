"""执行 Kimi 最小恢复对照中的一个独立条件。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter_ns
import traceback

from evaluation.rq5b_kimi_phase1 import ENGINE_ARGS, SAMPLING, tokens, digest, save, call, wait, generate
from evaluation.rq5b_kimi_phase2 import TPBatchClient
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation

H = 16384
FALLBACK = 8192
TAIL = 256


def workload():
    """冻结相同历史、公共 KDA 准备步骤与相同恢复请求。"""
    prefix8 = tokens(1101, FALLBACK)
    prefix16 = prefix8 + tokens(2202, H - FALLBACK)
    prefixes = {f'c{position}': prefix16[:position] for position in range(2048, H + 1, 2048)}
    history = [prefix16 + tokens(3303, TAIL), prefix8 + tokens(4404, TAIL), prefix16 + tokens(5505, TAIL)]
    target = prefix16 + tokens(6606, TAIL)
    return prefixes, history, target


def state_heg(view):
    """把全部实际祖先检查点交给既有前沿函数，禁止忽略中间状态。"""
    path = view['paths'][f'c{H}']
    if not path['path_full_all_present']:
        raise RuntimeError('MLA 历史未完整驻留')
    candidates = tuple(CheckpointCandidate(f"node{row['node_id']}", 'W', ('W',), row['position'], 1)
        for row in path['path_mamba_positions'])
    pending = PendingContinuation('resume', 'W', ('W',), H, path['prefix_tokens'])
    return {'H': pending.planning_target, 'E': executable_frontier(pending, candidates),
            'G': recovery_gap(pending, candidates)}


def selected_for_common():
    """公共准备只清除 8K 与 16K 之间的三个中间状态。"""
    return [f'c{position}' for position in (2048, 4096, 6144, 8192, 16384)]


def main():
    """独立物化、准备条件、测量一次恢复并验证逐 rank 状态。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--pair', type=int, required=True)
    parser.add_argument('--condition', choices=('A', 'B'), required=True)
    args = parser.parse_args()
    out = Path(args.output_root) / 'runs' / f'pair{args.pair:02d}_{args.condition}'
    out.mkdir(parents=True, exist_ok=False)
    record = {'pair': args.pair, 'condition': args.condition, 'status': 'FAIL', 'checks': {}}
    engine = None
    prefixes, history, target = workload()
    save(out / 'engine_args.json', ENGINE_ARGS)
    save(out / 'workload.json', {'history_token_digests': [digest(values) for values in history],
        'history_lengths': [len(values) for values in history], 'target_digest': digest(target),
        'target_length': len(target), 'sampling': SAMPLING,
        'common_remove_positions': [10240, 12288, 14336],
        '说明': '公共准备补齐固定条件所需的 KDA 驻留；A/B 的唯一条件差异是 16K KDA 驻留。'})
    try:
        from targeted_probe import ControlClient
        from rq5b_kimi_phase3_transport import KimiPhase3Engine
        engine = KimiPhase3Engine(**ENGINE_ARGS)
        clients = {rank: ControlClient(int(os.environ.get('FLOWSTATE_RQ5B_PORT', '49981')) + rank, timeout_s=300) for rank in (0, 1)}
        for client in clients.values():
            wait(client)
        gates = {rank: call(client, 'runtime_gate') for rank, client in clients.items()}
        save(out / 'runtime_gate.json', gates)
        assert all(g['initial']['tree']['node_count'] == 1 and g['initial']['tree']['mamba_node_count'] == 0 for g in gates.values())
        requests = [generate(engine, f'history{index}', values) for index, values in enumerate(history)]
        save(out / 'history_requests.json', requests)
        batch = TPBatchClient(clients)
        rows = [{'checkpoint_id': key, 'token_ids': list(values), 'expected_prefix_digest': digest(values)} for key, values in prefixes.items()]

        def observe():
            """在安全时点一次读取全部祖先状态及只读证据。"""
            return batch.submit('flowstate_batch_introspection', {rank: {'candidate_ids': list(prefixes), 'handles': rows} for rank in clients})

        def reconcile(observed, selected):
            """用既有 KDA-only 原语实现指定的准备状态。"""
            return batch.submit('flowstate_batch_reconciliation', {rank: {
                'candidate_ids': list(prefixes),
                'handles': [{**row, 'expected_node_id': observed[rank]['view']['paths'][row['checkpoint_id']]['node_id']} for row in rows],
                'selected_ids': selected, 'expected_view_digest': observed[rank]['view_digest'],
                'expected_replacement': observed[rank]['read_only_after']['replacement']} for rank in clients})

        initial = observe()
        save(out / 'initial_state.json', initial)
        assert all(not v['eviction_events'] for v in initial.values())
        common = reconcile(initial, selected_for_common())
        save(out / 'common_preparation.json', common)
        prepared = observe()
        save(out / 'common_state.json', prepared)
        assert all(state_heg(row['view']) == {'H': H, 'E': H, 'G': 0} for row in prepared.values())
        condition_change = None
        if args.condition == 'B':
            condition_change = reconcile(prepared, [key for key in selected_for_common() if key != f'c{H}'])
            save(out / 'condition_intervention.json', condition_change)
        before = observe()
        save(out / 'before_resume.json', before)
        expected = {'H': H, 'E': H if args.condition == 'A' else FALLBACK, 'G': 0 if args.condition == 'A' else H - FALLBACK}
        assert all(state_heg(item['view']) == expected for item in before.values())
        assert all(initial[rank]['view']['tree']['full_tree_sha256'] == before[rank]['view']['tree']['full_tree_sha256']
            and initial[rank]['view']['accounting']['full_allocator'] == before[rank]['view']['accounting']['full_allocator'] for rank in clients)
        for client in clients.values():
            call(client, 'arm_resume', request_id='measured_resume')
        started = perf_counter_ns()
        response = generate(engine, 'measured_resume', target)
        client_ms = (perf_counter_ns() - started) / 1e6
        metrics = {rank: call(client, 'resume_metrics') for rank, client in clients.items()}
        save(out / 'resume_metrics.json', metrics)
        save(out / 'resume_response.json', response)
        after = observe()
        save(out / 'after_resume.json', after)
        actual = {rank: {key: row['matches'][0][key] for key in ('H', 'E', 'G')} for rank, row in metrics.items()}
        expected_extend = H + TAIL - expected['E']
        checks = {
            'heg_before': all(state_heg(item['view']) == expected for item in before.values()),
            'heg_actual_match': all(value == expected for value in actual.values()),
            'tp_consistency': actual[0] == actual[1] and metrics[0]['output_ids'] == metrics[1]['output_ids'],
            'mla_preserved_by_intervention': all(initial[rank]['view']['tree']['full_tree_sha256'] == before[rank]['view']['tree']['full_tree_sha256'] for rank in clients),
            'mla_history_preserved_after_resume': all(before[rank]['view']['paths'][f'c{H}']['path_full_sha256'] == after[rank]['view']['paths'][f'c{H}']['path_full_sha256'] and after[rank]['view']['paths'][f'c{H}']['path_full_all_present'] for rank in clients),
            'actual_prefill_work': all(sum(b['extend_tokens'] for b in row['batches']) == expected_extend for row in metrics.values()),
            'no_native_eviction': all(all(event['authorized'] for event in row['eviction_events']) for row in metrics.values()),
            'only_expected_evictions': all(len(row['eviction_events']) == (3 if args.condition == 'A' else 4) for row in metrics.values()),
            'positive_finite_latency': all(0 < row['resume_latency_ms'] < 600000 and 0 < row['forward_cuda_ms'] < 600000 for row in metrics.values()),
            'no_retraction': response['metadata'].get('num_retractions', 0) == 0,
            'one_output_token': len(response['output_ids']) == 1,
            'fresh_lifecycle': all(g['initial']['tree']['node_count'] == 1 and g['initial']['tree']['mamba_node_count'] == 0 for g in gates.values()),
            'read_only_observation': all(row['read_only'] for collection in (initial, prepared, before, after) for row in collection.values()),
        }
        record.update({'checks': checks, 'expected_H_E_G': expected, 'actual_H_E_G': actual,
            'resume_latency_ms': max(row['resume_latency_ms'] for row in metrics.values()),
            'forward_cuda_ms': max(row['forward_cuda_ms'] for row in metrics.values()),
            'per_rank_resume_ms': {rank: row['resume_latency_ms'] for rank, row in metrics.items()},
            'client_request_ms': client_ms, 'server_e2e_ms': response['metadata'].get('e2e_latency', 0) * 1000,
            'actual_extend_tokens': {rank: sum(b['extend_tokens'] for b in row['batches']) for rank, row in metrics.items()},
            'mla_before_digest_per_rank': {rank: row['view']['tree']['full_tree_sha256'] for rank, row in before.items()},
            'common_recurrent_digest_per_rank': {rank: row['view']['tree']['mamba_tree_sha256'] for rank, row in prepared.items()},
            'output_ids': response['output_ids'],
            '说明': '主指标取两个 TP rank 调度器内部恢复时延的最大值；不将客户端时延或纯 CUDA 时间冒充主指标。'})
        assert all(checks.values()), checks
        record['status'] = 'PASS'
    except Exception as error:
        record.update({'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                record['checks']['engine_shutdown'] = True
            except Exception as error:
                record.update({'status': 'FAIL', 'shutdown_error': repr(error)})
        save(out / 'record.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
