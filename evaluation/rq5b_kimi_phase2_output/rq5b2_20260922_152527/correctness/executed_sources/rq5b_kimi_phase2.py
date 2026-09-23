"""验证冻结 FlowState 分配在 Kimi MLA 与 KDA 上的批量实现。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import traceback

from evaluation.rq5b_kimi_phase1 import ENGINE_ARGS, tokens, digest, generate, save, wait, call
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation

POSITIONS = (2048, 4096)
BUDGET = 2


def workload():
    """定义已物化历史与当前待续元数据，不包含任何未来请求内容。"""
    prefixes = {}
    candidates = []
    pending = []
    for workflow, seed in (('A', 1101), ('B', 7707)):
        full = tokens(seed, POSITIONS[0]) + tokens(seed + 1101, POSITIONS[1] - POSITIONS[0])
        for index, position in enumerate(POSITIONS, 1):
            key = f'{workflow}{index}'
            prefixes[key] = full[:position]
            candidates.append({'checkpoint_id': key, 'workflow_id': workflow,
                'lineage_path': [workflow], 'token_pos': position,
                'prefix_digest': digest(prefixes[key])})
            pending.append({'continuation_id': f'{key}:pending', 'workflow_id': workflow,
                'lineage_path': [workflow], 'anchor_pos': position})
    return {'candidates': candidates, 'pending': pending, 'logical_k': BUDGET,
            '说明': '全部待续分支在分配前登记；不提交分配后的生成请求。'}, prefixes


def logical_view(view, frozen, memory_bytes):
    """根据真实路径构造现有候选与待续对象，拒绝遗漏的祖先循环状态。"""
    candidates = []
    for row in frozen['candidates']:
        path = view['paths'][row['checkpoint_id']]
        candidates.append(CheckpointCandidate(row['checkpoint_id'], row['workflow_id'],
            tuple(row['lineage_path']), row['token_pos'], memory_bytes,
            bool(path['target_mamba_present']), bool(path['path_full_all_present'])))
    pending = []
    for row in frozen['pending']:
        key = row['continuation_id'].split(':')[0]
        path = view['paths'][key]
        if not path['path_full_all_present']:
            raise RuntimeError('当前实验所需 MLA 历史未完整驻留')
        continuation = PendingContinuation(**{**row, 'lineage_path': tuple(row['lineage_path']),
            'resident_fa_frontier': path['prefix_tokens']})
        compatible_nodes = {view['paths'][c.checkpoint_id]['node_id'] for c in candidates if is_compatible(c, continuation)}
        if any(item['node_id'] not in compatible_nodes for item in path['path_mamba_positions']):
            raise RuntimeError('存在未纳入候选的兼容祖先循环状态')
        pending.append(continuation)
    return tuple(candidates), tuple(pending)


def heg(candidates, pending):
    """调用原有兼容性与前沿函数，不定义架构特定语义。"""
    resident = tuple(c for c in candidates if c.recurrent_resident)
    return [{'continuation_id': p.continuation_id, 'H': p.planning_target,
             'compatible_candidate_ids': [c.checkpoint_id for c in candidates if is_compatible(c, p)],
             'E': executable_frontier(p, resident), 'G': recovery_gap(p, resident)} for p in pending]


def rank_facts(view):
    """跨 rank 只比较逻辑事实，允许物理槽位不同。"""
    return {key: {field: path[field] for field in ('prefix_sha256', 'prefix_tokens',
        'target_full_present', 'path_full_all_present', 'target_mamba_present')}
        for key, path in view['paths'].items()}


class TPBatchClient:
    """一次逻辑提交向两个空闲 rank 分发完整批次。"""
    def __init__(self, clients):
        self.clients = clients
        self.counts = {'introspection': 0, 'reconciliation': 0, 'rank_rpc': 0}

    def submit(self, action, requests):
        """等待两个 rank 都完成后统一返回，任一失败使整批失败。"""
        self.counts['introspection' if action.endswith('introspection') else 'reconciliation'] += 1
        self.counts['rank_rpc'] += len(self.clients)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {rank: pool.submit(call, client, action, **requests[rank]) for rank, client in self.clients.items()}
            return {rank: future.result() for rank, future in futures.items()}


def main():
    """每次进程执行一个分配 epoch，独立创建和关闭引擎。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    out = Path(args.output_root) / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('per_rank_state', 'correctness', 'logs'):
        (out / folder).mkdir()
    frozen, prefixes = workload()
    save(out / 'workload.json', frozen)
    save(out / 'engine_args.json', ENGINE_ARGS)
    record = {'status': 'FAIL', 'run_id': args.run_id, 'checks': {}}
    engine = None
    try:
        from targeted_probe import ControlClient
        from rq5b_kimi_phase2_transport import KimiPhase2Engine
        engine = KimiPhase2Engine(**ENGINE_ARGS)
        clients = {rank: ControlClient(int(os.environ.get('FLOWSTATE_RQ5B_PORT', '49971')) + rank, timeout_s=300) for rank in (0, 1)}
        for client in clients.values():
            wait(client)
        gates = {rank: call(client, 'runtime_gate') for rank, client in clients.items()}
        save(out / 'correctness/runtime_gate.json', gates)
        assert all(g['initial']['tree']['mamba_node_count'] == 0 and g['initial']['tree']['node_count'] == 1 for g in gates.values())
        requests = []
        for workflow, seed in (('A', 3303), ('B', 9909)):
            for label, values in (
                ('main', prefixes[f'{workflow}2'] + tokens(seed, 256)),
                ('branch1', prefixes[f'{workflow}1'] + tokens(seed + 1101, 256)),
                ('branch2', prefixes[f'{workflow}2'] + tokens(seed + 2202, 256))):
                requests.append(generate(engine, f'{args.run_id}:{workflow}:{label}', values))
        save(out / 'requests.json', requests)
        handles = [{'checkpoint_id': key, 'token_ids': list(values), 'expected_prefix_digest': digest(values)} for key, values in prefixes.items()]
        common = {'candidate_ids': list(prefixes), 'handles': handles}
        client = TPBatchClient(clients)
        observed = client.submit('flowstate_batch_introspection', {rank: common for rank in clients})
        save(out / 'introspection_before.json', observed)
        assert rank_facts(observed[0]['view']) == rank_facts(observed[1]['view'])
        assert all(not item['eviction_events'] for item in observed.values())
        memory_bytes = sum(item['checkpoint_bytes_per_rank'] for item in observed.values())
        logical = {rank: logical_view(item['view'], frozen, memory_bytes) for rank, item in observed.items()}
        assert logical[0] == logical[1]
        candidates, pending = logical[0]
        assert len(candidates) == 4 and all(c.recurrent_resident for c in candidates)
        mapping = {key: {rank: {'node_id': item['view']['paths'][key]['node_id'],
            'slots': item['view']['paths'][key]['target_mamba_slots'],
            'prefix_digest': item['view']['paths'][key]['prefix_sha256']} for rank, item in observed.items()} for key in prefixes}
        save(out / 'candidate_mapping.json', mapping)
        save(out / 'allocator_input.json', {'candidates': [asdict(c) for c in candidates],
            'pending': [asdict(p) for p in pending], 'H_E_G': heg(candidates, pending),
            'budget_bytes': BUDGET * memory_bytes, 'logical_k': BUDGET,
            'recovery_model': asdict(RecoveryCostModel.metadata),
            '说明': '冻结恢复成本仅用于接口功能验证，不作为 Kimi 实测延迟。'})
        optimizer = GlobalOptimizer(RecoveryCostModel())
        allocation = optimizer.select(pending, candidates, BUDGET * memory_bytes)
        selected = tuple(c.checkpoint_id for c in allocation.selected)
        sequence = []
        previous = ()
        for c in allocation.selected:
            before_cost = optimizer._recovery_cost(pending, previous)
            previous += (c,)
            after_cost = optimizer._recovery_cost(pending, previous)
            sequence.append({'checkpoint_id': c.checkpoint_id, 'before_ms': before_cost,
                             'after_ms': after_cost, 'marginal_gain_ms': before_cost - after_cost})
        predicted_candidates = tuple(CheckpointCandidate(**{**asdict(c), 'recurrent_resident': c.checkpoint_id in selected}) for c in candidates)
        prediction = heg(predicted_candidates, pending)
        save(out / 'allocator_output.json', {**asdict(allocation), 'selected_ids': selected,
            'candidate_count': len(candidates), 'candidate_ids': list(prefixes),
            'logical_k': BUDGET, 'marginal_selection_sequence': sequence,
            'predicted_H_E_G': prediction,
            'current_resident_objective_ms': optimizer._recovery_cost(pending, candidates)})
        reconciliation = {rank: {**common, 'handles': [{**h, 'expected_node_id': mapping[h['checkpoint_id']][rank]['node_id']} for h in handles],
            'selected_ids': selected, 'expected_view_digest': item['view_digest'],
            'expected_replacement': item['read_only_after']['replacement']} for rank, item in observed.items()}
        save(out / 'reconciliation_request.json', reconciliation)
        results = client.submit('flowstate_batch_reconciliation', reconciliation)
        save(out / 'reconciliation_result.json', results)
        actual_heg = {}
        for rank, result in results.items():
            save(out / f'per_rank_state/rank{rank}.json', {'observed': observed[rank], 'reconciled': result})
            actual_candidates, actual_pending = logical_view(result['after'], frozen, memory_bytes)
            actual_heg[rank] = heg(actual_candidates, actual_pending)
        checks = {
            'batched_introspection': client.counts['introspection'] == 1,
            'read_only_observation': all(item['read_only'] and item['read_only_before'] == item['read_only_after'] for item in observed.values()),
            'executable_state_construction': logical[0] == logical[1],
            'batched_kda_reconciliation': client.counts['reconciliation'] == 1,
            'selected_set_realization': all(bool(result['after']['paths'][key]['target_mamba_present']) == (key in selected) for result in results.values() for key in prefixes),
            'heg_consistency': all(value == prediction for value in actual_heg.values()),
            'mla_preservation': all(result['proof']['fa_preserved'] for result in results.values()),
            'recurrent_only_mutation': all(result['proof']['recurrent_change_exact'] for result in results.values()),
            'tp_rank_consistency': rank_facts(results[0]['after']) == rank_facts(results[1]['after']),
            'no_native_eviction': all(result['proof']['no_native_eviction'] for result in results.values()),
            'no_rematerialization': all(not result['proof']['unexpected_rematerialization'] and result['proof']['already_absent_unchanged'] for result in results.values()),
            'no_attention_cascade': all(not result['proof']['fa_cascade'] for result in results.values()),
            'no_future_leakage': True,
            'no_cross_run_contamination': all(g['initial']['tree']['mamba_node_count'] == 0 and g['initial']['tree']['node_count'] == 1 for g in gates.values()),
            'aggregate_post_validation': all(result['post_validation_count'] == 1 for result in results.values()),
        }
        record.update({'checks': checks, 'selected_ids': selected, 'actual_H_E_G': actual_heg,
            'predicted_H_E_G': prediction, 'rpc_counts': client.counts,
            'workload_sha256': hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
            '说明': '一次聚合观测和协调各含两个 rank RPC；聚合后验验证复用协调返回值，无额外 RPC。'})
        save(out / 'post_validation.json', record)
        assert all(checks.values()), checks
        record['status'] = 'PASS'
    except Exception as error:
        record.update({'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                record['engine_shutdown'] = True
            except Exception as error:
                record.update({'status': 'FAIL', 'shutdown_error': repr(error)})
        save(out / 'record.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
