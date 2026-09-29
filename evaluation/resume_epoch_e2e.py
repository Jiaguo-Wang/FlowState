"""在冻结快照上测量包含控制路径的单令牌恢复 epoch。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter_ns
import traceback

from evaluation import rq4_unified_runtime_harness as rq4
from evaluation import rq6_batched_control as batch
from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.scenario import CheckpointRecency
from evaluation.sota_metadata import build_marconi_flop_saved
from evaluation.sota_policies import MarconiStylePolicy
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652'
STAGES = ('common_observation_ms', 'policy_observation_ms', 'input_construction_ms',
          'allocation_ms', 'reconciliation_ms')


def require(value, message):
    """门禁不满足时立即终止当前运行。"""
    if not value:
        raise RuntimeError(message)


def write(path, value):
    """仅向本轮独立目录写出结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def construction(snapshot, states, history, frontiers, policy):
    """使用当前驻留观测与历史元数据构造本轮策略输入。"""
    candidates = tuple(replace(c.to_core(), recurrent_resident=states[c.checkpoint_id]['recurrent_resident'],
                               fa_resident=states[c.checkpoint_id]['fa_resident'])
                       for c in snapshot.eligible_candidates)
    if policy == 'LRU':
        inputs = tuple(CheckpointRecency(c.checkpoint_id, history[c.checkpoint_id].creation_order,
                                       history[c.checkpoint_id].last_access_order) for c in candidates)
    elif policy == 'Marconi':
        inputs = ({c.checkpoint_id: float(history[c.checkpoint_id].last_access_order) for c in candidates},
                  build_marconi_flop_saved(candidates))
    elif policy == 'FlowState':
        inputs = tuple(replace(p.to_core(), resident_fa_frontier=frontiers[p.workflow_id])
                       for p in snapshot.pending_continuations)
    else:
        raise ValueError('未知策略')
    return candidates, inputs


def selection(snapshot, candidates, inputs, policy):
    """调用冻结策略的真实选择算法，禁止使用参考选择代替计算。"""
    if policy == 'LRU':
        return tuple(select_global_lru(candidates, inputs, snapshot.budget_bytes))
    if policy == 'Marconi':
        return tuple(MarconiStylePolicy().select(candidates, snapshot.logical_budget_k,
                     inputs[0], inputs[1], snapshot.marconi_alpha).selected_checkpoint_ids)
    result = GlobalOptimizer(RecoveryCostModel()).select(inputs, candidates, snapshot.budget_bytes)
    return tuple(c.checkpoint_id for c in result.selected)


def stream_request(engine, request, clock=perf_counter_ns):
    """记录同一单调时钟上的提交、首令牌和流结束时间，单位为毫秒。"""
    submit = clock()
    stream = engine.generate(input_ids=list(request['input_ids']), rid=str(request['rid']),
                             sampling_params={'max_new_tokens': 1, 'temperature': 0, 'ignore_eos': True},
                             stream=True)
    first = None
    last = None
    for chunk in stream:
        require(isinstance(chunk, dict), '请求流块格式错误')
        if chunk.get('output_ids') and first is None:
            first = clock()
        last = chunk
    complete = clock()
    require(first is not None and last is not None, '请求未返回令牌')
    meta = last.get('meta_info') or {}
    require(len(last.get('output_ids') or []) == 1 and meta.get('completion_tokens') == 1, '输出不是单令牌')
    require(int(meta.get('num_retractions', 0) or 0) == 0, '发生请求回撤')
    require(meta.get('prompt_tokens') == len(request['input_ids']), '输入发生截断')
    require(submit <= first <= complete, '请求时钟顺序错误')
    return {'rid': request['rid'], 'workflow_id': request['workflow_id'],
            'submit_ts': submit / 1e6, 'first_token_ts': first / 1e6, 'completion_ts': complete / 1e6,
            'ttft_ms': (first-submit)/1e6, 'request_latency_ms': (complete-submit)/1e6,
            'completion_tokens': 1, 'server_metadata': meta}


def epoch_metrics(t0, control_end, requests):
    """从原始时间戳推导指标，不拼接不同运行的测量。"""
    require(len(requests) == 4, '恢复请求数不等于四')
    previous = control_end
    for row in requests:
        require(previous <= row['submit_ts'] <= row['first_token_ts'] <= row['completion_ts'], '串行时间边界错误')
        previous = row['completion_ts']
    end = requests[-1]['completion_ts']
    return {'epoch_start_ts': t0, 'epoch_end_ts': end,
            'control_inclusive_epoch_latency_ms': end-t0,
            'mean_barrier_to_completion_ms': sum(r['completion_ts']-t0 for r in requests)/4,
            'mean_request_execution_ms': sum(r['request_latency_ms'] for r in requests)/4,
            'total_control_latency_ms': control_end-t0,
            'between_request_validation_and_dispatch_ms': end-control_end-sum(r['request_latency_ms'] for r in requests)}


def worker(output, run_id):
    """在独立进程中重建历史并测量一个恢复 epoch。"""
    from transformers import AutoTokenizer
    from targeted_probe import ControlClient
    from rq6_batched_control_transport import RQ6BatchedControlEngine, requested_control_port
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
    from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES
    from evaluation.openhands_4workflow_occupancy_calibration import _runtime_metrics
    from evaluation.openhands_policy_to_actuator_mapping_gate import inspect_candidate_states

    plan = next(p for p in json.loads((output/'plan.json').read_text()) if p['run_id'] == run_id)
    directory = output/'runs'/run_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {**plan, 'status': 'FAIL', 'correctness': {}, 'requests': []}
    engine = None
    try:
        manifest = json.loads((rq4.DEFAULT_FORMAL_ROOT/'population_manifest.json').read_text())
        group = rq4._group_from_manifest(manifest, plan['group_ordinal'])
        frozen = rq4._load_allocation_snapshot(Path(plan['snapshot_artifact']))
        require(frozen.content_digest() == plan['snapshot_digest'], '冻结快照摘要错误')
        snapshot = rq4.create_budget_variant(frozen, plan['logical_k'])
        tokenizer = AutoTokenizer.from_pretrained(rq4.TOKENIZER_PATH, local_files_only=True)
        messages = {label: rq4.load_session_messages(sid, rq4.DATASET_PATH) for label,sid in group.session_by_label.items()}
        requests, audits = rq4.materialize_group_requests(tokenizer, messages, group=group,
                        normalize_message=rq4.normalize_message, template_input_ids=rq4._template_input_ids)
        require(not rq4._future_leakage(audits), '请求含未来信息')
        engine = RQ6BatchedControlEngine(**manifest['engine_configuration'])
        raw = ControlClient(requested_control_port())
        wait_for_transport(raw)
        runtime = rq4.SGLangGroupRuntime(engine, raw)
        initial = runtime.census(run_id+':initial', ordinal=0, request=None, previous=None)
        require(initial['mamba_node_count'] == 0, '初始缓存非空')
        trace = rq4.replay_group_to_barrier(runtime, group, requests)
        trace = replace(trace, boundary_audit=tuple(audits))
        assembly = rq4.assemble_group_snapshot(trace, checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES)
        require(assembly.status == 'ELIGIBLE', '历史重建不符合候选要求')
        validation = rq4.validate_rebuilt_snapshot(frozen, assembly.snapshot)
        handles = rq4.build_current_runtime_handles(trace, requests)
        descriptors = snapshot.eligible_candidates
        ids = tuple(c.checkpoint_id for c in descriptors)
        require(set(handles) == set(ids), '候选句柄集合错误')
        pending = [requests[(label, group.allocation_round+1)] for label in group.session_by_label]
        require(len(pending) == 4, 'pending 数量错误')
        history = {c.checkpoint_id: c for c in trace.checkpoints}
        client = batch.BatchedControlClient(raw)
        policy = plan['policy']
        stages = {}
        frontiers = {}
        observations = []
        t0 = perf_counter_ns()
        view = client.introspect(nonce=run_id+':common', candidates=descriptors, handles=handles)
        states = batch._state_from_view(view['view'], ids)
        require(view['state_mutated'] is False, '公共观测修改状态')
        require(all(s['recurrent_resident'] and s['fa_resident'] for s in states.values()), '初始候选未全部驻留')
        stop = perf_counter_ns()
        stages['common_observation_ms'] = (stop-t0)/1e6
        start = stop
        if policy == 'FlowState':
            for request in pending:
                obs = runtime.inspect_fa_frontier(request['input_ids'], nonce=run_id+':frontier:'+request['workflow_label'])
                require(obs['state_equal'] and obs['scope_before'] == obs['scope_after'], 'frontier 查询改变状态')
                frontiers[request['workflow_id']] = int(obs['resident_fa_frontier'])
                observations.append(dict(obs))
            require(all(frontiers[p.workflow_id] == p.resident_fa_frontier for p in snapshot.pending_continuations), '当前 frontier 不匹配')
        stop = perf_counter_ns()
        stages['policy_observation_ms'] = (stop-start)/1e6
        start = stop
        candidates, inputs = construction(snapshot, states, history, frontiers, policy)
        stop = perf_counter_ns()
        stages['input_construction_ms'] = (stop-start)/1e6
        start = stop
        selected = selection(snapshot, candidates, inputs, policy)
        stop = perf_counter_ns()
        stages['allocation_ms'] = (stop-start)/1e6
        start = stop
        require(set(selected) == set(plan['selected_candidate_ids']), '本轮选择与冻结结果不一致')
        reconciliation = client.reconcile(nonce=run_id+':reconcile', candidates=descriptors,
                handles=handles, selected_ids=selected, expected_view_digest=view['view_digest'])
        proof = reconciliation['proof']
        require(proof['status'] == 'PASS', '协调验证失败')
        after_states = batch._state_from_view(reconciliation['after'], ids)
        require(all(s['recurrent_resident'] == (cid in selected) for cid,s in after_states.items()), '选择集未实现')
        previous = batch.compact_census({'tree': reconciliation['after']['tree'],
                    'accounting': reconciliation['after']['accounting']},
                    ordinal=4*group.allocation_round, request=None, previous=trace.census_rows[-1])
        control_end = perf_counter_ns()
        stages['reconciliation_ms'] = (control_end-start)/1e6
        censuses = []
        for offset, request in enumerate(pending, 1):
            relevant = tuple(c for c in candidates if c.workflow_id == request['workflow_id'])
            pre, _ = inspect_candidate_states(raw, relevant, handles, phase=run_id+':pending:'+str(offset))
            require(all(s['recurrent_resident'] == (cid in selected) and s['fa_resident'] for cid,s in pre.items()), '请求前驻留被污染')
            row = stream_request(engine, request)
            record['requests'].append(row)
            metrics = _runtime_metrics(raw, request['rid'], len(request['input_ids']))
            require(metrics['runtime_metrics_valid'], '运行时 H/E/G 无效')
            row.update(metrics)
            census = runtime.census(run_id+':after:'+str(offset), ordinal=4*group.allocation_round+offset,
                                    request=request, previous=previous)
            require(not census['native_mamba_capacity_eviction_inferred'], '原生循环状态驱逐污染')
            require(not census['fa_kv_cascade_eviction_inferred'], 'attention 状态级联驱逐')
            censuses.append(dict(census))
            previous = census
        record.update(epoch_metrics(t0/1e6, control_end/1e6, record['requests']))
        record.update(stages)
        expected = rq4.evaluate_objective(snapshot, selected)
        expected_heg = {p.continuation_id: p for p in expected.per_continuation}
        for row in record['requests']:
            p = next(p for p in snapshot.pending_continuations if p.workflow_id == row['workflow_id'])
            e = expected_heg[p.continuation_id]
            require((row['h'],row['e'],row['g']) == (e.target_tokens,e.executable_frontier_tokens,e.recovery_gap_tokens), '实际 H/E/G 不匹配选择集')
        require(tuple(rq4.select_frozen_policy(snapshot, policy)) == tuple(plan['selected_candidate_ids']), '冻结选择参考不一致')
        require(frozen.content_digest() == plan['snapshot_digest'], '快照被修改')
        record['selected_candidate_ids_actual'] = selected
        record['correctness'] = {'candidate_population': True, 'selected_set_equivalent': True,
            'selected_residency': True, 'attention_preserved': proof['fa_preserved'],
            'heg_consistent': True, 'unexpected_recurrent_eviction_zero': not proof['native_recurrent_eviction'],
            'unexpected_rematerialization_zero': not proof['unexpected_rematerialization'],
            'native_eviction_contamination_zero': True, 'oom_retraction_truncation_zero': True,
            'fresh_engine': True, 'snapshot_unchanged': True, 'future_leakage_zero': True}
        require(all(record['correctness'].values()), '正确性汇总失败')
        record['status'] = 'PASS_PENDING_CLEANUP'
        write(directory/'evidence.json', {'replay_validation': validation, 'common': view,
            'policy_observation': observations, 'reconciliation': reconciliation, 'censuses': censuses})
    except Exception:
        record['status'] = 'FAIL'
        record['error'] = traceback.format_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                record['correctness']['engine_shutdown'] = True
            except Exception:
                record['status'] = 'FAIL'
                record['shutdown_error'] = traceback.format_exc()
    write(directory/'record.json', record)
    return 0 if record['status'] == 'PASS_PENDING_CLEANUP' else 1


def initialize(output):
    """仅创建新实验目录，复用冻结正式计划而不重新抽样。"""
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((FORMAL/'manifest.json').read_text())
    plan = manifest['runs']
    require(len(plan) == 216 and len({p['snapshot_id'] for p in plan}) == 24, '冻结计划规模错误')
    write(output/'plan.json', plan)
    write(output/'protocol.json', {'说明': '包含控制路径的四个串行单令牌恢复请求；历史维护不属于本 epoch。',
        '时钟': 'perf_counter_ns 转为毫秒；仅在同一 worker 内比较时间戳。',
        '计时边界': '历史重建后公共观测开始至第四请求流结束。第四请求后审计不计入。',
        '观测': '公共批量视图由三种策略承担；FlowState 在计时区间内重新查询四个 frontier。',
        '日志': 'worker标准输出由父进程在内存中收集，worker退出后写盘；关键路径不写artifact。',
        '同步验证': '协调验证、每请求前驻留检查及前三请求后的 H/E/G 查询和 census 均计入。',
        '初始化': '历史记录维护及参考快照验证在计时区间外，不代替区间内观测和选择。',
        '统计': '每快照策略先取三次重复均值，再以24快照配对；round分层bootstrap10000次，seed20260928。',
        '平局': '配对差绝对值不超过1e-6毫秒。正改善定义为基线减FlowState。',
        '规模': {'snapshots':24,'policies':3,'repetitions':3,'lifecycles':216},
        '冻结模型配置': json.loads((rq4.DEFAULT_FORMAL_ROOT/'population_manifest.json').read_text())['engine_configuration']})


def run(output, limit):
    """顺序运行冻结计划，已完成记录只读复用，任何失败立即停止。"""
    plan = json.loads((output/'plan.json').read_text())
    if limit:
        plan = plan[:limit]
    (output/'logs').mkdir(exist_ok=True)
    initial = rq4.wait_gpu_stable(gpu_index=0)
    write(output/'initial_gpu.json', initial)
    for index,p in enumerate(plan,1):
        path = output/'runs'/p['run_id']/'record.json'
        if path.exists():
            require(json.loads(path.read_text())['status'] == 'PASS', '已有失败记录，禁止自动重试')
            continue
        process = subprocess.Popen([sys.executable, '-m', 'evaluation.resume_epoch_e2e', '--worker',
                    '--output', str(output), '--run-id',p['run_id']], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, start_new_session=True)
        try:
            captured, _ = process.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            captured, _ = process.communicate()
        code = process.returncode
        (output/'logs'/(p['run_id']+'.log')).write_bytes(captured)
        cleanup = rq4.wait_gpu_stable(gpu_index=0)
        record = json.loads(path.read_text()) if path.exists() else {**p,'status':'FAIL','correctness':{}}
        record['correctness']['gpu_cleanup'] = cleanup['stable']
        record['gpu_cleanup'] = cleanup
        record['worker_exit_code'] = code
        passed = code == 0 and record['status'] == 'PASS_PENDING_CLEANUP' and all(record['correctness'].values())
        record['status'] = 'PASS' if passed else 'FAIL'
        write(path,record)
        write(output/'progress.json',{'completed':index,'planned':216,'latest':p['run_id'],'status':record['status']})
        print(index, p['run_id'], record['status'], flush=True)
        require(passed, '运行失败，停止后续实验：'+p['run_id'])
    return 0


def main():
    """提供初始化、隔离运行和正式采集入口。"""
    parser = argparse.ArgumentParser(description='单令牌恢复 epoch 实验')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    if args.initialize:
        initialize(args.output)
        return 0
    if args.worker:
        return worker(args.output,args.run_id)
    return run(args.output,args.limit)


if __name__ == '__main__':
    raise SystemExit(main())
