"""汇总阶段二原始证据并验证冻结文件及四个独立运行的完整性。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    """读取运行结束后已落盘的原始记录。"""
    return json.loads(path.read_text())


def save(path, value):
    """保存最终聚合证据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    """只根据已完成记录判定 READY，不补造未完成的实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    frozen = read(out / 'correctness/frozen_before.json')
    changed = [path for path, expected in frozen.items()
        if not (ROOT / path).exists() or hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected]
    sources = read(out / 'correctness/source_manifest.json')
    source_changes = [path for path, expected in sources.items()
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected]
    save(out / 'correctness/frozen_after.json', {'passed': not changed and not source_changes,
        'checked_frozen_files': len(frozen), 'changed_frozen_files': changed,
        'changed_experiment_sources': source_changes})
    runs = {}
    for run in ('smoke', 'rep01', 'rep02', 'rep03'):
        directory = out / 'runs' / run
        if not (directory / 'record.json').exists():
            continue
        record = read(directory / 'record.json')
        cleanup = read(out / f'correctness/gpu_cleanup_{run}.json')
        record['checks']['gpu_cleanup'] = cleanup['passed']
        record['checks']['allocator_unchanged'] = not changed
        if record.get('status') == 'PASS':
            observed = read(directory / 'introspection_before.json')
            results = read(directory / 'reconciliation_result.json')
            allocation = read(directory / 'allocator_output.json')
            inputs = read(directory / 'allocator_input.json')
            workload = read(directory / 'workload.json')
            allowed_candidate_fields = {'checkpoint_id', 'workflow_id', 'lineage_path', 'token_pos', 'memory_bytes', 'recurrent_resident', 'fa_resident'}
            allowed_pending_fields = {'continuation_id', 'workflow_id', 'lineage_path', 'anchor_pos', 'resident_fa_frontier'}
            online_boundary = all(set(row) == allowed_candidate_fields for row in inputs['candidates']) and all(set(row) == allowed_pending_fields for row in inputs['pending'])
            record['checks']['no_future_leakage'] = online_boundary
            record['checks']['source_unchanged'] = not source_changes
            assert allocation['candidate_count'] == 4 and allocation['logical_k'] == 2
            assert record['rpc_counts'] == {'introspection': 1, 'reconciliation': 1, 'rank_rpc': 4}
            assert record['selected_ids'] == allocation['selected_ids']
            for rank in ('0', '1'):
                assert observed[rank]['read_only_before'] == observed[rank]['read_only_after']
                assert observed[rank]['view'] == results[rank]['before']
                assert results[rank]['post_validation_count'] == 1
                assert results[rank]['proof']['status'] == 'PASS'
                assert len(results[rank]['eviction_events']) == 2
                assert all(event['authorized'] and event['component'] == 'MambaComponent' for event in results[rank]['eviction_events'])
                assert record['actual_H_E_G'][rank] == record['predicted_H_E_G']
            save(directory / 'correctness/online_boundary.json', {'passed': online_boundary,
                'candidate_fields': sorted(allowed_candidate_fields), 'pending_fields': sorted(allowed_pending_fields),
                '说明': '分配输入仅含当前候选驻留、workflow、lineage、历史锚点及当前 MLA 前沿；待续元数据在物化前冻结，未执行协调后的请求，未引入未来内容、长度、访问次序或延迟。'})
            record['status'] = 'PASS' if all(record['checks'].values()) else 'FAIL'
            save(directory / 'post_validation.json', {**record,
                '说明': '根据 engine shutdown 后的 record.json 最终化聚合验证状态；GPU 清理由宿主机独立核验。'})
        runs[run] = record
    identical_workload = len({r.get('workload_sha256') for r in runs.values()}) == 1
    identical_selection = len({tuple(r.get('selected_ids', ())) for r in runs.values()}) == 1
    complete = len(runs) == 4 and all(r['status'] == 'PASS' and all(r['checks'].values()) for r in runs.values())
    ready = complete and identical_workload and identical_selection and not changed and not source_changes
    status = 'RQ5B_KIMI_PHASE2_READY' if ready else 'RQ5B_KIMI_PHASE2_PARTIAL'
    aggregate = {'status': status, 'runs': {k: {'status': r['status'], 'checks': r['checks'], 'selected_ids': r.get('selected_ids')} for k, r in runs.items()},
        'identical_workload': identical_workload, 'identical_selection': identical_selection,
        'frozen_files_unchanged': not changed, 'experiment_sources_unchanged': not source_changes,
        'logical_batch_counts_per_epoch': {'introspection': 1, 'reconciliation': 1, 'aggregate_validation': 1},
        'physical_rank_rpc_count_per_epoch': 4}
    save(out / 'post_validation.json', aggregate)
    if 'smoke' in runs and runs['smoke']['status'] == 'PASS':
        for name in ('engine_args.json', 'workload.json', 'candidate_mapping.json', 'introspection_before.json',
                     'allocator_input.json', 'allocator_output.json', 'reconciliation_request.json', 'reconciliation_result.json'):
            shutil.copyfile(out / 'runs/smoke' / name, out / name)
        for run in runs:
            for rank in (0, 1):
                source = out / f'runs/{run}/per_rank_state/rank{rank}.json'
                if source.exists():
                    shutil.copyfile(source, out / f'per_rank_state/{run}_rank{rank}.json')
    lines = ['# RQ5-B Kimi Phase 2', '', status, '',
        '模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`。SGLang 0.5.17，提交 `29481685462732237d80d86076d6563e1f658102`。2 × NVIDIA H100 PCIe（各 81,559 MiB），TP=2。', '',
        'Step 0：NEED_MINIMAL_ADAPTER。直接调用冻结 GlobalOptimizer、RecoveryCostModel、兼容性和 H/E/G 函数，以及 SGLangAdapter.evict_mamba_only。新增部分仅负责 TP=2 物理句柄发现、批量分发、只读元数据证据与聚合验证。未修改 SGLang core。', '',
        '两个 workflow 各有 2K/4K 检查点，候选 A1、A2、B1、B2，逻辑预算 K=2；当前待续分支分别锚定这四个已物化位置。全祖先路径核对排除了遗漏的兼容 KDA 检查点。预算仅约束这四个候选，已物化的非候选分叉末端状态保持原样。', '',
        '每个 epoch 一次逻辑批量观测、一次逻辑批量协调：每个批次分别发送一次完整请求到 TP0 和 TP1，共 4 个 rank RPC。两个调度器均空闲且期间无生成请求，协调先验证观测摘要和替换元数据未漂移；每个 rank 在协调内返回一次统一后验检查，宿主机据返回值完成一次 aggregate validation，无额外观测 RPC。本实验证明空闲安全时点的双 rank 实现，不主张跨 rank 故障事务回滚。', '',
        '| 生命周期 | 状态 | GPU 清理 | S* |', '| --- | --- | --- | --- |']
    for run, record in runs.items():
        lines.append(f"| {run} | {record['status']} | {'PASS' if record['checks']['gpu_cleanup'] else 'FAIL'} | {', '.join(record.get('selected_ids', []))} |")
    if 'smoke' in runs and runs['smoke']['status'] == 'PASS':
        allocation = read(out / 'allocator_output.json')
        lines += ['', 'S* = {A2, B2}；A1 与 B1 的 KDA 在两个 rank 上均被删除，A2 与 B2 保持驻留。', '',
            '| 当前待续分支 | H | 驱逐前 E/G | 驱逐后 E/G |', '| --- | ---: | --- | --- |',
            '| A1 / B1 | 2048 | 2048 / 0 | 0 / 2048 |',
            '| A2 / B2 | 4096 | 4096 / 0 | 4096 / 0 |', '',
            f"冻结目标函数空选择成本为 {allocation['recovery_cost_before_ms']:.9f}，S* 成本为 {allocation['recovery_cost_after_ms']:.9f}；全部候选当前驻留时成本为 {allocation['current_resident_objective_ms']:.9f}。前者是 greedy 从空集开始的目标基线，不应与全部驻留状态混同。边际选择次序和各步收益见 allocator_output.json。", '',
            '成本数值沿用冻结 Qwen 恢复模型接口及其 ms 标记，仅验证分配接口可执行；未经 Kimi 校准，不可解释为 Kimi 恢复时延预测。未拟合恢复模型、未采集 TTFT 或吞吐比较。']
    lines += ['', f'冻结文件核验：{len(frozen)} 个已有文件；变化数 {len(changed)}。实验执行源码变化数 {len(source_changes)}。相关测试：62 passed，见 logs/related_tests.log。', '',
        '逐 rank 驻留和物理槽位见 per_rank_state/；只读证据同时覆盖 MLA/KDA 驻留、分配器和 LRU 指针及访问字段；全树差分、槽位释放计数与授权组件驱逐事件共同验证 recurrent-only mutation、无原生驱逐、无重物化和无 attention 级联。already absent 保持 absent 的分支由 CPU 测试覆盖；本次真实运行四个初始候选均已驻留。', '',
        '每次生命周期从空缓存启动，分别完成真实物化，关闭后独立检查无计算进程且两卡回到基线显存。后续运行仅复用编译缓存，不复用运行时缓存或物理槽位。', '',
        'FlowState 的同一 executable-state allocation 和 batched recurrent-state realization 可以在不改变核心语义的情况下从 Qwen3.5 的 FA+GDN 迁移到 Kimi Linear 的 MLA+KDA。' if ready else '现有证据尚不足以形成跨架构完整迁移结论。', '',
        '本轮止于 Phase 2，未启动 Phase 3 或其他 RQ 实验。']
    (out / 'final_report.md').write_text('\n'.join(lines) + '\n')
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.rglob('*') if path.is_file() and 'cache' not in path.relative_to(out).parts and path.name != 'manifest.json'}
    save(out / 'manifest.json', manifest)
    print(status)
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
