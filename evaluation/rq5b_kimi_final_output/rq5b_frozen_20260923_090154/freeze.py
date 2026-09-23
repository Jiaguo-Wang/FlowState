"""只读核验既有 RQ5-B 证据，并在本目录生成独立冻结报告与清单。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/home/wjg/code/FlowState')
OUT = Path(__file__).resolve().parent
PHASES = {
    'phase1': ROOT / 'evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043',
    'phase2': ROOT / 'evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527',
    'phase3': ROOT / 'evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555',
}


def read(path):
    """读取既有 JSON，不改写原始文件。"""
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    """计算文件内容摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    """仅新建当前收尾目录中的输出，拒绝覆盖。"""
    with (OUT / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    """核验现有证据与受保护文件，保持实验、测试及算法完全不动。"""
    if (OUT / 'frozen_manifest.json').exists():
        raise RuntimeError('当前冻结目录已经封存，拒绝覆盖')
    expected = {}
    phase_checks = {}
    for phase, root in PHASES.items():
        manifest = read(root / 'manifest.json')
        entries = ({name: row['SHA256'] for name, row in manifest['文件'].items()}
                   if phase == 'phase1' else manifest)
        for name, digest in entries.items():
            path = root / name
            assert path.is_file() and sha(path) == digest, f'阶段摘要不匹配：{path}'
            expected[path] = digest
        files = [path for path in root.rglob('*') if path.is_file() and 'cache' not in path.relative_to(root).parts]
        for path in files:
            expected.setdefault(path, sha(path))
        phase_checks[phase] = {'status': 'PASS', 'manifest_entries_verified': len(entries),
            'current_noncache_files': len(files), 'artifact_root': str(root)}
    p1, p2, p3 = (PHASES[key] for key in ('phase1', 'phase2', 'phase3'))
    protected = read(p3 / 'correctness/frozen_before.json')
    for name, digest in protected.items():
        path = ROOT / name
        assert sha(path) == digest, f'受保护文件变化：{path}'
        if path in expected:
            assert expected[path] == digest
        expected[path] = digest
    for filename in ('rq5b_kimi_phase1.py.sha256', 'rq5b_kimi_phase1_transport.py.sha256'):
        digest, name = (p1 / filename).read_text().split()
        assert sha(ROOT / name) == digest
        expected[ROOT / name] = digest
    for name, digest in read(p2 / 'correctness/source_manifest.json').items():
        assert sha(ROOT / name) == digest
        expected[ROOT / name] = digest
    for name in ('evaluation/rq5b_kimi_phase3.py', 'evaluation/run_rq5b_kimi_phase3.py',
                 'tests/runtime/rq5b_kimi_phase3_transport.py', 'tests/test_rq5b_kimi_phase3.py'):
        digest = sha(p3 / 'sources' / Path(name).name)
        assert sha(ROOT / name) == digest
        expected[ROOT / name] = digest
    assert read(p1 / 'summary.json')['最终状态'] == 'RQ5B_KIMI_PHASE1_READY'
    gate2 = read(p2 / 'post_validation.json')
    assert gate2['status'] == 'RQ5B_KIMI_PHASE2_READY'
    assert all(all(row['checks'].values()) for row in gate2['runs'].values())
    for phase, root in (('phase1', p1), ('phase2', p2)):
        for run in ('smoke', 'rep01', 'rep02', 'rep03'):
            record = read(root / 'runs' / run / 'record.json')
            assert record['status'] == 'PASS' and all(record['checks'].values())
            cleanup = read(root / (f'gpu_cleanup_{run}.json' if phase == 'phase1' else f'correctness/gpu_cleanup_{run}.json'))
            assert cleanup.get('状态') == 'PASS' if phase == 'phase1' else cleanup['passed']
    phase1_frontiers = []
    for run in ('smoke', 'rep01', 'rep02', 'rep03'):
        for rank in (0, 1):
            after = read(p1 / 'runs' / run / f'eviction_rank{rank}.json')['after']
            actual_e = max(row['position'] for row in after['paths']['c2']['path_mamba_positions'])
            assert actual_e == 14336
            phase1_frontiers.append({'run_id': run, 'rank': rank, 'resident_ancestor_E': actual_e})
    summary = read(p3 / 'summary.json')
    assert summary['status'] == 'RQ5B_KIMI_PHASE3_READY' and len(summary['pairs']) == 5
    assert len(summary['runs']) == 10 and summary['all_correctness_gates_pass']
    for run in summary['runs']:
        directory = p3 / 'runs' / run['run_id']
        record = read(directory / 'record.json')
        assert record['status'] == 'PASS' and all(record['checks'].values())
        assert read(p3 / 'correctness' / f"gpu_cleanup_{run['run_id']}.json")['passed']
        assert all(value == record['expected_H_E_G'] for value in record['actual_H_E_G'].values())
    pairs = summary['pairs']
    differences = [row['B_resume_ms'] - row['A_resume_ms'] for row in pairs]
    assert all(row['status'] == 'PASS' and all(row['checks'].values()) for row in pairs)
    assert all(value > 0 for value in differences)
    assert statistics.mean(differences) == summary['mean_paired_difference_ms']
    assert statistics.median(differences) == summary['median_paired_difference_ms']
    assert '62 passed' in (p2 / 'logs/related_tests.log').read_text()
    assert '46 passed' in (p3 / 'logs/related_tests.log').read_text()
    timestamp = datetime.now(timezone.utc).isoformat()
    integrity = {'status': 'PASS', 'final_status': 'RQ5B_KIMI_FROZEN', 'created_at_utc': timestamp,
        'phase_manifests': phase_checks, 'historical_protected_files_verified': len(protected),
        'unique_existing_files_verified': len(expected), 'changed_existing_files': [],
        'historical_tests': {'phase2': '62/62 PASS', 'phase3': '46 PASS'},
        'historical_lifecycles_verified': 18, 'historical_gpu_cleanup': '18/18 PASS',
        'gpu_experiments_started_this_session': 0, 'phase_reruns_this_session': 0,
        'test_reruns_this_session': 0, 'algorithm_modified': False,
        'phase1_frontier_evidence': phase1_frontiers,
        '说明': '测试和 GPU 清理结论来自已存记录，本轮只核验文件与证据。Phase 1 的两候选逻辑视图不代表完整物理路径的实际回退点；真正 8K 回退由 Phase 3 规范化公共准备和实际请求匹配证明。编译缓存与模型权重不纳入本次清单。'}
    save('integrity_summary.json', integrity)
    links = {key: f'[{path}]({path}/final_report.md)' for key, path in PHASES.items()}
    lines = ['# RQ5-B Kimi 跨架构泛化最终冻结报告', '', '最终状态：RQ5B_KIMI_FROZEN。', '',
        f'冻结时间：{timestamp}。本轮只做已有证据核验、汇总和封存，未启动 GPU、未重跑实验或测试、未改动任何旧 artifact 或 FlowState 算法。', '',
        '模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`；SGLang `0.5.17` / `29481685462732237d80d86076d6563e1f658102`；2 × NVIDIA H100 PCIe（各 81,559 MiB）；TP=2。', '',
        '| 阶段 | 冻结依据状态 | 原始 artifact |', '| --- | --- | --- |',
        f"| Phase 1 | RQ5B_KIMI_PHASE1_READY | {links['phase1']} |",
        f"| Phase 2 | RQ5B_KIMI_PHASE2_READY | {links['phase2']} |",
        f"| Phase 3 | RQ5B_KIMI_PHASE3_READY | {links['phase3']} |", '',
        'Phase 1 证明 KDA tracking、MLA/KDA 驻留观测、KDA-only eviction、MLA 保持及 TP rank 一致性。smoke 和随后 3/3 独立生命周期均通过。', '',
        '**前沿证据口径：**Phase 1 原报告的 `16K/16K/0 → 16K/8K/8K` 来自仅含 c1@8K、c2@16K 的逻辑视图。原始逐 rank 路径同时保留 10K、12K、14K KDA；只删除 16K 后，全祖先驻留证据给出的 E 是 14K、G 是 2K。最终冻结保留原 READY 状态和原文件，但不将该两候选视图表述为实际 8K 回退测量。Phase 1 支持组件隔离结论；真正的 `16K/8K/8K` 由 Phase 3 的实际请求匹配证明。', '',
        'Phase 2 直接复用现有兼容性、H/E/G、恢复成本接口及 greedy allocator。两个 workflow、四个候选 A1/A2/B1/B2，K=2，S*={A2,B2}；两个 rank 均正确保留 A2/B2 并仅删除 A1/B1 的 KDA。smoke 和 3/3 独立生命周期通过。批量观测、只读检查、executable-state 构造、协调实现、H/E/G 预测、MLA 保持及 TP 一致性均 PASS。每个 epoch 一次逻辑批量观测、一次逻辑批量协调，每批各含两个 rank RPC，聚合验证不额外读取；不主张并发请求下的跨 rank 事务容错。', '',
        'Phase 2 恢复成本数值只用于冻结接口功能验证，不是 Kimi 延迟校准。相关测试记录为 62/62 PASS；既有 5,161 文件核验无变化。', '',
        'Phase 3 在 A/B 公共准备中仅删除 10K/12K/14K KDA，保留 MLA 和 8K/16K；A/B 唯一条件差异是 16K KDA 是否驻留。实际请求在两个 rank 上均确认 A=16K/16K/0、B=16K/8K/8K。五组配对采用 AB/BA/AB/BA/AB，每个条件独立 engine，使用相同历史、请求及配置。', '',
        '主指标为两个 TP rank 内部恢复时延的较大者：首次实际前缀匹配至单令牌请求完成，包含相同的 256-token 后缀处理，排除启动、历史物化、观测和驱逐。该指标是实际 resume 路径时间，不是纯回放 kernel 时间。', '',
        '| 配对 | A：G=0（ms） | B：G=8K（ms） | B−A（ms） |', '| --- | ---: | ---: | ---: |']
    for row in pairs:
        lines.append(f"| {row['pair']} | {row['A_resume_ms']:.6f} | {row['B_resume_ms']:.6f} | {row['paired_difference_ms']:+.6f} |")
    lines += ['', f"配对差值均值 **+{statistics.mean(differences):.2f} ms**，中位数 **+{statistics.median(differences):.2f} ms**；正差值 **5/5**。MLA 保持、TP 一致性、正确性门禁及 GPU 清理均为 10/10 PASS。相关测试记录 46 PASS；Phase 3 原有 5,267 文件完整性记录无变化。", '',
        f"本轮重新核验三份既有 manifest：Phase 1 {phase_checks['phase1']['manifest_entries_verified']} 项、Phase 2 {phase_checks['phase2']['manifest_entries_verified']} 项、Phase 3 {phase_checks['phase3']['manifest_entries_verified']} 项，均 PASS；同时核验历史保护清单的 {len(protected)} 个文件及阶段源码对应关系。去重后共核验 {len(expected)} 个既有文件，变化数为 0。既有测试仅核验记录，未重跑，不将重叠测试相加为独立测试总数。", '',
        '冻结入口为 [frozen_manifest.json](frozen_manifest.json)，包含既有证据、既有保护文件、阶段源码及本次报告和完整性汇总的路径、字节数与 SHA-256。清单自身摘要见 frozen_manifest.sha256；其余核验说明见 [integrity_summary.json](integrity_summary.json)。排除各阶段 cache/ 下的编译缓存和模型权重内容；不改动旧文件权限或重写旧清单。', '',
        '结论范围限于该 Kimi 模型、固定 SGLang/TP=2 环境与小规模构造工作负载。未重拟合 Φ、未新增 baseline、未复制 RQ4，且不声称 TTFT 改善比例、严格线性或完整单调性。', '',
        'FlowState 的同一 executable-state allocation 与组件隔离的 batched recurrent-state realization 可在不改变核心语义的情况下迁移到 Kimi Linear 的 MLA+KDA；在固定 MLA 驻留下，真实 8K 恢复间隔在五个独立配对中均产生可测的额外恢复开销。']
    with (OUT / 'final_report.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    for path, digest in expected.items():
        assert sha(path) == digest, f'汇总期间文件发生变化：{path}'
    files = {str(path.relative_to(ROOT)): {'sha256': digest, 'bytes': path.stat().st_size}
             for path, digest in sorted(expected.items())}
    for name in ('freeze.py', 'final_report.md', 'integrity_summary.json'):
        path = OUT / name
        files[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    save('frozen_manifest.json', {'status': 'RQ5B_KIMI_FROZEN', 'created_at_utc': timestamp,
        'path_base': str(ROOT), 'file_count': len(files), 'files': files,
        '说明': '冻结清单覆盖所有列出的文件内容；不包含自身及自身校验文件以避免循环摘要。原阶段产物保持不变。'})
    with (OUT / 'frozen_manifest.sha256').open('x') as handle:
        handle.write(sha(OUT / 'frozen_manifest.json') + '  frozen_manifest.json\n')
    print(json.dumps({'status': 'RQ5B_KIMI_FROZEN', 'output': str(OUT), 'existing_files_verified': len(expected),
                      'manifest_files': len(files), 'phase_checks': phase_checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
