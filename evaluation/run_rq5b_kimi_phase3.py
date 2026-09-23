"""执行五个独立配对并汇总 Kimi 最小恢复 sanity 的证据。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from evaluation.run_rq5b_kimi_phase2 import ROOT, IMAGE, MODEL, gpu, command, save


def read(path):
    """读取已经完成的条件原始记录。"""
    return json.loads(path.read_text())


def freeze_manifest():
    """保护之前冻结核验涵盖的文件以及 Phase 1/2 新增证据。"""
    prior = ROOT / 'evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527'
    paths = {ROOT / p for p in read(prior / 'correctness/frozen_before.json')}
    paths.update(p for p in prior.rglob('*') if p.is_file() and 'cache' not in p.relative_to(prior).parts)
    paths.update(ROOT / p for p in read(prior / 'correctness/source_manifest.json'))
    paths.add(ROOT / 'evaluation/finalize_rq5b_kimi_phase2.py')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def paired_record(out, pair):
    """只有完全相同输入与初始 MLA 的有效 A/B 才构成一个配对。"""
    a_dir, b_dir = (out / 'runs' / f'pair{pair:02d}_{condition}' for condition in ('A', 'B'))
    a, b = read(a_dir / 'record.json'), read(b_dir / 'record.json')
    checks = {
        'both_conditions_pass': a['status'] == b['status'] == 'PASS',
        'identical_workload': read(a_dir / 'workload.json') == read(b_dir / 'workload.json'),
        'identical_engine_args': read(a_dir / 'engine_args.json') == read(b_dir / 'engine_args.json'),
        'identical_mla': a['mla_before_digest_per_rank'] == b['mla_before_digest_per_rank'],
        'identical_common_kda': a['common_recurrent_digest_per_rank'] == b['common_recurrent_digest_per_rank'],
        'same_output': a['output_ids'] == b['output_ids'],
        'heg_conditions': a['expected_H_E_G'] == {'H': 16384, 'E': 16384, 'G': 0}
            and b['expected_H_E_G'] == {'H': 16384, 'E': 8192, 'G': 8192},
    }
    return {'pair': pair, 'order': ['A', 'B'] if pair % 2 else ['B', 'A'],
        'A_resume_ms': a['resume_latency_ms'], 'B_resume_ms': b['resume_latency_ms'],
        'paired_difference_ms': b['resume_latency_ms'] - a['resume_latency_ms'],
        'A_forward_cuda_ms': a['forward_cuda_ms'], 'B_forward_cuda_ms': b['forward_cuda_ms'],
        'paired_cuda_difference_ms': b['forward_cuda_ms'] - a['forward_cuda_ms'],
        'checks': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL'}


def finalize(out, frozen, runs, pairs, error):
    """完整记录未完成状态，禁止凭不完整测量推断 READY。"""
    changed = [p for p, h in frozen.items() if not (ROOT / p).exists() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    save(out / 'correctness/frozen_after.json', {'passed': not changed, 'checked_files': len(frozen), 'changed_files': changed})
    differences = [p['paired_difference_ms'] for p in pairs if p['status'] == 'PASS']
    all_pass = len(runs) == 10 and len(pairs) == 5 and all(row['status'] == 'PASS' and row['cleanup_passed'] for row in runs) and all(row['status'] == 'PASS' for row in pairs)
    positives = sum(value > 0 for value in differences)
    ready = all_pass and positives == 5 and not changed and error is None
    status = 'RQ5B_KIMI_PHASE3_READY' if ready else 'RQ5B_KIMI_PHASE3_PARTIAL'
    summary = {'status': status, 'pairs': pairs, 'runs': runs,
        'valid_pairs': len(differences), 'positive_pairs': positives,
        'mean_paired_difference_ms': statistics.mean(differences) if differences else None,
        'median_paired_difference_ms': statistics.median(differences) if differences else None,
        'all_correctness_gates_pass': all_pass, 'frozen_unchanged': not changed, 'error': error}
    save(out / 'summary.json', summary)
    lines = ['# RQ5-B Kimi Phase 3：最小恢复验证', '', status, '',
        '模型为 `/data1/models/Kimi-Linear-48B-A3B-Instruct`，SGLang 0.5.17 / `29481685462732237d80d86076d6563e1f658102`，两张 H100 PCIe，TP=2。', '',
        '主指标为调度器首次实际前缀匹配至单令牌恢复请求完成的内部墙钟时延，取两个 rank 中较大者。包括 KDA 恢复、共同的 256-token 后缀前向和单令牌输出处理；不包括引擎启动、历史物化、控制观测、驱逐和网络传输。该指标是实际 resume 路径时间，不是纯回放 kernel 时间；两条件相减用于估计新增恢复开销。CUDA 累计前向时间作为独立辅助记录。', '',
        '每个条件使用新 engine；每组 A/B 使用相同历史、请求令牌、采样和 runtime 配置，交替执行 AB/BA/AB/BA/AB。每组两条件的 MLA 全树驻留摘要和公共 KDA 状态摘要必须一致。历史物化过程覆盖 2048-token 分块与 256-token 后缀前向，以预热相同形状的执行路径；不将物化时间纳入测量。', '',
        '只读检查发现 Phase 1 原始物理路径还保留 10K/12K/14K KDA，单删 16K 会回退到 14K。Phase 3 在两条件共有的准备步骤中仅清除这三个 KDA，保留 MLA 及 8K/16K；A/B 唯一条件差异仍是 16K KDA 是否驻留。所有祖先状态参与既有 E/G 计算，并用真正的请求匹配结果确认 A=16K/16K/0、B=16K/8K/8K。旧 Phase 1/2 artifact 未修改。', '',
        '| 配对 | 顺序 | A：G=0（ms） | B：G=8K（ms） | B−A（ms） |',
        '| --- | --- | ---: | ---: | ---: |']
    for pair in pairs:
        lines.append(f"| {pair['pair']} | {'→'.join(pair['order'])} | {pair['A_resume_ms']:.6f} | {pair['B_resume_ms']:.6f} | {pair['paired_difference_ms']:+.6f} |")
    if differences:
        lines += ['', f"有效配对 {len(differences)}/5；差值均值 {statistics.mean(differences):.6f} ms；中位数 {statistics.median(differences):.6f} ms；正差值 {positives}/5。"]
    lines += ['', f'全部正确性与清理门禁：{all_pass}；既有 {len(frozen)} 个文件摘要变化数：{len(changed)}。', '',
        '逐 rank 匹配、实际 prefill token 数、CUDA 分块计时、恢复前后 MLA 路径摘要和无原生驱逐证据保存在 runs/；每个条件的 GPU 清理证据保存在 correctness/。恢复请求可以重新物化缺失的 KDA，这是被测恢复行为；条件准备不允许重物化。MLA 保持要求同时检查干预前后全树驻留不变以及恢复后原 16K 历史路径驻留不变。', '',
        '未重拟合 Φ，未增加基线，未运行 RQ4、延迟曲线或完整单调性实验。五组观察只支持此固定历史与 G=8K 的最小 sanity，不主张严格线性。', '',
        '在 Kimi Linear 的 MLA 驻留不变时，KDA 可执行前沿从 16K 回退到 8K 在五个独立配对中均增加了实际恢复开销，支持 FlowState recovery-gap 语义的跨架构适用性。' if ready else '当前证据尚不足以确认固定 G=8K 在五个独立配对中稳定增加恢复开销。']
    if error:
        lines += ['', f'未完成原因：{error}']
    (out / 'final_report.md').write_text('\n'.join(lines) + '\n')
    save(out / 'manifest.json', {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob('*') if p.is_file() and 'cache' not in p.relative_to(out).parts and p.name != 'manifest.json'})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if ready else 1


def main():
    """先执行 CPU 门禁，再顺序执行固定的十个独立条件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('correctness', 'logs', 'sources'):
        (out / name).mkdir()
    frozen = freeze_manifest()
    save(out / 'correctness/frozen_before.json', frozen)
    source_paths = ['evaluation/rq5b_kimi_phase3.py', 'evaluation/run_rq5b_kimi_phase3.py',
                    'tests/runtime/rq5b_kimi_phase3_transport.py', 'tests/test_rq5b_kimi_phase3.py']
    for path in source_paths:
        (out / 'sources' / Path(path).name).write_bytes((ROOT / path).read_bytes())
    save(out / 'plan.json', {'pairs': 5, 'order': ['AB', 'BA', 'AB', 'BA', 'AB'],
        'primary_metric': 'max_rank_scheduler_resume_latency_ms', 'ready_requires_positive_pairs': 5,
        'engine_lifecycles': 10, '说明': '不根据测量结果改变样本数、顺序、指标或判定标准。'})
    tests = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
        '--mount', f'type=bind,src={ROOT},dst={ROOT},readonly', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', f'PYTHONPATH={ROOT}', '--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE,
        '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_rq5b_kimi_phase3.py',
        'tests/test_rq5b_kimi_phase2.py', 'tests/test_executable_state.py', 'tests/test_sglang_adapter.py']
    with (out / 'logs/related_tests.log').open('w') as log:
        tested = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT)
    if tested.returncode:
        return finalize(out, frozen, [], [], 'CPU 测试失败，未启动 GPU')
    initial = gpu()
    if initial['processes'] or any(int(row.split(',')[2]) > 100 for row in initial['gpus'].splitlines()):
        return finalize(out, frozen, [], [], 'GPU 非空闲，未启动实验')
    save(out / 'environment.json', {'gpu_before': initial,
        'image': json.loads(command(['docker', 'image', 'inspect', IMAGE]))[0], 'model': MODEL, 'tp': 2})
    runs, pairs, commands = [], [], []
    error = None
    for pair in range(1, 6):
        for condition in ('AB' if pair % 2 else 'BA'):
            run = f'pair{pair:02d}_{condition}'
            name = f'flowstate-{out.name}-{run}'.replace('_', '-')
            cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name,
                '--gpus', '"device=0,1"', '--network', 'host', '--shm-size', '32g',
                '--mount', f'type=bind,src={ROOT},dst={ROOT},readonly',
                '--mount', f'type=bind,src={MODEL},dst=/model,readonly',
                '--mount', f'type=bind,src={out},dst=/output']
            for key, value in {'HF_HOME': '/output/cache/huggingface', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820',
                'TRITON_CACHE_DIR': '/output/cache/triton', 'TORCHINDUCTOR_CACHE_DIR': '/output/cache/torchinductor',
                'FLOWSTATE_RQ5B_PORT': '49981'}.items():
                cmd.extend(['--env', f'{key}={value}'])
            cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE,
                '-m', 'evaluation.rq5b_kimi_phase3', '--output-root', '/output', '--pair', str(pair), '--condition', condition]
            commands.append(shlex.join(cmd))
            (out / 'launch_command.txt').write_text('\n\n'.join(commands) + '\n')
            print(f'启动 {run}', flush=True)
            with (out / f'logs/{run}.log').open('w') as log:
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'stop', '--time', '30', name], check=False)
                    code = 124
            cleanup = gpu()
            for _ in range(20):
                if not cleanup['processes'] and all(int(row.split(',')[2]) <= 100 for row in cleanup['gpus'].splitlines()):
                    break
                time.sleep(1)
                cleanup = gpu()
            cleanup['passed'] = not cleanup['processes'] and all(int(row.split(',')[2]) <= 100 for row in cleanup['gpus'].splitlines())
            save(out / f'correctness/gpu_cleanup_{run}.json', cleanup)
            path = out / 'runs' / run / 'record.json'
            record = read(path) if path.exists() else {'status': 'FAIL', 'error': '运行记录缺失'}
            row = {'run_id': run, 'status': record['status'], 'exit_code': code, 'cleanup_passed': cleanup['passed']}
            runs.append(row)
            save(out / 'progress.json', {'runs': runs, 'pairs': pairs})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if code or record['status'] != 'PASS' or not cleanup['passed']:
                error = f"{run} 条件失败：{record.get('error', row)}"
                break
        if error:
            break
        paired = paired_record(out, pair)
        pairs.append(paired)
        save(out / 'paired_measurements.json', pairs)
        print(json.dumps(paired, ensure_ascii=False), flush=True)
        if paired['status'] != 'PASS':
            error = f'配对 {pair} 的公共状态一致性失败'
            break
    return finalize(out, frozen, runs, pairs, error)


if __name__ == '__main__':
    raise SystemExit(main())
