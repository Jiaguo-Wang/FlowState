"""顺序执行一次烟测和三个独立生命周期，并逐次验证显存清理。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'lmsysorg/sglang:v0.5.17-cu129-runtime'
MODEL = '/data1/models/Kimi-Linear-48B-A3B-Instruct'


def save(path, value):
    """保存独立的阶段二证据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def command(args):
    """运行只读环境探针并要求成功。"""
    return subprocess.check_output(args, text=True).strip()


def gpu():
    """读取两卡显存和所有计算进程。"""
    return {'gpus': command(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits']),
            'processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])}


def main():
    """前一生命周期全部通过才进入下一生命周期，遇错停止。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    initial_gpu = gpu()
    if initial_gpu['processes'] or any(int(row.split(',')[2]) > 100 for row in initial_gpu['gpus'].splitlines()):
        raise RuntimeError('GPU 非空闲，停止启动')
    environment = {'image': json.loads(command(['docker', 'image', 'inspect', IMAGE]))[0],
        'gpu_before': initial_gpu, 'tp': 2, 'model': MODEL,
        '说明': '使用已验证镜像和本地只读模型；不下载模型，不运行其他阶段。'}
    save(out / 'environment.json', environment)
    save(out / 'model_info.json', {'path': MODEL, 'source': 'moonshotai/Kimi-Linear-48B-A3B-Instruct',
        'config': json.loads(Path(MODEL, 'config.json').read_text()),
        'integrity_evidence': str(ROOT / 'evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043/model_integrity.json'),
        '说明': '沿用阶段一完整性 PASS，无重复下载或权重扫描。'})
    tests = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
        '--mount', f'type=bind,src={ROOT},dst={ROOT},readonly',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', f'PYTHONPATH={ROOT}',
        '--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE,
        '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_rq5b_kimi_phase2.py', 'tests/test_rq6_batched_control.py',
        'tests/test_optimizer.py', 'tests/test_executable_state.py', 'tests/test_sglang_adapter.py']
    with (out / 'logs/related_tests.log').open('w') as log:
        tested = subprocess.run(tests, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if tested.returncode:
        raise RuntimeError('相关 CPU 测试失败，停止 GPU 实验')
    save(out / 'correctness/source_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / 'evaluation/rq5b_kimi_phase2.py', ROOT / 'evaluation/run_rq5b_kimi_phase2.py', ROOT / 'tests/runtime/rq5b_kimi_phase2_transport.py', ROOT / 'tests/test_rq5b_kimi_phase2.py')})
    summary = []
    commands = []
    for run in ('smoke', 'rep01', 'rep02', 'rep03'):
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
            'FLOWSTATE_RQ5B_PORT': '49971'}.items():
            cmd.extend(['--env', f'{key}={value}'])
        cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE,
                '-m', 'evaluation.rq5b_kimi_phase2', '--output-root', '/output', '--run-id', run]
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
        record_path = out / 'runs' / run / 'record.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'status': 'FAIL', 'error': '无运行记录'}
        row = {'run_id': run, 'exit_code': code, 'status': record['status'], 'cleanup_passed': cleanup['passed']}
        summary.append(row)
        save(out / 'summary.json', summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if code or record['status'] != 'PASS' or not cleanup['passed']:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
