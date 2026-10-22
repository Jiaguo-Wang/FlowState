"""每臂仅一次的顺序启动器：census 先行且通过结构门禁后才进入 strict/cold 臂。

- 每臂独立容器与 Engine 生命周期，臂间校验 GPU 归零与容器清理；
- 任一臂失败即停止（保留日志），不自动重试；
- 共享 GPU 礼仪：每臂启动前断言无其他计算进程。
"""
import hashlib
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main():
    assert not (OUT / 'execution_started.json').exists(), '禁止重跑已启动生命周期'
    source = read(OUT / 'source_identity.json')
    for p, h in source['sha256'].items():
        assert sha(p) == h, p
    image_id = source['image_id']
    image = json.loads(run(['docker', 'image', 'inspect', image_id]))
    assert image[0]['Id'] == image_id
    protocol = read(OUT / 'protocol.json')
    arms = protocol['arms']
    lock = {str(p): sha(p) for p in sorted(OUT.glob('*.py'))}
    lock.update({str(OUT / f): sha(OUT / f) for f in
                 ['protocol.json', 'policy_plans.json', 'prepared_inputs.json',
                  'cpu_candidate_prefix_mapping.json', 'cpu_gate.json', 'source_identity.json']})
    write(OUT / 'implementation_lock.json', {'files': lock, 'image': image_id})
    write(OUT / 'execution_started.json', {'unix_ns': time.time_ns(), 'arms': arms,
                                           'target': 'C Shared/Branched K=3 首次三策略真实 GPU Pilot'})
    shared_files = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json',
                    'cpu_candidate_prefix_mapping.json', 'policy_plans.json']
    strict_sources = {}  # pending -> 首个完成的 strict 臂目录（供 cold 臂取 target_request.json）
    status = {}
    try:
        for arm in arms:
            assert time.time() < protocol['deadline_unix'], '时限已到，剩余臂如实标记未测'
            case = OUT / arm
            case.mkdir(exist_ok=False)
            mode = arm.split('_')[0]
            for f in shared_files:
                shutil.copyfile(OUT / f, case / f)
            env = {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                   'FLOWSTATE_STEP5D_PORT': '49987', 'B5D4_RUN': 'run', 'PILOT_MODE': mode,
                   'TRITON_CACHE_DIR': '/b5out/triton_cache',
                   'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
            if mode == 'census':
                env['B5D4_ARM'] = 'census'
            elif mode == 'strict':
                _, policy, pending = arm.split('_', 2)
                env.update(B5D4_ARM='strict', PILOT_POLICY=policy, PILOT_PENDING=pending)
            else:
                pending = arm.split('_', 1)[1]
                if pending not in strict_sources:
                    raise RuntimeError('Cold 臂缺少同 Pending 的已完成 strict 臂：' + pending)
                shutil.copyfile(strict_sources[pending] / 'target_request.json', case / 'target_request.json')
                env.update(B5D4_ARM='cold', PILOT_PENDING=pending)
            for p, h in lock.items():
                assert sha(p) == h, p
            apps = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            assert len(apps.strip().splitlines()) == 1, 'GPU0 存在其他任务（共享GPU礼仪：不抢占，停止）'
            gpu = run(['nvidia-smi', '-i', '0', '--query-gpu=uuid,name,memory.used,memory.total', '--format=csv'])
            assert 'H100 PCIe' in gpu
            assert int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip()) < 1000
            name = 'flowstate-c-pilot-' + arm.replace('_', '-').replace('.', '-').lower()
            assert not run(['docker', 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}']).strip(), '专用容器已存在'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 49987))
            write(OUT / f'{arm}_gpu_before.json', {'gpu': gpu, 'processes': apps})
            cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device=0',
                   '--network', 'host', '--shm-size', '32g']
            for src, dst, ro in [(ROOT, ROOT, True),
                                 (Path('/home/wjg/models/qwen3.5-9b'), Path('/home/wjg/models/qwen3.5-9b'), True),
                                 (case, Path('/b5out'), False)]:
                cmd += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
            for k, v in env.items():
                cmd += ['--env', k + '=' + v]
            cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', image_id,
                    '-B', '-m', 'evaluation.c_shared_gpu_pilot_k3.runner']
            write(OUT / f'{arm}_command.json', cmd)
            try:
                with (OUT / f'{arm}.runtime.log').open('x') as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(1800, max(1, int(protocol['deadline_unix'] - time.time()))))
                write(OUT / f'{arm}_process.json', {'returncode': r.returncode})
            finally:
                removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True)
                obs = []
                for _ in range(3):
                    time.sleep(3)
                    mem = int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip())
                    apps2 = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
                    obs.append({'memory_mib': mem, 'processes': apps2})
                own = run(['docker', 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}']).strip()
                cleanup = {'status': 'PASS' if not own and all(x['memory_mib'] < 100 and len(x['processes'].strip().splitlines()) == 1 for x in obs) else 'FAIL',
                           'observations': obs, 'own_container': own, 'remove_stderr': removed.stderr}
                write(OUT / f'{arm}_cleanup.json', cleanup)
            assert cleanup['status'] == 'PASS', '清理失败：' + arm
            assert r.returncode == 0, '本臂失败，不继续：' + arm
            summary = read(case / 'run' / 'summary.json')
            assert summary['status'] == 'PASS', '缺少实际正确性摘要：' + arm
            if mode == 'census':
                # 结构门禁：物理域必须与 CPU 16 候选域一致，且四 Pending 活前沿与声明一致，否则不进入 strict
                assert summary['mapped_count'] == 16 and not summary['unmapped_extras'] and not summary['missing_physical'], \
                    'Census 结构断言失败：mapped=%d extras=%d missing=%d' % (summary['mapped_count'], len(summary['unmapped_extras']), len(summary['missing_physical']))
                assert all(p['match'] for p in summary['pendings_live_vs_declared']), 'Census：活前沿与 CPU 声明不一致'
                assert summary['slots_used'] + summary['free_slots'] == summary['pool_capacity'], '槽位记账不闭合'
            if mode == 'strict':
                pending = env['PILOT_PENDING']
                strict_sources.setdefault(pending, case)
            status[arm] = 'PASS'
            (OUT / 'arm_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        write(OUT / 'launch_complete.json', {'status': 'COMPLETE', 'arms': status, 'unix_ns': time.time_ns()})
    except BaseException as exc:
        status_file = OUT / 'arm_status.json'
        write(OUT / 'launch_failure.json', {'error': repr(exc), 'completed_arms': read(status_file) if status_file.exists() else {}})
        raise
    finally:
        logs = []
        for arm in arms:
            p = OUT / f'{arm}.runtime.log'
            if p.exists():
                logs.append('[生命周期 ' + arm + ']\n' + p.read_text())
        (OUT / 'runtime.log').write_text('\n'.join(logs))


if __name__ == '__main__':
    main()
