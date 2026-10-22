"""attempt2 续跑启动器：跳过已 PASS 的臂（census），其余臂按 protocol 顺序执行。

与 launch.py 的差别：
- 不断言 execution_started.json 不存在（attempt1 已写入），另写 execution3_started.json；
- 每臂若已存在 PASS summary 则跳过；attempt1 失败臂的证据已改名 *.attempt1* 保留；
- 身份锁另存 implementation_lock3.json（覆盖 v1.2 修正后的 control.py）。
臂内合同、GPU 礼仪、清理校验、失败即停与 launch.py 完全一致。
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
    assert not (OUT / 'launch3_complete.json').exists(), 'attempt3 已完成，禁止重跑'
    source = read(OUT / 'source_identity.json')
    for p, h in source['sha256'].items():
        assert sha(p) == h, p
    image_id = source['image_id']
    assert json.loads(run(['docker', 'image', 'inspect', image_id]))[0]['Id'] == image_id
    protocol = read(OUT / 'protocol.json')
    arms = protocol['arms']
    lock = {str(p): sha(p) for p in sorted(OUT.glob('*.py'))}
    lock.update({str(OUT / f): sha(OUT / f) for f in
                 ['protocol.json', 'policy_plans.json', 'prepared_inputs.json',
                  'cpu_candidate_prefix_mapping.json', 'cpu_gate.json', 'source_identity.json']})
    write(OUT / 'implementation_lock3.json', {'files': lock, 'image': image_id,
                                              'note': 'attempt3：含 v1.2 时间线合同修正后的 control.py'})
    write(OUT / 'execution3_started.json', {'unix_ns': time.time_ns(), 'arms': arms,
                                            'note': 'census=PASS 保留；attempt1/2 证据在 *.attempt1*/*.attempt2*'})
    shared_files = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json',
                    'cpu_candidate_prefix_mapping.json', 'policy_plans.json']
    strict_sources = {}
    status = {}
    # 已完成臂先登记（本轮跳过但作为 cold 目标请求来源时有效）
    for arm in arms:
        if (OUT / arm / 'run' / 'summary.json').exists() and read(OUT / arm / 'run' / 'summary.json').get('status') == 'PASS':
            status[arm] = 'PASS_PRIOR'
            if arm.startswith('strict_'):
                strict_sources.setdefault(arm.split('_')[-1], OUT / arm)
    try:
        for arm in arms:
            if status.get(arm) == 'PASS_PRIOR':
                continue
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
            if mode == 'strict':
                strict_sources.setdefault(env['PILOT_PENDING'], case)
            status[arm] = 'PASS'
            (OUT / 'arm_status3.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        write(OUT / 'launch3_complete.json', {'status': 'COMPLETE', 'arms': status, 'unix_ns': time.time_ns()})
    except BaseException as exc:
        sf = OUT / 'arm_status3.json'
        write(OUT / 'launch3_failure.json', {'error': repr(exc),
                                             'completed_arms': read(sf) if sf.exists() else {}})
        raise
    finally:
        logs = []
        for arm in arms:
            p = OUT / f'{arm}.runtime.log'
            if p.exists():
                logs.append('[生命周期 ' + arm + ']\n' + p.read_text())
        (OUT / 'runtime3.log').write_text('\n'.join(logs))


if __name__ == '__main__':
    main()
