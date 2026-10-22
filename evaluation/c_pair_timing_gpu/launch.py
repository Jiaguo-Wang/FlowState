"""phase1 启动器：按 protocol 冻结顺序执行 20 次独立 Engine 生命周期（4 条件 × 5 对）。

- 每生命周期独立容器：GPU 空闲礼仪、专用容器、清理校验、失败即停，与 pilot launch 纪律一致；
- 冻结输入四件套 + Cold 参照 + 运行时身份逐生命周期拷入 case 目录；
- 实现锁覆盖本目录全部 .py 与冻结文件，臂启动前逐文件复核。
"""
import hashlib
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/c_pair_timing_gpu'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


SHARED = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json',
          'cpu_candidate_prefix_mapping.json', 'policy_plans.json',
          'cold_reference_C_B.json', 'cold_reference_C_C.json']
LOCKED = SHARED + ['protocol.json']


def launch_phase(arms, phase_tag):
    """执行一组生命周期；phase_tag 区分 phase1/phase2 的状态与日志文件名。"""
    source = read(OUT / 'source_identity.json')
    image_id = source['image_id']
    assert json.loads(run(['docker', 'image', 'inspect', image_id]))[0]['Id'] == image_id
    protocol = read(OUT / 'protocol.json')
    lock = {str(p): sha(p) for p in sorted(OUT.glob('*.py'))}
    lock.update({str(OUT / f): sha(OUT / f) for f in LOCKED})
    write(OUT / f'implementation_lock_{phase_tag}.json', {'files': lock, 'image': image_id})
    write(OUT / f'execution_started_{phase_tag}.json',
          {'unix_ns': time.time_ns(), 'arms': [a['name'] for a in arms]})
    status = {}
    try:
        for arm in arms:
            assert time.time() < protocol['deadline_unix'], '时限已到，剩余生命周期如实标记未测'
            case = OUT / arm['name']
            case.mkdir(exist_ok=False)
            for f in SHARED:
                shutil.copyfile(OUT / f, case / f)
            env = {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                   'FLOWSTATE_STEP5D_PORT': '49987', 'B5D4_RUN': 'run', 'B5D4_ARM': 'strict',
                   'PILOT_MODE': arm['mode'], 'PILOT_POLICY': arm['policy'], 'PILOT_PENDING': arm['pending'],
                   'TRITON_CACHE_DIR': '/b5out/triton_cache',
                   'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
            for p, h in lock.items():
                assert sha(p) == h, p
            apps = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            assert len(apps.strip().splitlines()) == 1, 'GPU0 存在其他任务（共享GPU礼仪：不抢占，停止）'
            gpu = run(['nvidia-smi', '-i', '0', '--query-gpu=uuid,name,memory.used,memory.total', '--format=csv'])
            assert 'H100 PCIe' in gpu
            assert int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip()) < 1000
            name = 'flowstate-c-pair-' + arm['name'].replace('_', '-').lower()
            assert not run(['docker', 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}']).strip(), '专用容器已存在'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 49987))
            write(OUT / f"{arm['name']}_gpu_before.json", {'gpu': gpu, 'processes': apps})
            cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device=0',
                   '--network', 'host', '--shm-size', '32g']
            for src, dst, ro in [(ROOT, ROOT, True),
                                 (Path('/home/wjg/models/qwen3.5-9b'), Path('/home/wjg/models/qwen3.5-9b'), True),
                                 (case, Path('/b5out'), False)]:
                cmd += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
            for k, v in env.items():
                cmd += ['--env', k + '=' + v]
            cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', image_id,
                    '-B', '-m', 'evaluation.c_pair_timing_gpu.runner']
            write(OUT / f"{arm['name']}_command.json", cmd)
            try:
                with (OUT / f"{arm['name']}.runtime.log").open('x') as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(1800, max(1, int(protocol['deadline_unix'] - time.time()))))
                write(OUT / f"{arm['name']}_process.json", {'returncode': r.returncode})
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
                write(OUT / f"{arm['name']}_cleanup.json", cleanup)
            assert cleanup['status'] == 'PASS', '清理失败：' + arm['name']
            assert r.returncode == 0, '本生命周期失败，不继续：' + arm['name']
            summary = read(case / 'run' / 'summary.json')
            assert summary['status'] == 'PASS', '缺少实际正确性摘要：' + arm['name']
            status[arm['name']] = 'PASS'
            (OUT / f'arm_status_{phase_tag}.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        write(OUT / f'launch_{phase_tag}_complete.json',
              {'status': 'COMPLETE', 'arms': status, 'unix_ns': time.time_ns()})
    except BaseException as exc:
        sf = OUT / f'arm_status_{phase_tag}.json'
        write(OUT / f'launch_{phase_tag}_failure.json',
              {'error': repr(exc), 'completed_arms': read(sf) if sf.exists() else {}})
        raise
    finally:
        logs = []
        for arm in arms:
            p = OUT / f"{arm['name']}.runtime.log"
            if p.exists():
                logs.append('[生命周期 ' + arm['name'] + ']\n' + p.read_text())
        (OUT / f'runtime_{phase_tag}.log').write_text('\n'.join(logs))


def main():
    assert not (OUT / 'launch_p1_complete.json').exists(), 'phase1 已完成，禁止重跑'
    protocol = read(OUT / 'protocol.json')
    launch_phase(protocol['phase1_arms'], 'p1')


if __name__ == '__main__':
    main()
