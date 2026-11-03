"""T2-G1 Smoke 启动器：三策略各 1 臂（Pending C_B），每臂独立 Docker/Engine 生命周期。

复用 pilot launch3.py 的结构（身份锁/证据留档/清理校验/失败即停），差异仅两处并明示：
1. GPU 礼仪放宽：pilot 要求 GPU0 <1000MiB 且无他人进程；本轮 GPU0 有 1.5G 他人常驻进程
   （robot-learning），放宽为 空闲≥40000MiB + 利用率0% + 记录共存进程（engine 需 ~32.6G，
   无争用）。正式 60 臂实验仍须空卡或另行拍板。
2. 臂域为 T2-G1 三策略 smoke case；运行模块/共享输入零改动（smoke_lock.json 锁定摘要）。
"""
import json
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
SMOKE = ROOT / 'evaluation/t2g1_three_policy_gpu/smoke'
ARMS = ['strict_FlowState-Greedy_C_B', 'strict_Marconi-Evict_C_B', 'strict_KVFlow-inspired_C_B']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def main():
    assert not (SMOKE / 'smoke_complete.json').exists(), 'smoke 已完成，禁止重跑'
    image_id = read(SMOKE / 'strict_FlowState-Greedy_C_B' / 'source_identity.json')['image_id']
    assert json.loads(run(['docker', 'image', 'inspect', image_id]))[0]['Id'] == image_id
    lock = read(SMOKE / 'smoke_lock.json')['files']
    status = {}
    try:
        for arm in ARMS:
            for p, h in lock.items():
                assert sha(p) == h, '文件被修改：' + p
            # GPU 门禁（放宽版，见模块说明）：空闲≥40G、利用率0%、共存进程留档
            used = int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip())
            total = int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.total', '--format=csv,noheader,nounits']).strip())
            util = int(run(['nvidia-smi', '-i', '0', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).strip())
            apps = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            assert total - used >= 40000 and util == 0, 'GPU0 不满足放宽门禁（used=%dMiB util=%d%%）' % (used, util)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 49987))
            write(SMOKE / f'{arm}_gpu_before.json', {'used_mib': used, 'total_mib': total, 'util_pct': util, 'processes': apps,
                                                     'gate': '放宽版：空闲≥40G+利用率0%；正式实验须空卡'})
            _, policy, pending = arm.split('_', 2)
            name = 'flowstate-t2g1-smoke-' + arm.replace('_', '-').lower()
            assert not run(['docker', 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}']).strip()
            env = {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                   'FLOWSTATE_STEP5D_PORT': '49987', 'B5D4_RUN': 'run', 'PILOT_MODE': 'strict', 'B5D4_ARM': 'strict',
                   'PILOT_POLICY': policy, 'PILOT_PENDING': pending,
                   'TRITON_CACHE_DIR': '/b5out/triton_cache',
                   'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
            cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device=0',
                   '--network', 'host', '--shm-size', '32g']
            for src, dst, ro in [(ROOT, ROOT, True),
                                 (Path('/home/wjg/models/qwen3.5-9b'), Path('/home/wjg/models/qwen3.5-9b'), True),
                                 (SMOKE / arm, Path('/b5out'), False)]:
                cmd += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
            for k, v in env.items():
                cmd += ['--env', k + '=' + v]
            cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', image_id,
                    '-B', '-m', 'evaluation.c_shared_gpu_pilot_k3.runner']
            write(SMOKE / f'{arm}_command.json', cmd)
            with (SMOKE / f'{arm}.runtime.log').open('x') as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            write(SMOKE / f'{arm}_process.json', {'returncode': r.returncode})
            finally_removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True)
            obs = []
            for _ in range(3):
                time.sleep(3)
                obs.append({'memory_mib': int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip()),
                            'processes': run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])})
            own = run(['docker', 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}']).strip()
            cleanup = {'status': 'PASS' if not own and all(o['memory_mib'] < 4000 for o in obs) else 'FAIL',
                       'observations': obs, 'own_container': own, 'remove_stderr': finally_removed.stderr,
                       'note': '清理阈值放宽为<4000MiB（GPU0 有 1.5G 他人常驻进程的基线）'}
            write(SMOKE / f'{arm}_cleanup.json', cleanup)
            assert cleanup['status'] == 'PASS', '清理失败：' + arm
            assert r.returncode == 0, '本臂失败，不继续：' + arm
            summary = read(SMOKE / arm / 'run' / 'summary.json')
            assert summary['status'] == 'PASS', '摘要非PASS：' + arm
            status[arm] = 'PASS'
            (SMOKE / 'smoke_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        write(SMOKE / 'smoke_complete.json', {'status': 'COMPLETE', 'arms': status, 'unix_ns': time.time_ns()})
        print('smoke 全部完成：', status)
    except BaseException as exc:
        write(SMOKE / 'smoke_failure.json', {'error': repr(exc), 'completed': status})
        raise


if __name__ == '__main__':
    main()
