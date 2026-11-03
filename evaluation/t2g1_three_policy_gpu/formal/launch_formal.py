"""T2-G2 正式 GPU 对照启动器：门禁（3 臂）与主矩阵（60 臂）两阶段，按 protocol.json 冻结顺序执行。

- 复用 c_formal_perf_gpu 的 runner/bridge/control（零改动）：容器内仍运行
  `evaluation.c_formal_perf_gpu.runner`，测量合同（strict_warm + FS_MEASURE_MODE=1 + 形状预热 B v1.1 +
  请求窗口零审计零编译 + Cold 逐 Token + census/E·T·G 断言）全部由该栈自带；
- 与 c_formal launch.py 的差异仅三处并明示：(1) 臂域为 T2-G2 protocol 的 63 臂（策略含
  Marconi-Evict/KVFlow-inspired，经 case 目录 policy_plans.json 注入，control 泛型消费）；
  (2) 实现锁额外覆盖被复用的 c_formal 三个模块与其实现锁交叉核对；(3) deadline 于各阶段启动时
  固定 now+6h（c_formal 为协议内预置，本轮启动时刻待批准故改为启动时冻结）；
- GPU 空卡礼仪与 c_formal 完全一致：无其他 compute 进程、显存<1000MiB、H100 PCIe——不沿用 Smoke 放宽条件；
- 主矩阵前置：launch_gate_complete.json + 人工复核 gate_passed.json；已完成阶段禁止重跑；
- 每臂独立容器与 Engine/Triton 生命周期、专用容器名、3×清理校验、失败即停保留证据。
"""
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/t2g1_three_policy_gpu/formal'
CFORMAL = ROOT / 'evaluation/c_formal_perf_gpu'
RUNNER_MODULE = 'evaluation.c_formal_perf_gpu.runner'  # 零改动复用的测量栈入口
REUSED_MODULES = ['runner.py', 'bridge.py', 'control.py']  # 被复用的 c_formal 模块（逐臂哈希复核）
SHARED = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json',
          'cpu_candidate_prefix_mapping.json', 'policy_plans.json',
          'cold_reference_C_A.json', 'cold_reference_C_B.json',
          'cold_reference_C_C.json', 'cold_reference_C_D.json']
DEADLINE_HOURS = 6  # 各阶段启动时固定的截止窗口（覆盖 60 臂≈2–2.5h + 余量）


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def build_lock():
    """实现锁：本目录全部 .py + 冻结输入与调度 + 被复用的 c_formal 模块（与其实现锁交叉核对）。"""
    lock = {str(p): sha(p) for p in sorted(OUT.glob('*.py'))}
    lock.update({str(OUT / f): sha(OUT / f) for f in SHARED + ['protocol.json']})
    lock_main = read(CFORMAL / 'implementation_lock_main.json')['files']
    for m in REUSED_MODULES:
        src = CFORMAL / m
        assert sha(src) == lock_main[str(src)], 'c_formal 模块与其实现锁不符（禁止改动）：' + m
        lock[str(src)] = sha(src)
    return lock


def launch_phase(arms, phase_tag):
    """执行一组生命周期；phase_tag 区分 gate/main 的状态与日志文件名。"""
    source = read(OUT / 'source_identity.json')
    image_id = source['image_id']
    assert json.loads(run(['docker', 'image', 'inspect', image_id]))[0]['Id'] == image_id
    protocol = read(OUT / 'protocol.json')
    deadline = time.time() + DEADLINE_HOURS * 3600
    lock = build_lock()
    write(OUT / ('implementation_lock_%s.json' % phase_tag), {'files': lock, 'image': image_id})
    write(OUT / ('execution_started_%s.json' % phase_tag),
          {'unix_ns': time.time_ns(), 'deadline_unix': deadline, 'deadline_policy':
           '启动时固定 now+%dh（protocol.json deadline_policy）' % DEADLINE_HOURS,
           'arms': [a['name'] for a in arms]})
    status = {}
    try:
        for arm in arms:
            assert time.time() < deadline, '时限已到，剩余生命周期如实标记未测'
            case = OUT / arm['name']
            case.mkdir(exist_ok=False)
            for f in SHARED:
                (case / f).write_bytes((OUT / f).read_bytes())
            env = {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                   'FLOWSTATE_STEP5D_PORT': '49987', 'B5D4_RUN': 'run', 'B5D4_ARM': 'strict',
                   'PILOT_MODE': 'strict_warm', 'PILOT_POLICY': arm['policy'], 'PILOT_PENDING': arm['pending'],
                   'FS_MEASURE_MODE': '1', 'TRITON_CACHE_DIR': '/b5out/triton_cache',
                   'PYTHONPATH': '%s:%s:%s' % (ROOT, ROOT / 'tests/runtime',
                                               ROOT / 'motivation/artifacts/wp3b_gate_20260820')}
            for p, h in lock.items():
                assert sha(p) == h, p
            # GPU 空卡门禁（正式礼仪，与 c_formal 一致；不用 Smoke 放宽条件）
            apps = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            assert len(apps.strip().splitlines()) == 1, 'GPU0 存在其他任务（空卡礼仪：不抢占，停止）'
            gpu = run(['nvidia-smi', '-i', '0', '--query-gpu=uuid,name,memory.used,memory.total', '--format=csv'])
            assert 'H100 PCIe' in gpu
            assert int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                            '--format=csv,noheader,nounits']).strip()) < 1000, 'GPU0 显存基线异常'
            name = 'flowstate-t2g2-' + arm['name'].replace('_', '-').lower()
            assert not run(['docker', 'ps', '-a', '--filter', 'name=' + name,
                            '--format', '{{.Names}}']).strip(), '专用容器已存在'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 49987))
            write(OUT / ('%s_gpu_before.json' % arm['name']), {'gpu': gpu, 'processes': apps})
            cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device=0',
                   '--network', 'host', '--shm-size', '32g']
            for src, dst, ro in [(ROOT, ROOT, True),
                                 (Path('/home/wjg/models/qwen3.5-9b'), Path('/home/wjg/models/qwen3.5-9b'), True),
                                 (case, Path('/b5out'), False)]:
                cmd += ['--mount', 'type=bind,src=%s,dst=%s%s' % (src, dst, ',readonly' if ro else '')]
            for k, v in env.items():
                cmd += ['--env', k + '=' + v]
            cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', image_id,
                    '-B', '-m', RUNNER_MODULE]
            write(OUT / ('%s_command.json' % arm['name']), cmd)
            try:
                with (OUT / ('%s.runtime.log' % arm['name'])).open('x') as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(1800, max(1, int(deadline - time.time()))))
                write(OUT / ('%s_process.json' % arm['name']), {'returncode': r.returncode})
            finally:
                removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True)
                obs = []
                for _ in range(3):
                    time.sleep(3)
                    mem = int(run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits']).strip())
                    apps2 = run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_memory',
                                 '--format=csv'])
                    obs.append({'memory_mib': mem, 'processes': apps2})
                own = run(['docker', 'ps', '-a', '--filter', 'name=' + name,
                           '--format', '{{.Names}}']).strip()
                cleanup = {'status': 'PASS' if not own and all(
                    x['memory_mib'] < 100 and len(x['processes'].strip().splitlines()) == 1 for x in obs) else 'FAIL',
                    'observations': obs, 'own_container': own, 'remove_stderr': removed.stderr}
                write(OUT / ('%s_cleanup.json' % arm['name']), cleanup)
            assert cleanup['status'] == 'PASS', '清理失败：' + arm['name']
            assert r.returncode == 0, '本生命周期失败，不继续：' + arm['name']
            summary = read(case / 'run' / 'summary.json')
            assert summary['status'] == 'PASS', '缺少实际正确性摘要：' + arm['name']
            status[arm['name']] = 'PASS'
            (OUT / ('arm_status_%s.json' % phase_tag)).write_text(
                json.dumps(status, ensure_ascii=False, indent=2))
            print('PASS', arm['name'], 'TTFT', round(summary['ttft_ms']), 'ms', flush=True)
        write(OUT / ('launch_%s_complete.json' % phase_tag),
              {'status': 'COMPLETE', 'arms': status, 'unix_ns': time.time_ns()})
    except BaseException as exc:
        sf = OUT / ('arm_status_%s.json' % phase_tag)
        write(OUT / ('launch_%s_failure.json' % phase_tag),
              {'error': repr(exc), 'completed_arms': read(sf) if sf.exists() else {}})
        raise
    finally:
        logs = []
        for arm in arms:
            p = OUT / ('%s.runtime.log' % arm['name'])
            if p.exists():
                logs.append('[生命周期 ' + arm['name'] + ']\n' + p.read_text())
        (OUT / ('runtime_%s.log' % phase_tag)).write_text('\n'.join(logs))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'gate'
    protocol = read(OUT / 'protocol.json')
    assert read(OUT / 'formal_precheck.json')['status'] == 'READY', '预检未通过'
    if phase == 'gate':
        assert not (OUT / 'launch_gate_complete.json').exists(), '门禁已完成，禁止重跑'
        launch_phase(protocol['gate_arms'], 'gate')
    elif phase == 'main':
        assert (OUT / 'launch_gate_complete.json').exists(), '门禁未完成'
        gp = read(OUT / 'gate_passed.json')
        assert gp.get('passed') is True, '门禁未复核通过'
        assert not (OUT / 'launch_main_complete.json').exists(), '主矩阵已完成，禁止重跑'
        launch_phase(protocol['main_arms'], 'main')
    else:
        raise ValueError('未知阶段：' + phase)


if __name__ == '__main__':
    main()
