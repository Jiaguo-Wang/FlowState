"""P0-4 第一轮 GPU 验证·宿主启动器：前置检查 → 逐臂 docker 生命周期（串行）→ 记录。

纪律（Task 0/P0-4）：共享机器，每臂启动前 nvidia-smi 实查选卡；每臂前后 df -B1G / 复查
（<20G 暂停）；镜像冻结 sha256:1dfce903…（--pull never）；三臂串行执行，一臂失败即停止
（fail-loud，不自动重试——重试须人工审查后显式发起并记录）。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PKG = ROOT / 'evaluation/p0_control_cycle_gpu'
ARMS = PKG / 'arms'
IMAGE = 'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926'
MODEL = '/home/wjg/models/qwen3.5-9b'
PORT = '50087'
POLICY_ARM = {'FlowState-Greedy': 'r1_ck3_fs', 'Marconi-Evict': 'r1_ck3_mc', 'KVFlow-inspired': 'r1_ck3_kv'}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def df_root():
    line = sh("df -B1G / | tail -1")
    return int(line.split()[3])


def pick_gpu():
    """实查两卡占用，返回空闲卡 UUID（<100MiB 即视为空闲；都不空闲则失败）。"""
    out = sh("nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits")
    gpus = []
    for line in out.splitlines():
        uuid, used = [x.strip() for x in line.split(',')]
        gpus.append((uuid, int(used)))
    free = [u for u, m in gpus if m < 100]
    assert free, '无空闲GPU：' + repr(gpus)
    # 其他人进程检查（监控容器除外）：发现非 nvmon/fabricmanager 类进程即失败
    procs = sh("nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader")
    assert not procs.strip(), 'GPU 上存在他人计算进程：' + procs
    return free[0]


def docker_cmd(policy, arm, gpu_uuid, container_name):
    return ['docker', 'run', '--rm', '--pull', 'never',
            '--name', container_name,
            '--gpus', 'device=' + gpu_uuid,
            '--network', 'host',
            '--shm-size', '32g',
            '--mount', 'type=bind,src=%s,dst=%s,readonly' % (ROOT, ROOT),
            '--mount', 'type=bind,src=%s,dst=%s,readonly' % (MODEL, MODEL),
            '--mount', 'type=bind,src=%s,dst=/b5out' % (ARMS / arm),
            '--env', 'CUDA_VISIBLE_DEVICES=' + gpu_uuid,
            '--env', 'EXPECTED_GPU_UUID=' + gpu_uuid,
            '--env', 'HF_HUB_OFFLINE=1',
            '--env', 'TRANSFORMERS_OFFLINE=1',
            '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--env', 'PYTHONUNBUFFERED=1',
            '--env', 'FLOWSTATE_STEP5D_PORT=' + PORT,
            '--env', 'B5D4_RUN=run',
            '--env', 'B5D4_ARM=strict',
            '--env', 'PILOT_MODE=strict_warm',
            '--env', 'PILOT_POLICY=' + policy,
            '--env', 'FS_MEASURE_MODE=1',
            '--env', 'TRITON_CACHE_DIR=/b5out/triton_cache',
            '--env', 'PYTHONPATH=%s:%s:%s' % (ROOT, ROOT / 'tests/runtime',
                                              ROOT / 'motivation/artifacts/wp3b_gate_20260820'),
            '--workdir', str(ROOT),
            '--entrypoint', 'python3',
            IMAGE,
            '-B', str(PKG / 'device_entry.py')]


def run_arm(policy, arm, timeout_s=5400):
    run_dir = ARMS / arm / 'run'
    gpu_uuid = pick_gpu()
    disk_before = df_root()
    gpu_before = sh("nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader")
    name = 'flowstate-p04-%s' % arm
    log_path = ARMS / (arm + '.runtime.log')
    cmd = docker_cmd(policy, arm, gpu_uuid, name)
    (ARMS / (arm + '_command.json')).write_text(json.dumps(cmd, indent=2))
    print('启动', policy, 'GPU', gpu_uuid, '磁盘余', disk_before, 'G', flush=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        (ARMS / (arm + '_process.json')).write_text(json.dumps(
            {'pid': proc.pid, 'container': name, 'gpu_uuid': gpu_uuid,
             'disk_free_gb_before': disk_before, 'gpu_before': gpu_before}, indent=2))
        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                break
            assert time.monotonic() - t0 < timeout_s, '臂超时（%ds）：%s' % (timeout_s, arm)
            done = (run_dir / 'summary.json').exists() or (run_dir / 'failure.json').exists()
            time.sleep(30)
            if done and proc.poll() is None:
                # runner 已落盘结论，等待容器自然退出（engine shutdown + 清理文件）
                if time.monotonic() - t0 > timeout_s - 600:
                    break
    rc = proc.poll()
    if rc is None:
        proc.terminate()
        rc = proc.wait(30)
    disk_after = df_root()
    status = 'PASS' if (run_dir / 'summary.json').exists() else ('BLOCKED' if rc not in (0,) else 'UNKNOWN')
    (ARMS / (arm + '_cleanup.json')).write_text(json.dumps(
        {'returncode': rc, 'status': status,
         'disk_free_gb_after': disk_after, 'disk_free_gb_delta': disk_after - disk_before,
         'wall_s': time.monotonic() - t0}, indent=2))
    print('完成', policy, 'rc', rc, status, '磁盘余', disk_after, 'G', flush=True)
    return status, disk_after


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else list(POLICY_ARM)
    assert df_root() > 20, '根分区余量不足 20G，暂停'
    results = {}
    for policy in only:
        arm = POLICY_ARM[policy]
        assert (ARMS / arm / 'prepared_inputs.json').exists(), '臂目录未装配：' + arm
        status, free = run_arm(policy, arm)
        results[policy] = {'arm': arm, 'status': status, 'disk_free_gb': free}
        if status != 'PASS':
            break   # fail-loud：一臂失败即停止，不自动重试
        assert free > 20, '根分区余量不足 20G，暂停'
    (ARMS / 'arms_status.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
