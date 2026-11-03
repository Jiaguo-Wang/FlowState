"""新批次串行执行器：逐臂资源门禁、独立生命周期、保留失败且不自动重试。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
ARMS = PKG / 'arms'
GPU = 'GPU-bfff4d31-0b94-7de3-9946-f593437ae566'
IMAGE = 'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926'
MODEL = '/home/wjg/models/qwen3.5-9b'
NAMES = ['prepared_inputs.json', 'cpu_candidate_prefix_mapping.json', 'cpu_gate.json',
         'cold_reference_C_A.json', 'cold_reference_C_B.json', 'cold_reference_C_C.json',
         'cold_reference_C_D.json', 't2g1_plans.json', 'workloads_case_C.json', 'source_identity.json']
SOURCE = ROOT / 'evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def readcmd(args):
    return subprocess.check_output(args, text=True).strip()


def resource_gate(cleanup=False):
    assert shutil.disk_usage(ROOT).free > 20*1024**3, '根盘不足20GiB'
    output = readcmd(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    row = next(line for line in output.splitlines() if line.split(',')[0].strip() == GPU)
    assert int(row.split(',')[1]) < 100, '目标GPU显存非空闲：'+row
    if not cleanup:
        assert int(row.split(',')[2]) == 0, '启动前GPU存在执行活动：'+row
    processes = readcmd(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    assert not any(line.split(',')[0].strip() == GPU for line in processes.splitlines()), '目标GPU有计算进程'
    containers = readcmd(['docker', 'ps', '--format', '{{.Names}}'])
    assert not any(line.startswith(('flowstate-opt-', 'flowstate-p04-')) for line in containers.splitlines()), '存在并发实验容器'
    return {'gpu': output, 'processes': processes, 'containers': containers,
            'disk_free_bytes': shutil.disk_usage(ROOT).free, 'time_ns': time.time_ns()}


def freeze():
    files = list(PKG.glob('*.py')) + [PKG/'implementation_changes.md']
    code = {str(p): sha(p) for p in files}
    inputs = {name: sha(SOURCE/name) for name in NAMES}
    schedule = [{'arm': 'smoke_fs', 'policy': 'FlowState-Greedy', 'mode': 'batched', 'verify': True, 'round': 0, 'phase': 'smoke'}]
    for r in range(1, 6):
        for mode in (['original', 'batched'] if r % 2 else ['batched', 'original']):
            schedule.append({'arm': f'p{r}_{mode}_fs', 'policy': 'FlowState-Greedy', 'mode': mode,
                             'verify': False, 'round': r, 'phase': 'paired'})
    orders = [['fs', 'mc', 'kv'], ['mc', 'kv', 'fs'], ['kv', 'fs', 'mc'], ['fs', 'kv', 'mc'], ['mc', 'fs', 'kv']]
    policies = {'fs': 'FlowState-Greedy', 'mc': 'Marconi-Evict', 'kv': 'KVFlow-inspired'}
    for r, order in enumerate(orders, 1):
        for suffix in order:
            schedule.append({'arm': f't{r}_{suffix}', 'policy': policies[suffix], 'mode': 'batched',
                             'verify': False, 'round': r, 'phase': 'three_policy'})
    save(PKG/'execution_lock.json', {'code': code, 'inputs': inputs, 'gpu_uuid': GPU, 'image': IMAGE,
                                    'schedule': schedule, 'phase1_sha256': sha(PKG/'phase1_analysis.md'),
                                    'conditional_three_policy': '五轮配对控制差中位数严格降低且全部通过才执行'})


def check_lock(lock):
    for name, digest in lock['code'].items():
        assert sha(Path(name)) == digest, '新实现源码身份变化：'+name
    for name, digest in lock['inputs'].items():
        assert sha(SOURCE/name) == digest, '冻结输入变化：'+name
    for name, digest in json.loads((PKG/'original_code_manifest.json').read_text()).items():
        assert sha(Path(name)) == digest, '原版对照源码变化：'+name


def run(spec, lock):
    check_lock(lock)
    before = resource_gate()
    ARMS.mkdir(exist_ok=True)
    directory = ARMS/spec['arm']
    directory.mkdir(exist_ok=False)
    for name in NAMES:
        shutil.copyfile(SOURCE/name, directory/name)
    save(directory/'experiment_identity.json', {'spec': spec, 'code': lock['code'], 'input_sha256': lock['inputs'], 'resources_before': before})
    name = 'flowstate-opt-'+spec['arm'].replace('_', '-')
    cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device='+GPU,
           '--network', 'host', '--shm-size', '32g', '--mount', f'type=bind,src={ROOT},dst={ROOT},readonly',
           '--mount', f'type=bind,src={MODEL},dst={MODEL},readonly', '--mount', f'type=bind,src={directory},dst=/b5out']
    env = {'CUDA_VISIBLE_DEVICES': GPU, 'EXPECTED_GPU_UUID': GPU, 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'FLOWSTATE_STEP5D_PORT': '50087', 'B5D4_RUN': 'run', 'B5D4_ARM': 'strict',
           'PILOT_MODE': 'strict_warm', 'PILOT_POLICY': spec['policy'], 'FS_MEASURE_MODE': '1',
           'FS_OPT_MODE': spec['mode'], 'FS_OPT_VERIFY': '1' if spec['verify'] else '0',
           'TRITON_CACHE_DIR': '/b5out/triton_cache',
           'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
    for key, value in env.items():
        cmd += ['--env', key+'='+value]
    cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE, '-B', str(PKG/'device_entry.py')]
    save(directory/'command.json', cmd)
    print('启动', spec, flush=True)
    start = time.monotonic()
    rc = None
    try:
        with (directory/'runtime.log').open('x') as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            rc = p.wait(timeout=5400)
    except BaseException:
        subprocess.run(['docker', 'stop', '--time', '30', name], capture_output=True)
        raise
    finally:
        save(directory/'host_exit.json', {'returncode': rc, 'wall_s': time.monotonic()-start,
                                         'disk_free_bytes': shutil.disk_usage(ROOT).free})
    after = resource_gate(cleanup=True)
    save(directory/'gpu_cleanup.json', after)
    summary = directory/'run/summary.json'
    assert rc == 0 and summary.exists(), '实验臂未完整成功：'+spec['arm']
    result = json.loads(summary.read_text())
    assert result['status'] == 'PASS' and result['oracle_match'] and result['cycle']['identity_exact'], '门禁未通过'
    assert not (directory/'run/failure.json').exists() and not (directory/'run/failure_after_summary.json').exists(), '存在运行失败'
    assert json.loads((directory/'run/engine_cleanup.json').read_text())['shutdown_returned'], '引擎清理失败'
    if spec['verify']:
        rec = json.loads((directory/'run/reconciliation.json').read_text())
        checks = [m for m in rec['optimization_metrics'] if m['kind'] == 'equivalence']
        assert len(checks) >= 6 and all(m['verified'] for m in checks), '缺少真实GPU摘要等价证据'
    with (PKG/'ledger.jsonl').open('a') as f:
        f.write(json.dumps({'spec': spec, 'status': 'PASS', 'summary_sha256': sha(summary)})+'\n')
        f.flush()
        os.fsync(f.fileno())
    print('完成', spec['arm'], result['cycle']['control_cycle_ms'], result['cycle']['cycle_ms_full_incl_warm'], flush=True)


def main():
    phase = sys.argv[1]
    assert phase in ['paired', 'three_policy']
    lock = json.loads((PKG/'execution_lock.json').read_text())
    amendment = json.loads((PKG/'execution_amendment.json').read_text())
    assert sha(Path(__file__)) == amendment['execute_v2_sha256'], '收尾门禁修订身份不符'
    check_lock(lock)
    assert json.loads((ARMS/'smoke_fs/run/summary.json').read_text())['status'] == 'PASS'
    if phase == 'three_policy':
        differences = []
        for r in range(1, 6):
            a = json.loads((ARMS/f'p{r}_original_fs/run/summary.json').read_text())
            b = json.loads((ARMS/f'p{r}_batched_fs/run/summary.json').read_text())
            assert a['status'] == b['status'] == 'PASS'
            differences.append(b['cycle']['control_cycle_ms']-a['cycle']['control_cycle_ms'])
        assert statistics.median(differences) < 0, '控制无改善，按协议不执行三策略追加实验'
    for spec in lock['schedule']:
        if spec['phase'] != phase:
            continue
        if spec['arm'] in amendment['completed_arms_not_rerun']:
            directory = ARMS/spec['arm']
            summary = directory/'run/summary.json'
            assert sha(summary) == amendment['completed_arms_not_rerun'][spec['arm']]
            s = json.loads(summary.read_text())
            assert s['status'] == 'PASS' and s['oracle_match'] and s['cycle']['identity_exact']
            assert not (directory/'run/failure.json').exists()
            assert not (directory/'run/failure_after_summary.json').exists()
            assert json.loads((directory/'host_exit.json').read_text())['returncode'] == 0
            assert json.loads((directory/'run/engine_cleanup.json').read_text())['shutdown_returned']
            print('保留已完成臂，不重跑', spec['arm'], flush=True)
            continue
        run(spec, lock)


if __name__ == '__main__':
    main()
