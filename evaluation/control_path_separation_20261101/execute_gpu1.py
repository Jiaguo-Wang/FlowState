"""隔离批次执行：正确性Smoke、最小配对、五轮续接与条件性三策略验证。"""
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
ARMS = PKG/'arms'
GPU = 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'
IMAGE = 'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926'
MODEL = '/home/wjg/models/qwen3.5-9b'
SOURCE = ROOT/'evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs'
NAMES = ['prepared_inputs.json', 'cpu_candidate_prefix_mapping.json', 'cpu_gate.json',
         'cold_reference_C_A.json', 'cold_reference_C_B.json', 'cold_reference_C_C.json',
         'cold_reference_C_D.json', 't2g1_plans.json', 'workloads_case_C.json', 'source_identity.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def query(args):
    return subprocess.check_output(args, text=True).strip()


def resources():
    free = shutil.disk_usage(ROOT).free
    assert free > 20*1024**3, '磁盘余量不足20GiB'
    gpu = query(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    row = next(line for line in gpu.splitlines() if line.split(',')[0].strip() == GPU)
    assert int(row.split(',')[1]) < 100, '目标GPU显存被占用：'+row
    processes = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    assert not any(line.split(',')[0].strip() == GPU for line in processes.splitlines()), '目标GPU存在计算进程'
    containers = query(['docker', 'ps', '--format', '{{.Names}}'])
    assert not any(name.startswith(('flowstate-sep-', 'flowstate-p04-')) for name in containers.splitlines()), '存在相关实验容器'
    return {'gpu': gpu, 'compute_processes': processes, 'containers': containers,
            'free_bytes': free, 'time_ns': time.time_ns(),
            'utilization_semantics': '过去采样窗口，仅记录；独占门禁为目标卡显存与计算进程'}


def freeze():
    files = list(PKG.glob('*.py')) + [PKG/n for n in ['phase1_analysis.md', 'safety_equivalence.md', 'implementation_changes.md']]
    schedule = [{'arm': 'smoke_serving_fs', 'mode': 'serving', 'policy': 'FlowState-Greedy', 'round': 0, 'phase': 'smoke', 'verify': True}]
    for r in range(1, 6):
        for mode in (['original', 'serving'] if r % 2 else ['serving', 'original']):
            schedule.append({'arm': f'p{r}_{mode}_fs', 'mode': mode, 'policy': 'FlowState-Greedy',
                             'round': r, 'phase': 'paired', 'verify': False})
    names = {'fs': 'FlowState-Greedy', 'mc': 'Marconi-Evict', 'kv': 'KVFlow-inspired'}
    orders = [['fs', 'mc', 'kv'], ['mc', 'kv', 'fs'], ['kv', 'fs', 'mc'], ['fs', 'kv', 'mc'], ['mc', 'fs', 'kv']]
    for r, order in enumerate(orders, 1):
        for tag in order:
            schedule.append({'arm': f't{r}_{tag}', 'mode': 'serving', 'policy': names[tag],
                             'round': r, 'phase': 'three_policy', 'verify': False})
    save(PKG/'execution_lock_gpu1.json', {'image': IMAGE, 'model': MODEL, 'gpu_uuid': GPU,
         'code': {str(f): sha(f) for f in files}, 'input_sha256': {n: sha(SOURCE/n) for n in NAMES},
         'schedule': schedule, 'first_pair_gate': '控制差<0且Full差<=0，全部正确性通过',
         'five_pair_gate': '五轮控制配对差中位数<0且Full配对差中位数<=0，全部正确性通过',
         'T0_T3_T4': '原始定义不变', 'T5': '审计、重复展示报告和原全部正确性及Oracle完成，不含summary写盘/Engine退出'})


def check_identity(lock):
    for name, digest in lock['code'].items():
        assert sha(Path(name)) == digest, '本轮实现身份漂移：'+name
    for name, digest in lock['input_sha256'].items():
        assert sha(SOURCE/name) == digest, '原冻结输入漂移：'+name
    original = json.loads((PKG/'original_source_manifest.json').read_text())['files']
    for name, digest in original.items():
        assert sha(Path(name)) == digest, '原实验/生产代码身份漂移：'+name


def command(spec, directory):
    name = 'flowstate-sep-'+spec['arm'].replace('_', '-')
    cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--gpus', 'device='+GPU,
           '--network', 'host', '--shm-size', '32g', '--mount', f'type=bind,src={ROOT},dst={ROOT},readonly',
           '--mount', f'type=bind,src={MODEL},dst={MODEL},readonly', '--mount', f'type=bind,src={directory},dst=/b5out']
    env = {'CUDA_VISIBLE_DEVICES': GPU, 'EXPECTED_GPU_UUID': GPU, 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'FLOWSTATE_STEP5D_PORT': '50087', 'B5D4_RUN': 'run', 'B5D4_ARM': 'strict',
           'PILOT_MODE': 'strict_warm', 'PILOT_POLICY': spec['policy'], 'FS_MEASURE_MODE': '1',
           'FS_SEPARATION_MODE': spec['mode'], 'FS_SEPARATION_VERIFY': '1' if spec['verify'] else '0',
           'TRITON_CACHE_DIR': '/b5out/triton_cache',
           'PYTHONPATH': f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
    for key, value in env.items():
        cmd += ['--env', key+'='+value]
    return cmd + ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE, '-B', str(PKG/'device_entry.py')], name


def validate_arm(spec):
    directory = ARMS/spec['arm']
    assert json.loads((directory/'host_exit.json').read_text())['returncode'] == 0, '宿主进程异常'
    assert (directory/'gpu_cleanup.json').exists(), '资源清理证据缺失'
    run = directory/'run'
    assert not (run/'failure.json').exists() and not (run/'failure_after_summary.json').exists(), '存在执行失败'
    s = json.loads((run/'summary.json').read_text())
    assert s['status'] == 'PASS' and s['oracle_match'] and s['cycle']['identity_exact'], '原门禁失败'
    assert json.loads((run/'engine_cleanup.json').read_text())['shutdown_returned'], 'Engine清理失败'
    assert s['separation']['mode'] == spec['mode'] and s['separation']['t5_ns'] >= s['cycle']['t4_ns'], '模式/审计时钟错误'
    if spec['mode'] == 'serving':
        ts = s['separation']['transfers']
        assert len(ts) == 1 and ts[0]['digest_verified'], '完整响应校验缺失'
        assert s['separation']['report_timing']['start_ns'] >= s['cycle']['t4_ns'], '展示报告未移出窗口'
        if spec['verify']:
            assert ts[0]['encoding']['equivalence_verified'], '与原线协议逐字段对照缺失'
    return s


def run(spec, lock):
    check_identity(lock)
    before = resources()
    directory = ARMS/spec['arm']
    directory.mkdir(parents=True, exist_ok=False)
    for name in NAMES:
        shutil.copyfile(SOURCE/name, directory/name)
    save(directory/'experiment_identity.json', {'spec': spec, 'code': lock['code'], 'input_sha256': lock['input_sha256'], 'resources_before': before})
    cmd, name = command(spec, directory)
    save(directory/'command.json', cmd)
    print('启动', spec, flush=True)
    start = time.monotonic()
    rc = None
    try:
        with (directory/'runtime.log').open('x') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            rc = process.wait(timeout=5400)
    except BaseException:
        subprocess.run(['docker', 'stop', '--time', '30', name], capture_output=True)
        raise
    finally:
        save(directory/'host_exit.json', {'returncode': rc, 'wall_s': time.monotonic()-start,
                                         'free_bytes': shutil.disk_usage(ROOT).free})
    after = resources()
    save(directory/'gpu_cleanup.json', after)
    s = validate_arm(spec)
    with (PKG/'ledger.jsonl').open('a') as f:
        f.write(json.dumps({'spec': spec, 'status': 'PASS', 'summary_sha256': sha(directory/'run/summary.json')})+'\n')
        f.flush()
        os.fsync(f.fileno())
    print('完成', spec['arm'], 'Control', s['cycle']['control_cycle_ms'], 'Full', s['cycle']['cycle_ms_full_incl_warm'],
          'Audit', s['separation']['audit_inclusive_ms'], flush=True)


def paired_gate(lock, count):
    values = []
    for r in range(1, count+1):
        original = next(s for s in lock['schedule'] if s['arm'] == f'p{r}_original_fs')
        serving = next(s for s in lock['schedule'] if s['arm'] == f'p{r}_serving_fs')
        a, b = validate_arm(original), validate_arm(serving)
        assert a['selection']['selected_cpu'] == b['selection']['selected_cpu'], '同Epoch选集变化'
        assert a['selection']['plan_recovery_cost_ms'] == b['selection']['plan_recovery_cost_ms'], 'J变化'
        for key in ['E', 'T', 'G', 'prefill_token_count']:
            assert [p[key] for p in a['pending_rows']] == [p[key] for p in b['pending_rows']], '恢复合同变化'
        values.append({'round': r, 'control_delta_ms': b['cycle']['control_cycle_ms']-a['cycle']['control_cycle_ms'],
                       'full_delta_ms': b['cycle']['cycle_ms_full_incl_warm']-a['cycle']['cycle_ms_full_incl_warm'],
                       'audit_delta_ms': b['separation']['audit_inclusive_ms']-a['separation']['audit_inclusive_ms']})
    control = statistics.median(v['control_delta_ms'] for v in values)
    full = statistics.median(v['full_delta_ms'] for v in values)
    return {'count': count, 'values': values, 'median_control_delta_ms': control,
            'median_full_delta_ms': full, 'pass': control < 0 and full <= 0}


def main():
    action = sys.argv[1]
    if action == 'freeze':
        freeze()
        return
    lock = json.loads((PKG/'execution_lock_gpu1.json').read_text())
    check_identity(lock)
    if action == 'smoke':
        run(lock['schedule'][0], lock)
        return
    validate_arm(lock['schedule'][0])
    assert action in ['first_pair', 'remaining', 'three_policy']
    if action == 'first_pair':
        chosen = [s for s in lock['schedule'] if s['phase'] == 'paired' and s['round'] == 1]
    elif action == 'remaining':
        assert paired_gate(lock, 1)['pass'], '最小配对未通过，不执行剩余轮次'
        chosen = [s for s in lock['schedule'] if s['phase'] == 'paired' and s['round'] > 1]
    else:
        assert paired_gate(lock, 5)['pass'], '五轮门禁未通过，不追加三策略实验'
        chosen = [s for s in lock['schedule'] if s['phase'] == 'three_policy']
    for spec in chosen:
        run(spec, lock)
    if action in ['first_pair', 'remaining']:
        count = 1 if action == 'first_pair' else 5
        verdict = paired_gate(lock, count)
        save(PKG/f'pair_gate_{count}.json', verdict)
        print('配对门禁', verdict, flush=True)


if __name__ == '__main__':
    main()
