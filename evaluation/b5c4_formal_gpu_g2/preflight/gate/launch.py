"""G2-A 门禁单臂受控启动（宿主机执行）：2 求解臂 × 4 Pending = 8 臂。

用法：python3 launch.py Exact|Greedy A|B|C|D [重试序号]（RUN={solver}_{字母}，重试追加 _r{n}，
失败证据保留不覆盖）。每臂独立 Docker/GPU 生命周期：8 条冻结历史单轮重放→唯一 b5d4_apply
（G2 控制路径在线求解：Exact=生产 plan_strict 自动；Greedy=生产 Marginal Greedy 适配）→
事后 e3 参照 Oracle→形状预热 B（L=len(pt)−E，E 处分叉）→正式流式请求（测量模式）→
请求后 b5d4_state 全量合同（Prefill=len(pt)−E、FA命中==前沿、Cold 逐 Token、审计==2、
请求窗口零新增编译、TTFT 六段分解闭合）。已启动即拒绝重跑；GPU 须空闲。
"""
import json, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight/gate'
PKG = ROOT / 'evaluation/b5c4_formal_gpu_g2/pkg_g2'
assert len(sys.argv) >= 3, '用法：launch.py Exact|Greedy A|B|C|D [重试序号]'
SOLVER, LETTER = sys.argv[1], sys.argv[2].upper()
assert SOLVER in ('Exact', 'Greedy') and LETTER in 'ABCD'
RUN = SOLVER.lower() + '_' + LETTER.lower()
if len(sys.argv) > 3:
    RUN += '_r' + sys.argv[3]
PORT, GPU, TIMEOUT_S = 49992, 0, 2400


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def write(name, x):
    p = OUT / name
    assert not p.exists(), '拒绝覆盖：' + name
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2))


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (OUT / (RUN + '_execution_started.json')).exists(), '禁止重跑已启动验证：' + RUN
    source = json.loads((OUT / 'source_identity.json').read_text())
    image = json.loads(run(['docker', 'image', 'inspect', source['image_id']]))[0]
    assert image['Id'] == source['image_id']
    lock = {str(p): sha(p) for p in list(OUT.glob('*.py')) + list(OUT.glob('*.json'))}
    lock.update({str(p): sha(p) for p in PKG.glob('*.py')})
    lock.update(source['hybrid_entry_sources'])
    write('execution_lock_' + RUN + '.json', {'files': lock, 'image': image['Id'],
           'runner': 'evaluation.b5c4_formal_gpu_g2.pkg_g2.runner', 'scenario': 'G2-A_gate',
           'solver': SOLVER, 'pending': 'B5C2_' + LETTER + '_TURN_003', 'phase': 'gate', 'round': 0})
    write(RUN + '_execution_started.json', {'unix_ns': time.time_ns(), 'run': RUN})
    for p, h in lock.items():
        assert sha(p) == h, '锁定文件已变化：' + p
    apps = run(['nvidia-smi', '-i', str(GPU), '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
    assert len(apps.strip().splitlines()) == 1, 'GPU有其他任务'
    gpu = run(['nvidia-smi', '-i', str(GPU), '--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit', '--format=csv'])
    assert 'H100 PCIe' in gpu
    assert int(run(['nvidia-smi', '-i', str(GPU), '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip()) < 1000
    container = 'flowstate-g2-gate-' + RUN.replace('_', '-')
    assert not run(['docker', 'ps', '-a', '--filter', 'name=' + container, '--format', '{{.Names}}']).strip()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', PORT))
    write(RUN + '_gpu_before.json', {'gpu': gpu, 'apps': apps, 'unix_ns': time.time_ns()})
    cmd = ['docker', 'run', '--rm', '--pull', 'never', '--name', container, '--gpus', 'device=%d' % GPU, '--network', 'host', '--shm-size', '32g']
    for src, dst, ro in [(ROOT, ROOT, True), (Path('/home/wjg/models/qwen3.5-9b'), Path('/home/wjg/models/qwen3.5-9b'), True), (OUT, Path('/b5out'), False)]:
        cmd += ['--mount', 'type=bind,src=%s,dst=%s%s' % (src, dst, ',readonly' if ro else '')]
    env = {'CUDA_VISIBLE_DEVICES': str(GPU), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'FLOWSTATE_STEP5D_PORT': str(PORT), 'TRITON_CACHE_DIR': '/b5out/triton_cache_' + RUN,
           'B5D4_ARM': 'strict', 'B5D4_RUN': RUN, 'G2_SOLVER': SOLVER, 'G2_PENDING': 'B5C2_' + LETTER + '_TURN_003',
           'G2_PHASE': 'gate', 'G2_ROUND': '0', 'FS_MEASURE_MODE': '1',
           'PYTHONPATH': '%s:%s/tests/runtime:%s/motivation/artifacts/wp3b_gate_20260820' % (ROOT, ROOT, ROOT)}
    for k, v in env.items():
        cmd += ['--env', k + '=' + v]
    cmd += ['--workdir', str(ROOT), '--entrypoint', 'python3', image['Id'], '-B', '-m', 'evaluation.b5c4_formal_gpu_g2.pkg_g2.runner']
    write(RUN + '_command.json', cmd)
    print('启动', RUN, flush=True)
    r = None
    try:
        with (OUT / (RUN + '.runtime.log')).open('x') as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
        write(RUN + '_process.json', {'returncode': r.returncode})
    finally:
        removed = subprocess.run(['docker', 'rm', '-f', container], capture_output=True, text=True)
        obs = []
        for _ in range(3):
            time.sleep(3)
            mem = int(run(['nvidia-smi', '-i', str(GPU), '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip())
            apps = run(['nvidia-smi', '-i', str(GPU), '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            gpu_after = run(['nvidia-smi', '-i', str(GPU), '--query-gpu=clocks.sm,power.draw', '--format=csv,noheader'])
            obs.append({'memory_mib': mem, 'processes': apps, 'clocks_power': gpu_after.strip()})
        own = run(['docker', 'ps', '-a', '--filter', 'name=' + container, '--format', '{{.Names}}']).strip()
        cleanup = {'status': 'PASS' if not own and all(x['memory_mib'] < 100 and len(x['processes'].strip().splitlines()) == 1 for x in obs) else 'FAIL',
                   'observations': obs, 'own_container': own, 'remove_stderr': removed.stderr}
        write(RUN + '_cleanup.json', cleanup)
    assert cleanup['status'] == 'PASS' and r and r.returncode == 0, '执行或清理失败'
    assert json.loads((OUT / RUN / 'summary.json').read_text())['status'] == 'PASS', '缺少完整正确性摘要'
    print('G2-A 门禁臂完成', RUN, flush=True)


if __name__ == '__main__':
    main()
