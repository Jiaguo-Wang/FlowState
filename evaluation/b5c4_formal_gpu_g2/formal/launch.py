"""G2-B 正式单臂受控启动（宿主机执行）：按 schedule.json 顺序强制执行，前序未完成不得启动。

用法：python3 launch.py 轮次(1-5) Exact|Greedy A|B|C|D [重试序号]
（RUN=r{轮}_{策略}_{字母}，重试追加 _r{n}，失败证据保留不覆盖）。
生命周期与门禁臂相同（8 历史重放→在线求解→事后 Oracle→形状预热 B→正式流式请求→
全量合同）；正式值进入统计分析。每臂独立 Docker/GPU 生命周期。
"""
import json, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/b5c4_formal_gpu_g2/formal'
PKG = ROOT / 'evaluation/b5c4_formal_gpu_g2/pkg_g2'
assert len(sys.argv) >= 4, '用法：launch.py 轮次 Exact|Greedy A|B|C|D [重试序号]'
ROUND, SOLVER, LETTER = sys.argv[1], sys.argv[2], sys.argv[3].upper()
assert ROUND.isdigit() and 1 <= int(ROUND) <= 5 and SOLVER in ('Exact', 'Greedy') and LETTER in 'ABCD'
RUN = 'r%s_%s_%s' % (ROUND, SOLVER.lower(), LETTER.lower())
if len(sys.argv) > 4:
    RUN += '_r' + sys.argv[4]
PORT, GPU, TIMEOUT_S = 49993, 0, 2400


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def write(name, x):
    p = OUT / name
    assert not p.exists(), '拒绝覆盖：' + name
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2))


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arm_terminal(base):
    """该基础 RUN 是否已有终态：存在 PASS 摘要（含重试），或存在失败终证（failure/进程记录）。"""
    for candidate in [base] + [base + '_r%d' % i for i in range(1, 20)]:
        d = OUT / candidate
        if (d / 'summary.json').exists():
            return True, json.loads((d / 'summary.json').read_text())['status'] == 'PASS'
        if (d / 'failure.json').exists() and (OUT / (candidate + '_process.json')).exists():
            return True, False
    return False, None


def main():
    schedule = json.loads((OUT / 'schedule.json').read_text())['schedule']
    mine = next(a for a in schedule if a['run'] == 'r%s_%s_%s' % (ROUND, SOLVER.lower(), LETTER.lower()))
    for prev in schedule[:schedule.index(mine)]:  # 前序臂必须全部终态（PASS 或已留证的失败）
        terminal, ok = arm_terminal(prev['run'])
        assert terminal, '前序臂未完成，禁止跳序启动：%s（当前 %s）' % (prev['run'], RUN)
    assert not (OUT / (RUN + '_execution_started.json')).exists(), '禁止重跑已启动验证：' + RUN
    source = json.loads((OUT / 'source_identity.json').read_text())
    image = json.loads(run(['docker', 'image', 'inspect', source['image_id']]))[0]
    assert image['Id'] == source['image_id']
    lock = {str(p): sha(p) for p in list(OUT.glob('*.py')) + list(OUT.glob('*.json'))}
    lock.update({str(p): sha(p) for p in PKG.glob('*.py')})
    lock.update(source['hybrid_entry_sources'])
    write('execution_lock_' + RUN + '.json', {'files': lock, 'image': image['Id'],
           'runner': 'evaluation.b5c4_formal_gpu_g2.pkg_g2.runner', 'scenario': 'G2-B_formal',
           'solver': SOLVER, 'pending': 'B5C2_' + LETTER + '_TURN_003', 'phase': 'formal', 'round': int(ROUND)})
    write(RUN + '_execution_started.json', {'unix_ns': time.time_ns(), 'run': RUN,
           'schedule_index': schedule.index(mine), 'solver_first_in_round': mine['solver_first_in_round']})
    for p, h in lock.items():
        assert sha(p) == h, '锁定文件已变化：' + p
    apps = run(['nvidia-smi', '-i', str(GPU), '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
    assert len(apps.strip().splitlines()) == 1, 'GPU有其他任务'
    gpu = run(['nvidia-smi', '-i', str(GPU), '--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit', '--format=csv'])
    assert 'H100 PCIe' in gpu
    assert int(run(['nvidia-smi', '-i', str(GPU), '--query-gpu=memory.used', '--format=csv,noheader,nounits']).strip()) < 1000
    container = 'flowstate-g2-formal-' + RUN.replace('_', '-')
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
           'G2_PHASE': 'formal', 'G2_ROUND': ROUND, 'FS_MEASURE_MODE': '1',
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
    print('G2-B 正式臂完成', RUN, flush=True)


if __name__ == '__main__':
    main()
