"""P0-4 五轮矩阵·定卡串行启动器（宿主机执行；复用 launch.py 的 docker_cmd，与策略实现隔离）。

与 launch.py（r1 Pilot 启动器）的差异均为编排层，实验语义零变化：
- 固定物理 GPU（GPU-bfff4d31…，与 r1 三臂同卡）：每臂启动前实查该卡空闲（<100MiB 且无
  计算进程）；被他人占用 → 立即停止等待，绝不自动换卡、绝不杀死他人进程；
- 按 prepare_round.ROUND_ORDER（任务书固定拉丁方）串行执行，一臂非 PASS 即终止该轮
  （fail-loud：不静默重试、不跳过、不覆盖历史结果）；
- 每臂完成后向追加式总账 arms/five_round_ledger.jsonl 追加一行（只增不改，fsync 落盘），
  失败臂同样记账（status 非 PASS + 错误摘要）；
- 逐轮写 arms/arms_status_m{R}.json（一次性，拒绝覆盖），绝不触碰 arms_status.json；
- 每轮发射前门禁（§任务书四.4）：盘余>20G、定卡空闲、无 flowstate-p04-* 残留容器与
  launcher 残留进程、当前源码身份==冻结身份、该轮三臂目录已装配且 run/ 为空。

用法（仓库根目录执行）：
  python3 -m evaluation.p0_control_cycle_gpu.launch_round 2 --dry-run   # 演习：全门禁+命令构造，不启动容器
  python3 -m evaluation.p0_control_cycle_gpu.launch_round 2             # 正式执行第 2 轮
  python3 -m evaluation.p0_control_cycle_gpu.launch_round --ledger-selftest  # 总账追加/失败记录自检
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import launch, prepare
from .prepare_round import ROUND_ORDER, arm_name

PINNED_GPU = 'GPU-bfff4d31-0b94-7de3-9946-f593437ae566'   # r1 三臂同卡；被占即停等，不换卡
# 冻结源码身份（preflight §2；每轮发射前与当前文件完整哈希比对）
FROZEN_IDENTITY = {
    'runner.py': 'b6cee2d3ffa04e8f85e9658bf166a37cfb063ad21983c9488b37fb8b4eaf9f84',
    'observation.py': '7c1dcefcb5b456144a70908d7b48cba2b815584a5f0332c3e9f2586445392837',
}
ARMS = launch.ARMS
LEDGER = ARMS / 'five_round_ledger.jsonl'
SELFTEST_LEDGER = ARMS / 'five_round_ledger_selftest.jsonl'
TIMEOUT_S = 5400   # 单臂墙钟上限（与 launch.run_arm 一致）


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def disk_gate():
    free = launch.df_root()
    assert free > 20, '根分区余量不足 20G（当前 %dG），暂停' % free
    return free


def gpu_gate():
    """定卡门禁：实查 PINNED_GPU 存在、显存<100MiB、定卡无计算进程；不满足即失败（停等）。
    他人在另一卡上的进程与本单卡实验无干扰、不阻断（协议语义=定卡空闲；R3 中断教训：
    首版沿用 launch.py 的任意卡断言，严于协议，已按定卡过滤收窄）。"""
    out = sh("nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits")
    gpus = {line.split(',')[0].strip(): int(line.split(',')[1].strip()) for line in out.splitlines()}
    assert PINNED_GPU in gpus, '定卡不在位：' + repr(list(gpus))
    assert gpus[PINNED_GPU] < 100, '定卡被占用（%dMiB）：停止等待，不换卡、不杀他人进程' % gpus[PINNED_GPU]
    procs = sh("nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader")
    mine = [l for l in procs.splitlines() if l.strip() and l.split(',')[0].strip() == PINNED_GPU]
    assert not mine, '定卡上存在计算进程：' + '; '.join(mine)


def _ancestors():
    """当前进程的全部祖先 PID（排除调用链自身，杜绝 pgrep -f 自匹配误报——任务A已踩坑）。"""
    pids, p = set(), os.getppid()
    while p > 1:
        pids.add(p)
        try:
            with open('/proc/%d/status' % p) as f:
                for line in f:
                    if line.startswith('PPid:'):
                        p = int(line.split()[1])
                        break
        except OSError:
            break
    return pids


def residue_gate():
    """残留容器与 launcher 进程检查（发现即失败，绝不代杀；排除自身与祖先调用链）。"""
    containers = sh("docker ps -a --format '{{.Names}}' | grep '^flowstate-p04-' || true")
    assert not containers, '存在 p04 残留容器：' + containers
    family = _ancestors() | {os.getpid()}
    residue = []
    for pid_dir in Path('/proc').glob('[0-9]*'):
        pid = int(pid_dir.name)
        if pid in family:
            continue
        try:
            cmdline = (pid_dir / 'cmdline').read_bytes().decode(errors='replace')
        except OSError:
            continue
        if 'p0_control_cycle_gpu.launch' in cmdline:
            residue.append('%d %s' % (pid, cmdline.replace('\0', ' ')[:120]))
    assert not residue, '存在 launcher 残留进程：' + '; '.join(residue)


def identity_gate():
    """当前源码身份 == 冻结身份（完整 sha256，杜绝缩写误判）。"""
    for name, frozen in FROZEN_IDENTITY.items():
        cur = prepare.sha(Path(prepare.PKG) / name)
        assert cur == frozen, '源码身份漂移 %s：%s != %s' % (name, cur, frozen)


def arm_gate(arm):
    """该轮臂目录门禁：已装配（prepared_inputs 在位）且从未运行（run/ 不存在）。"""
    d = ARMS / arm
    assert (d / 'prepared_inputs.json').exists(), '臂目录未装配：' + str(d)
    assert not (d / 'run').exists(), '臂目录已有 run/（拒绝覆盖历史结果）：' + str(d / 'run')


def append_ledger(path, entry):
    """总账追加一行（只增不改）：JSON 行 + flush + fsync。"""
    with path.open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def ledger_entry(round_no, order_idx, policy, arm, cleanup, summary=None, error=None):
    """由臂 cleanup/结论文件构造总账行（真实完成与失败均记账；字段口径见 preflight §4.3）。"""
    entry = {
        'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'round': round_no, 'order_idx': order_idx, 'policy': policy, 'arm': arm,
        'gpu_uuid': PINNED_GPU, 'image': launch.IMAGE,
        'runner_sha256': FROZEN_IDENTITY['runner.py'],
        'observation_sha256': FROZEN_IDENTITY['observation.py'],
        'returncode': cleanup['returncode'], 'status': cleanup['status'],
        'wall_s': round(cleanup['wall_s'], 1), 'disk_free_gb': cleanup['disk_free_gb_after'],
        'control_cycle_ms': None, 'cycle_ms_full_incl_warm': None, 'request_exec_total_ms': None,
        'sum_ttft_ms': None, 'sum_cuda_prefill_ms': None, 'J_ms': None, 'selection_ids': None,
        'evidence_dir': str((ARMS / arm).relative_to(launch.ROOT)),
    }
    if summary is not None:   # 仅 PASS 臂有 summary：提取双口径计时与恢复指标
        cyc, rows, sel = summary['cycle'], summary['pending_rows'], summary['selection']
        entry.update({
            'control_cycle_ms': cyc['control_cycle_ms'],
            'cycle_ms_full_incl_warm': cyc['cycle_ms_full_incl_warm'],
            'request_exec_total_ms': cyc['request_exec_total_ms'],
            'sum_ttft_ms': round(sum(r['ttft_ms'] for r in rows), 3),
            'sum_cuda_prefill_ms': round(sum(r['cuda_prefill_ms'] for r in rows), 3),
            'J_ms': sel['plan_recovery_cost_ms'], 'selection_ids': sel['selected_cpu'],
        })
    else:
        entry['error'] = error   # 失败臂：错误摘要随账可溯（完整证据在臂目录 failure 文件）
    return entry


def run_round_arm(round_no, order_idx, policy):
    """执行单臂（docker 生命周期与监控语义同 launch.run_arm；定卡 + 总账记账为五轮扩展）。"""
    arm = arm_name(policy, round_no)
    gpu_gate()                                    # 每臂启动前重新实查定卡
    disk_before = launch.df_root()
    name = 'flowstate-p04-%s' % arm
    log_path = ARMS / (arm + '.runtime.log')
    cmd = launch.docker_cmd(policy, arm, PINNED_GPU, name)
    (ARMS / (arm + '_command.json')).write_text(json.dumps(cmd, indent=2))
    print('启动', 'R%d' % round_no, order_idx, policy, 'GPU', PINNED_GPU, '磁盘余', disk_before, 'G', flush=True)
    run_dir = ARMS / arm / 'run'
    with log_path.open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        (ARMS / (arm + '_process.json')).write_text(json.dumps(
            {'pid': proc.pid, 'container': name, 'gpu_uuid': PINNED_GPU,
             'disk_free_gb_before': disk_before, 'round': round_no, 'order_idx': order_idx}, indent=2))
        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                break
            assert time.monotonic() - t0 < TIMEOUT_S, '臂超时（%ds）：%s' % (TIMEOUT_S, arm)
            done = (run_dir / 'summary.json').exists() or (run_dir / 'failure.json').exists()
            time.sleep(30)
            if done and proc.poll() is None:
                # runner 已落盘结论，等待容器自然退出（engine shutdown + 清理文件）
                if time.monotonic() - t0 > TIMEOUT_S - 600:
                    break
    rc = proc.poll()
    if rc is None:
        proc.terminate()
        rc = proc.wait(30)
    disk_after = launch.df_root()
    status = 'PASS' if (run_dir / 'summary.json').exists() else ('BLOCKED' if rc != 0 else 'UNKNOWN')
    cleanup = {'returncode': rc, 'status': status,
               'disk_free_gb_after': disk_after, 'disk_free_gb_delta': disk_after - disk_before,
               'wall_s': time.monotonic() - t0}
    (ARMS / (arm + '_cleanup.json')).write_text(json.dumps(cleanup, indent=2))
    summary = json.loads((run_dir / 'summary.json').read_text()) if status == 'PASS' else None
    error = None
    if status != 'PASS':
        fail = run_dir / 'failure.json'
        error = json.loads(fail.read_text()).get('error') if fail.exists() else (
            '无结论文件（rc=%r）' % rc)
    append_ledger(LEDGER, ledger_entry(round_no, order_idx, policy, arm, cleanup, summary, error))
    print('完成', policy, 'rc', rc, status, '磁盘余', disk_after, 'G', flush=True)
    return status, disk_after


def round_gates(round_no, dry_run):
    """轮前门禁全集（盘/定卡/残留/身份/三臂干净）。"""
    disk_gate()
    gpu_gate()
    residue_gate()
    identity_gate()
    for order_idx, policy in enumerate(ROUND_ORDER[round_no], 1):
        arm_gate(arm_name(policy, round_no))
    print('轮前门禁通过 R%d（盘/定卡/残留/身份/臂目录%s）' % (
        round_no, '/命令构造' if dry_run else ''))


def dry_run_round(round_no):
    """演习模式：门禁 + 三臂启动命令构造落盘（不启动容器、不写总账）。"""
    for order_idx, policy in enumerate(ROUND_ORDER[round_no], 1):
        arm = arm_name(policy, round_no)
        cmd = launch.docker_cmd(policy, arm, PINNED_GPU, 'flowstate-p04-%s' % arm)
        path = ARMS / (arm + '_command_dryrun.json')
        path.write_text(json.dumps({'round': round_no, 'order_idx': order_idx,
                                    'policy': policy, 'gpu': PINNED_GPU, 'cmd': cmd}, indent=2))
        print('  [%d/3] %s -> %s' % (order_idx, policy, path.name))


def ledger_selftest():
    """总账自检：追加写入（PASS 样例含指标 / FAIL 样例含错误摘要）→ 回读校验只增不改。"""
    assert not SELFTEST_LEDGER.exists(), '自检账簿已存在，拒绝覆盖：' + str(SELFTEST_LEDGER)
    ok_cleanup = {'returncode': 0, 'status': 'PASS', 'disk_free_gb_after': 68,
                  'disk_free_gb_delta': -1, 'wall_s': 180.2}
    ok_summary = {'cycle': {'control_cycle_ms': 9523.7, 'cycle_ms_full_incl_warm': 23159.5,
                            'request_exec_total_ms': 1252.8},
                  'pending_rows': [{'ttft_ms': 123.8, 'cuda_prefill_ms': 116.2},
                                   {'ttft_ms': 161.3, 'cuda_prefill_ms': 153.9},
                                   {'ttft_ms': 205.2, 'cuda_prefill_ms': 196.6},
                                   {'ttft_ms': 242.8, 'cuda_prefill_ms': 233.7}],
                  'selection': {'plan_recovery_cost_ms': 703.45,
                                'selected_cpu': ['样例：不指向真实臂']}}
    bad_cleanup = {'returncode': 1, 'status': 'BLOCKED', 'disk_free_gb_after': 67,
                   'disk_free_gb_delta': 0, 'wall_s': 42.0}
    append_ledger(SELFTEST_LEDGER, ledger_entry(0, 1, 'SELFTEST-PASS', 'selftest_ok',
                                                ok_cleanup, ok_summary))
    append_ledger(SELFTEST_LEDGER, ledger_entry(0, 2, 'SELFTEST-FAIL', 'selftest_fail',
                                                bad_cleanup, None, 'ValueError: 样例失败记录'))
    lines = [json.loads(x) for x in SELFTEST_LEDGER.read_text().splitlines() if x.strip()]
    assert len(lines) == 2 and lines[0]['status'] == 'PASS' and lines[1]['status'] == 'BLOCKED'
    assert lines[0]['sum_ttft_ms'] == round(123.8 + 161.3 + 205.2 + 242.8, 3)
    assert lines[1]['error'] == 'ValueError: 样例失败记录' and lines[1]['J_ms'] is None
    print('总账自检通过：追加 2 行（PASS 含指标 / FAIL 含错误摘要），回读一致，样例账簿=%s'
          % SELFTEST_LEDGER.name)


def resume_arm(round_no, policy):
    """单臂续跑（人工授权后续接被轮前门禁中断的轮次）：只运行该轮顺序中第一个未完成臂。

    纪律：前序臂必须全部已在总账 PASS 且证据在位；目标臂与更后臂必须从未运行；
    轮状态文件按该轮总账行重建（一次性写入）。时间断口在审计报告中如实披露。
    """
    order = ROUND_ORDER[round_no]
    assert policy in order, '策略不在该轮顺序中'
    idx = order.index(policy) + 1
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []
    done = {r['policy']: r['status'] for r in rows if r['round'] == round_no}
    for i, p in enumerate(order, 1):
        has_run = (ARMS / arm_name(p, round_no) / 'run').exists()
        if i < idx:
            assert done.get(p) == 'PASS' and (ARMS / arm_name(p, round_no) / 'run' / 'summary.json').exists(), \
                '前序臂未完整 PASS，拒绝续跑：' + p
        else:
            assert p not in done and not has_run, \
                ('续跑目标臂已有记录或已运行：' if i == idx else '续跑目标之后的臂已有记录，顺序异常：') + p
    disk_gate()
    gpu_gate()
    residue_gate()
    identity_gate()
    arm_gate(arm_name(policy, round_no))
    print('续跑门禁通过 R%d 臂[%d/3] %s（前序臂均已 PASS；盘/定卡/残留/身份复核）' % (round_no, idx, policy))
    status, free = run_round_arm(round_no, idx, policy)
    if status != 'PASS':
        print('R%d 续跑臂非 PASS：%s（保留现场，停止）' % (round_no, policy), flush=True)
    else:
        assert free > 20, '根分区余量不足 20G，暂停'
    # 轮状态按该轮总账行重建（含续跑臂），一次性写入
    status_path = ARMS / ('arms_status_m%d.json' % round_no)
    assert not status_path.exists(), '轮状态文件已存在，拒绝覆盖：' + str(status_path)
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    res = {r['policy']: {'arm': r['arm'], 'order_idx': r['order_idx'], 'status': r['status'],
                         'disk_free_gb': r['disk_free_gb']} for r in rows if r['round'] == round_no}
    status_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return status


def main():
    arm_flag = next((a[len('--arm='):] for a in sys.argv[1:] if a.startswith('--arm=')), None)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry_run = '--dry-run' in sys.argv
    if '--ledger-selftest' in sys.argv:
        ledger_selftest()
        return
    assert len(args) == 1, '用法：launch_round R [--dry-run|--ledger-selftest]'
    round_no = int(args[0])
    assert round_no in ROUND_ORDER, '轮次须在 1..5：' % str(round_no)
    if arm_flag is not None:
        resume_arm(round_no, arm_flag)
        return
    round_gates(round_no, dry_run)
    if dry_run:
        dry_run_round(round_no)
        print('R%d dry-run 完成（未启动容器、未写总账）' % round_no)
        return
    results = {}
    for order_idx, policy in enumerate(ROUND_ORDER[round_no], 1):
        status, free = run_round_arm(round_no, order_idx, policy)
        results[policy] = {'arm': arm_name(policy, round_no), 'order_idx': order_idx,
                           'status': status, 'disk_free_gb': free}
        if status != 'PASS':
            print('R%d fail-loud：%s 非 PASS，终止本轮（保留现场，不重试不跳过）' % (round_no, policy),
                  flush=True)
            break
        assert free > 20, '根分区余量不足 20G，暂停'
    status_path = ARMS / ('arms_status_m%d.json' % round_no)
    assert not status_path.exists(), '轮状态文件已存在，拒绝覆盖：' + str(status_path)
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
