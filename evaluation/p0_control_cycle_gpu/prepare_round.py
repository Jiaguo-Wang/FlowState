"""P0-4 五轮矩阵·轮次装配器（宿主机执行，只读复制冻结输入并锁定身份；与策略实现隔离）。

复用 prepare.py 的冻结输入原语（write_once 拒绝覆盖、cpu_prechecks、collect_entry_sources、
serial_safety_check），为轮次 R∈[1,5] 装配三个全新臂目录 m{R}_ck3_{fs|mc|kv}
（共 15 臂，与 r1_* 及历史归档零冲突）。实验语义与 r1 Pilot 完全一致：
同源冻结输入（g1_CB_FS 8 文件 + t2g1_plans + workloads case C）、同 identity schema
（仅新增 'round' 字段）、同镜像与镜像内 Runtime 冻结哈希、同串行安全判定。
不修改 prepare.py / launch.py / runner / observation / 策略实现 / 冻结输入。

执行顺序（任务书固定拉丁方，装配与启动共用，禁止中途调整）：
  R1: FlowState-Greedy, Marconi-Evict, KVFlow-inspired
  R2: Marconi-Evict, KVFlow-inspired, FlowState-Greedy
  R3: KVFlow-inspired, FlowState-Greedy, Marconi-Evict
  R4: FlowState-Greedy, KVFlow-inspired, Marconi-Evict
  R5: Marconi-Evict, FlowState-Greedy, KVFlow-inspired

用法（仓库根目录执行）：
  python3 -m evaluation.p0_control_cycle_gpu.prepare_round 3     # 仅装配第 3 轮三臂
  python3 -m evaluation.p0_control_cycle_gpu.prepare_round all   # 装配全部五轮共 15 臂
"""
import json
import sys
import time

from . import prepare

POLICY_SUFFIX = {'FlowState-Greedy': 'fs', 'Marconi-Evict': 'mc', 'KVFlow-inspired': 'kv'}
# 任务书§四.2 固定执行顺序（launch_round 同表引用；任何改动须经人工确认）
ROUND_ORDER = {
    1: ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired'],
    2: ['Marconi-Evict', 'KVFlow-inspired', 'FlowState-Greedy'],
    3: ['KVFlow-inspired', 'FlowState-Greedy', 'Marconi-Evict'],
    4: ['FlowState-Greedy', 'KVFlow-inspired', 'Marconi-Evict'],
    5: ['Marconi-Evict', 'FlowState-Greedy', 'KVFlow-inspired'],
}


def arm_name(policy, round_no):
    """五轮矩阵臂目录名：m{轮次}_ck3_{策略缩写}（与 r1_* 命名空间零冲突）。"""
    return 'm%d_ck3_%s' % (round_no, POLICY_SUFFIX[policy])


def build_round_arm(policy, round_no, safety, g1_source, entry_sources):
    """装配单个五轮矩阵臂（语义同 prepare.build_arm，臂名按轮次参数化；拒绝覆盖既有目录）。"""
    NAMES, SRC, T2G1, MAIN = prepare.NAMES, prepare.SRC, prepare.T2G1, prepare.MAIN
    OUTBASE, sha, write_once = prepare.OUTBASE, prepare.sha, prepare.write_once
    arm = arm_name(policy, round_no)
    out = OUTBASE / arm
    assert not out.exists(), '臂目录已存在，拒绝覆盖：' + str(out)
    out.mkdir()
    for name in NAMES:
        write_once(out / name, (SRC / name).read_bytes())
    write_once(out / 't2g1_plans.json', (T2G1 / 't2g1_plans.json').read_bytes())
    case = next(c for c in json.loads((MAIN / 'workloads.json').read_text())['cases']
                if c['structure'] == 'C')
    write_once(out / 'workloads_case_C.json', case)   # 抽取副本；下方与源文件规范化一致性断言
    src_case_again = next(c for c in json.loads((MAIN / 'workloads.json').read_text())['cases']
                          if c['structure'] == 'C')
    assert json.loads((out / 'workloads_case_C.json').read_text()) == src_case_again, 'case 抽取副本与源不一致'
    identity = {
        'round': round_no,
        'image_id': g1_source['image_id'],
        'runtime_sources': g1_source['runtime_sources'],
        'hybrid_entry_sources': entry_sources,
        'frozen_input_hashes': {'g1_CB_FS/' + n: sha(SRC / n) for n in NAMES} |
                               {'t2g1_plans.json': sha(T2G1 / 't2g1_plans.json')},
        'serial_safety': safety,
        'note': '五轮矩阵臂（m%d）：冻结输入与 r1 Pilot 同源同款；t2g1_plans 仅供 runner 在 '
                'T4 之后窗口外 oracle 对照，控制路径（control.py）不读取' % round_no}
    write_once(out / 'source_identity.json', identity)
    print('臂目录就绪', policy, '->', out, '(round %d)' % round_no)
    return {'arm': arm, 'dir': str(out)}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    rounds = sorted(ROUND_ORDER) if arg == 'all' else [int(arg)]
    assert all(1 <= r <= 5 for r in rounds), '轮次须在 1..5：' + repr(rounds)
    prepare.OUTBASE.mkdir(exist_ok=True)
    safety = prepare.cpu_prechecks()          # census 16=16、三策略 E 向量、oracle schema、串行安全
    g1_source = json.loads((prepare.SRC / 'source_identity.json').read_text())
    entry_sources = prepare.collect_entry_sources()
    for r in rounds:
        prepared = {}
        for policy in ROUND_ORDER[r]:         # 装配顺序与执行顺序一致（仅目录创建，无运行语义）
            prepared[policy] = build_round_arm(policy, r, safety, g1_source, entry_sources)
        prepare.write_once(prepare.OUTBASE / ('prepare_record_m%d.json' % r), {
            'round': r, 'order': ROUND_ORDER[r], 'prepared': prepared,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_identity_note': 'runtime_sources 沿用 g1_CB_FS 冻结哈希；语义与 r1 装配一致'})


if __name__ == '__main__':
    main()
