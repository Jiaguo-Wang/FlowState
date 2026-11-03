"""T2-G2 正式 GPU 对照：CPU 侧预检与输入冻结（本脚本不启动 GPU、不运行任何臂）。

产出（均写入本 formal/ 目录）：
- policy_plans.json：t2g1_plans.json['plans'] 的原样投影（顶层裸键格式——c_formal control 以
  `policy in plans` 泛型消费，无策略名硬编码；只读 selected_checkpoint_ids 与 expected_E_G，
  其余键如 J_ms_current_scope 原样保留但不被控制路径使用）；
- 8 份冻结共享文件：逐字节复制 c_formal_perf_gpu 并与其 implementation_lock_main.json 交叉核对
  （prepared_inputs / source_identity / cpu_gate / cpu_candidate_prefix_mapping /
  cold_reference_C_{A,B,C,D}）——同镜像、同输入、同 Cold 参照，与 c_formal 主矩阵唯一差异=策略计划；
- protocol.json：63 臂冻结调度（3 门禁臂 g_CB_{FS,MC,KV} + 60 主臂 m{轮}{Pending}_{FS,MC,KV}；
  奇数轮 FS→MC→KV、偶数轮反转；同轮同 Pending 三策略时间相邻），一次生成后冻结不可变；
- formal_precheck.json：预检结论（三策略×四 Pending E/T/G 齐全、K=3、调度计数、镜像在场、来源哈希）。

所有写入均幂等：目标已存在时逐字节比对而非重写，保证冻结语义。
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
CFORMAL = ROOT / 'evaluation/c_formal_perf_gpu'
OUT = Path(__file__).resolve().parent

POLICIES = ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired']  # 冻结策略顺序（调度基准）
TOKEN = {'FlowState-Greedy': 'FS', 'Marconi-Evict': 'MC', 'KVFlow-inspired': 'KV'}
PENDINGS = ['C_A', 'C_B', 'C_C', 'C_D']
SHARED = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json',
          'cpu_candidate_prefix_mapping.json', 'cold_reference_C_A.json', 'cold_reference_C_B.json',
          'cold_reference_C_C.json', 'cold_reference_C_D.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write_frozen(p, obj):
    """冻结写入：已存在则断言内容一致（幂等重跑保护），不存在则以独占模式新建。"""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if Path(p).exists():
        assert Path(p).read_text() == text, '冻结文件与再生内容不一致（禁止改动）：' + str(p)
    else:
        with Path(p).open('x') as f:
            f.write(text)


def build_protocol():
    """生成 63 臂冻结调度：门禁 3 臂（C_B=三策略 E 全>0 且形状预热 E 两两不同的唯一 Pending）
    + 主矩阵 60 臂（奇数轮 FS→MC→KV、偶数轮反转，对冲时间漂移；同轮同 Pending 三策略相邻配对）。"""
    gate_arms = [{'name': 'g_CB_' + TOKEN[p], 'pair': 1, 'policy': p, 'pending': 'C_B'} for p in POLICIES]
    main_arms = []
    for rep in range(1, 6):
        order = POLICIES if rep % 2 == 1 else list(reversed(POLICIES))
        for pending in PENDINGS:
            for p in order:
                main_arms.append({'name': 'm%d%s_%s' % (rep, pending[-1], TOKEN[p]),
                                  'rep': rep, 'pending': pending, 'policy': p})
    return {
        'version': '1.0',
        'source': {
            'module': 'evaluation.c_formal_perf_gpu.runner/bridge/control 零改动复用（逐臂哈希复核）',
            'plans': '../t2g1_plans.json（plans 子字典原样投影为 policy_plans.json）',
            'shared_inputs': 'c_formal_perf_gpu 8 份冻结共享文件逐字节复制（与 implementation_lock_main.json 交叉核对）',
            'contract': '测量合同沿用 c_formal_perf_gpu protocol v1.2：strict_warm + FS_MEASURE_MODE=1 + '
                        '形状预热 B v1.1（pt[:E]+确定性随机尾，E 处分叉）+ 请求窗口零 A 类审计零新增编译 + '
                        'Cold 逐 Token 一致 + census 16=16 + E/T/G 逐格==冻结计划',
        },
        'policies': {tok: pol for pol, tok in TOKEN.items()},
        'pairing': '同轮同 Pending 三策略时间相邻；顺序冻结（奇数轮 FS→MC→KV、偶数轮反转），不依结果调整',
        'gate_arms': gate_arms,
        'gate_criteria': '三臂 PASS（census 16/16/0、E/G==冻结、COW/FA/Cold 逐 Token 一致、形状预热行 E 与 '
                         'FA 命中均恰为计划 E、请求窗口编译=0、窗口内审计项=2），人工复核后写 gate_passed.json 方可执行主矩阵',
        'main_arms': main_arms,
        'deadline_policy': '各阶段启动时固定 now+6h 写入 execution_started_<phase>.json；超时剩余臂如实标未测',
        'rules': '不修改三种策略的评分与驱逐算法、不修改 Runtime 状态控制语义、不改动 c_formal_perf_gpu 目录任何文件；'
                 '不修改生产 FlowState/Phi/策略实现/冻结 Token/预算；不改动镜像（runtime_sources 哈希逐生命周期由 runner 复核）；'
                 '不删除任何 B/C 类必要操作；每臂独立容器与 Engine/Triton 生命周期；GPU 空卡礼仪与 c_formal 一致'
                 '（无其他 compute 进程 + 显存<1000MiB + H100 PCIe），不沿用 Smoke 放宽条件；'
                 '任一臂失败即停保留证据；不把 5 次重复解释为充分统计显著性；负向/平局观测全保留',
        'stop_rule': '状态正确性失败 / 未解释编译污染 / 跨策略资源条件不一致 → 立即停止正式比较并保存证据；'
                     'deadline 截止未完成臂如实标未测',
    }


def main():
    # 1) 策略计划投影：断言三策略 × 四 Pending 齐全、K=3，然后原样落盘
    t2g1 = read(T2G1 / 't2g1_plans.json')
    plans = t2g1['plans']
    assert set(plans) == set(POLICIES), '策略集合不符：%s' % sorted(plans)
    for pol in POLICIES:
        row = plans[pol]
        assert len(row['selected_checkpoint_ids']) == 3, 'K!=3：' + pol
        assert set(row['expected_E_G']) == set(PENDINGS), 'E/G 预期不齐：' + pol
        for pid in PENDINGS:
            e = row['expected_E_G'][pid]
            assert set(e) >= {'E', 'T', 'G'} and e['G'] == e['T'] - e['E'], 'E/T/G 合同异常：%s/%s' % (pol, pid)
        assert row.get('selection_source'), '缺少选集来源声明：' + pol
    write_frozen(OUT / 'policy_plans.json', plans)

    # 2) 冻结共享文件：逐字节复制并与 c_formal 实现锁交叉核对（血缘证明）
    lock_main = read(CFORMAL / 'implementation_lock_main.json')['files']
    for f in SHARED:
        src = CFORMAL / f
        assert sha(src) == lock_main[str(src)], 'c_formal 文件与其实现锁不符：' + f
        dst = OUT / f
        if dst.exists():
            assert sha(dst) == sha(src), '共享文件与 c_formal 不一致（禁止改动）：' + f
        else:
            dst.write_bytes(src.read_bytes())

    # 3) 调度：一次生成，之后冻结
    protocol = build_protocol()
    assert len(protocol['main_arms']) == 60 and len(protocol['gate_arms']) == 3, '臂数异常'
    write_frozen(OUT / 'protocol.json', protocol)

    # 4) 镜像在场核对（只读 docker 检查，不启动容器）
    image_id = read(OUT / 'source_identity.json')['image_id']
    got = json.loads(subprocess.run(['docker', 'image', 'inspect', image_id], check=True,
                                    capture_output=True, text=True).stdout)[0]['Id']
    assert got == image_id, '镜像不在场或 Id 不符'

    # 5) 预检结论落盘
    write_frozen(OUT / 'formal_precheck.json', {
        'status': 'READY',
        'policies': POLICIES,
        'shared_files': {f: sha(OUT / f) for f in SHARED},
        'policy_plans': sha(OUT / 'policy_plans.json'),
        'source_t2g1_plans': sha(T2G1 / 't2g1_plans.json'),
        'source_formal_plan': sha(T2G1 / 'formal_plan.md'),
        'protocol': sha(OUT / 'protocol.json'),
        'image_id': image_id,
        'gate_arms': [a['name'] for a in protocol['gate_arms']],
        'main_arm_count': len(protocol['main_arms']),
        'note': 'CPU 预检通过：输入/调度/镜像冻结完成；正式启动尚需空卡复核+单独批准（launch_formal.py gate/main）',
    })
    print('T2-G2 预检 READY：门禁 %d 臂 + 主矩阵 %d 臂；镜像 %s' % (
        len(protocol['gate_arms']), len(protocol['main_arms']), image_id[:19]))


if __name__ == '__main__':
    main()
