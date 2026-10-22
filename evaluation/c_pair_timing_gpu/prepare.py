"""冻结输入准备：从已完成 PASS 的 c_shared_gpu_pilot_k3 复制冻结产物并交叉核验哈希。

- 冻结五件套（prepared_inputs/policy_plans/cpu_candidate_prefix_mapping/cpu_gate/source_identity 的 runtime_sources）
  必须与 pilot attempt3 实现锁逐字节一致，保证本轮候选域/策略选择/Token 与 Pilot 完全相同；
- Cold 参照输出取自 pilot cold_C_B / cold_C_C 臂，用于本轮每个生命周期的逐 Token 输出一致性断言；
- 本目录 control.py / bridge.py 为 pilot 逐字节副本（协调与观察合同不变，仅运行器增加计时仪表）。
所有文件 write-once：已存在即拒绝，防止静默改写。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/c_pair_timing_gpu'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'

FROZEN = ['prepared_inputs.json', 'policy_plans.json', 'cpu_candidate_prefix_mapping.json', 'cpu_gate.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def main():
    # 冻结文件不得已存在（write-once）
    for f in FROZEN + ['cold_reference_C_B.json', 'cold_reference_C_C.json', 'source_identity.json']:
        assert not (OUT / f).exists(), '已存在，禁止覆盖：' + f
    # 交叉核验：冻结输入与 pilot attempt3 实现锁一致
    lock3 = json.loads((PILOT / 'implementation_lock3.json').read_text())['files']
    for f in FROZEN:
        src = str(PILOT / f)
        assert src in lock3, 'pilot 实现锁缺少冻结文件：' + f
        assert sha(src) == lock3[src], '冻结文件与 pilot 锁不符：' + f
        (OUT / f).write_bytes(Path(src).read_bytes())
    # control/bridge 逐字节等于 pilot（协调合同不变）
    for f in ['control.py', 'bridge.py']:
        assert sha(OUT / f) == sha(PILOT / f), '控制栈副本与 pilot 不一致：' + f
    # Cold 参照：pilot cold 臂的真实输出
    for pending in ['C_B', 'C_C']:
        arm = PILOT / ('cold_' + pending)
        summary = json.loads((arm / 'run/summary.json').read_text())
        assert summary['status'] == 'PASS' and summary['H_physical_fa_hit'] == 0
        output = json.loads((arm / 'run/output.json').read_text())
        write(OUT / f'cold_reference_{pending}.json',
              {'pending': pending, 'source': 'c_shared_gpu_pilot_k3/cold_' + pending,
               'output_ids': output['output_ids'], 'completion_tokens': output['meta_info']['completion_tokens']})
    # 运行时身份：沿用 pilot 镜像与 runtime 源码哈希（容器内核验用）
    pilot_identity = json.loads((PILOT / 'source_identity.json').read_text())
    write(OUT / 'source_identity.json',
          {'image_id': pilot_identity['image_id'],
           'runtime_sources': pilot_identity['runtime_sources'],
           'note': '镜像与 runtime 源码哈希沿用 c_shared_gpu_pilot_k3（PASS 实验同一受控栈）；'
                   '本目录 control.py/bridge.py 为其逐字节副本，由 launch 实现锁另行锁定'})
    print('准备完成：冻结四件套 + Cold 参照 C_B/C_C + 运行时身份')


if __name__ == '__main__':
    main()
