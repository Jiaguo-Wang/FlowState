"""T2-G1 Smoke 准备：为三个策略臂创建独立 case 目录（挂载为 /b5out，含共享冻结文件+T2-G1 计划）。

臂选择：三策略各 1 臂、同一 Pending C_B（唯一使三策略 E 全>0 的 Pending，恢复路径全验证）；
FS 臂同时充当冒烟正对照（该臂组合已在 pilot 通过 GPU 验证）。
输出对照：复用 pilot 冻结 cold_C_B（独立空缓存生命周期）作为生成一致性参照，不新建 cold 臂。
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
SMOKE = ROOT / 'evaluation/t2g1_three_policy_gpu/smoke'
ARMS = ['strict_FlowState-Greedy_C_B', 'strict_Marconi-Evict_C_B', 'strict_KVFlow-inspired_C_B']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    SMOKE.mkdir(exist_ok=False)
    plans = json.loads((ROOT / 'evaluation/t2g1_three_policy_gpu/t2g1_plans.json').read_text())['plans']
    pilot_plans = json.loads((PILOT / 'policy_plans.json').read_text())
    # 合同：FS 条目必须与 pilot 逐位一致（正对照的前提）
    assert plans['FlowState-Greedy']['selected_checkpoint_ids'] == pilot_plans['FlowState-Greedy']['selected_checkpoint_ids']
    assert plans['FlowState-Greedy']['expected_E_G'] == pilot_plans['FlowState-Greedy']['expected_E_G']
    smoke_plans = {}
    for policy, v in plans.items():
        smoke_plans[policy] = {'selected_checkpoint_ids': v['selected_checkpoint_ids'],
                               'expected_E_G': v['expected_E_G'],
                               'selection_source': 'T2-G1 ' + v['selection_source']}
    shared = ['prepared_inputs.json', 'source_identity.json', 'cpu_gate.json', 'cpu_candidate_prefix_mapping.json']
    lock = {}
    for arm in ARMS:
        case = SMOKE / arm
        case.mkdir()
        for f in shared:
            shutil.copyfile(PILOT / f, case / f)
            lock[str(case / f)] = sha(case / f)
        (case / 'policy_plans.json').write_text(json.dumps(smoke_plans, ensure_ascii=False, indent=2))
        lock[str(case / 'policy_plans.json')] = sha(case / 'policy_plans.json')
    # 运行模块与共享输入不可变锁（复用 pilot runner/control/bridge，零代码改动）
    for f in ['runner.py', 'control.py', 'bridge.py'] + shared:
        lock[str(PILOT / f)] = sha(PILOT / f)
    (SMOKE / 'smoke_lock.json').write_text(json.dumps(
        {'files': lock,
         'note': '三个 smoke case 的共享文件与 pilot 逐字节一致；policy_plans.json 为 T2-G1 三策略版（FS==pilot 逐位）',
         'arms': ARMS}, ensure_ascii=False, indent=2))
    print('smoke case 准备完成：', ARMS)


if __name__ == '__main__':
    main()
