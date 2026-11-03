"""正式 GPU 性能配对实验输入准备：冻结四件套 + 四 Pending Cold 参照 + 运行时身份。

- 冻结输入逐字节复用 c_pair_timing_gpu（其 implementation_lock_p1.json 已与 pilot implementation_lock3.json
  交叉核对）；本目录 control.py 亦为其逐字节副本（协调/观察/审计合同不变）；
- Cold 参照：C_B/C_C 直接复用 c_pair_timing_gpu 文件；C_A/C_D 从 pilot 冷臂（空缓存全量 Prefill，
  PASS 生命周期）输出构建，格式一致；
- source_identity.json 复用同一受控镜像与 runtime 源码哈希（本轮不改动镜像内任何文件）。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PAIR = ROOT / 'evaluation/c_pair_timing_gpu'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
OUT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    # 1) 冻结四件套：逐字节复制并与 c_pair 实现锁交叉核对
    lock = json.loads((PAIR / 'implementation_lock_p1.json').read_text())['files']
    for f in ['prepared_inputs.json', 'cpu_gate.json', 'cpu_candidate_prefix_mapping.json', 'policy_plans.json']:
        assert sha(PAIR / f) == lock[str(PAIR / f)], '冻结输入与 c_pair 实现锁不符：' + f
        (OUT / f).write_bytes((PAIR / f).read_bytes())
    # 2) Cold 参照：C_B/C_C 复用；C_A/C_D 从 pilot 冷臂输出构建
    for pid in ['C_B', 'C_C']:
        src = PAIR / ('cold_reference_' + pid + '.json')
        assert sha(src) == lock[str(src)], 'Cold 参照与实现锁不符：' + pid
        (OUT / ('cold_reference_' + pid + '.json')).write_bytes(src.read_bytes())
    for pid in ['C_A', 'C_D']:
        arm = PILOT / ('cold_' + pid)
        summary = json.loads((arm / 'run' / 'summary.json').read_text())
        assert summary['status'] == 'PASS', 'pilot 冷臂未 PASS：' + pid
        out = json.loads((arm / 'run' / 'output.json').read_text())
        ref = {'pending': pid, 'source': 'c_shared_gpu_pilot_k3/cold_' + pid,
               'output_ids': out['output_ids'], 'completion_tokens': out['meta_info']['completion_tokens']}
        with (OUT / ('cold_reference_' + pid + '.json')).open('x') as f:
            json.dump(ref, f, ensure_ascii=False, indent=2)
    # 3) 运行时身份：同一镜像与 runtime 源码哈希
    (OUT / 'source_identity.json').write_bytes((PAIR / 'source_identity.json').read_bytes())
    # 4) 自检：四 Pending 三策略的 E/G 预期齐全（正式矩阵合同）
    plans = json.loads((OUT / 'policy_plans.json').read_text())
    for pol in ['FlowState-Greedy', 'StandaloneTopK', 'WFC']:
        assert set(plans[pol]['expected_E_G']) == {'C_A', 'C_B', 'C_C', 'C_D'}, 'E/G 预期不齐：' + pol
    print('准备完成：冻结输入 4 + Cold 参照 4 + source_identity；镜像',
          json.loads((OUT / 'source_identity.json').read_text())['image_id'][:19])


if __name__ == '__main__':
    main()
