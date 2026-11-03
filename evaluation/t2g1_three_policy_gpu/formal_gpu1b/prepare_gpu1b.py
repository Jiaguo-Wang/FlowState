"""GPU1 第二批次（formal_gpu1b）CPU 侧装配与预检：不启动 GPU。

来源与修正纪律（对 formal_gpu1 中断批次的最小修正续建）：
- 前置：test_spawn_reentry.py 必须已 PASS（入口 __main__ 保护 + spawn 重入回归含 RED 对照）；
- 10 项冻结输入：要求 formal_gpu1 与 GPU0 封存批次 formal/ 逐字节一致后，自 formal_gpu1 复制
  （血缘双源核验，任一漂移即停）；
- launch_formal.py / verify_formal.py / analyze_formal.py：自 formal_gpu1 派生，
  仅替换输出目录与容器名前缀（断言替换计数，防静默漂移）；
- takeover/verify_evidence.py：逐字节复制（gate 阶段双复核后落盘 gate_passed.json 的
  凭据链沿用）；
- device_entry.py / test_spawn_reentry.py：本目录新写入（入口保护修正与回归）。
所有写入幂等：已存在则逐字节比对而非重写。
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
SRC = T2G1 / 'formal_gpu1'      # GPU1 中断批次（冻结输入与适配脚本的直接来源）
GPU0 = T2G1 / 'formal'          # GPU0 封存批次（第二血缘源）
OUT = Path(__file__).resolve().parent

SHARED10 = ['protocol.json', 'policy_plans.json', 'prepared_inputs.json', 'source_identity.json',
            'cpu_gate.json', 'cpu_candidate_prefix_mapping.json',
            'cold_reference_C_A.json', 'cold_reference_C_B.json',
            'cold_reference_C_C.json', 'cold_reference_C_D.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def derive(src_path, dst_path, replacements):
    """自 formal_gpu1 派生脚本：逐对 (旧串, 新串, 期望次数) 替换并断言计数。"""
    text = Path(src_path).read_text()
    for old, new, want in replacements:
        got = text.count(old)
        assert got == want, '替换计数漂移：%s 期望%d 实得%d' % (old, want, got)
        text = text.replace(old, new)
    if Path(dst_path).exists():
        assert Path(dst_path).read_text() == text, '派生文件与现存内容不一致：' + str(dst_path)
    else:
        Path(dst_path).write_text(text)


def main():
    # 0) 前置：spawn 重入回归已 PASS；device_entry 目标 UUID 为 GPU1
    reg = read(OUT / 'spawn_regression.json')
    assert reg['status'] == 'PASS', 'spawn 重入回归未通过，禁止装配'
    entry_text = (OUT / 'device_entry.py').read_text()
    assert 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff' in entry_text, 'device_entry UUID 非 GPU1'

    # 1) 冻结输入：双血缘一致后复制
    for f in SHARED10:
        assert sha(SRC / f) == sha(GPU0 / f), '血缘不一致（formal_gpu1 vs formal）：' + f
        dst = OUT / f
        if dst.exists():
            assert sha(dst) == sha(SRC / f), '冻结输入与来源不一致（禁止改动）：' + f
        else:
            dst.write_bytes((SRC / f).read_bytes())

    # 2) 验证器逐字节复制；执行脚本派生（仅目录与前缀）
    (OUT / 'takeover').mkdir(exist_ok=True)
    ve_src, ve_dst = SRC / 'takeover/verify_evidence.py', OUT / 'takeover/verify_evidence.py'
    if ve_dst.exists():
        assert sha(ve_dst) == sha(ve_src), 'verify_evidence 与来源不一致'
    else:
        ve_dst.write_bytes(ve_src.read_bytes())
    derive(SRC / 'launch_formal.py', OUT / 'launch_formal.py',
           [("evaluation/t2g1_three_policy_gpu/formal_gpu1'", "evaluation/t2g1_three_policy_gpu/formal_gpu1b'", 1),
            ("'flowstate-t2g2-gpu1-'", "'flowstate-t2g2-g1b-'", 1)])
    derive(SRC / 'verify_formal.py', OUT / 'verify_formal.py',
           [("formal_gpu1'", "formal_gpu1b'", 1)])
    derive(SRC / 'analyze_formal.py', OUT / 'analyze_formal.py',
           [("formal_gpu1'", "formal_gpu1b'", 1)])

    # 3) 镜像在场核对（只读 docker）
    image_id = read(OUT / 'source_identity.json')['image_id']
    got = json.loads(subprocess.run(['docker', 'image', 'inspect', image_id], check=True,
                                    capture_output=True, text=True).stdout)[0]['Id']
    assert got == image_id, '镜像不在场或 Id 不符'

    # 4) 预检落盘（schema 兼容 verify_evidence 与 launcher）
    protocol = read(OUT / 'protocol.json')
    precheck = {
        'status': 'READY',
        'policies': ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired'],
        'shared_files': {f: sha(OUT / f) for f in SHARED10[2:]},  # 8 项共享输入（与 GPU0 批次同构）
        'policy_plans': sha(OUT / 'policy_plans.json'),
        'source_t2g1_plans': sha(T2G1 / 't2g1_plans.json'),
        'source_formal_plan': sha(T2G1 / 'formal_plan.md'),
        'protocol': sha(OUT / 'protocol.json'),
        'image_id': image_id,
        'gate_arms': [a['name'] for a in protocol['gate_arms']],
        'main_arm_count': len(protocol['main_arms']),
        'gpu_uuid': 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff',
        'spawn_regression': sha(OUT / 'spawn_regression.json'),
        'lineage': {'frozen_inputs': 'formal_gpu1 与 formal 双源逐字节一致后复制',
                    'scripts': 'launch/verify/analyze 自 formal_gpu1 派生（仅目录与容器前缀）；'
                               'verify_evidence 逐字节复制；device_entry=__main__ 保护修正版',
                    'predecessor': 'formal_gpu1 封存为中断批次（gate 1 臂 INITIALIZATION_FAILED，0 有效臂）'},
        'note': 'CPU 预检通过：spawn 回归 PASS、输入/镜像冻结完成；正式启动须 launcher 逐臂空卡门禁',
    }
    text = json.dumps(precheck, ensure_ascii=False, indent=2)
    pc = OUT / 'formal_precheck.json'
    if pc.exists():
        assert pc.read_text() == text, '预检文件与再生内容不一致'
    else:
        pc.write_text(text)
    print('formal_gpu1b 预检 READY：门禁 %d 臂 + 主矩阵 %d 臂；spawn 回归 PASS；镜像 %s' % (
        len(protocol['gate_arms']), len(protocol['main_arms']), image_id[:19]))


if __name__ == '__main__':
    main()
