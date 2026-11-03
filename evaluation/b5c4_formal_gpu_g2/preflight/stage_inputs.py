"""G2 阶段目录装配（宿主机执行）：把冻结输入复制进各阶段目录（该目录将挂载为容器 /b5out）。

用法：python3 stage_inputs.py cold|gate|formal
- cold（preflight/cold/）：prepared_inputs / cpu_gate / source_identity（Cold 补采臂所需）；
- gate / formal：另加 g2_reference.json 与 4 个 Pending 的 cold_reference_*.json
  （gate 要求 A/B/D 已由 promote_cold.py 落位，C 为 gate_a 冻结复制）。
全部按 preflight/ 根冻结副本的哈希核验后复制；拒绝覆盖既有文件。
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PRE = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight'
PENDINGS = ['B5C2_A_TURN_003', 'B5C2_B_TURN_003', 'B5C2_C_TURN_003', 'B5C2_D_TURN_003']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ''
    assert phase in ('cold', 'gate', 'formal'), '用法：stage_inputs.py cold|gate|formal'
    out = PRE / phase if phase != 'formal' else ROOT / 'evaluation/b5c4_formal_gpu_g2/formal'  # formal 位于 G2 根（与 launch.py/schedule.json 同目录，挂载为 /b5out）
    out.mkdir(exist_ok=True)
    names = ['prepared_inputs.json', 'cpu_gate.json', 'source_identity.json']
    if phase in ('gate', 'formal'):
        names += ['g2_reference.json'] + ['cold_reference_' + p + '.json' for p in PENDINGS]
    for name in names:
        src, dst = PRE / name, out / name
        assert src.exists(), '缺少冻结文件：' + str(src)
        assert not dst.exists(), '拒绝覆盖：' + str(dst)
        data = src.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha(src)
        dst.write_bytes(data)
    print('阶段目录装配完成：%s（%d 个冻结文件，哈希核验通过）' % (out, len(names)))


if __name__ == '__main__':
    main()
