"""phase2 启动器：形状预热修正生命周期（strict_warm），仅当计时污染被确认后执行。

- 门禁：contamination_confirmed.json 必须存在且载明证据摘要（由 phase1 离线分析写入）；
- 4 个生命周期：w1/w2 × {FlowState-Greedy, StandaloneTopK}，奇数对 FS 先手；
- 其余纪律与 launch.py 完全一致（复用 launch_phase）。
"""
import json
from pathlib import Path
from launch import OUT, launch_phase, read


def main():
    assert not (OUT / 'launch_p2_complete.json').exists(), 'phase2 已完成，禁止重跑'
    gate = OUT / 'contamination_confirmed.json'
    assert gate.exists(), 'phase1 未确认计时污染，phase2 不执行'
    evidence = read(gate)
    assert evidence.get('confirmed') is True and evidence.get('evidence'), '污染确认文件缺少证据摘要'
    protocol = read(OUT / 'protocol.json')
    launch_phase(protocol['phase2_arms'], 'p2')


if __name__ == '__main__':
    main()
