"""加载模型前验证唯一可见 CUDA 设备，随后运行本轮四 Pending Runner（容器入口）。

与 t2g1 formal_gpu1b/device_entry.py 同模式（spawn 重入教训：全部逻辑置于 __main__ 保护内）：
- 主进程（__main__）：设备校验（UUID 由 launch 注入 EXPECTED_GPU_UUID）→ 独占写证据
  → runpy 接力 evaluation.serving_control_gpu_gate_20261102.runner；
- spawn 子进程（__mp_main__）：重入为无操作，不触碰任何文件。
"""
import json
import os
import runpy
from pathlib import Path

import torch


def main():
    expected = os.environ['EXPECTED_GPU_UUID']
    assert torch.cuda.device_count() == 1, '可见CUDA设备不是单卡'
    p = torch.cuda.get_device_properties(0)
    actual = str(p.uuid)
    assert actual.removeprefix('GPU-').lower() == expected.removeprefix('GPU-').lower(), 'CUDA设备UUID不符'
    assert 'H100 PCIe' in p.name, '设备型号不符'
    with Path('/b5out/device_binding.json').open('x') as f:
        json.dump({'status': 'PASS', 'host_gpu_uuid': expected, 'cuda_device_count': 1,
                   'logical_device': 0, 'actual_uuid': actual, 'name': p.name},
                  f, ensure_ascii=False, indent=2)
    runpy.run_module('evaluation.serving_control_gpu_gate_20261102.runner', run_name='__main__')


if __name__ == '__main__':
    main()
