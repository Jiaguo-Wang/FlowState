"""加载模型前验证唯一可见 CUDA 设备，随后原样运行冻结 Runner（GPU1 编排入口）。

2026-10-27 首个 GPU1 批次（formal_gpu1）失败修正：原版顶层无 __main__ 保护，
SGLang Engine 以 multiprocessing spawn 创建 scheduler 子进程时，
spawn._fixup_main_from_path 会把本入口以 __mp_main__ 语义重新执行，子进程重复
执行独占写入 /b5out/device_binding.json → FileExistsError → Rank 0 scheduler
初始化死亡（g_CB_FS returncode=1）。GPU0 批次入口 `-m evaluation.c_formal_perf_gpu.runner`
自带 __main__ 保护，故同一 Runner 在 GPU0 从未触发。

现全部可执行逻辑收进 main() 并置于 `if __name__ == '__main__'` 保护内：
- 主进程（__main__）：行为与原版逐条一致（设备校验 → 独占写证据 → runpy 接力冻结 Runner）；
- spawn 子进程（__mp_main__）：重入为无操作，不触碰任何文件。
回归测试见 test_spawn_reentry.py（含对旧入口的 RED 对照）。
"""
import json
import runpy
from pathlib import Path

import torch

EXPECTED = 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'  # GPU1 UUID（与 launcher/nvidia-smi 检查同源）


def main():
    assert torch.cuda.device_count() == 1, '可见CUDA设备不是单卡'
    p = torch.cuda.get_device_properties(0)
    actual = str(p.uuid)
    assert actual.removeprefix('GPU-').lower() == EXPECTED.removeprefix('GPU-').lower(), 'CUDA设备UUID不符'
    assert 'H100 PCIe' in p.name, '设备型号不符'
    with Path('/b5out/device_binding.json').open('x') as f:
        json.dump({'status': 'PASS', 'host_gpu_uuid': EXPECTED, 'cuda_device_count': 1,
                   'logical_device': 0, 'actual_uuid': actual, 'name': p.name},
                  f, ensure_ascii=False, indent=2)
    runpy.run_module('evaluation.c_formal_perf_gpu.runner', run_name='__main__')


if __name__ == '__main__':
    main()
