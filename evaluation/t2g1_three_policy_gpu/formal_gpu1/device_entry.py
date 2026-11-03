"""在加载模型前验证唯一可见CUDA设备，随后原样运行冻结Runner。"""
import json
import runpy
from pathlib import Path
import torch
EXPECTED = "GPU-051c9abc-864a-9f31-c4f7-a737aed997ff"
assert torch.cuda.device_count() == 1, '可见CUDA设备不是单卡'
p = torch.cuda.get_device_properties(0)
actual = str(p.uuid)
assert actual.removeprefix('GPU-').lower() == EXPECTED.removeprefix('GPU-').lower(), 'CUDA设备UUID不符'
assert 'H100 PCIe' in p.name, '设备型号不符'
with Path('/b5out/device_binding.json').open('x') as f:
    json.dump({'status':'PASS','host_gpu_uuid':EXPECTED,'cuda_device_count':1,'logical_device':0,'actual_uuid':actual,'name':p.name},f,ensure_ascii=False,indent=2)
runpy.run_module('evaluation.c_formal_perf_gpu.runner', run_name='__main__')
