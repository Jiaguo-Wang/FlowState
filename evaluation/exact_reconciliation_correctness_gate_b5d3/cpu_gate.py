"""冻结镜像内仅用CPU验证真实Req/Batch和FA读取接口。"""
import hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import torch
from . import observation
from .control import fa_payload
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool,MHATokenToKVPool,UnquantizedKVCacheMethod
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

out=Path('/b5out');checks={}
checks['req_batch']=observation.real_api_preflight()
full=MHATokenToKVPool.__new__(MHATokenToKVPool);full.layer_transfer_counter=None;full.start_layer=0;full.quant_method=UnquantizedKVCacheMethod();full.dtype=torch.float32;full.store_dtype=torch.float32
full.k_buffer=[torch.arange(24,dtype=torch.float32).reshape(6,2,2)];full.v_buffer=[torch.arange(24,dtype=torch.float32).reshape(6,2,2)+1]
# 仅构造CPU缓冲，不调用任何GPU池构造器。
pool=HybridLinearKVPool.__new__(HybridLinearKVPool);pool.full_attention_layer_id_mapping={3:0};pool.layer_transfer_counter=None;pool.start_layer=0;pool.full_kv_pool=full
cache=SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda:pool))
before=[x.clone() for x in full.k_buffer+full.v_buffer]
h=fa_payload(cache,[1,3]);assert h==fa_payload(cache,[1,3])
assert all(torch.equal(x,y) for x,y in zip(before,full.k_buffer+full.v_buffer));checks['fa_cpu_readonly']='PASS'
assert h!=fa_payload(cache,[1,4]);checks['fa_distinct_slice']='PASS'
checks['allocator_api']='PASS' if callable(BaseTokenToKVPoolAllocator.get_kvcache) else 'FAIL'
files={str(Path(inspect.getfile(cls))):hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in [HybridLinearKVPool,BaseTokenToKVPoolAllocator]}
(out/'cpu_gate.json').write_text(json.dumps({'status':'PASS','checks':checks,'runtime_sources':files,'gpu_engine_started':False},ensure_ascii=False,indent=2))
print('真实Req/Batch及FA CPU只读门禁PASS')
