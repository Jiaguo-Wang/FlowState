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

# CUDA仅用替身测试计时包装合同，Req/Batch仍为冻结版本真实类型。
from unittest.mock import patch
from array import array
from sglang.srt.managers.schedule_batch import Req,ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.model_executor.forward_batch_info import ForwardMode
class Event:
 def record(self,stream=None):pass
 def synchronize(self):pass
 def elapsed_time(self,other):return 1.0
req=Req(observation.TARGET,'',array('q',[1,2,3,4]),SamplingParams(temperature=0,max_new_tokens=8,ignore_eos=True))
req._refresh_fill_ids();req.set_extend_range(2,4);req.prefix_indices=array('q',[10,11])
batch=ScheduleBatch(reqs=[req],device='cpu',forward_mode=ForwardMode.EXTEND,prefix_lens=[2],extend_lens=[2])
calls=[];rows=[]
with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',side_effect=lambda **kw:Event()),patch.object(torch.cuda,'current_stream',return_value=SimpleNamespace(cuda_stream=1)):
 wrapped=observation.wrap_run_batch(lambda *a:calls.append(1) or '完成',rows,lambda:None)
 assert wrapped(SimpleNamespace(enable_overlap=False),batch)=='完成' and calls==[1] and rows[0]['cuda_elapsed_ms']==1.0
 try:wrapped(SimpleNamespace(enable_overlap=True),batch)
 except ValueError:pass
 else:raise AssertionError('重叠调度未拒绝')
 assert calls==[1]
 def fail(*args):calls.append(2);raise RuntimeError('原生异常')
 try:observation.wrap_run_batch(fail,[],lambda:None)(SimpleNamespace(enable_overlap=False),batch)
 except RuntimeError:pass
 else:raise AssertionError('原生异常被吞')
 assert calls==[1,2] and req.extend_range.start==2 and list(req.origin_input_ids)==[1,2,3,4]
checks['cuda_wrapper_cpu_contract']='PASS：原生恰好一次、异常传播、重叠拒绝、Req不变；未认证GPU时间'
(out/'cpu_gate.json').write_text(json.dumps({'status':'PASS','checks':checks,'runtime_sources':files,'gpu_engine_started':False},ensure_ascii=False,indent=2))
print('成对计时包装CPU合同PASS')
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
assert callable(EagerRunner.execute)
with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',side_effect=lambda **kw:Event()),patch.object(torch.cuda,'current_stream',return_value=SimpleNamespace(cuda_stream=1)):
 calls=[];observation.COMPUTE=[];observation.ACTIVE=True
 wrapped=observation.wrap_eager(lambda *a:calls.append(1) or '原结果',lambda:None)
 assert wrapped(None,batch)=='原结果' and calls==[1] and observation.COMPUTE[0]['returned']
 observation.ACTIVE=False
 assert wrapped(None,batch)=='原结果' and calls==[1,1] and len(observation.COMPUTE)==1
checks['eager_wrapper_cpu_contract']='PASS：只计目标、原调用恰好一次、保留返回值'
for cls in [EagerRunner]:
 p=Path(inspect.getfile(cls));files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'cpu_gate.json').write_text(json.dumps({'status':'PASS','checks':checks,'runtime_sources':files,'gpu_engine_started':False},ensure_ascii=False,indent=2))
print('COW后模型边界CPU合同PASS')
