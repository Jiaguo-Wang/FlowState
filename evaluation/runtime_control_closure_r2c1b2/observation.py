"""只读观察冻结版本的真实请求与批次，不执行缓存操作。"""
import time
from array import array

TARGET='RQ3_G001_A_PENDING_TURN_004'

def require(value,message):
    if not value:raise ValueError(message)

def observe_batch(batch):
    """仅Extend模式解释extend_range；Decode不把残留区间当成当前计算范围。"""
    rows=[]
    for i,req in enumerate(batch.reqs):
        if str(req.rid)!=TARGET:continue
        row={'ns_before':time.monotonic_ns(),'forward_mode':str(batch.forward_mode),
             'original_input_length':len(req.origin_input_ids),
             'full_untruncated_length':len(req.full_untruncated_fill_ids),
             'prefix_indices_length':len(req.prefix_indices),
             'output_length_before':len(req.output_ids),
             'optional_extend_batch_idx':getattr(req,'extend_batch_idx','UNKNOWN')}
        if batch.forward_mode.is_extend():
            fill=req.get_fill_ids();start=len(req.prefix_indices);end=req.extend_range.end
            require(req.extend_range.start==start,'Extend起点与前缀长度不同')
            require(len(fill)==end and len(fill[start:])==req.extend_range.length,'真实输入区间不一致')
            require(batch.prefix_lens[i]==start and batch.extend_lens[i]==end-start,'Batch与Req区间不一致')
            row.update(extend_start=start,extend_end=end,extend_token_count=end-start,
                       batch_prefix_length=batch.prefix_lens[i],batch_extend_length=batch.extend_lens[i])
        else:
            row.update(extend_start='不适用：非Prefill',extend_end='不适用：非Prefill',extend_token_count='不适用：非Prefill')
        rows.append(row)
    return rows

def wrap_run_batch(original,records,persist):
    """原函数恰好调用一次；其异常原样传播，不将返回当作最终输出认证。"""
    def run(scheduler,batch,*args,**kwargs):
        rows=observe_batch(batch)
        records.extend(rows)
        if rows:persist()
        result=original(scheduler,batch,*args,**kwargs)
        for row in rows:row.update(native_run_returned=True,ns_after=time.monotonic_ns())
        if rows:persist()
        return result
    return run

def real_api_preflight():
    """使用真实Req与ScheduleBatch，提交目标之前验证所需接口，无Engine或GPU分配。"""
    from sglang.srt.managers.schedule_batch import Req,ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    req=Req(TARGET,'',array('q',[1,2,3,4]),SamplingParams(temperature=0,max_new_tokens=8,ignore_eos=True))
    req._refresh_fill_ids();req.set_extend_range(2,4);req.prefix_indices=array('q',[10,11])
    batch=ScheduleBatch(reqs=[req],device='cpu',forward_mode=ForwardMode.EXTEND,prefix_lens=[2],extend_lens=[2])
    before=(list(req.origin_input_ids),list(req.full_untruncated_fill_ids),req.extend_range,list(req.prefix_indices),list(batch.prefix_lens),list(batch.extend_lens))
    row=observe_batch(batch)[0]
    after=(list(req.origin_input_ids),list(req.full_untruncated_fill_ids),req.extend_range,list(req.prefix_indices),list(batch.prefix_lens),list(batch.extend_lens))
    require(before==after and row['extend_token_count']==2,'真实API只读检查失败')
    return {'status':'PASS','req_class':Req.__module__+'.'+Req.__name__,'batch_class':ScheduleBatch.__module__+'.'+ScheduleBatch.__name__,'device':'cpu'}
