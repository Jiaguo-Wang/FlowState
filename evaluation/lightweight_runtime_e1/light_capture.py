"""轻量层保留完整元数据读取与双读检查；张量内容审计独立执行。"""
import hashlib,json,os,threading,time,traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from evaluation.runtime_control_closure_r2c0 import bridge as prior
from evaluation.strict_compatibility_b1c.protocol import token_digest
from evaluation.strict_compatibility_b1c.runner import MATCHES,COPIES,install_observation
from evaluation.strict_compatibility_b4b0.contracts import build_edges
from flowstate.recovery_model import RecoveryCostModel
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/b5out')
EVENTS=[]

def payload(cache,view):
    """按既有R2-C1a只读方式对每个驻留槽的Conv及Temporal取内容摘要。"""
    import torch
    tensors=cache.req_to_token_pool.mamba_pool.mamba_cache
    rows={}
    for n in view.nodes:
        if not n.slots:continue
        rows[str(n.node_id)]=[{'slot':slot,'component':i,'sha256':hashlib.sha256(t[:,slot].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()} for slot in n.slots for i,t in enumerate(list(tensors.conv)+[tensors.temporal])]
    return rows

def observe_control(scheduler,request):
    """同一同步调用内读取前后视图、内容和FA；异常保持隔离并由父进程结束生命周期。"""
    import torch
    from evaluation.barrier_fa_frontier_control import inspect_resident_fa_frontier
    result={'ok':False,'status':'BLOCKED','destructive_calls':0,'pending_dispatches':0}
    scheduler._r2c0_owner=threading.get_ident()
    window=f'b5c3-{os.getpid()}-{time.monotonic_ns()}'
    try:
        inputs=json.loads((OUT/'prepared_inputs.json').read_text())
        config=inputs['configuration'];cd=prior.digest(config)
        epoch=SimpleNamespace(configuration_digest=cd)
        prior.scope(scheduler)
        effective={k:getattr(scheduler.server_args,k) for k in config['engine_configuration']}
        prior.require(effective==config['engine_configuration'],'实际配置变化')
        torch.cuda.synchronize()
        before,audit=prior.observe(scheduler,epoch,window)
        slots=[s for n in before.nodes for s in n.slots]
        prior.require(len(slots)==len(set(slots)),'驻留槽存在别名')
        prior.require(set(slots).isdisjoint(before.free_slots),'已用空闲重叠')
        prior.require(len(slots)+len(before.free_slots)==before.pool_capacity,'物理槽记账不闭合')
        final=request.get('final') is True
        hashes=None
        candidates=[];pendings=[];edges=[];fa_rows=[]
        if final:
            for n in before.nodes:
                if not n.slots:continue
                row=next(r for r in audit['state']['nodes'] if r['node_id']==n.node_id)
                prior.require(n.references==0 and not n.protected and not n.host_copy,'驻留状态有未解决保护或Host引用')
                candidates.append({'checkpoint_id':f'B5C3_NODE_{n.node_id}','token_ids':list(n.token_ids),'token_position_L':len(n.token_ids),'token_prefix_sha256':token_digest(n.token_ids),'configuration_digest':cd,'barrier_id':window,'state_identity':{'node_id':n.node_id,'target_mamba_slots':list(n.slots),'target_mamba_present':True,'target_full_present':n.attention_present,'path_full_all_present':all(x.attention_present for x in before.nodes if n.token_ids[:len(x.token_ids)]==x.token_ids),'references':{'target_full_lock_ref':row['references']['full_lock'],'target_mamba_lock_ref':row['references']['mamba_lock'],'target_mamba_host_lock_ref':row['references']['mamba_host_lock'],'target_mamba_session_ref':row['references']['mamba_session']}},'component_ids':list(n.component_ids),'payload':None,'ownership':'本窗口当前Node到槽及存储切片映射','slot_generation':'UNKNOWN；不建立跨窗口租约'})
            for p in inputs['pendings']:
                fa=inspect_resident_fa_frontier(scheduler.tree_cache,p['token_ids'])
                target=min(len(p['token_ids']),int(fa['resident_fa_frontier']))
                pendings.append({**p,'planning_target_T':target,'attention_frontier':fa['resident_fa_frontier'],'configuration_digest':cd,'barrier_id':window,'controlled_input_ready':True,'runtime_receive_identity':'未派发；仅CPU已准备输入','agent_native_ready':'未认证'})
                fa_rows.append({'pending_id':p['continuation_id'],'observation':fa})
            edges=build_edges(candidates,pendings,audit['state'],cd)
        after,last=prior.observe(scheduler,epoch,window)
        after_hashes=None
        prior.require(before==after and audit['semantic']==last['semantic'] and audit['component_slices']==last['component_slices'],'只读窗口改变物理或替换状态')
        prior.require(hashes==after_hashes,'只读窗口改变组件内容')
        prior.scope(scheduler)
        result.update({'ok':True,'status':'OBSERVED','window':window,'effective_configuration':effective,'before':asdict(before),'after':asdict(after),'audit_before':audit,'audit_after':last,'payload_before':hashes,'payload_after':after_hashes,'unchanged':True,'matches':MATCHES,'copies':COPIES,'native_eviction_events':EVENTS,'candidates':candidates,'pendings':pendings,'edges':edges,'fa_observations':fa_rows,'resident_outside_candidate_set':[] if final else '未进行候选构建'})
        if final:
            model=RecoveryCostModel();rows=[]
            for c in candidates:
                support=[];content=[];benefits={}
                for p in pendings:
                    e=next(e for e in edges if e['checkpoint_id']==c['checkpoint_id'] and e['continuation_id']==p['continuation_id'])
                    if c['token_ids']==p['token_ids'][:c['token_position_L']]:content.append(p['label'])
                    if e['status']=='EPOCH_ADMISSIBLE':
                        t=p['planning_target_T'];b=model.estimate(t,t)-model.estimate(t-c['token_position_L'],t)
                        prior.require(b>=0,'负收益违反冻结模型合同')
                        benefits[p['label']]=b
                        if b>0:support.append(p['label'])
                    else:benefits[p['label']]=None if e['status']=='UNKNOWN' else 0
                rows.append({'checkpoint_id':c['checkpoint_id'],'node':c['state_identity']['node_id'],'slots':c['state_identity']['target_mamba_slots'],'L':c['token_position_L'],'token_content_support':content,'strict_positive_support':support,'standalone_positive_evidence_ms':benefits,'resume_validated':False})
            supports={tuple(r['strict_positive_support']) for r in rows}
            wanted={('A','B','C','D'),('A','B'),('C','D')}
            complete=all(e['status']!='UNKNOWN' for e in edges)
            result['support_rows']=rows
            result['status']='READY' if wanted<=supports and complete else 'PARTIAL'
            result['strict_epoch']={'epoch_id':window,'configuration':config,'configuration_digest':cd,'workload_type':'Trace-derived Controlled Composition','candidates':candidates,'pendings':pendings,'edges':edges,'logical_budget':'未定义；本轮不分配','complete_matrix':complete,'pending_dispatched':False,'slot_generation':'UNKNOWN；以同一同步窗口身份保证观察一致性'}
        return result
    except BaseException as exc:
        scheduler._r2c0_failed=True
        result.update(error=repr(exc),traceback=traceback.format_exc())
        return result
    finally:scheduler._r2c0_owner=None
