"""独立Strict或Cold生命周期，仅执行一个目标Pending。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
from .bridge import ROOT,RAW,load,require,scheduler_process,TARGET

def write(out,name,value):
    with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def main():
    arm=os.environ['R2_ARM'];out=Path('/r2out')/arm;out.mkdir(exist_ok=False);engine=None;partial={};result=None
    try:
        source=json.loads((Path('/r2out')/'source_identity.json').read_text());require(source['status']=='PASS','来源身份门禁失败')
        frozen,epoch,plan=load();config=frozen['configuration'];current={}
        for name in ('sglang','transformers'):require(importlib.metadata.version(name)==config[name+'_version'],'依赖版本不符')
        for path,h in source['runtime_sources'].items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'镜像源码改变')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(config['engine_configuration']['model_path'],local_files_only=True)
        require(hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()==config['chat_template_sha256'],'模板身份变化')
        from evaluation.strict_compatibility_b1.run_pilot import _load_frozen_group
        from evaluation.strict_compatibility_b4b0.contracts import MANIFEST
        from evaluation.rq3_openhands_neutral_collector import SGLangGroupRuntime,replay_group_to_barrier
        group=_load_frozen_group(MANIFEST,1)
        require(group.allocation_round==3 and dict(group.session_by_label)==frozen['group']['session_by_label'],'原始分组改变')
        requests={(r['request']['workflow_label'],r['request']['turn']):r['request'] for r in frozen['history']}
        require(len(requests)==12,'历史数量错误')
        label_by_workflow={v:k for k,v in group.session_by_label.items()}
        for p in frozen['pendings']:
            label=label_by_workflow[p['workflow_id']]
            requests[(label,4)]={'workflow_label':label,'workflow_id':p['workflow_id'],'turn':4,'rid':p['request_id'],'input_ids':p['token_ids']}
        write(out,'prepared_inputs.json',{'requests':list(requests.values()),'snapshot_sha256':epoch.snapshot_sha256,'controlled_input_ready':True,'agent_native_ready':'未认证','pending_execution':False})
        from sglang.srt.entrypoints.engine import Engine
        from targeted_probe import ControlClient
        from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
        class ObservedEngine(Engine):
            run_scheduler_process_func=staticmethod(scheduler_process)
        engine=ObservedEngine(**config['engine_configuration'])
        client=ControlClient(int(os.environ['FLOWSTATE_STEP5D_PORT']));wait_for_transport(client)
        if arm=='strict':
            trace=replay_group_to_barrier(SGLangGroupRuntime(engine,client),group,requests,partial_rows=partial)
            write(out,'history_replay.json',partial)
            require(not any((trace.native_mamba_eviction,trace.fa_cascade,trace.oom,trace.truncation)) and trace.fa_query_side_effect_free,'历史回放正确性失败')
            require(len(trace.checkpoints)==12 and len(trace.pendings)==4,'当前候选或Pending数量变化')
            histories=[]
            for row in frozen['history']:
                request=row['request'];response=client._call({'op':'checkpoint_control','action':'flowstate_runtime_metrics','request_id':request['rid'],'nonce':str(time.time_ns())})
                from evaluation.strict_compatibility_b1c.protocol import token_digest
                require(all(x['input_token_digest']==token_digest(request['input_ids']) and x['input_token_count']==len(request['input_ids']) for x in response['records']),'历史实际Runtime输入不一致')
                histories.append(response)
            for c in trace.checkpoints:
                old=next(x for x in frozen['candidates'] if x['checkpoint_id']==c.checkpoint_id)
                require(c.token_pos==old['token_position_L'],'当前Checkpoint位置变化')
            for p in trace.pendings:
                old=next(x for x in frozen['pendings'] if x['continuation_id']==p.continuation_id)
                require(p.resident_fa_frontier==old['attention_frontier'] and min(p.anchor_pos,p.resident_fa_frontier)==old['planning_target_T'],'当前Planning Target变化')
            write(out,'history_identity.json',histories)
            result=client._call({'op':'checkpoint_control','action':'r2c1b2_reconcile','nonce':str(time.time_ns())})
            write(out,'reconciliation_trace.json',result)
            require(result.get('ok') is True,'严格协调失败，不提交Pending')
        else:
            result=client._call({'op':'checkpoint_control','action':'r2c1b2_state','nonce':str(time.time_ns())})
            require(not any(n['slots'] for n in result['state']['nodes']),'Cold初始缓存不为空')
            write(out,'cold_initial.json',result)
        require(result['observer_api_preflight']['status']=='PASS','观测接口前置检查失败')
        pending=next(p for p in frozen['pendings'] if p['continuation_id']==TARGET)
        sampling={'temperature':0,'max_new_tokens':8,'ignore_eos':True}
        write(out,'request.json',{'pending':pending,'sampling_params':sampling,'arm':arm})
        output=engine.generate(input_ids=pending['token_ids'],sampling_params=sampling,rid=TARGET)
        write(out,'output.json',output)
        state=client._call({'op':'checkpoint_control','action':'r2c1b2_state','nonce':str(time.time_ns())})
        write(out,'post_request.json',state)
        require(len(output.get('output_ids',[]))==8 and output['meta_info']['completion_tokens']==8,'输出长度不符')
        require(int(output['meta_info'].get('num_retractions',0) or 0)==0,'发生请求回撤')
        matches=[m['native'] for m in state['trace']['matches'] if m.get('native',{}).get('cow_requested')]
        require(matches,'没有原生实际匹配证据')
        require(state['trace']['execution'] and all(x.get('native_run_returned') for x in state['trace']['execution']),'实际run_batch未完整返回')
        actual=matches[-1]
        require(actual['input_token_digest']==pending['token_prefix_sha256'],'Pending实际输入摘要不符')
        if arm=='cold':require(actual['physical_fa_hit']==0 and actual['executable_prefix']==0,'Cold发生缓存命中')
        else:
            expected=next(p for p in plan.pending if p.pending_id==TARGET)
            bound=next(b for b in result['reconciliation']['binding']['candidates'] if b['candidate_id']=='RQ3_G001_B_TURN_001')['node']
            require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'恢复来源不符')
            require(actual['executable_prefix']==expected.E and actual['physical_fa_hit']==pending['attention_frontier'],'原生恢复前沿不符')
            copies=state['trace']['copies']
            require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in copies),'缺少正确COW复制证据')
            protected_slots={s for b in result['reconciliation']['binding']['candidates'] if b['candidate_id'] in plan.selected_ids for s in b['node']['slots']}
            require(not protected_slots.intersection(actual['working_slots']),'工作Slot与Selected别名')
            require(state['state']['selected_payload']==result['selected_payload_after'],'请求破坏Selected内容')
            require(state['state']['selected_components']==result['admission_baseline']['selected_components'],'Selected物理切片变化')
        write(out,'summary.json',{'status':'PASS','arm':arm,'pending_count':1,'native_match':actual,'output_ids':output['output_ids']})
        print(json.dumps({'status':'PASS','arm':arm,'E':actual['executable_prefix']},ensure_ascii=False),flush=True)
    except BaseException as exc:
        failure={'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc(),'partial_replay':partial,'eviction_attempts':len((result or {}).get('operations',[]))}
        write(out,'failure.json',failure)
        if not (out/'reconciliation_trace.json').exists():write(out,'reconciliation_trace.json',failure)
        if not (out/'physical_postcheck.json').exists():write(out,'physical_postcheck.json',{'status':'UNKNOWN','reason':'前置门禁失败，未取得完整当前视图'})
        print(failure['traceback'],flush=True)
        raise
    finally:
        if engine is not None:engine.shutdown()
        write(out,'engine_cleanup.json',{'engine_created':engine is not None,'shutdown_returned':True,'unix_ns':time.time_ns(),'说明':'GPU显存及容器退出由宿主机另行验证'})
if __name__=='__main__':main()
