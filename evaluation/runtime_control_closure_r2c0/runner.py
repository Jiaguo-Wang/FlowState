"""一个g001生命周期：只回放冻结历史，不提交Pending，不释放状态。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
from .bridge import ROOT,RAW,load,require,scheduler_process

def write(out,name,value):
    with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def main():
    out=Path('/r2out');engine=None;partial={};result=None
    try:
        source=json.loads((out/'source_identity.json').read_text());require(source['status']=='PASS','来源身份门禁失败')
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
        result=client._call({'op':'checkpoint_control','action':'r2c0_readonly','nonce':str(time.time_ns())})
        write(out,'preflight_results.json',{k:v for k,v in result.items() if k!='runtime_view_audit'})
        write(out,'runtime_view_audit.json',result.get('runtime_view_audit',{'status':'UNKNOWN','reason':result.get('error','控制调用未返回视图')}))
        print(json.dumps({'status':result.get('status'),'error':result.get('error'),'destructive_calls':0},ensure_ascii=False),flush=True)
        require(result.get('ok') is True,'真实只读Preflight未通过')
    except BaseException as exc:
        failure={'status':'BLOCKED','error':repr(exc),'traceback':traceback.format_exc(),'partial_replay':partial,'destructive_calls':0}
        write(out,'failure.json',failure)
        if not (out/'preflight_results.json').exists():write(out,'preflight_results.json',failure)
        if not (out/'runtime_view_audit.json').exists():write(out,'runtime_view_audit.json',{'status':'UNKNOWN','reason':'前置门禁失败，未取得完整当前视图'})
        print(failure['traceback'],flush=True)
        raise
    finally:
        if engine is not None:engine.shutdown()
        write(out,'engine_cleanup.json',{'engine_created':engine is not None,'shutdown_returned':True,'unix_ns':time.time_ns(),'说明':'GPU显存及容器退出由宿主机另行验证'})
if __name__=='__main__':main()
