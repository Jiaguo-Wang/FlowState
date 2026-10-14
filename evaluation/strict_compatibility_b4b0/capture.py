"""仅由下一阶段显式授权启动的单组采集入口；模块导入不创建Engine。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from .contracts import *

def capture(ordinal,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    engine=None;status='BLOCKED';partial={};snapshot_digest=None
    append_event(out,'开始',group=ordinal)
    try:
        lock=verify_lock();row=group_contract(ordinal)
        write_new(out/'source.json',{'group':row,'execution_lock':lock,'sampling_protocol':read(PROTOCOL),'native_root_identity':'UNKNOWN','agent_native_ready':'UNKNOWN'})
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.engine import Engine
        from evaluation.strict_compatibility_b1c.runner import scheduler_process
        from evaluation.strict_compatibility_b1.run_pilot import _load_frozen_group,_path_evidence
        from evaluation.rq3_openhands_neutral_collector import SGLangGroupRuntime,materialize_group_requests,replay_group_to_barrier
        from evaluation.openhands_single_workflow_smoke import TOKENIZER_PATH,normalize_message,_template_input_ids
        from evaluation.rq3_openhands_population import load_session_messages
        from targeted_probe import ControlClient
        from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
        identity=read(B1)['configuration_identity'];write_new(out/'configuration.json',identity)
        for name in ('sglang','transformers'):require(importlib.metadata.version(name)==identity[name+'_version'],'版本不一致')
        import sglang
        source_root=Path(sglang.__file__).parent/'srt'
        expected=read(ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2/retained/evidence.json')['source_identity']
        for old,digest in expected.items():
            relative=old.split('/srt/',1)[1];require(sha(source_root/relative)==digest,'Runtime源码版本变化')
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH,local_files_only=True)
        require(hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()==identity['chat_template_sha256'],'模板变化')
        group=_load_frozen_group(MANIFEST,ordinal)
        requests,boundary=materialize_group_requests(tokenizer,{label:load_session_messages(sid,DATA) for label,sid in group.session_by_label.items()},group=group,normalize_message=normalize_message,template_input_ids=_template_input_ids)
        for a in boundary:
            require(all(a[k] is False for k in ('pending_turn_output_read','r_plus_2_message_consumed','r_plus_2_request_materialized','future_timing_read','future_checkpoint_read')),'未来输入污染')
        write_new(out/'prepared_inputs.json',{'requests':list(requests.values()),'boundary_audit':boundary,'controlled_input_ready_ns':time.monotonic_ns()})
        append_event(out,'controlled_input_ready')
        class ObservedEngine(Engine):
            run_scheduler_process_func=staticmethod(scheduler_process)
        engine=ObservedEngine(**identity['engine_configuration'])
        client=ControlClient(int(os.environ.get('FLOWSTATE_STEP5D_PORT','49986')));wait_for_transport(client)
        runtime=SGLangGroupRuntime(engine,client)
        def call(action,**kw):return client._call({'op':'checkpoint_control','action':action,'nonce':str(time.time_ns()),**kw})
        def state():return call('b1c_state')['state']
        def records(rid):return call('b1c_records',request_id=rid)
        trace=replay_group_to_barrier(runtime,group,requests,partial_rows=partial)
        require(not any((trace.native_mamba_eviction,trace.fa_cascade,trace.oom,trace.truncation)) and trace.fa_query_side_effect_free,'历史回放门禁失败')
        barrier=state();write_new(out/'barrier_capture.json',barrier);safe_barrier(barrier,trace.native_mamba_eviction)
        barrier_id=f'b4b1-g{ordinal:03d}-{barrier["monotonic_ns"]}'
        config_digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        history=[]
        for (label,turn),request in requests.items():
            if turn<=group.allocation_round:
                evidence=records(request['rid'])
                require(evidence['records'] and all(x['input_token_digest']==token_digest(request['input_ids']) for x in evidence['records']),'实际历史输入身份不符')
                history.append({'request':request,'native_match_and_copy':evidence})
        write_new(out/'history_input_evidence.json',history)
        candidates=[]
        for c in trace.checkpoints:
            tokens=requests[(c.workflow_label,c.turn)]['input_ids'][:c.token_pos]
            path=runtime.inspect_checkpoint(c.checkpoint_id,tokens)['after']['path']
            require(path['node_id']==c.node_id and path['prefix_tokens']==c.token_pos and path['prefix_sha256']==token_digest(tokens),'候选身份变化')
            candidates.append({'checkpoint_id':c.checkpoint_id,'workflow_id':c.workflow_id,'workflow_label':c.workflow_label,'token_ids':tokens,'token_position_L':c.token_pos,'token_prefix_sha256':token_digest(tokens),'state_identity':_path_evidence(path),'configuration_digest':config_digest,'barrier_id':barrier_id,'metadata':{'memory_bytes':read(MANIFEST)['checkpoint_size_bytes'],'lineage_path':[c.workflow_id],'creation_order':c.creation_order,'last_access_order':c.last_access_order,'access_rule':'原collector基于实际历史match E且同Workflow同位置更新；不是原生全树LRU','contributing_request_ids':list(c.contributing_request_ids)}})
        resident={n['node_id'] for n in barrier['nodes'] if n['mamba_slots']}
        require(resident=={c['state_identity']['node_id'] for c in candidates},'存在候选外驻留状态，前沿合同未证明')
        pendings=[]
        for p in trace.pendings:
            req=requests[(p.workflow_label,group.allocation_round+1)];tokens=req['input_ids']
            pendings.append({'continuation_id':p.continuation_id,'request_id':req['rid'],'workflow_id':req['workflow_id'],'token_ids':tokens,'token_prefix_sha256':token_digest(tokens),'planning_target_T':min(p.anchor_pos,p.resident_fa_frontier),'attention_frontier':p.resident_fa_frontier,'configuration_digest':config_digest,'barrier_id':barrier_id,'controlled_input_ready':True,'runtime_receive_identity':'待冻结后单独验证'})
        end=state();safe_barrier(end)
        require(barrier['nodes']==end['nodes'] and barrier['tree']==end['tree'] and barrier['accounting']==end['accounting'],'屏障采集期间状态变化')
        from evaluation.sota_metadata import build_marconi_flop_saved,CONTROLLED_MARCONI_ALPHA
        from flowstate.state_catalog import CheckpointCandidate
        from evaluation.openhands_common_barrier_snapshot_gate import lineage_path
        raw_candidates=[CheckpointCandidate(c['checkpoint_id'],c['workflow_id'],lineage_path(c['workflow_id']),c['token_position_L'],c['metadata']['memory_bytes']) for c in candidates]
        flop=build_marconi_flop_saved(raw_candidates)
        for c in candidates:
            c['metadata'].update({'lineage_path':list(lineage_path(c['workflow_id'])),'marconi_flop_saved':flop[c['checkpoint_id']],'marconi_alpha':CONTROLLED_MARCONI_ALPHA})
        edges=build_edges(candidates,pendings,barrier,config_digest)
        labels=structure_labels(candidates,pendings,edges)
        snapshot={'group':row,'barrier_id':barrier_id,'configuration':identity,'barrier_start':barrier,'barrier_end':end,'candidates':candidates,'pendings':pendings,'edges':edges,'structure':labels,'history':history,'history_request_rows':list(trace.request_rows),'history_census':list(trace.census_rows),'future_boundary_audit':boundary,'cost_matrix':'本轮不计算收益','slot_generation':'UNKNOWN','native_ready':'未认证'}
        snapshot_digest=freeze_snapshot(out,snapshot);append_event(out,'屏障证据已冻结',sha256=snapshot_digest)
        checks=[]
        for p in pendings:
            rid=p['request_id'];append_event(out,'Pending身份核对提交',request_id=rid)
            output=engine.generate(input_ids=p['token_ids'],sampling_params={'temperature':0,'max_new_tokens':1,'ignore_eos':True},rid=rid)
            received=records(rid)
            write_new(out/f'pending_{rid}.json',{'output':output,'received':received})
            require(received['records'] and all(x['input_token_digest']==p['token_prefix_sha256'] and x['input_token_count']==len(p['token_ids']) for x in received['records']),'Pending实际输入身份变化')
            require(output['meta_info']['completion_tokens']==1 and int(output['meta_info'].get('num_retractions',0) or 0)==0,'Pending截断或回撤')
            unchanged_snapshot(out,snapshot_digest)
            checks.append({'request_id':rid,'records':received,'output':output,'说明':'冻结后身份核对，不回填原始快照的E或Handle'})
            append_event(out,'Pending身份核对通过',request_id=rid)
        write_new(out/'pending_identity_checks.json',checks)
        after=state();safe_barrier(after);write_new(out/'post_pending_state.json',after)
        prior={n['prefix_digest']:n['mamba_slots'] for n in barrier['nodes'] if n['mamba_slots']}
        later={n['prefix_digest']:n['mamba_slots'] for n in after['nodes'] if n['mamba_slots']}
        require(all(later.get(k)==v for k,v in prior.items()),'Pending期间出现原有状态非预期驱逐或替换')
        status='CAPTURED' if all(e['status']!='UNKNOWN' for e in edges) else 'PARTIAL'
        write_new(out/'summary.json',{'status':status,'structure':labels,'edge_counts':{s:sum(e['status']==s for e in edges) for s in ('EPOCH_ADMISSIBLE','REJECTED','UNKNOWN')},'snapshot_sha256':snapshot_digest,'allocation_comparison_run':False})
    except BaseException as exc:
        append_event(out,'失败',error=repr(exc),traceback=traceback.format_exc())
        write_new(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'partial_replay':partial})
        raise
    finally:
        try:
            if engine is not None:engine.shutdown()
            append_event(out,'清理返回',engine_created=engine is not None,status=status)
        except BaseException as exc:
            append_event(out,'清理失败',error=repr(exc));raise
        finally:
            try:
                if snapshot_digest:unchanged_snapshot(out,snapshot_digest)
            finally:
                finalize_manifest(out)

def main():
    p=argparse.ArgumentParser(description='下一阶段单组受控采集入口')
    p.add_argument('--execute-b4b1',action='store_true');p.add_argument('--group',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(a.execute_b4b1,'缺少下一阶段显式执行参数');capture(a.group,a.out)
if __name__=='__main__':main()
