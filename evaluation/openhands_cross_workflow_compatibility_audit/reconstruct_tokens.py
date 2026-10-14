"""用现存本地 tokenizer 重建历史输入，并逐 token 检查跨 workflow 前缀。"""
import json,hashlib,importlib.metadata,csv
from collections.abc import Mapping
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
MODEL=Path('/home/wjg/models/qwen3.5-9b')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(raw):
    msg={k:raw[k] for k in ['role','content'] if raw.get(k) is not None}
    calls=raw.get('tool_calls_json')
    if calls:
        calls=json.loads(calls) if isinstance(calls,str) else calls
        for call in calls:
            args=call.get('function',{}).get('arguments')
            if isinstance(args,str):call['function']['arguments']=json.loads(args)
        msg['tool_calls']=calls
    for k in ['tool_call_id','name']:
        if raw.get(k) is not None:msg[k]=raw[k]
    return msg
def tok(t,messages):
    x=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    return list(x['input_ids'] if isinstance(x,Mapping) else x)
def json_bytes(x):return json.dumps(x,separators=(',',':')).encode()
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
identity={f:sha(MODEL/f) for f in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json']}
assert t.chat_template==(MODEL/'chat_template.jinja').read_text()
index=read(OUT/'snapshot_inputs.json');all_rows=[];edge_rows=[]
for case in index:
    bylabel={};r=case['allocation_round']
    for s in case['sessions']:
        raw=read(s['message_file'])['raw_messages_through_pending_marker'];history=[];turn=0;inputs={}
        for msg in raw:
            if msg.get('role')=='assistant':
                turn+=1
                if turn<=r+1:inputs[turn]=tok(t,list(history))
                if turn==r+1:break
            history.append(normalize(msg))
        if set(inputs)!=set(range(1,r+2)):raise ValueError(f"请求轮次缺失 {s['session_id']}")
        bylabel[s['label']]=inputs
    runtime=read(case['runtime_path'])
    assert runtime['abort_reason'] is None and all(q['request_completed'] and not q['oom'] and not q['truncation_or_clipping'] for q in runtime['request_rows'])
    assert all(not b['pending_turn_output_read'] and not b['r_plus_2_message_consumed'] and not b['future_checkpoint_read'] for b in runtime['boundary_audit'])
    reqmap={(q['workflow'],q['turn']):q for q in runtime['request_rows']}
    candmap={};census_nodes={n['node_id']:n for n in case['census']['resident_mamba_nodes']}
    for c in case['candidates']:
        label=c['checkpoint_id'].split('_')[2];turn=int(c['checkpoint_id'].split('_')[-1]);matches=[n for n in census_nodes.values() if n['token_position']==c['token_pos']];assert matches;node=matches[0] if len(matches)==1 else None
        source=bylabel[label][turn]
        candmap[c['checkpoint_id']]=(label,turn,c['token_pos'],source,node)
        assert c['recurrent_resident'] and c['fa_resident'] and matches
        assert len(source)>=c['token_pos']
        logged=reqmap[(label,turn)]
        assert len(source)==logged['offline_input_tokens'],(case['snapshot_id'],label,turn,len(source),logged['offline_input_tokens'])
    pend={p['continuation_id'].split('_')[2]:p for p in case['pending']}
    for p_label,p in pend.items():
        pids=bylabel[p_label][r+1];assert len(pids)==p['anchor_pos']
        for cid,(c_label,turn,pos,cids,node) in candmap.items():
            if c_label==p_label:continue
            equal=pos<=len(pids) and cids[:pos]==pids[:pos]
            candidate_sha=hashlib.sha256(json_bytes(cids[:pos])).hexdigest()
            pending_sha=hashlib.sha256(json_bytes(pids[:pos])).hexdigest() if pos<=len(pids) else None
            edge_rows.append(dict(snapshot_id=case['snapshot_id'],ordinal=case['ordinal'],checkpoint_id=cid,candidate_workflow=c_label,candidate_turn=turn,candidate_position=pos,pending_id=p['continuation_id'],pending_workflow=p_label,pending_target=p['resident_fa_frontier'],reconstructed_exact_prefix=equal,reconstructed_prefix_sha256=candidate_sha,reconstructed_pending_prefix_sha256=pending_sha,candidate_recurrent_resident=True,candidate_fa_resident=True,physical_slot=node['slots'][0] if node else None,physical_node=node['node_id'] if node else None,physical_candidate_mapping='UNIQUE_POSITION_ONLY_NOT_IDENTITY_CERTIFIED' if len(matches)==1 else 'AMBIGUOUS_POSITION_MAPPING_USE_RESIDENT_FLAG_ONLY',runtime_token_identity='RECONSTRUCTED_TRANSFORMERS_5_13_NOT_ORIGINAL_5_12_1',runtime_pending_event='CONTROLLED_BARRIER_PENDING',historical_prefix_digest_retained=False))
    all_rows.append(dict(snapshot_id=case['snapshot_id'],ordinal=case['ordinal'],allocation_round=r,session_count=4,checkpoint_count=len(candmap),pending_count=len(pend),reconstructed_request_lengths_match_logged=True,all_original_requests_pass=all(q['request_completed'] and not q['oom'] and not q['truncation_or_clipping'] for q in runtime['request_rows']),future_boundary_pass=all(not b['pending_turn_output_read'] and not b['r_plus_2_message_consumed'] and not b['future_checkpoint_read'] for b in runtime['boundary_audit']),barrier_abort_reason=runtime['abort_reason'],candidate_positions_map_uniquely_to_resident_node=all(sum(n['token_position']==c['token_pos'] for n in census_nodes.values())==1 for c in case['candidates']),ambiguous_position_candidate_count=sum(sum(n['token_position']==c['token_pos'] for n in census_nodes.values())!=1 for c in case['candidates']),tokenizer_version=importlib.metadata.version('transformers'),template_sha256=identity['chat_template.jinja'],reconstructed_cross_edges=sum(e['snapshot_id']==case['snapshot_id'] and e['reconstructed_exact_prefix'] for e in edge_rows),candidate_digest_archived=False,snapshot_path=case['snapshot_path'],snapshot_sha256=case['snapshot_sha256'],runtime_path=case['runtime_path'],runtime_sha256=case['runtime_sha256']))
with (OUT/'reconstructed_edges.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(edge_rows[0]));w.writeheader();w.writerows(edge_rows)
with (OUT/'snapshot_identity_audit.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
(OUT/'reconstruction_identity.json').write_text(json.dumps({'tokenizer_model_path':str(MODEL),'transformers_version':importlib.metadata.version('transformers'),'tokenizer_file_sha256':identity,'frozen_runtime_transformers_version':'5.12.1','frozen_runtime_model_path':'/model','historical_runtime_token_ids_archived':False,'historical_candidate_prefix_digest_archived':False,'all_168_snapshots_request_lengths_match_original_logs':True,'reconstructed_cross_workflow_edges':sum(e['reconstructed_exact_prefix'] for e in edge_rows),'candidate_pending_cross_pairs':len(edge_rows)},ensure_ascii=False,indent=2)+'\n')
print('重建候选-跨请求边',len(edge_rows),'token前缀重合',sum(e['reconstructed_exact_prefix'] for e in edge_rows))
