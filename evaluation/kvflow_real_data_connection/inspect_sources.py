"""仅提取现存来源字段和因果截断证据，不构建推测ASG或运行策略。"""
import json,csv,hashlib
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sources={}
def read(p):
    data=p.read_bytes();sources[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def keys(x,c):
    if isinstance(x,dict):
        c.update(x.keys())
        for v in x.values():keys(v,c)
    elif isinstance(x,list):
        for v in x:
            if isinstance(v,(dict,list)):keys(v,c)

def main():
    refs=[];plan=read(ROOT/'rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/run_plan.json')['runs'];seen=set()
    for r in plan:
        if r['snapshot_id'] in seen:continue
        seen.add(r['snapshot_id']);s=json.loads(read(Path(r['snapshot_artifact']))['canonical_snapshot'])
        for p in s['pending_continuations']:
            refs.append(dict(snapshot=r['snapshot_id'],session=p['workflow_id'],turn=int(p['continuation_id'].rsplit('_',1)[1]),kind='旧Runtime24'))
    for name in ['w1','w2','w3','w4','B5-C4']:
        path=ROOT/'evaluation'/('multishare_runtime_gate_b5c3' if name=='B5-C4' else 'independent_runtime_allocation_e2/'+name)
        s=read(path/'strict_epoch_snapshot.json');read(path/'prepared_inputs.json')
        for p in s['pendings']:
            refs.append(dict(snapshot=name,session=p['workflow_id'],turn=p['turn'],kind='严格快照'))
    dataset=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')
    columns=['session_id','source_dataset','agent_framework','recorded_model','messages_json','ground_truth_meta_json']
    table=pq.read_table(dataset,filters=[('session_id','in',sorted({r['session'] for r in refs}))],columns=columns)
    rows={r['session_id']:r for r in table.to_pylist()};assert len(rows)==len({r['session'] for r in refs})
    outputs=[];allkeys=Counter()
    for ref in refs:
        row=rows[ref['session']];messages=json.loads(row['messages_json']);cut=[];turn=0
        for msg in messages:
            if msg.get('role')=='assistant':
                turn+=1
                if turn==ref['turn']:break
            cut.append(msg)
        assert turn==ref['turn']
        counts=Counter();keys(cut,counts);allkeys.update(counts)
        systems=[m for m in cut if m.get('role')=='system']
        metadata=json.loads(row['ground_truth_meta_json'] or '{}')
        outputs.append({**ref,'known_message_count':len(cut),'message_field_names':sorted(counts),'system_messages':len(systems),'system_content_sha256':[hashlib.sha256(json.dumps(m,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for m in systems],'recorded_model':row['recorded_model'],'metadata_keys':sorted(metadata) if isinstance(metadata,dict) else [],'source_record_sha256':hashlib.sha256(row['messages_json'].encode()).hexdigest(),'cut_sha256':hashlib.sha256(json.dumps(cut,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'pending_output_consumed':False})
    save('openhands_source_inspection.json',{'dataset_path':str(dataset),'schema':str(pq.read_schema(dataset)),'unique_sessions':len(rows),'snapshot_count':len({r['snapshot'] for r in refs}),'epoch_session_pairs':len(refs),'known_message_fields':dict(allkeys),'records':outputs})
    # 全文件流式扫描只核对字段，不用后续事件推导决策时可见依赖。
    path=Path('/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl');sha=hashlib.sha256();rootkeys=Counter();requestkeys=Counter();nestedkeys=Counter();traces=0;requests=0
    def visit(items):
        nonlocal requests
        if isinstance(items,list):
            for it in items:visit(it)
        elif isinstance(items,dict):
            nestedkeys.update(items.keys())
            if 'model' in items and 'in' in items:
                requests+=1;requestkeys.update(items.keys())
            for k,v in items.items():
                if k not in ('hash_ids','models') and isinstance(v,(dict,list)):visit(v)
    with path.open('rb') as f:
        for line in f:
            sha.update(line)
            if not line.strip():continue
            obj=json.loads(line);traces+=1;rootkeys.update(obj.keys());visit(obj.get('requests',[]))
    save('agentx_source_inspection.json',{'path':str(path),'sha256':sha.hexdigest(),'traces':traces,'request_records':requests,'root_fields':dict(rootkeys),'request_fields':dict(requestkeys),'nested_fields':dict(nestedkeys),'说明':'字段扫描不是ASG提取；未从后续事件构建图'})
    for name in ['rq3_openhands_neutral_collector.py','rq3_openhands_population.py','agentx_chain_detection.py','agentx_runtime_snapshot_collection.py','agentx_qwen_replay_preflight.py','kvflow_inspired_retention_cpu/policy.py']:
        p=ROOT/'evaluation'/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save('source_identity.json',{'read_files_unchanged':True,'sha256':sources,'说明':'Parquet以schema和命中原始记录摘要认证；未宣称本轮重算整个Parquet SHA'})
    print('OpenHands',len(rows),'sessions',len(refs),'epoch-session pairs',dict(allkeys));print('AgentX',traces,requests,dict(requestkeys),dict(nestedkeys))
if __name__=='__main__':main()
