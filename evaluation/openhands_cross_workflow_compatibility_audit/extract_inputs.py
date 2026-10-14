"""从已冻结 OpenHands session parquet 提取已有快照对应的消息前缀。"""
import json,hashlib
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
ART=ROOT/'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017'
DATA=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
index=[];wanted={}
for snap in sorted((ART/'snapshots').glob('g*.json')):
    x=read(snap);canonical=json.loads(x['canonical_snapshot']);ordinal=x['group_ordinal']
    runtime=read(ART/'runtime_correctness'/f'g{ordinal:03d}.json')
    sessions=x['session_ids'];assert len(sessions)==4
    case={'ordinal':ordinal,'snapshot_id':x['snapshot_id'],'allocation_round':x['allocation_round'],'snapshot_path':str(snap),'snapshot_sha256':sha(snap),'runtime_path':str(ART/'runtime_correctness'/f'g{ordinal:03d}.json'),'runtime_sha256':sha(ART/'runtime_correctness'/f'g{ordinal:03d}.json'),'sessions':[],'candidates':canonical['eligible_candidates'],'pending':canonical['pending_continuations'],'census':runtime['census_rows'][-1],'boundary_audit':runtime['boundary_audit'],'abort_reason':runtime['abort_reason']}
    for label,sid in zip('ABCD',sessions):
        wanted[sid]=(ordinal,label,x['allocation_round'])
        case['sessions'].append({'label':label,'session_id':sid})
    index.append(case)
assert len(index)==168 and len(wanted)==672
messages={};pf=pq.ParquetFile(DATA)
for batch in pf.iter_batches(columns=['session_id','messages_json'],batch_size=32):
    data=batch.to_pydict()
    for sid,value in zip(data['session_id'],data['messages_json']):
        if sid not in wanted:continue
        ordinal,label,round_id=wanted[sid];raw=json.loads(value);seen=0;prefix=[]
        for msg in raw:
            prefix.append(msg)
            if msg.get('role')=='assistant':
                seen+=1
                if seen==round_id+1:break
        if seen<round_id+1:raise ValueError(f'消息不完整 {sid}')
        messages[sid]={'ordinal':ordinal,'label':label,'allocation_round':round_id,'raw_messages_through_pending_marker':prefix}
assert len(messages)==672
for sid,row in messages.items():
    path=OUT/'messages'/f'{wanted[sid][0]:03d}_{wanted[sid][1]}.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
for case in index:
    for session in case['sessions']:session['message_file']=str(OUT/'messages'/f"{case['ordinal']:03d}_{session['label']}.json")
(OUT/'snapshot_inputs.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
print('提取快照',len(index),'会话前缀',len(messages))
